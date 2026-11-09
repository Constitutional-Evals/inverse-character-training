"""
Stage 1: choose a subset of the shortlisted criteria that apply broadly and differ from each other.

1. Applicability filter: an LLM rates each criterion 1-5 for how often it would favour one
   reasonable response over another across everyday assistant requests. Only criteria rated
   >= --min-applicability are kept. (--min-applicability 0 skips this step.)
2. Diversity: embed the remaining criterion texts, then pick one at a time, each time taking
   the criterion least similar to everything already picked (farthest-point selection), with
   at most one criterion per tradition. The first pick is the criterion least similar to the
   average. Deterministic: same inputs and ratings, same subset.

    python pipeline/select_criteria.py --k 15

Writes (to --out-dir, default outputs/criteria):
  criteria_pilot.jsonl       the chosen criteria, in pipeline order (pilot_id 1..k)
  constitution_pilot.json    input file for the scenario pipeline
  applicability.jsonl        the 1-5 rating and reason for every criterion
  selection_log.json         settings, pick order and distances
Ratings and embeddings are cached under outputs/cache, so reruns cost nothing.

Needs OPENROUTER_API_KEY.
"""

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

OVERVIEW = (
    "This is not the constitution of a single tradition. It is a set of {k} independent criteria, "
    "each drawn from a different value tradition, used to build scenarios in which an assistant's "
    "choices can be attributed to individual criteria. Treat each criterion on its own terms: in a "
    "given situation two criteria may agree, conflict, or be unrelated."
)


APPLICABILITY_PROMPT = """You are helping design a benchmark of everyday requests to a general AI assistant: practical advice, decisions, work, money, health habits, family and relationships, conflicts with other people, community and civic matters, and so on.

Here is one criterion for judging assistant responses:
"{criterion}"

Across such everyday requests, how often would this criterion favour one reasonable response over another reasonable response?
1 = almost never; it only matters for a narrow topic most users never raise
2 = rarely; a few specific kinds of situation
3 = sometimes; a recognisable but limited range of situations
4 = often; many ordinary situations
5 = very often; most ordinary advice or decision requests

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=120)


def rate_applicability(rows: list[dict], model: str, cache_path: Path) -> dict[str, dict]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = lambda r: f"{model}::{r['criterion_id']}"
    todo = [r for r in rows if key(r) not in cache]
    if todo:
        c = client()

        def ask(r):
            for _ in range(4):
                out = c.chat.completions.create(
                    model=model, temperature=0, max_tokens=200,
                    messages=[{"role": "user", "content": APPLICABILITY_PROMPT.format(criterion=r["criterion"].strip())}],
                )
                m = re.search(r"\{.*\}", out.choices[0].message.content or "", re.DOTALL)
                try:
                    obj = json.loads(m.group(0)) if m else {}
                    score = int(obj["score"])
                    if 1 <= score <= 5:
                        return r, {"score": score, "reason": str(obj.get("reason", ""))}
                except (ValueError, KeyError, TypeError):
                    pass
            raise RuntimeError(f"Could not get a rating for {r['criterion_id']}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            for r, rating in pool.map(ask, todo):
                cache[key(r)] = rating
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1))
    return {r["criterion_id"]: cache[key(r)] for r in rows}


def load_criteria(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if r.get("recommended_pool", "True") == "True"]
    if not keep:
        raise SystemExit(f"No criteria found in {path}")
    return keep


def embed(texts: list[str], model: str, cache_path: Path) -> np.ndarray:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = lambda t: f"{model}::{t}"
    todo = [t for t in texts if key(t) not in cache]
    if todo:
        c = client()
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            resp = c.embeddings.create(model=model, input=batch)
            for item in sorted(resp.data, key=lambda d: d.index):
                cache[key(batch[item.index])] = item.embedding
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    X = np.array([cache[key(t)] for t in texts], dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def select(X: np.ndarray, anchors: list[str], k: int, one_per_anchor: bool) -> list[tuple[int, float | None]]:
    sim = X @ X.T
    centroid = X.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    first = int(np.argmin(X @ centroid))
    picks = [(first, None)]
    used = {anchors[first]}
    while len(picks) < k:
        chosen = [i for i, _ in picks]
        best, best_d = None, -1.0
        for i in range(len(anchors)):
            if i in chosen or (one_per_anchor and anchors[i] in used):
                continue
            d = 1.0 - float(sim[i, chosen].max())  # distance to the nearest already-picked criterion
            if d > best_d:
                best, best_d = i, d
        if best is None:
            print(f"Only {len(picks)} criteria available under the constraints.")
            break
        picks.append((best, best_d))
        used.add(anchors[best])
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/recommended_78_criteria.csv")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    ap.add_argument("--rater-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--min-applicability", type=int, default=4, help="keep criteria rated at least this (0 = no filter)")
    ap.add_argument("--allow-same-anchor", action="store_true", help="drop the one-per-tradition rule")
    ap.add_argument("--exclude", nargs="*", default=[], help="criterion_ids to leave out, e.g. maori#02")
    ap.add_argument("--out-dir", default="outputs/criteria")
    ap.add_argument("--cache", default="outputs/cache/criterion_embeddings.json")
    ap.add_argument("--rating-cache", default="outputs/cache/applicability_ratings.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = [r for r in load_criteria(Path(args.csv)) if r["criterion_id"] not in set(args.exclude)]

    ratings = {}
    if args.min_applicability > 0:
        ratings = rate_applicability(all_rows, args.rater_model, Path(args.rating_cache))
        with (out / "applicability.jsonl").open("w", encoding="utf-8") as f:
            for r in sorted(all_rows, key=lambda r: (-ratings[r["criterion_id"]]["score"], r["criterion_id"])):
                f.write(json.dumps({"criterion_id": r["criterion_id"], **ratings[r["criterion_id"]]}, ensure_ascii=False) + "\n")
        counts = {s: sum(v["score"] == s for v in ratings.values()) for s in range(1, 6)}
        print(f"Applicability ratings (1-5): {counts}")
        rows = [r for r in all_rows if ratings[r["criterion_id"]]["score"] >= args.min_applicability]
        print(f"{len(rows)} of {len(all_rows)} criteria rated >= {args.min_applicability}, "
              f"from {len({r['anchor'] for r in rows})} traditions\n")
    else:
        rows = all_rows
    if not rows:
        raise SystemExit("No criteria left after the applicability filter. Lower --min-applicability.")

    texts = [r["criterion"].strip() for r in rows]
    anchors = [r["anchor"] for r in rows]
    X = embed(texts, args.embedding_model, Path(args.cache))
    picks = select(X, anchors, args.k, not args.allow_same_anchor)

    chosen = []
    for order, (i, d) in enumerate(picks, 1):
        r = rows[i]
        chosen.append({
            "pilot_id": order,  # = the criterion's id inside the scenario pipeline
            "criterion_id": r["criterion_id"],
            "anchor": r["anchor"],
            "anchor_title": r["anchor_title"],
            "criterion": r["criterion"].strip(),
            "reasoning": r["reasoning"].strip(),
            "applicability": ratings.get(r["criterion_id"], {}).get("score"),
            "distance_to_nearest_pick": d,
        })
    with (out / "criteria_pilot.jsonl").open("w", encoding="utf-8") as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    constitution = {
        "title": f"Pilot set of {len(chosen)} independent criteria",
        "overview": OVERVIEW.format(k=len(chosen)),
        "criteria": [{"comparative": c["criterion"], "reasoning": c["reasoning"]} for c in chosen],
    }
    (out / "constitution_pilot.json").write_text(json.dumps(constitution, indent=2, ensure_ascii=False))

    (out / "selection_log.json").write_text(json.dumps({
        "csv": args.csv, "n_after_exclusions": len(all_rows), "n_candidates": len(rows), "k": args.k,
        "rater_model": args.rater_model if args.min_applicability > 0 else None,
        "min_applicability": args.min_applicability,
        "embedding_model": args.embedding_model, "one_per_anchor": not args.allow_same_anchor,
        "excluded": args.exclude, "first_pick": "least similar to the average criterion",
        "picks": [{"pilot_id": c["pilot_id"], "criterion_id": c["criterion_id"],
                   "distance_to_nearest_pick": c["distance_to_nearest_pick"]} for c in chosen],
    }, indent=2))

    print(f"Picked {len(chosen)} of {len(rows)} criteria:\n")
    for c in chosen:
        d = "  first" if c["distance_to_nearest_pick"] is None else f"{c['distance_to_nearest_pick']:.3f}"
        a = f"applicability {c['applicability']}" if c["applicability"] is not None else ""
        print(f"{c['pilot_id']:>3}  {c['criterion_id']:<38} {d}   {a}")
        print(f"     {c['criterion']}\n")
    print(f"Wrote {out}/criteria_pilot.jsonl, constitution_pilot.json, selection_log.json")


if __name__ == "__main__":
    main()
