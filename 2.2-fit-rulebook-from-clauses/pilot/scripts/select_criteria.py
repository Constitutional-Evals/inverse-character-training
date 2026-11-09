"""Step 1: pick the pilot criteria from the shortlist.

Embed each criterion's text, then greedily pick the criterion farthest (in cosine distance) from those already picked,
at most one per tradition (`anchor`). The rule makes no assumption about which criteria conflict.

The first pick decides much of the set: started from a random criterion, two seeds share only ~40% of their 15 picks.
So by default (`--start centroid`) the first pick is the criterion farthest from the pool's mean embedding, and the
whole selection is deterministic. `--start seed --seed N` starts from a seeded random criterion instead.

    python scripts/select_criteria.py artifacts/inputs/criteria_78.csv --n 15 --out artifacts/step1

Writes, under --out:
    criteria_pilot.jsonl   the picked criteria, in pick order, with how they were picked
    constitution.json      the same criteria as one constitution, the input the scenario pipeline reads
    embeddings.npy, embeddings_meta.json   cached embeddings of all shortlisted criteria

Needs OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import urllib.request
from pathlib import Path

import numpy as np

API = "https://openrouter.ai/api/v1/embeddings"


def embed(texts: list[str], model: str) -> np.ndarray:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(API, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)["data"]
    vecs = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float64)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def cached_embeddings(rows: list[dict], model: str, out: Path) -> np.ndarray:
    texts = [r["criterion"] for r in rows]
    digest = hashlib.sha256(json.dumps([model, texts]).encode()).hexdigest()
    meta_path, vec_path = out / "embeddings_meta.json", out / "embeddings.npy"
    if meta_path.exists() and vec_path.exists() and json.loads(meta_path.read_text()).get("sha256") == digest:
        return np.load(vec_path)
    vecs = embed(texts, model)
    np.save(vec_path, vecs)
    meta_path.write_text(json.dumps({"model": model, "sha256": digest, "ids": [r["criterion_id"] for r in rows]}, indent=1))
    return vecs


def select(rows: list[dict], vecs: np.ndarray, n: int, seed: int, start: str = "centroid") -> list[dict]:
    """Farthest-point selection on cosine distance, one criterion per anchor."""
    if n > len({r["anchor"] for r in rows}):
        raise SystemExit(f"--n {n} exceeds the number of traditions")
    dist = 1.0 - vecs @ vecs.T
    if start == "centroid":
        mean = vecs.mean(axis=0)
        first = int(np.argmin(vecs @ (mean / np.linalg.norm(mean))))
    else:
        first = random.Random(seed).randrange(len(rows))
    picked, used, log = [first], {rows[first]["anchor"]}, [None]
    while len(picked) < n:
        best, best_d = None, -1.0
        for i, r in enumerate(rows):  # ties go to the earlier row, so the result is deterministic
            if r["anchor"] in used:
                continue
            d = float(dist[i, picked].min())
            if d > best_d:
                best, best_d = i, d
        picked.append(best)
        used.add(rows[best]["anchor"])
        log.append(round(best_d, 4))
    return [dict(rows[i], pick=k + 1, min_distance_at_pick=log[k]) for k, i in enumerate(picked)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shortlist", type=Path)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--start", choices=("centroid", "seed"), default="centroid")
    ap.add_argument("--seed", type=int, default=0, help="used only with --start seed")
    ap.add_argument("--model", default="openai/text-embedding-3-small")
    ap.add_argument("--out", type=Path, default=Path("artifacts/step1"))
    a = ap.parse_args()

    rows = list(csv.DictReader(a.shortlist.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("recommended_pool", "True") == "True"]
    a.out.mkdir(parents=True, exist_ok=True)
    vecs = cached_embeddings(rows, a.model, a.out)
    picks = select(rows, vecs, a.n, a.seed, a.start)

    rule = {"rule": "farthest-point on cosine distance, at most one per anchor", "n": a.n, "start": a.start,
            "seed": a.seed if a.start == "seed" else None,
            "embedding_model": a.model, "embedded_field": "criterion", "pool_size": len(rows),
            "shortlist_sha256": hashlib.sha256(a.shortlist.read_bytes()).hexdigest()}
    keep = ("criterion_id", "anchor", "anchor_title", "criterion", "reasoning", "pick", "min_distance_at_pick")
    with (a.out / "criteria_pilot.jsonl").open("w", encoding="utf-8") as f:
        for p in picks:
            f.write(json.dumps({**{k: p[k] for k in keep}, "selection": rule}, ensure_ascii=False) + "\n")
    constitution = {"name": f"pilot-{a.n}-" + ("centroid" if a.start == "centroid" else f"seed{a.seed}"),
                    "description": "Pilot criteria, one per tradition, chosen to be mutually different.",
                    "criteria": [{"criterion": p["criterion"], "reasoning": p["reasoning"], "tradition": p["anchor_title"],
                                  "criterion_id": p["criterion_id"]} for p in picks]}
    (a.out / "constitution.json").write_text(json.dumps(constitution, indent=1, ensure_ascii=False))
    for p in picks:
        print(f"{p['pick']:>2}  {p['criterion_id']:<40} {p['min_distance_at_pick']}")


if __name__ == "__main__":
    main()
