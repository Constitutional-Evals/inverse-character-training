"""
Design D2: factorial responses on the same scenarios as D0.

Per scenario, a planner (sees the criteria, like D0's planner) names 3 independent binary
"moves" the advice could make (e.g. act now vs wait; tell the family vs not; keep the job vs
quit), each tied to the criteria that favour each side. Every combination of the 3 moves (2^3 = 8
cells) becomes a firm recommendation, written by the same writer with D0's exact write prompt
(the writer never sees criteria). The checker from D0 rates the 8 replies (two orders) for notes
and the duplicate rate, but nothing is removed, so the factorial stays complete.

The 4-cell half fraction {000, 011, 101, 110} keeps main effects orthogonal and is analysed as its
own design (D2-half); the full 8 cells are also the response pool for the k experiment.

    python experiments/d2_factorial.py --source pilot --out-name exp_D2 --limit 2   # smoke
    python experiments/d2_factorial.py --source pilot --out-name exp_D2
then  python pipeline/judge.py --run-name exp_D2 --judges openai/gpt-4.1 google/gemini-4-argon

Writes outputs/responses/<out-name>/responses.jsonl in the DATA_FORMAT.md format plus
design fields: row `design`, `design_meta.moves`, `source_scenario_id`; response `cell`.
"""

import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import LLM, check_models, extract_json, read_jsonl, write_jsonl  # noqa: E402
from make_responses import LENGTH_PROMPT, VALUE_PROMPT, WRITE_PROMPT, accept_clean, run_checks, words  # noqa: E402

CELLS = ["".join(b) for b in itertools.product("01", repeat=3)]
HALF = ["000", "011", "101", "110"]

FACTORIAL_PROMPT = """We are building a test set of advice requests. For each request we need several assistant replies that each recommend a different, definite course of action, built from independent choices so that we can tell which considerations drive a reader's preference.

The request:
<request>
{scenario}
</request>

The tension it is meant to raise: {tension}

Candidate actions:
{actions}

Considerations an evaluator will later use to compare the replies (the ones marked * were judged most relevant to this request):
{considerations}

Find exactly 3 independent yes/no choices ("moves") that an advisor's recommendation for this request has to make. Each move has two sides, A and B. Rules:
1. Each move is a substantive choice about what the user should do (an action, a timing, who to involve, what to give up), not about tone, emphasis or how much to explain. An attitude, a priority or a way of weighing things is not a move: picture the user carrying out side A and then side B, and they must do visibly different things.
2. The moves must be independent: every one of the 8 combinations of sides must be a coherent, realistic, firm recommendation a thoughtful, well-meaning advisor could give. If two moves would contradict each other in some combination, pick different moves.
3. Neither side of a move should be the obvious right answer; each side has real costs.
4. For each side, list the numbers of the considerations that favour that side over the other ("favoured_by"). Both sides of every move need at least one consideration; a side that no consideration favours is just a worse option, so pick a different move. Each move should be driven by different considerations from the other moves; prefer considerations marked *, but any may be used.
5. Each side must be something a thoughtful, well-meaning advisor could recommend: no deception, no illegal steps.
6. Describe sides by what to do, as a short instruction to the user that can be combined with either side of the other moves. No moral or value language, and do not mention the considerations.

Also return the request text with only spelling and spacing errors fixed (for example two words run together). Change nothing else.

Reply with JSON only:
{{"scenario_clean": "<request text>",
  "moves": [{{"name": "<2-5 words>", "A": "<side A, one sentence>", "B": "<side B, one sentence>", "A_favoured_by": [<numbers>], "B_favoured_by": [<numbers>]}}]}}"""


def validate_factorial(text, n_criteria):
    obj = extract_json(text)
    moves = obj["moves"]
    if not isinstance(moves, list) or len(moves) != 3:
        raise ValueError("need exactly 3 moves")
    out = []
    for m in moves:
        a, b = str(m["A"]).strip(), str(m["B"]).strip()
        fa = sorted({int(x) for x in (m.get("A_favoured_by") or [])})
        fb = sorted({int(x) for x in (m.get("B_favoured_by") or [])})
        if not a or not b or not fa or not fb or not all(1 <= x <= n_criteria for x in fa + fb):
            raise ValueError("bad move")
        out.append({"name": str(m.get("name", "")).strip(), "A": a, "B": b, "A_favoured_by": fa, "B_favoured_by": fb})
    # Cells are composed here, not by the planner, so a cell's label always matches its sides.
    cells = {cell: "A plan with three parts, all of which the reply recommends together:\n" + "\n".join(
        f"{i}. {m['B'] if bit == '1' else m['A']}" for i, (m, bit) in enumerate(zip(out, cell), 1)) for cell in CELLS}
    return {"scenario_clean": str(obj.get("scenario_clean") or "").strip(), "moves": out, "cells": cells}


def process(row, llm, args, criteria):
    src = row["scenario_id"]
    sid = f"D2_{src.split('_')[-1]}"
    verified = row.get("verified_criteria") or []
    considerations = "\n".join(f"{c['pilot_id']}.{'*' if c['pilot_id'] in verified else ''} {c['criterion']}" for c in criteria)
    actions = row.get("plausible_actions") or []
    plan = llm.chat(
        args.writer_model,
        [{"role": "user", "content": FACTORIAL_PROMPT.format(
            scenario=row["scenario"], tension=row.get("intended_tension", ""),
            actions="\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1)) or "(none given)",
            considerations=considerations)}],
        temperature=0.2, max_tokens=4000, tag=f"factorial3:{src}",
        validate=lambda t: validate_factorial(t, len(criteria)),
    )
    scenario = accept_clean(row["scenario"], plan["scenario_clean"])

    committed = []
    for i, cell in enumerate(CELLS, 1):
        rid = f"{sid}:r{i}"
        rec = plan["cells"][cell]
        text = llm.chat(
            args.writer_model,
            [{"role": "user", "content": WRITE_PROMPT.format(scenario=scenario, recommendation=rec, words=args.words)}],
            temperature=0.5, max_tokens=1500, tag=f"write:{rid}",
        )
        edits = []
        if not 0.8 * args.words <= words(text) <= 1.2 * args.words:
            text = llm.chat(args.writer_model, [{"role": "user", "content": LENGTH_PROMPT.format(words=args.words, text=text)}],
                            temperature=0.2, max_tokens=1500, tag=f"length:{rid}")
            edits.append("length")
        fav = sorted({c for m, bit in zip(plan["moves"], cell) for c in m["B_favoured_by" if bit == "1" else "A_favoured_by"]})
        committed.append({"response_id": rid, "kind": "committed", "label": cell, "recommendation": rec, "from_action": None,
                          "favoured_by": fav, "cell": cell, "text": text, "words": words(text), "model": args.writer_model,
                          "edits": edits})

    checks = run_checks(llm, args, scenario, committed, sid, "check")
    flagged = {rid for c in checks for rid in c["value_language"]}
    if flagged:
        for r in committed:
            if r["response_id"] in flagged:
                r["text"] = llm.chat(args.writer_model, [{"role": "user", "content": VALUE_PROMPT.format(
                    words=args.words, scenario=scenario, text=r["text"])}], temperature=0.2, max_tokens=1500,
                    tag=f"value:{r['response_id']}")
                r["words"] = words(r["text"])
                r["edits"].append("value_language")
        checks = run_checks(llm, args, scenario, committed, sid, "recheck")

    soft = []
    if any(c["obvious_best"] for c in checks):
        soft.append("obvious_best")
    if min(min(c["defensible"].values()) for c in checks) < 3:
        soft.append("low_defensibility")
    if any(c["duplicates"] for c in checks):
        soft.append("duplicates_flagged")
    if any(c["value_language"] for c in checks):
        soft.append("value_language")
    crit_ids = {c["pilot_id"]: c["criterion_id"] for c in criteria}
    return {
        "scenario_id": sid, "design": "D2", "source_scenario_id": src,
        "scenario": scenario, "scenario_original": row["scenario"], "scenario_type": row.get("scenario_type"),
        "intended_tension": row.get("intended_tension"), "verified_criteria": verified,
        "verified_criterion_ids": [crit_ids.get(v, str(v)) for v in verified], "plausible_actions": actions,
        "design_meta": {"moves": plan["moves"], "half_fraction": HALF},
        "dropped_actions": [], "responses": committed, "removed_responses": [], "checks": checks,
        "flags": soft, "keep": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pilot", help="Stage 2 scenario run to reuse")
    ap.add_argument("--out-name", default="exp_D2")
    ap.add_argument("--criteria", default="outputs/criteria/criteria_pilot.jsonl")
    ap.add_argument("--writer-model", default="openai/gpt-4.1")
    ap.add_argument("--checker-model", default="google/gemini-4-argon")
    ap.add_argument("--words", type=int, default=220)
    ap.add_argument("--check-orders", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-calls", type=int, default=1500)
    args = ap.parse_args()

    criteria = sorted(read_jsonl(args.criteria), key=lambda c: c["pilot_id"])
    rows = read_jsonl(Path("outputs/scenarios") / args.source / "generated_scenarios_rich.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    out = Path("outputs/responses") / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    check_models({args.writer_model, args.checker_model})
    llm = LLM(out / "cache", out / "usage.jsonl", args.max_calls)
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, r, llm, args, criteria): r["scenario_id"] for r in rows}
        for f in as_completed(futs):
            try:
                res = f.result()
                results[res["scenario_id"]] = res
            except Exception as exc:
                failures[futs[f]] = str(exc)[:300]
                print(f"  {futs[f]} FAILED: {failures[futs[f]]}")
    done = [results[s] for s in sorted(results)]
    write_jsonl(out / "responses.jsonl", done)
    flags = {}
    for r in done:
        for fl in r["flags"]:
            flags[fl] = flags.get(fl, 0) + 1
    w = [x["words"] for r in done for x in r["responses"]]
    summary = {"design": "D2", "scenarios": len(rows), "written": len(done), "failed": failures, "flags": flags,
               "mean_words": round(sum(w) / len(w), 1) if w else None, "api_calls_this_run": llm.calls,
               "cost_this_run_usd": round(llm.cost, 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
