"""
Print one scenario readably: the question, each response, and its 1-10 scores on every criterion
(mean of the judges).

    python pipeline/show_scenario.py --run pilot_v2                 # list scenarios
    python pipeline/show_scenario.py --run pilot_v2 gen_00000       # show one
    python pipeline/show_scenario.py --run pilot_v2 gen_00000 --full   # also print the full response texts
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_id", nargs="?")
    ap.add_argument("--run", default="pilot_v2")
    ap.add_argument("--full", action="store_true", help="print the full response texts")
    args = ap.parse_args()

    rows = {r["scenario_id"]: r for r in map(json.loads, open(Path("outputs/responses") / args.run / "responses.jsonl"))}
    mpath = Path("outputs/ratings") / args.run / "ratings_matrix.json"
    matrix = json.loads(mpath.read_text()) if mpath.exists() else {"criteria": [], "scenarios": {}}
    crit = {str(c["pilot_id"]): c["criterion_id"] for c in matrix["criteria"]}

    if not args.scenario_id:
        for sid, r in rows.items():
            m = matrix["scenarios"].get(sid, {})
            status = "useful" if m.get("informative") else ("not useful" if m else "not rated")
            print(f"{sid}  {status:<10}  {r['scenario'][:90]}...")
        print("\nShow one with: python pipeline/show_scenario.py --run", args.run, "<scenario_id>")
        return

    r = rows[args.scenario_id]
    m = matrix["scenarios"].get(args.scenario_id, {})
    print("=" * 100)
    print(r["scenario"])
    print("\nVerified criteria:", ", ".join(crit.get(str(v), str(v)) for v in r["verified_criteria"]))
    print("Useful:", m.get("informative"), "| dominated:", m.get("dominated"))

    resps = r["responses"]
    print()
    for i, x in enumerate(resps, 1):
        designed = ", ".join(crit.get(str(c), str(c)) for c in x.get("favoured_by") or []) or "-"
        print(f"[{i}] {x['response_id']}  {x['kind']}  {x['words']} words")
        print(f"    {x['label'] or '(natural answer)'}" + (f": {x['recommendation']}" if x["recommendation"] else ""))
        print(f"    designed for: {designed}")
        if args.full:
            print("\n" + x["text"] + "\n")

    if m:
        print("\nScores 1-10 (mean of judges)")
        print(f"{'criterion':<38}" + "".join(f"{'[' + str(i) + ']':>7}" for i in range(1, len(resps) + 1)))
        for c, name in crit.items():
            vals = []
            for x in resps:
                rr = m["responses"].get(x["response_id"])
                vals.append("-" if rr is None else f"{rr['ratings'][c] * 9 + 1:.1f}")
            print(f"{c + ' ' + name:<38}" + "".join(f"{v:>7}" for v in vals))


if __name__ == "__main__":
    main()
