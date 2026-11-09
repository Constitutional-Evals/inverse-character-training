"""
Grade every design's scenarios the same way: Anand's quality-judge prompt, rubric and payload
(imported from the vendor file) with one judge that is used nowhere else in any design
(default anthropic/claude-sonnet-5.5), so no design is graded by a model that also filtered it.

    python experiments/grade_scenario_quality.py --name D0 --scenarios outputs/responses/pilot_v2/responses.jsonl
    python experiments/grade_scenario_quality.py --name D1 --scenarios outputs/responses/exp_D1/responses.jsonl

Writes outputs/quality/<name>.jsonl (one row per scenario).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(HERE / "experiments"))
from common import LLM, check_models, read_jsonl, write_jsonl  # noqa: E402
from d1_pair_first import load_constitution, quality_judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenarios", required=True, help="a responses.jsonl or scenarios jsonl with scenario_id and scenario")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    ap.add_argument("--max-calls", type=int, default=200)
    args = ap.parse_args()
    out = HERE / "outputs/quality"
    out.mkdir(parents=True, exist_ok=True)
    check_models({args.model})
    llm = LLM(out / "cache", out / "usage.jsonl", args.max_calls)
    constitution = load_constitution()
    rows = read_jsonl(args.scenarios)

    def grade(r):
        q = quality_judge(llm, args.model, constitution, r["scenario"], f"grade:{r['scenario_id']}")
        s = q["scores"]
        return {"scenario_id": r["scenario_id"], "design": args.name, "grader": args.model, "scores": s,
                "average": round(sum(s.values()) / len(s), 1), "surface_cue_neutrality": q["surface_cue_neutrality"],
                "flaw_type": q["flaw_type"], "verified_criteria": q.get("verified_criteria"), "feedback": q.get("feedback")}

    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(grade, rows))
    write_jsonl(out / f"{args.name}.jsonl", res)
    dims = list(res[0]["scores"])
    print(args.name, "n", len(res), "avg", round(sum(r["average"] for r in res) / len(res), 1),
          {d: round(sum(r["scores"][d] for r in res) / len(res), 1) for d in dims},
          "neutrality", round(sum(r["surface_cue_neutrality"] for r in res) / len(res), 1),
          "flaws", sum(r["flaw_type"] != "none" for r in res), f"cost ${llm.cost:.3f}")


if __name__ == "__main__":
    main()
