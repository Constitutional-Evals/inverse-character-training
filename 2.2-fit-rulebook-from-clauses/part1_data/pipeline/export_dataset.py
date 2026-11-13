"""Export the standard pilot dataset (criteria + scenarios with their criteria and responses) to dataset/.

    python pipeline/export_dataset.py                      # pilot scenarios, pilot_v2 responses
    python pipeline/export_dataset.py --responses-run pilot_v2 --out dataset

Reads  outputs/criteria/criteria_pilot.jsonl and outputs/responses/<run>/responses.jsonl
Writes dataset/criteria.jsonl and dataset/scenarios.jsonl (format in dataset/README.md).
"""
import argparse
from pathlib import Path

from common import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", default="outputs/criteria/criteria_pilot.jsonl")
    ap.add_argument("--responses-run", default="pilot_v2")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    criteria = sorted(read_jsonl(Path(args.criteria)), key=lambda c: c["pilot_id"])
    rows = [r for r in read_jsonl(Path("outputs/responses") / args.responses_run / "responses.jsonl") if r["keep"]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "criteria.jsonl", [
        {"pilot_id": c["pilot_id"], "criterion_id": c["criterion_id"], "tradition": c["anchor_title"], "criterion": c["criterion"]}
        for c in criteria
    ])
    write_jsonl(out / "scenarios.jsonl", [
        {
            "scenario_id": r["scenario_id"],
            "scenario": r["scenario"],
            "verified_criteria": r["verified_criteria"],
            "verified_criterion_ids": r["verified_criterion_ids"],
            "responses": [
                {"response_id": a["response_id"], "kind": a["kind"], "recommendation": a.get("recommendation"), "text": a["text"]}
                for a in r["responses"]
            ],
        }
        for r in rows
    ])
    n_resp = sum(len(r["responses"]) for r in rows)
    print(f"wrote {out}/criteria.jsonl ({len(criteria)} criteria), {out}/scenarios.jsonl ({len(rows)} scenarios, {n_resp} responses)")


if __name__ == "__main__":
    main()
