"""Label free answers written on the GPU pod (free_raw_*.jsonl) with the plausible action they recommend, as
collect_choices.py does for the API models; writes free_answers.jsonl next to each input file's choices.

    PYTHONPATH=scripts python scripts/label_free.py --scenarios S --responses R --raw out/free_raw_*.jsonl --out artifacts/step5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_responses import LABEL, LABELLER
from openrouter import Client, parse_json, spent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--raw", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    api = Client(a.out / "label_cache")
    scenarios = {s["scenario_id"]: s for s in map(json.loads, a.scenarios.open(encoding="utf-8"))}
    written = {}
    for r in map(json.loads, a.responses.open(encoding="utf-8")):
        if r["kept"] and r["source"] == "written":
            written[(r["scenario_id"], r["intended_action"])] = r["response_id"]

    for path in a.raw:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]

        def label(row):
            s = scenarios[row["scenario_id"]]
            acts = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(s["plausible_actions"]))
            try:
                lab = parse_json(api.chat(LABELLER, [{"role": "user", "content": LABEL.format(
                    scenario=s["scenario"], actions=acts, reply=row["text"])}], temperature=0)["text"]).get("action")
            except (ValueError, json.JSONDecodeError):
                lab = "unparsed"
            if isinstance(lab, (int, float)) or (isinstance(lab, str) and lab.isdigit()):
                lab = int(lab) - 1
            return {**row, "action": lab, "matches_response": written.get((row["scenario_id"], lab))}

        out = api.map(label, rows)
        name = rows[0]["model"].split("+")[-1] if "+" in rows[0]["model"] else "base"
        target = a.out / f"qwen2.5-7b-it-{name}"
        target.mkdir(parents=True, exist_ok=True)
        with (target / "free_answers.jsonl").open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"labelled {len(a.raw)} files; spent ${spent(a.out / 'label_cache'):.2f}")


if __name__ == "__main__":
    main()
