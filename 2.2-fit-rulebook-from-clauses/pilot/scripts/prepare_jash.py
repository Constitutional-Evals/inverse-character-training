"""Turn Jash's criterion-groups snapshot into this pipeline's inputs, unchanged in content.

The snapshot (https://jashvira.com/scenarios/criterion-groups-preview/snapshot.json) has 6 groups of 6 criteria and 47
accepted scenarios, each verified on criteria of its own group (group-local ids 1-6). Criteria get global ids 1-36
(group order, then criterion order), so the scenarios' verified criteria map onto one criteria file and every later
step runs as it does for the pilot; a scenario is rated only on its own group's verified criteria.

The snapshot keeps no plausible actions, which step 3 writes responses from: ACTION_MODEL lists 3-4 materially different
courses of action per scenario from the scenario text alone (not from the criteria).

    PYTHONPATH=scripts python scripts/prepare_jash.py --snapshot snapshot.json --out artifacts/jash
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openrouter import Client, parse_json, spent

ACTION_MODEL = "openai/gpt-5.6-terra"
ACTIONS = """A person sent this message asking for advice:

<message>
{scenario}
</message>

List the 3 or 4 materially different courses of action they could take (different decisions, not different wordings
or explanations). Use the options the message names where it names them. Describe each in one neutral sentence, with
no judgement of which is better and no moral vocabulary.

Answer with JSON only: {{"actions": ["...", "..."]}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/jash"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    snap = json.loads(a.snapshot.read_text(encoding="utf-8"))

    crit, glob = [], {}
    for g in snap["groups"]:
        for c in g["criteria"]:
            glob[(g["id"], c["id"])] = len(crit) + 1
            crit.append({"criterion_id": f"{g['id']}/{c['source_criterion_id']}", "anchor": c["source_constitution"],
                         "group": g["id"], "criterion": c["comparative"], "reasoning": c.get("reasoning", ""),
                         "pick": len(crit) + 1})
    with (a.out / "criteria_jash.jsonl").open("w", encoding="utf-8") as f:
        for c in crit:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    api = Client(a.out / "cache")

    def actions(s):
        for attempt in range(3):
            try:
                out = parse_json(api.chat(ACTION_MODEL, [{"role": "user", "content": ACTIONS.format(scenario=s["scenario"])}],
                                          temperature=0.0, seed=attempt)["text"])
                acts = [x.strip() for x in out.get("actions", []) if isinstance(x, str) and x.strip()]
                if 2 <= len(acts) <= 5:
                    return acts
            except (ValueError, json.JSONDecodeError):
                pass
        return []

    rows = snap["scenarios"]
    acts = api.map(actions, rows)
    with (a.out / "scenarios_jash.jsonl").open("w", encoding="utf-8") as f:
        for s, act in zip(rows, acts):
            f.write(json.dumps({"scenario_id": s["id"], "group": s["group"], "scenario": s["scenario"],
                                "verified_criteria": sorted(glob[(s["group"], i)] for i in s["pipeline_verified_criteria"]),
                                "plausible_actions": act, "generator_model": s.get("generator_model"),
                                "source": "jashvira.com criterion-groups-preview snapshot"}, ensure_ascii=False) + "\n")
    print(f"{len(crit)} criteria, {len(rows)} scenarios, {sum(1 for x in acts if x)} with actions; spent ${spent(a.out / 'cache'):.2f}")


if __name__ == "__main__":
    main()
