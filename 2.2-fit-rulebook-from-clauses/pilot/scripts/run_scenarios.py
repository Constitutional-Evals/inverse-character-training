"""Step 2: generate the pilot scenarios with the team's scenario pipeline.

The pipeline is `mass_scenario_gen` in Constitutional-Evals/constitutions (branch massscenariogen), used unchanged;
this script only overrides its CONFIG:
- the held-out judge is GPT-5.6 Terra instead of Grok 4.7, because Grok is a target model;
- coverage asks for MIN_PER_CRITERION scenarios per criterion and allows a top-up round to reach it;
- a call cap as a backstop (the dollar cap is enforced from outside by watching the key's spend).

    PYTHONPATH=scripts python scripts/run_scenarios.py \
        --pipeline "<clone>/mass_scenario_gen/mass_scenario_generation_pipeline_10_24_26 (1).py" \
        --constitution artifacts/step1/constitution.json --run-id scenarios-pilot15 --candidates 120 --topup 60 \
        --min-per-criterion 6

Writes artifacts/runs/<run-id>/{scenarios.jsonl, report.json, summary.md, meta.json}; the pipeline's own audit files are
in pipeline/. Rerunning with the same --run-id resumes from the pipeline's cache. Needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

HELDOUT = [{"name": "GPT", "model": "openai/gpt-5.6-terra", "reasoning": ""}]
OVERRIDES = {
    "generation": {"candidate_budget": 60},
    "coverage": {"min_per_criterion": 5, "min_per_pair": 2, "topup_rounds": 1, "topup_candidates": 30},
    "runtime": {"max_api_calls": 2500},
}


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("scenario_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", type=Path, required=True)
    ap.add_argument("--constitution", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--topup", type=int, default=30)
    ap.add_argument("--min-per-criterion", type=int, default=5)
    a = ap.parse_args()
    OVERRIDES["generation"]["candidate_budget"] = a.candidates
    OVERRIDES["coverage"].update(min_per_criterion=a.min_per_criterion, topup_candidates=a.topup)

    pipe = load_pipeline(a.pipeline)
    cfg = copy.deepcopy(pipe.CONFIG)
    cfg["models"]["heldout"] = HELDOUT
    for section, values in OVERRIDES.items():
        cfg[section].update(values)
    cfg["paths"].update(use_google_drive=False, constitution_path=str(a.constitution.resolve()),
                        constitution_name=a.constitution.stem, run_name=a.run_id)
    pipe.validate_config(cfg)

    local = Path("artifacts/runs") / a.run_id
    work = local / "pipeline"
    raw = json.loads(a.constitution.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "-C", str(a.pipeline.parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    local.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": a.run_id, "pipeline": "Constitutional-Evals/constitutions mass_scenario_gen", "pipeline_commit": commit,
            "pipeline_version": getattr(pipe, "VERSION", ""), "development_models": cfg["models"]["development"],
            "heldout_models": HELDOUT, "overrides": OVERRIDES,
            "note": "Held-out judge swapped from Grok 4.7 to GPT-5.6 Terra: Grok is a target model."}
    (local / "meta.json").write_text(json.dumps(meta, indent=1))
    accepted, report = pipe.run_pipeline(raw, cfg, work)

    rows = accepted  # the pipeline's rich rows: text, verified criteria, plausible actions, judge history
    with (local / "scenarios.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (local / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    cov = report.get("final_coverage", {}).get("criteria", {})
    (local / "summary.md").write_text(
        f"# Pilot scenarios\n\n{len(rows)} accepted scenarios for the 15 pilot criteria. Coverage goals met: "
        f"{report.get('minimums_met')}.\n\n| Criterion | Scenarios |\n|---|---|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in cov.items()) + "\n")
    print(f"{len(rows)} accepted scenarios; coverage goals met: {report.get('minimums_met')}")

if __name__ == "__main__":
    main()
