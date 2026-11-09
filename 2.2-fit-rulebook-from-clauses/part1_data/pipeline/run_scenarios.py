"""
Stage 2: run Anand's scenario pipeline (vendor/scenario_pipeline_10_24_26.py, unchanged)
on the pilot constitution, with a small budget.

    python pipeline/run_scenarios.py --run-name smoke --budget 5 --max-api-calls 400
    python pipeline/run_scenarios.py --run-name pilot --budget 50

The pipeline's models, prompts and quality thresholds are left exactly as Anand set them;
only the candidate budget and the API-call cap are changed here. Rerunning with the same
--run-name resumes from its cache. Use a new run name for a fresh run.

Main output: outputs/scenarios/<run-name>/generated_scenarios_rich.jsonl (accepted
scenarios with verified criteria, type, setting, intended tension and plausible actions).
API usage per call: outputs/scenarios/<run-name>/api_usage.jsonl

Needs OPENROUTER_API_KEY.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # part1_data/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", default=str(HERE / "outputs/criteria/constitution_pilot.json"))
    ap.add_argument("--pipeline", default=str(HERE / "vendor/scenario_pipeline_10_24_26.py"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--budget", type=int, required=True, help="number of candidate scenarios to generate")
    ap.add_argument("--max-api-calls", type=int, default=1500)
    ap.add_argument("--output-root", default=str(HERE / "outputs/scenarios"))
    args = ap.parse_args()

    if not Path(args.constitution).exists():
        raise SystemExit(f"{args.constitution} not found. Run select_criteria.py first.")

    spec = importlib.util.spec_from_file_location("scenario_pipeline", args.pipeline)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)

    gen = pipeline.CONFIG["generation"]
    gen["candidate_budget"] = args.budget
    if args.budget < gen["batch_size"]:  # keep the pipeline's own validity rules satisfied
        gen["batch_size"] = args.budget
    pipeline.CONFIG["runtime"]["max_api_calls"] = args.max_api_calls

    print(f"Budget: {args.budget} candidates, at most {args.max_api_calls} API calls\n")
    return pipeline.main([
        args.constitution,
        "--output-root", args.output_root,
        "--run-name", args.run_name,
        "--constitution-name", "pilot_criteria",
        "--full-outputs",
    ])


if __name__ == "__main__":
    sys.exit(main())
