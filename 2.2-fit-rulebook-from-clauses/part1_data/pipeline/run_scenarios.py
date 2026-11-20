"""Generate scenarios with Anand's pipeline or import its benchmark for Stage 3."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from common import read_jsonl, write_jsonl


HERE = Path(__file__).resolve().parents[1]
ANAND_PIPELINE = HERE / "vendor/constitutions/Completed_Scenario_Generation"


def pilot_constitutions(criteria_path):
    """Keep the globally unique pilot IDs when grouping criteria for M3."""
    criteria = sorted(read_jsonl(criteria_path), key=lambda row: row["pilot_id"])
    if not criteria or [row["pilot_id"] for row in criteria] != list(range(1, len(criteria) + 1)):
        raise ValueError("criteria pilot_ids must be 1..k")

    groups = {}
    for row in criteria:
        tradition = row["criterion_id"].rsplit("#", 1)[0]
        group = groups.setdefault(tradition, {"criteria": []})
        group["criteria"].append({"id": row["pilot_id"], "text": row["criterion"]})
    return {"constitutions": groups}


def response_scenarios(bank, target=""):
    rows = []
    for row in bank:
        relevance = row["relevance"]
        target_ids = set(relevance.get(target, []))
        verified_ids = {key for ids in relevance.values() for key in ids}
        rows.append({
            "scenario_id": row["scenario_id"],
            "scenario": row["scenario"],
            "verified_criteria": sorted(target_ids) + sorted(verified_ids - target_ids),
            "scenario_type": row.get("scenario_type"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--budget", type=int, help="number of new scenario candidates to attempt")
    mode.add_argument("--benchmark", type=Path, help="import a labeled Anand benchmark without API calls")
    parser.add_argument("--criteria", type=Path, default=HERE / "outputs/criteria/criteria_pilot.jsonl")
    parser.add_argument("--target", default="", help="target one group for generation or prioritize its verified IDs on import")
    parser.add_argument("--existing", type=Path, help="existing Anand benchmark to extend with --budget")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs/scenarios")
    parser.add_argument("--max-api-calls", type=int, default=1500)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--quota", type=int, help="coverage quota per input group")
    parser.add_argument("--min-criteria", type=int, help="coverage breadth target per input group")
    args = parser.parse_args()

    if args.benchmark and args.existing:
        parser.error("--existing is for generation with --budget")
    if not (ANAND_PIPELINE / "modules/core.py").is_file():
        parser.error("Initialize Anand's pinned checkout with git submodule update --init.")

    sys.path.insert(0, str(ANAND_PIPELINE))
    from modules import config as cfg
    from modules.core import atomic_json, initialize_workspace, load_constitutions, load_existing_bank, read_json, slug

    root = args.output_root / args.run_name
    pool_path = root / "pilot_constitutions.json"
    pool = pilot_constitutions(args.criteria)
    target = slug(args.target)
    group_ids = sorted(slug(name) for name in pool["constitutions"])
    if args.target and target not in group_ids:
        parser.error(f"Unknown target constitution {args.target!r}. Available IDs: {', '.join(group_ids)}")
    if pool_path.exists() and read_json(pool_path) != pool:
        parser.error("The criteria differ from this run's saved pool. Use a new run name.")
    atomic_json(pool_path, pool)
    if args.benchmark:
        const = load_constitutions(pool_path)
        bank = load_existing_bank(args.benchmark, const)
        if not bank or any(set(row["relevance"]) != set(const) for row in bank):
            parser.error("--benchmark requires a benchmark export with criterion relevance labels")
    else:
        from modules.client import OpenRouterClient, obtain_openrouter_key
        from modules.pipeline import run_pipeline
        from modules.stages import show_coverage

        cfg.CONSTITUTION_INPUT = str(pool_path)
        cfg.TARGET_CONSTITUTION = target
        cfg.EXISTING_BANK_INPUT = str(args.existing) if args.existing else ""
        cfg.LOCAL_RUN_PARENT = str(args.output_root)
        cfg.RUN_NAME = args.run_name
        cfg.MAX_NEW_ATTEMPTS = args.budget
        cfg.MAX_API_REQUESTS = args.max_api_calls
        if args.batch_size is not None:
            cfg.GENERATION_BATCH_SIZE = args.batch_size
        if args.quota is not None:
            cfg.SCENARIO_QUOTA = args.quota
        if args.min_criteria is not None:
            cfg.MIN_DISTINCT_CRITERIA = args.min_criteria

        root, const, existing = initialize_workspace()
        results = run_pipeline(root, const, existing, OpenRouterClient(root, obtain_openrouter_key()))
        show_coverage(results["report"])
        bank = results["existing"] + results["selected"]

    rows = response_scenarios(bank, target)
    path = root / "generated_scenarios_rich.jsonl"
    with TemporaryDirectory(dir=root) as directory:
        temporary = Path(directory) / path.name
        write_jsonl(temporary, rows)
        temporary.replace(path)
    print(f"Wrote {len(rows)} scenarios to {path}")


if __name__ == "__main__":
    main()
