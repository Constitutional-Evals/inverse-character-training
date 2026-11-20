"""Generate scenarios with Anand's pipeline or import its benchmark for Stage 3."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock, Timeout

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
        scheduled_ids = target_ids.intersection(row.get("target_criteria") or ())
        verified_ids = {key for ids in relevance.values() for key in ids}
        verified_order = sorted(scheduled_ids) + sorted(target_ids - scheduled_ids) + sorted(verified_ids - target_ids)
        rows.append({
            "scenario_id": row["scenario_id"],
            "scenario": row["scenario"],
            "verified_criteria": verified_order,
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
    root = args.output_root / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / ".scenario-run.lock"), timeout=0):
            run_scenarios(args, parser, root)
    except Timeout:
        parser.error("This scenario run is already in progress.")


def run_scenarios(args, parser, root):
    from modules import config as cfg
    from modules.core import atomic_json, digest, initialize_workspace, load_constitutions, load_existing_bank, read_json, slug

    pool_path = root / "pilot_constitutions.json"
    path = root / "generated_scenarios_rich.jsonl"
    import_manifest = root / "benchmark_import.json"
    has_generation_state = (root / "run_manifest.json").exists() or (root / "inputs").exists()
    if args.benchmark and has_generation_state:
        parser.error("This run name is already used for generation. Import under a new run name.")
    if not args.benchmark and (import_manifest.exists() or (path.exists() and not has_generation_state)):
        parser.error("This run name is already used for benchmark import. Generate under a new run name.")

    pool = pilot_constitutions(args.criteria)
    target = slug(args.target)
    group_ids = sorted(slug(name) for name in pool["constitutions"])
    if args.target and target not in group_ids:
        parser.error(f"Unknown target constitution {args.target!r}. Available IDs: {', '.join(group_ids)}")
    if pool_path.exists() and read_json(pool_path) != pool:
        parser.error("The criteria differ from this run's saved pool. Use a new run name.")
    if args.benchmark:
        # Validate the incoming pool and benchmark before writing either into the run.
        with TemporaryDirectory(dir=root) as directory:
            proposed_pool = Path(directory) / pool_path.name
            atomic_json(proposed_pool, pool)
            const = load_constitutions(proposed_pool)
        bank = load_existing_bank(args.benchmark, const)
        if not bank or any(set(row["relevance"]) != set(const) for row in bank):
            parser.error("--benchmark requires a benchmark export with criterion relevance labels")
        rows = response_scenarios(bank, target)
        identity = digest({"criteria": pool, "benchmark": bank, "target": target})
        saved_import = read_json(import_manifest)
        if saved_import is not None and saved_import["identity"] != identity:
            parser.error("The benchmark or target differs from this run's saved import. Use a new run name.")
        if saved_import is None and path.exists():
            if not pool_path.exists() or read_jsonl(path) != rows:
                parser.error("The benchmark or pool does not match this run's existing import. Use a new run name.")
        if not pool_path.exists():
            atomic_json(pool_path, pool)
        if saved_import is None:
            atomic_json(import_manifest, {
                "identity": identity, "target_constitution": target, "scenario_count": len(rows),
            })
    else:
        from modules.client import OpenRouterClient, obtain_openrouter_key
        from modules.pipeline import run_pipeline
        from modules.stages import show_coverage

        if not pool_path.exists():
            atomic_json(pool_path, pool)
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
        rows = response_scenarios(results["existing"] + results["selected"], target)

    with TemporaryDirectory(dir=root) as directory:
        temporary = Path(directory) / path.name
        write_jsonl(temporary, rows)
        temporary.replace(path)
    print(f"Wrote {len(rows)} scenarios to {path}")


if __name__ == "__main__":
    main()
