"""
Turn 3.2's recorded target choices (AIRiskDilemmas, A/B in both orders) into a WIMHF
dataset, and write a matching WIMHF config.

The 3.2 *test* split is left out on purpose: the SAE and regression never see it, so
3.2's evaluate.py can score the 3.1 rulebook on held-out dilemmas.

    python from_3p2_choices.py \
        --choices <3.2 folder>/data/m0_synthetic/choices.jsonl \
        --name m0_synthetic

Writes:
  data/wimhf_3p2_<name>.jsonl       prompt, response_A, response_B, label (1 = target chose A)
  configs/wimhf_3p2_<name>.json     config for WIMHF's scripts/run_wimhf.py
"""

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_text(action: str, dilemma: str, context: str) -> str:
    """WIMHF embeds each response alone, so short actions need their situation attached.
    The action comes first so WIMHF's word truncation only ever cuts the situation."""
    if context == "action":
        return action.strip()
    return f"Chosen action: {action.strip()}\n\nSituation: {dilemma.strip()}"


def make_config(name: str, data_path: Path, n_rows: int, args) -> dict:
    m = args.sae_m
    return {
        "dataset": {
            "name": f"ICT_3p2_{name}",
            "path": str(data_path.resolve()),
            "split_columns": ["prompt"],
            "train_split_size": 0.8,
            "split_random_seed": 42,
            "random_seed": 42,
            "max_words_prompt": 256,
            "max_words_response": 160,
        },
        "embedding": {"model": args.embedding_model, "n_workers": 2},
        "sae": {
            "M": m,
            "K": args.sae_k,
            "prefix_lengths": [max(1, m // 4), m],
            "batch_size": 64,
            "n_epochs": 100,
            "min_epochs": 20,
            "patience": 5,
            "learning_rate": 0.0005,
            "dead_neuron_threshold_steps": 64,
        },
        "interpretation": {
            "interpreter_model": args.interpreter_model,
            "annotator_model": args.annotator_model,
            "abbreviator_model": args.annotator_model,
            "n_candidates": 5,
            "interpret_n_examples": 10,
            "interpret_sampling_kwargs": {
                "percentile_bin": [90, 100],
                "nonzero_only": True,
                "swap_negative_response_pairs": True,
            },
            # small dataset: score each label on at most a third of the rows
            "scoring_n_examples": max(20, min(100, n_rows // 3)),
            "scoring_sampling_kwargs": {
                "high_percentile": [50, 100],
                "low_percentile": [0, 50],
                "nonzero_only": True,
            },
            "min_correlation": 0.0,
            "p_value_threshold": 0.05 / m,  # Bonferroni over the M features
            "n_workers_interpretation": 16,
            "n_workers_annotation": 32,
            "timeout": 120.0,
            "max_interpretation_tokens": 20000,
            # OpenAI's `reasoning_effort` parameter is left out on OpenRouter, which may not accept it
            "completion_kwargs": {} if args.openrouter else {"reasoning_effort": "low"},
        },
        "selection": {
            "controls": ["length_delta"],
            "lasso_top_k": [5, 10],
            "max_samples_lasso": None,
            "classification": True,
            "standardize": True,
        },
        "runtime": {
            "checkpoint_dir": f"outputs/{name}/checkpoints",
            "cache_dir": f"outputs/{name}/emb_cache",
            "retrain_sae": True,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choices", required=True, help="3.2 choices.jsonl")
    ap.add_argument("--name", required=True, help="run name, e.g. m0_synthetic")
    ap.add_argument("--context", choices=["in_context", "action"], default="in_context",
                    help="embed the action with its dilemma (default) or the action alone")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sae-m", type=int, default=16, help="number of SAE features")
    ap.add_argument("--sae-k", type=int, default=3, help="active features per pair")
    ap.add_argument("--openrouter", action="store_true",
                    help="write OpenRouter model names (run WIMHF with OPENAI_BASE_URL set to OpenRouter)")
    ap.add_argument("--embedding-model", default=None, help="default: text-embedding-3-small")
    ap.add_argument("--interpreter-model", default=None, help="default: gpt-5")
    ap.add_argument("--annotator-model", default=None, help="default: gpt-5-mini")
    args = ap.parse_args()

    prefix = "openai/" if args.openrouter else ""
    args.embedding_model = args.embedding_model or f"{prefix}text-embedding-3-small"
    args.interpreter_model = args.interpreter_model or f"{prefix}gpt-5"
    args.annotator_model = args.annotator_model or f"{prefix}gpt-5-mini"

    rng = random.Random(args.seed)
    rows, n_test, n_dropped = [], 0, 0
    for c in read_jsonl(Path(args.choices)):
        if c.get("target_pick") is None or not c.get("consistent", True):
            n_dropped += 1
            continue
        if c.get("split") == "test":
            n_test += 1
            continue
        # Random A/B assignment, so "A" never systematically means AIRisk's Action 1.
        a_idx = 1 if rng.random() < 0.5 else 0
        b_idx = 1 - a_idx
        rows.append({
            "pair_id": c["id"],
            "prompt": c["dilemma"],
            "response_A": action_text(c["actions"][a_idx], c["dilemma"], args.context),
            "response_B": action_text(c["actions"][b_idx], c["dilemma"], args.context),
            "label": 1 if c["target_pick"] == a_idx else 0,
            "split_3p2": c.get("split", ""),
            "action_index_A": a_idx,
        })

    if not rows:
        raise SystemExit("No usable choices (all unparsed, inconsistent, or test).")

    data_path = Path("data") / f"wimhf_3p2_{args.name}.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg_path = Path("configs") / f"wimhf_3p2_{args.name}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(make_config(args.name, data_path, len(rows), args), indent=2))

    share_a = sum(r["label"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} pairs to {data_path} (held out {n_test} test, dropped {n_dropped} unparsed/inconsistent)")
    print(f"Share labelled A: {share_a:.0%} (near 50% expected after random assignment)")
    print(f"Config: {cfg_path}")
    if len(rows) < 200:
        print("Warning: under 200 pairs is thin for an SAE. Consider more train/val dilemmas in 3.2's make_scenarios.py.")


if __name__ == "__main__":
    main()
