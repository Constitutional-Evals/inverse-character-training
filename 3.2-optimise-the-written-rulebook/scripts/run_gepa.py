"""Stage 2: optimise the rulebook with GEPA so that R's picks match the target's."""

import argparse
import json

import _bootstrap  # noqa: F401
from rulebook_opt.adapter import COMPONENT, RulebookAdapter
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.data import by_split, load_choices
from rulebook_opt.interpreter import Interpreter
from rulebook_opt.metrics import evaluate_rulebook
from rulebook_opt.optimize import run_optimization
from rulebook_opt.rulebooks import load_rulebook
from rulebook_opt.runtime import build_client, data_path, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = cfg.get("gepa", {})

    splits = by_split(load_choices(data_path(cfg, "choices")))
    train, val = splits.get("train", []), splits.get("val", [])
    if not train or not val:
        raise SystemExit(f"need train and val choices, got {len(train)} train / {len(val)} val; run collect_choices first")
    print(f"train {len(train)}, val {len(val)} (test {len(splits.get('test', []))} is held out)")

    r_client = build_client(cfg, "interpreter")
    interpreter = Interpreter(r_client, orders=int(cfg["interpreter"].get("orders", 1)))
    adapter = RulebookAdapter(interpreter, g.get("max_criteria", 15), g.get("max_chars", 3500))
    opt_client = build_client(cfg, "optimizer", cache=False)
    seed_rulebook = load_rulebook(resolve_path(cfg, g.get("seed_rulebook")) if g.get("seed_rulebook") not in (None, "empty") else None)

    out = results_dir(cfg)
    result = run_optimization(
        train, val, adapter, opt_client, seed_rulebook,
        max_metric_calls=int(g.get("max_metric_calls", 3000)),
        reflection_minibatch_size=int(g.get("reflection_minibatch_size", 8)),
        use_merge=bool(g.get("use_merge", True)),
        seed=int(g.get("seed", 0)),
        run_dir=str(out / "gepa"),
    )

    best = result.best_candidate[COMPONENT]
    (out / "best_rulebook.md").write_text(best + "\n", encoding="utf-8")
    history = [
        {"idx": i, "val_score": float(s), "rulebook": c[COMPONENT]}
        for i, (c, s) in enumerate(zip(result.candidates, result.val_aggregate_scores))
    ]
    (out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    seed_score = evaluate_rulebook(interpreter, seed_rulebook, val)["agreement"]
    best_score = evaluate_rulebook(interpreter, best, val)["agreement"]
    print(f"val agreement: seed {seed_score:.3f} -> best {best_score:.3f}  ({len(history)} candidates)")
    print(f"R calls: {r_client.stats}; optimiser calls: {opt_client.stats['calls']}")
    print(f"best rulebook written to {out / 'best_rulebook.md'}\n\n{best}")


if __name__ == "__main__":
    main()
