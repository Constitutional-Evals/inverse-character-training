"""Stage 2: optimise the rulebook with GEPA so that R matches the target (choices or free-form replies)."""

import argparse
import json

import _bootstrap  # noqa: F401
from rulebook_opt.adapter import COMPONENT
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.modes import build
from rulebook_opt.optimize import run_optimization
from rulebook_opt.rulebooks import load_rulebook
from rulebook_opt.runtime import build_client, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = cfg.get("gepa", {})

    s = build(cfg)
    train, val = s.splits.get("train", []), s.splits.get("val", [])
    if not train or not val:
        raise SystemExit(f"need train and val data, got {len(train)} train / {len(val)} val; run the collect script first")
    print(f"mode {s.mode}: train {len(train)}, val {len(val)} (test {len(s.splits.get('test', []))} held out)"
          + (f"; chance {s.chance:.3f}" if s.chance else ""))

    opt_client = build_client(cfg, "optimizer", cache=False)
    seed_spec = g.get("seed_rulebook")
    seed_rulebook = load_rulebook(None if seed_spec in (None, "empty") else resolve_path(cfg, seed_spec))

    out = results_dir(cfg)
    result = run_optimization(
        train, val, s.adapter, opt_client, seed_rulebook,
        max_metric_calls=int(g.get("max_metric_calls", 3000)),
        reflection_minibatch_size=int(g.get("reflection_minibatch_size", 8)),
        use_merge=bool(g.get("use_merge", True)),
        seed=int(g.get("seed", 0)),
        run_dir=str(out / "gepa"),
        mode=s.mode,
    )

    best = result.best_candidate[COMPONENT]
    (out / "best_rulebook.md").write_text(best + "\n", encoding="utf-8")
    history = [
        {"idx": i, "parent": (result.parents[i] if result.parents else None), "val_score": float(sc), "rulebook": c[COMPONENT]}
        for i, (c, sc) in enumerate(zip(result.candidates, result.val_aggregate_scores))
    ]
    (out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    seed_score = s.score(seed_rulebook, val)["agreement"]
    best_score = s.score(best, val)["agreement"]
    print(f"val score: seed {seed_score:.3f} -> best {best_score:.3f}  ({len(history)} candidates, {result.total_metric_calls} metric calls)")
    print("calls:", {c.cfg.model: c.stats for c in s.clients}, "| optimiser calls:", opt_client.stats["calls"])
    print(f"best rulebook written to {out / 'best_rulebook.md'}\n\n{best}")


if __name__ == "__main__":
    main()
