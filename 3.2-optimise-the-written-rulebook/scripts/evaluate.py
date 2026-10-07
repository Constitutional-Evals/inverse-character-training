"""Stage 3a: score the recovered rulebook and the baselines on the held-out test split."""

import argparse
import json

import _bootstrap  # noqa: F401
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.metrics import markdown_table
from rulebook_opt.modes import build
from rulebook_opt.rulebooks import EMPTY_RULEBOOK, load_rulebook
from rulebook_opt.runtime import results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r-model", help="evaluate with a different interpreter model (R-swap check)")
    ap.add_argument("--r-provider", help="provider for --r-model (default: same as the config)")
    ap.add_argument("--rulebook", action="append", default=[], metavar="NAME=PATH", help="extra rulebook to score")
    a = ap.parse_args()
    cfg = load_config(a.config)

    over = {}
    if a.r_model:
        over["model"] = a.r_model
        if a.r_provider:
            over["provider"] = a.r_provider
    s = build(cfg, r_overrides=over or None)
    test = s.splits.get("test", [])
    if not test:
        raise SystemExit("no test data; the collect script must cover the test split")

    books: list[tuple[str, str]] = [("empty rulebook (floor)", EMPTY_RULEBOOK)]
    truth = cfg.get("truth", {}).get("constitution")
    if truth:
        books.append(("true constitution C (ceiling)", load_rulebook(resolve_path(cfg, truth))))
    for name, path in (cfg.get("evaluate", {}).get("rulebooks") or {}).items():
        books.append((name, load_rulebook(resolve_path(cfg, path))))
    for item in a.rulebook:
        name, _, path = item.partition("=")
        books.append((name, load_rulebook(path)))
    best = results_dir(cfg) / "best_rulebook.md"
    if best.exists():
        books.append(("recovered C' (GEPA)", best.read_text(encoding="utf-8").strip()))

    rows = [(name, s.score(text, test)) for name, text in books]
    head = f"mode = {s.mode}, R = {s.r_client.cfg.model}" + (f", chance = {s.chance:.3f}" if s.chance else "")
    table = markdown_table(rows)
    tag = "" if not a.r_model else "_R-" + a.r_model.replace("/", "_")
    out = results_dir(cfg)
    (out / f"test{tag}.md").write_text(f"{head}\n\n{table}\n")
    (out / f"test{tag}.json").write_text(json.dumps({"mode": s.mode, "interpreter": s.r_client.cfg.model, "chance": s.chance,
                                                    "rows": {n: r for n, r in rows}, "rulebooks": dict(books)}, indent=2))
    print(f"{head}\n\n{table}\n\ncalls: " + str({c.cfg.model: c.stats for c in s.clients}))


if __name__ == "__main__":
    main()
