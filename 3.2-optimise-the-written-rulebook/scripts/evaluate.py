"""Stage 3a: agreement on the held-out test split for the recovered rulebook and the baselines."""

import argparse
import json

import _bootstrap  # noqa: F401
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.data import by_split, load_choices
from rulebook_opt.interpreter import Interpreter
from rulebook_opt.metrics import evaluate_rulebook, markdown_table
from rulebook_opt.rulebooks import EMPTY_RULEBOOK, load_rulebook
from rulebook_opt.runtime import build_client, data_path, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r-model", help="evaluate with a different interpreter model (R-swap check)")
    ap.add_argument("--r-provider", help="provider for --r-model (default: same as the config)")
    ap.add_argument("--rulebook", action="append", default=[], metavar="NAME=PATH", help="extra rulebook to score")
    a = ap.parse_args()
    cfg = load_config(a.config)

    test = by_split(load_choices(data_path(cfg, "choices"))).get("test", [])
    if not test:
        raise SystemExit("no test choices; collect_choices must cover the test split")
    over = {}
    if a.r_model:
        over["model"] = a.r_model
        if a.r_provider:
            over["provider"] = a.r_provider
    client = build_client(cfg, "interpreter", overrides=over or None)
    interpreter = Interpreter(client, orders=int(cfg["interpreter"].get("orders", 1)))

    books: list[tuple[str, str]] = [("empty rulebook", EMPTY_RULEBOOK)]
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

    rows = [(name, evaluate_rulebook(interpreter, text, test)) for name, text in books]
    table = markdown_table(rows)
    tag = "" if not a.r_model else "_R-" + a.r_model.replace("/", "_")
    out = results_dir(cfg)
    (out / f"test{tag}.md").write_text(f"R = {client.cfg.model}\n\n{table}\n")
    (out / f"test{tag}.json").write_text(json.dumps({n: {k: v for k, v in r.items()} for n, r in rows}, indent=2))
    print(f"R = {client.cfg.model}\n\n{table}\n\ncalls: {client.stats}")


if __name__ == "__main__":
    main()
