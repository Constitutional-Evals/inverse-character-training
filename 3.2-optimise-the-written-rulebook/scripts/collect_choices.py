"""Stage 1: record the target's choices (resumable; every API reply is cached)."""

import argparse
import json

import _bootstrap  # noqa: F401
from rulebook_opt.collect import collect_choices, summarize
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.data import load_scenarios, save_choices
from rulebook_opt.rulebooks import load_constitution
from rulebook_opt.runtime import build_client, data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="only the first N scenarios (smoke test)")
    a = ap.parse_args()
    cfg = load_config(a.config)

    scenarios = load_scenarios(data_path(cfg, "scenarios"))
    if a.limit:
        scenarios = scenarios[: a.limit]
    t = cfg["target"]
    constitution = None
    if t.get("system_constitution"):  # synthetic target: a model prompted with a known constitution
        constitution = load_constitution(resolve_path(cfg, t["system_constitution"]))
    client = build_client(cfg, "target")
    choices = collect_choices(scenarios, client, orders=int(t.get("orders", 2)), constitution=constitution)

    out = data_path(cfg, "choices")
    save_choices(out, choices)
    summary = summarize(choices) | {"api_calls": client.stats["calls"], "cache_hits": client.stats["cache_hits"]}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
