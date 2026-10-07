"""Free-form stage 1: record the target's own replies (resumable; every API reply is cached)."""

import argparse

import _bootstrap  # noqa: F401
from rulebook_opt.config import load_config, resolve_path
from rulebook_opt.data import load_scenarios
from rulebook_opt.freeform import collect_replies, save_replies
from rulebook_opt.rulebooks import load_constitution
from rulebook_opt.runtime import build_client, data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    limits = cfg["data"].get("limits") or {}
    by = {}
    for s in load_scenarios(data_path(cfg, "scenarios")):
        by.setdefault(s.split, []).append(s)
    scen = [s for split, ss in by.items() for s in ss[: limits.get(split, len(ss))]]

    t = cfg["target"]
    constitution = load_constitution(resolve_path(cfg, t["system_constitution"])) if t.get("system_constitution") else None
    client = build_client(cfg, "target")
    replies = collect_replies(scen, client, constitution)
    out = data_path(cfg, "replies")
    save_replies(out, replies)
    empty = sum(r.target_reply is None for r in replies)
    print(f"wrote {len(replies)} replies ({empty} empty) to {out}; calls {client.stats}")


if __name__ == "__main__":
    main()
