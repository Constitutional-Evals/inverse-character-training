"""Does a constitution change the target's A/B choices at all? Run this before spending on GEPA.

Asks the same model each dilemma with no constitution and with each given constitution (in its system
prompt), both presentation orders, and reports how often the constitution flips the choice. If it almost
never does, recovery from choices cannot work for that constitution: its effect is in style, not decisions.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from rulebook_opt.collect import collect_choices, summarize
from rulebook_opt.config import ModelConfig
from rulebook_opt.data import load_scenarios
from rulebook_opt.llm import LLMClient
from rulebook_opt.rulebooks import load_constitution
from rulebook_opt.runtime import PROJECT_ROOT, _slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--constitution", action="append", required=True, metavar="PATH")
    ap.add_argument("--scenarios", default=str(PROJECT_ROOT / "data" / "scenarios.jsonl"))
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    scen = [s for s in load_scenarios(Path(a.scenarios)) if s.split == a.split]
    mc = ModelConfig(model=a.model, provider=a.provider, max_tokens=16)
    client = LLMClient(mc, cache_path=PROJECT_ROOT / ".cache" / "llm" / f"target-{_slug(a.provider)}-{_slug(a.model)}.jsonl")

    base = collect_choices(scen, client, orders=2, verbose=False)
    out = {"model": a.model, "split": a.split, "n": len(scen), "none": summarize(base), "constitutions": {}}
    b = {c.id: c.target_pick for c in base}
    for path in a.constitution:
        ch = collect_choices(scen, client, orders=2, constitution=load_constitution(path), verbose=False)
        both = [c for c in ch if c.target_pick is not None and b[c.id] is not None]
        flips = sum(c.target_pick != b[c.id] for c in both)
        out["constitutions"][Path(path).stem] = summarize(ch) | {"compared": len(both), "flips": flips,
                                                               "flip_rate": flips / len(both) if both else None}
        print(f"{Path(path).stem:24s} flips the choice on {flips}/{len(both)} dilemmas")
    dest = PROJECT_ROOT / "results" / "signal" / f"{_slug(a.model)}_{a.split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"calls {client.stats}; wrote {dest}")


if __name__ == "__main__":
    main()
