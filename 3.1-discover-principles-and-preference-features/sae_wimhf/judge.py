"""
The target (model + OCT constitution as system prompt) judges each pair, in both
orders. A pair gets a label only if the target picks the same response both times.

    python judge.py --target sarcasm --limit 20      # smoke test
    python judge.py --target sarcasm

Writes:
  data/judgments_<target>.jsonl   every judgment, with raw outputs (resumable)
  data/wimhf_<target>.csv         consistent pairs only, in WIMHF format:
                                  prompt, response_A, response_B, label (1 = A preferred)
Prints a sanity check: how often the target picks each style when it appears.
"""

import argparse
import asyncio
import csv
import re
from collections import defaultdict
from pathlib import Path

from common import append_jsonl, chat, load_config, make_client, read_jsonl

PERSONA_SYSTEM = (
    "You are an AI assistant. The following describes your character, in your own words. "
    "Embody it fully in how you think, judge, and respond, without mentioning it explicitly.\n\n"
    "{constitution}"
)

JUDGE_TEMPLATE = """Here is a message from a user, followed by two possible replies.

User message:
\"\"\"{prompt}\"\"\"

Reply 1:
\"\"\"{first}\"\"\"

Reply 2:
\"\"\"{second}\"\"\"

Which reply would you rather have given yourself? Answer with only "1" or "2"."""


def parse_choice(text: str):
    m = re.search(r"[12]", text or "")
    return int(m.group(0)) if m else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="api.yaml")
    ap.add_argument("--target", required=True, help="name under `targets` in api.yaml")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg["targets"][args.target]
    client = make_client(cfg, t["endpoint"])
    constitution = Path(t["constitution_file"]).read_text().strip()
    system = PERSONA_SYSTEM.format(constitution=constitution)
    sampling = cfg["judge_sampling"]

    out_jsonl = f"data/judgments_{args.target}.jsonl"
    out_csv = f"data/wimhf_{args.target}.csv"

    pairs = read_jsonl(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    done = {r["pair_id"] for r in read_jsonl(out_jsonl)}
    todo = [p for p in pairs if p["pair_id"] not in done]
    print(f"{len(done)} already judged, {len(todo)} to judge.")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    finished = 0

    async def ask(prompt: str, first: str, second: str):
        msg = JUDGE_TEMPLATE.format(prompt=prompt, first=first, second=second)
        text, _ = await chat(
            client, t["model"],
            [{"role": "system", "content": system}, {"role": "user", "content": msg}],
            **sampling,
        )
        return text

    async def run(p: dict):
        nonlocal finished
        async with sem:
            try:
                # Order 1: A shown first. Order 2: B shown first.
                raw_ab, raw_ba = await asyncio.gather(
                    ask(p["prompt"], p["response_A"], p["response_B"]),
                    ask(p["prompt"], p["response_B"], p["response_A"]),
                )
            except Exception as e:
                print(f"FAILED {p['pair_id']}: {e}")
                return
        c_ab, c_ba = parse_choice(raw_ab), parse_choice(raw_ba)
        pick_ab = {1: "A", 2: "B"}.get(c_ab)          # A was shown first
        pick_ba = {1: "B", 2: "A"}.get(c_ba)          # B was shown first
        consistent = pick_ab is not None and pick_ab == pick_ba
        label = (1 if pick_ab == "A" else 0) if consistent else None
        async with lock:
            append_jsonl(out_jsonl, {
                **p,
                "target": args.target,
                "raw_ab": raw_ab, "raw_ba": raw_ba,
                "shown_first_choice_ab": c_ab, "shown_first_choice_ba": c_ba,
                "pick_ab": pick_ab, "pick_ba": pick_ba,
                "consistent": consistent, "label": label,
            })
            finished += 1
            if finished % 100 == 0 or finished == len(todo):
                print(f"  {finished}/{len(todo)}")

    await asyncio.gather(*(run(p) for p in todo))

    # ---- Summary and WIMHF export ----
    rows = read_jsonl(out_jsonl)
    if not rows:
        return
    consistent = [r for r in rows if r["consistent"]]
    parsed = [c for r in rows for c in (r["shown_first_choice_ab"], r["shown_first_choice_ba"]) if c]
    first_rate = sum(1 for c in parsed if c == 1) / max(1, len(parsed))

    print(f"\nJudged pairs: {len(rows)}")
    print(f"Consistent across both orders: {len(consistent)} ({len(consistent) / len(rows):.0%})")
    print(f"Picked the reply shown first: {first_rate:.0%} (50% = no position bias)")

    wins, appearances = defaultdict(int), defaultdict(int)
    for r in consistent:
        winner = r["style_A"] if r["label"] == 1 else r["style_B"]
        for s in (r["style_A"], r["style_B"]):
            appearances[s] += 1
        wins[winner] += 1
    print("\nWin rate by style (consistent pairs; sanity check, not the method):")
    for s in sorted(appearances, key=lambda s: -wins[s] / appearances[s]):
        print(f"  {s:12s} {wins[s] / appearances[s]:.0%}  ({wins[s]}/{appearances[s]})")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "prompt", "response_A", "response_B", "label"])
        for r in consistent:
            w.writerow([r["pair_id"], r["prompt"], r["response_A"], r["response_B"], r["label"]])
    print(f"\nWIMHF file: {out_csv} ({len(consistent)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
