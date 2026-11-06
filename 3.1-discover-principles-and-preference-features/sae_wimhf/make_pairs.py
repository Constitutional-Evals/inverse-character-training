"""
Build response pairs from the pool: two different styles answering the same prompt,
with similar lengths so the target judges on character, not effort.

    python make_pairs.py --pairs-per-prompt 6

Output rows: {"pair_id", "prompt_id", "category", "prompt",
              "response_A", "response_B", "style_A", "style_B"}
"""

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from common import read_jsonl


def n_words(text: str) -> int:
    return max(1, len(text.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool.jsonl")
    ap.add_argument("--out", default="data/pairs.jsonl")
    ap.add_argument("--pairs-per-prompt", type=int, default=6)
    ap.add_argument("--max-length-ratio", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_prompt = defaultdict(list)
    for r in read_jsonl(args.pool):
        if r["response"].strip() and r["finish_reason"] != "length":
            by_prompt[r["prompt_id"]].append(r)

    pairs, skipped_prompts = [], 0
    for prompt_id, rows in by_prompt.items():
        candidates = []
        for a, b in itertools.combinations(rows, 2):
            la, lb = n_words(a["response"]), n_words(b["response"])
            if max(la, lb) / min(la, lb) <= args.max_length_ratio:
                candidates.append((a, b))
        if not candidates:
            skipped_prompts += 1
            continue
        for a, b in rng.sample(candidates, min(args.pairs_per_prompt, len(candidates))):
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append({
                "pair_id": f"{prompt_id}__{a['style']}__{b['style']}",
                "prompt_id": prompt_id,
                "category": a.get("category"),
                "prompt": a["prompt"],
                "response_A": a["response"],
                "response_B": b["response"],
                "style_A": a["style"],
                "style_B": b["style"],
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    style_counts = defaultdict(int)
    for p in pairs:
        style_counts[p["style_A"]] += 1
        style_counts[p["style_B"]] += 1
    print(f"{len(pairs)} pairs from {len(by_prompt)} prompts ({skipped_prompts} prompts had no length-matched pair)")
    print("Appearances per style:", dict(sorted(style_counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
