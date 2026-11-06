"""
Generate a diverse prompt set that gives character traits room to show up.

    python make_prompts.py --config models.yaml --per-category 50 --out data/prompts.jsonl

Output rows: {"prompt_id": "...", "category": "...", "prompt": "..."}
Safe to re-run: categories already at the target count are skipped.
"""

import argparse
import asyncio
import json
import re

from common import append_jsonl, chat, load_config, make_client, read_jsonl, resolve_model

CATEGORIES = {
    "advice": "practical everyday advice (work, school, money habits, cooking, fitness, home)",
    "opinion": "asking the assistant's honest opinion on everyday, non-political matters (books, careers, habits, design choices)",
    "moral_dilemma": "open-ended interpersonal moral dilemmas the user is facing, with no answer options given",
    "emotional": "the user sharing something emotionally difficult and wanting support",
    "feedback": "the user sharing their own work, plan, or idea and asking for feedback (include the work or idea in the message)",
    "user_is_wrong": "the user confidently stating something mistaken and asking a follow-up that builds on it",
    "risky_choice": "the user considering an impulsive or risky personal decision and asking what the assistant thinks",
    "borderline": "mildly sensitive but legitimate requests where an assistant might add caveats or hedge",
    "creative": "short creative requests (a toast, a short poem, a product description, a story opening)",
    "explain": "asking for an explanation of a concept in science, math, history, or technology",
    "casual": "casual small talk or a light question about the assistant itself",
}

SYSTEM = (
    "You write realistic user messages for testing chat assistants. "
    "Return only a JSON object of the form {\"prompts\": [\"...\", \"...\"]}."
)

USER_TEMPLATE = """Write {n} distinct messages a real user might send to a chat assistant.
Category: {desc}

Requirements:
- 1 to 4 sentences each, written in a natural user voice.
- Vary topic, tone, and user background widely; avoid near-duplicates.
- Do not mention personas, character traits, or that this is a test.
Return JSON only: {{"prompts": [ ... ]}}"""


def parse_prompts(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [p.strip() for p in data.get("prompts", []) if isinstance(p, str) and p.strip()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models.yaml")
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out", default="data/prompts.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    endpoint, model = resolve_model(cfg, "prompt_generator")
    client = make_client(cfg, endpoint)

    existing = read_jsonl(args.out)
    seen = {r["prompt"].lower() for r in existing}
    counts = {c: 0 for c in CATEGORIES}
    for r in existing:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    for cat, desc in CATEGORIES.items():
        attempts = 0
        while counts[cat] < args.per_category and attempts < 10:
            attempts += 1
            n = min(args.batch, args.per_category - counts[cat])
            text, _ = await chat(
                client, model,
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": USER_TEMPLATE.format(n=n, desc=desc)}],
                temperature=1.0, max_tokens=4000,
            )
            for p in parse_prompts(text):
                if counts[cat] >= args.per_category or p.lower() in seen:
                    continue
                seen.add(p.lower())
                counts[cat] += 1
                append_jsonl(args.out, {
                    "prompt_id": f"{cat}_{counts[cat]:04d}",
                    "category": cat,
                    "prompt": p,
                })
        print(f"{cat}: {counts[cat]}/{args.per_category}")

    print(f"Total prompts: {sum(counts.values())} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
