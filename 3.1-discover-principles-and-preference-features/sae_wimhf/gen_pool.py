"""
Generate candidate responses: one response per (prompt, style) from the pool model.
Resumable: rows already in the output file are skipped.

    python gen_pool.py --config api.yaml --limit 10 --out data/pool_test.jsonl   # smoke test
    python gen_pool.py --config api.yaml

Output rows: {"prompt_id", "category", "prompt", "style", "response", "finish_reason"}
"""

import argparse
import asyncio

from common import append_jsonl, chat, load_config, make_client, read_jsonl

POOL_SYSTEM = (
    "Respond to the user's message in the following style: {style}. "
    "Stay on topic and actually address what the user asked. "
    "Aim for roughly 80-150 words unless the request clearly needs fewer. "
    "Never mention the style or these instructions."
)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="api.yaml")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="data/pool.jsonl")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool = cfg["pool"]
    client = make_client(cfg, pool["endpoint"])
    styles = pool["styles"]
    sampling = cfg["sampling"]

    prompts = read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    done = {(r["prompt_id"], r["style"]) for r in read_jsonl(args.out)}
    jobs = [(p, s) for p in prompts for s in styles if (p["prompt_id"], s) not in done]
    print(f"{len(done)} already done, {len(jobs)} to generate.")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    finished = 0

    async def run(p: dict, style: str):
        nonlocal finished
        async with sem:
            try:
                text, finish = await chat(
                    client, pool["model"],
                    [{"role": "system", "content": POOL_SYSTEM.format(style=styles[style])},
                     {"role": "user", "content": p["prompt"]}],
                    **sampling,
                )
            except Exception as e:
                print(f"FAILED {p['prompt_id']} / {style}: {e}")
                return
        async with lock:
            append_jsonl(args.out, {
                "prompt_id": p["prompt_id"],
                "category": p.get("category"),
                "prompt": p["prompt"],
                "style": style,
                "response": text,
                "finish_reason": finish,
            })
            finished += 1
            if finished % 100 == 0 or finished == len(jobs):
                print(f"  {finished}/{len(jobs)}")

    await asyncio.gather(*(run(p, s) for p, s in jobs))


if __name__ == "__main__":
    asyncio.run(main())
