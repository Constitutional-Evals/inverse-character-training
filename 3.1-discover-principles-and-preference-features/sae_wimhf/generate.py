"""
Generate one response per (prompt, model). Resumable: rows already in the
output file are skipped, so you can stop and re-run at any time.

    python generate.py --models base sarcasm --out data/responses.jsonl
    python generate.py --models base sarcasm goodness humor loving --concurrency 32

Output rows:
  {"prompt_id", "category", "prompt", "model", "response", "finish_reason", "sampling"}
"""

import argparse
import asyncio
import time

from common import append_jsonl, chat, load_config, make_client, read_jsonl, resolve_model


async def wait_until_ready(client, model_id: str, max_wait: int = 1200) -> None:
    """Block until the server lists the model (covers Modal cold starts)."""
    start = time.time()
    while True:
        try:
            models = await client.models.list()
            ids = {m.id for m in models.data}
            if model_id in ids:
                return
            raise RuntimeError(f"'{model_id}' not served. Available: {sorted(ids)}")
        except RuntimeError:
            raise
        except Exception as e:
            if time.time() - start > max_wait:
                raise TimeoutError(f"Server not ready after {max_wait}s: {e}")
            print(f"  waiting for server ({type(e).__name__})...")
            await asyncio.sleep(15)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models.yaml")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--models", nargs="+", required=True, help="names from models.yaml")
    ap.add_argument("--out", default="data/responses.jsonl")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompts (for testing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sampling = cfg["sampling"]
    prompts = read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    clients, model_ids = {}, {}
    for name in args.models:
        endpoint, model_id = resolve_model(cfg, name)
        if endpoint not in clients:
            clients[endpoint] = make_client(cfg, endpoint)
        model_ids[name] = (endpoint, model_id)

    for name, (endpoint, model_id) in model_ids.items():
        await wait_until_ready(clients[endpoint], model_id)
    print("All models ready.")

    done = {(r["prompt_id"], r["model"]) for r in read_jsonl(args.out)}
    jobs = [(p, name) for name in args.models for p in prompts if (p["prompt_id"], name) not in done]
    print(f"{len(done)} rows already done, {len(jobs)} to generate.")

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    finished = 0

    async def run(p: dict, name: str):
        nonlocal finished
        endpoint, model_id = model_ids[name]
        async with sem:
            try:
                text, finish = await chat(
                    clients[endpoint], model_id,
                    [{"role": "user", "content": p["prompt"]}],
                    **sampling,
                )
            except Exception as e:
                print(f"FAILED {p['prompt_id']} / {name}: {e}")
                return
        async with write_lock:
            append_jsonl(args.out, {
                "prompt_id": p["prompt_id"],
                "category": p.get("category"),
                "prompt": p["prompt"],
                "model": name,
                "response": text,
                "finish_reason": finish,
                "sampling": sampling,
            })
            finished += 1
            if finished % 50 == 0 or finished == len(jobs):
                print(f"  {finished}/{len(jobs)}")

    await asyncio.gather(*(run(p, name) for p, name in jobs))

    truncated = sum(1 for r in read_jsonl(args.out) if r["finish_reason"] == "length")
    print(f"Done. {truncated} responses hit max_tokens (consider raising it if many).")


if __name__ == "__main__":
    asyncio.run(main())
