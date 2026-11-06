import asyncio
import json
import os
import random
from pathlib import Path

import yaml
from openai import AsyncOpenAI


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_client(cfg: dict, endpoint_name: str) -> AsyncOpenAI:
    ep = cfg["endpoints"][endpoint_name]
    key = os.environ.get(ep["api_key_env"])
    if not key:
        raise RuntimeError(f"Environment variable {ep['api_key_env']} is not set")
    # Long timeout: the first request may wait for a Modal cold start.
    return AsyncOpenAI(base_url=ep["base_url"], api_key=key, timeout=600, max_retries=0)


def resolve_model(cfg: dict, name: str) -> tuple[str, str]:
    """Return (endpoint_name, model_id) for a model name in models.yaml."""
    if name == "prompt_generator":
        m = cfg["prompt_generator"]
    else:
        if name not in cfg["models"]:
            raise KeyError(f"Model '{name}' not found in models.yaml")
        m = cfg["models"][name]
    return m["endpoint"], m["model"]


async def chat(client: AsyncOpenAI, model: str, messages: list[dict],
               retries: int = 6, **sampling) -> tuple[str, str]:
    """Chat completion with exponential backoff. Returns (text, finish_reason)."""
    for attempt in range(retries):
        try:
            r = await client.chat.completions.create(model=model, messages=messages, **sampling)
            choice = r.choices[0]
            return choice.message.content or "", choice.finish_reason
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = min(60, 2 ** attempt) + random.random()
            print(f"  [{model}] {type(e).__name__}: {e}. Retrying in {wait:.1f}s")
            await asyncio.sleep(wait)


def read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
