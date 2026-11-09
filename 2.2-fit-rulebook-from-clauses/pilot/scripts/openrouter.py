"""Minimal OpenRouter chat client with a content-addressed disk cache, so every step can be rerun and resumed.

Uses OPENROUTER_API_KEY only. Each response's usage (tokens, cost) is appended to <cache>/usage.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
_lock = threading.Lock()


class Client:
    def __init__(self, cache: Path, workers: int = 16):
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set")
        self.headers = {"Authorization": f"Bearer {key}", "X-Title": "constitution-recovery"}
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def chat(self, model: str, messages: list[dict], **params) -> dict:
        """Return {"text", "logprobs", "usage"}; cached on (model, messages, params)."""
        payload = {"model": model, "messages": messages, "usage": {"include": True}, **params}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(6):
            try:
                r = requests.post(URL, headers=self.headers, json=payload, timeout=(30, 600))
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 60))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 60))
                continue
            r.raise_for_status()
            obj = r.json()
            if "error" in obj or not obj.get("choices"):
                time.sleep(min(2 ** attempt, 60))
                continue
            choice = obj["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                time.sleep(min(2 ** attempt, 60))
                continue
            out = {"text": text, "logprobs": choice.get("logprobs"), "usage": obj.get("usage", {}), "model": obj.get("model")}
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(out))
            with _lock, (self.cache / "usage.jsonl").open("a") as f:
                f.write(json.dumps({"time": time.time(), "model": model, "usage": out["usage"]}) + "\n")
            return out
        raise RuntimeError(f"{model}: no usable response after retries")

    def map(self, fn, items: list) -> list:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(fn, items))


def spent(cache: Path) -> float:
    path = Path(cache) / "usage.jsonl"
    if not path.exists():
        return 0.0
    return sum(float(json.loads(line)["usage"].get("cost") or 0) for line in path.open())


def parse_json(text: str) -> dict:
    """The first JSON object in a model reply (tolerates code fences and prose around it)."""
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0 and start >= 0:
            return json.loads(text[start:i + 1])
    raise ValueError("no JSON object in reply")
