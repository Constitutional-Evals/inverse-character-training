"""Shared helpers for stages 3 and 4: OpenRouter calls with a disk cache, usage logging, JSON parsing."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests
from openai import OpenAI

OPENROUTER = "https://openrouter.ai/api/v1"


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    return key


def check_models(models) -> None:
    """Fail early, with a clear message, if a model id is not on OpenRouter."""
    resp = requests.get(f"{OPENROUTER}/models", timeout=60)
    resp.raise_for_status()
    ids = {m["id"] for m in resp.json()["data"]}
    missing = sorted(m for m in models if m not in ids)
    if missing:
        raise SystemExit(
            "Not available on OpenRouter: " + ", ".join(missing)
            + "\nPick another model id from openrouter.ai/models and pass it with the matching flag."
        )


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict:
    """Parse the first {...} object in a model reply (tolerates code fences and surrounding prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


class LLM:
    """Chat calls through OpenRouter. Every valid reply is cached on disk, so reruns resume for free."""

    def __init__(self, cache_dir, usage_path, max_calls: int):
        self.client = OpenAI(base_url=OPENROUTER, api_key=api_key(), timeout=300, max_retries=0)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_path = Path(usage_path)
        self.max_calls = max_calls
        self.calls = 0
        self.cost = 0.0
        self.lock = threading.Lock()

    def chat(self, model, messages, *, temperature, max_tokens, tag, validate=None, attempts=4):
        """Return the reply text, or validate(text) if given. validate raises ValueError on a bad reply,
        which triggers a retry; only replies that pass are cached. tag keeps otherwise-identical
        requests apart (for example repeat ratings)."""
        key = hashlib.sha256(
            json.dumps([model, messages, temperature, max_tokens, tag], sort_keys=True).encode()
        ).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
            return validate(text) if validate else text

        last = None
        for attempt in range(attempts):
            with self.lock:
                if self.calls >= self.max_calls:
                    raise RuntimeError("call limit reached (--max-calls); rerun with a higher value, finished calls are cached")
                self.calls += 1
            try:
                out = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                    extra_body={"usage": {"include": True}},
                )
            except Exception as exc:  # network errors, rate limits, provider errors
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            self._log(model, tag, out.usage)
            text = (out.choices[0].message.content or "").strip() if out.choices else ""
            if not text:
                last = "empty reply"
                continue
            try:
                result = validate(text) if validate else text
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                last = f"invalid reply: {exc}"
                continue
            path.write_text(json.dumps({"model": model, "tag": tag, "text": text}, ensure_ascii=False), encoding="utf-8")
            return result
        raise RuntimeError(f"{model} failed on {tag}: {last}")

    def _log(self, model, tag, usage) -> None:
        u = usage.model_dump() if usage is not None else {}
        cost = float(u.get("cost") or 0.0)
        with self.lock:
            self.cost += cost
            with self.usage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "time": time.time(), "model": model, "tag": tag,
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "cost": cost,
                }) + "\n")
