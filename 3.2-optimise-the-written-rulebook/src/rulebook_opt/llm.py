"""OpenAI-compatible chat client with a disk cache, retries and thread-pool batching.

Calls are cached on (provider, model, messages, temperature, max_tokens), so a re-run, a resumed
run, or a second candidate that produces an identical prompt costs nothing. Empty replies are
not cached. Failures are never silently scored: transient errors are retried with backoff and
then raised; auth / missing-model errors are raised immediately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .config import ModelConfig

log = logging.getLogger(__name__)

Messages = list[dict[str, str]]
# transport(messages, temperature, max_tokens) -> reply text
Transport = Callable[[Messages, float, int], str]


class LLMError(RuntimeError):
    pass


class LLMFatalError(LLMError):
    """Not worth retrying: bad key, unknown model, no permission."""


def openai_transport(cfg: ModelConfig) -> tuple[Transport, tuple[type, ...], tuple[type, ...]]:
    import openai
    from openai import OpenAI

    base_url, key = cfg.resolve()
    client = OpenAI(
        base_url=base_url,
        api_key=key,
        default_headers=cfg.extra_headers or None,
        timeout=120,
        max_retries=0,  # retries are handled here
    )

    def call(messages: Messages, temperature: float, max_tokens: int) -> str:
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=cfg.extra_body or None,
            )
        except openai.BadRequestError as e:  # e.g. content filter on one item: treat as empty reply
            log.warning("bad request for %s: %s", cfg.model, str(e)[:200])
            return ""
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    retryable = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )
    fatal = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)
    return call, retryable, fatal


class DiskCache:
    def __init__(self, path: Path | None):
        self.path = path
        self._d: dict[str, str] = {}
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # blank, or torn by a concurrent writer: just a cache miss
                        continue
                    self._d[rec["k"]] = rec["v"]

    def get(self, key: str) -> str | None:
        return self._d.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._d[key] = value
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


class LLMClient:
    def __init__(
        self,
        cfg: ModelConfig,
        cache_path: Path | None = None,
        transport: Transport | None = None,
        max_workers: int = 8,
        retries: int = 6,
    ):
        self.cfg = cfg
        self.cache = DiskCache(cache_path)
        self.max_workers = max_workers
        self.retries = retries
        self.stats = {"calls": 0, "cache_hits": 0}
        self._stats_lock = threading.Lock()
        if transport is not None:
            self._transport, self._retryable, self._fatal = transport, (Exception,), ()
        else:
            self._transport, self._retryable, self._fatal = openai_transport(cfg)

    def _key(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        blob = json.dumps(
            [self.cfg.provider, self.cfg.model, messages, temperature, max_tokens], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, messages: Messages, temperature: float | None = None, max_tokens: int | None = None) -> str:
        t = self.cfg.temperature if temperature is None else temperature
        m = self.cfg.max_tokens if max_tokens is None else max_tokens
        key = self._key(messages, t, m)
        hit = self.cache.get(key)
        if hit is not None:
            with self._stats_lock:
                self.stats["cache_hits"] += 1
            return hit
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                out = self._transport(messages, t, m)
                with self._stats_lock:
                    self.stats["calls"] += 1
                if out:
                    self.cache.put(key, out)
                return out
            except self._fatal as e:
                raise LLMFatalError(f"{self.cfg.provider}:{self.cfg.model}: {e}") from e
            except self._retryable as e:
                last = e
                wait = min(60.0, 2.0**attempt) + random.random()
                log.warning("retry %d/%d for %s after %s", attempt + 1, self.retries, self.cfg.model, type(e).__name__)
                time.sleep(wait)
        raise LLMError(f"{self.cfg.provider}:{self.cfg.model} failed after {self.retries} attempts: {last}")

    def complete_many(self, batch: list[Messages], **kw) -> list[str]:
        if len(batch) <= 1 or self.max_workers <= 1:
            return [self.complete(m, **kw) for m in batch]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda m: self.complete(m, **kw), batch))
