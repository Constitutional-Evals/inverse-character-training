"""Shared helpers for the scripts: building clients from a YAML config."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import ModelConfig, resolve_path
from .llm import LLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def build_client(cfg: dict[str, Any], section: str, cache: bool = True, overrides: dict[str, Any] | None = None) -> LLMClient:
    sec = dict(cfg[section])
    if overrides:
        sec.update(overrides)
    mc = ModelConfig.from_dict({k: v for k, v in sec.items() if k in ModelConfig._KEYS})
    cache_path = PROJECT_ROOT / ".cache" / "llm" / f"{section}-{_slug(mc.provider)}-{_slug(mc.model)}.jsonl" if cache else None
    return LLMClient(mc, cache_path=cache_path, max_workers=int(sec.get("max_workers", 8)))


def results_dir(cfg: dict[str, Any]) -> Path:
    d = PROJECT_ROOT / "results" / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_path(cfg: dict[str, Any], key: str) -> Path:
    p = resolve_path(cfg, cfg["data"][key])
    assert p is not None
    return p
