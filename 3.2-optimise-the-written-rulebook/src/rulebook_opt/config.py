"""Model endpoints and YAML run configs.

Every model role (target T, interpreter R, optimiser) is an OpenAI-compatible chat endpoint,
so OpenRouter, the Hugging Face router, a dedicated HF Inference Endpoint, vLLM and OpenAI
all work through one client. Pick one with ``provider`` or give ``base_url`` explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# provider -> (base_url, api-key environment variable)
PROVIDERS: dict[str, tuple[str | None, str | None]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "hf": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "custom": (None, None),  # vLLM, HF dedicated endpoint, ...: set base_url (and api_key_env if needed)
}


@dataclass
class ModelConfig:
    model: str
    provider: str = "openrouter"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    _KEYS = (
        "model",
        "provider",
        "base_url",
        "api_key_env",
        "temperature",
        "max_tokens",
        "extra_headers",
        "extra_body",
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        unknown = set(d) - set(cls._KEYS)
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        if "model" not in d:
            raise ValueError("model config needs a 'model'")
        return cls(**{k: d[k] for k in cls._KEYS if k in d})

    def resolve(self) -> tuple[str, str]:
        """Return (base_url, api_key). Raises a clear error if a required key is missing."""
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; choose from {sorted(PROVIDERS)}")
        default_url, default_env = PROVIDERS[self.provider]
        base_url = self.base_url or default_url
        if not base_url:
            raise ValueError(f"provider {self.provider!r} needs an explicit base_url")
        env = self.api_key_env or default_env
        if env:
            key = os.environ.get(env)
            if not key:
                raise RuntimeError(f"set the {env} environment variable for {self.provider}:{self.model}")
        else:
            key = "EMPTY"  # local servers that do not check keys
        return base_url, key


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    cfg["_dir"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: dict[str, Any], p: str | None) -> Path | None:
    """Resolve a path from a config: absolute stays; relative is taken from the project root."""
    if p is None:
        return None
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg["_dir"]).parent / path
