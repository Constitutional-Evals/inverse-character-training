"""Build the train/val/test data, adapter and scorer for a config's mode ('choice' or 'freeform')."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapter import RulebookAdapter
from .config import resolve_path
from .data import by_split, load_choices
from .freeform import FreeformAdapter, Lineup, evaluate_freeform, load_replies
from .interpreter import Interpreter
from .llm import LLMClient
from .metrics import evaluate_rulebook
from .rulebooks import load_constitution
from .runtime import build_client, data_path


@dataclass
class Setup:
    mode: str
    splits: dict[str, list]
    adapter: Any
    score: Callable[[str, list], dict]  # (rulebook, items) -> result dict with "agreement"
    r_client: LLMClient
    clients: list[LLMClient]
    chance: float | None = None


def build(cfg: dict[str, Any], r_overrides: dict[str, Any] | None = None) -> Setup:
    mode = cfg.get("mode", "choice")
    g = cfg.get("gepa", {})
    r_client = build_client(cfg, "interpreter", overrides=r_overrides)
    if mode == "choice":
        splits = by_split(load_choices(data_path(cfg, "choices")))
        interp = Interpreter(r_client, orders=int(cfg["interpreter"].get("orders", 1)))
        adapter = RulebookAdapter(interp, g.get("max_criteria", 15), g.get("max_chars", 3500))
        return Setup(mode, splits, adapter, lambda rb, items: evaluate_rulebook(interp, rb, items), r_client, [r_client])
    if mode == "freeform":
        splits = load_replies(data_path(cfg, "replies"), cfg["data"].get("limits"))
        judge = build_client(cfg, "judge")
        lu = cfg.get("lineup", {})
        decoys: dict[str, str | None] = {}
        if lu.get("include_default", True):
            decoys["default"] = None
        truth = cfg.get("truth", {}).get("constitution")
        for p in lu.get("decoys", []):
            path = resolve_path(cfg, p)
            if truth and Path(path).resolve() == Path(resolve_path(cfg, truth)).resolve():
                raise ValueError("the true constitution must not be a decoy")
            decoys[Path(path).stem] = load_constitution(path)
        lineup = Lineup(r_client, judge, decoys, seed=int(g.get("seed", 0)))
        adapter = FreeformAdapter(lineup, g.get("max_criteria", 15), g.get("max_chars", 3500))
        return Setup(mode, splits, adapter, lambda rb, items: evaluate_freeform(lineup, rb, items), r_client,
                     [r_client, judge], chance=1 / lineup.size)
    raise ValueError(f"unknown mode {mode!r}")
