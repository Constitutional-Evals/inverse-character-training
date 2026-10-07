"""Scenarios (AIRiskDilemmas) and recorded target choices.

AIRiskDilemmas has two rows per dilemma, one per action ("Action 1: ...", "Action 2: ..."), so every
scenario is a binary choice. We keep the dilemma text and the two action texts; the dataset's
free-text value tags are not used by 3.2.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

AIRISK_DATASET = "kellycyy/AIRiskDilemmas"
AIRISK_REVISION = "8674d1f5844c3909b05e06d9f30bbc2b7c753f39"  # pinned, same as EigenBench
AIRISK_FILE = "model_eval.jsonl"
AIRISK_URL = f"https://huggingface.co/datasets/{AIRISK_DATASET}/resolve/{AIRISK_REVISION}/{AIRISK_FILE}"

_ACTION_RE = re.compile(r"^\s*Action\s*(\d+)\s*:\s*", re.I)


@dataclass
class Scenario:
    id: str
    dilemma: str
    actions: list[str]  # exactly two, in dataset order (Action 1, Action 2)
    split: str = ""


@dataclass
class Choice:
    """One scenario plus the target's pick, as an index into ``actions`` (None if unparsed)."""

    id: str
    dilemma: str
    actions: list[str]
    target_pick: int | None
    consistent: bool = True  # same action picked in both presentation orders (True if only one order was run)
    picks_by_order: dict[str, int | None] = field(default_factory=dict)
    split: str = ""


def scenario_id(dilemma: str) -> str:
    return hashlib.sha1(dilemma.encode("utf-8")).hexdigest()[:12]


def download_airisk(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(AIRISK_URL, timeout=120) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return dest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_scenarios(rows: Iterable[dict]) -> list[Scenario]:
    """Group action rows by dilemma; keep dilemmas with exactly two distinct actions."""
    by_dilemma: dict[str, dict[int, str]] = {}
    order: list[str] = []
    for r in rows:
        d, a = r.get("dilemma"), r.get("action")
        if not isinstance(d, str) or not isinstance(a, str) or not d.strip():
            continue
        m = _ACTION_RE.match(a)
        idx = int(m.group(1)) if m else len(by_dilemma.get(d, {})) + 1
        text = _ACTION_RE.sub("", a).strip()
        if d not in by_dilemma:
            by_dilemma[d] = {}
            order.append(d)
        by_dilemma[d].setdefault(idx, text)
    out = []
    for d in order:
        acts = by_dilemma[d]
        if len(acts) != 2:
            continue
        a1, a2 = (acts[k] for k in sorted(acts))
        if a1 == a2 or not a1 or not a2:
            continue
        out.append(Scenario(id=scenario_id(d), dilemma=d.strip(), actions=[a1, a2]))
    return out


def make_splits(scenarios: list[Scenario], n_train: int, n_val: int, n_test: int, seed: int = 0) -> list[Scenario]:
    """Sample n_train+n_val+n_test distinct dilemmas and assign splits (by dilemma, so no leakage)."""
    need = n_train + n_val + n_test
    if need > len(scenarios):
        raise ValueError(f"asked for {need} scenarios but only {len(scenarios)} available")
    rng = random.Random(seed)
    pool = list(scenarios)
    rng.shuffle(pool)
    out = []
    for i, sc in enumerate(pool[:need]):
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        out.append(Scenario(sc.id, sc.dilemma, sc.actions, split))
    return out


def load_scenarios(path: Path) -> list[Scenario]:
    return [Scenario(**r) for r in read_jsonl(path)]


def save_scenarios(path: Path, scenarios: Iterable[Scenario]) -> None:
    write_jsonl(path, (asdict(s) for s in scenarios))


def load_choices(path: Path, keep_inconsistent: bool = False, drop_unparsed: bool = True) -> list[Choice]:
    out = []
    for r in read_jsonl(path):
        c = Choice(**r)
        if drop_unparsed and c.target_pick is None:
            continue
        if not keep_inconsistent and not c.consistent:
            continue
        out.append(c)
    return out


def save_choices(path: Path, choices: Iterable[Choice]) -> None:
    write_jsonl(path, (asdict(c) for c in choices))


def by_split(choices: Iterable[Choice]) -> dict[str, list[Choice]]:
    out: dict[str, list[Choice]] = {}
    for c in choices:
        out.setdefault(c.split, []).append(c)
    return out
