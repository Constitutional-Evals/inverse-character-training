"""Rulebook text: loading known constitutions, rendering, and length checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

EMPTY_RULEBOOK = "(No constitution provided. Choose as you normally would.)"

_LABEL = re.compile(r"^\s*Criterion\s*\d+\s*(?:for\s+[^:]+)?:\s*", re.I)  # "Criterion 3 for Misalignment: "
_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def _clean(item: str) -> str:
    """Drop persona labels such as 'Criterion 3 for Misalignment:' (they would name the answer)."""
    item = _LABEL.sub("", item.strip())
    return item[:1].upper() + item[1:] if item else item


def render(items: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))


def load_constitution(path: str | Path) -> str:
    """Load a constitution as a numbered list.

    Accepts an EigenBench/OCT JSON list of criterion strings, a Constitutional-Evals JSON object with a
    ``criteria`` list (uses each ``comparative``), or a plain-text / markdown file (returned as is).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return raw.strip()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("criteria", [])
    items: list[str] = []
    for c in data:
        if isinstance(c, dict):
            c = c.get("comparative") or c.get("criterion") or c.get("text") or ""
        if isinstance(c, str) and c.strip():
            items.append(_clean(c))
    if not items:
        raise ValueError(f"no criteria found in {path}")
    return render(items)


def load_rulebook(spec: str | Path | None) -> str:
    """'empty' / None -> the empty rulebook; otherwise a constitution or text file."""
    if spec is None or str(spec).lower() == "empty":
        return EMPTY_RULEBOOK
    return load_constitution(spec)


def count_criteria(text: str) -> int | None:
    """Number of numbered/bulleted lines, or None if the text has no list structure."""
    n = sum(1 for line in text.splitlines() if _ITEM.match(line))
    return n or None


def violation(text: str, max_criteria: int | None, max_chars: int | None) -> str | None:
    """Reason a candidate rulebook breaks the format limits, or None if it is fine."""
    if not text.strip():
        return "the rulebook is empty"
    if max_chars and len(text) > max_chars:
        return f"the rulebook is {len(text)} characters long; the limit is {max_chars}"
    n = count_criteria(text)
    if max_criteria and n and n > max_criteria:
        return f"the rulebook has {n} principles; the limit is {max_criteria}"
    return None
