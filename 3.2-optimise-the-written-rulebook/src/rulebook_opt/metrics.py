"""Agreement scoring and reporting."""

from __future__ import annotations

import numpy as np

from .data import Choice
from .interpreter import Interpreter, agreement


def bootstrap_ci(scores: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not scores:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def evaluate_rulebook(interpreter: Interpreter, rulebook: str, choices: list[Choice]) -> dict:
    """Agreement of R (prompted with ``rulebook``) with the target on ``choices``."""
    decisions = interpreter.decide(rulebook, choices)
    scores = [agreement(d, c.target_pick) for d, c in zip(decisions, choices)]
    parsed = [p is not None for d in decisions for p in d.picks]
    lo, hi = bootstrap_ci(scores)
    return {
        "n": len(choices),
        "agreement": float(np.mean(scores)) if scores else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "parse_rate": float(np.mean(parsed)) if parsed else float("nan"),
        "scores": scores,
    }


def markdown_table(rows: list[tuple[str, dict]]) -> str:
    lines = ["| rulebook | agreement | 95% CI | parsed | n |", "|---|---|---|---|---|"]
    for name, r in rows:
        lines.append(
            f"| {name} | {r['agreement']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['parse_rate']:.2f} | {r['n']} |"
        )
    return "\n".join(lines)
