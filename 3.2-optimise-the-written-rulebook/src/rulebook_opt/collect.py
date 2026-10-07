"""Stage 1: record the target's choices on the scenarios."""

from __future__ import annotations

import sys

from .data import Choice, Scenario
from .llm import LLMClient
from .prompts import parse_choice, target_messages

ORDERS = {"fwd": (0, 1), "rev": (1, 0)}  # which action is shown as A, which as B


def collect_choices(
    scenarios: list[Scenario],
    client: LLMClient,
    orders: int = 2,
    constitution: str | None = None,
    chunk: int = 50,
    verbose: bool = True,
) -> list[Choice]:
    """Ask the target each scenario in ``orders`` presentation orders.

    With two orders, a scenario is ``consistent`` only if the target picks the same action both times;
    inconsistent ones mostly reflect position bias and are dropped by default when loading.
    ``constitution`` is for a synthetic target (a model prompted with a known constitution).
    """
    names = list(ORDERS)[:orders]
    out: list[Choice] = []
    for start in range(0, len(scenarios), chunk):
        part = scenarios[start : start + chunk]
        reqs = [
            target_messages(sc.dilemma, sc.actions[ORDERS[n][0]], sc.actions[ORDERS[n][1]], constitution)
            for sc in part
            for n in names
        ]
        replies = client.complete_many(reqs)
        for i, sc in enumerate(part):
            picks: dict[str, int | None] = {}
            for j, n in enumerate(names):
                letter = parse_choice(replies[i * len(names) + j])
                picks[n] = None if letter is None else ORDERS[n][letter]
            vals = list(picks.values())
            if len(names) == 2:
                consistent = vals[0] is not None and vals[0] == vals[1]
                pick = vals[0] if consistent else None
            else:
                consistent, pick = vals[0] is not None, vals[0]
            out.append(Choice(sc.id, sc.dilemma, sc.actions, pick, consistent, picks, sc.split))
        if verbose:
            print(f"collected {min(start + chunk, len(scenarios))}/{len(scenarios)}", file=sys.stderr)
    return out


def summarize(choices: list[Choice]) -> dict:
    n = len(choices)
    parsed = [c for c in choices if any(v is not None for v in c.picks_by_order.values())]
    kept = [c for c in choices if c.target_pick is not None and c.consistent]
    # share of answers that were the letter A: position bias if far from 0.5
    a_letters = [
        (v == ORDERS[name][0]) for c in parsed for name, v in c.picks_by_order.items() if v is not None
    ]
    return {
        "scenarios": n,
        "any_parsed": len(parsed) / n if n else float("nan"),
        "kept": len(kept),
        "kept_rate": len(kept) / n if n else float("nan"),
        "share_action1_among_kept": (sum(1 for c in kept if c.target_pick == 0) / len(kept)) if kept else float("nan"),
        "share_letter_A": (sum(a_letters) / len(a_letters)) if a_letters else float("nan"),
    }
