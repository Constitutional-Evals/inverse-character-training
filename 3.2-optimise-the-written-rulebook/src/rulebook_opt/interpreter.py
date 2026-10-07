"""R: reads a rulebook and predicts which option a model following it would pick."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .data import Choice, Scenario
from .llm import LLMClient
from .prompts import LETTERS, interpreter_messages, parse_choice


@dataclass
class Decision:
    picks: list[int | None]  # per presentation order, as an index into scenario.actions (None = unparsed)
    texts: list[str]  # R's raw replies (reasoning + answer)
    orders: list[tuple[int, int]]  # which action was shown as A and as B


def agreement(decision: Decision, target_pick: int) -> float:
    """Fraction of presentation orders in which R picked the target's action (unparsed counts as 0)."""
    return sum(1 for p in decision.picks if p == target_pick) / len(decision.picks)


class Interpreter:
    def __init__(self, client: LLMClient, orders: int = 1):
        if orders not in (1, 2):
            raise ValueError("orders must be 1 or 2")
        self.client = client
        self.orders = orders

    @staticmethod
    def _base_flip(scenario_id: str) -> int:
        return hashlib.sha1(f"R|{scenario_id}".encode()).digest()[0] & 1

    def order_for(self, scenario_id: str, variant: int) -> tuple[int, int]:
        flip = self._base_flip(scenario_id) ^ (variant & 1)
        return (1, 0) if flip else (0, 1)

    def decide(self, rulebook: str, scenarios: list[Scenario] | list[Choice]) -> list[Decision]:
        requests, meta = [], []
        for sc in scenarios:
            for v in range(self.orders):
                order = self.order_for(sc.id, v)
                a, b = sc.actions[order[0]], sc.actions[order[1]]
                requests.append(interpreter_messages(rulebook, sc.dilemma, a, b))
                meta.append(order)
        replies = self.client.complete_many(requests)
        out = []
        for i in range(len(scenarios)):
            picks, texts, orders = [], [], []
            for v in range(self.orders):
                k = i * self.orders + v
                letter = parse_choice(replies[k], strict=True)
                picks.append(None if letter is None else meta[k][letter])
                texts.append(replies[k])
                orders.append(meta[k])
            out.append(Decision(picks, texts, orders))
        return out

    def describe_order(self, order: tuple[int, int]) -> str:
        return f"A=action {order[0] + 1}, B=action {order[1] + 1}" if order else ""


__all__ = ["Interpreter", "Decision", "agreement", "LETTERS"]
