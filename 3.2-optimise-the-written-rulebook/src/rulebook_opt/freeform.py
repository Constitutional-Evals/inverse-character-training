"""Free-form mode: compare replies instead of A/B choices.

T answers each dilemma in its own words. For a candidate rulebook, R answers the same dilemma under it.
A judge sees T's reply as the REFERENCE and a line-up of R's replies: one under the candidate rulebook and
the rest under fixed decoy constitutions (plus R's default character). The judge picks the reply whose author
is most like the reference's. Score = 1 if it picks the candidate's reply, else 0; chance = 1 / line-up size.
Decoy replies depend only on the scenario, so they are generated once and cached.
"""

from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from .data import Scenario, read_jsonl, write_jsonl
from .llm import LLMClient
from .metrics import bootstrap_ci
from .prompts import interpreter_free_messages, judge_messages, parse_number, strip_answer, target_free_messages
from .rulebooks import violation

COMPONENT = "rulebook"
CANDIDATE = "candidate"


@dataclass
class Reply:
    id: str
    dilemma: str
    split: str
    target_reply: str | None


def collect_replies(
    scenarios: list[Scenario], client: LLMClient, constitution: str | None = None, chunk: int = 50, verbose: bool = True
) -> list[Reply]:
    out: list[Reply] = []
    for start in range(0, len(scenarios), chunk):
        part = scenarios[start : start + chunk]
        texts = client.complete_many([target_free_messages(s.dilemma, constitution) for s in part])
        out += [Reply(s.id, s.dilemma, s.split, t.strip() or None) for s, t in zip(part, texts)]
        if verbose:
            print(f"collected {min(start + chunk, len(scenarios))}/{len(scenarios)}", file=sys.stderr)
    return out


def save_replies(path: Path, replies: list[Reply]) -> None:
    write_jsonl(path, (asdict(r) for r in replies))


def load_replies(path: Path, limits: dict[str, int] | None = None) -> dict[str, list[Reply]]:
    """Replies by split (empty target replies dropped), truncated to ``limits[split]`` if given."""
    out: dict[str, list[Reply]] = {}
    for r in read_jsonl(path):
        rep = Reply(**r)
        if rep.target_reply:
            out.setdefault(rep.split, []).append(rep)
    if limits:
        out = {k: v[: limits.get(k, len(v))] for k, v in out.items()}
    return out


@dataclass
class Judgement:
    correct: bool
    candidate_reply: str
    picked: str | None  # line-up entry name the judge picked (CANDIDATE, a decoy name, or None if unparsed)
    picked_reply: str | None
    reason: str
    lineup_size: int


class Lineup:
    """Scores a rulebook by whether a judge can pick R's reply under it out of a line-up of decoys."""

    def __init__(self, r_client: LLMClient, judge_client: LLMClient, decoys: dict[str, str | None], seed: int = 0):
        if CANDIDATE in decoys:
            raise ValueError(f"'{CANDIDATE}' is reserved")
        self.r = r_client
        self.judge = judge_client
        self.decoys = decoys  # name -> rulebook text, or None for R's default character (no system prompt)
        self.seed = seed
        self._decoy_cache: dict[tuple[str, str], str] = {}

    @property
    def size(self) -> int:
        return len(self.decoys) + 1

    def replies(self, rulebook: str | None, items: list[Reply]) -> list[str]:
        return [t.strip() for t in self.r.complete_many([interpreter_free_messages(rulebook, it.dilemma) for it in items])]

    def _decoy_replies(self, items: list[Reply]) -> dict[str, list[str]]:
        for name, rb in self.decoys.items():
            missing = [it for it in items if (name, it.id) not in self._decoy_cache]
            for it, text in zip(missing, self.replies(rb, missing)):
                self._decoy_cache[(name, it.id)] = text
        return {name: [self._decoy_cache[(name, it.id)] for it in items] for name in self.decoys}

    def _order(self, item_id: str, rulebook: str) -> list[str]:
        names = [CANDIDATE, *self.decoys]
        h = hashlib.sha1(f"{self.seed}|{item_id}|{rulebook}".encode()).hexdigest()
        random.Random(h).shuffle(names)
        return names

    def score(self, rulebook: str, items: list[Reply]) -> list[Judgement]:
        cand = self.replies(rulebook, items)
        decoys = self._decoy_replies(items)
        requests, orders = [], []
        for i, it in enumerate(items):
            order = self._order(it.id, rulebook)
            texts = [cand[i] if n == CANDIDATE else decoys[n][i] for n in order]
            requests.append(judge_messages(it.dilemma, it.target_reply or "", texts))
            orders.append((order, texts))
        verdicts = self.judge.complete_many(requests)
        out = []
        for i, v in enumerate(verdicts):
            order, texts = orders[i]
            k = parse_number(v, len(order))
            picked = None if k is None else order[k]
            out.append(
                Judgement(
                    correct=picked == CANDIDATE,
                    candidate_reply=cand[i],
                    picked=picked,
                    picked_reply=None if k is None else texts[k],
                    reason=strip_answer(v),
                    lineup_size=len(order),
                )
            )
        return out


def evaluate_freeform(lineup: Lineup, rulebook: str, items: list[Reply]) -> dict:
    js = lineup.score(rulebook, items)
    scores = [1.0 if j.correct else 0.0 for j in js]
    lo, hi = bootstrap_ci(scores)
    return {
        "n": len(items),
        "agreement": sum(scores) / len(scores) if scores else float("nan"),  # identification rate
        "ci_low": lo,
        "ci_high": hi,
        "parse_rate": sum(j.picked is not None for j in js) / len(js) if js else float("nan"),
        "chance": 1 / lineup.size,
        "scores": scores,
    }


def _clip(s: str | None, n: int = 900) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " [...]"


@dataclass
class _Traj:
    item: Reply
    judgement: Judgement | None
    rejected: str | None = None


class FreeformAdapter:
    """GEPA adapter for free-form mode; the GEPA budget counts scored dilemmas (one R reply + one judge call each)."""

    propose_new_texts = None  # use GEPA's own LLM reflection

    def __init__(self, lineup: Lineup, max_criteria: int | None = 15, max_chars: int | None = 3500):
        self.lineup = lineup
        self.max_criteria = max_criteria
        self.max_chars = max_chars

    def evaluate(self, batch: list[Reply], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        rulebook = candidate[COMPONENT]
        reason = violation(rulebook, self.max_criteria, self.max_chars)
        if reason:
            traj = [_Traj(it, None, reason) for it in batch] if capture_traces else None
            return EvaluationBatch(outputs=[None] * len(batch), scores=[0.0] * len(batch), trajectories=traj, num_metric_calls=0)
        js = self.lineup.score(rulebook, batch)
        traj = [_Traj(it, j) for it, j in zip(batch, js)] if capture_traces else None
        return EvaluationBatch(
            outputs=[j.candidate_reply for j in js],
            scores=[1.0 if j.correct else 0.0 for j in js],
            trajectories=traj,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        records = []
        for t in eval_batch.trajectories or []:
            if t.rejected:
                records.append(
                    {"Inputs": "(not run)", "Generated Outputs": "(none)", "Feedback": f"Rejected without evaluation: {t.rejected}."}
                )
                continue
            j = t.judgement
            assert j is not None
            inputs = f"Dilemma:\n{t.item.dilemma}\n\nThe target's reply:\n{_clip(t.item.target_reply)}"
            if j.correct:
                fb = f"Correct: the judge matched the target's reply to this reply. Judge: {j.reason}"
            elif j.picked is None:
                fb = "Incorrect: the judge's answer could not be parsed."
            else:
                fb = (
                    "Incorrect: the judge found another assistant's reply closer to the target's character:\n"
                    f'"{_clip(j.picked_reply, 600)}"\nJudge: {j.reason}\n'
                    "Change the principles so that an assistant following them replies with the target's character, "
                    "stated in general terms and without mentioning this scenario."
                )
            records.append({"Inputs": inputs, "Generated Outputs": _clip(j.candidate_reply), "Feedback": fb})
        return {c: records for c in components_to_update}
