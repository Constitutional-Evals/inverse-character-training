"""GEPA adapter: a candidate is one text component, the rulebook; its score is agreement with T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gepa.core.adapter import EvaluationBatch

from .data import Choice
from .interpreter import Decision, Interpreter, agreement
from .prompts import LETTERS
from .rulebooks import violation

COMPONENT = "rulebook"


@dataclass
class Trajectory:
    choice: Choice
    decision: Decision | None
    rejected: str | None = None


class RulebookAdapter:
    """Implements gepa's GEPAAdapter protocol.

    Score per scenario = fraction of R's presentation orders in which R, prompted with the candidate
    rulebook, picks the same action as the target T. Candidates that break the length limits are
    rejected without calling R (score 0, no budget used).

    The GEPA call budget (``max_metric_calls``) is counted in R calls: one per scenario per order.
    """

    # GEPA looks this up on the adapter; None means "use GEPA's own LLM reflection"
    propose_new_texts = None

    def __init__(self, interpreter: Interpreter, max_criteria: int | None = 15, max_chars: int | None = 3500):
        self.interpreter = interpreter
        self.max_criteria = max_criteria
        self.max_chars = max_chars

    def evaluate(self, batch: list[Choice], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        rulebook = candidate[COMPONENT]
        reason = violation(rulebook, self.max_criteria, self.max_chars)
        if reason:
            traj = [Trajectory(c, None, reason) for c in batch] if capture_traces else None
            return EvaluationBatch(
                outputs=[None] * len(batch), scores=[0.0] * len(batch), trajectories=traj, num_metric_calls=0
            )
        decisions = self.interpreter.decide(rulebook, batch)
        scores = [agreement(d, c.target_pick) for d, c in zip(decisions, batch)]
        traj = [Trajectory(c, d) for c, d in zip(batch, decisions)] if capture_traces else None
        return EvaluationBatch(
            outputs=[d.texts[0] for d in decisions],
            scores=scores,
            trajectories=traj,
            num_metric_calls=len(batch) * self.interpreter.orders,
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        for traj, score in zip(eval_batch.trajectories or [], eval_batch.scores):
            if traj.rejected:
                records.append(
                    {
                        "Inputs": "(the rulebook was not run)",
                        "Generated Outputs": "(none)",
                        "Feedback": f"Rejected without evaluation: {traj.rejected}. Rewrite it within the limits.",
                    }
                )
                continue
            records.append(self._record(traj, score))
        return {c: records for c in components_to_update}

    def _record(self, traj: Trajectory, score: float) -> dict[str, Any]:
        c, d = traj.choice, traj.decision
        assert d is not None
        # show the first presentation order where R disagreed with T, else the first one
        k = next((i for i, p in enumerate(d.picks) if p != c.target_pick), 0)
        order = d.orders[k]
        shown = [c.actions[order[0]], c.actions[order[1]]]
        inputs = f"{c.dilemma}\nOption A: {shown[0]}\nOption B: {shown[1]}"
        target_text = c.actions[c.target_pick]
        target_letter = LETTERS[order.index(c.target_pick)]
        if score >= 1.0:
            fb = f"Correct. The target chose {target_letter}: {target_text}"
        else:
            pick = d.picks[k]
            if pick is None:
                fb = (
                    f"Incorrect: the answer could not be parsed. The target chose {target_letter}: {target_text}. "
                    "Principles must let the assistant commit to one option."
                )
            else:
                r_letter = LETTERS[order.index(pick)]
                fb = (
                    f"Incorrect. The assistant chose {r_letter} but the target chose {target_letter}: {target_text}. "
                    "Change the principles so that an assistant following them would choose what the target chose, "
                    "stated as general values and without mentioning this scenario."
                )
        return {"Inputs": inputs, "Generated Outputs": d.texts[k], "Feedback": fb}
