import json
from itertools import combinations
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

PART = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PART), str(PART / "experiments"), str(PART / "pipeline")]

import anchored_extremes as anchored
from experiments.anchored_menu_checks import anand, parse_pairs, parse_quality


class AnchoredCheckTest(unittest.TestCase):
    def setUp(self):
        self.to_id = {label: f"stable:{label.lower()}" for label in "ABCDE"}
        self.quality = {"scores": dict.fromkeys(anand.QUALITY_DIMS, 75),
                        "reason": "Distinct motivations.", "repair": "", "flaws": []}
        self.pairs = [{"pair": [a, b], "score": 75 if (a, b) == ("A", "E") else 15,
                       "reason": "Same decision and tradeoff." if (a, b) == ("A", "E") else "Different priorities."}
                      for a, b in combinations("ABCDE", 2)]

    def test_quality_gates_and_local_evidence(self):
        self.quality.update(flaws=[{"reply": "C", "dimension": "information_realism",
                                    "evidence": "Assumes the employee can guarantee a trial."}])
        for minimum, passed, repairable in [(100, True, True), (60, False, True), (59, False, True), (49, False, False)]:
            with self.subTest(minimum=minimum):
                self.quality["scores"]["information_realism"] = minimum
                result = parse_quality(json.dumps(self.quality), self.to_id)
                self.assertEqual((result["passed"], result["repairable"]), (passed, repairable))
                self.assertEqual(result["flaws"][0]["reply_id"], "stable:c")

    def test_rejects_unusable_evidence_scores_and_incomplete_pairs(self):
        for mutation in [dict(reply="F", dimension="coherence", evidence="x"),
                         dict(reply="A", dimension="unknown", evidence="x"),
                         dict(reply="A", dimension="coherence", evidence="")]:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                parse_quality(json.dumps({**self.quality, "flaws": [mutation]}), self.to_id)
        for bad_rows in [self.pairs[:-1], self.pairs + [self.pairs[0]],
                         [{**self.pairs[0], "pair": ["A", "A"]}, *self.pairs[1:]],
                         [{**self.pairs[0], "score": True}, *self.pairs[1:]]]:
            with self.subTest(rows=len(bad_rows)), self.assertRaises(ValueError):
                parse_pairs(json.dumps({"pairs": bad_rows}), self.to_id)

    def test_order_reversal_preserves_pair_and_blinds_writer_metadata(self):
        llm = Mock()
        prompts = []

        def chat(model, messages, *, validate, tag, **kwargs):
            self.assertEqual(model, "reviewer")
            prompts.append(messages[1]["content"])
            # A/E in the reverse order are the same E/A pair.
            data = self.quality if "-quality:" in tag else {"pairs": self.pairs}
            return validate(json.dumps(data))

        llm.chat.side_effect = chat
        replies = [{"response_id": rid, "text": f"Advice {i}", "anchor": "hidden criterion"}
                   for i, rid in enumerate(self.to_id.values())]
        checks = anchored.run_anchored_checks(llm, SimpleNamespace(checker_model="reviewer", check_orders=2),
                                              "A request", replies, "case")
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all("hidden criterion" not in p and "stable:" not in p for p in prompts))
        first, second = checks
        self.assertEqual(first["order"], list(reversed(second["order"])))
        self.assertEqual(first["duplicates"], second["duplicates"])
        self.assertEqual(anchored.confirmed_menu_issues(checks),
                         {"weak": [], "quality_failed": False, "duplicates": first["duplicates"]})
