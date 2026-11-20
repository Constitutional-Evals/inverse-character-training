import json
from itertools import combinations
from pathlib import Path
import sys
import unittest

PART = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PART), str(PART / "experiments"), str(PART / "pipeline")]

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
