import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PART = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART / "experiments"))

import anchored_extremes as anchored


IDS = ["A", "B", "C", "D", "E"]


def check(*, weak=(), duplicates=(), menu_failure=False, repairable=True, dimension="coherence"):
    return {
        "order": IDS,
        "quality": {
            "passed": not (weak or menu_failure), "repairable": repairable,
            "reason": "Review of the whole menu.", "repair": "Fix cited flaws.",
            "flaws": [{"reply_id": rid, "dimension": dimension, "evidence": "Assumed an unavailable option."}
                      for rid in weak],
        },
        "duplicates": list(duplicates),
        "pairs": [{"pair": list(pair), "score": 85, "reason": "Same advice and same tradeoff."}
                  for pair in duplicates],
    }


class MenuRepairTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(model="writer", words=220)
        self.replies = [
            {"response_id": rid, "anchor": i, "recommendation": f"Original {rid}",
             "text": f"Original reply {rid}.", "edits": []}
            for i, rid in enumerate(IDS, 1)
        ]
        self.criteria = {i: f"Criterion {i}" for i in range(1, 6)}
        self.llm = Mock()

    def review(self, initial, revised=None):
        rounds = [initial] if revised is None else [initial, revised]
        with patch.object(anchored, "run_anchored_checks", side_effect=rounds) as reviewer:
            outcome = anchored.review_menu(
                self.llm, self.args, "Help me decide.", self.replies, [2], self.criteria, "case"
            )
        return outcome, reviewer.call_args_list

    def test_repair_targets_only_confirmed_defects_and_rechecks_whole_menu_once(self):
        initial = [check(weak=["C", "D"], duplicates=[["A", "B"]]),
                   check(weak=["C"], duplicates=[["B", "A"]])]
        for revised, expected_keep in [([check(), check()], True),
                                       ([check(weak=["C"]), check(weak=["C"])], False)]:
            with self.subTest(keep=expected_keep):
                self.llm.reset_mock()

                def repair(model, messages, *, validate, **kwargs):
                    user_prompt = messages[1]["content"]
                    targets = json.loads(user_prompt.split("<repairs_requested>\n", 1)[1].split("\n</repairs_requested>", 1)[0])
                    self.assertEqual([(t["id"], t["priority"]) for t in targets],
                                     [("A", "Criterion 1"), ("C", "Criterion 3")])
                    self.assertEqual(targets[0]["duplicates"], ["B"])
                    notes = json.loads(user_prompt.split("<reviewer_notes>\n", 1)[1].split("\n</reviewer_notes>", 1)[0])
                    self.assertEqual(notes[0]["pair_evidence"][0]["reason"], "Same advice and same tradeoff.")
                    self.assertEqual([f["reply_id"] for f in notes[0]["local_evidence"]], ["C"])
                    return validate(json.dumps({"repairs": [
                        {"id": rid, "recommendation": f"Changed {rid}", "text": f"Repaired reply {rid}."}
                        for rid in ["A", "C"]
                    ]}))

                self.llm.chat.side_effect = repair
                outcome, calls = self.review(initial, revised)
                self.llm.chat.assert_called_once()
                self.assertEqual([c.args[5] for c in calls], ["check", "recheck"])
                self.assertEqual(calls[0].args[3], self.replies)
                self.assertEqual([r["response_id"] for r in calls[1].args[3]], IDS)
                self.assertEqual(outcome["responses"][1], self.replies[1])
                self.assertEqual(outcome["responses"][2]["text"], "Repaired reply C.")
                self.assertEqual(self.replies[2]["text"], "Original reply C.")
                self.assertEqual(outcome["menu_repair"]["before_replies"], [self.replies[0], self.replies[2]])
                self.assertEqual(outcome["checks"], revised)
                self.assertEqual(outcome["keep"], expected_keep)

    def test_order_disagreements_remain_inspectable(self):
        raw = [check(weak=["A"], duplicates=[["A", "B"]]), check(duplicates=[["C", "D"]])]
        outcome, calls = self.review(raw)
        self.llm.chat.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertTrue(outcome["keep"])
        self.assertEqual(outcome["checks"], raw)
        self.assertCountEqual(outcome["flags"], ["menu_quality_flagged", "duplicates_flagged"])

    def test_confirmed_menu_failure_without_confirmed_local_defect_is_held_without_repair(self):
        for raw in [
            [check(menu_failure=True), check(menu_failure=True)],
            [check(weak=["A"]), check(weak=["A"], dimension="naturalness")],
            [check(weak=["A"], repairable=False, duplicates=[["C", "D"]]),
             check(weak=["A"], duplicates=[["C", "D"]])],
        ]:
            with self.subTest(raw=raw):
                outcome, calls = self.review(raw)
                self.assertEqual(len(calls), 1)
                self.assertFalse(outcome["keep"])
                self.llm.chat.assert_not_called()

    def test_clean_menu_is_retained_without_repair(self):
        outcome, calls = self.review([check(), check()])
        self.assertTrue(outcome["keep"])
        self.assertEqual(len(calls), 1)
        self.llm.chat.assert_not_called()

    def test_repair_validation_rejects_missing_extra_duplicate_and_overlong_replies(self):
        reply = {"id": "A", "recommendation": "Choose this.", "text": "One two."}
        for replacements in [[], [reply, reply], [{**reply, "id": "B"}], [{**reply, "text": "One two three."}]]:
            with self.subTest(replacements=replacements), self.assertRaises(ValueError):
                anchored.validate_menu_repair(json.dumps({"repairs": replacements}), {"A"}, 2)
