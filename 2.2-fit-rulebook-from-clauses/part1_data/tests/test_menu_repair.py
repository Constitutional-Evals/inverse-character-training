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


def check(*, weak=(), duplicates=(), winner=None, value=()):
    return {
        "order": IDS,
        "defensible": {rid: 2 if rid in weak else 4 for rid in IDS},
        "duplicates": list(duplicates),
        "obvious_best": winner,
        "value_language": list(value),
        "note": "Some options recommend the same action." if duplicates else "",
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
        with patch.object(anchored, "run_checks", side_effect=rounds) as reviewer:
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

    def test_order_disagreement_and_value_flags_remain_inspectable(self):
        raw = [
            check(weak=["A"], duplicates=[["A", "B"]], winner="B", value=["D"]),
            check(weak=["C"], duplicates=[["C", "D"]], winner="E"),
        ]
        outcome, calls = self.review(raw)
        self.llm.chat.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertTrue(outcome["keep"])
        self.assertEqual(outcome["checks"], raw)
        self.assertCountEqual(outcome["flags"],
                              ["obvious_best", "low_defensibility", "duplicates_flagged", "value_language"])

    def test_clean_menu_and_consensus_winner_need_no_rewrite(self):
        for winner, expected_keep in [(None, True), ("A", False)]:
            with self.subTest(winner=winner):
                outcome, calls = self.review([check(winner=winner), check(winner=winner)])
                self.assertEqual(len(calls), 1)
                self.assertEqual(outcome["keep"], expected_keep)
        self.llm.chat.assert_not_called()

    def test_repair_validation_rejects_missing_extra_duplicate_and_overlong_replies(self):
        reply = {"id": "A", "recommendation": "Choose this.", "text": "One two."}
        for replacements in [[], [reply, reply], [{**reply, "id": "B"}], [{**reply, "text": "One two three."}]]:
            with self.subTest(replacements=replacements), self.assertRaises(ValueError):
                anchored.validate_menu_repair(json.dumps({"repairs": replacements}), {"A"}, 2)
