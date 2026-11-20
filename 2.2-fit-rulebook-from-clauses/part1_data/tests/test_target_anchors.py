import json
from pathlib import Path
import sys
import unittest

PART = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART / "pipeline"))
sys.path.insert(0, str(PART / "experiments"))

from anchored_extremes import validate_plan
from run_scenarios import response_scenarios


class TargetAnchorsTest(unittest.TestCase):
    def test_verified_target_is_reserved_when_six_criteria_apply(self):
        bank = [{
            "scenario_id": "housing",
            "scenario": "What should our co-op fund?",
            "relevance": {"kindness": [1, 2, 3, 4, 5], "oct-goodness": [6]},
        }]
        [scenario] = response_scenarios(bank, target="oct-goodness")
        verified = scenario["verified_criteria"]
        self.assertEqual(verified, [6, 1, 2, 3, 4, 5])
        def plan(ids):
            return json.dumps({"anchors": [
                {"criterion": i, "label": f"Anchor {i}", "recommendation": "Fund an action.", "manner": "Plain."}
                for i in ids
            ]})

        self.assertEqual(validate_plan(plan([6, 1, 2, 3, 4]), verified, 6)[0]["criterion"], 6)
        with self.assertRaisesRegex(ValueError, "verified criteria"):
            validate_plan(plan([1, 2, 3, 4, 5]), verified, 6)
        self.assertEqual(response_scenarios(bank)[0]["verified_criteria"], [1, 2, 3, 4, 5, 6])
