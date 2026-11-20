from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import anchored_extremes


class AnchoredIdTests(unittest.TestCase):
    def test_cli_rejects_id_collisions_before_model_calls(self):
        for first, second, output_id in [("gen_1", "1", "DA_1"), ("bench", "bench", "DA_bench")]:
            with self.subTest(first=first, second=second):
                rows = [{"scenario_id": first}, {"scenario_id": second}]
                errors = StringIO()
                with (
                    patch.object(sys, "argv", ["anchored_extremes.py"]),
                    patch.object(anchored_extremes, "read_jsonl", side_effect=[[], rows]),
                    patch.object(anchored_extremes.Path, "mkdir") as make_output,
                    patch.object(anchored_extremes, "check_models") as check_models,
                    patch.object(anchored_extremes, "LLM") as model,
                    redirect_stderr(errors),
                    self.assertRaises(SystemExit) as exit_error,
                ):
                    anchored_extremes.main()

                self.assertEqual(exit_error.exception.code, 2)
                self.assertIn(f"{first!r} and {second!r} both map to {output_id!r}", errors.getvalue())
                make_output.assert_not_called()
                check_models.assert_not_called()
                model.assert_not_called()

    def test_existing_ids_keep_their_format(self):
        for source_id, expected in [
            ("gen_00014", "DA_00014"),
            ("cand_slot_0000042", "DA_cand_slot_0000042"),
            ("1", "DA_1"),
        ]:
            with self.subTest(source_id=source_id):
                self.assertEqual(anchored_extremes.anchored_scenario_id(source_id), expected)
