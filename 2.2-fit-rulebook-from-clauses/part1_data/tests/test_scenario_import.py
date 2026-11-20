import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from filelock import FileLock

PART = Path(__file__).resolve().parents[1]


class ScenarioImportTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "runs" / "pilot"
        self.export = self.run / "generated_scenarios_rich.jsonl"
        self.manifest = self.run / "benchmark_import.json"
        self.criteria = self.root / "criteria.jsonl"
        self.criteria.write_text("".join(
            json.dumps({"pilot_id": i, "criterion_id": f"{group}#{i}", "criterion": f"Criterion {i}."}) + "\n"
            for i, group in enumerate(["kindness"] * 5 + ["oct-goodness"], 1)
        ), encoding="utf-8")
        self.source = self.benchmark("first")

    def benchmark(self, name):
        source = self.root / f"{name}.json"
        source.write_text(json.dumps([{
            "scenario_id": name,
            "scenario": f"What is the best course of action for {name}?",
            "relevance": {"kindness": [1, 2, 3, 4, 5], "oct-goodness": [6]},
        }]), encoding="utf-8")
        return source

    def command(self, *args):
        return subprocess.run([
            sys.executable, str(PART / "pipeline/run_scenarios.py"),
            "--output-root", str(self.run.parent), "--run-name", self.run.name,
            "--criteria", str(self.criteria), *map(str, args),
        ], capture_output=True, text=True)

    def test_same_import_can_resume_and_different_source_or_target_cannot(self):
        imported = self.command("--benchmark", self.source, "--target", "OCT Goodness")
        self.assertEqual(imported.returncode, 0, imported.stderr)
        original = self.export.read_bytes()
        identity = self.manifest.read_bytes()
        self.assertEqual(json.loads(original)["verified_criteria"], [6, 1, 2, 3, 4, 5])

        compressed = self.root / "copy.zip"
        with ZipFile(compressed, "w") as archive:
            archive.write(self.source, "benchmark.json")
        resumed = self.command("--benchmark", compressed, "--target", "oct-goodness")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)

        different = self.command("--benchmark", self.benchmark("different"), "--target", "oct-goodness")
        retargeted = self.command("--benchmark", self.source, "--target", "kindness")
        self.assertIn("differs from this run's saved import", different.stderr)
        self.assertIn("differs from this run's saved import", retargeted.stderr)
        self.assertEqual(self.export.read_bytes(), original)
        self.assertEqual(self.manifest.read_bytes(), identity)

    def test_legacy_import_only_adopts_matching_rows(self):
        result = self.command("--benchmark", self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest.unlink()
        original = self.export.read_bytes()
        result = self.command("--benchmark", self.benchmark("different"))
        self.assertIn("does not match this run's existing import", result.stderr)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(self.export.read_bytes(), original)
        retry = self.command("--benchmark", self.source)
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertTrue(self.manifest.is_file())

    def test_legacy_import_requires_saved_pool(self):
        result = self.command("--benchmark", self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest.unlink()
        (self.run / "pilot_constitutions.json").unlink()
        result = self.command("--benchmark", self.source)
        self.assertIn("does not match this run's existing import", result.stderr)
        self.assertFalse(self.manifest.exists())

    def test_generation_and_import_cannot_share_a_run_name(self):
        self.run.mkdir(parents=True)
        generation_manifest = self.run / "run_manifest.json"
        generation_manifest.write_text('{"identity": "generation"}', encoding="utf-8")
        result = self.command("--benchmark", self.source)
        self.assertIn("already used for generation", result.stderr)
        self.assertFalse(self.export.exists())
        generation_manifest.unlink()

        result = self.command("--benchmark", self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = self.export.read_bytes()
        result = self.command("--budget", 1)
        self.assertIn("already used for benchmark import", result.stderr)
        self.assertEqual(self.export.read_bytes(), original)

    def test_second_writer_cannot_enter_run(self):
        self.run.mkdir(parents=True)
        with FileLock(str(self.run / ".scenario-run.lock"), timeout=0):
            result = self.command("--benchmark", self.source)
        self.assertIn("already in progress", result.stderr)
        self.assertFalse(self.manifest.exists())
