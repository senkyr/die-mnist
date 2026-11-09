import json
import tempfile
import unittest
from pathlib import Path

from _util import ROOT

from die_common import paths


class PathsTest(unittest.TestCase):
    def test_paths_are_anchored_to_repository(self):
        self.assertEqual(paths.ROOT, ROOT)
        self.assertTrue((paths.ROOT / "die_common").is_dir())
        for p in (paths.COLLECTED_DIR, paths.SYNTHETIC_DIR, paths.COMPOSED_DIR,
                  paths.MODELS_DIR, paths.TESTS_DIR):
            self.assertTrue(p.is_absolute())
            self.assertEqual(p.relative_to(ROOT).parts[0], "shared")
        self.assertEqual(paths.LEARNER_CONFIG, ROOT / "DigitLearner" / "config.json")

    def test_find_model_runs_skips_unfinished_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            for name in ("run_20251024_120000", "run_20251024_120000_01", "run_20251023_090000"):
                (models / name).mkdir()
                (models / name / paths.MODEL_FILENAME).write_bytes(b"x")
            (models / "run_20251025_000000").mkdir()  # přerušený trénink, bez modelu
            (models / "other").mkdir()

            runs = [p.name for p in paths.find_model_runs(models)]
            self.assertEqual(runs, ["run_20251023_090000", "run_20251024_120000", "run_20251024_120000_01"])
            self.assertEqual(paths.latest_model_run(models).name, "run_20251024_120000_01")

    def test_missing_models_dir(self):
        missing = Path(tempfile.gettempdir()) / "die-mnist-neexistuje-xyz"
        self.assertEqual(paths.find_model_runs(missing), [])
        self.assertIsNone(paths.latest_model_run(missing))


class RunSampleCountTest(unittest.TestCase):
    """DigitTeaser ukazuje počet vzorků běhu; zdrojem je jen test_metrics.json z train.py."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        (self.run / name).write_text(json.dumps(data), encoding="utf-8")

    def test_reads_num_samples_from_test_metrics(self):
        self.write("test_metrics.json", {"test_accuracy": 0.9, "num_samples": 120})
        self.assertEqual(paths.run_sample_count(self.run), 120)

    def test_history_list_does_not_carry_count(self):
        # train.py zapisuje training_history.json jako seznam epoch, ne dict.
        self.write("training_history.json", [{"epoch": 1, "train_loss": 0.5, "train_acc": 0.8,
                                              "val_loss": 0.6, "val_acc": 0.7}])
        self.assertIsNone(paths.run_sample_count(self.run))

    def test_missing_broken_or_unexpected_metrics(self):
        self.assertIsNone(paths.run_sample_count(self.run))
        (self.run / "test_metrics.json").write_text("{nedopsany", encoding="utf-8")
        self.assertIsNone(paths.run_sample_count(self.run))
        for data in ([1, 2], {"num_samples": "120"}, {"num_samples": True}):
            self.write("test_metrics.json", data)
            self.assertIsNone(paths.run_sample_count(self.run))


if __name__ == "__main__":
    unittest.main()
