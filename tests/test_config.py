import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import _util  # noqa: F401  (sys.path)

from die_common import paths
from die_common.config import (
    HyperParams,
    load_hyperparams,
    overrides_from_args,
    parse_train_args,
    split_lengths,
)


class LoadHyperparamsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.default = self.dir / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data, name="config.json"):
        path = self.dir / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_missing_default_config_uses_builtin_values(self):
        hp, path, loaded = load_hyperparams(None, {}, default_config_path=self.default)
        self.assertFalse(loaded)
        self.assertEqual(path, self.default)
        self.assertEqual(hp, HyperParams())

    def test_missing_explicit_config_is_error(self):
        with self.assertRaises(FileNotFoundError):
            load_hyperparams(self.dir / "jiny.json", {}, default_config_path=self.default)

    def test_values_are_cast_to_field_types(self):
        # marshall ani ruční úprava JSONu nemusí trefit typ (20.0, "32")
        self.write({"epochs": 20.0, "batch_size": "32", "learning_rate": "0.01"})
        hp, _, loaded = load_hyperparams(None, {}, default_config_path=self.default)
        self.assertTrue(loaded)
        self.assertIsInstance(hp.epochs, int)
        self.assertEqual((hp.epochs, hp.batch_size, hp.learning_rate), (20, 32, 0.01))

    def test_invalid_value_reports_field(self):
        self.write({"epochs": "hodně"})
        with self.assertRaisesRegex(ValueError, "epochs"):
            load_hyperparams(None, {}, default_config_path=self.default)

    def test_non_object_json_is_error(self):
        self.write([1, 2, 3])
        with self.assertRaises(ValueError):
            load_hyperparams(None, {}, default_config_path=self.default)

    def test_cli_overrides_config_and_none_keeps_it(self):
        self.write({"epochs": 5, "dropout": 0.3})
        hp, _, _ = load_hyperparams(None, {"epochs": 7, "dropout": None}, default_config_path=self.default)
        self.assertEqual(hp.epochs, 7)
        self.assertEqual(hp.dropout, 0.3)

    def test_validation_rejects_too_little_training_data(self):
        with self.assertRaises(ValueError):
            load_hyperparams(None, {"val_ratio": 0.4, "test_ratio": 0.4}, default_config_path=self.default)
        with self.assertRaises(ValueError):
            load_hyperparams(None, {"lr_gamma": 0.0}, default_config_path=self.default)

    def test_repository_config_is_valid(self):
        hp, _, loaded = load_hyperparams(None, {})
        self.assertTrue(loaded)
        hp.validate()


class TrainArgsTest(unittest.TestCase):
    def test_defaults_point_to_shared(self):
        args = parse_train_args([])
        self.assertEqual(args.data_dir, paths.COMPOSED_DIR)
        self.assertEqual(args.output_dir, paths.MODELS_DIR)
        self.assertEqual(args.config, paths.LEARNER_CONFIG)
        self.assertFalse(args.use_cpu)
        self.assertTrue(all(v is None for v in overrides_from_args(args).values()))

    def test_overrides_are_typed(self):
        args = parse_train_args(["--epochs", "3", "--learning_rate", "0.005", "--use_cpu",
                                 "--data_dir", "neco"])
        overrides = overrides_from_args(args)
        self.assertEqual(overrides["epochs"], 3)
        self.assertEqual(overrides["learning_rate"], 0.005)
        self.assertEqual(args.data_dir, Path("neco"))
        self.assertTrue(args.use_cpu)

    def test_action_word_is_rejected(self):
        # run.bat musí slovo "train" odfiltrovat – argparse by ho odmítl
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_train_args(["train", "--epochs", "3"])


class SplitLengthsTest(unittest.TestCase):
    def test_lengths_sum_and_remainder_goes_to_train(self):
        self.assertEqual(split_lengths(101, 0.2, 0.1), (71, 20, 10))
        for total in (3, 10, 57, 1000):
            self.assertEqual(sum(split_lengths(total, 0.2, 0.1)), total)

    def test_zero_ratios(self):
        self.assertEqual(split_lengths(10, 0.0, 0.0), (10, 0, 0))

    def test_too_small_or_nothing_left(self):
        with self.assertRaises(ValueError):
            split_lengths(2, 0.2, 0.1)
        with self.assertRaises(ValueError):
            split_lengths(10, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
