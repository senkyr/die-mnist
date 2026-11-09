"""Testy závislé na PyTorch – bez torch se přeskočí (spouštěj ve .venv DigitLearneru).

    DigitLearner\\.venv\\Scripts\\python -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _util import HAS_PIL, HAS_TORCH, load_tool, make_class_tree

# Klíče state dictu uložených digit_cnn.pt. Změna = staré modely nepůjdou načíst.
EXPECTED_KEYS = [
    f"features.{i}.{name}"
    for i, kind in ((0, "conv"), (1, "bn"), (4, "conv"), (5, "bn"), (8, "conv"), (9, "bn"))
    for name in (("weight", "bias") if kind == "conv"
                 else ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"))
] + ["classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"]


@unittest.skipUnless(HAS_TORCH, "PyTorch není nainstalovaný")
class SimpleCNNTest(unittest.TestCase):
    def test_state_dict_keys_are_stable(self):
        from die_common.model import SimpleCNN

        self.assertEqual(list(SimpleCNN().state_dict().keys()), EXPECTED_KEYS)

    def test_forward_shape(self):
        import torch

        from die_common.model import SimpleCNN

        model = SimpleCNN().eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 1, 32, 32)).shape), (2, 10))

    def test_load_model_roundtrip_in_eval_mode(self):
        import torch

        from die_common.model import SimpleCNN, load_model

        original = SimpleCNN(dropout=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "digit_cnn.pt"
            torch.save(original.state_dict(), path)
            loaded = load_model(path, dropout=0.3)
        self.assertFalse(loaded.training)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)


@unittest.skipUnless(HAS_TORCH and HAS_PIL, "PyTorch/Pillow nejsou nainstalované")
class TrainHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_tool("digit_learner_train", "DigitLearner/train.py")

    def test_snapshot_is_independent_of_further_training(self):
        import torch

        model = self.train.SimpleCNN()
        snapshot = self.train.snapshot_state(model)
        with torch.no_grad():
            model.classifier[4].weight.add_(1.0)
        self.assertFalse(torch.equal(snapshot["classifier.4.weight"], model.classifier[4].weight))

    def test_random_split_keeps_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_class_tree(Path(tmp), {i: 3 for i in range(10)})
            dataset = self.train.DigitDataset(Path(tmp))
            loaders = self.train.create_loaders(dataset, batch_size=4, val_ratio=0.2, test_ratio=0.1, seed=1)
            self.assertEqual(set(loaders), {"train", "val", "test"})
            self.assertEqual(sum(len(l.dataset) for l in loaders.values()), 30)
            images, labels = next(iter(loaders["train"]))
            self.assertEqual(tuple(images.shape[1:]), (1, 32, 32))

    def test_empty_split_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            make_class_tree(base / "train", {0: 2, 1: 2})
            (base / "val" / "0").mkdir(parents=True)
            loaders = self.train.create_loaders_from_subdirectories(base, batch_size=2)
            self.assertEqual(list(loaders), ["train"])

    def test_bad_data_dir_leaves_no_run_folder(self):
        # Dřív vznikla složka run_* s config.json dřív, než se zkontrolovala data.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "empty").mkdir()
            for data_dir, error in ((base / "neni", FileNotFoundError), (base / "empty", RuntimeError)):
                argv = ["train.py", "--data_dir", str(data_dir), "--output_dir", str(base / "models"),
                        "--use_cpu"]
                with self.subTest(data_dir=data_dir.name), mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(error):
                        self.train.main()
                    self.assertEqual(list((base / "models").glob("run_*")), [])


if __name__ == "__main__":
    unittest.main()
