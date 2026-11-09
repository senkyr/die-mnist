"""DigitDreamer: rozdělení do splitů, formát souborů a bezpečné --clean (malé počty, rychlé)."""

import random
import tempfile
import unittest
from pathlib import Path

from _util import HAS_NUMPY, HAS_PIL, load_tool, make_bmp


@unittest.skipUnless(HAS_NUMPY and HAS_PIL, "DigitDreamer potřebuje numpy a Pillow")
class DreamerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dreamer = load_tool("digit_dreamer_main", "DigitDreamer/main.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / "synthetic"
        random.seed(0)
        import numpy as np

        np.random.seed(0)

    def tearDown(self):
        self.tmp.cleanup()

    def generate(self, samples, split, **kw):
        params = dict(noise_sigma=10.0, val_split=0.2, test_split=0.1, blur_max=0.0,
                      invert_prob=0.0, max_retry=2, use_extra_shapes=False)
        params.update(kw)
        return self.dreamer.generate_dataset(base_dir=self.base, samples_per_class=samples,
                                             split_mode=split, **params)

    def count(self, directory):
        return len(list(Path(directory).glob("*.bmp")))

    def test_split_allocation_per_class(self):
        total, by_tag = self.generate(10, split=True)
        self.assertEqual(total, 100)
        self.assertEqual(dict(by_tag), {"train": 70, "val": 20, "test": 10})
        for label in range(10):
            self.assertEqual(self.count(self.base / "train" / str(label)), 7)
            self.assertEqual(self.count(self.base / "val" / str(label)), 2)
            self.assertEqual(self.count(self.base / "test" / str(label)), 1)

    def test_split_never_exceeds_samples(self):
        # Dřív round() u train i val: 3 vzorky, val 0,5, test 0 → 2+2 = 4 na třídu (40 místo 30).
        total, by_tag = self.generate(3, split=True, val_split=0.5, test_split=0.0)
        self.assertEqual(total, 30)
        self.assertEqual(sum(by_tag.values()), 30)

    def test_flat_output_is_numbered_32x32_grayscale(self):
        from PIL import Image

        self.generate(3, split=False)
        files = sorted(p.name for p in (self.base / "4").glob("*.bmp"))
        self.assertEqual(files, ["0001.bmp", "0002.bmp", "0003.bmp"])  # formát, který čte Composer
        with Image.open(self.base / "4" / "0001.bmp") as img:
            self.assertEqual((img.mode, img.size), ("L", (32, 32)))

    def test_append_continues_numbering(self):
        self.generate(2, split=False)
        self.generate(2, split=False)
        self.assertEqual(sorted(p.name for p in (self.base / "0").glob("*.bmp")),
                         ["0001.bmp", "0002.bmp", "0003.bmp", "0004.bmp"])

    def append_run(self, seed, samples):
        """Jako main() při volbě Doplnit: seed podle stavu složky, pak generování."""
        existing = len(self.dreamer.own_images(self.base))
        self.dreamer.seed_generators(self.dreamer.run_seed(seed, existing))
        self.generate(samples, split=False, max_retry=5)

    def pixel_digests(self):
        from PIL import Image

        digests = []
        for path in self.dreamer.own_images(self.base):
            with Image.open(path) as img:
                digests.append(self.dreamer.pixel_digest(img))
        return digests

    def test_append_with_same_seed_gives_new_images(self):
        # Dřív druhý běh se stejným --seed zopakoval tutéž sekvenci: 100 souborů, 50 unikátních.
        self.append_run(1234, 5)
        self.append_run(1234, 5)
        digests = self.pixel_digests()
        self.assertEqual(len(digests), 100)
        self.assertEqual(len(set(digests)), 100)

    def test_run_seed(self):
        run_seed = self.dreamer.run_seed
        self.assertEqual(run_seed(1234, 0), 1234)  # bez dat přesně --seed (reprodukovatelnost)
        self.assertNotEqual(run_seed(1234, 50), 1234)
        self.assertEqual(run_seed(1234, 50), run_seed(1234, 50))
        self.assertNotEqual(run_seed(1234, 50), run_seed(1234, 100))
        self.assertLess(run_seed(1234, 50), 2 ** 32)  # numpy přijme jen 32bitový seed

    def test_existing_digests_match_generated(self):
        self.generate(2, split=False)
        self.assertEqual(len(self.dreamer.existing_digests(self.base)), 20)

    def test_limit_9999(self):
        with self.assertRaises(ValueError):
            self.generate(10_000, split=False)

    def test_clean_removes_only_own_structure(self):
        make_bmp(self.base / "3" / "0001.bmp")
        make_bmp(self.base / "train" / "3" / "0001.bmp")
        foreign = make_bmp(self.base / "collected" / "3" / "0001.bmp")  # --output_dir omylem výš
        readme = self.base / "README.md"
        readme.write_text("x", encoding="utf-8")

        self.dreamer.prepare_output_dir(self.base, clean=True)

        self.assertFalse((self.base / "3" / "0001.bmp").exists())
        self.assertFalse((self.base / "train" / "3" / "0001.bmp").exists())
        self.assertTrue(foreign.exists())
        self.assertTrue(readme.exists())


if __name__ == "__main__":
    unittest.main()
