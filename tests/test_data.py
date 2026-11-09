import tempfile
import unittest
from pathlib import Path

from _util import make_bmp, make_class_tree

from die_common.data import (
    allocate_counts,
    check_data_dir,
    collect_samples,
    count_seen,
    detect_data_structure,
    file_digest,
    has_samples,
    is_split_layout,
    non_empty_splits,
    read_data_info,
    resolve_eval_dirs,
    seen_digests,
    write_data_info,
)


class CollectSamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_come_from_folder_names(self):
        make_class_tree(self.base, {0: 2, 3: 1, 9: 1})
        (self.base / "README.md").write_text("x", encoding="utf-8")  # soubor v kořeni se ignoruje
        samples = collect_samples(self.base)
        self.assertEqual(sorted(label for _, label in samples), [0, 0, 3, 9])
        self.assertTrue(all(p.suffix == ".bmp" for p, _ in samples))

    def test_strict_rejects_non_digit_folder(self):
        make_class_tree(self.base, {1: 1}, names=["jedna"])
        with self.assertRaisesRegex(ValueError, "jedna"):
            collect_samples(self.base)

    def test_strict_rejects_extra_class(self):
        # DigitDreamer --extra_shapes vytváří třídu 10 – síť má jen 10 výstupů
        make_class_tree(self.base, {1: 1, 10: 1})
        with self.assertRaisesRegex(ValueError, "mimo rozsah"):
            collect_samples(self.base)

    def test_non_strict_ignores_unknown_folders(self):
        make_class_tree(self.base, {2: 2, 10: 1}, names=["jina"])
        samples = collect_samples(self.base, strict=False)
        self.assertEqual([label for _, label in samples], [2, 2])

    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            collect_samples(self.base / "neni")

    def test_empty_folder_is_empty_list(self):
        self.assertEqual(collect_samples(self.base), [])


class CheckDataDirTest(unittest.TestCase):
    """train.py volá check_data_dir před založením běhu run_* (jinak zůstávaly prázdné běhy)."""

    def test_missing_or_file_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Výchozí composed/ plní DigitComposer, ne DigitDreamer – hláška musí radit správně.
            with self.assertRaisesRegex(FileNotFoundError, "DigitComposer.*--data_dir"):
                check_data_dir(base / "neni")
            (base / "soubor.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                check_data_dir(base / "soubor.txt")

    def test_existing_folder_is_returned_as_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check_data_dir(tmp), Path(tmp))


class AllocateCountsTest(unittest.TestCase):
    """Rozdělení vzorků jedné třídy do train/val/test (DigitComposer)."""

    def test_small_class_does_not_inflate_test(self):
        # Dřív int() u train i val a zbytek do testu: 9 vzorků → 6/1/2.
        self.assertEqual(allocate_counts(9, (0.7, 0.2, 0.1)), [6, 2, 1])

    def test_sum_is_always_total(self):
        for ratios in ((0.7, 0.2, 0.1), (0.8, 0.1, 0.1), (0.5, 0.25, 0.25), (1.0, 0.0, 0.0)):
            for total in range(0, 60):
                counts = allocate_counts(total, ratios)
                self.assertEqual(sum(counts), total, (total, ratios))
                self.assertTrue(all(c >= 0 for c in counts))

    def test_exact_ratios_and_zero_parts(self):
        self.assertEqual(allocate_counts(100, (0.7, 0.2, 0.1)), [70, 20, 10])
        self.assertEqual(allocate_counts(7, (1.0, 0.0, 0.0)), [7, 0, 0])
        self.assertEqual(allocate_counts(0, (0.7, 0.2, 0.1)), [0, 0, 0])

    def test_each_part_within_one_of_its_quota(self):
        for total in range(1, 200):
            for count, ratio in zip(allocate_counts(total, (0.7, 0.2, 0.1)), (0.7, 0.2, 0.1)):
                self.assertLess(abs(count - total * ratio), 1.0)

    def test_ratios_are_normalized(self):
        # Posuvníky dávají součet „skoro 1" (0.7000000001 apod.).
        self.assertEqual(allocate_counts(10, (7, 2, 1)), [7, 2, 1])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            allocate_counts(5, (0, 0, 0))
        with self.assertRaises(ValueError):
            allocate_counts(5, (0.5, -0.1, 0.6))


class DataInfoTest(unittest.TestCase):
    """data.json běhu: Tester podle něj pozná vzorky, které model viděl při učení."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.run = self.base / "run"
        self.run.mkdir()
        # Různý obsah, ať hashe nesplývají (make_bmp kreslí jednu hodnotu na celý obrázek).
        self.a = make_bmp(self.base / "data" / "train" / "0" / "010001.bmp", 10)
        self.b = make_bmp(self.base / "data" / "val" / "0" / "010002.bmp", 20)
        self.c = make_bmp(self.base / "data" / "test" / "0" / "010003.bmp", 30)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_seen_set(self):
        write_data_info(self.run, self.base / "data", "split",
                        {"train": [self.a], "val": [self.b], "test": [self.c]})
        info = read_data_info(self.run)
        self.assertEqual(info["layout"], "split")
        self.assertEqual(info["splits"]["test"]["count"], 1)
        self.assertEqual(seen_digests(info), {file_digest(self.a), file_digest(self.b)})

    def test_recomposed_test_split_is_detected(self):
        write_data_info(self.run, self.base / "data", "split",
                        {"train": [self.a], "val": [self.b], "test": [self.c]})
        seen = seen_digests(read_data_info(self.run))
        self.assertEqual(count_seen([self.c], seen), 0)  # původní test model neviděl
        # Composer po tréninku přeskládal: vzorek z train je teď v testu (jiná cesta, týž obsah).
        moved = self.base / "data2" / "test" / "0" / "010001.bmp"
        moved.parent.mkdir(parents=True)
        moved.write_bytes(self.a.read_bytes())
        self.assertEqual(count_seen([self.c, moved], seen), 1)

    def test_missing_or_broken_info(self):
        self.assertIsNone(read_data_info(self.run))  # starší běh bez data.json
        (self.run / "data.json").write_text("{nejson", encoding="utf-8")
        self.assertIsNone(read_data_info(self.run))
        (self.run / "data.json").write_text("[]", encoding="utf-8")
        self.assertIsNone(read_data_info(self.run))


class SplitLayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        make_class_tree(self.base / "train", {0: 3, 1: 2})
        (self.base / "val" / "0").mkdir(parents=True)  # Composer s val 0 %: prázdné složky tříd
        make_class_tree(self.base / "test", {1: 1})

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_split_is_skipped(self):
        self.assertTrue(is_split_layout(self.base))
        self.assertTrue(has_samples(self.base / "train"))
        self.assertFalse(has_samples(self.base / "val"))
        self.assertEqual(non_empty_splits(self.base), ["train", "test"])

    def test_flat_folder_is_not_split_layout(self):
        flat = make_class_tree(self.base / "flat", {0: 1})
        self.assertFalse(is_split_layout(flat))
        self.assertEqual(non_empty_splits(flat), [])

    def test_detect_data_structure_counts(self):
        info = detect_data_structure(self.base)
        self.assertTrue(info["has_splits"])
        self.assertEqual(info["splits"], {"train": 5, "val": 0, "test": 1})
        self.assertEqual(info["total"], 6)

        with tempfile.TemporaryDirectory() as tmp:
            flat = make_class_tree(Path(tmp), {4: 2})
            self.assertEqual(detect_data_structure(flat), {"has_splits": False, "splits": {}, "total": 2})

    def test_detect_missing(self):
        self.assertEqual(detect_data_structure(self.base / "neni")["total"], 0)

    def test_resolve_eval_dirs(self):
        # „kompletní dataset" = všechny splity
        self.assertEqual(resolve_eval_dirs(self.base),
                         [self.base / "train", self.base / "val", self.base / "test"])
        # pseudo-složka train+val
        self.assertEqual(resolve_eval_dirs(self.base / "train+val"),
                         [self.base / "train", self.base / "val"])
        # konkrétní split
        self.assertEqual(resolve_eval_dirs(self.base / "test"), [self.base / "test"])

    def test_make_bmp_helper(self):
        path = make_bmp(self.base / "x" / "0001.bmp")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
