"""DigitComposer: detekce zdrojů a ochrana výstupní složky (bez GUI – okno se nevytváří)."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _util import HAS_PIL, HAS_TKINTER, load_tool, make_class_tree


@unittest.skipUnless(HAS_TKINTER and HAS_PIL, "DigitComposer potřebuje tkinter a Pillow")
class ComposerDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.composer = load_tool("digit_composer_main", "DigitComposer/main.py")
        # Metody detekce nepoužívají stav okna – instance bez __init__ (bez Tk).
        cls.app = object.__new__(cls.composer.DigitComposerApp)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def roots(self, path):
        return [Path(p).relative_to(self.base).as_posix() for p in self.app._find_data_roots(str(path))]

    def test_flat_source(self):
        make_class_tree(self.base / "collected", {i: 1 for i in range(10)})
        self.assertEqual(self.roots(self.base / "collected"), ["collected"])

    def test_split_source_gives_all_splits(self):
        # Dřív se našel jen první split (test/) a zbytek dat se tiše ignoroval.
        for split in ("train", "val", "test"):
            make_class_tree(self.base / "synthetic" / split, {i: 1 for i in range(10)})
        self.assertEqual(self.roots(self.base / "synthetic"),
                         ["synthetic/train", "synthetic/val", "synthetic/test"])

    def test_nested_digits_folder(self):
        # Struktura ZIPu z jiné instance DigitCollectoru: jmeno/digits/0..9
        make_class_tree(self.base / "jmeno" / "digits", {i: 1 for i in range(10)})
        self.assertEqual(self.roots(self.base), ["jmeno/digits"])

    def test_needs_at_least_five_classes(self):
        make_class_tree(self.base / "few", {0: 1, 1: 1, 2: 1, 3: 1})
        self.assertEqual(self.app._find_data_roots(str(self.base / "few")), [])
        make_class_tree(self.base / "few", {4: 1})
        self.assertEqual(self.roots(self.base / "few"), ["few"])

    def test_several_datasets_are_all_found(self):
        # Výběr shared/data: dřív jen collected/ a synthetic/ se tiše vynechal.
        make_class_tree(self.base / "collected", {i: 1 for i in range(10)})
        for split in ("train", "val", "test"):
            make_class_tree(self.base / "synthetic" / split, {i: 1 for i in range(10)})
        self.assertEqual(self.roots(self.base),
                         ["collected", "synthetic/train", "synthetic/val", "synthetic/test"])

    def test_zip_with_several_students(self):
        make_class_tree(self.base / "anna" / "digits", {i: 1 for i in range(10)})
        make_class_tree(self.base / "petr" / "digits", {i: 1 for i in range(10)})
        self.assertEqual(self.roots(self.base), ["anna/digits", "petr/digits"])

    def test_nested_composed_output_is_skipped(self):
        composed = self.base / "composed"
        make_class_tree(self.base / "collected", {i: 1 for i in range(10)})
        make_class_tree(composed / "train", {i: 1 for i in range(10)})
        with mock.patch.object(self.composer, "COMPOSED_DIR", composed):
            self.assertEqual(self.roots(self.base), ["collected"])
            # Vybere-li ho uživatel přímo, je to jeho volba.
            self.assertEqual(self.roots(composed), ["composed/train"])

    def test_several_datasets_ask_first(self):
        make_class_tree(self.base / "a", {i: 1 for i in range(10)})
        make_class_tree(self.base / "b", {i: 1 for i in range(10)})
        with mock.patch.object(self.composer, "messagebox") as box:
            box.askyesno.return_value = False
            self.assertIsNone(self.app._confirm_datasets(str(self.base)))
            box.askyesno.return_value = True
            self.assertEqual(len(self.app._confirm_datasets(str(self.base))), 2)
        with mock.patch.object(self.composer, "messagebox") as box:
            self.assertEqual(len(self.app._confirm_datasets(str(self.base / "a"))), 1)
            box.askyesno.assert_not_called()  # jeden dataset: bez ptaní

    def test_is_within(self):
        is_within = self.composer._is_within
        self.assertTrue(is_within(self.base / "a" / "b", self.base))
        self.assertTrue(is_within(self.base, self.base))
        self.assertFalse(is_within(self.base, self.base / "a"))
        self.assertFalse(is_within(self.base / "ab", self.base / "a"))  # prefix textu ≠ podsložka
        if os.name == "nt":
            self.assertFalse(is_within("D:\\data", "C:\\data"))  # jiné disky: bez výjimky

    def test_safe_output_entries(self):
        self.assertEqual(self.composer.COMPOSED_ENTRIES, {"train", "val", "test", "README.md"})


@unittest.skipUnless(HAS_TKINTER and HAS_PIL, "DigitComposer potřebuje tkinter a Pillow")
class ComposerSourceIdTest(unittest.TestCase):
    """ID zdroje je předpona SS v názvu SSOOOO.bmp, musí zůstat v rozsahu 01-99."""

    @classmethod
    def setUpClass(cls):
        cls.composer = load_tool("digit_composer_main", "DigitComposer/main.py")

    def app_with_ids(self, ids):
        app = object.__new__(self.composer.DigitComposerApp)
        app.sources = [SimpleNamespace(id=i) for i in ids]
        return app

    def test_lowest_free_id(self):
        free = self.composer.lowest_free_source_id
        self.assertEqual(free([]), 1)
        self.assertEqual(free([1, 2, 4]), 3)
        self.assertEqual(free(range(1, 99)), 99)

    def test_exhausted_ids_give_clear_error(self):
        with self.assertRaisesRegex(ValueError, "99 ID"):
            self.composer.lowest_free_source_id(range(1, 100))

    def test_repeated_add_and_remove_keeps_two_digit_ids(self):
        # Dřív počítadlo jen rostlo: po 150 přidáních s odebíráním mělo ID 150
        # a soubory se jmenovaly 1500001.bmp místo SSOOOO.
        app = self.app_with_ids([])
        for _ in range(150):
            source_id = app._next_source_id()
            self.assertTrue(1 <= source_id <= self.composer.MAX_SOURCES)
            app.sources.append(SimpleNamespace(id=source_id))
            self.assertEqual(len(f"{source_id:02d}{1:04d}.bmp"), len("SSOOOO.bmp"))
            if len(app.sources) > 3:
                app.sources.pop(0)

    def test_removed_id_is_reused(self):
        app = self.app_with_ids([1, 2, 3])
        del app.sources[1]
        self.assertEqual(app._next_source_id(), 2)


@unittest.skipUnless(HAS_TKINTER and HAS_PIL, "DigitComposer potřebuje tkinter a Pillow")
class ComposerOutputTest(unittest.TestCase):
    """Výstup do shared/data/composed: založení, start dialogu, --pipeline (okna nahrazena mocky)."""

    @classmethod
    def setUpClass(cls):
        cls.composer = load_tool("digit_composer_main", "DigitComposer/main.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.composed = self.base / "shared" / "data" / "composed"
        self.source = make_class_tree(self.base / "shared" / "data" / "synthetic", {i: 9 for i in range(10)})
        patches = [
            mock.patch.object(self.composer, "COMPOSED_DIR", self.composed),
            mock.patch.object(self.composer, "tk", mock.MagicMock()),
            mock.patch.object(self.composer, "ttk", mock.MagicMock()),
            mock.patch.object(self.composer, "messagebox"),
            mock.patch.object(self.composer, "filedialog"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def make_app(self):
        app = object.__new__(self.composer.DigitComposerApp)
        app.master = mock.MagicMock()
        app.sources = [self.composer.DataSource(1, "dir", "synthetic", str(self.source))]
        app.composed_written = False
        app.train_ratio = SimpleNamespace(get=lambda: 0.7)
        app.val_ratio = SimpleNamespace(get=lambda: 0.2)
        app.test_ratio = SimpleNamespace(get=lambda: 0.1)
        app.seed_var = SimpleNamespace(get=lambda: 42)
        return app

    def compose_into(self, output_dir):
        self.composer.filedialog.askdirectory.return_value = str(output_dir) if output_dir else ""
        app = self.make_app()
        app.compose_dataset()
        return app

    def test_missing_composed_is_created_and_dialog_starts_there(self):
        self.assertFalse(self.composed.exists())  # čerstvý klon: composed/ nikdy nebyla v repu
        self.compose_into(None)  # uživatel dialog zavře
        self.assertTrue(self.composed.is_dir())
        initialdir = self.composer.filedialog.askdirectory.call_args.kwargs["initialdir"]
        self.assertEqual(Path(initialdir), self.composed)

    def test_confirming_default_writes_dataset_and_counts_per_class(self):
        app = self.compose_into(self.composed)
        self.assertTrue(app.composed_written)
        # 9 vzorků na třídu a 70/20/10 → 6/2/1 (dřív 6/1/2)
        for split, per_class in (("train", 6), ("val", 2), ("test", 1)):
            for c in range(10):
                self.assertEqual(len(list((self.composed / split / str(c)).glob("*.bmp"))), per_class)
        message = self.composer.messagebox.showinfo.call_args.args[1]
        self.assertNotIn("--data_dir", message)

    def test_other_folder_warns_and_does_not_count_as_composed(self):
        other = self.base / "jinde"
        app = self.compose_into(other)
        self.assertFalse(app.composed_written)
        self.assertTrue((other / "train" / "0").is_dir())
        message = self.composer.messagebox.showinfo.call_args.args[1]
        self.assertIn("--data_dir", message)

    def test_pipeline_exit_code(self):
        code = self.composer.pipeline_exit_code
        self.assertEqual(code(pipeline=True, composed_written=False), 1)
        self.assertEqual(code(pipeline=True, composed_written=True), 0)
        self.assertEqual(code(pipeline=False, composed_written=False), 0)  # samostatný run.bat beze změny
        self.assertTrue(self.composer.parse_args(["--pipeline"]).pipeline)
        self.assertFalse(self.composer.parse_args([]).pipeline)


if __name__ == "__main__":
    unittest.main()
