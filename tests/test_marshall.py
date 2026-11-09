"""DigitLearner/marshall.py: editor neuloží hodnoty, které train.py odmítne (bez GUI)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _util import HAS_TKINTER, load_tool


@unittest.skipUnless(HAS_TKINTER, "marshall.py importuje tkinter")
class MarshallSaveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marshall = load_tool("digit_learner_marshall", "DigitLearner/marshall.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def editor(self, **values):
        """Instance bez __init__ (bez Tk); posuvníky nahrazují objekty s get()."""
        data = dict(self.marshall.DEFAULTS, **values)
        editor = object.__new__(self.marshall.HyperparamEditor)
        editor.vars = {key: SimpleNamespace(get=lambda v=value: v) for key, value in data.items()}
        return editor

    def save(self, editor):
        with mock.patch.object(self.marshall, "CONFIG_PATH", self.config_path), \
                mock.patch.object(self.marshall.messagebox, "showerror") as showerror:
            saved = editor.save()
        return saved, showerror

    def test_slider_limits_allow_invalid_split(self):
        # Předpoklad testu: posuvníky samy val 0,5 + test 0,5 dovolí.
        limits = {p["key"]: p["max"] for p in self.marshall.PARAMS}
        self.assertEqual((limits["val_ratio"], limits["test_ratio"]), (0.5, 0.5))
        with self.assertRaises(ValueError):
            self.marshall.validate_values({"val_ratio": 0.5, "test_ratio": 0.5})

    def test_invalid_values_are_not_saved_and_reported(self):
        saved, showerror = self.save(self.editor(val_ratio=0.5, test_ratio=0.5))
        self.assertFalse(saved)
        self.assertFalse(self.config_path.exists())
        showerror.assert_called_once()
        self.assertIn("20 %", showerror.call_args.args[1])

    def test_valid_values_are_saved(self):
        saved, showerror = self.save(self.editor(epochs=5, val_ratio=0.25))
        self.assertTrue(saved)
        showerror.assert_not_called()
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(data["epochs"], 5)
        self.assertEqual(data["val_ratio"], 0.25)

    def test_enter_keeps_window_open_on_invalid_values(self):
        editor = self.editor(val_ratio=0.5, test_ratio=0.5)
        editor.close = mock.Mock()
        with mock.patch.object(self.marshall, "CONFIG_PATH", self.config_path), \
                mock.patch.object(self.marshall.messagebox, "showerror"):
            editor._save_and_close()
        editor.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
