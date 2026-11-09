"""DigitCollector: uložení vzorku se zatržítky „Přidávat šum“ a „Negovat barvy“ (okno nahrazeno mocky)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _util import HAS_NUMPY, HAS_PIL, HAS_TKINTER, load_tool


@unittest.skipUnless(HAS_TKINTER and HAS_PIL and HAS_NUMPY, "DigitCollector potřebuje tkinter, Pillow a numpy")
class CollectorSaveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = load_tool("digit_collector_main", "DigitCollector/main.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(self.collector, "DIGITS_DIR", self.tmp.name)
        p.start()
        self.addCleanup(p.stop)
        for digit in range(10):  # v aplikaci je zakládá setup_directories
            (Path(self.tmp.name) / str(digit)).mkdir()

    def save(self, noise, invert):
        from PIL import Image, ImageDraw

        app = object.__new__(self.collector.DigitCollectorApp)
        app.master = mock.MagicMock()
        app.target_digit = 7
        app.has_drawing = True
        app.noise_var = SimpleNamespace(get=lambda: noise)
        app.invert_var = SimpleNamespace(get=lambda: invert)
        app.show_noisy_preview = mock.MagicMock()
        app._post_save_reset = mock.MagicMock()
        app.image = Image.new("L", (256, 256), 0)
        ImageDraw.Draw(app.image).line([20, 128, 236, 128], fill=255, width=15)
        app.save_digit()
        saved = sorted((Path(self.tmp.name) / "7").glob("*.bmp"))[-1]
        with Image.open(saved) as img:
            return img.convert("L").copy(), app

    def test_plain_sample_has_dark_background(self):
        img, app = self.save(noise=False, invert=False)
        self.assertEqual(img.getpixel((0, 0)), 0)
        self.assertGreater(img.getpixel((16, 16)), 200)  # tah uprostřed (po LANCZOS ne přesně 255)
        app.show_noisy_preview.assert_not_called()

    def test_negated_sample_is_exact_negative(self):
        plain, _ = self.save(noise=False, invert=False)
        negated, app = self.save(noise=False, invert=True)
        self.assertEqual(negated.getpixel((0, 0)), 255)
        self.assertEqual(list(negated.getdata()), [255 - v for v in plain.getdata()])
        app.show_noisy_preview.assert_called_once()  # ukáže, co se uložilo

    def test_noise_then_negation(self):
        # Šum zvedá tmavé pozadí, po negaci tedy pozadí zůstane světlé, ale ne všude 255.
        img, _ = self.save(noise=True, invert=True)
        corner = [img.getpixel((x, y)) for x in range(4) for y in range(4)]
        self.assertGreater(min(corner), 128)
        self.assertLess(min(corner), 255)

    def test_window_builds_with_both_checkboxes(self):
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError as exc:  # bez displeje
            self.skipTest(f"Tk nejde spustit: {exc}")
        try:
            app = self.collector.DigitCollectorApp(root)
            root.update()
            texts = [w.cget("text") for w in app.options_frame.winfo_children()]
            self.assertEqual(texts, ["Přidávat šum", "Negovat barvy"])
            app._enter_stats_view()
            self.assertFalse(app.options_frame.winfo_ismapped())
            app._enter_draw_view()
            root.update()
            self.assertTrue(app.options_frame.winfo_ismapped())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
