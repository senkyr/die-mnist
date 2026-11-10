"""die_common/venv_check.py: pozná venv, ze kterého Windows smazal část souborů v TEMP."""

import importlib
import tempfile
import unittest
from pathlib import Path

from _util import ROOT, load_tool

venv_check = load_tool("die_common_venv_check", "die_common/venv_check.py")


def fake_package(site: Path, name: str, files=("__init__.py", "core.py")) -> Path:
    """site/<name>/… a site/<name>-1.0.dist-info s METADATA a RECORD jako po pip install."""
    pkg = site / name
    pkg.mkdir(parents=True)
    for f in files:
        (pkg / f).write_text("x = 1\n", encoding="utf-8")
    info = site / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8")
    record = [f"{name}/{f},sha256=abc,6" for f in files]
    record.append(f"{name}/__pycache__/core.cpython-313.pyc,,")  # .pyc se nekontroluje
    record.append(f"{name}-1.0.dist-info/METADATA,,")
    record.append(f"{name}-1.0.dist-info/RECORD,,")
    (info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
    return pkg


class VenvCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.site = Path(self.tmp.name) / "site-packages"
        self.site.mkdir()
        self.pkg = fake_package(self.site, "foo")

    def tearDown(self):
        self.tmp.cleanup()

    def problems(self):
        return venv_check.incomplete_distributions(paths=[str(self.site)])

    def test_complete_package_passes(self):
        self.assertEqual(self.problems(), [])  # chybějící .pyc nevadí

    def test_missing_init_is_detected(self):
        # Import by to nepoznal: balíček bez __init__.py se naimportuje jako namespace.
        # Distribution.files chybějící soubory od Pythonu 3.12 vyřazuje, proto čteme RECORD.
        (self.pkg / "__init__.py").unlink()
        self.assertEqual(self.problems(), ["foo: chybí foo/__init__.py"])

    def test_missing_requirement_when_dist_info_deleted(self):
        req = Path(self.tmp.name) / "req.txt"
        req.write_text("foo>=1.0\nbar_baz==2\n", encoding="utf-8")
        self.assertEqual(venv_check.missing_requirements(req, paths=[str(self.site)]),
                         ["chybí balíček bar_baz"])
        fake_package(self.site, "bar-baz")  # jméno se porovnává normalizované (bar_baz = bar-baz)
        # importlib.metadata drží výpis složky v cache podle jejího mtime. Když se balíček
        # založí ve stejném tiku hodin jako první dotaz, mtime se nezmění a druhý dotaz by
        # dostal starý výpis (test pak náhodně padal, v těsné smyčce zhruba v polovině běhů).
        importlib.invalidate_caches()
        self.assertEqual(venv_check.missing_requirements(req, paths=[str(self.site)]), [])

    def test_requirement_names_follow_nested_files(self):
        names = venv_check.requirement_names(ROOT / "DigitLearner" / "requirements.txt")
        self.assertIn("torch", names)
        self.assertIn("Pillow", names)
        self.assertEqual(venv_check.requirement_names(ROOT / "DigitComposer" / "requirements.txt"),
                         ["numpy", "Pillow"])

    def test_every_tool_points_to_existing_shared_requirements(self):
        for tool in ("DigitCollector", "DigitComposer", "DigitDreamer",
                     "DigitLearner", "DigitTeaser", "DigitTester"):
            bat = (ROOT / tool / "run.bat").read_text(encoding="ascii")
            kind = bat.split('set "VENV_KIND=', 1)[1].split('"', 1)[0]
            self.assertTrue((ROOT / "die_common" / f"requirements-{kind}.txt").is_file(), tool)
            self.assertIn(f"requirements-{kind}.txt",
                          (ROOT / tool / "requirements.txt").read_text(encoding="utf-8"), tool)


if __name__ == "__main__":
    unittest.main()
