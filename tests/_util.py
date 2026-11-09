"""Pomůcky pro testy: kořen repozitáře v sys.path, načtení nástroje, tvorba BMP."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TKINTER = importlib.util.find_spec("tkinter") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def load_tool(module_name: str, relative_path: str):
    """Načte skript nástroje (všechny se jmenují main.py) pod unikátním jménem modulu."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses potřebují modul v sys.modules
    spec.loader.exec_module(module)
    return module


def make_bmp(path: Path, value: int = 128) -> Path:
    """Minimální BMP soubor. Bez PIL stačí prázdný soubor – testy čtou jen názvy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PIL:
        from PIL import Image

        Image.new("L", (32, 32), value).save(path)
    else:
        path.write_bytes(b"BM")
    return path


def make_class_tree(base: Path, counts: dict, names: Iterable[str] = ()) -> Path:
    """base/<třída>/0001.bmp … podle counts {třída: počet}."""
    for label, count in counts.items():
        for i in range(1, count + 1):
            make_bmp(base / str(label) / f"{i:04d}.bmp")
    for name in names:
        (base / name).mkdir(parents=True, exist_ok=True)
    return base
