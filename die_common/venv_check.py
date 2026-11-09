"""Kontrola úplnosti virtuálního prostředí; volá ji ensure_venv.bat Pythonem z venv.

Venv leží v TEMP a Windows obsah TEMP občas maže po souborech, takže může
zmizet jen část balíčku. Import to nepozná spolehlivě: numpy bez __init__.py
se naimportuje jako prázdný namespace balíček. Proto se kontroluje, že na disku
jsou všechny soubory, které má každý nainstalovaný balíček zapsané v RECORD,
a že sedí závislosti (pip check).

    python venv_check.py requirements.txt

Kód 0 = v pořádku, 1 = venv je poškozený (založit znovu), 2 = jen chybí
balíčky z requirements (doinstalovat). Vypíše první problémy. Jen standardní knihovna.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Optional


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(req_file: Path) -> List[str]:
    """Jména balíčků z requirements (bez verzí); vnořené -r soubory se projdou taky."""
    names = []
    for raw in Path(req_file).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            nested = line.split(None, 1)[1].strip()
            names.extend(requirement_names(Path(req_file).parent / nested))
            continue
        if line.startswith("-"):
            continue  # další volby pipu (--index-url apod.)
        names.append(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0])
    return names


def recorded_files(dist: metadata.Distribution) -> List[str]:
    """Cesty z RECORD tak, jak jsou zapsané.

    Distribution.files nejde použít: od Pythonu 3.12 chybějící soubory sám
    vyřadí, takže by smazaný numpy/__init__.py nikdy neukázal.
    """
    text = dist.read_text("RECORD")
    if text is None:
        return []
    return [row[0] for row in csv.reader(text.splitlines()) if row and row[0]]


def incomplete_distributions(paths: Optional[Iterable[str]] = None, limit: int = 5) -> List[str]:
    """Popisy problémů: balíček bez jména (poškozená metadata) nebo chybějící soubor z RECORD.

    .pyc se nekontrolují: vznikají a mizí nezávisle na instalaci a Python si je
    při chybění vytvoří znovu.
    """
    problems = []
    dists = metadata.distributions(path=list(paths)) if paths is not None else metadata.distributions()
    for dist in dists:
        name = dist.metadata["Name"] if dist.metadata else None
        if not name:
            problems.append(f"poškozená metadata balíčku v {getattr(dist, '_path', '?')}")
        else:
            for entry in recorded_files(dist):
                if entry.endswith(".pyc"):
                    continue
                if not Path(dist.locate_file(entry)).exists():
                    problems.append(f"{name}: chybí {entry}")
                    break
        if len(problems) >= limit:
            break
    return problems


def missing_requirements(req_file: Path, paths: Optional[Iterable[str]] = None) -> List[str]:
    dists = metadata.distributions(path=list(paths)) if paths is not None else metadata.distributions()
    installed = {normalize(d.metadata["Name"]) for d in dists if d.metadata and d.metadata["Name"]}
    return [f"chybí balíček {n}" for n in requirement_names(req_file) if normalize(n) not in installed]


def pip_check() -> List[str]:
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if result.returncode == 0:
        return []
    lines = (result.stdout + result.stderr).strip().splitlines()
    return ["pip check: " + (lines[0] if lines else f"kód {result.returncode}")]


def main(argv: List[str]) -> int:
    """0 = v pořádku, 1 = venv je poškozený (založit znovu), 2 = chybí jen balíčky (doinstalovat)."""
    problems = incomplete_distributions()
    if not problems:
        problems = pip_check()
    code = 1 if problems else 0
    if not problems and len(argv) > 1:
        # Celá složka dist-info smazaná z TEMP: balíček v seznamu není, pip install ho doplní.
        problems = missing_requirements(Path(argv[1]))
        code = 2 if problems else 0
    for problem in problems:
        print(f"  {problem}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
