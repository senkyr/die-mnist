"""Procházení datových složek: <třída 0-9>/*.bmp, případně <split>/<třída>/*.bmp."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from die_common.paths import DATA_INFO_FILENAME

NUM_CLASSES = 10  # číslice 0-9; výstup sítě má vždy 10 tříd
SPLITS = ("train", "val", "test")
TRAIN_VAL_MARKER = "train+val"  # DigitTester: pseudo-složka „train a val dohromady"

Sample = Tuple[Path, int]


def check_data_dir(data_dir: Path) -> Path:
    """Ověří, že datová složka existuje; train.py to volá dřív, než založí běh run_*."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Create the dataset with DigitComposer (default output shared/data/composed) "
            "or point --data_dir to another dataset."
        )
    return data_dir


def allocate_counts(total: int, ratios: Sequence[float]) -> List[int]:
    """Rozdělí `total` vzorků podle poměrů metodou největšího zbytku; součet = total.

    Zaokrouhlit každý počet dolů a zbytek dát poslední části systematicky
    zvětšuje poslední část na úkor ostatních (u 9 vzorků a 70/20/10 vyšlo
    6/1/2 místo 6/2/1). Tady dostanou zbylé vzorky části s největším
    zlomkovým zbytkem, při shodě dřívější. Poměry se normalizují součtem.
    """
    if total < 0:
        raise ValueError("Počet vzorků nesmí být záporný.")
    if any(r < 0 for r in ratios):
        raise ValueError("Poměry nesmí být záporné.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("Součet poměrů musí být kladný.")
    quotas = [total * r / ratio_sum for r in ratios]
    counts = [int(q) for q in quotas]
    leftover = total - sum(counts)
    by_remainder = sorted(range(len(ratios)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in by_remainder[:leftover]:
        counts[i] += 1
    return counts


def collect_samples(root_dir: Path, strict: bool = True) -> List[Sample]:
    """Seznam (cesta, třída) pro všechny BMP v podsložkách tříd.

    strict=True (trénink): každá podsložka musí být číslo 0-9, jinak ValueError –
    překlep v názvu nebo třída navíc (Dreamer --extra_shapes) nesmí projít potichu.
    strict=False (testování): bere se jen 0/ až 9/, ostatní se ignoruje.
    Prázdný výsledek není chyba; rozhoduje volající.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Složka s daty '{root_dir}' neexistuje.")

    samples: List[Sample] = []
    if not strict:
        for label in range(NUM_CLASSES):
            class_dir = root_dir / str(label)
            if class_dir.is_dir():
                samples.extend((p, label) for p in sorted(class_dir.glob("*.bmp")))
        return samples

    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        try:
            label = int(label_dir.name)
        except ValueError as exc:
            raise ValueError(
                f"Očekávám složky pojmenované čísly 0-9, našel jsem '{label_dir.name}'."
            ) from exc
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(
                f"Třída '{label_dir.name}' je mimo rozsah 0-{NUM_CLASSES - 1} "
                "(DigitDreamer --extra_shapes vytváří třídy navíc, ty se trénovat nedají)."
            )
        samples.extend((p, label) for p in sorted(label_dir.glob("*.bmp")))
    return samples


def has_samples(split_dir: Path) -> bool:
    """Obsahuje složka aspoň jedno BMP v některé podsložce třídy?"""
    return Path(split_dir).is_dir() and next(Path(split_dir).glob("*/*.bmp"), None) is not None


def is_split_layout(base_dir: Path) -> bool:
    """Je dataset rozdělený do train/val/test? (Rozhoduje existence train/.)"""
    return (Path(base_dir) / "train").is_dir()


def non_empty_splits(base_dir: Path) -> List[str]:
    """Splity (v pořadí train, val, test), které obsahují data; prázdné se přeskočí."""
    return [split for split in SPLITS if has_samples(Path(base_dir) / split)]


def count_samples(directory: Path) -> int:
    directory = Path(directory)
    return sum(
        len(list((directory / str(digit)).glob("*.bmp")))
        for digit in range(NUM_CLASSES)
        if (directory / str(digit)).is_dir()
    )


def detect_data_structure(data_dir: Path) -> Dict:
    """{'has_splits', 'splits': {split: počet}, 'total'} pro nabídku v DigitTesteru."""
    data_dir = Path(data_dir)
    info: Dict = {"has_splits": False, "splits": {}, "total": 0}
    if not data_dir.exists():
        return info

    if is_split_layout(data_dir):
        info["has_splits"] = True
        for split in SPLITS:
            split_dir = data_dir / split
            if split_dir.is_dir():
                count = count_samples(split_dir)
                info["splits"][split] = count
                info["total"] += count
    else:
        info["total"] = count_samples(data_dir)
    return info


def file_digest(path: Path) -> str:
    """MD5 obsahu souboru. Composer kopíruje beze změny, takže vzorek pozná i pod jiným jménem."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_data_info(run_dir: Path, data_dir: Path, layout: str,
                    split_files: Dict[str, Iterable[Path]]) -> Path:
    """Uloží do běhu data.json: odkud data byla a hashe vzorků po splitech.

    Bez toho model nevěděl, na čem se učil. Po přeskládání composed/ (jiný
    seed, poměry, zdroje) pak Tester nabízel test split jako data, která
    model neviděl, přestože je část viděl.
    """
    info = {
        "data_dir": str(Path(data_dir).resolve()),
        "layout": layout,
        "splits": {},
    }
    for split, files in split_files.items():
        digests = sorted(file_digest(p) for p in files)
        info["splits"][split] = {"count": len(digests), "digests": digests}
    path = Path(run_dir) / DATA_INFO_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=1)
    return path


def read_data_info(run_dir: Path) -> Optional[Dict]:
    """data.json běhu, nebo None (starší běh bez něj, poškozený soubor)."""
    try:
        with (Path(run_dir) / DATA_INFO_FILENAME).open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or not isinstance(info.get("splits"), dict):
        return None
    return info


def seen_digests(info: Dict) -> Set[str]:
    """Hashe vzorků, které model při učení viděl (train a val; val vybírá nejlepší epochu)."""
    seen: Set[str] = set()
    for split in ("train", "val"):
        seen.update(info["splits"].get(split, {}).get("digests", []))
    return seen


def count_seen(paths: Iterable[Path], seen: Set[str]) -> int:
    return sum(1 for p in paths if file_digest(p) in seen)


def resolve_eval_dirs(selected_path: Path) -> List[Path]:
    """Složky, ze kterých DigitTester skládá testovací data.

    - <base>/train+val → train a val (co existuje),
    - složka s train/ → všechny splity,
    - jinak ta složka sama.
    """
    selected_path = Path(selected_path)
    if selected_path.name == TRAIN_VAL_MARKER:
        base = selected_path.parent
        return [base / split for split in ("train", "val") if (base / split).exists()]
    if is_split_layout(selected_path):
        return [selected_path / split for split in SPLITS if (selected_path / split).exists()]
    return [selected_path]
