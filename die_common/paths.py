"""Cesty ke sdíleným datům a modelům, nezávislé na aktuální pracovní složce."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = ROOT / "shared"
DATA_DIR = SHARED_DIR / "data"
COLLECTED_DIR = DATA_DIR / "collected"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
COMPOSED_DIR = DATA_DIR / "composed"
MODELS_DIR = SHARED_DIR / "models"
TESTS_DIR = SHARED_DIR / "tests"
LEARNER_CONFIG = ROOT / "DigitLearner" / "config.json"

MODEL_FILENAME = "digit_cnn.pt"
DATA_INFO_FILENAME = "data.json"  # na jakých datech se běh učil (die_common.data.write_data_info)


def find_model_runs(models_dir: Path = MODELS_DIR) -> List[Path]:
    """Běhy DigitLearneru (run_*) s uloženým modelem, od nejstaršího.

    Název run_YYYYMMDD_HHMMSS[_NN] se řadí chronologicky i jako text.
    Složky bez digit_cnn.pt (přerušený trénink) se vynechají.
    """
    if not models_dir.is_dir():
        return []
    return sorted(p for p in models_dir.glob("run_*") if (p / MODEL_FILENAME).is_file())


def latest_model_run(models_dir: Path = MODELS_DIR) -> Optional[Path]:
    runs = find_model_runs(models_dir)
    return runs[-1] if runs else None


def run_sample_count(run_dir: Path) -> Optional[int]:
    """Počet vzorků, na kterých běh pracoval (num_samples z test_metrics.json), jinak None.

    train.py ho zapisuje jen do test_metrics.json, a to jen při neprázdném testu;
    training_history.json je seznam statistik po epochách a počet vzorků nenese.
    """
    try:
        with (Path(run_dir) / "test_metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, ValueError):  # soubor chybí nebo nejde přečíst, neplatný JSON
        return None
    count = metrics.get("num_samples") if isinstance(metrics, dict) else None
    return count if isinstance(count, int) and not isinstance(count, bool) else None
