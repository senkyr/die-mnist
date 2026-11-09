"""Hyperparametry DigitLearneru: výchozí hodnoty, config.json, CLI a validace.

Bez PyTorch – sdílí ho train.py i editor marshall.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from die_common.paths import COMPOSED_DIR, LEARNER_CONFIG, MODELS_DIR


@dataclass
class HyperParams:
    """Konfigurace tréninku načítaná z konfigu nebo CLI."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    step_size: int = 10
    lr_gamma: float = 0.5
    dropout: float = 0.2
    seed: int = 42

    def update_from_dict(self, data: Dict[str, object]) -> None:
        for field in fields(self):
            if field.name in data:
                # JSON může nést "20" nebo 20.0 místo 20 – přetypuji podle výchozí hodnoty.
                caster = type(getattr(self, field.name))
                try:
                    setattr(self, field.name, caster(data[field.name]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Neplatná hodnota '{data[field.name]}' pro '{field.name}' v konfiguraci."
                    ) from exc

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def as_dict(self) -> Dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def validate(self) -> None:
        if not 0.0 <= self.val_ratio <= 0.5:
            raise ValueError("val_ratio ocekavam v intervalu <0.0, 0.5>.")
        if not 0.0 <= self.test_ratio <= 0.5:
            raise ValueError("test_ratio ocekavam v intervalu <0.0, 0.5>.")
        if self.val_ratio + self.test_ratio >= 0.8:
            raise ValueError("Na trenink musi zustat alespon 20 % dat (upravte pomery val/test).")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout ocekavam v intervalu <0.0, 1.0>.")
        if self.step_size <= 0:
            raise ValueError("step_size musi byt kladne cislo.")
        if self.lr_gamma <= 0.0 or self.lr_gamma > 1.0:
            raise ValueError("lr_gamma ocekavam v rozsahu (0.0, 1.0].")
        if self.epochs <= 0:
            raise ValueError("epochs musi byt kladne cislo.")
        if self.batch_size <= 0:
            raise ValueError("batch_size musi byt kladne cislo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate musi byt kladne cislo.")


def load_hyperparams(
    config_path: Optional[Path],
    overrides: Dict[str, object],
    default_config_path: Path = LEARNER_CONFIG,
) -> Tuple[HyperParams, Path, bool]:
    """Výchozí hodnoty ← config.json ← CLI overrides (None = nepřepisovat), pak validace.

    Chybějící výchozí config.json není chyba (použijí se vestavěné hodnoty),
    chybějící explicitně zadaný config ano. Vrací (hp, cesta, zda se config načetl).
    """
    if config_path is None:
        config_path = default_config_path
    config_path = Path(config_path)

    hp = HyperParams()
    loaded = False
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Konfiguracni JSON musi obsahovat objekt (dict).")
        hp.update_from_dict(data)
        loaded = True
    elif config_path.resolve() != Path(default_config_path).resolve():
        raise FileNotFoundError(f"Konfiguracni soubor {config_path} neexistuje.")

    hp.apply_overrides(overrides)
    hp.validate()
    return hp, config_path, loaded


def overrides_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {field.name: getattr(args, field.name, None) for field in fields(HyperParams)}


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trains the digit classifier on data from DigitCollector."
    )
    parser.add_argument("--config", type=Path, default=LEARNER_CONFIG,
                        help="Path to the JSON hyperparameter file.")
    parser.add_argument("--data_dir", type=Path, default=COMPOSED_DIR,
                        help="Directory with subfolders 0-9 (or train/val/test) containing BMP images "
                             "(default shared/data/composed).")
    parser.add_argument("--output_dir", type=Path, default=MODELS_DIR,
                        help="Root directory for all runs (default shared/models).")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Override validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=None, help="Override test split ratio.")
    parser.add_argument("--step_size", type=int, default=None, help="Override StepLR step_size.")
    parser.add_argument("--lr_gamma", type=float, default=None, help="Override StepLR gamma.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout probability.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser


def parse_train_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_train_parser().parse_args(argv)


def split_lengths(total: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Počty (train, val, test) pro náhodný split; zbytek po zaokrouhlení jde do train."""
    if total < 3:
        raise ValueError("Dataset je příliš malý na rozdělení.")
    val_len = int(total * val_ratio)
    test_len = int(total * test_ratio)
    train_len = total - val_len - test_len
    if train_len <= 0:
        raise ValueError("Zkontrolujte poměry pro validaci/test – na trénink nezbyla data.")
    return train_len, val_len, test_len
