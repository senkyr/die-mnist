"""
DigitTester - Kvantifikované testování natrénovaných modelů.

CLI nástroj pro evaluaci modelů z DigitLearner na testovacích datech.
Vypočítá detailní metriky včetně confusion matrix a per-class accuracy.

Features:
- Interaktivní výběr modelu ze seznamu dostupných modelů
- Interaktivní výběr testovacích dat (kompletní dataset / test split / train+val)
- Automatická vizualizace výsledků s interaktivní confusion matrix
- Export grafů jako PNG

Spuštění:
    python main.py                    # Interaktivní výběr
    python main.py --model_dir PATH   # Explicitní model
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Kořen repozitáře do sys.path kvůli sdílenému balíčku die_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from die_common.data import (  # noqa: E402
    NUM_CLASSES,
    TRAIN_VAL_MARKER,
    collect_samples,
    count_seen,
    detect_data_structure,
    is_split_layout,
    read_data_info,
    resolve_eval_dirs,
    seen_digests,
)
from die_common.model import load_model as load_weights  # noqa: E402
from die_common.paths import COMPOSED_DIR, MODELS_DIR, TESTS_DIR, find_model_runs  # noqa: E402

DEFAULT_MODELS_DIR = MODELS_DIR
DEFAULT_DATA_BASE = COMPOSED_DIR
DEFAULT_TESTS_DIR = TESTS_DIR


class DigitDataset(Dataset):
    """Dataset, který čte BMP soubory ze struktury složek 0-9."""

    def __init__(self, root_dirs: List[Path], verify_size: bool = True) -> None:
        if isinstance(root_dirs, Path):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.verify_size = verify_size
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for root_dir in self.root_dirs:
            self.samples.extend(collect_samples(root_dir, strict=False))

        if not self.samples:
            raise ValueError(f"V zadaných složkách nejsou žádná .bmp data.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")

        if self.verify_size and img.size != (32, 32):
            raise ValueError(f"Obrázek {img_path} má velikost {img.size}, očekává se (32, 32).")

        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor, label, str(img_path)


def load_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Načte model a konfiguraci ze složky."""
    config_path = model_dir / "config.json"
    model_path = model_dir / "digit_cnn.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurace nenalezena: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model nenalezen: {model_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    dropout = config.get("dropout", 0.2)
    num_classes = config.get("num_classes", NUM_CLASSES)

    model = load_weights(model_path, num_classes=num_classes, dropout=dropout, device=device)

    return model, config


def evaluate_with_confusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
) -> Tuple[float, float, np.ndarray, List[Dict[str, object]]]:
    """Evaluuje model a vrátí loss, accuracy, confusion matrix a predictions."""
    model.eval()
    running_loss = 0.0
    total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            _, predicted = torch.max(outputs, 1)
            for path, true_label, pred_label in zip(paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true_label][pred_label] += 1
                predictions.append({
                    "path": path,
                    "true": int(true_label),
                    "pred": int(pred_label)
                })

    average_loss = running_loss / total if total else 0.0
    accuracy = confusion.trace() / total if total else 0.0

    return average_loss, accuracy, confusion, predictions


def compute_per_class_metrics(confusion: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Vypočítá per-class precision, recall, f1-score."""
    num_classes = confusion.shape[0]
    metrics = {}

    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        tn = confusion.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / confusion[cls, :].sum() if confusion[cls, :].sum() > 0 else 0.0

        metrics[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "accuracy": float(accuracy),
            "samples": int(confusion[cls, :].sum()),
        }

    return metrics


def save_results(results: Dict, output_path: Path) -> None:
    """Uloží výsledky do JSON souboru."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nVýsledky uloženy do: {output_path}")


def print_results(results: Dict) -> None:
    """Vytiskne výsledky do konzole."""
    print("\n" + "=" * 80)
    print("VÝSLEDKY TESTOVÁNÍ")
    print("=" * 80)
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Average Loss:     {results['average_loss']:.4f}")
    print(f"Total Samples:    {results['total_samples']}")
    print(f"\nModel:            {results['model_dir']}")
    print(f"Data:             {results['data_dir']}")
    print(f"Device:           {results['device']}")

    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)
    print(f"{'Class':>5} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Samples':>10}")
    print("-" * 80)

    per_class = results["per_class_metrics"]
    for cls in sorted(per_class.keys(), key=int):
        m = per_class[cls]
        print(
            f"{cls:>5} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['samples']:>10}"
        )

    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print("Řádky = skutečné třídy, Sloupce = predikované třídy")
    print()

    confusion = np.array(results["confusion_matrix"])
    num_classes = confusion.shape[0]

    # Header
    print("    ", end="")
    for i in range(num_classes):
        print(f"{i:>6}", end="")
    print()
    print("    " + "-" * (6 * num_classes))

    # Rows
    for i in range(num_classes):
        print(f"{i:>3} |", end="")
        for j in range(num_classes):
            print(f"{confusion[i, j]:>6}", end="")
        print()

    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DigitTester - Kvantifikované testování modelů z DigitLearner"
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Cesta ke složce s modelem (default: nejnovější v ../shared/models/)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Cesta k testovacím datům (default: interaktivní výběr v ../shared/data/composed/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Velikost batche pro evaluaci (default: 64)",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Vynutit použití CPU i když je GPU dostupné",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Cesta k výstupnímu JSON souboru (default: ../shared/tests/test_results_TIMESTAMP.json)",
    )

    return parser.parse_args()


def get_model_info(model_dir: Path) -> Dict:
    """Načte informace o modelu z config.json a test_metrics.json."""
    info = {
        "name": model_dir.name,
        "path": model_dir,
        "epochs": "?",
        "accuracy": "?",
        "test_acc": None,
    }

    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                info["epochs"] = config.get("epochs", "?")
        except Exception:
            pass

    # Load test metrics if available
    test_metrics_path = model_dir / "test_metrics.json"
    if test_metrics_path.exists():
        try:
            with test_metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
                info["test_acc"] = metrics.get("test_accuracy")
        except Exception:
            pass

    # Load training history for final accuracy
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as f:
                history = json.load(f)
                if history:
                    last_epoch = history[-1]
                    info["accuracy"] = last_epoch.get("val_acc", last_epoch.get("train_acc", "?"))
        except Exception:
            pass

    return info


def report_seen_samples(model_dir: Path, selected_path: Path, dataset: DigitDataset) -> Tuple[Optional[int], bool]:
    """Kolik vyhodnocovaných vzorků model viděl při učení (train/val podle data.json běhu).

    Vypíše varování, když jde o data, která měla být pro model nová (test split
    nebo jiná složka), a přesto je zčásti viděl – typicky po přeskládání
    composed/ Composerem až po tréninku. Vrací (počet nebo None u běhu bez
    data.json, zda se varovalo).
    """
    info = read_data_info(model_dir)
    if info is None:
        print("Poznámka: model nemá data.json (starší běh), nelze ověřit, "
              "jestli testovací data neviděl při učení.")
        return None, False

    seen = count_seen((path for path, _ in dataset.samples), seen_digests(info))
    total = len(dataset)
    print(f"Model se učil na: {info.get('data_dir', '?')}")
    print(f"Vzorky, které model viděl při učení (train/val): {seen} z {total}")
    deliberate = selected_path.name == TRAIN_VAL_MARKER or is_split_layout(selected_path)
    warn = bool(seen) and not deliberate
    if warn:
        print()
        print(f"POZOR: {seen} z {total} testovacích vzorků model viděl při učení, "
              "přesnost bude nadsazená.")
        print("Data se nejspíš od tréninku přeskládala (DigitComposer) nebo obsahují kopie. "
              "Pro poctivý odhad model znovu natrénujte.")
    return seen, warn


def warn_full_dataset(selected_path: Path) -> bool:
    """Varuje, když --data_dir ukazuje na kořen se splity: testuje se i na train/val.

    Interaktivní nabídka to u volby „Kompletní dataset“ říká, CLI dřív mlčelo.
    """
    if not is_split_layout(selected_path):
        return False
    print(f"POZOR: {selected_path} obsahuje train/val/test a testuje se na všech splitech včetně dat, "
          "na kterých se model učil – přesnost bude nadsazená.")
    print(f"Pro poctivý odhad použijte --data_dir {Path(selected_path) / 'test'}")
    return True


def create_test_dataset(selected_path: Path) -> DigitDataset:
    """Vytvoří dataset na základě vybrané cesty (viz die_common.data.resolve_eval_dirs)."""
    return DigitDataset(resolve_eval_dirs(selected_path))


def select_test_data_interactive(base_data_dir: Path) -> Optional[Path]:
    """Interaktivní výběr testovacích dat."""
    if not base_data_dir.exists():
        print(f"Chyba: Složka '{base_data_dir}' neexistuje")
        return None

    structure = detect_data_structure(base_data_dir)

    if not structure["has_splits"]:
        # Flat structure - just use it directly
        if structure["total"] == 0:
            print(f"Chyba: V '{base_data_dir}' nejsou žádná data")
            return None
        print(f"\nPoužívám kompletní dataset ({structure['total']} vzorků)")
        return base_data_dir

    # Has train/val/test splits - offer choices
    splits = structure["splits"]

    print("\n" + "=" * 80)
    print("VÝBĚR TESTOVACÍCH DAT")
    print("=" * 80)
    print(f"Dostupné splits v {base_data_dir.name}/:")

    split_info = []
    for split_name in ["train", "val", "test"]:
        count = splits.get(split_name, 0)
        if count > 0:
            split_info.append(f"{split_name}: {count} vzorků")

    print("  " + "  |  ".join(split_info))
    print("-" * 80)

    # Build options
    options = []
    option_paths = []

    # Poctivý odhad přesnosti dává jen test split – train i val model během
    # učení viděl (val vybírá nejlepší epochu), takže výsledek nadsazují.
    if "test" in splits and splits["test"] > 0:
        option_paths.append("test")
        options.append(f"{len(option_paths)}. Pouze test split ({splits['test']} vzorků) [DOPORUČENO]")
        options.append("   → Data, která model při učení neviděl – poctivý odhad přesnosti")

    option_paths.append("all")
    options.append(f"\n{len(option_paths)}. Kompletní dataset ({structure['total']} vzorků)")
    options.append("   → Obsahuje i trénovací data, přesnost bude nadsazená")

    train_val_count = splits.get("train", 0) + splits.get("val", 0)
    if train_val_count > 0:
        option_paths.append("train+val")
        options.append(f"\n{len(option_paths)}. Train + Val splits ({train_val_count} vzorků)")
        options.append("   → Data, která model viděl během učení (kontrola přeučení)")

    for opt in options:
        print(opt)

    print("=" * 80)
    print()

    # Get user choice
    while True:
        try:
            default_label = "test split" if option_paths[0] == "test" else "kompletní dataset"
            choice = input(f"Vyberte [1-{len(option_paths)}] (Enter = {default_label}): ").strip()

            if choice == "":
                choice = "1"

            choice_num = int(choice)
            if 1 <= choice_num <= len(option_paths):
                selected_option = option_paths[choice_num - 1]

                if selected_option == "all":
                    print(f"\nVybrána všechna data ({structure['total']} vzorků)")
                    return base_data_dir
                elif selected_option == "test":
                    test_dir = base_data_dir / "test"
                    print(f"\nVybrán test split ({splits['test']} vzorků)")
                    return test_dir
                elif selected_option == "train+val":
                    # For train+val, we need to combine them
                    # Return base_data_dir with a marker
                    print(f"\nVybrány train+val splits ({train_val_count} vzorků)")
                    return base_data_dir / "train+val"  # Special marker
            else:
                print(f"Chyba: Zadejte číslo mezi 1 a {len(option_paths)}")

        except ValueError:
            print("Chyba: Zadejte platné číslo nebo Enter")
        except KeyboardInterrupt:
            print("\n\nTestování zrušeno.")
            return None


def select_model_interactive(models_dir: Path) -> Optional[Path]:
    """Interaktivní výběr modelu ze seznamu."""
    if not models_dir.exists():
        print(f"Chyba: Složka {models_dir} neexistuje")
        print("Spusťte nejprve DigitLearner nebo zadejte --model_dir explicitně")
        return None

    run_dirs = sorted(
        find_model_runs(models_dir),
        key=lambda p: p.name,
        reverse=True
    )

    if not run_dirs:
        print(f"Chyba: Žádné modely nenalezeny v {models_dir}")
        print("Spusťte nejprve DigitLearner nebo zadejte --model_dir explicitně")
        return None

    # Show available models
    print("\n" + "=" * 80)
    print("DOSTUPNÉ MODELY")
    print("=" * 80)
    print(f"{'#':<4} {'Model':<25} {'Epochs':<8} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 80)

    model_infos = []
    for idx, run_dir in enumerate(run_dirs, 1):
        info = get_model_info(run_dir)
        model_infos.append(info)

        acc_str = f"{info['accuracy']:.4f}" if isinstance(info['accuracy'], float) else str(info['accuracy'])
        test_acc_str = f"{info['test_acc']:.4f}" if info['test_acc'] is not None else "-"

        print(f"{idx:<4} {info['name']:<25} {info['epochs']:<8} {acc_str:<10} {test_acc_str:<10}")

    print("=" * 80)
    print()

    # Get user choice
    while True:
        try:
            choice = input(f"Vyberte model [1-{len(run_dirs)}] (Enter = nejnovější): ").strip()

            if choice == "":
                # Default: newest model
                return run_dirs[0]

            choice_num = int(choice)
            if 1 <= choice_num <= len(run_dirs):
                selected = run_dirs[choice_num - 1]
                print(f"\nVybrán model: {selected.name}")
                return selected
            else:
                print(f"Chyba: Zadejte číslo mezi 1 a {len(run_dirs)}")

        except ValueError:
            print("Chyba: Zadejte platné číslo nebo Enter")
        except KeyboardInterrupt:
            print("\n\nTestování zrušeno.")
            return None


def main() -> int:
    """Vrací kód procesu: 1, když se nic netestovalo (start_*.bat jinak hlásil úspěch)."""
    args = parse_args()

    # Select model if not specified
    if args.model_dir is None:
        args.model_dir = select_model_interactive(DEFAULT_MODELS_DIR)

        if args.model_dir is None:
            return 1

    # Determine device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Používám zařízení: {device}")

    # Load model
    print(f"Načítám model z: {args.model_dir}")
    model, config = load_model(args.model_dir, device)
    num_classes = config.get("num_classes", 10)

    # Select test data if not specified via CLI
    if args.data_dir is None:
        selected_data_path = select_test_data_interactive(DEFAULT_DATA_BASE)

        if selected_data_path is None:
            return 1
    else:
        # User specified via CLI - use directly
        selected_data_path = args.data_dir
        print(f"\nPoužívám data z: {selected_data_path}")
        warn_full_dataset(selected_data_path)

    # Load test data
    print(f"\nNačítám testovací data...")
    test_dataset = create_test_dataset(selected_data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Nalezeno {len(test_dataset)} testovacích vzorků")
    seen_count, seen_warning = report_seen_samples(args.model_dir, selected_data_path, test_dataset)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    print("\nProvádím evaluaci...")
    avg_loss, accuracy, confusion, predictions = evaluate_with_confusion(
        model, test_loader, criterion, device, num_classes
    )

    # Compute per-class metrics
    per_class_metrics = compute_per_class_metrics(confusion)

    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "model_dir": str(args.model_dir.absolute()),
        "data_dir": str(selected_data_path.absolute()),
        "device": str(device),
        "batch_size": args.batch_size,
        "overall_accuracy": float(accuracy),
        "average_loss": float(avg_loss),
        "total_samples": len(test_dataset),
        "seen_in_training": seen_count,  # None = běh bez data.json (starší model)
        "seen_warning": seen_warning,
        "num_classes": num_classes,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions,
        "model_config": config,
    }

    # Print results
    print_results(results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = DEFAULT_TESTS_DIR / f"test_results_{timestamp}.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_results(results, output_path)

    # Launch visualization
    try:
        from visualize import ResultsVisualizer
        print("\nSpouštím vizualizaci...")
        viz = ResultsVisualizer(output_path)
        viz.show()
        # Vizualizace úspěšně dokončena, skript automaticky skončí
    except ImportError as e:
        print(f"\nVizualizace není dostupná: {e}")
        print("Pro vizualizaci nainstalujte: pip install matplotlib")
        input("\nStiskněte Enter pro ukončení...")
    except Exception as e:
        print(f"\nChyba při spuštění vizualizace: {e}")
        input("\nStiskněte Enter pro ukončení...")
    # Testování proběhlo a výsledky jsou uložené i tehdy, když selže jen vizualizace.
    return 0


if __name__ == "__main__":
    sys.exit(main())
