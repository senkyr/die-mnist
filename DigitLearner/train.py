"""
Jednoduchý trénovací skript pro neuronovou síť nad vlastní verzí datasetu číslic.

Spuštění (z libovolné složky):
    python train.py                                   # data z ../shared/data/composed
    python train.py --data_dir ../shared/data/synthetic --epochs 20
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

# Kořen repozitáře do sys.path kvůli sdílenému balíčku die_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from die_common.config import (  # noqa: E402
    HyperParams,
    load_hyperparams,
    overrides_from_args,
    parse_train_args,
    split_lengths,
)
from die_common.data import (  # noqa: E402
    NUM_CLASSES,
    check_data_dir,
    collect_samples,
    is_split_layout,
    non_empty_splits,
    write_data_info,
)
from die_common.model import SimpleCNN  # noqa: E402
from die_common.paths import COMPOSED_DIR, LEARNER_CONFIG, MODELS_DIR  # noqa: E402

# Zpětná kompatibilita pro skripty, které importovaly tyto názvy z train.py.
DEFAULT_CONFIG_PATH = LEARNER_CONFIG
DEFAULT_MODELS_DIR = MODELS_DIR
DEFAULT_DATA_DIR = COMPOSED_DIR
__all__ = ["HyperParams", "SimpleCNN", "NUM_CLASSES", "DigitDataset"]

@dataclass
class EpochStats:
    """Pomocná struktura pro ukládání průběžných výsledků."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

class DigitDataset(Dataset):
    """Dataset, který čte BMP soubory ze struktury složek 0-9."""

    def __init__(self, root_dir: Path, verify_size: bool = True) -> None:
        self.root_dir = root_dir
        self.verify_size = verify_size
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        self.samples = collect_samples(self.root_dir, strict=True)
        if not self.samples:
            raise RuntimeError("Nenašel jsem žádné BMP soubory. Zkontrolujte cestu k datasetu.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        # BMP načtu přes PIL a okamžitě překlopím do odstínů šedi.
        image = Image.open(image_path).convert("L")

        if self.verify_size and image.size != (32, 32):
            raise ValueError(
                f"Soubor {image_path} má rozměr {image.size}, očekáváno 32x32 px."
            )

        # Z obrázku udělám pole hodnot v rozsahu 0-1 a přidám kanál [1, 32, 32].
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)

        return tensor, torch.tensor(label, dtype=torch.long)

def set_seed(seed: int) -> None:
    """Nastaví seed pro reprodukovatelnost."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_loaders(
    dataset: DigitDataset,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, DataLoader]:
    """Rozdělí dataset na train/val/test a připraví DataLoadery."""
    train_len, val_len, test_len = split_lengths(len(dataset), val_ratio, test_ratio)

    generator = torch.Generator().manual_seed(seed)
    parts = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    loaders = {
        "train": DataLoader(parts[0], batch_size=batch_size, shuffle=True, num_workers=0),
    }
    if val_len > 0:
        loaders["val"] = DataLoader(parts[1], batch_size=batch_size, shuffle=False, num_workers=0)
    if test_len > 0:
        loaders["test"] = DataLoader(parts[2], batch_size=batch_size, shuffle=False, num_workers=0)
    return loaders


def create_loaders_from_subdirectories(
    base_dir: Path,
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Vytvoří DataLoadery přímo ze složek train/val/test."""
    loaders: Dict[str, DataLoader] = {}
    # Prázdný split (např. Composer s val 0 %) přeskočím – DigitDataset by na něm spadl.
    for split in non_empty_splits(base_dir):
        loaders[split] = DataLoader(
            DigitDataset(base_dir / split),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
    if "train" not in loaders:
        raise ValueError(f"Očekávám složku '{base_dir / 'train'}' s daty.")
    return loaders

def samples_by_split(loaders: Dict[str, DataLoader]) -> Dict[str, List[Tuple[Path, int]]]:
    """(cesta, třída) vzorků každého splitu.

    random_split vrací Subset bez atributu .samples – vzorky beru z podkladového
    datasetu podle indexů.
    """
    result: Dict[str, List[Tuple[Path, int]]] = {}
    for split, loader in loaders.items():
        subset = loader.dataset
        source = getattr(subset, "dataset", subset)
        indices = getattr(subset, "indices", range(len(source.samples)))
        result[split] = [source.samples[i] for i in indices]
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Provede jeden trenovaci epoch vcetne sberu statistik."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += inputs.size(0)

        avg_loss = running_loss / total if total else 0.0
        avg_acc = correct / total if total else 0.0
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

    progress.close()

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: Optional[str] = None,
) -> Tuple[float, float]:
    """Spocita chybu a presnost na validacnich nebo testovacich datech."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = None
    iterator = loader
    if desc:
        progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        iterator = progress

    with torch.no_grad():
        for inputs, targets in iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += inputs.size(0)

            if progress is not None:
                avg_loss = running_loss / total if total else 0.0
                avg_acc = correct / total if total else 0.0
                progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

    if progress is not None:
        progress.close()

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy

def snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Nezávislá kopie vah na CPU.

    Samotné `.cpu()` na CPU vrací tentýž tensor, takže by se „nejlepší“ stav
    dál přepisoval dalším tréninkem a uložil by se ve skutečnosti poslední.
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_history(history: List[EpochStats], output_dir: Path) -> None:
    """Uloží historii trénování do JSONu."""
    history_path = output_dir / "training_history.json"
    serialized = [asdict(entry) for entry in history]
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=False)

def main() -> None:
    args = parse_train_args()
    hyperparams, config_path, config_loaded = load_hyperparams(args.config, overrides_from_args(args))

    set_seed(hyperparams.seed)

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    config_state = "loaded" if config_loaded else "defaults used"
    config_status = f"{config_path} ({config_state})"

    # Data kontroluji a načítám dřív, než založím složku běhu: chybná cesta
    # nebo prázdná data dřív nechávala v shared/models prázdné běhy run_*.
    data_root = check_data_dir(args.data_dir)

    use_directory_split = is_split_layout(data_root)
    if use_directory_split:
        loaders = create_loaders_from_subdirectories(data_root, hyperparams.batch_size)
        data_layout_desc = "pre-split directories (train/val/test)"
    else:
        dataset = DigitDataset(data_root)
        loaders = create_loaders(
            dataset,
            batch_size=hyperparams.batch_size,
            val_ratio=hyperparams.val_ratio,
            test_ratio=hyperparams.test_ratio,
            seed=hyperparams.seed,
        )
        data_layout_desc = f"random split (val={hyperparams.val_ratio:.2f}, test={hyperparams.test_ratio:.2f})"

    split_samples = samples_by_split(loaders)
    label_set = {label for samples in split_samples.values() for _, label in samples}
    if not label_set:
        raise ValueError(f"V datové složce '{data_root}' chybí podsložky 0-9 s obrázky.")
    num_classes = NUM_CLASSES
    missing = sorted(set(range(NUM_CLASSES)) - label_set)
    data_summary = {split: len(loader.dataset) for split, loader in loaders.items()}
    train_loader = loaders.get("train")
    if train_loader is None:
        raise ValueError("Nebyla nalezena žádná trénovací data. Zkontrolujte složku s daty.")

    base_dir: Path = args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    config_output_path = run_dir / "config.json"
    run_config = hyperparams.as_dict()
    run_config["num_classes"] = NUM_CLASSES
    with config_output_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    # Na čem se běh učil – Tester podle toho pozná vzorky, které model viděl.
    write_data_info(
        run_dir,
        data_root,
        layout="split" if use_directory_split else "random",
        split_files={split: [path for path, _ in samples] for split, samples in split_samples.items()},
    )

    tqdm.write("")
    tqdm.write("+++ DigitLearner training setup +++")
    tqdm.write("")
    setup_rows = [
        ("Config file", config_status, "--config"),
        ("Run directory", run_dir, "(auto)"),
        ("Data directory", args.data_dir, "--data_dir"),
        ("Models root", base_dir, "--output_dir"),
        ("Hardware target", device, "--use_cpu  force CPU"),
        ("Epochs / batch size", f"{hyperparams.epochs} / {hyperparams.batch_size}", "--epochs / --batch_size"),
        ("Learning rate", f"{hyperparams.learning_rate:g}", "--learning_rate"),
        ("Val/Test split", f"{hyperparams.val_ratio:.2f} / {hyperparams.test_ratio:.2f}", "--val_ratio / --test_ratio"),
        ("Dropout", f"{hyperparams.dropout:.2f}", "--dropout"),
        ("LR step/gamma", f"{hyperparams.step_size} / {hyperparams.lr_gamma}", "--step_size / --lr_gamma"),
        ("Seed", hyperparams.seed, "--seed"),
    ]

    if use_directory_split:
        setup_rows = [
            (label, "from data split" if label == "Val/Test split" else value,
             "(auto)" if label == "Val/Test split" else flag)
            for label, value, flag in setup_rows
        ]

    setup_rows.append(("Data layout", data_layout_desc, "(auto)"))
    for split_name in ("train", "val", "test"):
        if split_name in data_summary:
            setup_rows.append((f"{split_name.capitalize()} samples", data_summary[split_name], "(auto)"))
    setup_rows.append(("Classes", num_classes, "(fixed 0-9)"))
    if missing:
        setup_rows.append(("Missing classes", ", ".join(map(str, missing)), "(warning)"))
    for label, value, flag in setup_rows:
        tqdm.write(f"{label:<20} : {str(value):<25} {flag}")
    tqdm.write("")
    tqdm.write("Progress bar shows processed mini-batches plus running loss/accuracy averages.")
    tqdm.write("")
    tqdm.write("[train]    : batches from the training split.")
    tqdm.write("[val]      : batches from the validation split.")
    tqdm.write("")
    tqdm.write("it/s       : iterations (mini-batches) processed per second.")
    tqdm.write("loss       : running mean cross-entropy over batches processed so far in the epoch.")
    tqdm.write("acc        : running accuracy (fraction correct) over batches processed so far.")
    tqdm.write("")
    tqdm.write("train_loss : mean cross-entropy over processed training samples.")
    tqdm.write("train_acc  : fraction of correct predictions over processed training samples.")
    tqdm.write("val_loss   : mean cross-entropy over processed validation samples.")
    tqdm.write("val_acc    : fraction of correct predictions over processed validation samples.")
    tqdm.write("")

    model = SimpleCNN(num_classes=num_classes, dropout=hyperparams.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    scheduler = StepLR(optimizer, step_size=hyperparams.step_size, gamma=hyperparams.lr_gamma)

    output_dir: Path = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history: List[EpochStats] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")

    for epoch in range(1, hyperparams.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=hyperparams.epochs,
        )

        if "val" in loaders:
            val_desc = f"Epoch {epoch:02d}/{hyperparams.epochs:02d} [val]"
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device, desc=val_desc)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        scheduler.step()

        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )

        tqdm.write(
            f"Epoch {epoch:02d}/{hyperparams.epochs} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = snapshot_state(model)

    if best_state is None:
        # Pokud jsme neměli validační data, uložíme poslední stav.
        best_state = snapshot_state(model)

    model_path = output_dir / "digit_cnn.pt"
    torch.save(best_state, model_path)
    save_history(history, output_dir)

    tqdm.write(f"Natrénovaný model uložen do: {model_path}")
    tqdm.write(f"Artefakty uložené v: {output_dir}")

    test_results: Optional[Tuple[float, float]] = None

    if "test" in loaders and len(loaders["test"].dataset) > 0:
        # Načteme uložené váhy do modelu na aktuálním zařízení a spočítáme výsledky na testu.
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device, desc="Testing")
        tqdm.write(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.3f}")
        test_results = (test_loss, test_acc)

        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "device": str(device),
            "epochs": hyperparams.epochs,
            "batch_size": hyperparams.batch_size,
            "learning_rate": hyperparams.learning_rate,
            # Při rozdělení podle složek train/val/test se poměry z konfigurace nepoužily.
            "data_layout": "split" if use_directory_split else "random",
            "val_ratio": None if use_directory_split else hyperparams.val_ratio,
            "test_ratio": None if use_directory_split else hyperparams.test_ratio,
            "step_size": hyperparams.step_size,
            "lr_gamma": hyperparams.lr_gamma,
            "dropout": hyperparams.dropout,
            "seed": hyperparams.seed,
            "num_classes": num_classes,
            "num_samples": sum(data_summary.values()),
            "split_samples": data_summary,
        }
        metrics_path = output_dir / "test_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        tqdm.write(f"Souhrnné metriky z testu uloženy do: {metrics_path}")

    valid_val_history = [stats for stats in history if not math.isnan(stats.val_loss)]
    if valid_val_history:
        best_epoch_stats = min(valid_val_history, key=lambda stats: stats.val_loss)
    else:
        best_epoch_stats = max(history, key=lambda stats: stats.train_acc)

    final_stats = history[-1]

    summary_lines = [
        "",
        "===== Training summary =====",
        f"Output directory : {output_dir}",
        f"Epochs total     : {hyperparams.epochs}",
        f"Best epoch       : {best_epoch_stats.epoch:02d} "
        f"(train_loss {best_epoch_stats.train_loss:.4f}, train_acc {best_epoch_stats.train_acc:.3f}, "
        f"val_loss {best_epoch_stats.val_loss:.4f}, val_acc {best_epoch_stats.val_acc:.3f})",
        f"Final epoch      : {final_stats.epoch:02d} "
        f"(train_loss {final_stats.train_loss:.4f}, train_acc {final_stats.train_acc:.3f}, "
        f"val_loss {final_stats.val_loss:.4f}, val_acc {final_stats.val_acc:.3f})",
        f"Model path       : {model_path}",
    ]

    if test_results is not None:
        t_loss, t_acc = test_results
        summary_lines.append(f"Test results     : loss {t_loss:.4f}, acc {t_acc:.3f}")

    summary_lines.append("=============================")

    for line in summary_lines:
        tqdm.write(line)

if __name__ == "__main__":
    main()
