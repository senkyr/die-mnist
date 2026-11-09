# models/

Centrální úložiště natrénovaných modelů pro projekt DIE-MNIST.

## Struktura

Každý tréninkový běh vytvoří vlastní složku s časovým razítkem:

```
models/
├── run_20251024_120000/
│   ├── config.json          # Konfigurace tréninku
│   ├── data.json            # Na jakých datech se běh učil
│   ├── digit_cnn.pt         # Váhy modelu (SimpleCNN)
│   ├── training_history.json  # Historie metrik během tréninku
│   └── test_metrics.json    # Výsledky testování (pokud bylo spuštěno)
├── run_20251024_143000/
│   └── ...
└── run_20251025_091500/
    └── ...
```

## Formát složek

- **Název**: `run_YYYYMMDD_HHMMSS[_NN]`
  - `YYYYMMDD` = datum
  - `HHMMSS` = čas
  - `_NN` = volitelná sekvence při kolizi (00, 01, 02...)

## Soubory v modelu

### config.json

Hyperparametry použité při tréninku (epochs, batch_size, learning_rate, val/test ratio, step_size, lr_gamma, dropout, seed) a `num_classes`. Cestu k datům nese `data.json`.

### data.json

Na jakých datech se běh učil: `data_dir` (absolutní cesta), `layout` (`split` = složky train/val/test, `random` = náhodné rozdělení ploché složky) a pro každý split počet vzorků a MD5 hashe jejich obsahu. DigitTester podle hashů train a val pozná vyhodnocované vzorky, které model viděl při učení (např. po přeskládání `composed/` až po tréninku), a varuje. `dispatch.py` z `data_dir` bere ukázkovou číslici. Starší běhy soubor nemají.

### digit_cnn.pt

PyTorch state dict nejlepšího modelu (podle validační loss). Architektura SimpleCNN:

- 3 konvoluční vrstvy
- 2 plně propojené vrstvy
- Dropout pro regularizaci

### training_history.json

Historie metrik pro každou epochu:

- Train loss/accuracy
- Val loss/accuracy
- Learning rate

### test_metrics.json (volitelný)

Detailní metriky z testování:

- Overall accuracy
- Per-class precision/recall/F1-score
- Confusion matrix

## Workflow

1. **DigitLearner** trénuje model a ukládá ho sem
2. **DigitTeaser** načítá model pro interaktivní testování
3. **DigitTester** načítá model pro kvantifikované testování

## Automatická detekce

- **DigitTester**: Interaktivně nabídne seznam modelů k výběru (default: nejnovější)
- **DigitTeaser**: Dialog se otevře přímo v této složce

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
