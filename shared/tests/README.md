# tests/

Centrální úložiště výsledků testování pro projekt DIE-MNIST.

## Struktura

Každý testovací běh vytvoří JSON soubor s časovým razítkem:

```
tests/
├── test_results_20251024_120000.json
├── test_results_20251024_143000.json
└── test_results_20251025_091500.json
```

## Formát souborů

- **Název**: `test_results_YYYYMMDD_HHMMSS.json`
  - `YYYYMMDD` = datum
  - `HHMMSS` = čas

## Obsah JSON souboru

Každý soubor obsahuje kompletní výsledky testování:

```json
{
  "timestamp": "20251024_120000",
  "model_dir": "/path/to/model",
  "data_dir": "/path/to/test/data",
  "device": "cuda",
  "batch_size": 64,
  "overall_accuracy": 0.9523,
  "average_loss": 0.1847,
  "total_samples": 1000,
  "seen_in_training": 0,
  "seen_warning": false,
  "num_classes": 10,
  "per_class_metrics": {
    "0": {
      "precision": 0.96,
      "recall": 0.94,
      "f1_score": 0.95,
      "accuracy": 0.94,
      "samples": 100
    },
    ...
  },
  "confusion_matrix": [[94, 1, 0, ...], ...],
  "model_config": {...}
}
```

## Workflow

1. **DigitTester** spustí testování modelu na testovacích datech
2. Vypočítá metriky (accuracy, precision, recall, F1, confusion matrix)
3. Uloží výsledky jako JSON do této složky
4. Zobrazí interaktivní vizualizaci výsledků

## Použití výsledků

- **Porovnání modelů**: Srovnejte JSON soubory z různých běhů
- **Tracking výkonu**: Sledujte vývoj metrik napříč experimenty
- **Reprodukovatelnost**: JSON obsahuje kompletní konfiguraci testu
- **Vizualizace**: DigitTester načte JSON a zobrazí interaktivní grafy

## Poznámka

README.md soubory v této struktuře jsou automaticky ignorovány při zpracování dat.

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
