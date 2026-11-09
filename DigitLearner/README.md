# DigitLearner

Trénovací část projektu pro vlastní alternativu datasetu MNIST. Skript `train.py` načítá obrázky vytvořené nástroji DigitCollector i DigitDreamer, trénuje na nich malou konvoluční síť a ukládá nejlepší model společně s historií učení.

## Jak začít

```bash

pip install -r requirements.txt

```

Pohodlnější je spustit `run.bat`: připraví sdílené virtuální prostředí `%TEMP%\DIE-MNIST\venv-torch` mimo projekt (viz hlavní README) a závislosti do něj instaluje, jen když chybí nebo se změnily. Ruční `pip install -r requirements.txt` instaluje do právě aktivního prostředí. Balíčky: PyTorch, NumPy, Pillow, tqdm, matplotlib.

Spuštění `run.bat` bez parametrů nejprve otevře editor hyperparametrů `marshall.py`. Po jeho zavření (a případném uložení změn do `config.json`) se automaticky spustí `train.py` se stejnými argumenty, které byly předány dávce. Po dokončení tréninku se automaticky spustí vizualizace modelu (`dispatch.py`) a okno zůstane otevřené, dokud nepotvrdíte zprávu. Pokud spouštíte dávku z vlastního terminálu, přidejte jako první argument `--no-pause`.

## Konfigurace

Hyperparametry (počty epoch, velikost batch, tempo učení, dropout…) jsou v souboru `config.json`. Pokud chybí, `train.py` použije vestavěné výchozí hodnoty (soubor sám nevytváří – zapíše ho až uložení v `marshall.py`). Editor uloží jen hodnoty, které trénink přijme: např. u poměrů val/test musí na trénink zbýt aspoň 20 % dat, jinak ukáže chybu a `config.json` nezmění. Cesty se berou relativně ke složce skriptu, takže `train.py` jde spustit odkudkoli. Každý parametr lze přepsat argumentem příkazové řádky (`--epochs`, `--learning_rate`, `--step_size`, `--dropout`, `--seed`, …).

Spuštěním `run.bat marshall` otevřete pouze editor konfigurace (bez následného tréninku); volbou `run.bat train ...` zase spustíte jen trénink a editor přeskočíte.

## Struktura dat

Data se očekávají ve složce `../shared/data/composed/` nebo v cestě zadané parametrem `--data_dir`. Lze použít buď jednoduchou strukturu

```
data/
 ├─0/00001.bmp
 ├─1/00001.bmp
 └─… (až 9/)
```

nebo variantu se splitem (doporučeno):

```
composed/
 ├─train/0/…
 ├─val/0/…
 └─test/0/…
```

V druhém případě se poměry `--val_ratio` a `--test_ratio` ignorují a dataset se použije tak, jak je.

## Spuštění tréninku

```bash

run.bat --epochs 25

```

Pro čistě dávkové spuštění bez editoru použijte `run.bat train --epochs 25`.

Skript nahraje data, vytrénuje model a do složky `../shared/models/run_YYYYMMDD_HHMMSS[_NN]/` uloží níže uvedené soubory. Složku běhu zakládá až po načtení dat, takže chybná `--data_dir` nebo prázdná data běh nezaloží.

- `config.json` – použitou konfiguraci (včetně parametrů z CLI),

- `data.json` – cestu k datům a hashe vzorků po splitech (DigitTester podle nich pozná vzorky, které model viděl),

- `digit_cnn.pt` – váhy modelu z epochy s nejnižší validační chybou (bez validačních dat z poslední epochy),

- `training_history.json` – průběh metrik po epochách,

- `test_metrics.json` – loss a přesnost nejlepšího modelu na test splitu (jen pokud nějaká testovací data jsou).

Síť má vždy 10 výstupů (číslice 0–9). Chybí-li v datech některá třída, trénink proběhne s varováním; složka s jiným názvem než 0–9 je chyba.

Parametr `--use_cpu` vynutí běh na CPU; jinak se automaticky použije GPU, pokud je dostupné.

## Vizualizace modelu

Po úspěšném natrénování se automaticky spustí vizualizace modelu (`dispatch.py`), která zobrazí:

- **Architekturu** - textový summary s počtem parametrů
- **Konvoluční filtry** - naučené váhy všech tří vrstev (32, 64, 128 filtrů)
- **Feature maps** - průchod ukázkové číslice sítí po jednotlivých vrstvách

**Manuální spuštění vizualizace:**

```bash
python dispatch.py                              # Auto-detekce nejnovějšího modelu
python dispatch.py --model_dir PATH             # Specifický model
```

Vizualizace se automaticky volá v rámci `run.bat` workflow, ale můžete ji spustit samostatně kdykoliv později.

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST (Digital Identification Exercise - MNIST), který používá centrální adresářovou strukturu:

**Vstupní data**: `../shared/data/composed/` (vytvořená pomocí DigitComposer)

- train/ - trénovací data
- val/ - validační data
- test/ - testovací data

**Výstupní modely**: `../shared/models/run_YYYYMMDD_HHMMSS/`

- config.json - konfigurace běhu
- digit_cnn.pt - trénovaný model
- training_history.json - metriky z tréninku
- test_metrics.json - výsledky testování (pokud bylo spuštěno)

Tyto modely jsou pak automaticky detekovány v DigitTeaser a DigitTester.

Viz hlavní README pro kompletní workflow a strukturu projektu.

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
