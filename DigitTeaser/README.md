# DigitTeaser

Real-time interaktivní aplikace pro vizualizaci klasifikační přesnosti natrénovaných modelů z DigitLearner. Umožňuje uživateli "poškádlit" model kreslením číslic v canvasu a okamžitě vidět pravděpodobnosti jejich zařazení do jednotlivých kategorií (0-9).

## Instalace

```bash
pip install -r requirements.txt
```

Pohodlnější je spustit `run.bat`: připraví sdílené virtuální prostředí `%TEMP%\DIE-MNIST\venv-torch` mimo projekt (viz hlavní README) a závislosti do něj instaluje, jen když chybí nebo se změnily. Ruční `pip install -r requirements.txt` instaluje do právě aktivního prostředí.

## Spuštění

```bash
run.bat
```

Otevře se okno s GUI aplikací.

## Jak použít

### 1. Načtení modelu

Klikněte na tlačítko **Načíst model...** a vyberte složku obsahující natrénovaný model z DigitLearner. Složka musí obsahovat:

- `config.json` - konfigurace modelu (včetně dropout parametru)
- `digit_cnn.pt` - uložené váhy modelu

**Inteligentní výchozí adresář:**

- Dialog automaticky otevře složku `../shared/models/` (pokud existuje), kde DigitLearner ukládá natrénované modely

**Příklad struktury modelu:**

```
models/run_20251017_221105/
├── config.json
├── digit_cnn.pt
├── training_history.json
└── test_metrics.json
```

Po úspěšném načtení se nad statistikami zobrazí název modelu a v závorce počet vzorků z `test_metrics.json` (ten DigitLearner zapisuje jen u běhu s testovacími daty).

### 2. Kreslení a klasifikace

- **Kreslete myší** v černém canvasu (256×256 px)
- **Real-time inference** se kontroluje každých 50 ms a přepočítá se jen po změně kresby
- **Pravděpodobnosti** se aktualizují průběžně během kreslení
- **Nejvyšší pravděpodobnost** je zvýrazněna světle modrou barvou

### 3. Vyčištění canvasu

Klikněte na tlačítko **Smazat** pro vymazání kreslení a reset pravděpodobností na rovnoměrné rozdělení (10% pro každou číslici).

## Funkce aplikace

### Real-time inference

- **Frekvence:** kontrola každých 50 ms, výpočet jen po změně kresby nebo modelu
- **Tah:** 15 px s kulatými spoji, stejně jako v DigitCollector (jiná tloušťka = jiná data, než na jakých se model učil)
- **Preprocessing:** Canvas (256×256) → resize na 32×32 → normalizace [0,1] → tensor [1,1,32,32]
- **Model:** SimpleCNN načtený z vybrané složky
- **Output:** Softmax pravděpodobnosti pro 10 tříd (0-9)

### Vizualizace pravděpodobností

- **Pruhové grafy** pro každou číslici (0-9)
- **Procenta** zobrazená za každým grafem
- **Zvýraznění** nejvyšší pravděpodobnosti světle modrou barvou
- **Prázdný canvas** → rovnoměrné rozdělení bez zvýraznění

## Technické detaily

- **GUI:** Tkinter s tmavým barevným schématem konzistentním s ostatními nástroji
- **Model:** SimpleCNN (3 conv layers + 2 fc layers)
- **Device:** CPU inference (dostatečně rychlý pro real-time)
- **Preprocessing:** Stejný jako v DigitLearner (normalizace na [0,1])
- **Canvas:** PIL Image backend pro kreslení + downsampling
- **Detekce prázdného canvasu:** Směrodatná odchylka pravděpodobností < 0.01

## Struktura modelu

Aplikace očekává strukturu složky modelu vytvořenou DigitLearner:

```
models/run_YYYYMMDD_HHMMSS/
├── config.json           # Hyperparametry (včetně dropout)
├── digit_cnn.pt          # State dict modelu
├── training_history.json # Historie trénování
└── test_metrics.json     # Testovací metriky
```

**Důležité:** Aplikace načítá `dropout` parametr z `config.json`, aby správně rekonstruovala architekturu SimpleCNN.

## Příklad použití

1. Natrénujte model v DigitLearner:

   ```bash
   cd ../DigitLearner
   run.bat train --epochs 20
   ```

2. Spusťte DigitTeaser:

   ```bash
   cd ../DigitTeaser
   run.bat
   ```

3. Načtěte model pomocí tlačítka "Načíst model..." a vyberte složku `models/run_*/`

4. Kreslete číslice a sledujte real-time klasifikaci!

## Tipy pro testování

- **Různé styly psaní:** Zkuste kreslit číslice různými způsoby (tlusté/tenké, velké/malé)
- **Částečné kreslení:** Sledujte, jak se pravděpodobnosti mění během kreslení
- **Edge cases:** Zkuste ambivalentní tvary (např. 0 vs O, 1 vs I)
- **Model comparison:** Porovnejte různé natrénované modely na stejných vzorcích

## Kompatibilita

Plně kompatibilní s modely natrénovanými v DigitLearner na datech z DigitCollector nebo DigitDreamer.

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST (Digital Identification Exercise - MNIST), který používá centrální adresářovou strukturu:

- **Modely**: `../shared/models/run_YYYYMMDD_HHMMSS/` (vytvořené pomocí DigitLearner)
- **Smart dialog**: Při kliknutí na "Načíst model..." se automaticky otevře složka `../shared/models/`

Viz hlavní README pro kompletní workflow a strukturu projektu.

## Známá omezení

- **CPU only:** Aplikace používá pouze CPU (GPU inference není nutný pro rychlost)
- **Pevná architektura:** Podporuje pouze SimpleCNN z DigitLearner
- **10 tříd:** Fixně nastaveno pro číslice 0-9

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
