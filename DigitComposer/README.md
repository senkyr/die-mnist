# DigitComposer

Desktopová aplikace v Tkinteru pro slučování datasetů ručně kreslených číslic z různých instancí DigitCollector do jednoho kompaktního tréninkového datasetu s automatickým rozdělením na train/val/test sady. Umožňuje importovat data z adresářů i ZIP archivů, zobrazuje inline statistiky s náhledy vzorků, nastavit poměry splitování a seed pro reprodukovatelnost, a kompiluje finální dataset s inteligentním přečíslováním zachovávajícím trasovatelnost původu každého vzorku.

## Instalace

```bash

pip install -r requirements.txt

```

Pohodlnější je spustit `run.bat`: připraví sdílené virtuální prostředí `%TEMP%\DIE-MNIST\venv-base` mimo projekt (viz hlavní README) a závislosti do něj instaluje, jen když chybí nebo se změnily. Ruční `pip install -r requirements.txt` instaluje do právě aktivního prostředí.

## Spuštění

```bash

run.bat

```

Otevře se okno s GUI aplikací.

Workflow skripty `start_collect.bat` a `start_dream.bat` volají Composer s přepínačem `--pipeline`. V tomto režimu skončí s kódem 1, pokud do `shared/data/composed/` nic nezapsal (okno zavřené bez komponování nebo uložení jinam), a workflow se zastaví dřív, než by DigitLearner spadl na chybějících datech.

## Jak použít

### 1. Přidání zdrojů dat

Kliknutím na tlačítka **Přidat adresář** nebo **Přidat ZIP** vyberte datasety vytvořené v různých instancích DigitCollector.

**Inteligentní výchozí adresáře:**

- Tlačítko **Přidat adresář** automaticky otevře dialog ve složce `../shared/data/` (pokud existuje), kde jsou uložena collected a synthetic data
- Tlačítko **Komponovat a uložit** založí `../shared/data/composed/`, pokud chybí, a otevře dialog přímo v ní – stačí potvrdit

**Podporované struktury:**

- Adresáře:
  - `digits/0/`, `digits/1/`, ... `digits/9/`
  - Nebo přímo `0/`, `1/`, ... `9/` v root

- ZIP archivy (automaticky se rozbalí):
  - `archive.zip → 0/, 1/, 2/, ...` (přímo v archivu)
  - `archive.zip → digits/0/, digits/1/, ...` (v podsložce digits)
  - `archive.zip → jmeno/digits/0/` nebo `jmeno/0/` (vnořená struktura)

Aplikace automaticky detekuje správnou úroveň s adresáři tříd 0-9. Je-li zdroj už rozdělený na `train/`, `val/`, `test/` (např. `DigitDreamer --split`), přidá se každý split jako samostatný zdroj a Composer je znovu promíchá podle svých poměrů.

### 2. Statistiky

Po přidání zdrojů se automaticky zobrazí inline statistiky pro každý zdroj přímo v seznamu:

```
[01] C:\path\to\data  73 vzorků --> | 09 | 09 | 09 | 05 | 04 | 04 | 08 | 15 | 04 | 06 |
```

- **[01]** - ID zdroje
- **C:\path\to\data** - úplná cesta ke zdrojové složce
- **73 vzorků** - celkový počet vzorků ve zdroji
- **| 09 | 09 | ...** - počty vzorků pro každou třídu (0-9)

### 3. Náhled vzorků

Tlačítko **Náhled vzorků** otevře okno s gridem náhodných vzorků z každého zdroje a každé třídy. Pomáhá vizuálně zkontrolovat kvalitu dat před sloučením.

### 4. Nastavení rozdělení datasetu

Použijte slidery pro nastavení poměrů rozdělení:

- **Train** - trénovací sada (výchozí 70%)
- **Val** - validační sada (výchozí 20%)
- **Test** - testovací sada (výchozí 10%)

Aplikace automaticky validuje, že součet poměrů je 100% (✓/⚠ indikátor).

**Seed** - nastavte seed (1-100, výchozí 42) pro reprodukovatelné rozdělení. Se stejným seedem dostanete vždy stejné rozdělení dat.

### 5. Komponování a uložení datasetu

1. Klikněte na **Komponovat a uložit**
2. Potvrďte předvybranou složku `shared/data/composed`, ze které čtou DigitLearner i DigitTester. Jiná složka je možná (ne přímo složka programu), závěrečné okno pak připomene, že Learner a Tester ji potřebují dostat přes `--data_dir`
3. Pokud složka obsahuje dřívější výstup (jen `train/`, `val/`, `test/`, případně `README.md`), aplikace se zeptá na potvrzení a smaže jen tyto splity. Složku s jiným obsahem (např. `shared/data` s `collected/`) odmítne přepsat, stejně jako složku, ve které leží některý ze zdrojů.
4. Dataset se automaticky zkompiluje s rozdělením na train/val/test sady

**Přečíslování souborů:**

Každý soubor dostane nový název ve formátu `SSOOOO.bmp`:

- `SS` = ID zdroje (01-99): nový zdroj dostane nejnižší volné ID, odebraný zdroj své ID uvolní
- `OOOO` = původní 4-ciferné číslo ze zdrojového souboru (0001-9999)

**Příklad:**

```
Zdroj 1 (první přidaný):
  digits/0/0001.bmp → output/train/0/010001.bmp
  digits/0/0042.bmp → output/val/0/010042.bmp

Zdroj 2 (druhý přidaný):
  digits/0/0001.bmp → output/train/0/020001.bmp
  digits/5/1234.bmp → output/test/5/021234.bmp
```

**Logika rozdělení:**

Aplikace používá **Global Split se stratifikací**:

1. Pro každou třídu (0-9) samostatně:
   - Sesbírá všechny vzorky ze všech zdrojů
   - Promíchá je podle nastaveného seedu
   - Rozdělí podle poměrů train/val/test; počty se zaokrouhlují metodou největšího zbytku (`die_common.data.allocate_counts`), takže např. 9 vzorků při 70/20/10 dá 6/2/1
2. Zajišťuje, že každá třída má stejné poměry v každé sadě

Toto schéma zajišťuje:

- **Žádné kolize názvů** mezi zdroji
- **Trasovatelnost** - z názvu poznáte původní zdroj i číslo (díky SSOOOO formátu)
- **Správné řazení** - díky fixní délce čísel
- **Reprodukovatelnost** - stejný seed = stejné rozdělení
- **Stratifikaci** - každá třída má konzistentní zastoupení napříč sadami

### 6. Limity

- **Maximální počet zdrojů:** 99
- **Maximální počet vzorků na zdroj:** 9999

Při překročení limitů se zobrazí chybová hláška.

## Výstup

Uživatelem zvolená složka (výchozí `shared/data/composed/`) s rozdělením na train/val/test:

```
output_folder/
 ├─train/
 │  ├─0/
 │  │  ├─010001.bmp
 │  │  ├─010002.bmp
 │  │  └─…
 │  ├─1/
 │  └─… (až 9/)
 ├─val/
 │  ├─0/
 │  ├─1/
 │  └─… (až 9/)
 └─test/
    ├─0/
    ├─1/
    └─… (až 9/)
```

Výstup je **přímo kompatibilní s DigitLearner**, který automaticky detekuje strukturu train/val/test a použije ji pro trénink modelu.

## Automatické čištění

Dočasné soubory rozbalené ze ZIP archivů se automaticky smažou po:

- Odebrání zdroje ze seznamu
- Zavření aplikace

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST (Digital Identification Exercise - MNIST), který používá centrální adresářovou strukturu:

**Doporučený workflow:**

1. DigitCollector ukládá data do `../shared/data/collected/`
2. DigitDreamer generuje data do `../shared/data/synthetic/`
3. DigitComposer slučuje oboje do `../shared/data/composed/` s rozdělením train/val/test
4. DigitLearner trénuje na `../shared/data/composed/` a ukládá modely do `../shared/models/`

Viz hlavní README pro kompletní workflow a strukturu projektu.

## Technické detaily

- **GUI:** Tmavé barevné schéma konzistentní s DigitCollector, responsivní layout
- **Statistiky:** Inline zobrazení přímo v listboxu se zdrojovými daty
- **ZIP extrakce:** Dočasné adresáře (`tempfile.mkdtemp`)
- **Kopírování:** `shutil.copy2()` pro zachování metadat
- **Split logika:**
  - Global Split se stratifikací (per-class)
  - Použití `random.seed()` pro reprodukovatelnost
  - Validace poměrů s real-time feedbackem (✓/⚠)
  - Automatický výpočet absolutních počtů vzorků
- **Ochrana:** Zabránění uložení přímo do složky programu, do složky s cizími daty nebo do složky obsahující zdroj
- **Validace:**
  - Kontrola neprázdné složky: přepíše jen dřívější výstup Composeru, a to po potvrzení
  - Kontrola součtu poměrů (musí být 100%)
  - Detekce BMP formátu
- **Progress bar:** Zobrazuje průběh po třídách (0-9)

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
