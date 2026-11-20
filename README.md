# DIE (**D**igital **I**dentification **E**xercise) MNIST

Tento projekt obsahuje šest na sebe navazujících aplikací pro sběr, slučování, syntetické generování, učení modelu a testování nad vlastní variantou datasetu ručně psaných číslic.

## Python: verze a instalace

**Potřebuješ Windows a Python 3.11 až 3.15 s Tcl/Tk (`tkinter`). Doporučený je 3.14, na kterém projekt vzniká.** Tkinter není balíček z PyPI, je součástí instalace Pythonu: `requirements.txt` ani `pip` ho nedoplní. Python bez něj se musí přeinstalovat.

Nejjednodušší je Python install manager (příkaz `py`, stáhneš ho z [python.org/downloads/windows](https://www.python.org/downloads/windows/)):

```bat
py install 3.14
python --version
python -c "import tkinter; tkinter.Tk().destroy(); print('tkinter OK')"
```

- **Dávky volají `python` z `PATH`.** `python --version` proto musí ukázat verzi z rozsahu výše. Install manager dá příkaz `python` nejnovější nainstalované verzi, takže když máš víc verzí, ověř, kterou dostaneš.
- **Klasický instalátor z python.org:** nech zapnutou volbu „tcl/tk and IDLE“. Ve výchozím stavu zapnutá je, vypnout se dá jen v „Customize installation“.
- **Nefunguje:** Python 3.10 z install manageru (stahuje se jako NuGet balíček bez Tcl/Tk), embeddable ZIP distribuce a NuGet balíček `python`. Python 3.10 z klasického instalátoru ani verze starší než 3.10 ověřené nejsou.
- **Po výměně Pythonu smaž složku `%TEMP%\DIE-MNIST`.** Dávky si venv založí jednou a dál ho používají s Pythonem, ze kterého vznikl. Venv založený z Pythonu bez Tkinteru zůstane nefunkční, dokud ho nesmažeš.

Ověřeno 20. 11. 2026 na Windows 11 s Pythony z install manageru 26.4. Závislosti se ve všech verzích nainstalovaly z PyPI (PyTorch 2.15.0, u 3.10 2.14.1):

| Python | Tcl/Tk | Výsledek |
|---|---|---|
| 3.10.11 | chybí | okna nejdou. Prošly unit testy (26 testů oken se přeskočilo), DigitDreamer a `train.py`; ostatní nástroje stojí na oknech nebo grafech v Tk |
| 3.11.9 | 8.6.12 | vše prošlo |
| 3.12.10 | 8.6.15 | vše prošlo |
| 3.13.12 | 8.6.15 | vše prošlo |
| 3.14.3 | 8.6.15 | vše prošlo |
| 3.15.2 | 9.0.4 | vše prošlo |

„Vše prošlo“ znamená: unit testy, průchod DigitDreamer → `train.py` → `dispatch.py` → DigitTester s vizualizací a otevření oken DigitCollector, DigitComposer, `marshall.py` a DigitTeaser s načtením modelu. Okna se jen otevřela a zavřela, bez kreslení a klikání, a nástroje běžely přímo z venv, ne přes `.bat`.

## Upozornění

- **Ručně sebraná data nejsou v gitu a nic je nezálohuje.** Složky `shared/data/collected/0`–`9` jsou v `.gitignore`, existují jen na disku. Kdo o ně nechce přijít, ať si je zálohuje sám (ZIP mimo projekt).
- **DigitDreamer umí sebraná data smazat, když ho na ně namíříš.** Přepínač `--clean` a volba „Přepsat“ mažou všechny BMP ve struktuře `<třída>/` cílové složky. Výchozí cíl je `shared/data/synthetic`; s `--output_dir` na `shared/data/collected` zmizí ručně kreslené vzorky bez dalšího ptaní. Do složky se sebranými daty Dreamer nepouštěj.
- **DigitComposer po potvrzení smaže každou složku, která vypadá jako jeho dřívější výstup**, tedy obsahuje jen `train/`, `val/`, `test/` a případně `README.md`. Tak vypadá i `shared/data/synthetic` po `DigitDreamer --split`. Jako výstup potvrzuj předvybranou `shared/data/composed`; složku s jiným obsahem a složku, ve které leží některý zdroj, Composer přepsat odmítne.
- **Composer neumí znovu složit dataset, který už jednou složil.** Soubory z jeho výstupu se jmenují `SSOOOO.bmp` a jako zdroj je odmítne (číslo nad 9999), výsledkem je prázdný dataset a seznam chyb. Skládej vždy z původních zdrojů (`collected`, `synthetic`, ZIPy ze sběru).
- **Poctivý odhad přesnosti dává jen `test` split.** DigitTester ho nabízí jako výchozí volbu (Enter). „Kompletní dataset“, `--data_dir` namířený na složku se splity a plochá složka bez splitů zahrnují i data, na kterých se model učil, a přesnost vyjde nadsazená; Tester na to upozorní. Když dataset `test` split nemá (Composer s testem 0 %), je výchozí volbou kompletní dataset.
- **Kontrola „viděl model tato data při učení?“ stojí na souboru `data.json` v běhu.** Zapisuje ho `train.py`. U běhu bez něj (starší model, ručně složená složka) Tester jen vypíše poznámku a nadsazenou přesnost nepozná. Po přeskládání `composed/` Composerem model znovu natrénuj, jinak část nového `test` splitu při učení viděl.
- **Invertované vzorky** (světlé pozadí, tmavý tah): DigitDreamer jich ve výchozím stavu přidává 15 % (`--invert_prob 0.15`, bez nich `--invert_prob 0`), DigitCollector je ukládá se zatržítkem „Negovat barvy“. DigitTeaser kreslí vždy světle na tmavém.
- **Virtuální prostředí leží v `%TEMP%\DIE-MNIST` a Windows ho může smazat.** Dávky to před spuštěním poznají a založí ho znovu; u nástrojů s PyTorch to znamená znovu stáhnout několik set MB a počkat několik minut. První spuštění potřebuje internet.
- **Python ukládá `__pycache__` do složek nástrojů a do `die_common/`.** V gitu nejsou, ale leží-li projekt v OneDrive, synchronizují se. Zabrání tomu proměnná prostředí `PYTHONPYCACHEPREFIX` nastavená na složku mimo projekt.
- **Testy nepokrývají ovládání oken ani dávky.** Okna (Collector, Composer, Teaser, `marshall.py`, vizualizace Testeru) jsou v testech nahrazená maketami a `.bat` soubory žádný test nespouští. Když se po úpravě chová okno nebo dávka divně, testy to nezachytí.

## Aplikace

### DigitCollector

Desktopová aplikace pro kreslení a ukládání číslic. Automaticky vyvažuje zastoupení tříd - méně zastoupené číslice mají vyšší pravděpodobnost výběru. Data se ukládají do `shared/data/collected/`.

### DigitComposer

Slučuje datasety z více instancí DigitCollector do jednoho s automatickým rozdělením na train/val/test sady. Podporuje import z adresářů i ZIP archivů. Inteligentní přečíslování souborů zachovává trasovatelnost původu každého vzorku. Nastavitelné poměry rozdělení a seed pro reprodukovatelnost.

### DigitDreamer

Generuje syntetické číslice z geometrických tvarů. Výstupní formát kompatibilní s DigitCollector. Umožňuje generovat dataset do podsložek `train/`, `val/`, `test` nebo navázat na existující obsah.

### DigitLearner

Trénuje konvoluční síť **SimpleCNN** (3× Conv2d+BatchNorm+ReLU+MaxPool → 2× Linear, 32×32 vstup, 10 tříd) na datech z `shared/data/composed/`. Pokud datová složka obsahuje podsložky `train/val/test`, použije toto rozdělení; jinak data náhodně rozdělí podle poměrů z konfigurace. Každý běh ukládá do `shared/models/run_<timestamp>/` váhy (`digit_cnn.pt`), použitou konfiguraci (`config.json`), popis dat, na kterých se učil (`data.json`), historii učení (`training_history.json`) a metriky z testu (`test_metrics.json`).

Skládá se ze tří skriptů s „železniční" metaforou:

- **`marshall.py`** – GUI editor hyperparametrů (epochy, batch size, learning rate, poměry val/test, dropout, StepLR step/gamma, seed), ukládá do `config.json`.
- **`train.py`** – samotný trénink (Adam + CrossEntropy + StepLR), ukládá nejlepší model podle validační loss.
- **`dispatch.py`** – po tréninku automaticky vizualizuje architekturu, naučené konvoluční filtry a feature mapy ukázkové číslice (matplotlib okna).

Defaultní průchod `run.bat` spustí `marshall` → `train` → `dispatch`.

### DigitTeaser

Interaktivní GUI aplikace pro real-time vizualizaci klasifikace. Umožňuje kreslit číslice a okamžitě vidět pravděpodobnosti klasifikace zobrazené jako pruhové grafy s vizuálním zvýrazněním nejvyšší hodnoty. Ideální pro "poškádlení" modelu a intuitivní pochopení jeho chování.

### DigitTester

CLI nástroj pro kvantifikované testování natrénovaných modelů s navazující grafickou vizualizací. V konzoli interaktivně nabídne výběr modelu (nebo se zadá přes `--model_dir`) a testovacích dat (`test` split – výchozí a jediný poctivý odhad / kompletní dataset / `train`+`val` pro kontrolu přeučení). Vypočítá detailní metriky (overall accuracy, per-class precision/recall/F1-score, confusion matrix), uloží je jako JSON do `shared/tests/` a automaticky otevře vizualizaci (matplotlib). Klikací confusion matrix umožňuje prozkoumat konkrétní příklady chyb modelu, grafy lze exportovat jako PNG. Vhodný pro seriózní evaluaci a porovnávání modelů.

### (OLD)MNIST – referenční baseline

Samostatná, na zbytek pipeline nenavázaná složka. Klasická „ML Hello World" implementace nad hotovým MNIST datasetem (`mnist.py` stáhne data přes torchvision a natrénuje CNN, `show_data.py` prohlíží dataset, `show_model.py` vizualizuje model). Slouží jako srovnání „použít hotové API" vs. „postavit celou pipeline od nuly" a jako demonstrace problému *dying ReLU* u staré architektury bez BatchNorm. Detaily v `(OLD)MNIST/README.md`.

## Struktura projektu

Projekt používá centrální adresářovou strukturu pro data a modely, což zajišťuje přehlednost a snadnou správu:

```
DIE-MNIST/
├── shared/                   # Centrální úložiště pro sdílená data a modely
│   ├── data/                 # Datové úložiště
│   │   ├── collected/        # ← DigitCollector ukládá sem
│   │   │   ├── 0/
│   │   │   ├── 1/
│   │   │   └── ... (0-9)
│   │   ├── synthetic/        # ← DigitDreamer ukládá sem
│   │   │   ├── train/0-9/
│   │   │   ├── val/0-9/
│   │   │   └── test/0-9/
│   │   └── composed/         # ← DigitComposer ukládá sem
│   │       ├── train/0-9/    # → DigitLearner čte odtud
│   │       ├── val/0-9/
│   │       └── test/0-9/     # → DigitTester čte odtud
│   ├── models/               # Úložiště natrénovaných modelů
│   │   ├── run_20251024_120000/  # ← DigitLearner ukládá sem
│   │   │   ├── config.json
│   │   │   ├── data.json
│   │   │   ├── digit_cnn.pt  # → DigitTeaser/DigitTester čtou odtud
│   │   │   ├── training_history.json
│   │   │   └── test_metrics.json
│   │   └── ...
│   └── tests/                # Výsledky testování
│       ├── test_results_20251024_120000.json  # ← DigitTester ukládá sem
│       └── ...
├── die_common/               # Sdílený kód: cesty, hyperparametry, procházení dat, SimpleCNN
├── tests/                    # Unit testy (python -m unittest discover -s tests)
└── [nástroje; virtuální prostředí jsou mimo projekt v %TEMP%\DIE-MNIST]
```

### Workflow

1. **DigitCollector** → vytváří ručně kreslená data v `shared/data/collected/`
2. **DigitDreamer** → generuje syntetická data v `shared/data/synthetic/`
3. **DigitComposer** → slučuje collected + synthetic → `shared/data/composed/`
4. **DigitLearner** → trénuje na `composed/` → ukládá model do `shared/models/`
5. **DigitTeaser** → načítá model z `shared/models/` pro interaktivní testování
6. **DigitTester** → načítá model z `shared/models/` a data z `composed/test/` pro metriky

**Automatizace:** Kroky 1-4 nebo 2-4 lze automaticky provést jedním příkazem pomocí `start_collect.bat` nebo `start_dream.bat`.

### Výhody

- ✅ **Jasná struktura** - vše na jednom místě
- ✅ **Oddělení nástrojů od dat** - snadné zálohování a cleanup
- ✅ **Automatické cesty** - nástroje fungují "out of the box"
- ✅ **CLI override** - pokročilí uživatelé mohou přepsat defaulty

## Spuštění

Předpoklady: Windows, Python v `PATH` podle [Python: verze a instalace](#python-verze-a-instalace) (3.11 až 3.15 s `tkinter`) a při prvním spuštění internet kvůli instalaci závislostí. Nic dalšího se neinstaluje ručně, virtuální prostředí si dávky připraví samy.

### Automatizované workflow (doporučeno)

Pro kompletní průchod celou pipeline od začátku do konce jsou k dispozici dva workflow skripty:

#### start_collect.bat

Workflow s **ručně kreslenými** daty:

```bash
start_collect.bat
```

Automaticky provede: **DigitCollector** → **DigitComposer** → **DigitLearner** → **DigitTester**

#### start_dream.bat

Workflow se **syntetickými** daty:

```bash
start_dream.bat
```

Automaticky provede: **DigitDreamer** → **DigitComposer** → **DigitLearner** → **DigitTester**

Workflow skripty automaticky řetězí jednotlivé nástroje, přeskakují pauzy mezi kroky a na konci zobrazí vizualizaci výsledků testování. V DigitComposeru stačí při ukládání potvrdit předvybranou složku `shared\data\composed`. Když do ní Composer nic nezapíše, workflow se zastaví, protože DigitLearner by neměl data.

### Ruční spuštění jednotlivých nástrojů

Pro nezávislé spuštění konkrétního nástroje:

- `DigitCollector\run.bat` – sběr ručně psaných číslic
- `DigitComposer\run.bat` – sloučení datasetů
- `DigitDreamer\run.bat [volby]` – generování syntetických dat (např. `run.bat --samples 500 --split`)
- `DigitLearner\run.bat [train|marshall] [volby]` – trénink modelu nebo editor konfigurace
- `DigitTeaser\run.bat` – interaktivní testování modelu kreslením
- `DigitTester\run.bat [volby]` – kvantifikované testování s vizualizací

Každý skript si připraví virtuální prostředí sám. Nástroje sdílejí dvě: `base` (DigitCollector, Composer, Dreamer) a `torch` (DigitLearner, Tester, Teaser), se závislostmi v `die_common\requirements-base.txt` a `requirements-torch.txt`. Leží v dočasné složce uživatele `%TEMP%\DIE-MNIST\venv-<druh>`, ne v projektu. Projekt bývá v OneDrive a venv s PyTorch (stovky MB, tisíce souborů) se do něj synchronizoval a na jiném stroji nefungoval. Windows obsah TEMP občas maže, a to po souborech. Před každým spuštěním proto `die_common\venv_check.py` ověří, že na disku jsou všechny soubory nainstalovaných balíčků (zhruba 4 s u torch). Chybí-li něco, venv se založí znovu, chybí-li jen celý balíček, doinstaluje se. První spuštění torch nástroje stahuje několik set MB a trvá několik minut. `pip install` se jinak spouští jen při změně requirements (otisk SHA256 ve venv). Společnou logiku drží `die_common\ensure_venv.bat`.

Kořenový `requirements.txt` obsahuje závislosti všech nástrojů (sjednocení obou sad) a slouží pro ruční prostředí a testy. Testy se pouštějí z kořene projektu Pythonem, který má nainstalované závislosti: venv mimo projekt, v něm `pip install -r requirements.txt` a `python -m unittest discover -s tests`. Bez PyTorch se testy modelu a tréninku přeskočí, bez `tkinter` testy oken.

> **Poznámka:** Skripty (`run.bat`, `start_collect.bat`, `start_dream.bat`) jsou psané pro **Windows** (dávkové soubory). Hlavní závislosti jsou `torch`, `numpy`, `Pillow`, `tqdm` a `matplotlib` (`torchvision` potřebuje jen referenční `(OLD)MNIST`). Okna (DigitCollector, Composer, Teaser, `marshall.py`, grafy `dispatch.py` a vizualizace Testeru) navíc potřebují `tkinter`, který musí mít samotná instalace Pythonu (viz [Python: verze a instalace](#python-verze-a-instalace)).

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
