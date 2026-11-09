# data/

Datové úložiště projektu DIE-MNIST. Tento adresář obsahuje všechna data v různých fázích zpracování.

## Struktura

### collected/

Ručně kreslená data nasbíraná pomocí **DigitCollector**. Každá číslice je uložena do podsložky 0-9 ve formátu `XXXX.bmp`.

### synthetic/

Synteticky generovaná data vytvořená pomocí **DigitDreamer**. Lze vytvořit s nebo bez rozdělení na train/val/test sady.

### composed/

Finální sloučený dataset vytvořený pomocí **DigitComposer** (výchozí výstup; složku Composer založí sám, v repu není). Obsahuje:

- `train/` - trénovací data
- `val/` - validační data
- `test/` - testovací data

Tento dataset je použit pro trénování v **DigitLearner**.

## Poznámka

README.md soubory v této struktuře jsou automaticky ignorovány při zpracování dat všemi nástroji.

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
