# shared/

Centrální úložiště sdílených dat a modelů pro projekt DIE-MNIST.

## Struktura

```
shared/
├── data/           # Všechna datová úložiště
│   ├── collected/  # Ručně kreslená data (DigitCollector)
│   ├── synthetic/  # Syntetická data (DigitDreamer)
│   └── composed/   # Sloučený dataset (DigitComposer)
├── models/         # Natrénované modely (DigitLearner)
└── tests/          # Výsledky testování (DigitTester)
```

## Účel

Tato složka odděluje sdílená data a modely od jednotlivých nástrojů. To umožňuje:

- ✅ Snadné zálohování - celá složka `shared/` obsahuje vše důležité
- ✅ Čistou strukturu - nástroje mají vlastní kód, data jsou mimo (a virtuální prostředí mimo projekt)
- ✅ Jednoduchý cleanup - smazání `shared/` vyresetuje všechna data
- ✅ Přehlednost - jasně oddělené "co" od "jak"

## Poznámka

README.md soubory v této struktuře jsou automaticky ignorovány při zpracování dat všemi nástroji (používají glob pattern `*.bmp`).

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
