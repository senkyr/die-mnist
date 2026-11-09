# DigitDreamer

Generátor syntetických číslic složených z jednoduchých geometrických tvarů. Výstupem jsou 32×32 šedotónové BMP soubory s numerickým názvoslovím (0001.bmp, 0002.bmp, ...) kompatibilní s DigitCollector i DigitComposer.

## Instalace

```bash
pip install -r requirements.txt
```

Pohodlnější je spustit `run.bat`: připraví sdílené virtuální prostředí `%TEMP%\DIE-MNIST\venv-base` mimo projekt (viz hlavní README) a závislosti do něj instaluje, jen když chybí nebo se změnily. Ruční `pip install -r requirements.txt` instaluje do právě aktivního prostředí.

## Základní použití

```bash
run.bat --samples 600 --split
```

Pokud není zadáno jinak, dataset se uloží do centrálního úložiště `../shared/data/synthetic/`. Při opakovaném spuštění skript detekuje existující data a nabídne tři možnosti: přepsat, zachovat, nebo doplnit (nové soubory pokračují v číslování tam, kde původní končí). Při doplňování se seed odvodí z `--seed` a počtu existujících obrázků a nové obrázky se porovnají i s těmi uloženými, takže se neopakují. Se stejným seedem by jinak vznikly přesné kopie a Composer by je rozdělil do train i test.

Po dokončení se v konzoli zobrazí souhrn vygenerovaných souborů a okno zůstane otevřené, dokud nestisknete libovolnou klávesu. Pokud spouštíte z již otevřeného terminálu, přidejte jako první argument `--no-pause`, pauza se přeskočí.

## Důležité přepínače

- `--output_dir` – cílová složka (výchozí `../shared/data/synthetic/`).
- `--samples` – počet obrazů na třídu (při `--split` se rozdělí na train/val/test podle `--val_split` a `--test_split`).
- `--noise` – směrodatná odchylka Gaussova šumu.
- `--blur` – maximální poloměr náhodného rozostření.
- `--invert_prob` – pravděpodobnost převrácení popředí a pozadí.
- `--split` – vytvoří podsložky `train/`, `val/`, `test` podle poměrů `--val_split` a `--test_split`.
- `--extra_shapes` – přidá třídu 10 (cik-cak). Zbytek pipeline počítá jen s číslicemi 0–9: DigitComposer ji ignoruje a DigitLearner trénink s ní odmítne.
- `--seed` – seed generátoru (výchozí 1234); stejný seed, parametry a stav cílové složky dají stejná data.
- `--clean` – bez ptaní smaže dřívější obrázky ve struktuře `<třída>/` nebo `<split>/<třída>/` cílové složky.

## Vygenerované tvary

Každá třída (číslice 0-9) odpovídá jinému geometrickému tvaru:

| Třída | Tvar | Anglický název |
|-------|------|----------------|
| 0 | Kruh | circle |
| 1 | Svislá čára | vertical_line |
| 2 | Vodorovná čára | horizontal_line |
| 3 | Úhlopříčka (\) | diagonal_tl_br |
| 4 | Úhlopříčka (/) | diagonal_tr_bl |
| 5 | Čtverec | square |
| 6 | Trojúhelník ▲ | triangle_up |
| 7 | Trojúhelník ▼ | triangle_down |
| 8 | Kosočtverec | diamond |
| 9 | Kříž | cross |

Poloha, tloušťka, šum i další parametry se náhodně mění; pomocí hashů se minimalizují duplicitní obrázky.

## Formát souborů a kompatibilita

Soubory jsou pojmenovány čtyřmístným číslem (0001.bmp až 9999.bmp), což zajišťuje kompatibilitu s DigitComposer. Maximální počet vzorků na třídu je tedy 9999.

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST (Digital Identification Exercise - MNIST), který používá centrální adresářovou strukturu pro snadnou správu dat a modelů. Data se ukládají do `../shared/data/synthetic/` a je možné je následně sloučit s ručně kreslenými vzorky pomocí DigitComposer. Viz hlavní README pro kompletní workflow.

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>
