"""Sdílený kód nástrojů DIE-MNIST.

- paths  – cesty do shared/ (relativně k repozitáři, ne k aktuální složce)
- config – hyperparametry DigitLearneru a parser jeho argumentů
- data   – procházení složek s BMP (třídy 0-9, splity train/val/test)
- model  – SimpleCNN (jediný modul, který potřebuje PyTorch)

Nástroje si kořen repozitáře přidávají do sys.path samy (každý má vlastní
.venv a spouští se ze své složky), proto tu nejsou žádné importy navíc –
`import die_common.paths` nesmí stahovat torch.
"""
