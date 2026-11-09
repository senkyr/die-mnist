@echo off
rem Sdilena priprava virtualniho prostredi pro nastroje DIE-MNIST.
rem Vola se z run.bat nastroje (call ..\die_common\ensure_venv.bat).
rem Vstup:  VENV_KIND = base (Collector, Composer, Dreamer) nebo torch (Learner, Tester, Teaser).
rem Vystup: VENV_DIR a VENV_PY pro volajici run.bat.
rem
rem Venv lezi v docasne slozce uzivatele (TEMP\DIE-MNIST\venv-<druh>), ne ve slozce
rem projektu: projekt byva v OneDrive a venv s torch (stovky MB, tisice souboru) se do
rem nej synchronizoval a na jinem stroji byl nefunkcni. Nastroje se stejnym druhem
rem sdileji jeden venv, torch se tak instaluje jednou, ne trikrat.
rem
rem Windows obsah TEMP obcas maze, a to po souborech, takze venv muze zmizet jen zcasti.
rem Proto se pred kazdym spustenim overi, ze na disku jsou vsechny soubory balicku
rem (venv_check.py; import to nepozna, numpy bez __init__.py se naimportuje jako prazdny
rem balicek). Kdyz neco chybi nebo venv nefunguje (odinstalovany Python), zalozi se znovu.
rem Zavislosti se instaluji jen pri novem venv nebo zmene requirements-<druh>.txt
rem (otisk SHA256 v VENV_DIR\.requirements.sha256).

set "REQ_FILE=%~dp0requirements-%VENV_KIND%.txt"
set "VENV_CHECK_PY=%~dp0venv_check.py"
if "%VENV_KIND%"=="" goto unknown_kind
if not exist "%REQ_FILE%" goto unknown_kind

set "VENV_DIR=%TEMP%\DIE-MNIST\venv-%VENV_KIND%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_STAMP=%VENV_DIR%\.requirements.sha256"

rem venv_check.py: 0 = v poradku, 1 = poskozeny (zalozit znovu), 2 = chybi balicky (doinstalovat)
if exist "%VENV_PY%" (
    "%VENV_PY%" -I "%VENV_CHECK_PY%" "%REQ_FILE%" >nul 2>&1
    if errorlevel 2 (
        echo Ve virtualnim prostredi chybi balicky, doinstaluji je...
        del /q "%REQ_STAMP%" 2>nul
    ) else if errorlevel 1 (
        echo Virtualni prostredi je neuplne nebo nefunkcni, zakladam ho znovu...
        rmdir /s /q "%VENV_DIR%"
    )
) else if exist "%VENV_DIR%" (
    rem Slozka bez python.exe: zbytek po ciste docasnych souboru
    rmdir /s /q "%VENV_DIR%"
)

if not exist "%VENV_PY%" (
    echo Vytvarim virtualni prostredi v %VENV_DIR% ...
    echo ^(prvni spusteni stahuje zavislosti, u torch to muze trvat nekolik minut^)
    python -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

set "REQ_HASH="
set "OLD_HASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%REQ_FILE%" SHA256 2^>nul') do (
    if not defined REQ_HASH set "REQ_HASH=%%H"
)
if exist "%REQ_STAMP%" set /p OLD_HASH=<"%REQ_STAMP%"

rem Bez otisku (certutil nedostupny) se instaluje vzdy - bezpecnejsi nez nic neinstalovat.
if defined REQ_HASH if "%REQ_HASH%"=="%OLD_HASH%" (
    echo Zavislosti beze zmeny, instalaci preskakuji.
    goto venv_ready
)

echo Instaluji zavislosti z %REQ_FILE% ...
"%VENV_PY%" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install --disable-pip-version-check -r "%REQ_FILE%"
if errorlevel 1 exit /b 1
if defined REQ_HASH (>"%REQ_STAMP%" echo %REQ_HASH%)

rem Kontrola po instalaci: chybejici balicek nebo soubor se ukaze tady, ne az v nastroji.
rem (Bez instalace ji uz udelala kontrola nahore.)
"%VENV_PY%" -I "%VENV_CHECK_PY%" "%REQ_FILE%"
if errorlevel 1 (
    echo Virtualni prostredi se nepodarilo pripravit. Smazte slozku %VENV_DIR% a zkuste to znovu.
    exit /b 1
)

:venv_ready
rem Stara .venv ve slozce nastroje (pred 9. 11. 2026) se uz nepouziva.
if exist ".venv\Scripts\python.exe" echo Poznamka: slozka .venv v %CD% se uz nepouziva, muzete ji smazat.
exit /b 0

:unknown_kind
echo Neznamy druh virtualniho prostredi "%VENV_KIND%" (chybi %REQ_FILE%).
exit /b 1
