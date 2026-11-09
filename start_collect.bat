@echo off
setlocal
cd /d "%~dp0"

echo.
echo ========================================================================
echo  DIE-MNIST - Kompletni workflow s rucne kreslenymi daty
echo ========================================================================
echo.
echo Tento skript provede celou pipeline:
echo   1. DigitCollector - Sber rucne kreslenych cislic
echo   2. DigitComposer  - Slouceni a rozdeleni datasetu
echo   3. DigitLearner   - Trenovani neuronove site
echo   4. DigitTester    - Testovani a vizualizace vysledku
echo.
echo ========================================================================
echo.

rem ========================================
rem Krok 1: DigitCollector
rem ========================================
echo.
echo [1/4] Spoustim DigitCollector pro sber dat...
echo.
call DigitCollector\run.bat --no-pause
if errorlevel 1 (
    echo.
    echo CHYBA: DigitCollector selhal.
    goto error
)

echo.
echo ----------------------------------------
echo Krok 1/4 dokoncen: Data byla uspesne sebrana
echo Pokracujeme...
echo ----------------------------------------
echo.
timeout /t 2 >nul

rem ========================================
rem Krok 2: DigitComposer
rem ========================================
echo.
echo [2/4] Spoustim DigitComposer pro sluceni datasetu...
echo.
echo NAPOVEDA PRO COMPOSER:
echo   1. Pridat adresar: Vyberte "shared\data\collected"
echo   2. Komponovat a ulozit: potvrdte predvybranou slozku "shared\data\composed"
echo   3. Pokud composed uz obsahuje dataset, potvrdte prepis
echo   Bez ulozeni do composed se workflow zastavi (Learner by nemel data).
echo.
call DigitComposer\run.bat --no-pause --pipeline
if errorlevel 1 (
    echo.
    echo CHYBA: DigitComposer selhal.
    goto error
)

echo.
echo ----------------------------------------
echo Krok 2/4 dokoncen: Dataset byl uspesne zkompilovan
echo Pokracujeme...
echo ----------------------------------------
echo.
timeout /t 2 >nul

rem ========================================
rem Krok 3: DigitLearner
rem ========================================
echo.
echo [3/4] Spoustim DigitLearner pro trenovani modelu...
echo.
call DigitLearner\run.bat --no-pause
if errorlevel 1 (
    echo.
    echo CHYBA: DigitLearner selhal.
    goto error
)

echo.
echo ----------------------------------------
echo Krok 3/4 dokoncen: Model byl uspesne natrenovan
echo Pokracujeme k testovani...
echo ----------------------------------------
echo.
timeout /t 2 >nul

rem ========================================
rem Krok 4: DigitTester
rem ========================================
echo.
echo [4/4] Spoustim DigitTester pro testovani a vizualizaci...
echo.
echo NAPOVEDA PRO TESTER:
echo   1. Vyber modelu: Stisknete Enter pro vyber nejnovejsiho modelu
echo   2. Vyber dat: Stisknete Enter pro testovani na test splitu (data, ktera model nevidel)
echo   3. Po testovani se automaticky otevre vizualizace
echo.
call DigitTester\run.bat
if errorlevel 1 (
    echo.
    echo CHYBA: DigitTester selhal.
    goto error
)

rem ========================================
rem Uspech
rem ========================================
echo.
echo ========================================================================
echo  WORKFLOW USPESNE DOKONCEN!
echo ========================================================================
echo.
echo Kompletni pipeline byl proveden:
echo   [OK] Sber dat (DigitCollector)
echo   [OK] Slouceni datasetu (DigitComposer)
echo   [OK] Trenovani modelu (DigitLearner)
echo   [OK] Testovani s vizualizaci (DigitTester)
echo.
echo Vysledky naleznete v:
echo   - Modely:       shared\models\
echo   - Test results: shared\tests\test_results_*.json
echo.
echo ========================================================================
endlocal & exit /b 0

:error
echo.
echo ========================================================================
echo  WORKFLOW SELHAL
echo ========================================================================
echo.
echo Zkontrolujte chybove hlasky vyse.
echo Muzete zkusit jednotlive kroky spustit rucne:
echo   - DigitCollector\run.bat
echo   - DigitComposer\run.bat
echo   - DigitLearner\run.bat
echo   - DigitTester\run.bat
echo.
echo ========================================================================
echo.
echo Stisknete libovolnou klavesu pro zavreni okna.
pause >nul
endlocal & exit /b 1
