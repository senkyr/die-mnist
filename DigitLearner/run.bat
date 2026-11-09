@echo off
setlocal
cd /d "%~dp0"

rem Druh sdileneho virtualniho prostredi (die_common\requirements-torch.txt)
set "VENV_KIND=torch"

set "SKIP_PAUSE=0"
:parse_flags
if /I "%~1"=="--no-pause" (
    set "SKIP_PAUSE=1"
    shift
    goto parse_flags
)

rem Sdileny venv v TEMP a zavislosti (pip jen pri zmene requirements), viz die_common\ensure_venv.bat;
rem ten nastavi VENV_DIR a VENV_PY
rem Relativne: cwd je slozka nastroje (cd /d nahore); %~dp0 by po shift ukazovalo na argument
call "..\die_common\ensure_venv.bat"
if errorlevel 1 goto fail

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto fail

echo(
set "MODE="
set "ACTION=%~1"
if "%ACTION%"=="" goto run_pipeline
if "%ACTION:~0,1%"=="-" goto run_pipeline
if /I "%ACTION%"=="train" goto do_train
if /I "%ACTION%"=="marshall" goto do_marshall

echo Neznamy prikaz: %ACTION%
echo Pouzijte bez parametru pro marshall -^> train, pripadne "train" nebo "marshall" pro jednotlive kroky.
set "APP_EXIT=1"
goto result

:do_train
set "MODE=train"
rem Seznam vsech argumentu shift nemeni, slovo "train" proto odfiltruje :filter_args
rem dispatch.py dostane tytez argumenty jako train.py (shodna --output_dir), ostatni ignoruje
set "SKIP_ACTION=1"
call :filter_args TRAIN_ARGS %*
echo(
echo Spoustim trenink (train.py)...
"%VENV_PY%" train.py %TRAIN_ARGS%
set "APP_EXIT=%ERRORLEVEL%"
if "%APP_EXIT%"=="0" (
    echo(
    echo Vypravuji model ^(vizualizace^)...
    "%VENV_PY%" dispatch.py %TRAIN_ARGS%
    if errorlevel 1 (
        echo VAROVANI: Vizualizace modelu selhala
        echo Pokracuji dale...
    )
)
goto result

:do_marshall
set "MODE=marshall"
echo(
echo Spoustim editor konfigurace (marshall.py)...
"%VENV_PY%" marshall.py
set "APP_EXIT=%ERRORLEVEL%"
goto result

:run_pipeline
set "MODE=pipeline"
call :filter_args TRAIN_ARGS %*
echo(
echo Nejprve oteviram editor konfigurace (marshall.py)...
"%VENV_PY%" marshall.py
if errorlevel 1 (
    echo Editor konfigurace skoncil s chybou nebo byl zavren.
    set "APP_EXIT=%ERRORLEVEL%"
    goto result
)

echo(
echo Spoustim trenink (train.py)...
"%VENV_PY%" train.py %TRAIN_ARGS%
set "APP_EXIT=%ERRORLEVEL%"
if "%APP_EXIT%"=="0" (
    echo(
    echo Vypravuji model ^(vizualizace^)...
    "%VENV_PY%" dispatch.py %TRAIN_ARGS%
    if errorlevel 1 (
        echo VAROVANI: Vizualizace modelu selhala
        echo Pokracuji dale...
    )
)
goto result

:fail
echo(
echo Spusteni se nezdarilo.
set "APP_EXIT=1"
goto result

:result
if "%APP_EXIT%"=="0" goto success
goto failure

:success
if /I "%MODE%"=="marshall" goto msg_marshall
set "RESULT_MSG=Trenink dokoncen. Souhrn je vypsany vyse."
goto after_result

:msg_marshall
set "RESULT_MSG=Editor konfigurace byl ukoncen."
goto after_result

:failure
set "RESULT_MSG=Skript skoncil s chybou (kod %APP_EXIT%)."
goto after_result

:after_result
echo(
echo %RESULT_MSG%

if "%SKIP_PAUSE%"=="0" (
    echo(
    echo Stisknete libovolnou klavesu pro zavreni okna.
    pause >nul
)

:end
endlocal & exit /b %APP_EXIT%

:filter_args
rem Sklada %1 (s uvozovkami, ne %~1): cesta s mezerou musi zustat jednim argumentem.
rem Bez delayed expansion, ta by snedla vykricniky. --no-pause plati na kterekoli pozici.
set "FA_DEST=%~1"
shift
set "FA_OUT="
:filter_loop
if "%~1"=="" goto filter_done
if /I "%~1"=="--no-pause" (
    set "SKIP_PAUSE=1"
    shift
    goto filter_loop
)
if defined SKIP_ACTION (
    set "SKIP_ACTION="
    shift
    goto filter_loop
)
set "FA_OUT=%FA_OUT% %1"
shift
goto filter_loop
:filter_done
set "%FA_DEST%=%FA_OUT%"
set "FA_DEST="
set "FA_OUT="
goto :eof
