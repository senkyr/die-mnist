@echo off
setlocal
cd /d "%~dp0"

rem Druh sdileneho virtualniho prostredi (die_common\requirements-base.txt)
set "VENV_KIND=base"

set "SKIP_PAUSE="
if /I "%~1"=="--no-pause" (
    set "SKIP_PAUSE=1"
    shift
)

set "PY_ARGS="
:collect_args
if "%~1"=="" goto args_ready
if defined PY_ARGS (
    set "PY_ARGS=%PY_ARGS% %1"
) else (
    set "PY_ARGS=%1"
)
shift
goto collect_args

:args_ready

rem Sdileny venv v TEMP a zavislosti (pip jen pri zmene requirements), viz die_common\ensure_venv.bat;
rem ten nastavi VENV_DIR a VENV_PY
rem Relativne: cwd je slozka nastroje (cd /d nahore); %~dp0 by po shift ukazovalo na argument
call "..\die_common\ensure_venv.bat"
if errorlevel 1 goto fail

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto fail

echo(
echo Spoustim DigitComposer...
"%VENV_PY%" main.py %PY_ARGS%
set "APP_EXIT=%ERRORLEVEL%"
goto result

:fail
set "APP_EXIT=1"
echo(
echo Spusteni se nezdarilo.
goto result

:result
if "%APP_EXIT%"=="0" goto success

echo(
echo Aplikace skoncila s chybou (kod %APP_EXIT%).
goto status_done

:success
echo(
echo Slucovani datasetu bylo dokonceno.

:status_done

if defined SKIP_PAUSE goto end

echo(
echo Stisknete libovolnou klavesu pro zavreni okna.
pause >nul

:end
endlocal & exit /b %APP_EXIT%
