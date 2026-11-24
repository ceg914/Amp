@echo off
rem SONATLAS launcher.
rem  1) find an official python.org Python (MSYS2/Cygwin Python cannot install PySide6, so it is skipped)
rem  2) create the virtual environment (.venv) and install packages (first run only)
rem  3) start the program
rem Keep this file ASCII-only. cmd.exe misreads multi-byte characters in batch files
rem (parts of Korean comment lines were executed as commands). Korean messages are in
rem scripts\messages\*.txt (UTF-8) and are printed with "type" after "chcp 65001".
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "MSG=scripts\messages"

if exist "%VENV_PY%" (
    "%VENV_PY%" -m amp --check-python
    if errorlevel 1 goto bad_venv
    goto venv_ready
)
if exist ".venv" goto bad_venv

set "BASE_PY="
py -3 -m amp --check-python >nul 2>&1 && set "BASE_PY=py -3"
if not defined BASE_PY (
    python -m amp --check-python >nul 2>&1 && set "BASE_PY=python"
)
if not defined BASE_PY goto no_python

type "%MSG%\creating_venv.txt"
echo   %BASE_PY%
%BASE_PY% -m venv .venv
if not exist "%VENV_PY%" goto venv_failed

:venv_ready
"%VENV_PY%" -c "import PySide6, numpy, soundfile, scipy, librosa, mido, yaml, guitarpro" >nul 2>&1
if errorlevel 1 (
    type "%MSG%\installing_packages.txt"
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto install_failed
)

"%VENV_PY%" -m amp %*
if errorlevel 1 pause
exit /b 0

:no_python
echo.
type "%MSG%\no_python.txt"
py -0p
pause
exit /b 1

:bad_venv
echo.
type "%MSG%\bad_venv.txt"
pause
exit /b 1

:venv_failed
echo.
type "%MSG%\venv_failed.txt"
pause
exit /b 1

:install_failed
echo.
type "%MSG%\install_failed.txt"
pause
exit /b 1
