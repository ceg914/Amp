@echo off
rem Installs PyTorch and demucs (vocal separation, Demucs htdemucs) into .venv.
rem Run run.bat once first so that .venv exists.
rem Options: --cpu (CPU build even with a GPU), --build cu126 / cu130 / cu132 (pick a GPU build), --dry-run
rem Keep this file ASCII-only (see scripts\messages\README.txt).
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    type "scripts\messages\no_venv_for_separation.txt"
    pause
    exit /b 1
)
"%VENV_PY%" scripts\install_separation.py %*
if errorlevel 1 (
    pause
    exit /b 1
)
pause
exit /b 0
