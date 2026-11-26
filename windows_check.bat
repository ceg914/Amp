@echo off
rem SONATLAS v0.7.0 check: installed models, GPU, profiles. Writes sonatlas_windows_check_*.txt/json here.
rem Options: --hash (checkpoint sha256, slow for big files)  --no-probe
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
"%VENV_PY%" scripts\windows_check.py %*
pause
exit /b 0
