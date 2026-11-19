@echo off
rem Checks lyrics recognition (Whisper) and alignment (MMS) on the NVIDIA GPU and writes a report.
rem Options: --audio PATH  --models large-v3,turbo  --seconds 60  --no-cancel  --no-mms
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
"%VENV_PY%" scripts\gpu_check.py %*
pause
exit /b 0
