@echo off
rem Repeats play/pause/stop/seek/loop/track switch/close/open with the app player and counts Qt audio warnings.
rem Options: --audio PATH  --rounds 3  --device-change 20  (--rounds 0 = device switch check only)
rem Keep this file ASCII-only (see scripts\messages\README.txt).
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo .venv not found. Run run.bat once first.
    pause
    exit /b 1
)
"%VENV_PY%" scripts\playback_check.py %*
pause
exit /b 0
