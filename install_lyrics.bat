@echo off
rem Installs openai-whisper (lyrics recognition) into .venv. Run install_separation.bat first (PyTorch).
rem Options: --dry-run, --japanese (MeCab+UniDic kanji readings, about 250MB),
rem          --dictionaries-only (pronunciation dictionaries only, no whisper, no PyTorch needed)
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
"%VENV_PY%" scripts\install_lyrics.py %*
if errorlevel 1 (
    pause
    exit /b 1
)
pause
exit /b 0
