@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem 보컬 분리(Demucs htdemucs)용 PyTorch와 demucs를 .venv에 설치한다.
rem 먼저 run.bat 을 한 번 실행해 .venv를 만들어 두어야 한다.
rem 옵션: --cpu (GPU가 있어도 CPU판), --dry-run (명령만 보기)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [SONATLAS] .venv가 없습니다. run.bat 을 먼저 실행해 주세요.
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
