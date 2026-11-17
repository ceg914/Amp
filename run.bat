@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Amp 실행기: 가상 환경(.venv) 만들기 -> 패키지 설치 -> 프로그램 실행
rem 두 번째 실행부터는 설치 단계를 건너뛴다.

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [Amp] 가상 환경을 만듭니다...
    rem python 명령이 없으면 Python 런처(py)로 한 번 더 시도한다
    python -m venv .venv 2>nul || py -3 -m venv .venv
    if not exist "%VENV_PY%" goto no_python
)

"%VENV_PY%" -c "import PySide6, numpy, soundfile" >nul 2>&1
if errorlevel 1 (
    echo [Amp] 필요한 패키지를 설치합니다. 처음 한 번은 몇 분 걸릴 수 있습니다.
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto install_failed
)

"%VENV_PY%" -m amp %*
if errorlevel 1 pause
exit /b 0

:no_python
echo.
echo [Amp] python 명령으로 가상 환경을 만들지 못했습니다.
echo Python 3.10 이상을 설치하고, 설치 화면에서 "Add python.exe to PATH"를 체크했는지 확인하세요.
pause
exit /b 1

:install_failed
echo.
echo [Amp] 패키지 설치에 실패했습니다. 인터넷 연결과 위쪽 오류 메시지를 확인하세요.
echo 설치를 처음부터 다시 하려면 .venv 폴더를 지운 뒤 run.bat 을 다시 실행하세요.
pause
exit /b 1
