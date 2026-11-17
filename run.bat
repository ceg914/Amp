@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem SONATLAS 실행기
rem 1) python.org 공식 Python을 찾는다 (MSYS2·Cygwin Python은 PySide6를 설치할 수 없어 건너뜀)
rem 2) 그 Python으로 가상 환경(.venv)을 만들고 패키지를 설치한다 (처음 한 번)
rem 3) 프로그램을 실행한다

set "VENV_PY=.venv\Scripts\python.exe"

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

echo [Amp] 가상 환경을 만듭니다 (%BASE_PY%)
%BASE_PY% -m venv .venv
if not exist "%VENV_PY%" goto venv_failed

:venv_ready
"%VENV_PY%" -c "import PySide6, numpy, soundfile, scipy, librosa" >nul 2>&1
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
echo [Amp] 사용할 수 있는 Python을 찾지 못했습니다.
echo python.org 에서 받은 Windows용 Python 3.10 이상이 필요합니다.
echo MSYS2(C:\msys64\...)나 Cygwin에 들어 있는 Python으로는 PySide6를 설치할 수 없습니다.
echo.
echo   1. https://www.python.org/downloads/windows/ 에서 Python을 설치합니다.
echo   2. 이 창을 닫고 run.bat 을 다시 실행합니다.
echo.
echo 아래는 찾은 Python 목록입니다 (py 런처가 없으면 오류가 나옵니다).
py -0p
pause
exit /b 1

:bad_venv
echo.
echo [Amp] 기존 .venv 폴더를 쓸 수 없습니다. 지원하지 않는 Python으로 만들어졌거나 손상되었습니다.
echo 아래 명령으로 .venv 폴더를 지운 뒤 run.bat 을 다시 실행하세요.
echo   rmdir /s /q .venv
pause
exit /b 1

:venv_failed
echo.
echo [Amp] 가상 환경을 만들지 못했습니다. 위쪽 오류 메시지를 확인하세요.
pause
exit /b 1

:install_failed
echo.
echo [Amp] 패키지 설치에 실패했습니다. 인터넷 연결과 위쪽 오류 메시지를 확인하세요.
echo 처음부터 다시 하려면 rmdir /s /q .venv 로 .venv 폴더를 지운 뒤 run.bat 을 다시 실행하세요.
pause
exit /b 1
