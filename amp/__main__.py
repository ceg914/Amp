"""``python -m amp`` 진입점.

필요한 패키지가 없으면 트레이스백 대신 해결 방법을 한국어로 알려 준다.
이 파일은 표준 라이브러리만 import해야 한다 (패키지가 없어도 메시지를 띄워야 하므로).

``python -m amp --check-python`` 은 이 Python으로 SONATLAS를 설치·실행할 수 있는지만 검사한다.
run.bat이 공식 Python을 고를 때 쓴다 (종료 코드 0 = 사용 가능).
"""

from __future__ import annotations

import importlib.util
import os
import sys

MIN_PYTHON = (3, 10)
REQUIRED_MODULES = ("PySide6", "numpy", "soundfile", "scipy", "librosa", "mido", "yaml")  # yaml = PyYAML
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def missing_modules(find_spec=importlib.util.find_spec) -> list[str]:
    return [name for name in REQUIRED_MODULES if find_spec(name) is None]


def _venv_dir(repo_root: str) -> str | None:
    """저장소 폴더에 만든 가상 환경 경로. 없으면 None."""
    venv = os.path.join(repo_root, ".venv")
    py = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
    return venv if os.path.exists(py) else None


def help_message(
    missing: list[str],
    executable: str = sys.executable,
    prefix: str = sys.prefix,
    repo_root: str = REPO_ROOT,
) -> str:
    req = os.path.join(repo_root, "requirements.txt")
    lines = [
        f"[SONATLAS] 실행에 필요한 패키지가 없습니다: {', '.join(missing)}",
        f"지금 실행한 Python: {executable}",
        "",
    ]
    venv = _venv_dir(repo_root)
    if venv and os.path.normcase(os.path.abspath(prefix)) != os.path.normcase(os.path.abspath(venv)):
        activate = r".venv\Scripts\activate.bat" if os.name == "nt" else "source .venv/bin/activate"
        lines += [
            "이 폴더에 가상 환경(.venv)이 있지만 지금은 켜져 있지 않습니다.",
            "가상 환경에 설치했다면 먼저 켠 뒤 다시 실행하세요.",
            f"    {activate}",
            "    python -m amp",
            "",
        ]
    if os.name == "nt":
        lines += [
            "가장 쉬운 방법: 저장소 폴더의 run.bat 을 실행하세요.",
            "가상 환경을 만들고 패키지를 설치한 뒤 프로그램을 켭니다.",
            "",
        ]
    lines += [
        "지금 쓰는 Python에 직접 설치하려면 아래 명령을 실행한 뒤 다시 시작하세요.",
        f'    "{executable}" -m pip install -r "{req}"',
    ]
    return "\n".join(lines)


def interpreter_problem(
    version_info=sys.version_info,
    version: str = sys.version,
    os_name: str = os.name,
    platform: str = sys.platform,
) -> str | None:
    """이 Python으로 SONATLAS를 설치할 수 없는 이유. 문제가 없으면 None."""
    if tuple(version_info[:2]) < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        return f"Python {need} 이상이 필요합니다 (지금 {version_info[0]}.{version_info[1]})."
    # python.org·Microsoft Store Python은 MSVC로 빌드된다. MSYS2(GCC/Clang)·Cygwin 빌드에는
    # PyPI의 PySide6 휠이 맞지 않아 pip으로 설치할 수 없다.
    if platform == "cygwin" or (os_name == "nt" and "MSC v." not in version):
        return "MSYS2·Cygwin용 Python이라 pip으로 PySide6를 설치할 수 없습니다 (PyPI에 이 Python용 설치 파일이 없음)."
    return None


def interpreter_help(problem: str, executable: str = sys.executable) -> str:
    lines = [
        "[SONATLAS] 지금 실행한 Python으로는 SONATLAS를 설치할 수 없습니다.",
        f"    {executable}",
        problem,
        "",
        "python.org에서 받은 Windows용 Python 3.10 이상이 필요합니다.",
        "1) 이미 설치되어 있는지 확인:  py -0p",
        "   목록에 경로가 나오면 설치된 것입니다. 'py'를 찾을 수 없다고 나오면 2)로 가세요.",
        "2) https://www.python.org/downloads/windows/ 에서 설치 (설치 화면 기본값 그대로 두면 py 런처도 함께 설치됨)",
        "3) 저장소 폴더에서 run.bat 실행. 공식 Python을 찾아 가상 환경을 만들고 패키지를 설치합니다.",
        "",
        "'python' 명령은 PATH 순서 때문에 계속 MSYS2 쪽을 가리킬 수 있습니다. 직접 설치할 때는 py를 쓰세요.",
        "    py -3 -m venv .venv",
    ]
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    problem = interpreter_problem()
    if argv[:1] == ["--check-python"]:
        if problem:
            print(f"{sys.executable}: {problem}", file=sys.stderr)
        return 1 if problem else 0
    if sys.version_info < MIN_PYTHON:
        print(f"[SONATLAS] {problem}", file=sys.stderr)
        return 1
    for flag, value in (("--lite", "1"), ("--full", "0")):  # 가벼운 모드 (amp/app/lite.py)
        if flag in argv:
            os.environ["SONATLAS_LITE"] = value
            argv = [a for a in argv if a != flag]
            sys.argv = [sys.argv[0], *argv]
    missing = missing_modules()
    if missing:
        # 패키지가 없을 때만 빌드 종류를 따진다 (다른 경로로 이미 설치했다면 막지 않는다)
        print(interpreter_help(problem) if problem else help_message(missing), file=sys.stderr)
        return 1
    from amp.app.main import main

    return main()


if __name__ == "__main__":
    sys.exit(run())
