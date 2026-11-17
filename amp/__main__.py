"""``python -m amp`` 진입점.

필요한 패키지가 없으면 트레이스백 대신 해결 방법을 한국어로 알려 준다.
이 파일은 표준 라이브러리만 import해야 한다 (패키지가 없어도 메시지를 띄워야 하므로).
"""

from __future__ import annotations

import importlib.util
import os
import sys

MIN_PYTHON = (3, 10)
REQUIRED_MODULES = ("PySide6", "numpy", "soundfile")
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
        f"[Amp] 실행에 필요한 패키지가 없습니다: {', '.join(missing)}",
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


def run() -> int:
    if sys.version_info < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        print(f"[Amp] Python {need} 이상이 필요합니다. 지금 버전: {sys.version.split()[0]}", file=sys.stderr)
        return 1
    missing = missing_modules()
    if missing:
        print(help_message(missing), file=sys.stderr)
        return 1
    from amp.app.main import main

    return main()


if __name__ == "__main__":
    sys.exit(run())
