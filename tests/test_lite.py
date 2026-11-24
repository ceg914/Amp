"""크롬북·가벼운 모드: 켜는 조건, 설치 안내 문구(운영체제별), 실행 인자, 설치 스크립트 점검."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from amp.app.lite import lite_mode, separation_installed
from amp.core.install_hints import script

ROOT = Path(__file__).resolve().parents[1]


def _spec(found: set):
    return lambda name: object() if name in found else None


def test_lite_mode_switch():
    assert lite_mode({}) is False  # Windows run.bat: 변수 없음 → 기존 그대로
    assert lite_mode({"SONATLAS_LITE": "1"}) is True
    assert lite_mode({"SONATLAS_LITE": "0"}) is False
    assert lite_mode({"SONATLAS_LITE": "auto"}, _spec(set())) is True  # PyTorch 없음
    assert lite_mode({"SONATLAS_LITE": "auto"}, _spec({"torch"})) is False
    assert separation_installed(_spec({"torch", "demucs"})) and not separation_installed(_spec({"torch"}))


def test_install_hints_per_os():
    assert script("separation", "nt") == "install_separation.bat"
    linux = script("separation", "posix")
    assert linux.startswith("./install_chromebook.sh --separation") and "install_separation.bat" in linux
    assert script("run", "posix").startswith("./run.sh")


def test_main_flag_sets_environment(monkeypatch):
    import amp.__main__ as M

    monkeypatch.delenv("SONATLAS_LITE", raising=False)
    monkeypatch.setattr(M, "missing_modules", lambda: ["PySide6"])  # 앱을 띄우지 않고 인자 처리만
    monkeypatch.setattr(M, "help_message", lambda missing: "")
    assert M.run(["--lite"]) == 1
    assert os.environ["SONATLAS_LITE"] == "1"
    M.run(["--full"])
    assert os.environ["SONATLAS_LITE"] == "0"


@pytest.mark.parametrize("name", ["install_chromebook.sh", "run.sh"])
def test_shell_scripts(name):
    path = ROOT / name
    data = path.read_bytes()
    assert b"\r\n" not in data  # 리눅스 셸 스크립트는 LF
    assert os.access(path, os.X_OK)
    subprocess.run(["bash", "-n", str(path)], check=True)  # 문법


def test_install_script_covers_qt_runtime_and_cpu_torch():
    text = (ROOT / "install_chromebook.sh").read_text(encoding="utf-8")
    for pkg in ("python3-venv", "libxcb-cursor0", "libxkbcommon-x11-0", "libegl1", "libpulse0", "libwayland-client0"):
        assert pkg in text
    assert "requirements.txt" in text and "install_separation.py --cpu" in text
    assert "sonatlas.desktop" in text
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "torch" not in req and "demucs" not in req and "whisper" not in req  # 기본 설치는 가볍게
