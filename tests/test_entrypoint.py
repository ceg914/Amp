import os
import subprocess
import sys

from amp.__main__ import REPO_ROOT, help_message, missing_modules


def test_missing_modules_uses_find_spec():
    assert missing_modules(lambda name: None if name == "PySide6" else object()) == ["PySide6"]
    assert missing_modules(lambda name: object()) == []


def test_help_message_has_exact_install_command(tmp_path):
    msg = help_message(["PySide6"], executable="/x/python", prefix="/x", repo_root=str(tmp_path))
    assert "PySide6" in msg
    assert '"/x/python" -m pip install -r' in msg
    assert str(tmp_path / "requirements.txt") in msg
    assert "켜져 있지 않습니다" not in msg  # .venv가 없으면 가상 환경 안내를 하지 않는다


def test_help_message_detects_inactive_venv(tmp_path):
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    py = tmp_path / ".venv" / sub[0] / sub[1]
    py.parent.mkdir(parents=True)
    py.write_text("")
    msg = help_message(["numpy"], executable="/sys/python", prefix="/sys", repo_root=str(tmp_path))
    assert "켜져 있지 않습니다" in msg
    msg = help_message(["numpy"], executable=str(py), prefix=str(tmp_path / ".venv"), repo_root=str(tmp_path))
    assert "켜져 있지 않습니다" not in msg


def test_python_m_amp_without_packages_prints_help():
    """site-packages 없이(-S) 실행해 패키지가 없는 Python을 흉내 낸다."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-S", "-m", "amp"], cwd=REPO_ROOT, env=env, capture_output=True, timeout=60
    )
    err = proc.stderr.decode("utf-8")
    assert proc.returncode == 1
    assert "Traceback" not in err
    assert "필요한 패키지가 없습니다" in err and "pip install -r" in err


MSYS2_VERSION = "3.12.9 (main, Feb 2025, 10:00:00)  [GCC 14.2.0 64 bit (AMD64)]"
OFFICIAL_VERSION = "3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)]"


def test_interpreter_problem_detects_msys2_and_old_python():
    from amp.__main__ import interpreter_problem

    assert interpreter_problem((3, 12), MSYS2_VERSION, "nt", "win32") is not None
    assert interpreter_problem((3, 13), OFFICIAL_VERSION, "nt", "win32") is None
    assert interpreter_problem((3, 12), "3.12 [GCC 13]", "posix", "linux") is None  # 리눅스 GCC 빌드는 정상
    assert interpreter_problem((3, 12), "3.12 [GCC 13]", "posix", "cygwin") is not None
    assert "3.10" in interpreter_problem((3, 9), OFFICIAL_VERSION, "nt", "win32")


def test_interpreter_help_points_to_py_launcher():
    from amp.__main__ import interpreter_help

    msg = interpreter_help("문제", executable=r"C:\msys64\ucrt64\bin\python.exe")
    assert r"C:\msys64\ucrt64\bin\python.exe" in msg
    assert "py -0p" in msg and "run.bat" in msg and "py -3 -m venv .venv" in msg


def test_check_python_flag():
    proc = subprocess.run([sys.executable, "-m", "amp", "--check-python"], cwd=REPO_ROOT, capture_output=True, timeout=60)
    assert proc.returncode == 0
