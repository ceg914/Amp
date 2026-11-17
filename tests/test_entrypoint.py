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
