"""재생 점검 도구의 출력 장치 바꾸기 진단 (STEP 3.5 후속).

Windows 실행 결과가 '다시 연 횟수 0'만으로는 장치를 안 바꾼 것인지, Qt가 알리지 않은 것인지, 앱이 무시한 것인지 알 수 없었다.
도구가 운영체제에 기본 장치를 직접 묻고 시간순으로 적는지 본다. Linux에서는 pactl을 운영체제 조회 대신 쓴다.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from amp.devtools import playback_check as pc


def test_verdict_tells_apart_no_change_qt_silence_and_app_ignoring():
    v = pc.device_change_verdict
    assert "다시 열었습니다 (2회)" in v({"switches": 2, "qt_defaults": ["A", "B", "A"], "windows_defaults": ["A", "B", "A"]})
    assert "앱 쪽 문제" in v({"switches": 0, "qt_defaults": ["A", "B"], "windows_defaults": ["A", "B"], "qt_signals": 1})
    assert "알림을 보내지 않았습니다" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A", "B", "A"],
                                       "qt_signals": 0})
    assert "그대로였습니다 (Qt 쪽 문제)" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A", "B"],
                                          "qt_signals": 2})
    assert "바뀌지 않았습니다" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A"], "qt_signals": 0})
    assert "알 수 없습니다" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": None, "qt_signals": 0})


def test_windows_probe_is_none_off_windows():
    if pc.sys.platform == "win32":
        pytest.skip("Windows에서는 실제 값을 돌려준다")
    assert pc.windows_default_output_id() is None


def _pactl_ready() -> bool:
    return bool(shutil.which("pactl")) and subprocess.run(["pactl", "info"], capture_output=True).returncode == 0


def test_device_change_window_records_os_qt_and_player(qapp, tmp_path):
    if not _pactl_ready():
        pytest.skip("PulseAudio(pactl)가 없어 가상 출력 장치를 만들 수 없다")
    from PySide6.QtCore import QTimer

    def pactl(*a):
        return subprocess.run(["pactl", *a], capture_output=True, text=True, check=True).stdout.strip()

    old = pactl("get-default-sink")
    mods = [pactl("load-module", "module-null-sink", f"sink_name={n}", f"sink_properties=device.description={n}")
            for n in ("amp_pc_a", "amp_pc_b")]
    try:
        pactl("set-default-sink", "amp_pc_a")
        # Qt가 a를 기본으로 볼 때까지 기다린다
        from PySide6.QtMultimedia import QMediaDevices

        for _ in range(200):
            qapp.processEvents()
            if bytes(QMediaDevices.defaultAudioOutput().id()) == b"amp_pc_a":
                break
            subprocess.run(["sleep", "0.01"])
        # 창이 열린 뒤 1초에 b로 바꾸고 2초에 a로 되돌린다 (사용자가 Windows 소리 설정에서 하는 일)
        QTimer.singleShot(1000, lambda: pactl("set-default-sink", "amp_pc_b"))
        QTimer.singleShot(2000, lambda: pactl("set-default-sink", "amp_pc_a"))
        out, md = pc.run(0, None, 3.0, tmp_path, probe=lambda: pactl("get-default-sink"))
    finally:
        for m in mods:
            subprocess.run(["pactl", "unload-module", m])
        if old:
            subprocess.run(["pactl", "set-default-sink", old])
    dc = out["device_change"]
    assert dc["windows_probe"] == "확인함"
    assert dc["windows_defaults"] == ["amp_pc_a", "amp_pc_b", "amp_pc_a"]
    assert dc["qt_signals"] >= 2 and dc["qt_defaults"] == ["amp_pc_a", "amp_pc_b", "amp_pc_a"]
    assert dc["switches"] == 2 and dc["state_after"] == "playing"
    assert "다시 열었습니다 (2회)" in dc["verdict"]
    sources = [e["source"] for e in dc["timeline"]]
    assert sources[0] == "start" and {"windows", "qt_signal", "qt_default", "player"} <= set(sources)
    text = md.read_text(encoding="utf-8")
    assert "판단: 새 기본 장치로" in text and "출력 장치 바꾸기 기록" in text
    assert json.loads((md.parent / "playback_check.json").read_text(encoding="utf-8"))["device_change"]["switches"] == 2
    assert out["device_warnings"] == [] and out["player_errors"] == []
