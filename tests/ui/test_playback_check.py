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
    ok = v({"switches": 2, "qt_defaults": ["A", "B", "A"], "windows_defaults": ["A", "B", "A"]})
    assert "다시 열었습니다 (2회)." == ok[-len("다시 열었습니다 (2회)."):]
    assert "Qt 쪽 문제, 앱이 Windows에 직접 물어 우회" in v({"switches": 2, "qt_defaults": ["A"],
                                                       "windows_defaults": ["A", "B", "A"], "qt_signals": 2})
    assert "알림을 보내지 않아" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A", "B", "A"],
                                  "qt_signals": 0})
    assert "앱 쪽 문제" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A", "B"], "qt_signals": 2})
    assert "앱 쪽 문제" in v({"switches": 0, "qt_defaults": ["A", "B"], "windows_defaults": None, "qt_signals": 1})
    assert "통신 장치" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A"],
                          "windows_other_roles_changed": True})
    said = v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A"],
              "user_report": {"changed": "예", "method": "앱별 출력 설정"}})
    assert "바꿨다고 답했지만" in said and "앱별 출력 설정은 Windows 기본 장치를 바꾸지 않습니다" in said
    assert "바뀌지 않았습니다" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": ["A"], "qt_signals": 0})
    assert "알 수 없습니다" in v({"switches": 0, "qt_defaults": ["A"], "windows_defaults": None, "qt_signals": 0})


def test_windows_lookup_is_none_off_windows():
    from amp.audio import windows_output

    if pc.sys.platform == "win32":
        pytest.skip("Windows에서는 실제 값을 돌려준다")
    assert pc.windows_default_output_id() is None
    assert windows_output.default_output_ids() is None and windows_output.default_output_id() is None


def test_questions_are_skipped_without_console(monkeypatch):
    monkeypatch.setattr(pc.sys, "stdin", None)
    assert pc._ask_user() == {}


def _pactl_ready() -> bool:
    return bool(shutil.which("pactl")) and subprocess.run(["pactl", "info"], capture_output=True).returncode == 0


def test_device_change_window_records_os_qt_and_player(qapp, tmp_path):
    if not _pactl_ready():
        pytest.skip("PulseAudio(pactl)가 없어 가상 출력 장치를 만들 수 없다")
    from PySide6.QtCore import QTimer
    from PySide6.QtMultimedia import QMediaDevices

    def pactl(*a):
        return subprocess.run(["pactl", *a], capture_output=True, text=True, check=True).stdout.strip()

    old = pactl("get-default-sink")
    mods = [pactl("load-module", "module-null-sink", f"sink_name={n}", f"sink_properties=device.description={n}")
            for n in ("amp_pc_a", "amp_pc_b")]
    try:
        pactl("set-default-sink", "amp_pc_a")
        for _ in range(200):  # Qt가 a를 기본으로 볼 때까지
            qapp.processEvents()
            if bytes(QMediaDevices.defaultAudioOutput().id()) == b"amp_pc_a":
                break
            subprocess.run(["sleep", "0.01"])
        # 준비 3초 뒤 점검 창이 열린다. 창이 열리고 1초에 b로, 2.5초에 a로 (사용자가 Windows에서 하는 일)
        QTimer.singleShot(4000, lambda: pactl("set-default-sink", "amp_pc_b"))
        QTimer.singleShot(5500, lambda: pactl("set-default-sink", "amp_pc_a"))
        out, md = pc.run(0, None, 4.0, tmp_path, probe=lambda: {"multimedia": pactl("get-default-sink")},
                         ask=lambda: {"changed": "예", "method": "작업 표시줄 스피커 아이콘", "sound": "새 장치로 옮겨 감"})
    finally:
        for m in mods:
            subprocess.run(["pactl", "unload-module", m])
        if old:
            subprocess.run(["pactl", "set-default-sink", old])
    dc = out["device_change"]
    assert dc["windows_probe"] == "확인함"
    assert dc["windows_defaults"] == ["amp_pc_a", "amp_pc_b", "amp_pc_a"]
    assert dc["qt_signals"] >= 2 and dc["qt_defaults"] == ["amp_pc_a", "amp_pc_b", "amp_pc_a"]
    assert dc["qt_flagged"] == [["amp_pc_a"], ["amp_pc_b"], ["amp_pc_a"]]
    assert dc["switches"] == 2 and dc["state_after"] == "playing"
    assert dc["player_os_lookup"] == "실패 없음"
    assert dc["verdict"] == "새 기본 장치로 재생 출력을 다시 열었습니다 (2회)."
    assert dc["user_report"]["sound"] == "새 장치로 옮겨 감"
    sources = [e["source"] for e in dc["timeline"]]
    assert sources[0] == "start" and "window_open" in sources and sources[-1] in ("window_closed", "playing_on", "player")
    assert {"windows", "qt_signal", "qt_default", "qt_flagged", "player", "playing_on"} <= set(sources)
    playing = [e["device"] for e in dc["timeline"] if e["source"] == "playing_on" and e["phase"] == "점검"]
    assert "amp_pc_b" in playing  # 앱이 실제로 b에서 재생한 순간이 있다
    text = md.read_text(encoding="utf-8")
    assert "판단: 새 기본 장치로" in text and "출력 장치 바꾸기 기록" in text and "사용자 답:" in text
    assert json.loads((md.parent / "playback_check.json").read_text(encoding="utf-8"))["device_change"]["switches"] == 2
    assert out["device_warnings"] == [] and out["player_errors"] == []
