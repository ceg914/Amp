"""STEP 5 청취 도구 화면 시험: 반복 구간, 음높이 유지 느린 재생, 음표 미리 듣기, 음표 소리 트랙, 스펙트로그램 레인.

재생 시험은 실제 출력 장치(PulseAudio 가상 장치 포함)가 있어야 돈다.
"""

import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest

from amp.app.player import AudioPlayer, PlaybackState
from amp.audio.synth import midi_to_hz
from amp.core.model import AnalysisResult
from amp.core.tempo import TempoMap
from tests.conftest import note

needs_device = pytest.mark.skipif(not AudioPlayer.output_available(), reason="오디오 출력 장치 없음")


def create(qtbot, w, path):
    w.create_project_from_audio(path, None, "copy")
    qtbot.waitUntil(lambda: w.session.audio is not None, timeout=20_000)
    return w.session.folder


def _f0(y, sr=44100):
    n = min(len(y), 8192)
    sp = np.abs(np.fft.rfft(y[:n] * np.hanning(n), 8 * n))
    return np.argmax(sp) * sr / (8 * n)


def test_waveform_drag_selects_loop_region_and_marks_views(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    w.resize(1400, 900)
    qtbot.waitUntil(lambda: w.waveform.width() > 300, timeout=5000)
    wave = w.waveform
    tl = w.timeline
    x0, x1 = tl.time_to_x(1.0), tl.time_to_x(2.0)
    y = wave.height() // 2
    QTest.mousePress(wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x0), y))
    QTest.mouseMove(wave, QPoint(int((x0 + x1) / 2), y))
    QTest.mouseMove(wave, QPoint(int(x1), y))
    QTest.mouseRelease(wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x1), y))
    a, b = w.player.loop_region
    assert a == pytest.approx(1.0, abs=2 / tl.px_per_sec) and b == pytest.approx(2.0, abs=2 / tl.px_per_sec)
    assert w.player.loop_enabled and w.listening.act_loop.isChecked()
    assert wave.region == (a, b) and w.view.roll.loop_region == (a, b)
    assert w.listening.start_spin.value() == pytest.approx(a, abs=0.01)
    # 짧게 누르기만 하면 기존처럼 이동만 한다
    QTest.mouseClick(wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(tl.time_to_x(3.0)), y))
    assert w.player.loop_region == (a, b)
    # 시작·끝 직접 입력
    w.listening.start_spin.setValue(0.5)
    w.listening.end_spin.setValue(1.25)
    w.listening._on_spin_edited()
    assert w.player.loop_region == pytest.approx((0.5, 1.25))
    w.listening.act_loop.trigger()  # 끄기
    assert not w.player.loop_enabled and w.view.roll.loop_region is None
    assert w.errors == []


def test_shortcuts_do_not_conflict(window):
    seen = {}
    for a in window.findChildren(QAction):
        for ks in a.shortcuts():
            key = ks.toString()
            if key:
                assert key not in seen, f"단축키 {key} 겹침: {seen.get(key)} / {a.text()}"
                seen[key] = a.text()
    for key in ("Alt+L", "Alt+[", "Alt+]", "Alt+,", "Alt+.", "Alt+/", "Alt+P", "Alt+M", "Ctrl+Shift+R"):
        assert key in seen


@needs_device
def test_loop_playback_stays_in_region_and_stop_returns(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    w.listening.set_region(1.0, 1.5, enable=True)
    w.seek(1.0)
    w.player.play()
    qtbot.waitUntil(lambda: w.player.state is PlaybackState.PLAYING, timeout=3000)
    # 이 컨테이너의 PulseAudio 가상 출력은 소리가 나가기 시작하는 데 1.7~1.9초 걸린다(예전 코드도 같음). 움직이기 시작한 뒤 잰다
    qtbot.waitUntil(lambda: w.player.position() > 1.05, timeout=6000)
    seen = []
    t_end = time.monotonic() + 1.4  # 구간(0.5초)을 두 번 넘게 돈다
    while time.monotonic() < t_end:
        qtbot.wait(30)
        seen.append(w.player.position())
    assert min(seen) >= 1.0 - 1e-6 and max(seen) < 1.5
    assert any(b < a for a, b in zip(seen, seen[1:]))  # 되돌아간 적이 있다
    w.player.stop()
    assert w.player.state is PlaybackState.STOPPED and w.player.position() == pytest.approx(1.0)
    # 반복 중 구간 밖으로 이동하면 구간 시작으로
    w.seek(3.0)
    assert w.player.position() == pytest.approx(1.0)
    assert w.errors == []


@needs_device
def test_slow_playback_keeps_pitch_and_reports_original_time(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    w.listening.act_slower.trigger()
    assert w.player.speed == 0.9
    w.listening.set_speed(0.5)
    assert w.listening.speed_combo.currentData() == 0.5
    w.seek(1.0)
    w.player.play()  # 늘린 오디오를 만든 뒤 시작한다
    qtbot.waitUntil(lambda: w.player.state is PlaybackState.PLAYING, timeout=20_000)
    assert w.listening.cache.misses >= 1
    qtbot.waitUntil(lambda: w.player.position() > 1.02, timeout=6000)  # 출력이 움직이기 시작한 뒤
    p0, t0 = w.player.position(), time.monotonic()
    qtbot.wait(1000)
    rate = (w.player.position() - p0) / (time.monotonic() - t0)
    assert rate == pytest.approx(0.5, abs=0.12)  # 화면 위치 = 원본 시간
    # 반복 + 느린 재생
    w.listening.set_region(1.0, 1.6, enable=True)
    qtbot.waitUntil(lambda: w.player.state is PlaybackState.PLAYING and not w.player.waiting_for_stretch, timeout=20_000)
    qtbot.wait(2500)
    seen = []
    for _ in range(40):
        qtbot.wait(40)
        seen.append(w.player.position())
    assert min(seen) >= 1.0 - 1e-6 and max(seen) < 1.6 + 1e-6
    # 정상 속도로 곧바로
    w.listening.act_normal.trigger()
    assert w.player.speed == 1.0 and w.player.state is PlaybackState.PLAYING
    w.player.stop()
    assert w.errors == []


def _setup_notes(w):
    notes = (note("a", 1.0, 1.5, 60), note("b", 1.6, 2.0, 64), note("c", 2.1, 2.6, 67))
    w.doc.set_analysis(AnalysisResult(10.0, TempoMap.constant(120), notes))
    return notes


def _note_point(w, n):
    m = w.view.mapper
    sp = QPointF(m.time_to_x((n.start + n.end) / 2), m.pitch_to_y(n.pitch) + m.row_height / 2)
    return w.view.mapFromScene(sp)


def test_note_click_drag_and_sequence_preview(qtbot, window, song, monkeypatch):
    w = window
    create(qtbot, w, song)
    notes = _setup_notes(w)
    w.resize(1400, 900)
    w.view.center_on_pitch(64)
    qtbot.wait(50)
    pv = w.listening.preview
    played = []
    monkeypatch.setattr(pv, "_ensure_sink", lambda: (played.append(1), False)[1])  # 장치 없이도 소리 내용은 확인
    vp = w.view.viewport()
    pt = _note_point(w, notes[1])
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    assert pv.triggers == 1 and abs(_f0(pv.last_sound) - midi_to_hz(64)) < 3
    up = QPoint(pt.x(), int(pt.y() - 2 * w.view.mapper.row_height))
    QTest.mouseMove(vp, QPoint(pt.x(), pt.y() - 3))
    QTest.mouseMove(vp, up)
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, up)
    assert pv.triggers == 2 and abs(_f0(pv.last_sound) - midi_to_hz(66)) < 3
    assert w.doc.score.get("b").pitch == 66
    # 방향키로 옮겨도 들려준다
    QTest.keyClick(w.view, Qt.Key.Key_Down)
    assert pv.triggers == 3 and w.doc.score.get("b").pitch == 65
    # 여러 음표를 원래 간격대로
    w.view.roll.select_ids(["a", "c"])
    w.listening.play_selected_notes()
    assert len(pv.last_sound) / 44100 == pytest.approx(2.6 - 1.0 + 0.04, abs=0.02)
    # 끄면 울리지 않는다
    w.listening.act_preview.trigger()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _note_point(w, notes[0]))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _note_point(w, notes[0]))
    assert pv.triggers == 4
    assert w.errors == []


@needs_device
def test_preview_voice_replaces_previous_sound(qtbot, window):
    pv = window.listening.preview
    for p in range(60, 72):  # 빠르게 연달아 눌러도 소리 하나만 남는다
        assert pv.play_note(p)
    buf = pv._voice._buf
    assert len(buf) <= len(pv.last_sound) + 1
    assert abs(_f0(buf[200:]) - midi_to_hz(71)) < 4
    pv.shutdown()
    assert pv._sink is None


def test_midi_track_follows_notes(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    _setup_notes(w)
    w.listening.act_midi.trigger()
    qtbot.waitUntil(lambda: "midi" in w.player.tracks, timeout=10_000)
    midi = w.player.tracks["midi"]
    assert midi.frames == w.session.audio.frames
    i = int(1.2 * midi.sample_rate)
    assert abs(_f0(midi.samples[i:i + 8192, 0]) - midi_to_hz(60)) < 3
    w.doc.move_notes(["a"], 0.0, 12)
    qtbot.waitUntil(lambda: w.player.tracks["midi"] is not midi, timeout=10_000)
    m2 = w.player.tracks["midi"]
    assert abs(_f0(m2.samples[i:i + 8192, 0]) - midi_to_hz(72)) < 4
    w.listening.act_midi.trigger()
    assert "midi" not in w.player.tracks
    assert w.errors == []


def test_spectrogram_lane_modes_tiles_and_height(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    w.resize(1400, 900)
    sp = w.spectrogram
    assert not sp.isVisible()
    w.listening.lane_actions["both"].trigger()
    assert sp.isVisible() and w.waveform.isVisible()
    sp.grab()  # 그리기 → 보이는 타일 요청
    assert sp.wait_idle(10_000)
    qtbot.waitUntil(lambda: len(sp.cache) > 0, timeout=5000)
    n_tiles = len(sp.cache)
    sp.grab()
    assert len(sp.cache) == n_tiles  # 같은 화면은 다시 계산하지 않는다
    w.listening.lane_actions["spec"].trigger()
    assert not w.waveform.isVisible() and sp.isVisible()
    w.listening.lane_actions["spec_curve"].trigger()
    assert sp.show_curve
    w.listening.lane_actions["hidden"].trigger()
    assert not sp.isVisible() and not w.waveform.isVisible()
    w.listening.lane_actions["both"].trigger()
    h0 = sp.height()
    y = h0 - 2
    QTest.mousePress(sp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, y))
    QTest.mouseMove(sp, QPoint(100, y + 60))
    QTest.mouseRelease(sp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, y + 60))
    assert sp.height() > h0 + 40
    # 확대하면 더 촘촘한 hop을 쓴다
    hop_before = sp.params().hop
    w.view.zoom_time(0.05)
    assert sp.params().hop >= hop_before
    assert w.errors == []
