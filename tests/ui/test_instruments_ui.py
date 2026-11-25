"""v0.6.4 오른쪽 탭 묶음(노래 분석·악기)과 악기 소리 패널: 분리 → 상태 표시 → 내보내기, 다시 누르면 분리하지 않음,
보컬 분석 소스·보컬 음표는 그대로, 작업 공간(고른 탭) 저장. 분리 모델은 시험용(_six)이다."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QFileDialog

from amp.app import instrument_controller as IC
from tests.ui.test_main_window import create


@pytest.fixture
def six(monkeypatch):
    monkeypatch.setattr(IC, "KINDS", {"basic": ("_six", ("drums", "bass", "other")),
                                      "six": ("_six", ("drums", "bass", "guitar", "piano", "other"))})
    monkeypatch.setattr("amp.app.lite.separation_installed", lambda: True)


def test_tabs_are_grouped(window):
    w = window
    assert [w.right_tabs.tabText(i) for i in range(w.right_tabs.count())] == ["속성·분석", "노래 분석", "악기"]
    assert [w.song_tabs.tabText(i) for i in range(w.song_tabs.count())] == ["가사", "검토", "교정 제안", "품질", "악보 비교"]
    assert [w.instrument_tabs.tabText(i) for i in range(w.instrument_tabs.count())] == ["악기 소리", "베이스 TAB"]
    w.show_side_panel(w.suggest.panel)
    assert w.right_tabs.currentWidget() is w.song_tabs and w.side_panel() is w.suggest.panel
    w.show_side_panel(w.bass.panel)
    assert w.side_panel() is w.bass.panel and w._side_path() == ["악기", "베이스 TAB"]
    w._apply_ui_state({"right_tab": 2})  # 0.6.3 저장값: 한 줄 탭의 2번 = 검토
    assert w.side_panel() is w.review.panel
    w._apply_ui_state({"right_panel": ["악기", "악기 소리"]})
    assert w.side_panel() is w.instruments.panel


def test_extract_and_export_instruments(qtbot, window, song, six, tmp_path, monkeypatch):
    w = window
    folder = create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.run_sample_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=20_000)
    vocal = [(n.id, n.pitch) for n in w.doc.score.notes()]
    p = w.instruments.panel
    w.show_side_panel(p)
    assert not p.export["drums"].isEnabled() and p.status["drums"].text() == "없음"

    p.basic.click()
    qtbot.waitUntil(lambda: "drums" in w.instruments.found and not w.session.busy, timeout=60_000)
    assert set(w.instruments.found) == {"drums", "bass", "other"} and p.export["drums"].isEnabled()
    assert "있음" in p.status["drums"].text() and not p.export["guitar"].isEnabled()
    assert "저장했습니다" in p.message.text()
    p.six.click()
    qtbot.waitUntil(lambda: "piano" in w.instruments.found and not w.session.busy, timeout=60_000)
    assert "이미 있어 그대로" in p.message.text()  # 드럼·베이스·나머지는 다시 만들지 않음
    p.six.click()
    qtbot.waitUntil(lambda: "다시 분리하지 않았습니다" in p.message.text() and not w.session.busy, timeout=60_000)

    assert w.session.model_store().pointer() is None  # 보컬 분석 소스는 정하지 않음
    assert [(n.id, n.pitch) for n in w.doc.score.notes()] == vocal

    src = w.instruments.found["guitar"].path
    ref, sr = sf.read(str(src), dtype="float32", always_2d=True)
    dst = tmp_path / "기타.wav"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(dst), ""))
    p.export["guitar"].click()
    qtbot.waitUntil(lambda: dst.exists() and not w.runner.active_count, timeout=60_000)
    got, gsr = sf.read(str(dst), dtype="float32", always_2d=True)
    assert gsr == sr and got.shape == ref.shape and np.max(np.abs(got - ref)) < 1e-6

    assert w.save()
    meta = json.loads(folder.paths.project_json.read_text(encoding="utf-8"))
    assert meta["ui"]["right_panel"] == ["악기", "악기 소리"]

    # 듣기: 켠 악기만 재생기 트랙(음소거 + 솔로)으로 올라가고, 끄면 내린다. 원본·보컬·반주 상태는 그대로
    from amp.audio.pcm import PcmMixer

    before = {n: w.player.track_state(n) for n in ("original", "vocals", "instrumental")}
    p.listen["drums"].click()
    qtbot.waitUntil(lambda: "inst_drums" in w.player.tracks, timeout=20_000)
    p.listen["bass"].click()
    qtbot.waitUntil(lambda: "inst_bass" in w.player.tracks, timeout=20_000)
    assert w.player.track_state("inst_drums") == {"gain": 1.0, "mute": True, "solo": True}
    assert "드럼·베이스" in p.listen_info.text()

    def heard():
        tracks = {n: a.samples for n, a in w.player.tracks.items()}
        m = PcmMixer(tracks, 44100, 44100, 2)
        for n in tracks:
            m.set_state(n, **w.player.track_state(n))
        return sorted(n for n, _g in m.audible())

    assert heard() == ["inst_bass", "inst_drums"]  # 켠 악기만 들린다
    w.player.set_track_states({n: {"solo": False} for n in w.player.tracks})  # 다른 쪽이 솔로를 모두 끈 경우 (A/B 비교 등)
    assert heard() == ["original"]  # 악기 트랙은 음소거라 기존 재생과 같다
    w.player.set_track_states({n: {"solo": True} for n in ("inst_drums", "inst_bass")})
    p.listen["drums"].click()
    assert "inst_drums" not in w.player.tracks and heard() == ["inst_bass"]
    assert w.bass.panel.listen.isChecked()  # 베이스 TAB 탭의 [베이스만 듣기]도 같은 상태
    w.bass.panel.listen.click()
    assert "inst_bass" not in w.player.tracks and not p.listen["bass"].isChecked()
    w.bass.panel.listen.click()
    qtbot.waitUntil(lambda: "inst_bass" in w.player.tracks, timeout=20_000)
    assert p.listen["bass"].isChecked()
    p.stop_listen.click()
    assert not any(n.startswith("inst_") for n in w.player.tracks) and not p.listen["bass"].isChecked()
    assert heard() == ["original"]
    assert {n: w.player.track_state(n) for n in before} == before


def test_extract_disabled_without_separation(qtbot, window, song, monkeypatch):
    monkeypatch.setattr("amp.app.lite.separation_installed", lambda: False)
    w = window
    w.instruments._installed = None
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    p = w.instruments.panel
    assert not p.basic.isEnabled() and not p.six.isEnabled()
    assert "설치" in p.basic.toolTip()
