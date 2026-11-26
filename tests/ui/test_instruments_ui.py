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
    assert [w.song_tabs.tabText(i) for i in range(w.song_tabs.count())] == ["가사", "보컬 정밀", "검토", "교정 제안", "품질",
                                                                           "악보 비교"]  # v0.7.0: 보컬 정밀 추가
    assert [w.instrument_tabs.tabText(i) for i in range(w.instrument_tabs.count())] == ["악기 소리", "베이스 TAB"]
    w.show_side_panel(w.suggest.panel)
    assert w.right_tabs.currentWidget() is w.song_tabs and w.side_panel() is w.suggest.panel
    w.show_side_panel(w.bass.panel)
    assert w.side_panel() is w.bass.panel and w._side_path() == ["악기", "베이스 TAB"]
    w._apply_ui_state({"right_tab": 2})  # 0.6.3 저장값: 한 줄 탭의 2번 = 검토
    assert w.side_panel() is w.review.panel
    w._apply_ui_state({"right_panel": ["악기", "악기 소리"]})
    assert w.side_panel() is w.instruments.panel


def test_extract_and_export_instruments(qtbot, make_window, song, six, tmp_path, monkeypatch):
    w = make_window()
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

    # 왼쪽 트랙 패널의 악기 줄 (v0.6.6): 뽑은 악기만 보이고, 처음에는 음소거라 재생기에 올리지 않는다
    from amp.audio.pcm import PcmMixer
    from PySide6.QtCore import Qt

    rows = w.tracks.inst_rows
    assert all(not rows[f"inst_{n}"].isHidden() for n in ("drums", "bass", "guitar", "piano", "other"))
    assert not w.tracks.inst_title.isHidden()
    assert w.player.track_state("inst_drums")["mute"] and rows["inst_drums"].mute.isChecked()
    assert not any(n.startswith("inst_") for n in w.player.tracks)
    before = {n: w.player.track_state(n) for n in ("original", "vocals", "instrumental")}

    def heard():
        tracks = {n: a.samples for n, a in w.player.tracks.items()}
        m = PcmMixer(tracks, 44100, 44100, 2)
        for n in tracks:
            m.set_state(n, **w.player.track_state(n))
        return sorted(n for n, _g in m.audible())

    rows["inst_drums"].solo.click()  # 헤드폰 = 드럼만
    qtbot.waitUntil(lambda: "inst_drums" in w.player.tracks, timeout=20_000)
    assert heard() == ["inst_drums"]
    rows["inst_bass"].mute.click()  # 음소거 풀기 = 지금 소리에 더하기 (드럼이 솔로라 아직은 드럼만)
    qtbot.waitUntil(lambda: "inst_bass" in w.player.tracks, timeout=20_000)
    assert heard() == ["inst_drums"]
    rows["inst_drums"].solo.click()  # 솔로 해제: 드럼은 음소거라 내리고, 원본 + 베이스
    assert "inst_drums" not in w.player.tracks and heard() == ["inst_bass", "original"]
    assert {n: w.player.track_state(n) for n in before} == before

    rows["inst_bass"].solo.click()
    assert w.bass.panel.listen.isChecked()  # 베이스 TAB 탭의 [베이스만 듣기]는 Bass 줄 헤드폰과 같은 상태
    w.bass.panel.listen.click()
    assert not rows["inst_bass"].solo.isChecked() and not w.player.track_state("inst_bass")["solo"]

    qtbot.mouseClick(rows["inst_piano"], Qt.MouseButton.LeftButton)  # 줄을 누르면 그 악기 파형
    qtbot.waitUntil(lambda: w.instruments.peaks("inst_piano") is not None, timeout=20_000)
    assert w.tracks.selected_track == "inst_piano" and w.wave_label.text() == "피아노"
    assert "inst_piano" not in w.player.tracks  # 파형만 보고 소리는 음소거 그대로
    qtbot.mouseClick(w.tracks.rows["original"], Qt.MouseButton.LeftButton)
    assert w.instruments.audio("inst_piano") is None  # 다른 줄을 고르면 메모리에서 내린다

    assert w.save()  # 악기 트랙 상태도 저장: 다시 열면 음소거를 푼 베이스를 불러 둔다
    meta = json.loads(folder.paths.project_json.read_text(encoding="utf-8"))
    assert meta["playback"]["tracks"]["inst_bass"]["mute"] is False
    assert meta["playback"]["tracks"]["inst_drums"]["mute"] is True
    w.close()
    w2 = make_window("w2")
    assert w2.open_project(str(folder.root))
    qtbot.waitUntil(lambda: "inst_bass" in w2.player.tracks, timeout=20_000)
    assert w2.tracks.inst_rows["inst_bass"].mute.isChecked() is False and "inst_drums" not in w2.player.tracks


def test_extract_disabled_without_separation(qtbot, window, song, monkeypatch):
    monkeypatch.setattr("amp.app.lite.separation_installed", lambda: False)
    w = window
    w.instruments._installed = None
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    p = w.instruments.panel
    assert not p.basic.isEnabled() and not p.six.isEnabled()
    assert "설치" in p.basic.toolTip()


def test_extract_from_vocal_analysis_instrumental(qtbot, window, song, six, monkeypatch):
    """v0.6.7: 보컬 분석을 했으면 기본은 그 반주에서 분리하고, 결과는 그 분리 결과에 더한다 (새 결과를 만들지 않음)."""
    from amp.engine.model_store import ModelStore
    from amp.engine.transcription import TranscriptionPipeline

    class HalfPipeline(TranscriptionPipeline):
        def __init__(self, *a, **k):
            super().__init__(*a, separation_model="_half", **k)

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", HalfPipeline)
    w = window
    p = w.instruments.panel
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    assert p.source.currentData() == "original"  # 보컬 분석 전: 반주가 없다
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    qtbot.waitUntil(lambda: p.source.currentData() == "instrumental", timeout=10_000)
    store = w.session.model_store()
    active = store.active_key()
    p.basic.click()
    qtbot.waitUntil(lambda: "drums" in w.instruments.found and not w.session.busy, timeout=60_000)
    assert w.instruments.found["drums"].key == active and len(store.entries()) == 1
    assert "반주에서" in w.tracks.inst_rows["inst_drums"].info.text()
    assert "반주에서" in p.message.text()
    p.source.setCurrentIndex(p.source.findData("original"))  # 원본에서도 고를 수 있다 (새 보관 결과)
    p.six.click()
    qtbot.waitUntil(lambda: "piano" in w.instruments.found and not w.session.busy, timeout=60_000)
    assert len(ModelStore(w.session.folder.paths.stems, w.session.folder.meta.original.sha256).entries()) == 2
    assert w.instruments.found["drums"].key == active  # 같은 악기는 지금 쓰는 결과(반주에서 뽑은 것)를 먼저


def test_group_mute_buttons(qtbot, window, song, six):
    """v0.6.7 트랙 창의 전부 음소거·묶음 음소거: 켜면 그 묶음이 들리지 않고(솔로도 끔), 다시 누르면 전 상태로."""
    from amp.audio.pcm import PcmMixer

    w = window
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.instruments.panel.basic.click()
    qtbot.waitUntil(lambda: "drums" in w.instruments.found and not w.session.busy, timeout=60_000)
    tp, rows = w.tracks, w.tracks.inst_rows

    def heard():
        tracks = {n: a.samples for n, a in w.player.tracks.items()}
        m = PcmMixer(tracks, 44100, 44100, 2)
        for n in tracks:
            m.set_state(n, **w.player.track_state(n))
        return sorted(n for n, _g in m.audible())

    assert tp.mute_inst.isChecked() and not tp.mute_main.isChecked() and not tp.mute_all.isChecked()  # 악기는 처음에 음소거
    rows["inst_bass"].solo.click()
    qtbot.waitUntil(lambda: "inst_bass" in w.player.tracks, timeout=20_000)
    assert not tp.mute_inst.isChecked() and heard() == ["inst_bass"]

    tp.mute_all.click()  # 전부 음소거: 솔로도 꺼서 아무것도 들리지 않는다
    assert heard() == [] and tp.mute_main.isChecked() and tp.mute_inst.isChecked()
    assert rows["inst_bass"].mute.isChecked() and not rows["inst_bass"].solo.isChecked()
    tp.mute_all.click()  # 다시 누르면 전 상태 (원본 켜짐, 베이스 솔로)
    qtbot.waitUntil(lambda: "inst_bass" in w.player.tracks, timeout=20_000)
    assert heard() == ["inst_bass"] and rows["inst_bass"].solo.isChecked() and not tp.mute_all.isChecked()

    tp.mute_inst.click()  # 악기 묶음만 음소거 → 원본만
    assert heard() == ["original"] and tp.mute_inst.isChecked()
    tp.mute_inst.click()
    qtbot.waitUntil(lambda: "inst_bass" in w.player.tracks, timeout=20_000)
    assert heard() == ["inst_bass"]

    rows["inst_bass"].solo.click()  # 솔로 해제 → 원본
    tp.mute_main.click()  # 원본·분리 트랙 묶음 음소거
    assert tp.rows["original"].mute.isChecked() and heard() == []
    tp.rows["original"].mute.click()  # 줄 하나를 직접 켜면 묶음 단추가 풀린다
    assert not tp.mute_main.isChecked() and heard() == ["original"]
    tp.mute_all.click()
    tp.mute_all.click()  # 되돌릴 상태가 있으면 그 상태로: 원본만 (악기는 원래대로 음소거)
    assert heard() == ["original"] and rows["inst_drums"].mute.isChecked()
