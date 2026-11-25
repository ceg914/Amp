"""v0.6.3 베이스 TAB 화면: 분석 → 미리보기, 베이스 편집 모드(피아노 롤 바꾸기·실행 취소 분리), 다시 분석해도 고친 음표 유지,
저장·다시 열기, GP5·텍스트 내보내기. 분리 모델은 시험용(_withbass: 입력 전체를 베이스로)이다."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QFileDialog

from amp.core.tempo import TempoMap
from amp.engine.bass_pipeline import BassPipeline
from tests.test_bass import LINE, T0, _bass_audio
from tests.ui.test_main_window import create

pytest.importorskip("librosa")


class WithBassPipeline(BassPipeline):
    def __init__(self, *a, **k):
        super().__init__(*a, separation_model="_withbass", **k)


@pytest.fixture
def bass_song(tmp_path):
    from amp.devtools.synth import SR

    y = _bass_audio()
    p = tmp_path / "베이스 연습곡.wav"
    sf.write(p, np.stack([y, y], 1), SR)
    return str(p)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr("amp.engine.bass_pipeline.BassPipeline", WithBassPipeline)
    monkeypatch.setattr("amp.app.lite.separation_installed", lambda: True)


def _analyze(qtbot, w):
    before = w.bass.doc.project.analysis
    w.act_bass_analyze.trigger()
    qtbot.waitUntil(lambda: w.bass.doc.project.analysis is not before and not w.session.busy, timeout=120_000)


def test_bass_analysis_edit_mode_undo_and_save(qtbot, make_window, bass_song, patched, tmp_path, monkeypatch):
    w = make_window()
    folder = create(qtbot, w, bass_song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.doc.set_tempo(TempoMap.constant(120.0, T0))
    w.bass.panel.tuning.setCurrentIndex(w.bass.panel.tuning.findData("5_standard"))
    _analyze(qtbot, w)
    bass = w.bass.doc
    assert [n.pitch for n in bass.score.notes()] == [p for p, _d in LINE]
    assert w.doc.project.bass is not None and w.doc.is_dirty
    assert w.act_bass_gp5.isEnabled() and w.act_bass_text.isEnabled()
    assert len(w.doc.score) == 0  # 보컬 음표는 그대로 (없음)
    assert "E|5" in w.bass.panel.preview.toPlainText().replace(" ", "") or "|5" in w.bass.panel.preview.toPlainText()
    store = w.session.model_store()
    assert any(e.manifest.files.get("bass") for e in store.entries())  # 베이스를 분리 결과에 더했다

    # 베이스 편집 모드: 가운데 피아노 롤이 베이스로 바뀌고, 삭제·실행 취소가 베이스 쪽에 걸린다
    w.act_bass_edit.trigger()
    assert w.bass.active and w.bass.view.isVisible() and not w.view.isVisible()
    assert w.bass.panel.edit.isChecked()
    first = bass.score.notes()[0]
    w.bass.view.roll.select_ids([first.id])
    w.delete_selected()
    assert len(bass.score) == len(LINE) - 1
    w.act_undo.trigger()
    assert len(bass.score) == len(LINE)
    w.act_redo.trigger()
    assert len(bass.score) == len(LINE) - 1
    target = bass.score.notes()[0]
    assert bass.move_notes([target.id], 0.0, 12)  # 한 옥타브 올림 (사용자 수정)
    moved = target.pitch + 12
    w.act_bass_edit.trigger()
    assert not w.bass.active and w.view.isVisible() and not w.bass.view.isVisible()
    assert w.undo_group.activeStack() is w.doc.undo_stack

    # 다시 분석해도 고친 음표는 지킨다
    _analyze(qtbot, w)
    assert any(n.pitch == moved and abs(n.start - target.start) < 1e-6 for n in bass.score.notes())

    assert w.save()
    d = json.loads(folder.paths.bass_json.read_text(encoding="utf-8"))
    assert d["schema"] == "sonatlas.bass" and d["options"]["tuning"] == "5_standard"
    assert any(n["status"] == "modified" for n in d["edits"]["notes"])
    saved = [(n.pitch, round(n.start, 4)) for n in bass.score.notes()]

    # 내보내기
    out = {"gp5": str(tmp_path / "out" / "베이스.gp5"), "txt": str(tmp_path / "out" / "베이스.txt")}
    (tmp_path / "out").mkdir()
    for kind, act in (("gp5", w.act_bass_gp5), ("txt", w.act_bass_text)):
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, k=kind, **kw: (out[k], ""))
        act.trigger()
    from amp.export.gp5 import read_gp5

    heads = [g for g in read_gp5(out["gp5"]) if not g["tie"]]
    assert [g["midi"] for g in heads] == [p for p, _s in sorted(saved, key=lambda x: x[1])]
    assert "현" in open(out["txt"], encoding="utf-8-sig").read()

    # 베이스 소리 내보내기: WAV는 분리해 둔 베이스와 같은 소리, MP3도 쓴다. 분리 파일은 그대로
    stem = w.bass.stem_path()
    stem_bytes = open(stem, "rb").read()
    ref, sr = sf.read(stem, dtype="float32", always_2d=True)
    for name in ("베이스.wav", "베이스.mp3"):
        dst = str(tmp_path / "out" / name)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, d=dst, **kw: (d, ""))
        w.act_bass_audio.trigger()
        qtbot.waitUntil(lambda d=dst: (tmp_path / "out" / name).exists() and not w.runner.active_count, timeout=60_000)
    got, gsr = sf.read(str(tmp_path / "out" / "베이스.wav"), dtype="float32", always_2d=True)
    assert gsr == sr and got.shape == ref.shape and np.max(np.abs(got - ref)) < 1e-6
    assert sf.info(str(tmp_path / "out" / "베이스.mp3")).format == "MP3"
    assert open(stem, "rb").read() == stem_bytes
    w.close()

    w2 = make_window("w2")
    assert w2.open_project(str(folder.root))
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    assert [(n.pitch, round(n.start, 4)) for n in w2.bass.doc.score.notes()] == saved
    assert w2.bass.panel.options().tuning == "5_standard"
    assert not w2.doc.is_dirty
    assert w2.bass.doc.note_status(next(n for n in w2.bass.doc.score.notes() if n.pitch == moved)) == "modified"


def test_bass_without_separation_reports_and_keeps_vocals(qtbot, window, song, monkeypatch):
    """분리를 설치하지 않았고 베이스 분리 결과도 없으면 안내만 하고, 보컬 쪽은 바뀌지 않는다."""
    monkeypatch.setattr("amp.app.lite.separation_installed", lambda: False)
    w = window
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.run_sample_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=20_000)
    vocal = [(n.id, n.pitch) for n in w.doc.score.notes()]
    w.act_bass_analyze.trigger()
    qtbot.waitUntil(lambda: bool(w.errors) and not w.session.busy, timeout=60_000)
    assert "베이스 분리 결과가 없습니다" in w.errors[-1]
    assert w.doc.project.bass is None and [(n.id, n.pitch) for n in w.doc.score.notes()] == vocal
    assert not w.act_bass_gp5.isEnabled()
    shown = []
    monkeypatch.setattr("amp.app.bass_controller.QMessageBox.information", lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: pytest.fail("파일을 묻지 않아야 함"))
    w.act_bass_audio.trigger()
    assert shown and "베이스 분석(F8)" in shown[0]
