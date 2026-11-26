"""v0.7.0 Phase 7 화면: 분석 결과 내보내기(PNG·구간 WAV 포함)와 검증 자료 창(저장·비교·확인 뒤 지우기)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from PySide6.QtGui import QImage

from tests.ui.test_main_window import IdentityPipeline, create


def _analysed(qtbot, w, song, monkeypatch):
    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    folder = create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    w.precision.acoustics.run()
    qtbot.waitUntil(lambda: w.precision.acoustics.res is not None and not w.session.busy, timeout=120_000)
    return folder


def test_analysis_export_with_png_and_region_wav(qtbot, window, song, monkeypatch, tmp_path):
    from amp.app.analysis_export_dialog import AnalysisExportDialog, run_export

    w = window
    _analysed(qtbot, w, song, monkeypatch)
    dlg = AnalysisExportDialog(w, str(tmp_path), (1.0, 3.0), {"original", "vocals"})
    qtbot.addWidget(dlg)
    assert dlg.loop.isChecked() and not dlg.wav["instrumental"].isEnabled()
    for cb in (*dlg.png.values(), dlg.wav["vocals"]):
        cb.setChecked(True)
    dlg.folder.setText(str(tmp_path / "out"))
    choice = dlg.choice()
    assert choice["region"] == (1.0, 3.0) and set(choice["png"]) == {"pitch", "spectrogram", "vibrato"}
    out = []
    handle = w.runner.submit("시험", run_export(w, choice))
    handle.finished.connect(out.append)
    qtbot.waitUntil(lambda: bool(out), timeout=60_000)
    written, skipped = out[0]
    names = {p.rsplit("/", 1)[-1].rsplit("\\\\", 1)[-1] for p in written}
    pngs = [p for p in written if p.endswith(".png")]
    assert len(pngs) == 3 and all(QImage(p).width() == 1400 for p in pngs)
    wav = next(p for p in written if p.endswith(".wav"))
    info = sf.info(wav)
    assert abs(info.duration - 2.0) < 0.01 and info.subtype == "PCM_24"
    assert any(n.endswith("-analysis.json") for n in names) and "csv:voices" in skipped


def test_validation_dialog_save_compare_and_confirmed_delete(qtbot, window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from amp.app.validation_dialog import ValidationDialog
    from amp.devtools.synth import SR, sung
    from amp.engine import validation as V

    y = sung([(0.5, 3.0, 64), (3.5, 7.0, 67), (7.5, 13.0, 69)], 14.0, vibrato_cents=40)  # 10초 이상 구간이 필요
    long_song = tmp_path / "긴 노래.wav"
    sf.write(long_song, np.stack([y, y], 1), SR)
    w = window
    folder = _analysed(qtbot, w, str(long_song), monkeypatch)
    dlg = ValidationDialog(w)
    qtbot.addWidget(dlg)
    dlg.start.setValue(0.0)
    dlg.end.setValue(4.0)
    dlg.items["pitch"].setChecked(True)
    assert dlg.save_current() is None and "10~30초" in dlg.result.toPlainText()  # 구간이 짧으면 저장하지 않음
    dlg.end.setValue(min(w.session.audio.duration, 12.0))
    dlg.start.setValue(max(0.0, dlg.end.value() - 11.0))
    dlg.category.setCurrentIndex(dlg.category.findData("dev_real"))
    dlg.items["vibrato"].setChecked(True)
    vs = dlg.save_current()
    assert vs is not None and dlg.list.count() == 1 and (folder.root / "validation" / f"{vs.set_id}.json").is_file()
    res = dlg.compare_selected()
    assert res["pitch"]["raw_pitch_accuracy_50c"] == 1.0 and "독립 검증 자료가 아닙니다" in res["warning"]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert not dlg.delete_selected() and dlg.list.count() == 1  # 묻고 '아니요'면 지우지 않는다
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert dlg.delete_selected() and V.load_all(folder.root) == []
    assert np.isfinite(w.doc.project.analysis.pitch_curve.f0_hz).any()  # 분석 결과는 그대로
