"""STEP 5 Phase C 화면: 채보·가사 내보내기 대화상자와 형식별 저장 (실제 작업 스레드 경로)."""

import json

import numpy as np
import yaml

from amp.app.export_controller import DataExportDialog
from amp.core.model import AnalysisResult, PitchCurve
from amp.core.tempo import TempoMap
from amp.export.common import ExportOptions
from amp.lyrics import linking, ops
from tests.conftest import note


def create(qtbot, w, path):
    w.create_project_from_audio(path, None, "copy")
    qtbot.waitUntil(lambda: w.session.audio is not None, timeout=20_000)
    return w.session.folder


def test_export_menu_formats_user_edits_and_reports(qtbot, window, song, tmp_path, monkeypatch):
    w = window
    folder = create(qtbot, w, song)
    notes = [note("a", 1.0, 1.8, 64), note("b", 1.9, 2.7, 67), note("c", 2.8, 4.0, 69)]
    f0 = np.full(600, np.nan)
    for n in notes:
        f0[int(n.start * 100):int(n.end * 100)] = 440 * 2 ** ((n.pitch - 69) / 12)
    w.doc.set_analysis(AnalysisResult(6.0, TempoMap.constant(120), tuple(notes),
                                      pitch_curve=PitchCurve(0.0, 100.0, f0, np.where(np.isfinite(f0), 0.9, 0.0))))
    st = ops.apply_text(w.doc.lyrics, "미 솔 라")
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {ids[0]: (1.0, 1.8), ids[1]: (1.9, 2.7), ids[2]: (2.8, 4.0)}, source="model")
    w.doc.push_lyrics(linking.auto_link(st, w.doc.score.notes()), "가사")
    w.doc.move_notes(["c"], 0.0, 1)  # 사용자 수정
    assert "예정" not in w.exporter.act.text() and w.exporter.act.isEnabled()
    dlg = DataExportDialog(str(folder.paths.exports), False, None, None)
    assert dlg.path.text().endswith("transcription.ustx")
    dlg.format.setCurrentIndex(dlg.format.findData("lrc"))
    assert dlg.path.text().endswith("lyrics.lrc") and not dlg.pitch.isEnabled()
    shown = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    out = {}
    for kind, ext in (("ustx", ".ustx"), ("midi", ".mid"), ("musicxml", ".musicxml"), ("json", ".json"),
                      ("csv", ".csv"), ("lrc", ".lrc")):
        path = str(folder.paths.exports / f"내보내기{ext}")
        h = w.exporter.export(kind, path, ExportOptions(pitch="simple" if kind in ("ustx", "json") else "none"))
        qtbot.waitSignal(h.ended, timeout=20_000)
        qtbot.waitUntil(lambda: len(shown) == len(out) + 1, timeout=5000)
        out[kind] = path
    doc = yaml.safe_load(open(out["ustx"], encoding="utf-8"))
    assert [n["tone"] for n in doc["voice_parts"][0]["notes"]] == [64, 67, 70]  # 사용자 수정이 들어간다
    assert [n["lyric"] for n in doc["voice_parts"][0]["notes"]] == ["미", "솔", "라"]
    j = json.load(open(out["json"], encoding="utf-8"))
    assert j["notes"][-1]["status"] == "modified" and "pitch_curve" in j
    assert open(out["lrc"], encoding="utf-8").read().splitlines()[-1] == "[00:01.00]미 솔 라"
    assert "담지 못한 정보" in shown[2]  # MusicXML 보고에 누락 정보 안내
    assert w.errors == []


def test_export_dialog_ufdata_transpose_encoding_and_range(qtbot, window, song, monkeypatch):
    """VOX Factory 등 보컬 합성 프로그램용: UFData 형식, 조옮김(합성 형식만), MIDI 가사 인코딩, 곡 음역 표시."""
    w = window
    folder = create(qtbot, w, song)
    notes = [note("a", 1.0, 1.8, 57), note("b", 1.9, 2.7, 64), note("c", 2.8, 4.0, 76)]
    w.doc.set_analysis(AnalysisResult(6.0, TempoMap.constant(120), tuple(notes)))
    dlg = DataExportDialog(str(folder.paths.exports), False, None, None,
                           notes_for_range=[(n.pitch, n.end - n.start) for n in notes])
    qtbot.addWidget(dlg)
    dlg.format.setCurrentIndex(dlg.format.findData("ufdata"))
    assert dlg.path.text().endswith("transcription.ufdata")
    assert dlg.transpose.isEnabled() and dlg.extend_lyric.isEnabled() and not dlg.midi_encoding.isEnabled()
    assert dlg.range_info.text().startswith("A3(57) ~ E5(76)")
    dlg.transpose.setValue(-3)
    assert "조옮김 뒤: F#3(54) ~ C#5(73)" in dlg.range_info.text()
    kind, _path, opts = dlg.values()
    assert kind == "ufdata" and opts.transpose == -3 and opts.extend_lyric == "-"
    dlg.format.setCurrentIndex(dlg.format.findData("midi"))
    dlg.midi_encoding.setCurrentIndex(dlg.midi_encoding.findData("cp949"))
    assert dlg.values()[2].midi_lyric_encoding == "cp949" and dlg.values()[2].transpose == -3
    dlg.format.setCurrentIndex(dlg.format.findData("csv"))
    assert not dlg.transpose.isEnabled() and dlg.values()[2].transpose == 0  # 표·가사 형식은 조옮김 안 함
    # 실제 작업 스레드로 UFData 저장
    shown = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    path = str(folder.paths.exports / "보컬.ufdata")
    h = w.exporter.export("ufdata", path, ExportOptions(transpose=-3))
    qtbot.waitSignal(h.ended, timeout=20_000)
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    d = json.load(open(path, encoding="utf-8"))
    assert [n["key"] for n in d["project"]["tracks"][0]["notes"]] == [54, 61, 73]
    assert "첫 음:" in shown[0] and "음역:" in shown[0]
    assert w.errors == []
