"""STEP 5 Phase B 화면 시험: 분석 오류 검토 목록, 참조 악보 가져오기·정렬·확정, 음정 비교, 저장·다시 열기."""

import os

import mido
import numpy as np
import pytest

from amp.app import dialogs
from amp.core.model import AnalysisResult, PitchCurve
from amp.core.tempo import TempoMap
from tests.conftest import note

FR = 100.0
MELODY = [(1.0, 1.8, 64), (1.9, 2.7, 67), (2.8, 4.0, 69)]


def create(qtbot, w, path):
    w.create_project_from_audio(path, None, "copy")
    qtbot.waitUntil(lambda: w.session.audio is not None, timeout=20_000)
    return w.session.folder


def _curve(segments, dur=6.0):
    f0 = np.full(int(dur * FR), np.nan)
    for a, b, p in segments:
        f0[int(a * FR):int(b * FR)] = 440 * 2 ** ((p - 69) / 12)
    return PitchCurve(0.0, FR, f0, np.where(np.isfinite(f0), 0.9, 0.05))


def _set_analysis(w, notes, curve):
    w.doc.set_analysis(AnalysisResult(6.0, TempoMap.constant(120), tuple(notes), pitch_curve=curve,
                                      onset_envelope=np.zeros(curve.f0_hz.size)))


def _midi_file(tmp_path, notes, name="ref.mid"):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="Vocal", time=0))
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    last = 0
    for start, dur, p in notes:  # 초 → 틱 (120BPM: 1초 = 960틱)
        on, off = int(start * 960), int((start + dur) * 960)
        tr.append(mido.Message("note_on", note=p, velocity=90, time=on - last))
        tr.append(mido.Message("note_off", note=p, velocity=0, time=off - on))
        last = off
    path = tmp_path / name
    mid.save(str(path))
    return str(path)


def test_review_list_navigation_status_undo_and_save(qtbot, window, song):
    w = window
    folder = create(qtbot, w, song)
    notes = [note("n1", 1.0, 1.8, 64, confidence=0.15), note("n2", 1.9, 2.7, 67, confidence=0.9)]
    _set_analysis(w, notes, _curve(MELODY))  # 2.8~4.0초 유성인데 음표 없음
    rc = w.review
    rc.refresh()
    qtbot.waitUntil(lambda: len(rc.items) > 0, timeout=10_000)
    kinds = {i.kind for i in rc.items}
    assert {"note_low_conf", "voiced_uncovered"} <= kinds
    before = [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()]
    panel = rc.panel
    row = next(r for r, (it, _s) in enumerate(panel._rows) if it.kind == "note_low_conf")
    panel.select_row(row)
    assert w.view.roll.selected_ids() == ["n1"]
    panel.actionRequested.emit("checked")
    item = next(i for i in rc.items if i.kind == "note_low_conf")
    assert w.doc.project.review.status_of(item) == "checked"
    # 필터: 미검토만이면 확인한 항목은 숨는다
    panel.unreviewed.setChecked(True)
    assert all(it.key != item.key for it, _s in panel._rows)
    panel.unreviewed.setChecked(False)
    w.doc.undo_stack.undo()
    assert w.doc.project.review.status_of(item) == "open"
    w.doc.undo_stack.redo()
    # 목록은 음표를 바꾸지 않는다
    assert [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()] == before
    assert w.save()
    assert (folder.paths.review_json).is_file()
    # 다시 열기: 상태 유지 → 근거가 바뀌면 '재검토 필요'
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    assert w.open_project(os.path.join(root, "project.json"))
    qtbot.waitUntil(lambda: w.session.audio is not None and not w.session.busy, timeout=20_000)
    assert w.doc.project.review.status_of(item) == "checked"
    changed = note("n1", 1.0, 1.8, 64, confidence=0.1)
    _set_analysis(w, [changed, notes[1]], _curve(MELODY))
    rc.refresh()
    qtbot.waitUntil(lambda: any(i.kind == "note_low_conf" and i.fingerprint != item.fingerprint for i in rc.items),
                    timeout=10_000)
    new = next(i for i in rc.items if i.kind == "note_low_conf")
    assert w.doc.project.review.status_of(new) == "recheck"
    assert w.errors == []


def test_reference_import_align_confirm_intonation_and_reopen(qtbot, window, song, tmp_path, monkeypatch):
    w = window
    folder = create(qtbot, w, song)
    auto = [note("a", 1.0, 1.8, 64), note("b", 1.9, 2.7, 67), note("c", 2.8, 4.0, 69)]
    # 실제 피치: 두 번째 음을 +60cent 높게 부름
    curve = _curve([(1.0, 1.8, 64), (1.9, 2.7, 67.6), (2.8, 4.0, 69)])
    _set_analysis(w, auto, curve)
    w.doc.move_notes(["a"], 0.0, 1)  # 사용자 편집 (보호 확인용)
    user_before = [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()]
    # 악보는 0초에서 시작 (음원은 1초 뒤에 시작) → 자동 정렬이 오프셋을 찾아야 한다
    path = _midi_file(tmp_path, [(0.0, 0.8, 64), (0.9, 0.8, 67), (1.8, 1.2, 69)], "참조 악보.mid")
    monkeypatch.setattr(dialogs.ScoreImportDialog, "exec", lambda self: 1)
    cc = w.compare
    cc.import_dialog(path)
    qtbot.waitUntil(lambda: w.doc.project.reference is not None, timeout=10_000)
    assert (folder.paths.scores / "참조 악보.mid").is_file()  # 사본을 프로젝트에 둔다
    qtbot.waitUntil(lambda: w.doc.project.reference.alignment.method == "auto", timeout=20_000)
    ref = w.doc.project.reference
    assert not ref.alignment.confirmed  # 미리보기
    assert ref.audio_span(ref.notes[0])[0] == pytest.approx(1.0, abs=0.06)
    assert w.view.roll.reference_preview and len(w.view.roll.reference_notes) == 3
    cc.confirm()
    assert w.doc.project.reference.alignment.confirmed
    qtbot.waitUntil(lambda: cc.report is not None and cc.report.confirmed, timeout=10_000)
    second = cc.report.notes[1]
    assert second.status == "scored" and second.median_cents == pytest.approx(60, abs=3)
    assert cc.flags()[ref.notes[1].id] == "high"
    cc.set_mode("diff")
    assert any(f == "high" for *_x, f in w.view.roll.reference_notes)
    cc.set_mode("ref")
    assert not w.view.roll.items_by_id["a"].isVisible()
    cc.set_mode("both")
    assert w.view.roll.items_by_id["a"].isVisible()
    # 수동 조절은 확정을 푼다
    cc.panel.offset.setValue(cc.panel.offset.value() + 0.1)
    assert not w.doc.project.reference.alignment.confirmed
    w.doc.undo_stack.undo()
    assert w.doc.project.reference.alignment.confirmed
    # 사용자 음표는 그대로
    assert [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()] == user_before
    assert w.save()
    assert folder.paths.reference_json.is_file()
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    assert w.open_project(os.path.join(root, "project.json"))
    qtbot.waitUntil(lambda: w.session.audio is not None and not w.session.busy, timeout=20_000)
    back = w.doc.project.reference
    assert back is not None and back.alignment.confirmed and back.notes == ref.notes
    assert [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()] == user_before
    # 지우기는 실행 취소할 수 있다
    cc.clear()
    assert w.doc.project.reference is None
    w.doc.undo_stack.undo()
    assert w.doc.project.reference is not None
    assert w.errors == []
