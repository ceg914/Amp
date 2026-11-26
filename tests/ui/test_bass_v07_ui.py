"""v0.7.0 베이스 화면: 사용자가 정한 줄(실행 취소·저장), 박자표 변화 입력, 셋잇단·못갖춘마디 설정 저장."""

from __future__ import annotations

from amp.core.model import AnalysisResult, Note
from amp.core.tempo import TempoMap


def _bass_notes(w):
    notes = (Note("b1", 1.0, 1.5, 33), Note("b2", 1.5, 2.0, 35), Note("b3", 2.0, 3.0, 36))
    w.doc.set_tempo(TempoMap.constant(120.0, 1.0))
    w.bass.doc.set_analysis(AnalysisResult(10.0, w.doc.score.tempo, notes))
    return notes


def test_user_string_choice_is_undoable_and_stored(qtbot, window):
    w = window
    _bass_notes(w)
    w.bass.set_active(True)
    w.bass.view.roll.select_ids(["b1"])
    assert w.bass.panel.finger_apply.isEnabled()
    assert next(n for n in w.bass.tab().notes if n.note_id == "b1").string == 0  # 기본: E줄 5프렛
    w.bass.panel.finger_string.setCurrentIndex(w.bass.panel.finger_string.findData(1))
    w.bass.panel.finger_apply.click()
    n = next(n for n in w.bass.tab().notes if n.note_id == "b1")
    assert (n.string, n.fret, n.flags) == (1, 0, ("fixed",))
    assert w.doc.project.bass.fingering == {"b1": 1}
    assert w.bass.doc.undo_stack.undoText().startswith("줄 지정")
    w.bass.doc.undo_stack.undo()
    assert w.bass.fingering == {} and w.doc.project.bass.fingering == {}
    w.bass.doc.undo_stack.redo()
    assert w.bass.fingering == {"b1": 1}
    assert w.bass.apply_fingering(None, ["b1"]) and w.bass.fingering == {}  # 자동으로
    w.bass.set_active(False)
    assert not w.bass.panel.finger_apply.isEnabled()


def test_meter_text_and_new_options_are_stored(qtbot, window):
    w = window
    _bass_notes(w)
    p = w.bass.panel
    p.meter.setText("2:3")
    p.meter.editingFinished.emit()
    assert p.options().meter == ((2, 3),) and w.doc.project.bass.options.meter == ((2, 3),)
    p.meter.setText("두:셋")
    p.meter.editingFinished.emit()
    assert p.meter_error and p.options().meter == ((2, 3),)  # 잘못 적으면 이전 값을 지킨다
    w.bass.refresh_preview()
    assert "박자표 변화 오류" in p.preview.toPlainText()
    p.tuplets.setCurrentIndex(p.tuplets.findData("auto"))
    p.pickup.setCurrentIndex(p.pickup.findData("partial"))
    o = w.doc.project.bass.options
    assert o.tuplets == "auto" and o.pickup == "partial" and not o.riff
    p.set_options(o)
    assert p.meter.text() == "2:3" and p.tuplets.currentData() == "auto"
