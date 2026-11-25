"""v0.6 교정 제안 탭: 찾기(음표 불변), 근거 표시, 미리보기 그림, 여러 개 적용·실행 취소 한 번, 사용자 음표 보호,
가사 연결 유지, 무시·숨김, 지난 제안, 저장·다시 열기.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from amp.app.document import ProjectDocument
from amp.compare import suggestions as S
from amp.core.model import AnalysisResult, Note, PitchCurve
from amp.core.pitch import midi_to_hz
from amp.core.tempo import TempoMap
from amp.lyrics import linking, ops
from tests.ui.test_main_window import create

FR = 100.0


def _analysis(duration=6.0):
    """곡선: 1.0~2.0 65 (음표 64 → 음높이 제안), 3.0~3.5 62·3.5~4.0 64 (한 음표 62 → 나누기 제안)."""
    n = int(duration * FR)
    midi = np.full(n, np.nan)
    midi[100:200] = 65.0
    midi[300:350] = 62.0
    midi[350:400] = 64.0
    f0 = np.where(np.isfinite(midi), midi_to_hz(np.nan_to_num(midi, nan=60)), np.nan).astype(np.float32)
    curve = PitchCurve(0.0, FR, f0, np.where(np.isfinite(midi), 0.9, 0.0).astype(np.float32))
    notes = (Note("p", 1.0, 2.0, 64), Note("s", 3.0, 4.0, 62))
    return AnalysisResult(duration=duration, tempo=TempoMap.constant(120), notes=notes, pitch_curve=curve,
                          onset_envelope=np.zeros(n, np.float32))


@pytest.fixture
def doc():
    d = ProjectDocument()
    d.set_analysis(_analysis())
    st = ops.apply_text(d.lyrics, "가 나")
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {ids[0]: (1.0, 2.0), ids[1]: (3.0, 4.0)}, source="model")
    st = linking.auto_link(st, d.score.notes())
    d.push_lyrics(st, "가사")
    d.undo_stack.clear()
    return d


def test_apply_two_suggestions_in_one_undo_step_keeps_links(doc):
    sugs = S.detect(doc.project)
    kinds = {s.suggestion_type for s in sugs}
    assert {"pitch", "split"} <= kinds
    assert [n.pitch for n in doc.score.notes()] == [64, 62]  # 찾기만으로는 바뀌지 않는다
    doc.set_suggestions(S.SuggestionState(tuple(sugs)))
    ids = [s.suggestion_id for s in sugs if s.suggestion_type in ("pitch", "split")]
    before_links = {sid: [l.note_id for l in v] for sid, v in doc.lyrics.alignment.links.items()}
    n0 = doc.undo_stack.count()
    applied, skipped = doc.apply_suggestions(ids)
    assert applied == 2 and not skipped
    notes = sorted(doc.score.notes(), key=lambda n: n.start)
    assert [n.pitch for n in notes] == [65, 62, 64]
    assert all(doc.note_status(n) != "original" for n in notes)  # 적용한 음표는 사용자 수정 음표 (재분석 보호)
    links = doc.lyrics.alignment.links
    assert all(links.get(sid) for sid in before_links)  # 두 음절 모두 연결 유지
    assert all(s.status == S.ACCEPTED for s in doc.project.suggestions.items if s.suggestion_id in ids)
    assert doc.undo_stack.count() == n0 + 1
    doc.undo_stack.undo()  # 한 번에 되돌림
    assert [n.pitch for n in sorted(doc.score.notes(), key=lambda n: n.start)] == [64, 62]
    assert {sid: [l.note_id for l in v] for sid, v in doc.lyrics.alignment.links.items()} == before_links
    assert all(s.status == S.PENDING for s in doc.project.suggestions.items)


def test_user_edit_makes_suggestion_outdated_and_skipped(doc):
    sugs = S.detect(doc.project)
    doc.set_suggestions(S.SuggestionState(tuple(sugs)))
    pitch = next(s for s in sugs if s.suggestion_type == "pitch")
    doc.move_notes(["p"], 0.05, 0)  # 사용자가 먼저 고침
    applied, skipped = doc.apply_suggestions([pitch.suggestion_id])
    assert applied == 0 and skipped
    assert all(n.pitch != 65 for n in doc.score.notes())
    n = doc.undo_stack.count()
    doc.apply_suggestions([pitch.suggestion_id])
    assert doc.undo_stack.count() == n  # 빈 실행 취소 기록을 만들지 않는다


def test_panel_detect_preview_apply_reject_and_reopen(qtbot, window, song, monkeypatch):
    w = window
    folder = create(qtbot, w, song)
    w.doc.set_analysis(dataclasses.replace(_analysis(5.0), duration=5.0))
    sc = w.suggest
    w.show_side_panel(sc.panel)
    n = sc.detect()
    assert n >= 2 and sc.panel.table.rowCount() == n
    assert [x.pitch for x in w.doc.score.notes()] == [64, 62]
    header = sc.panel.table.horizontalHeaderItem(3)
    assert header.text() == "근거 점수" and "확률이 아닙니다" in header.toolTip()
    sc.go(1)
    assert "근거 점수" in sc.panel.evidence.text() and "확률 아님" in sc.panel.evidence.text()
    assert w.view.roll.preview.get("added")  # 미리보기 그림
    sc.panel.preview.setChecked(False)
    assert not w.view.roll.preview
    sc.panel.preview.setChecked(True)
    # 첫 제안 무시, 나머지 적용
    first = sc.panel.selected_ids()[0]
    sc.on_action("reject")
    assert w.doc.project.suggestions.get(first).status == S.REJECTED
    assert first not in [s.suggestion_id for s in sc.rows]
    sc.panel.table.selectAll()
    sc.on_action("apply")
    assert any(s.status == S.ACCEPTED for s in w.doc.project.suggestions.items)
    # 다시 찾아도 무시한 제안은 대기로 돌아오지 않는다
    sc.detect()
    assert w.doc.project.suggestions.get(first).status == S.REJECTED
    assert w.save()
    assert folder.paths.suggestions_json.exists()
    from amp.project import workspace

    reopened, report = workspace.open_folder(folder.root)[0].load_project()
    assert not report.warnings
    assert reopened.suggestions.get(first).status == S.REJECTED
    assert {s.suggestion_id: s.status for s in reopened.suggestions.items} == \
        {s.suggestion_id: s.status for s in S.refresh(w.doc.project.suggestions, w.doc.project).items}
