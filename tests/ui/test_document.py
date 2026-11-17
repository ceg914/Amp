import pytest

from amp.app.document import ProjectDocument
from amp.core.errors import EditError
from amp.core.model import AnalysisResult, LyricsTrack, Syllable
from amp.core.tempo import TempoMap
from tests.conftest import note


@pytest.fixture
def doc(qapp):
    d = ProjectDocument()
    result = AnalysisResult(
        4.0,
        TempoMap.constant(100),
        (note("a", 0.0, 1.0, 60, syllable_id="s1"), note("b", 1.0, 2.0, 62, syllable_id="s2")),
        lyrics=LyricsTrack("ko", (Syllable("s1", "가"), Syllable("s2", "나"))),
    )
    d.set_analysis(result)
    d.mark_saved("/tmp/x.ampproj")
    return d


def test_move_undo_redo(doc):
    events = []
    doc.notesReplaced.connect(lambda r, a: events.append((r, [n.id for n in a])))
    assert doc.move_notes(["a"], 0.5, 2)
    assert doc.score.get("a").pitch == 62 and doc.score.get("a").start == 0.5
    assert doc.is_dirty
    doc.undo_stack.undo()
    assert doc.score.get("a").pitch == 60
    assert not doc.is_dirty
    doc.undo_stack.redo()
    assert doc.score.get("a").start == 0.5
    assert events[0] == (["a"], ["a"])


def test_noop_move_does_not_push(doc):
    assert not doc.move_notes(["a"], 0.0, 0)
    assert doc.undo_stack.count() == 0


def test_split_merge_delete_add(doc):
    left, right = doc.split_note("a", 0.4)
    assert len(doc.score) == 3
    assert doc.score.get(left).syllable_id == doc.score.get(right).syllable_id == "s1"
    merged = doc.merge_notes([left, right])
    assert doc.score.get(merged).end == 1.0
    doc.delete_notes([merged])
    assert "a" not in doc.score and len(doc.score) == 1
    nid = doc.add_note(2.0, 2.5, 70)
    assert doc.note_status(doc.score.get(nid)) == "user"
    for _ in range(4):
        doc.undo_stack.undo()
    assert sorted(n.id for n in doc.score.notes()) == ["a", "b"]


def test_split_at_requires_inside(doc):
    with pytest.raises(EditError):
        doc.split_notes_at(["a", "b"], 3.5)
    assert doc.split_notes_at(["a", "b"], 0.5) == 1


def test_selection_hint_after_undo(doc):
    selected = []
    doc.selectionRequested.connect(selected.append)
    left, right = doc.split_note("a", 0.5)
    assert selected[-1] == [left, right]
    doc.undo_stack.undo()
    assert selected[-1] == ["a"]


def test_revert_to_analysis_is_undoable(doc):
    doc.move_notes(["a", "b"], 0.0, 5)
    doc.set_bpm(140)
    doc.revert_to_analysis()
    assert doc.score.get("a").pitch == 60 and doc.score.tempo.initial_bpm == 100
    assert doc.note_status(doc.score.get("a")) == "original"
    doc.undo_stack.undo()
    assert doc.score.get("a").pitch == 65 and doc.score.tempo.initial_bpm == 140


def test_bpm_changes_merge(doc):
    doc.set_bpm(101)
    doc.set_bpm(102)
    doc.set_bpm(103)
    assert doc.undo_stack.count() == 1
    doc.undo_stack.undo()
    assert doc.score.tempo.initial_bpm == 100


def test_set_analysis_clears_history_and_marks_dirty(doc):
    doc.move_notes(["a"], 0.1, 0)
    doc.set_analysis(doc.project.analysis)
    assert doc.undo_stack.count() == 0
    assert doc.is_dirty
    assert doc.score.get("a").start == 0.0


def test_revision_increases(doc):
    r = doc.revision
    doc.move_notes(["a"], 0.1, 0)
    doc.undo_stack.undo()
    assert doc.revision == r + 2
