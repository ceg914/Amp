import pytest

from amp.core import editing
from amp.core.errors import EditError
from tests.conftest import note


def test_move_clamps_to_zero_and_midi_range():
    notes = [note("a", 0.2, 0.5, 126), note("b", 1.0, 2.0, 60)]
    moved = editing.move_notes(notes, -1.0, 5)
    assert moved[0].start == pytest.approx(0.0)
    assert moved[0].pitch == 127 and moved[1].pitch == 61
    assert moved[1].start == pytest.approx(0.8)


def test_resize_keeps_min_duration():
    notes = [note("a", 0.0, 0.1), note("b", 1.0, 2.0)]
    out = editing.resize_notes(notes, "end", -1.0)
    assert min(n.duration for n in out) == pytest.approx(editing.MIN_NOTE_DURATION)
    out = editing.resize_notes(notes, "start", 5.0)
    assert out[0].duration == pytest.approx(editing.MIN_NOTE_DURATION)
    out = editing.resize_notes([note("c", 0.5, 1.0)], "start", -2.0)
    assert out[0].start == 0.0


def test_split_inherits_syllable_and_origin():
    n = note("a", 0.0, 1.0, syllable_id="s1", origin_ids=("o1",))
    left, right = editing.split_note(n, 0.4, ids=("l", "r"))
    assert (left.end, right.start) == (0.4, 0.4)
    assert left.syllable_id == right.syllable_id == "s1"
    assert left.origin_ids == right.origin_ids == ("o1",)
    with pytest.raises(EditError):
        editing.split_note(n, 0.01)
    with pytest.raises(EditError):
        editing.split_note(n, 1.5)


def test_merge_rules():
    a = note("a", 0.0, 0.3, 60, syllable_id=None, origin_ids=("o1",), confidence=0.9)
    b = note("b", 0.3, 1.0, 62, syllable_id="s2", origin_ids=("o2",), confidence=0.5)
    m = editing.merge_notes([b, a], note_id="m")
    assert (m.start, m.end) == (0.0, 1.0)
    assert m.pitch == 62  # 더 길게 유지된 음
    assert m.syllable_id == "s2"
    assert m.origin_ids == ("o1", "o2")
    assert m.confidence == 0.5
    with pytest.raises(EditError):
        editing.merge_notes([a])


def test_merge_tie_prefers_earlier_pitch():
    m = editing.merge_notes([note("a", 0, 0.5, 60), note("b", 0.5, 1.0, 64)])
    assert m.pitch == 60


def test_create_note_validation():
    with pytest.raises(EditError):
        editing.create_note(-0.1, 0.5, 60)
    with pytest.raises(EditError):
        editing.create_note(0.0, 0.01, 60)
    assert editing.create_note(0.0, 0.5, 60).pitch == 60
