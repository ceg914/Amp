import numpy as np
import pytest

from amp.core.errors import EditError
from amp.core.model import AnalysisResult, ModelInfo, PitchCurve, Project, Score
from amp.core.pitch import hz_to_midi, midi_to_hz, note_name
from amp.core.tempo import TempoMap
from tests.conftest import note


def test_note_validation():
    with pytest.raises(EditError):
        note("a", 1.0, 1.0)
    with pytest.raises(EditError):
        note("a", 0.0, 1.0, pitch=128)


def test_pitch_conversions():
    assert float(midi_to_hz(69)) == pytest.approx(440.0)
    assert hz_to_midi(np.array([440.0, 0.0, np.nan]))[0] == pytest.approx(69.0)
    assert np.isnan(hz_to_midi(np.array([0.0, np.nan]))).all()
    assert note_name(60) == "C4" and note_name(61) == "C#4"


def test_score_replace_is_atomic():
    s = Score(notes=[note("a", 0, 1), note("b", 1, 2)])
    with pytest.raises(EditError):
        s.replace_notes(["a", "missing"], [])
    assert len(s) == 2
    with pytest.raises(EditError):
        s.replace_notes([], [note("b", 3, 4)])  # id 중복
    s.replace_notes(["a"], [note("a", 0.5, 1.0)])
    assert s.get("a").start == 0.5
    assert [n.id for n in s.notes()] == ["a", "b"]


def test_score_from_analysis_records_origin_and_keeps_original():
    a = AnalysisResult(1.0, TempoMap.constant(100), (note("x", 0, 1),))
    s = Score.from_analysis(a)
    assert s.get("x").origin_ids == ("x",)
    s.replace_notes(["x"], [s.get("x").with_(pitch=62)])
    assert a.notes[0].pitch == 60  # 원본은 그대로


def test_note_status():
    a = AnalysisResult(2.0, TempoMap.constant(100), (note("x", 0, 1), note("y", 1, 2)))
    p = Project(analysis=a, score=Score.from_analysis(a))
    x = p.score.get("x")
    assert p.note_status(x) == "original"
    assert p.note_status(x.with_(end=0.9)) == "modified"
    assert p.note_status(note("z", 0, 1)) == "user"


def test_pitch_curve_ranges_and_readonly():
    c = PitchCurve(0.0, 100.0, np.full(1000, 220.0), np.ones(1000))
    assert c.end_time == pytest.approx(10.0)
    assert c.frame_range(1.0, 2.0) == (100, 201)
    assert c.frame_range(-5, 50) == (0, 1000)
    with pytest.raises(ValueError):
        c.f0_hz[0] = 1.0
    with pytest.raises(ValueError):
        PitchCurve(0.0, 100.0, np.ones(3), np.ones(4))


def test_placeholder_flag():
    a = AnalysisResult(1.0, TempoMap.constant(100), (), models={"pitch": ModelInfo("sample", placeholder=True)})
    assert a.is_placeholder
