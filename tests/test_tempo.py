import pytest

from amp.core.errors import EditError
from amp.core.tempo import TempoMap, TempoPoint


def test_constant_tempo_roundtrip():
    t = TempoMap.constant(120)
    assert t.beat_at(1.0) == pytest.approx(2.0)
    assert t.time_at(4.0) == pytest.approx(2.0)
    assert t.bar_beat(2.25) == (2, pytest.approx(1.5))


def test_first_beat_offset():
    t = TempoMap.constant(60, first_beat=0.5)
    assert t.beat_at(0.5) == pytest.approx(0.0)
    assert t.beat_at(0.0) == pytest.approx(-0.5)
    assert t.time_at(2.0) == pytest.approx(2.5)


def test_piecewise_tempo():
    t = TempoMap((TempoPoint(0.0, 60), TempoPoint(4.0, 120)))
    assert t.beat_at(4.0) == pytest.approx(4.0)
    assert t.beat_at(5.0) == pytest.approx(6.0)
    assert t.time_at(6.0) == pytest.approx(5.0)
    for b in (0.5, 3.9, 4.0, 7.3):
        assert t.beat_at(t.time_at(b)) == pytest.approx(b)


def test_snap_and_grid():
    t = TempoMap.constant(120)  # 1박 = 0.5초
    assert t.snap(0.37, 0.25) == pytest.approx(0.375)
    assert t.snap(0.37, 0) == 0.37
    assert t.grid_beats(0.0, 1.0, 1.0) == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("bpm", [0, 5, 1000, float("nan")])
def test_invalid_bpm(bpm):
    with pytest.raises(EditError):
        TempoMap.constant(bpm)


def test_with_initial_bpm_keeps_other_points():
    t = TempoMap((TempoPoint(0.0, 60), TempoPoint(4.0, 120)))
    t2 = t.with_initial_bpm(90)
    assert t2.points[0].bpm == 90 and t2.points[1].bpm == 120
