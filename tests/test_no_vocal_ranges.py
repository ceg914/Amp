"""보컬 없는 구간: 이 구간에서는 음표를 만들지 않는다 (분리 결과에 섞인 반주 대책)."""

import numpy as np
import pytest

from amp.app.dialogs import format_time, parse_time
from amp.core.model import PitchCurve
from amp.core.pitch import midi_to_hz
from amp.engine.notes import detect_notes
from amp.engine.settings import AnalysisSettings, SettingsError, normalize_ranges


def curve(m, conf=0.9):
    m = np.asarray(m, float)
    f0 = np.where(np.isfinite(m), midi_to_hz(np.nan_to_num(m)), np.nan)
    return PitchCurve(0.0, 100.0, f0, np.where(np.isfinite(m), conf, 0.0))


def test_normalize_sorts_merges_and_validates():
    assert normalize_ranges([(30, 40), (0, 28), (27.5, 29)]) == ((0.0, 29.0), (30.0, 40.0))
    assert normalize_ranges([(0, 5), (5, 8)]) == ((0.0, 8.0),)  # 맞닿으면 합친다
    with pytest.raises(SettingsError):
        normalize_ranges([(5, 3)])
    assert normalize_ranges([(5, 3), ("x", 1), (1, 2)], strict=False) == ((1.0, 2.0),)


def test_settings_roundtrip_and_bad_ranges_do_not_reset_other_settings():
    s = AnalysisSettings(min_note_ms=120, no_vocal_ranges=[(0, 28)])
    d = s.to_dict()
    assert d["no_vocal_ranges"] == ((0.0, 28.0),) or d["no_vocal_ranges"] == [[0.0, 28.0]]
    assert AnalysisSettings.from_dict(d) == s
    bad = AnalysisSettings.from_dict({"min_note_ms": 120, "no_vocal_ranges": [[0, 28], [9, 3], "x"]})
    assert bad.min_note_ms == 120 and bad.no_vocal_ranges == ((0.0, 28.0),)
    assert s.note_params().no_vocal_ranges == ((0.0, 28.0),)
    assert AnalysisSettings.for_sensitivity("precise", s).no_vocal_ranges == ((0.0, 28.0),)


def test_notes_inside_range_removed_and_straddling_note_cut():
    m = np.full(600, np.nan)
    m[50:150] = 60   # 0.5~1.5초: 반주가 섞인 전주라고 지정할 곳
    m[200:400] = 64  # 2.0~4.0초: 보컬. 2.0~2.5초는 구간에 걸친다
    m[450:550] = 67
    base = AnalysisSettings()
    assert [n.pitch for n in detect_notes(curve(m), None, base.note_params())] == [60, 64, 67]
    s = AnalysisSettings(no_vocal_ranges=[(0.0, 1.6), (1.9, 2.5)])
    notes = detect_notes(curve(m), None, s.note_params())
    assert [n.pitch for n in notes] == [64, 67]
    assert notes[0].start >= 2.5 and notes[0].end == pytest.approx(4.0, abs=0.02)


def test_short_range_is_not_bridged_over():
    m = np.full(400, np.nan)
    m[50:350] = 62
    notes = detect_notes(curve(m), None, AnalysisSettings(no_vocal_ranges=[(1.5, 1.52)]).note_params())
    assert len(notes) == 2 and notes[0].end <= 1.5 + 1e-6 and notes[1].start >= 1.52 - 1e-6


def test_time_text_helpers():
    assert parse_time("0:28.5") == 28.5 and parse_time("75") == 75.0 and parse_time("1:02") == 62.0
    assert format_time(75.25) == "1:15.25" and parse_time(format_time(75.25)) == 75.25
    for bad in ("", "x", "1:75", "-3"):
        with pytest.raises(ValueError):
            parse_time(bad)
