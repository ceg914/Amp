"""요구 테스트 2(음높이 연속), 3(비브라토), 4(무음·잡음) — 음표 검출."""

import numpy as np
import pytest

from amp.core.model import PitchCurve
from amp.core.pitch import midi_to_hz
from amp.engine.notes import detect_notes
from amp.engine.settings import AnalysisSettings

FR = 100.0


def curve(midi, conf=0.9):
    midi = np.asarray(midi, dtype=float)
    f0 = np.where(np.isfinite(midi), midi_to_hz(np.nan_to_num(midi)), np.nan)
    return PitchCurve(0.0, FR, f0, np.where(np.isfinite(midi), conf, 0.0))


def params(sens="standard", **kw):
    from dataclasses import replace

    return replace(AnalysisSettings.for_sensitivity(sens), **kw).note_params()


def summary(notes):
    return [(round(n.start, 2), round(n.end, 2), n.pitch) for n in notes]


def test_sequence_with_gaps_and_legato():
    m = np.full(500, np.nan)
    m[50:120] = 60
    m[120:200] = 64  # 레가토
    m[210:300] = 67  # 100ms 쉼 후
    m[300:400] = 72
    notes = detect_notes(curve(m), None, params())
    got = summary(notes)
    assert [p for _, _, p in got] == [60, 64, 67, 72]
    for (s, e, _), (es, ee) in zip(got, [(0.5, 1.2), (1.2, 2.0), (2.1, 3.0), (3.0, 4.0)]):
        assert abs(s - es) <= 0.04 and abs(e - ee) <= 0.04


def test_portamento_not_over_split():
    m = np.full(400, np.nan)
    m[50:150] = 60
    m[150:180] = np.linspace(60, 67, 30)  # 300ms 동안 완전5도 미끄러짐
    m[180:300] = 67
    for sens in ("simple", "standard", "precise"):
        notes = detect_notes(curve(m), None, params(sens))
        assert [n.pitch for n in notes] == [60, 67], sens


@pytest.mark.parametrize("sens", ["simple", "standard", "precise"])
def test_vibrato_stays_one_note(sens):
    t = np.arange(400) / FR
    m = np.full(400, np.nan)
    m[50:350] = 67 + 0.6 * np.sin(2 * np.pi * 5.5 * t[50:350])  # ±60cent
    notes = detect_notes(curve(m), None, params(sens))
    assert len(notes) == 1 and notes[0].pitch == 67
    assert notes[0].vibrato is not None
    assert 4.5 < notes[0].vibrato.rate_hz < 6.5
    assert 40 < notes[0].vibrato.extent_cents < 75


def test_new_held_pitch_splits_but_min_length_respected():
    m = np.full(400, np.nan)
    m[50:150] = 60
    m[150:156] = 62  # 60ms → 최소 길이(100ms) 미만
    m[156:250] = 60
    m[250:350] = 62  # 1초 유지 → 별도 음표
    notes = detect_notes(curve(m), None, params("simple"))
    assert [n.pitch for n in notes] == [60, 62]
    assert all(n.duration >= 0.1 for n in notes)


def test_ornament_by_confidence():
    m = np.full(300, np.nan)
    m[50:110] = 72
    m[110:115] = 74  # 50ms 장식음
    m[115:200] = 72
    high = detect_notes(curve(m, 0.95), None, params("standard"))
    assert [(n.pitch, n.ornament) for n in high] == [(72, False), (74, True), (72, False)]
    low = detect_notes(curve(m, 0.3), None, params("standard"))
    assert [n.pitch for n in low] == [72]  # 신뢰도가 낮으면 장식음으로 보지 않는다
    simple = detect_notes(curve(m, 0.95), None, params("simple"))
    assert [n.pitch for n in simple] == [72]


def test_silence_and_noise_produce_no_notes():
    assert detect_notes(curve(np.full(500, np.nan)), None, params()) == []
    rng = np.random.default_rng(0)
    m = np.full(500, np.nan)
    idx = rng.choice(500, 60, replace=False)
    m[idx] = rng.uniform(40, 80, 60)  # 흩어진 잡음 프레임
    assert detect_notes(curve(m), None, params()) == []


def test_breath_like_low_confidence_frames_ignored():
    m = np.full(300, np.nan)
    m[50:250] = 60
    notes = detect_notes(curve(m, conf=0.005), None, params("standard"))  # voiced_threshold 0.02 미만
    assert notes == []


def test_pitch_range_filter():
    m = np.full(300, np.nan)
    m[50:150] = 30  # 범위 밖(F#1)
    m[160:260] = 60
    notes = detect_notes(curve(m), None, params(pitch_low=36, pitch_high=84))
    assert [n.pitch for n in notes] == [60]


def test_onset_splits_repeated_same_pitch():
    m = np.full(300, np.nan)
    m[50:250] = 62
    onset = np.ones(300)
    onset[150] = 50.0  # 같은 음 재발음
    notes = detect_notes(curve(m), onset, params("standard"))
    assert [(round(n.start, 2), n.pitch) for n in notes] == [(0.5, 62), (1.5, 62)]
    simple = detect_notes(curve(m), onset, params("simple"))  # 단순 모드는 온셋으로 나누지 않는다
    assert len(simple) == 1


def test_notes_have_required_fields():
    m = np.full(200, np.nan)
    m[20:150] = 69
    n = detect_notes(curve(m), None, params())[0]
    assert n.id and n.name == "A4" and n.confidence is not None and n.origin_ids == ()


def test_confidence_finite_when_merging_across_unvoiced_gap():
    """실제 곡에서 발견: 짧은 무성 틈을 넘어 같은 음을 합치면 신뢰도가 NaN이 되어 프로젝트 저장이 실패했다."""
    import math

    m = np.full(300, np.nan)
    m[50:120] = 62
    m[125:200] = 62  # 50ms 틈: 표준의 틈 잇기(40ms)보다 길고 같은 음 병합(50ms) 이내
    notes = detect_notes(curve(m), None, params("standard"))
    assert len(notes) == 1 and math.isfinite(notes[0].confidence) and notes[0].confidence > 0.5
