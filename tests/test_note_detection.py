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


# --- STEP 3.5: 접근음(끌어올림) 흡수 --------------------------------------------------------------

def _scoop(lead_hold=6, rise=8, low=62, high=64, start=50, target=60, n=300):
    """프레이즈 시작: low에 lead_hold프레임 머문 뒤 rise프레임 동안 high로 올라가 target프레임 유지."""
    m = np.full(n, np.nan)
    a = start
    m[a:a + lead_hold] = low
    m[a + lead_hold:a + lead_hold + rise] = np.linspace(low, high, rise + 1)[1:]
    m[a + lead_hold + rise:a + lead_hold + rise + target] = high
    return m


def _onset(n=300, peaks=()):
    o = np.zeros(n)
    for k, v in peaks:
        o[k] = v
    return o


def test_approach_scoop_merges_into_target_and_keeps_curve():
    m = _scoop()
    c = curve(m)
    f0_before, conf_before = c.f0_hz.copy(), c.confidence.copy()
    off = detect_notes(c, _onset(peaks=[(50, 8.0)]), params(merge_approach=False))
    on = detect_notes(c, _onset(peaks=[(50, 8.0)]), params())
    assert [n.pitch for n in off] == [62, 64]  # 이전 동작: 끌어올림이 따로 음표
    assert [(round(n.start, 2), n.pitch) for n in on] == [(0.5, 64)]
    assert abs(on[0].end - off[-1].end) < 1e-9
    # 피치 곡선은 바꾸지 않는다
    np.testing.assert_array_equal(c.f0_hz, f0_before)
    np.testing.assert_array_equal(c.confidence, conf_before)


def test_approach_needs_articulation_evidence():
    m = _scoop()
    assert len(detect_notes(curve(m), None, params())) == 2  # 재발음 지표가 없으면 합치지 않는다


def test_reattacked_target_is_not_an_approach():
    m = _scoop()
    on = detect_notes(curve(m), _onset(peaks=[(50, 8.0), (64, 5.0)]), params())
    assert [n.pitch for n in on] == [62, 64]  # 뒤 음 시작에 재발음 → 따로 부른 장식음


def test_flat_short_note_before_step_is_kept():
    m = _scoop(lead_hold=18, rise=1)  # 180ms 평평하게 머문 뒤 바로 위 음 (한 모음 안의 실제 음)
    on = detect_notes(curve(m), _onset(peaks=[(50, 8.0)]), params())
    assert [n.pitch for n in on] == [62, 64]


def test_fall_and_mid_phrase_are_unchanged():
    fall = _scoop(low=66, high=64)  # 위에서 내려오며 시작
    mid = np.full(300, np.nan)
    mid[30:80] = 60  # 앞 음에서 이어지는 중간 음 (노래 시작 자리가 아님)
    mid[80:150] = _scoop(start=0, n=70)[0:70]
    for m in (fall, mid):
        o = _onset(peaks=[(30, 8.0), (50, 8.0)])
        assert summary(detect_notes(curve(m), o, params())) == summary(detect_notes(curve(m), o, params(merge_approach=False)))


def test_long_approach_or_big_flat_interval_is_kept():
    long_lead = _scoop(lead_hold=20, rise=10)  # 300ms
    assert len(detect_notes(curve(long_lead), _onset(peaks=[(50, 8.0)]), params())) == 2
    big = _scoop(lead_hold=10, rise=2, low=59, high=64)  # 5반음 아래에서 거의 머물다 뛰어오름
    assert len(detect_notes(curve(big), _onset(peaks=[(50, 8.0)]), params())) == 2


def test_syllable_start_on_target_blocks_merge():
    from dataclasses import replace

    m = _scoop()
    p = replace(params(), syllable_starts=(0.5, 0.64))  # 뒤 음 시작(0.64초)에서 다음 음절이 시작
    assert [n.pitch for n in detect_notes(curve(m), _onset(peaks=[(50, 8.0)]), p)] == [62, 64]
    p2 = replace(params(), syllable_starts=(0.5,))  # 음절은 앞에서 한 번만 시작
    assert [n.pitch for n in detect_notes(curve(m), _onset(peaks=[(50, 8.0)]), p2)] == [64]


def test_merge_approach_setting_round_trip():
    s = AnalysisSettings(merge_approach=False)
    assert AnalysisSettings.from_dict(s.to_dict()).merge_approach is False
    assert AnalysisSettings.from_dict({"sensitivity": "standard"}).merge_approach is True  # 이전 프로젝트 설정
    assert s.note_params().merge_approach is False
