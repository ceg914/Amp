"""v0.7.0 Feature D 비브라토 (명세 7.6): 합성 피치 곡선과 합성 노래(pYIN 경유)에서 속도 오차 0.3Hz·폭 오차 10센트 이내.
합성 자료의 목표이며 실제 가창의 정확도가 아니다."""

from __future__ import annotations

import numpy as np
import pytest

from amp.engine.acoustics.vibrato import VibratoSettings, analyze_curve, summary

FR = 100.0


def curve(dur, base=64.0, rate=5.5, ext=50.0, start=0.0, stop=None, drift=0.0, rate_end=None, ext_end=None):
    t = np.arange(int(dur * FR)) / FR
    r = np.full(len(t), rate) if rate_end is None else np.linspace(rate, rate_end, len(t))
    e = np.full(len(t), ext) if ext_end is None else np.linspace(ext, ext_end, len(t))
    ph = 2 * np.pi * np.cumsum(r) / FR
    on = (t >= start) & (t < (stop if stop is not None else dur + 1))
    midi = base + drift / 100 * t / dur + np.where(on, e * np.sin(ph), 0.0) / 100
    return 440 * 2 ** ((midi - 69) / 12)


def one(segs):
    ok = [s for s in segs if s.status == "completed"]
    assert len(ok) == 1, [(s.start, s.end, s.status) for s in segs]
    return ok[0]


@pytest.mark.parametrize("rate,ext", [(5.5, 50.0), (4.0, 30.0), (6.5, 80.0), (8.0, 35.0), (3.5, 40.0), (5.0, 15.0)])
def test_constant_vibrato_rate_and_extent(rate, ext):
    s = one(analyze_curve(curve(2.5, rate=rate, ext=ext)))
    assert abs(s.rate_hz - rate) <= 0.3
    assert abs(s.extent_cents - ext) <= 10  # 한쪽 진폭
    assert abs(s.extent_pp_cents - 2 * ext) <= 20  # 봉우리 간 폭 = 2 × 한쪽 진폭
    assert s.regularity > 0.8 and s.cycles >= 2
    assert abs(s.center_midi - 64.0) < 0.1


def test_accelerating_and_growing():
    s = one(analyze_curve(curve(2.5, rate=4.0, rate_end=7.0)))
    assert s.rate_slope_hz_per_s == pytest.approx(1.2, abs=0.3) and abs(s.rate_hz - 5.5) <= 0.3
    s = one(analyze_curve(curve(2.5, ext=20, ext_end=80)))
    assert s.extent_slope_cents_per_s == pytest.approx(24, abs=6) and abs(s.extent_cents - 50) <= 10
    assert s.cycle_extents[0] < s.cycle_extents[-1]


def test_drifting_center_and_onset_delay():
    s = one(analyze_curve(curve(2.0, drift=100)))  # 2초 동안 1반음 올라가며 떠는 소리
    assert abs(s.rate_hz - 5.5) <= 0.3 and abs(s.extent_cents - 50) <= 10
    s = one(analyze_curve(curve(2.0, start=0.5)))
    assert s.onset_delay_s == pytest.approx(0.5, abs=0.1)
    s = one(analyze_curve(curve(2.0, start=0.5), onsets=[0.3]))
    assert s.onset_delay_s == pytest.approx(0.2, abs=0.1)  # 음 시작부터


def test_interrupted_vibrato_gives_two_segments():
    f = curve(3.0, stop=1.2)
    g = curve(3.0, start=1.8)
    t = np.arange(len(f)) / FR
    x = np.where(t < 1.5, f, g)
    segs = [s for s in analyze_curve(x) if s.status == "completed"]
    assert len(segs) == 2 and segs[0].end < 1.35 and segs[1].start > 1.65


def test_no_false_vibrato():
    rng = np.random.default_rng(0)
    flat = 440 * 2 ** ((64 + rng.normal(0, 0.03, 300) - 69) / 12)  # 3센트 흔들림
    assert not [s for s in analyze_curve(flat) if s.status == "completed"]
    t = np.arange(300) / FR
    porta = 64 + np.clip((t - 1.0) / 0.2, 0, 1) * 3  # 200ms 동안 3반음 미끄러짐
    assert not analyze_curve(440 * 2 ** ((porta - 69) / 12))
    steps = 64 + 2 * (np.floor(t * 4) % 2)  # 음 바뀜(250ms마다 2반음)
    assert not analyze_curve(440 * 2 ** ((steps - 69) / 12))
    octave = 440 * 2 ** ((64 - 69) / 12) * np.where((t > 1.0) & (t < 1.05), 2.0, 1.0)  # 옥타브 오류 튐
    assert not analyze_curve(octave)
    short = curve(0.3)  # 너무 짧음 (2주기 미만)
    assert not [s for s in analyze_curve(short) if s.status == "completed"]


def test_trill_and_irregular_marked_uncertain():
    t = np.arange(300) / FR
    trill = 64 + 2.0 * (np.sin(2 * np.pi * 6.5 * t) > 0)  # 2반음을 6.5Hz로 오감
    segs = analyze_curve(440 * 2 ** ((trill - 69) / 12), settings=VibratoSettings(jump_cents=400))
    assert segs and all(s.status == "uncertain" for s in segs) and "트릴" in segs[0].reason
    assert summary(segs)["completed"] == 0


def test_end_to_end_synthetic_voice_through_pyin():
    """합성 노래(포먼트) → pYIN → 비브라토. 합성 자료 기준 목표: 속도 0.3Hz, 폭 10센트."""
    from amp.devtools.synth import SR, sung
    from amp.engine.dsp.pitch import track_pitch

    y = sung([(0.5, 3.0, 67)], 3.5, vibrato_cents=45, vibrato_hz=5.8)
    frames = track_pitch(y, SR)
    s = max(analyze_curve(frames.f0_hz, frames.frame_rate), key=lambda s: s.duration)
    assert s.status == "completed"
    assert abs(s.rate_hz - 5.8) <= 0.3 and abs(s.extent_cents - 45) <= 10
