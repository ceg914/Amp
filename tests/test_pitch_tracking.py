"""요구 테스트 1(일정 주파수), 4(무음), 5(시간 정렬 일부) — pYIN + 후처리."""

import numpy as np
import pytest

pytest.importorskip("librosa")

from amp.core.pitch import hz_to_midi
from amp.engine.dsp.pitch import FRAME_RATE, track_pitch
from amp.engine.pitch_post import build_curve, clean_midi, energy_gate, voiced_runs
from amp.devtools.synth import SR, sung


def _cents(a_hz, b_hz):
    return 1200 * np.log2(a_hz / b_hz)


@pytest.mark.parametrize("kind", ["sine", "harmonic"])
def test_constant_frequency(kind):
    t = np.arange(int(SR * 3)) / SR
    if kind == "sine":
        y = 0.3 * np.sin(2 * np.pi * 220 * t)
    else:
        y = sum(0.3 / k * np.sin(2 * np.pi * 220 * k * t) for k in range(1, 8))
    y[t < 1.0] = 0
    f = track_pitch(y.astype(np.float32), SR)
    voiced = np.isfinite(f.f0_hz)
    tone = slice(int(1.1 * FRAME_RATE), int(2.9 * FRAME_RATE))
    assert voiced[tone].mean() > 0.95
    assert abs(_cents(np.nanmedian(f.f0_hz[tone]), 220.0)) < 10
    first = np.argmax(voiced) / FRAME_RATE
    assert abs(first - 1.0) < 0.04  # 시작 시각이 오디오와 맞는다
    assert not voiced[: int(0.9 * FRAME_RATE)].any()  # 무음 구간은 무성


def test_formant_voice_and_silence():
    y = sung([(0.5, 1.5, 60), (2.5, 3.5, 67)], 4.0)
    f = track_pitch(y, SR)
    c = build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate)
    m = c.midi()
    assert abs(np.nanmedian(m[60:140]) - 60) < 0.3
    assert abs(np.nanmedian(m[260:340]) - 67) < 0.3
    assert np.isnan(m[170:240]).all()  # 쉬는 구간은 NaN → 곡선이 이어지지 않는다
    assert len(voiced_runs(np.isfinite(m))) == 2


def test_clean_midi_fixes_octave_errors_and_spikes():
    m = np.full(300, 60.0)
    m[100:104] = 72.0  # 4프레임 옥타브 위 오류
    m[200] = 64.5  # 1프레임 튐
    m[150:153] = np.nan  # 무성 틈은 그대로
    out = clean_midi(m, 100.0)
    assert np.nanmax(np.abs(out[np.isfinite(out)] - 60)) < 0.6
    assert np.isnan(out[150:153]).all()


def test_short_islands_removed_and_energy_gate():
    m = np.full(200, np.nan)
    m[50:52] = 65  # 20ms짜리 조각
    m[100:180] = 62
    out = clean_midi(m, 100.0)
    assert np.isnan(out[50:52]).all() and np.isfinite(out[100:180]).all()
    rms = np.r_[np.full(100, 1e-5), np.full(100, 0.1)]
    gate = energy_gate(rms)
    assert not gate[:100].any() and gate[100:].all()


def test_hz_to_midi_roundtrip():
    assert hz_to_midi(np.array([261.6256]))[0] == pytest.approx(60, abs=1e-3)


def test_articulation_envelope_needs_energy_dip():
    from amp.engine.pitch_post import articulation_envelope

    n = 300
    onset = np.ones(n)
    onset[100] = onset[200] = 10.0
    rms = np.full(n, 0.1)
    rms[195:200] = 0.01  # 200 프레임 직전 20dB 하락 (재발음), 100은 하락 없음 (비브라토 흔들림)
    art = articulation_envelope(onset, rms, 100.0)
    assert art[200] > 8.0
    assert art[100] < 1.0


def test_octave_reference_does_not_cross_long_gaps():
    """실제 곡에서 발견: 쉼 뒤 높은 프레이즈의 첫 프레임이 몇 초 전 낮은 프레이즈 높이에 끌려 한 옥타브 내려갔다."""
    m = np.full(400, np.nan)
    m[0:200] = 67.0  # 낮은 프레이즈 2초
    m[215:229] = 80.8  # 150ms 쉼 뒤 짧은 높은 프레이즈 (115.41초 부근과 같은 모양)
    m[215:217] = (80.0, 80.2)  # 첫 두 프레임은 조금 낮다 → 67과의 차이가 12±1.5 안
    m[245:400] = 67.0
    out = clean_midi(m, 100.0)
    assert np.all(out[215:229] > 79)
    d = np.diff(out)
    assert not np.any(np.isfinite(d) & (np.abs(np.abs(d) - 12) < 1))
    spike = m.copy()
    spike[100:110] = 79.0  # 프레이즈 안의 짧은 옥타브 오류는 여전히 고친다
    assert np.allclose(clean_midi(spike, 100.0)[100:110], 67.0)
