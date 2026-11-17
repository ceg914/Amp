"""요구 테스트 1(일정 주파수), 4(무음), 5(시간 정렬 일부) — pYIN + 후처리."""

import numpy as np
import pytest

pytest.importorskip("librosa")

from amp.core.pitch import hz_to_midi
from amp.engine.dsp.pitch import FRAME_RATE, track_pitch
from amp.engine.pitch_post import build_curve, clean_midi, energy_gate, voiced_runs
from tests.audio_synth import SR, sung


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
