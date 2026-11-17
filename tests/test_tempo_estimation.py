"""BPM 추정 (요구: 자동 검출, 2배·절반 후보)."""

import numpy as np
import pytest

pytest.importorskip("librosa")

from amp.engine.dsp.tempo import estimate_tempo, refine_with_beats
from tests.audio_synth import SR, clicks


@pytest.mark.parametrize("bpm", [96.0, 120.0])
def test_click_track(bpm):
    r = estimate_tempo(clicks(15.0, bpm), SR)
    cands = [c["bpm"] for c in r["candidates"]]
    assert any(abs(c - bpm) < 1.0 for c in cands), r
    assert r["bpm"] in cands
    assert any(abs(c - r["bpm"] * 2) < 0.5 or abs(c - r["bpm"] / 2) < 0.5 for c in cands)
    period = 60.0 / r["bpm"]
    phase = (0.25 % period)
    assert min(abs(r["first_beat"] - phase), period - abs(r["first_beat"] - phase)) < 0.05


def test_short_or_silent_audio_uses_default():
    r = estimate_tempo(np.zeros(SR * 2, np.float32), SR)
    assert r["bpm"] == 120.0 and not r["reliable"] and r["reason"]


def test_refine_with_beats():
    beats = 0.3 + np.arange(20) * 0.5 + np.random.default_rng(0).normal(0, 0.01, 20)
    bpm, first = refine_with_beats(beats, 117.45)
    assert bpm == pytest.approx(120.0, abs=0.5) and first == pytest.approx(0.3, abs=0.02)
    assert refine_with_beats(beats[:5], 120) is None
