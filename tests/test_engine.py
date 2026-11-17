import threading
from collections import Counter

import numpy as np
import pytest

from amp.audio.data import AudioData
from amp.core.errors import AnalysisCancelled, AnalysisError
from amp.core.model import ModelInfo, PitchCurve
from amp.engine.backends.sample import build_plan
from amp.engine.context import CancelToken
from amp.engine.interfaces import PitchTracker
from amp.engine.pipeline import AnalysisPipeline, PipelineConfig
from amp.engine.registry import BackendRegistry, StageKind, default_registry


@pytest.fixture
def mix():
    return AudioData(np.zeros((22050 * 30, 2), np.float32), 22050)


def test_sample_pipeline_end_to_end(mix):
    seen = []
    result = AnalysisPipeline(default_registry()).run(mix, lambda f, m: seen.append(f), audio_fingerprint="fp")
    assert result.is_placeholder
    assert result.audio_fingerprint == "fp"
    assert result.notes and all(0 <= n.start < n.end <= 30 for n in result.notes)
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert set(result.models) == {k.value for k in StageKind}
    assert len(result.pitch_curve) == 3000
    # 모든 음표가 음절에 연결되고, 일부 음절은 여러 음표(멜리스마)를 가진다
    counts = Counter(n.syllable_id for n in result.notes)
    assert None not in counts
    assert max(counts.values()) >= 2
    syl = {s.id: s for s in result.lyrics.syllables}
    for sid, cnt in counts.items():
        linked = [n for n in result.notes if n.syllable_id == sid]
        assert syl[sid].start == min(n.start for n in linked)


def test_pitch_curve_follows_notes(mix):
    result = AnalysisPipeline(default_registry()).run(mix)
    curve = result.pitch_curve
    midi = curve.midi()
    for n in result.notes[:20]:
        i0, i1 = curve.frame_range(n.start + 0.1, n.end - 0.05)
        seg = midi[i0:i1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            assert abs(np.median(seg) - n.pitch) < 0.5


def test_plan_is_deterministic():
    assert build_plan(60.0) == build_plan(60.0)
    assert build_plan(0.5).notes == ()


def test_skip_lyrics(mix):
    result = AnalysisPipeline(default_registry(), PipelineConfig(lyrics=None, aligner=None)).run(mix)
    assert result.lyrics is None
    assert all(n.syllable_id is None for n in result.notes)


def test_cancel(mix):
    token = CancelToken()
    token.cancel()
    with pytest.raises(AnalysisCancelled):
        AnalysisPipeline(default_registry()).run(mix, token=token)


def test_cancel_from_progress(mix):
    token = CancelToken()

    def progress(f, m):
        if f > 0.3:
            token.cancel()

    with pytest.raises(AnalysisCancelled):
        AnalysisPipeline(default_registry()).run(mix, progress, token)


class BrokenTracker(PitchTracker):
    @property
    def info(self):
        return ModelInfo("broken")

    def track(self, vocals, ctx):
        raise RuntimeError("model crashed")


def test_backend_error_is_wrapped(mix):
    reg = default_registry()
    reg.register(StageKind.PITCH, "broken", BrokenTracker)
    with pytest.raises(AnalysisError, match="model crashed"):
        AnalysisPipeline(reg, PipelineConfig(pitch="broken")).run(mix)


def test_registry_checks():
    reg = BackendRegistry()
    reg.register(StageKind.PITCH, "x", BrokenTracker)
    with pytest.raises(ValueError):
        reg.register(StageKind.PITCH, "x", BrokenTracker)
    with pytest.raises(KeyError):
        reg.create(StageKind.PITCH, "nope")
    reg.register(StageKind.TEMPO, "wrong", BrokenTracker)
    with pytest.raises(TypeError):
        reg.create(StageKind.TEMPO, "wrong")


def test_pipeline_runs_in_thread(mix):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", AnalysisPipeline(default_registry()).run(mix)))
    t.start()
    t.join(10)
    assert isinstance(out["r"].pitch_curve, PitchCurve)
