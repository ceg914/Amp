"""v0.6.1 피치 다시 찾기 구간: 설정 검사, 구간 추적의 프레임 격자, 서브하모닉 버림, 파이프라인 끼워 넣기·캐시.

사용자 곡 보고(0:34~0:51 남성 보컬 음표 누락)에서 출발했다. 그 곡에서는 여러 소리가 겹쳐 pYIN이 대부분을 무성으로
버렸고(YIN 최소값 중앙 0.46~0.52), 음역을 좁혀 다시 찾으면 높은 목소리의 절반 주파수를 잡는 일도 있었다.
"""

from __future__ import annotations

import io
import threading
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from amp.devtools.synth import SR, sung  # noqa: E402
from amp.engine.dsp import pitch as P  # noqa: E402
from amp.engine.settings import AnalysisSettings, SettingsError, normalize_pitch_regions  # noqa: E402

MALE = [(0.3, 1.0, 50), (1.0, 1.7, 52), (1.7, 2.4, 53), (2.4, 3.2, 55)]


def _midi(f):
    return 69 + 12 * np.log2(np.where(np.isfinite(f), f, np.nan) / 440)


def test_settings_validate_and_roundtrip():
    r = normalize_pitch_regions([(40.0, 46.0, 70, 420), (34.0, 38.0, 140, 1100)])
    assert r == ((34.0, 38.0, 140.0, 1100.0), (40.0, 46.0, 70.0, 420.0))
    with pytest.raises(SettingsError):
        normalize_pitch_regions([(34, 40, 70, 420), (39, 45, 70, 420)])  # 겹침
    with pytest.raises(SettingsError):
        normalize_pitch_regions([(34, 40, 420, 70)])  # 음역 거꾸로
    with pytest.raises(SettingsError):
        normalize_pitch_regions([(34, 34.1, 70, 420)])  # 0.2초 미만
    assert normalize_pitch_regions([(34, 40, 70, 420), (39, 45, 70, 420), ("x",)], strict=False) == ((34.0, 40.0, 70.0, 420.0),)
    s = AnalysisSettings(pitch_regions=[(34.0, 52.0, 70.0, 420.0)])
    assert AnalysisSettings.from_dict(s.to_dict()) == s
    bad = {**s.to_dict(), "pitch_regions": [[1, 0, 70, 420]], "sensitivity": "precise"}
    t = AnalysisSettings.from_dict(bad)  # 잘못된 구간 하나 때문에 다른 설정을 잃지 않는다
    assert t.pitch_regions == () and t.sensitivity == "precise"


def test_region_tracking_uses_the_same_frame_grid():
    y = sung(MALE, 3.5)
    full = P.track_pitch(y, SR)
    f0, prob, g0 = P.track_region(y, SR, 0.5, 3.0, 70, 420)
    assert g0 == 50 and len(f0) == 250 and len(prob) == 250
    a, b = _midi(full.f0_hz[g0:g0 + len(f0)]), _midi(f0)
    both = np.isfinite(a) & np.isfinite(b)
    assert both.mean() > 0.9
    assert np.mean(np.abs(a[both] - b[both]) < 0.1) > 0.95  # 같은 시각에 같은 높이


def test_subharmonic_of_a_higher_voice_is_rejected():
    male = sung(MALE, 3.5)
    female = sung([(0.3, 3.2, 72)], 3.5, vibrato_cents=40)
    stats: dict = {}
    f0, _p, _g0 = P.track_region(male + 0.9 * female, SR, 0.4, 3.1, 70, 420, stats=stats)
    m = _midi(f0)
    assert stats["subharmonic_frames"] > 100  # 72(C5)의 절반 60(C4)을 잡은 프레임을 버렸다
    ok = np.isfinite(m)
    assert not ok.any() or np.mean(np.abs(m[ok] - 60) > 1.0) > 0.9
    stats2: dict = {}
    f0m, _p, _g = P.track_region(male, SR, 0.4, 3.1, 70, 420, stats=stats2)
    assert stats2["subharmonic_frames"] < 10 and np.isfinite(f0m).mean() > 0.9  # 진짜 낮은 목소리는 남긴다
    ratio = P.odd_even_ratio(P.to_pitch_sr(male, SR), np.full(len(f0m), np.nan))
    assert np.isnan(ratio).all()


@pytest.fixture()
def song(tmp_path):
    y = sung([(0.3, 1.2, 60), (1.3, 2.2, 62), (2.3, 3.2, 64), (3.3, 4.2, 65)], 4.6)
    path = tmp_path / "곡.wav"
    sf.write(path, np.stack([y, y], 1), SR)
    return path


def _runner(calls: Counter):
    from amp.engine import worker

    lock = threading.Lock()

    def run(command, args, progress=None, token=None, numba_cache_dir=None):
        with lock:
            calls[command] += 1
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    return run


def test_pipeline_splices_region_and_caches_it(song, tmp_path):
    from amp.audio.decoder import load_audio
    from amp.engine.cache import file_content_key
    from amp.engine.transcription import TranscriptionPipeline, TranscriptionRequest

    def request(regions):
        st = AnalysisSettings(device="cpu", separation="none", pitch_regions=regions)
        return TranscriptionRequest(load_audio(str(song)), str(song), file_content_key(str(song)), st, None,
                                    str(tmp_path / "cache"))

    calls: Counter = Counter()
    base = TranscriptionPipeline(_runner(calls)).run(request(()))
    reg = TranscriptionPipeline(_runner(calls)).run(request(((1.3, 3.2, 140.0, 1100.0),)))
    assert calls["pitch"] == 1 and calls["pitch_region"] == 1  # 곡 전체 피치는 캐시, 구간만 계산
    info = reg.result.models["pitch"].params["regions"][0]
    assert info["start"] == 1.3 and info["replaced"] > 0.5
    a, b = base.result.pitch_curve, reg.result.pitch_curve
    outside = np.r_[0:130, 320:len(a.f0_hz)]
    assert np.array_equal(np.nan_to_num(a.f0_hz[outside]), np.nan_to_num(b.f0_hz[outside]))  # 구간 밖은 그대로
    again = TranscriptionPipeline(_runner(calls)).run(request(((1.3, 3.2, 140.0, 1100.0),)))
    assert calls["pitch_region"] == 1 and again.reused["pitch"]  # 같은 구간이면 다시 계산하지 않음
    TranscriptionPipeline(_runner(calls)).run(request(((1.3, 3.0, 140.0, 1100.0),)))
    assert calls["pitch_region"] == 2  # 구간이 바뀌면 다시 계산
    assert [n.pitch for n in reg.result.notes] == [60, 62, 64, 65]  # 한 목소리 곡은 결과가 같다
