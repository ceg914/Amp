"""STEP 5 청취 도구(Qt 없음): 음높이 유지 시간 늘리기, 반복·속도 재생 스트림."""

import numpy as np
import pytest

from amp.audio.pcm import PcmMixer, SampleFormat
from amp.audio.stream import SEAM_FADE_S, PlaybackStream
from amp.audio.stretch import StretchCache, time_stretch, time_stretch_many

SR = 44100


def _f0(y, sr=SR):
    n = 32768
    seg = y[len(y) // 4: len(y) // 4 + n] * np.hanning(n)
    sp = np.abs(np.fft.rfft(seg, 4 * n))
    k = int(np.argmax(sp[: int(2000 * 4 * n / sr)]))
    a, b, c = np.log(sp[k - 1:k + 2] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / (4 * n)


def _tone(hz, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    y = 0.3 * np.sin(2 * np.pi * hz * t) + 0.12 * np.sin(2 * np.pi * 2 * hz * t)
    return np.stack([y, y], 1).astype(np.float32)


@pytest.mark.parametrize("speed", [0.5, 0.75, 0.9])
@pytest.mark.parametrize("hz", [220.0, 440.0, 659.25])
def test_time_stretch_keeps_pitch_and_scales_length(speed, hz):
    x = _tone(hz, 2.0)
    y = time_stretch(x, speed, SR)
    assert y.shape == (round(x.shape[0] / speed), 2)
    cents = 1200 * np.log2(_f0(y[:, 0]) / _f0(x[:, 0]))
    assert abs(cents) < 2.0  # 샘플레이트만 바꾸면 0.5배에서 -1200cent
    rms = lambda a: float(np.sqrt(np.mean(a[SR // 2:-SR // 2] ** 2)))  # noqa: E731
    assert rms(y) == pytest.approx(rms(x), rel=0.05)


def test_time_stretch_maps_time_linearly():
    """원본 1.0초의 소리 시작이 0.5배에서 2.0초에 나온다 (위치 환산이 선형이라는 가정 확인)."""
    x = np.zeros((SR * 3, 1), np.float32)
    t = np.arange(SR // 2) / SR
    x[SR:SR + SR // 2, 0] = 0.5 * np.sin(2 * np.pi * 440 * t)
    y = time_stretch(x, 0.5, SR)
    env = np.convolve(np.abs(y[:, 0]), np.ones(441) / 441, mode="same")
    onset = np.argmax(env > 0.1) / SR
    assert onset == pytest.approx(2.0, abs=0.03)


def test_unity_speed_is_identity_and_tracks_stay_separate():
    a, b = _tone(220, 1.0), _tone(330, 1.0)[:, :1]
    ya, yb = time_stretch_many([a, b], 1.0, SR)
    assert np.array_equal(ya, a) and np.array_equal(yb, b)
    ya, yb = time_stretch_many([a, b], 0.75, SR)
    assert ya.shape[1] == 2 and yb.shape[1] == 1
    assert abs(1200 * np.log2(_f0(yb[:, 0]) / 330)) < 2.0


def test_silence_stays_silent():
    y = time_stretch(np.zeros((SR, 2), np.float32), 0.5, SR)
    assert y.shape == (2 * SR, 2) and float(np.abs(y).max()) < 1e-6


def test_stretch_cache_reuses_and_limits_memory():
    cache = StretchCache(max_bytes=3 * SR * 2 * 4 * 2)  # 늘린 2초 스테레오 결과 약 1.5개
    x = _tone(440, 4.0)
    tracks = {"vocals": ("v1", x), "original": ("o1", x)}
    out1 = cache.stretch_tracks(tracks, SR, 2 * SR, 0.5, SR)
    assert cache.misses == 2 and out1["vocals"].shape == (2 * SR, 2)
    out2 = cache.stretch_tracks({"vocals": ("v1", x)}, SR, 2 * SR, 0.5, SR)
    assert out2["vocals"] is out1["vocals"] or cache.hits >= 1
    assert cache.size_bytes <= cache.max_bytes or len(cache._items) == 1
    cache.stretch_tracks({"vocals": ("v1", x)}, SR, 2 * SR, 0.75, SR)  # 속도가 다르면 따로
    assert cache.misses >= 3


def _mixer(x, sr=100):
    return PcmMixer({"original": x}, sr, sr, x.shape[1], SampleFormat.FLOAT32)


def _floats(b):
    return np.frombuffer(b, dtype="<f4")


def test_stream_plain_maps_frames_to_time():
    x = np.arange(1000, dtype=np.float32).reshape(-1, 1) / 1000
    m = _mixer(x)
    s = PlaybackStream(m, 0, m.total_out_frames, 0.0)
    assert s.time_at(250) == pytest.approx(2.5) and s.frame_for_time(2.5) == 250
    assert np.allclose(_floats(s.read(250, 3)), [0.25, 0.251, 0.252])
    assert s.read(999, 10) and not s.read(1000, 10)


def test_stream_loop_wraps_to_region_start_with_seam_fades():
    sr = 1000
    x = np.ones((10 * sr, 1), np.float32)
    m = PcmMixer({"original": x}, sr, sr, 1, SampleFormat.FLOAT32)
    a, b = 2.0, 3.0
    s = PlaybackStream(m, m.out_frame_at(a), m.out_frame_at(b), a, 1.0, loop=True)
    assert s.time_at(0) == pytest.approx(2.0) and s.time_at(999) == pytest.approx(2.999)
    assert s.time_at(1000) == pytest.approx(2.0) and s.time_at(2500) == pytest.approx(2.5)  # 되돌아감
    y = _floats(s.read(0, 3000))
    assert y.shape == (3000,)
    f = int(SEAM_FADE_S * sr)
    assert y[500] == 1.0 and y[0] < 0.5 and y[999] < 0.5  # 이음매 앞뒤만 줄인다
    assert np.all(y[f:1000 - f] == 1.0)
    assert np.array_equal(y[:1000], y[1000:2000])  # 매번 같은 소리
    assert s.total_out_frames > 10**12


def test_stream_speed_reports_original_time():
    """0.5배로 늘린 2초 구간(출력 4초): 출력 3초 지점 = 원본 구간 시작 + 1.5초."""
    sr = 100
    stretched = np.zeros((400, 1), np.float32)
    m = _mixer(stretched, sr)
    s = PlaybackStream(m, 0, m.total_out_frames, 10.0, speed=0.5, loop=False)
    assert s.time_at(300) == pytest.approx(11.5)
    assert s.frame_for_time(11.5) == 300
    assert s.time_at(400) == pytest.approx(12.0) and s.time_at(10_000) == pytest.approx(12.0)
    loop = PlaybackStream(m, 0, m.total_out_frames, 10.0, speed=0.5, loop=True)
    assert loop.time_at(500) == pytest.approx(10.5)


# --- 스펙트로그램 ------------------------------------------------------------------

from amp.audio import spectrogram as SP  # noqa: E402


def _sine(hz, dur, amp=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_spectrogram_single_sine_peak_and_level():
    p = SP.SpecParams(4096, 1024)
    tile = SP.compute_tile(_sine(1000, 3.0), SR, p, 0).astype(np.float32)
    col = tile[:, 60]
    hz = SP.bin_hz(p, SR)
    assert abs(hz[int(np.argmax(col))] - 1000) <= SR / p.n_fft  # 한 칸 이내
    assert -9.0 < col.max() < -5.0  # 진폭 0.5 = −6dBFS (칸 사이 손실 포함)


def test_spectrogram_multiple_frequencies_and_silence():
    p = SP.SpecParams(4096, 1024)
    x = _sine(300, 2.0) + _sine(2000, 2.0, 0.1)
    col = SP.compute_tile(x, SR, p, 0).astype(np.float32)[:, 40]
    hz = SP.bin_hz(p, SR)
    near = lambda f: col[np.abs(hz - f) < 30].max()  # noqa: E731
    assert near(300) > near(2000) > near(1000) + 30  # 두 봉우리, 그 사이는 낮다
    assert near(300) - near(2000) == pytest.approx(20 * np.log10(0.5 / 0.1), abs=1.5)
    silent = SP.compute_tile(np.zeros(SR, np.float32), SR, p, 0).astype(np.float32)
    assert silent.max() <= -190


def test_spectrogram_frequency_axis_maps_rows_and_back():
    rows = 200
    for log_axis in (True, False):
        ys = SP.freq_to_y([100.0, 1000.0, 4000.0], rows, 50.0, 8000.0, log_axis)
        f = SP.row_frequencies(rows, 50.0, 8000.0, log_axis)
        for y, want in zip(ys, (100.0, 1000.0, 4000.0)):
            got = f[min(int(y), rows - 1)]
            assert abs(np.log2(got / want)) < 0.06 if log_axis else abs(got - want) < 45
    # 1kHz 사인파가 그 행에 그려진다
    p = SP.SpecParams(4096, 1024)
    tile = SP.compute_tile(_sine(1000, 2.0), SR, p, 0)
    img = SP.rows_from_bins(tile, p, SR, rows, 50.0, 8000.0, True)
    assert abs(int(np.argmax(img[:, 40])) - int(SP.freq_to_y(1000.0, rows, 50.0, 8000.0, True))) <= 2


def test_spectrogram_time_axis_within_half_frame():
    """1.000초에 시작하는 소리: 그 시각을 덮는 프레임부터 에너지가 보인다(반 프레임 이내)."""
    x = np.zeros(SR * 3, np.float32)
    x[SR:] = _sine(800, 2.0)
    for n_fft in (1024, 2048, 4096):
        p = SP.SpecParams(n_fft, n_fft // 4)
        frames = np.concatenate([SP.compute_tile(x, SR, p, k).astype(np.float32) for k in range(3)], axis=1)
        b = int(round(800 * n_fft / SR))
        on = frames[b] > frames[b].max() - 6  # 정상 세기에서 6dB 안
        first = int(np.argmax(on))
        t_first = SP.frame_time(first, p.hop, SR)
        # 창이 n_fft라 중심이 소리 시작보다 반 창 앞이어도 소리가 들어온다. 정상 세기(−6dB)는 중심이 시작 근처일 때
        assert abs(t_first - 1.0) <= (n_fft / 2) / SR + p.hop / SR / 2


def test_spectrogram_hop_follows_zoom_and_tiles_cover_range():
    assert SP.choose_hop(2048, SR, 400) == 512  # 확대: 기본 hop
    assert SP.choose_hop(2048, SR, 5) >= 4096  # 축소: 1px당 프레임 하나 이상만
    tiles = SP.tiles_for_range(10.0, 20.0, 512, SR)
    covered = (tiles.start * SP.TILE_FRAMES * 512 / SR, (tiles.stop) * SP.TILE_FRAMES * 512 / SR)
    assert covered[0] <= 10.0 and covered[1] >= 20.0


def test_spectrogram_cache_reuse_and_limit():
    cache = SP.TileCache(max_bytes=3 * 2049 * SP.TILE_FRAMES * 2)
    p = SP.SpecParams(4096, 1024)
    x = _sine(440, 10.0)
    for k in range(5):
        key = ("t", p.n_fft, p.hop, k)
        if cache.get(key) is None:
            cache.put(key, SP.compute_tile(x, SR, p, k))
    assert len(cache) == 3 and cache.size_bytes <= cache.max_bytes
    assert cache.get(("t", p.n_fft, p.hop, 4)) is not None and cache.hits == 1


def test_spectrogram_long_audio_only_visible_tiles():
    """6분 음원: 화면(20초)에 보이는 타일만 계산하면 금방 끝난다(곡 전체를 만들지 않는다)."""
    import time

    x = (np.random.default_rng(0).standard_normal(SR * 360) * 0.05).astype(np.float32)
    hop = SP.choose_hop(2048, SR, 1600 / 20)
    t0 = time.perf_counter()
    tiles = [SP.compute_tile(x, SR, SP.SpecParams(2048, hop), k) for k in SP.tiles_for_range(200.0, 220.0, hop, SR)]
    whole = 360 * SR / hop / SP.TILE_FRAMES
    assert len(tiles) <= int(np.ceil(20 * SR / hop / SP.TILE_FRAMES)) + 2 and len(tiles) < whole / 10
    assert time.perf_counter() - t0 < 2.0
