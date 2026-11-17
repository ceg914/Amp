import numpy as np
import pytest

from amp.audio.data import AudioData
from amp.audio.decoder import file_fingerprint, load_audio
from amp.audio.peaks import WaveformPeaks
from amp.audio.pcm import PcmConverter, SampleFormat
from amp.core.errors import AnalysisCancelled, AudioLoadError


def test_load_wav(wav_file):
    calls = []
    a = load_audio(wav_file, progress=lambda f, m: calls.append(f))
    assert a.sample_rate == 22050 and a.channels == 2
    assert a.duration == pytest.approx(2.0)
    assert calls and calls[-1] == pytest.approx(1.0)
    assert not a.samples.flags.writeable


def test_load_mp3(mp3_file):
    a = load_audio(mp3_file)
    assert a.sample_rate == 44100
    assert a.duration == pytest.approx(8.0, abs=0.1)
    assert 0.3 < float(np.abs(a.samples).max()) < 0.5


def test_load_errors(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(str(tmp_path / "missing.mp3"))
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(AudioLoadError, match="지원하지 않는"):
        load_audio(str(txt))
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio at all" * 10)
    with pytest.raises(AudioLoadError):
        load_audio(str(broken))


def test_load_cancel(wav_file):
    with pytest.raises(AnalysisCancelled):
        load_audio(wav_file, is_cancelled=lambda: True)


def test_fingerprint_changes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"a" * 200_000)
    f1 = file_fingerprint(str(p))
    p.write_bytes(b"a" * 199_999 + b"b")
    assert file_fingerprint(str(p)) != f1


def test_audio_data_mono():
    a = AudioData(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), 10)
    assert a.mono().tolist() == [0.5, 0.5]


def _brute(x, sr, t0, t1, width):
    spp = (t1 - t0) * sr / width
    mins, maxs = [], []
    for i in range(width):
        a = int(np.floor(t0 * sr + i * spp))
        b = int(np.ceil(t0 * sr + (i + 1) * spp))
        seg = x[max(a, 0):max(b, 0)]
        mins.append(seg.min() if seg.size else np.nan)
        maxs.append(seg.max() if seg.size else np.nan)
    return np.array(mins), np.array(maxs)


def test_peaks_bound_true_envelope():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 100_000).astype(np.float32)
    p = WaveformPeaks.build(x, 10_000)
    assert len(p.levels) > 2
    mins, maxs = p.columns(1.0, 6.0, 300)
    bmin, bmax = _brute(x, 10_000, 1.0, 6.0, 300)
    # 버킷 경계 때문에 요약값은 실제 구간보다 약간 넓을 수 있지만 좁아서는 안 된다
    assert np.all(mins <= bmin + 1e-6) and np.all(maxs >= bmax - 1e-6)


def test_peaks_outside_range_is_nan():
    p = WaveformPeaks.build(np.ones(1000, np.float32), 1000)
    mins, _ = p.columns(-1.0, 2.0, 30)
    assert np.isnan(mins[:9]).all() and np.isnan(mins[-9:]).all()
    assert not np.isnan(mins[11:19]).any()


def test_pcm_same_rate_int16():
    s = np.array([[0.5, -0.5], [1.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    c = PcmConverter(s, 100, 100, 2, SampleFormat.INT16)
    data = np.frombuffer(c.read(0, 10), dtype="<i2")
    assert data.tolist() == [16383, -16383, 32767, -32767, 0, 0]
    assert c.read(3, 5) == b""


def test_pcm_resample_and_channels():
    s = np.arange(10, dtype=np.float32)[:, None] / 10
    c = PcmConverter(s, 100, 200, 2, SampleFormat.FLOAT32)
    assert c.total_out_frames == 19
    out = np.frombuffer(c.read(0, 4), dtype="<f4").reshape(-1, 2)
    assert out[:, 0] == pytest.approx([0.0, 0.05, 0.1, 0.15])
    assert (out[:, 0] == out[:, 1]).all()
    # 이어 읽어도 한 번에 읽은 것과 같다
    whole = c.read(0, 19)
    assert c.read(0, 7) + c.read(7, 12) == whole


def test_pcm_downmix_and_seek():
    s = np.stack([np.ones(100), -np.ones(100) * 0.5], 1).astype(np.float32)
    c = PcmConverter(s, 100, 100, 1, SampleFormat.INT32)
    assert c.out_frame_at(0.5) == 50 and c.out_frame_at(99) == 100
    v = np.frombuffer(c.read(10, 1), dtype="<i4")[0]
    assert v == pytest.approx(0.25 * 2147483647, rel=1e-6)


def test_without_miniaudio_fallback(tmp_path, monkeypatch):
    import amp.audio.decoder as dec

    def no_miniaudio(*a, **k):
        raise ImportError("miniaudio 없음")

    monkeypatch.setattr(dec, "_load_with_miniaudio", no_miniaudio)
    broken = tmp_path / "x.wav"
    broken.write_bytes(b"RIFF0000WAVEjunk")
    with pytest.raises(AudioLoadError, match="손상"):
        load_audio(str(broken))


def test_pcm_mixer_mute_solo_gain():
    from amp.audio.pcm import PcmMixer

    a = np.full((10, 1), 0.25, np.float32)
    b = np.full((10, 1), 0.5, np.float32)
    m = PcmMixer({"original": a, "vocals": b}, 100, 100, 1, SampleFormat.FLOAT32)
    read = lambda: np.frombuffer(m.read(0, 1), dtype="<f4")[0]
    assert read() == pytest.approx(0.75)
    m.set_state("vocals", mute=True)
    assert read() == pytest.approx(0.25)
    m.set_state("vocals", solo=True)  # 솔로는 음소거보다 우선
    assert read() == pytest.approx(0.5)
    m.set_state("vocals", gain=0.5)
    assert read() == pytest.approx(0.25)
    m.set_state("vocals", solo=False, mute=False, gain=1.0)
    m.set_state("original", gain=0.0)
    assert read() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        PcmMixer({"a": a, "b": np.zeros((5, 1), np.float32)}, 100, 100, 1)
