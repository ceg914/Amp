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


def test_mono_matches_channel_mean_for_any_channel_count():
    """mono()를 열 단위 덧셈으로 바꿨다(속도). 이전 mean(axis=1, float32)과 값이 같아야 한다."""
    rng = np.random.default_rng(3)
    for ch in range(1, 7):
        x = rng.standard_normal((20_001, ch)).astype(np.float32)
        m = AudioData(x, 44100).mono()
        ref = x[:, 0] if ch == 1 else x.mean(axis=1, dtype=np.float32)
        assert m.dtype == np.float32 and np.array_equal(m, ref)
    x = rng.standard_normal((1000, 2)).astype(np.float32)
    m = AudioData(x, 44100).mono()
    m[:] = 0  # 돌려준 배열을 고쳐도 원본은 그대로 (복사본)
    assert np.any(x != 0)


def test_no_vocal_mask_moves_vocals_into_instrumental_inside_ranges_only():
    from amp.audio.mask import FADE_S, apply_no_vocal, keep_gain

    sr = 1000
    rng = np.random.default_rng(5)
    voc = AudioData(rng.standard_normal((5000, 2)).astype(np.float32) * 0.3, sr)
    ins = AudioData(rng.standard_normal((5000, 2)).astype(np.float32) * 0.3, sr)
    out = apply_no_vocal({"vocals": voc, "instrumental": ins}, [(1.0, 2.0), (4.9, 9.0)])
    v, i = out["vocals"].samples, out["instrumental"].samples
    f = int(FADE_S * sr)
    assert np.array_equal(v[:1000], voc.samples[:1000]) and np.array_equal(v[2000:4900], voc.samples[2000:4900])  # 구간 밖 그대로
    assert np.all(v[1000 + f:2000 - f] == 0)  # 구간 안 무음
    assert np.allclose(v + i, voc.samples + ins.samples, atol=1e-6)  # 보컬 + 반주 = 원래 합
    gain = keep_gain(5000, sr, [(1.0, 2.0)])
    assert gain[999] == 1.0 and 0 < gain[1000 + f // 2] < 1 and gain[1000 + f - 1] == 0  # 구간 안쪽으로 페이드 아웃
    assert gain[2000 - f] == 0 and 0 < gain[2000 - f // 2] < 1 and gain[2000] == 1  # 페이드 인
    assert np.allclose(v[1000:1000 + f], voc.samples[1000:1000 + f] * gain[1000:1000 + f, None])
    g = keep_gain(5000, sr, [(4.9, 9.0)])
    assert g[-1] == 0 and g[4899] == 1  # 끝을 넘는 구간은 잘라서
    tiny = keep_gain(5000, sr, [(1.0, 1.01)])  # 페이드보다 짧은 구간
    assert tiny[1000:1010].max() <= 1.0 and tiny[999] == 1 and tiny[1010] == 1
    same = {"vocals": voc}
    assert apply_no_vocal(same, []) is same and apply_no_vocal(same, [(9.0, 10.0)]) is same
    only_v = apply_no_vocal({"vocals": voc}, [(1.0, 2.0)])  # 반주가 없으면 보컬만 비운다
    assert set(only_v) == {"vocals"} and np.all(only_v["vocals"].samples[1000 + f:2000 - f] == 0)


def test_pcm_mixer_add_and_remove_track_while_playing():
    from amp.audio.pcm import PcmMixer

    """v0.6.5 악기 듣기: 재생 중 트랙을 더하고 빼도 출력 장치를 다시 열지 않는다. 새 트랙은 0에서 올라와 딸깍 소리가 없다."""
    n = 4800
    orig = np.full((n, 2), 0.5, dtype=np.float32)
    inst = np.full((n, 2), -0.25, dtype=np.float32)
    m = PcmMixer({"original": orig}, 48000, 48000, 2, SampleFormat.FLOAT32)
    first = m.read_float(0, 1000)
    assert np.allclose(first, 0.5)
    assert m.add_track("inst_drums", inst, {"mute": True, "solo": True})
    assert not m.add_track("short", inst[:100])  # 길이가 다르면 넣지 않는다
    ramp = int(PcmMixer.RAMP_S * 48000)
    block = m.read_float(1000, 2000)
    assert np.all(np.abs(np.diff(block[:, 0])) < 0.75 / ramp + 1e-6)  # 원본은 내려가고 악기는 올라온다 (급격한 변화 없음)
    assert np.allclose(block[ramp + 5:, 0], -0.25)  # 솔로인 악기만
    m.set_state("inst_drums", solo=False)  # 솔로가 풀리면 음소거라 들리지 않고 원본이 돌아온다
    assert np.allclose(m.read_float(3000, 1000)[ramp + 5:, 0], 0.5)
    m.remove_track("inst_drums")
    assert "inst_drums" not in m.state and np.allclose(m.read_float(4000, 500), 0.5)
    m.remove_track("original")  # 마지막 트랙은 빼지 않는다
    assert "original" in m.state
