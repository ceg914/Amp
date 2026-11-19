"""외부 분리 결과 가져오기: 원본에 맞추기(어긋남·샘플레이트·채널·길이), 다른 곡 거부, stems/ 기록, 파이프라인 재사용."""

import io
import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("scipy")

from amp.audio.data import AudioData
from amp.audio.decoder import load_audio
from amp.core.errors import AnalysisError
from amp.engine import stem_import
from amp.engine import stems as stem_store

SR = 44100
DUR = 8.0


def _band_noise(n, seed, lo=80.0, hi=12000.0, sr=SR):
    """대역 제한 잡음 (리샘플 왕복 오차를 작게 하려고 높은 대역을 뺀다)."""
    from scipy.signal import butter, sosfilt

    x = np.random.default_rng(seed).standard_normal(n)
    return sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def make_song(seed=0, dur=DUR):
    """(원본, 보컬, 반주) 모두 (frames, 2). 보컬은 켜졌다 꺼지는 대역 잡음, 반주는 다른 잡음 + 저음."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    gate = (np.sin(2 * np.pi * 0.7 * t + seed) > -0.2).astype(float)
    v = _band_noise(n, seed + 10, 200, 6000) * gate * 0.25
    inst = _band_noise(n, seed + 20) * 0.12 + 0.15 * np.sin(2 * np.pi * 55 * t)
    vocals = np.stack([v, 0.9 * v], 1).astype(np.float32)
    instrumental = np.stack([inst, np.roll(inst, 7)], 1).astype(np.float32)
    return (vocals + instrumental).astype(np.float32), vocals, instrumental


def _write(path, x, sr=SR):
    sf.write(str(path), x, sr, subtype="FLOAT")
    return str(path)


@pytest.fixture
def song(tmp_path):
    mix, v, i = make_song()
    return AudioData(mix, SR), v, i, tmp_path


def test_identical_files_import_without_change(song):
    original, v, i, d = song
    vp, ip = _write(d / "v.wav", v), _write(d / "i.wav", i)
    vo, io_, rep = stem_import.prepare(original, vp, ip, tool="시험")
    assert rep.lag_ms == 0 and rep.resampled_from is None and rep.instrumental == "imported"
    assert rep.residual_db < -100 and rep.warnings == []
    assert np.array_equal(vo.samples, v) and np.array_equal(io_.samples, i)
    assert "시간 어긋남 없음" in rep.summary()


def test_delay_and_length_difference_are_corrected(song):
    """MP3 인코더 지연처럼 앞에 10ms 무음이 붙고 끝이 0.5초 짧은 파일 → 어긋남을 찾아 원본 위치로 되돌린다."""
    original, v, i, d = song
    pad = 441
    cut = int(0.5 * SR)
    shift = lambda x: np.concatenate([np.zeros((pad, 2), np.float32), x])[:-cut]  # noqa: E731
    vp, ip = _write(d / "v.wav", shift(v)), _write(d / "i.wav", shift(i))
    vo, io_, rep = stem_import.prepare(original, vp, ip)
    assert rep.lag_ms == pytest.approx(-10.0, abs=0.01)  # 앞으로 441샘플 당김
    assert rep.length_diff_ms == pytest.approx((pad - cut) / SR * 1000, abs=0.1)
    assert vo.frames == original.frames and vo.sample_rate == SR
    end = original.frames - cut  # 이 뒤는 파일에 없어서 0으로 채운다
    assert np.array_equal(vo.samples[:end], v[:end]) and np.all(vo.samples[end:] == 0)
    assert np.array_equal(io_.samples[:end], i[:end])


def test_delayed_file_the_other_way(song):
    original, v, i, d = song
    vp = _write(d / "v.wav", v[1000:])  # 앞 1000샘플이 잘린 파일 → 뒤로 민다
    vo, io_, rep = stem_import.prepare(original, vp)
    assert rep.lag_ms == pytest.approx(1000 / SR * 1000, abs=0.01)
    assert np.all(vo.samples[:1000] == 0) and np.array_equal(vo.samples[1000:], v[1000:])


def test_other_sample_rate_and_mono_are_matched(song):
    from scipy.signal import resample_poly

    original, v, i, d = song
    v48 = resample_poly(v, 160, 147, axis=0).astype(np.float32)
    i48 = resample_poly(i, 160, 147, axis=0).astype(np.float32)
    vo, io_, rep = stem_import.prepare(original, _write(d / "v.wav", v48, 48000), _write(d / "i.wav", i48, 48000))
    assert rep.resampled_from == 48000 and abs(rep.lag_ms) < 0.05
    assert vo.frames == original.frames and vo.sample_rate == SR
    err = vo.samples[SR:-SR] - v[SR:-SR]  # 양 끝은 필터 가장자리
    assert 10 * np.log10(np.mean(err ** 2) / np.mean(v[SR:-SR] ** 2)) < -40
    assert rep.residual_db < -30
    # 모노 보컬 → 원본 채널 수로 복사
    vo, io_, rep = stem_import.prepare(original, _write(d / "m.wav", v[:, 0]))
    assert rep.channels_from == 1 and vo.channels == 2
    assert np.array_equal(vo.samples[:, 1], v[:, 0])


def test_vocals_only_makes_instrumental_from_original(song):
    original, v, i, d = song
    vo, io_, rep = stem_import.prepare(original, _write(d / "v.wav", v))
    assert rep.instrumental == "original_minus_vocals" and rep.residual_db is None
    assert rep.correlation > 0.3
    assert np.allclose(vo.samples + io_.samples, original.samples, atol=1e-6)
    assert np.allclose(io_.samples, i, atol=1e-6)
    assert "원본 − 보컬" in rep.summary()


def test_other_song_is_rejected(song):
    original, v, i, d = song
    _mix, v2, i2 = make_song(seed=5)
    with pytest.raises(AnalysisError, match="맞지 않습니다"):
        stem_import.prepare(original, _write(d / "v2.wav", v2), _write(d / "i2.wav", i2))
    with pytest.raises(AnalysisError, match="거의 맞지 않습니다"):
        stem_import.prepare(original, _write(d / "v2.wav", v2))


def test_length_mismatch_is_rejected(song):
    original, v, i, d = song
    longer = np.concatenate([v, np.zeros((4 * SR, 2), np.float32)])
    with pytest.raises(AnalysisError, match="길이"):
        stem_import.prepare(original, _write(d / "v.wav", longer))
    with pytest.raises(AnalysisError, match="보컬과 반주"):
        stem_import.prepare(original, _write(d / "v.wav", v), _write(d / "i.wav", i[: 4 * SR]))


def test_volume_change_is_a_warning_not_an_error(song):
    original, v, i, d = song
    vo, io_, rep = stem_import.prepare(original, _write(d / "v.wav", v * 0.8), _write(d / "i.wav", i * 0.8))
    assert -20 < rep.residual_db < -6 and rep.warnings


def test_write_stems_records_external_manifest(song):
    original, v, i, d = song
    vp, ip = _write(d / "v.wav", v), _write(d / "i.wav", i)
    vo, io_, rep = stem_import.prepare(original, vp, ip, tool="UVR MelBand RoFormer")
    stems_dir = d / "proj" / "stems"
    m = stem_import.write_stems(stems_dir, vo, io_, "a" * 64, rep, vp, ip)
    assert m.model == stem_import.EXTERNAL and m.source_sha256 == "a" * 64
    assert m.model_info["tool"] == "UVR MelBand RoFormer" and m.settings == {"imported": True}
    assert len(m.model_info["vocals_sha256"]) == 64 and m.model_info["instrumental"] == "imported"
    ok, why = stem_store.validate(stems_dir, stem_store.read_manifest(stems_dir), "a" * 64)
    assert ok, why
    loaded = stem_store.load(stems_dir, m)
    assert loaded["vocals"].frames == original.frames
    assert np.allclose(loaded["vocals"].samples, v, atol=2 ** -22)  # PCM 24bit
    assert not [p for p in stems_dir.iterdir() if p.name.startswith(".tmp-")]
    # 같은 초 안에 다른 파일을 다시 가져와도 stems_id가 바뀐다 (피치 캐시 키가 달라진다)
    vp2 = _write(d / "v2.wav", v * 0.5)
    vo2, io2, rep2 = stem_import.prepare(original, vp2, ip)
    m2 = stem_import.write_stems(stems_dir, vo2, io2, "a" * 64, rep2, vp2, ip)
    assert m2.stems_id != m.stems_id


def test_pipeline_uses_imported_stems_instead_of_separating(tmp_path):
    """가져온 결과(model='external')가 있으면 분석이 Demucs를 다시 돌리지 않고 그 보컬로 피치를 계산한다.
    '다시 분리'를 명시하면 그때만 분리한다."""
    pytest.importorskip("librosa")
    from dataclasses import replace

    from amp.devtools.synth import backing, sung
    from amp.engine import worker
    from amp.engine.cache import file_content_key
    from amp.engine.settings import AnalysisSettings
    from amp.engine.transcription import TranscriptionPipeline, TranscriptionRequest

    calls = Counter()

    def runner(command, args, progress=None, token=None, numba_cache_dir=None):
        calls[command] += 1
        if command == "pitch":
            calls["pitch_input:" + args["input"].replace("\\", "/").rsplit("/", 2)[-2]] += 1
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    melody = [(1.0, 1.8, 64), (1.9, 2.7, 67), (2.8, 4.0, 69)]
    voice = sung(melody, 5.0)
    band = backing(5.0) * 0.3
    mix = np.stack([voice + band, voice + band], 1).astype(np.float32)
    path = tmp_path / "곡.wav"
    sf.write(path, mix, SR)
    audio = load_audio(str(path))
    sha = file_content_key(str(path))
    vp = _write(tmp_path / "vox.wav", np.stack([voice, voice], 1).astype(np.float32))
    vo, io_, rep = stem_import.prepare(audio, vp)
    stems_dir = tmp_path / "proj" / "stems"
    imported = stem_import.write_stems(stems_dir, vo, io_, sha, rep, vp, None)
    req = TranscriptionRequest(audio, str(path), sha, AnalysisSettings(), str(stems_dir), str(tmp_path / "proj" / "cache"))
    pipe = TranscriptionPipeline(runner, separation_model="_identity")
    out = pipe.run(req)
    assert calls["separate"] == 0 and out.reused["separation"] is True
    assert calls["pitch_input:stems"] == 1  # stems/vocals.wav로 피치
    assert out.manifest.stems_id == imported.stems_id
    assert out.result.models["separation"].name == "external"
    assert [n.pitch for n in out.result.notes] == [64, 67, 69]
    # 다른 보컬 파일을 다시 가져오면 피치를 새로 계산한다 (캐시 키 = stems_id)
    vp2 = _write(tmp_path / "vox2.wav", np.stack([voice, voice], 1).astype(np.float32) * 0.9)
    vo2, io2, rep2 = stem_import.prepare(audio, vp2)
    stem_import.write_stems(stems_dir, vo2, io2, sha, rep2, vp2, None)
    pipe.run(req)
    assert calls["pitch"] == 2 and calls["separate"] == 0
    # 명시적으로 다시 분리하면 그때 분리한다
    out = pipe.run(replace(req, force_separation=True))
    assert calls["separate"] == 1 and out.manifest.model == "_identity"
    info = json.loads((stems_dir / stem_store.MANIFEST).read_text(encoding="utf-8"))
    assert info["model"] == "_identity"
