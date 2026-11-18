"""분석 안정성 (STEP 2.5 6장): 캐시 손상, 실패 결과 미저장, 강제 종료, 디스크 부족, 손상·잘린 파일, GPU 메모리 부족, 모델·라이브러리 누락."""

import glob
import json
import sys

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from amp.audio.decoder import load_audio
from amp.core.errors import AnalysisError
from amp.engine import worker
from amp.engine.dsp import separation as sep
from amp.engine.transcription import TranscriptionPipeline
from amp.engine.worker_client import WorkerError, run_worker
from amp.devtools.synth import SR, sung
from tests.test_transcription import InProcessRunner, make_request


@pytest.fixture
def song(tmp_path):
    y = sung([(0.5, 1.5, 64), (1.6, 2.6, 67)], 3.0)
    p = tmp_path / "곡 (안정성).wav"
    sf.write(p, np.stack([y, y], 1), SR)
    return str(p)


def test_corrupt_pitch_cache_is_detected_and_recomputed(song, tmp_path):
    runner = InProcessRunner()
    pipe = TranscriptionPipeline(runner, separation_model="_identity")
    req = make_request(song, tmp_path)
    first = pipe.run(req)
    npz = glob.glob(str(tmp_path / "proj" / "cache" / "*" / "pitch-*" / "pitch.npz"))[0]
    with open(npz, "wb") as f:
        f.write(b"not a zip")  # 손상
    second = pipe.run(req)
    assert runner.calls["pitch"] == 2 and not second.reused["pitch"]
    assert any("캐시가 손상" in w for w in second.result.warnings)
    assert [n.pitch for n in second.result.notes] == [n.pitch for n in first.result.notes]
    tempo_json = glob.glob(str(tmp_path / "proj" / "cache" / "*" / "tempo-*" / "tempo.json"))[0]
    with open(tempo_json, "w", encoding="utf-8") as f:
        json.dump({"bpm": -5}, f)
    pipe.run(req)
    assert runner.calls["tempo"] == 2


def test_invalid_new_result_is_not_committed(song, tmp_path):
    class BadPitchRunner(InProcessRunner):
        def __call__(self, command, args, progress=None, token=None, numba_cache_dir=None):
            out = super().__call__(command, args, progress, token, numba_cache_dir)
            if command == "pitch":  # 프레임 수가 틀린 결과를 흉내
                np.savez_compressed(f"{args['out_dir']}/pitch.npz", f0_hz=np.ones(5, np.float32),
                                    voiced_prob=np.ones(5, np.float32), onset=np.ones(5, np.float32),
                                    rms=np.ones(5, np.float32), frame_rate=np.float64(100), start_time=np.float64(0))
            return out

    req = make_request(song, tmp_path)
    with pytest.raises(AnalysisError, match="프레임 수"):
        TranscriptionPipeline(BadPitchRunner(), separation_model="_identity").run(req)
    assert glob.glob(str(tmp_path / "proj" / "cache" / "*" / "pitch-*")) == []  # 실패 결과는 캐시에 남지 않는다
    assert glob.glob(str(tmp_path / "proj" / "cache" / "*" / ".tmp-*")) == []


def test_killed_worker_leaves_no_result(song, tmp_path):
    req = make_request(song, tmp_path)
    with pytest.raises(WorkerError) as e:
        TranscriptionPipeline(separation_model="_crash").run(req)
    if sys.platform != "win32":
        assert e.value.code == "killed"
    assert "완성되지 않은 결과는 저장하지 않았습니다" in str(e.value) or e.value.code == "internal"
    stems = tmp_path / "proj" / "stems"
    assert list(stems.iterdir()) == []  # stems.json도 임시 폴더도 없음 → '분리 전'으로 보인다


def test_worker_classifies_disk_full_and_permission():
    code, msg = worker._classify(OSError(28, "No space left on device", "vocals.wav"))
    assert code == "disk_full" and "디스크" in msg
    code, _ = worker._classify(PermissionError(13, "denied", "x.wav"))
    assert code == "permission"


def test_truncated_and_garbage_mp3(tmp_path, caplog):
    y = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 4) / SR)).astype(np.float32)
    p = tmp_path / "잘린 파일.mp3"
    sf.write(p, np.stack([y, y], 1), SR, format="MP3")
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # 중간에서 잘림
    with caplog.at_level("INFO"):
        a = load_audio(str(p))  # 예외 없이 앞부분만 읽는다
    assert 1.0 < a.duration < 3.5
    assert "잘렸을 수 있습니다" in caplog.text  # 길이 헤더(Info)가 있는 MP3라 잘림을 알 수 있다
    with pytest.raises(WorkerError) as e:
        run_worker("pitch", {"input": str(tmp_path / "없는.ogg"), "out_dir": str(tmp_path)})
    assert e.value.code == "decode"


# --- 분리 모델 (torch가 있을 때만, 같은 프로세스에서 demucs 함수를 바꿔 끼운다) ---------------------



class _FakeModel:
    sources = ["drums", "bass", "other", "vocals"]
    segment = 1.0

    def eval(self):
        return self


def _install_fake_demucs(monkeypatch, apply_model):
    pytest.importorskip("demucs")
    import demucs.apply
    import demucs.pretrained

    monkeypatch.setattr(demucs.pretrained, "get_model", lambda name: _FakeModel())
    monkeypatch.setattr(demucs.apply, "apply_model", apply_model)


def test_gpu_oom_falls_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    used = []

    def apply_model(model, x, device=None, **kw):
        used.append(device)
        if device == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return torch.zeros((1, 4) + tuple(x.shape[1:]))

    _install_fake_demucs(monkeypatch, apply_model)
    monkeypatch.setattr(sep, "choose_device", lambda requested: ("cuda", ""))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    sources, info = sep._run_demucs("htdemucs", np.zeros((2, 4410), np.float32), "auto", lambda f, m: None)
    assert used == ["cuda", "cpu"] and info["device"] == "cpu" and "GPU 메모리" in info["note"]
    assert set(sources) == {"drums", "bass", "other", "vocals"}


def test_separation_is_repeatable(monkeypatch):
    """demucs의 임의 이동(shifts) 때문에 같은 곡을 다시 분리하면 결과가 달라지던 문제 (Windows 실행 기록에서 발견)."""
    torch = pytest.importorskip("torch")
    import random

    def apply_model(model, x, device=None, shifts=0, **kw):
        offset = random.randint(0, 22050) if shifts else 0  # demucs와 같은 방식으로 이동량을 고른다
        return torch.full((1, 4) + tuple(x.shape[1:]), float(offset))

    _install_fake_demucs(monkeypatch, apply_model)
    monkeypatch.setattr(sep, "choose_device", lambda requested: ("cpu", ""))
    x = np.zeros((2, 4410), np.float32)
    runs = []
    for _ in range(3):
        random.random()  # 앞선 작업이 난수를 얼마나 썼는지와 상관없어야 한다
        sources, info = sep._run_demucs("htdemucs", x, "cpu", lambda f, m: None)
        runs.append(float(sources["vocals"].flat[0]))
    assert len(set(runs)) == 1 and info["shift_seed"] == sep.SHIFT_SEED


def test_cpu_oom_reports_clear_error(monkeypatch):
    pytest.importorskip("torch")

    def apply_model(model, x, device=None, **kw):
        raise RuntimeError("DefaultCPUAllocator: can't allocate memory: out of memory")

    _install_fake_demucs(monkeypatch, apply_model)
    monkeypatch.setattr(sep, "choose_device", lambda requested: ("cpu", ""))
    with pytest.raises(sep.SeparationError) as e:
        sep._run_demucs("htdemucs", np.zeros((2, 4410), np.float32), "cpu", lambda f, m: None)
    assert e.value.code == "oom"


def test_model_download_failure(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("demucs")
    import demucs.pretrained

    def boom(name):
        raise OSError("Connection refused")

    monkeypatch.setattr(demucs.pretrained, "get_model", boom)
    with pytest.raises(sep.SeparationError) as e:
        sep._run_demucs("htdemucs", np.zeros((2, 4410), np.float32), "cpu", lambda f, m: None)
    assert e.value.code == "model_download"


def test_missing_torch_or_demucs(monkeypatch):
    monkeypatch.setitem(sys.modules, "demucs.apply", None)  # import 실패 흉내
    with pytest.raises(sep.SeparationError) as e:
        sep._run_demucs("htdemucs", np.zeros((2, 4410), np.float32), "cpu", lambda f, m: None)
    assert e.value.code == "missing_dependency"


def test_failed_analysis_does_not_change_project_files(song, tmp_path):
    """분석이 실패해도 이전 분리 결과·캐시는 그대로다 (실패 결과를 완료로 저장하지 않음)."""
    runner = InProcessRunner()
    TranscriptionPipeline(runner, separation_model="_identity").run(make_request(song, tmp_path))
    manifest = (tmp_path / "proj" / "stems" / "stems.json").read_bytes()
    req = make_request(song, tmp_path, force_separation=True)
    with pytest.raises(WorkerError):
        TranscriptionPipeline(separation_model="_crash").run(req)
    # 강제 재분리가 실패하면 기존 목록은 무효화되지 않는다 (교체는 새 결과가 완성된 뒤에만 시작)
    assert (tmp_path / "proj" / "stems" / "stems.json").read_bytes() == manifest


def test_worker_peak_memory_is_its_own(tmp_path):
    """작업 프로세스 최대 메모리에 부모 프로세스(GUI) 메모리가 섞이지 않는다 (STEP 2.5에서 발견한 측정 오류)."""
    big = np.ones(600 * 2**20 // 8)  # 부모가 600MB를 잡고 있는 상태
    big[::4096] += 1  # 실제로 페이지를 쓴다
    sr = 16000
    p = tmp_path / "짧은.wav"
    sf.write(p, np.zeros(sr, np.float32), sr)
    out = run_worker("tempo", {"input": str(p), "out_dir": str(tmp_path)})
    assert 0 < out["resources"]["peak_rss_mb"] < 500
    del big


def test_mp3_without_length_header_is_not_reported_as_truncated(tmp_path, caplog):
    """실제 곡에서 발견: Xing·Info 헤더 없는 MP3는 libsndfile이 파일 크기로 길이를 어림해 실제보다 길게 잡는다."""
    from amp.audio.decoder import mp3_has_length_header

    y = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 3) / SR)).astype(np.float32)
    src = tmp_path / "src.mp3"
    sf.write(src, np.stack([y, y], 1), SR, format="MP3")
    data = src.read_bytes()
    i = data.index(b"\xff\xfb") if b"\xff\xfb" in data else data.index(b"\xff\xfa")
    h = int.from_bytes(data[i:i + 4], "big")
    br = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320][(h >> 12) & 15] * 1000
    first = 144 * br // 44100 + ((h >> 9) & 1)
    assert b"Info" in data[i:i + first] or b"Xing" in data[i:i + first]
    # 길이 헤더가 든 첫 프레임을 빼고, 앞에 큰 ID3v2 태그(앨범 아트 흉내 400KB)를 붙인다
    art = 400_000
    size = bytes([(art >> 21) & 0x7F, (art >> 14) & 0x7F, (art >> 7) & 0x7F, art & 0x7F])
    p = tmp_path / "길이 헤더 없음.mp3"
    p.write_bytes(b"ID3\x03\x00\x00" + size + b"\x00" * art + data[i + first:])
    assert not mp3_has_length_header(str(p)) and mp3_has_length_header(str(src))
    with caplog.at_level("INFO"):
        a = load_audio(str(p))
    assert 2.9 < a.duration < 3.1
    assert "잘렸을 수 있습니다" not in caplog.text


def test_cpu_fallback_with_nvidia_gpu_is_reported(monkeypatch, song, tmp_path):
    """GPU가 있는데 PyTorch가 GPU를 못 쓰면 이유와 해결법을 경고로 알린다 (사용자 Windows 실행에서 조용히 CPU로 돈 문제)."""
    import types

    from amp.engine.dsp import separation as sep

    fake_torch = types.SimpleNamespace(__version__="2.15.0+cpu", cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sep, "nvidia_gpu", lambda: {"name": "NVIDIA GeForce RTX 3060", "driver": "581.15"})
    dev, note = sep.choose_device("auto")
    assert dev == "cpu" and "RTX 3060" in note and "install_separation.bat" in note and "2.15.0+cpu" in note
    monkeypatch.setattr(sep, "nvidia_gpu", lambda: None)
    assert sep.choose_device("auto") == ("cpu", "")  # GPU가 없으면 조용히 CPU
    assert sep.choose_device("cuda")[1]  # GPU를 고르고 못 쓰면 알린다

    # 파이프라인: 새로 분리했을 때 분리 단계 안내가 분석 경고로 나온다
    real = sep.separate

    def separate_with_note(*a, **k):
        v, i, info = real(*a, **k)
        info["note"] = "NVIDIA GPU(RTX 3060)가 있지만 … CPU로 분리했습니다."
        return v, i, info

    monkeypatch.setattr(sep, "separate", separate_with_note)
    from tests.test_transcription import InProcessRunner, make_request

    from amp.engine.transcription import TranscriptionPipeline

    req = make_request(song, tmp_path)
    out = TranscriptionPipeline(InProcessRunner(), separation_model="_identity").run(req)
    assert any("RTX 3060" in w for w in out.result.warnings)
    again = TranscriptionPipeline(InProcessRunner(), separation_model="_identity").run(make_request(song, tmp_path))
    assert not any("RTX 3060" in w for w in again.result.warnings)  # 기존 분리 결과를 재사용하면 다시 알리지 않는다
