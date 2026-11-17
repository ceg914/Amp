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
    a = load_audio(str(p))  # 예외 없이 앞부분만 읽는다
    assert 1.0 < a.duration < 3.5
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
