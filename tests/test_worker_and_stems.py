"""분석 작업 프로세스와 분리 결과 관리 (요구 4·5·8 일부). 실제 프로세스를 띄운다."""

import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from amp.core.errors import AnalysisCancelled
from amp.engine import stems as stem_store
from amp.engine.context import CancelToken
from amp.engine.worker_client import WorkerError, run_worker


@pytest.fixture
def stereo48k(tmp_path):
    d = tmp_path / "한글 경로 & 특수#문자"
    d.mkdir()
    sr = 48000
    t = np.arange(int(sr * 3)) / sr
    x = np.where(t > 1.0, 0.3 * np.sin(2 * np.pi * 330 * t), 0).astype(np.float32)
    path = d / "원본 노래.wav"
    sf.write(path, np.stack([x, -x], 1), sr)
    return str(path), x, sr


def test_separate_identity_keeps_length_and_alignment(stereo48k, tmp_path):
    path, x, sr = stereo48k
    out = tmp_path / "out"
    out.mkdir()
    events = []
    info = run_worker("separate", {"input": path, "out_dir": str(out), "model": "_identity"},
                      lambda f, m: events.append(f))
    assert info["frames"] == len(x) and info["sample_rate"] == sr and events
    v, vsr = sf.read(out / "vocals.wav", dtype="float32")
    inst, _ = sf.read(out / "instrumental.wav", dtype="float32")
    assert vsr == sr and v.shape == (len(x), 2) and inst.shape == v.shape
    seg = slice(int(1.5 * sr), int(1.5 * sr) + 4800)
    lags = [float(np.dot(v[seg.start + k:seg.stop + k, 0], x[seg])) for k in range(-50, 51)]
    assert int(np.argmax(lags)) - 50 == 0  # 원본과 시간 차 없음
    assert np.max(np.abs(v[:, 0] - x)) < 1e-3


def test_stems_commit_validate_and_detect_problems(stereo48k, tmp_path):
    path, x, sr = stereo48k
    tmp = tmp_path / "stems" / ".tmp-1"
    tmp.mkdir(parents=True)
    run_worker("separate", {"input": path, "out_dir": str(tmp), "model": "_identity"})
    stems_dir = tmp_path / "stems"
    m = stem_store.commit(tmp, stems_dir, "sha1", {"device": "cpu"})
    assert stem_store.validate(stems_dir, m, "sha1") == (True, "")
    assert m.offset_frames == 0 and m.frames == len(x) and m.sample_rate == sr
    assert stem_store.read_manifest(stems_dir) == m
    assert not stem_store.validate(stems_dir, m, "other-sha")[0]  # 원본이 바뀌면 무효
    with open(stems_dir / "vocals.wav", "ab") as f:
        f.write(b"\0" * 10)  # 손상(크기 변화)
    assert not stem_store.validate(stems_dir, stem_store.read_manifest(stems_dir), "sha1")[0]
    (stems_dir / "stems.json").write_text("{broken", encoding="utf-8")
    assert stem_store.read_manifest(stems_dir) is None


def test_worker_cancel_terminates_process(stereo48k, tmp_path):
    path, _, _ = stereo48k
    token = CancelToken()
    threading.Timer(1.5, token.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(AnalysisCancelled):
        run_worker("separate", {"input": path, "out_dir": str(tmp_path), "model": "_sleep"}, token=token)
    assert time.monotonic() - t0 < 10


def test_worker_error_codes(tmp_path):
    with pytest.raises(WorkerError) as e:
        run_worker("pitch", {"input": str(tmp_path / "없는 파일.mp3"), "out_dir": str(tmp_path)})
    assert e.value.code == "decode"
    bad = tmp_path / "x.wav"
    sf.write(bad, np.zeros((4800, 1), np.float32), 48000)
    with pytest.raises(WorkerError) as e:
        run_worker("separate", {"input": str(bad), "out_dir": str(tmp_path), "model": "no-such-model"})
    assert e.value.code == "internal" and "알 수 없는 분리 모델" in str(e.value)


def test_silent_input_skips_separation(tmp_path):
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros((48000, 2), np.float32), 48000)
    info = run_worker("separate", {"input": str(p), "out_dir": str(tmp_path), "model": "_identity"})
    assert info.get("silent_input") is True
    assert json.loads((tmp_path / "info.json").read_text())["silent_input"] is True
