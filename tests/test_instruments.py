"""v0.6.4 악기 소리 분리: 고른 악기만 저장, 보관 결과에 빠진 악기만 더하기(보컬·있는 파일 그대로), 다시 분리하지 않기,
악기별 파일 찾기 순서. 분리 모델은 시험용(_six: 입력을 여섯으로 나눈 비율)이다."""

from __future__ import annotations

import io
import threading
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from amp.engine import instruments as I
from amp.engine.model_store import ModelStore, identity_for, store_key

SR = 44100


@pytest.fixture
def song(tmp_path):
    t = np.arange(int(SR * 2.0)) / SR
    x = (0.4 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    p = tmp_path / "악기 곡.wav"
    sf.write(p, np.stack([x, x], 1), SR)
    return str(p), x


def _runner(calls: Counter):
    from amp.engine import worker

    lock = threading.Lock()

    def run(command, args, progress=None, token=None, numba_cache_dir=None):
        with lock:
            calls[command] += 1
            calls["extras:" + ",".join(args.get("extras") or ())] += 1
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    return run


def test_worker_keeps_only_requested_instruments(song, tmp_path):
    path, x = song
    out = tmp_path / "out"
    out.mkdir()
    _runner(Counter())("separate", {"input": path, "out_dir": str(out), "model": "_six", "extras": ["drums", "guitar"]})
    assert sorted(p.name for p in out.glob("*.wav")) == ["drums.wav", "guitar.wav", "instrumental.wav", "vocals.wav"]
    d, sr = sf.read(out / "drums.wav", dtype="float32", always_2d=True)
    assert sr == SR and d.shape == (len(x), 2)
    assert np.max(np.abs(d[:, 0] - 0.3 * x)) < 2e-3  # _six: 드럼 = 입력의 0.3 (시간 어긋남 없음)
    default = tmp_path / "default"
    default.mkdir()
    _runner(Counter())("separate", {"input": path, "out_dir": str(default), "model": "_six"})
    assert (default / "bass.wav").is_file() and not (default / "drums.wav").exists()  # 기본은 베이스만 (v0.6.3과 같음)


def test_extract_adds_missing_instruments_only(song, tmp_path):
    path, _x = song
    sha = "a" * 64
    stems = tmp_path / "stems"
    calls: Counter = Counter()
    run = _runner(calls)
    r1 = I.extract(stems, sha, path, "_six", run, names=("drums", "bass", "other"))
    assert r1.separated and r1.added == ["drums", "bass", "other"] and r1.key == store_key(identity_for("_six", sha))
    store = ModelStore(stems, sha)
    assert store.pointer() is None  # 지금 쓰는 분리 결과를 정하지 않는다 (보컬 분석 소스는 그대로)
    entry = store.entry(r1.key)
    before = {n: (entry.path / f"{n}.wav").read_bytes() for n in ("vocals", "instrumental", "drums")}

    r2 = I.extract(stems, sha, path, "_six", run, names=("drums", "bass", "guitar", "piano", "other"))
    assert r2.added == ["guitar", "piano"] and r2.kept == ["drums", "bass", "other"]
    assert calls["extras:guitar,piano"] == 1  # 빠진 악기만 요청
    entry = store.entry(r1.key)
    assert {n: (entry.path / f"{n}.wav").read_bytes() for n in before} == before  # 보컬·반주·있던 악기 그대로
    found = I.find_stems(stems, sha)
    assert set(found) == set(I.INSTRUMENTS) and all(f.key == r1.key for f in found.values())
    assert len(store.entries()) == 1

    r3 = I.extract(stems, sha, path, "_six", run, names=("guitar", "piano"))
    assert not r3.separated and calls["separate"] == 2  # 다 있으면 다시 분리하지 않는다


def test_find_prefers_active_result(song, tmp_path):
    path, _x = song
    sha = "b" * 64
    stems = tmp_path / "stems"
    run = _runner(Counter())
    a = I.extract(stems, sha, path, "_six", run, names=("drums",))
    b = I.extract(stems, sha, path, "_withbass", run, names=("bass",))
    store = ModelStore(stems, sha)
    store.activate(b.key)
    found = I.find_stems(stems, sha)
    assert found["drums"].key == a.key and found["bass"].key == b.key
    store.activate(a.key)
    assert I.find_stems(stems, sha)["drums"].key == a.key


def test_six_source_identity_matches_htdemucs_options():
    six = identity_for("htdemucs_6s", "c" * 64)
    four = identity_for("htdemucs", "c" * 64)
    assert {k: v for k, v in six.items() if k != "model"} == {k: v for k, v in four.items() if k != "model"}
    assert set(four) == {"model", "source", "demucs", "shifts", "overlap", "split", "shift_seed", "normalize"}
    assert store_key(six) != store_key(four)
