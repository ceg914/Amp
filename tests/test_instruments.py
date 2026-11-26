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


def test_extract_from_instrumental_adds_to_active_result(song, tmp_path):
    """v0.6.7: 보컬 분석에 쓰는 분리 결과의 반주를 분리해 그 결과 폴더에 악기를 더한다. 새 '보컬' 결과는 만들지 않는다."""
    path, x = song
    sha = "d" * 64
    stems = tmp_path / "stems"
    calls: Counter = Counter()
    run = _runner(calls)
    with pytest.raises(Exception, match="보컬 분석"):
        I.extract(stems, sha, path, "_six", run, names=("drums",), source="instrumental")  # 분리 결과가 없음
    store = ModelStore(stems, sha)
    tmp = store.new_tmp()
    run("separate", {"input": path, "out_dir": str(tmp), "model": "_half"})  # 보컬 = 입력의 절반, 반주 = 나머지 절반
    entry = store.commit(tmp, identity_for("_half", sha), {"device": "cpu"})
    store.activate(entry.key)
    before = {n: (entry.path / f"{n}.wav").read_bytes() for n in ("vocals", "instrumental")}

    r = I.extract(stems, sha, path, "_six", run, names=("drums", "guitar"), source="instrumental")
    assert r.key == entry.key and r.added == ["drums", "guitar"] and r.source_model == "_half"
    assert len(store.entries()) == 1  # 반주를 분리하며 나온 보컬(남은 소리)로 새 결과를 만들지 않는다
    after = store.entry(entry.key)
    assert {n: (after.path / f"{n}.wav").read_bytes() for n in before} == before
    d, _ = sf.read(after.path / "drums.wav", dtype="float32", always_2d=True)
    assert np.max(np.abs(d[:, 0] - 0.15 * x)) < 3e-3  # 반주(0.5)의 드럼 몫(0.3) = 입력의 0.15: 원본이 아니라 반주에서 뽑았다
    meta = after.manifest.files["drums"]
    assert meta["made_by"] == "_six" and meta["input"] == "instrumental"
    assert "drums" in store.pointer().files  # 지금 쓰는 결과라 위쪽 목록에도 들어간다
    found = I.find_stems(stems, sha)
    assert found["drums"].from_instrumental and found["drums"].model == "_six" and "반주에서" in found["drums"].model_label
    r2 = I.extract(stems, sha, path, "_six", run, names=("drums", "guitar"), source="instrumental")
    assert not r2.separated and calls["separate"] == 2


# --- v0.7.0: 입력별 악기 결과 구분 (명세 15.1, 14.3) --------------------------------------------------

def _vocal_entry(stems, sha, path, run, model="_six"):
    store = ModelStore(stems, sha)
    tmp = store.new_tmp()
    run("separate", {"input": path, "out_dir": str(tmp), "model": model})
    entry = store.commit(tmp, identity_for(model, sha), {"device": "cpu"})
    store.activate(entry.key)
    return store, entry


def test_instrumental_request_not_skipped_by_original_stem(song, tmp_path):
    """보컬 분리가 원본에서 만든 베이스가 있어도 반주에서 베이스를 요청하면 분리한다. 원본 베이스 파일은 그대로."""
    path, x = song
    sha = "f" * 64
    stems = tmp_path / "stems"
    calls: Counter = Counter()
    run = _runner(calls)
    store, entry = _vocal_entry(stems, sha, path, run)
    orig_bass = (entry.path / "bass.wav").read_bytes()
    r = I.extract(stems, sha, path, "_six", run, names=("bass", "drums"), source="instrumental")
    assert r.separated and sorted(r.added) == ["bass", "drums"] and r.kept == [] and r.input == "instrumental"
    assert calls["extras:bass,drums"] == 1
    e = store.entry(entry.key)
    assert (e.path / "bass.wav").read_bytes() == orig_bass  # 원본에서 만든 베이스는 덮어쓰지 않는다
    variants = {(v.name, v.input) for v in I.stem_variants(e)}
    assert {("bass", "original"), ("bass", "instrumental"), ("drums", "instrumental")} <= variants
    inst = next(v for v in I.stem_variants(e) if v.name == "bass" and v.input == "instrumental")
    assert inst.slot == "bass@instrumental-_six" and inst.input_sha256 == file_sha(e.path / "instrumental.wav")
    b, _ = sf.read(inst.path, dtype="float32", always_2d=True)
    assert np.max(np.abs(b[:, 0] - 0.2 * 0.9 * x)) < 3e-3  # 반주(_six: 입력의 0.9)의 베이스 몫(0.2)
    assert [v.input for v in I.find_all_stems(stems, sha)["bass"]] == ["original", "instrumental"]
    assert I.find_stems(stems, sha, prefer_input="instrumental")["bass"].input == "instrumental"
    r2 = I.extract(stems, sha, path, "_six", run, names=("bass", "drums"), source="instrumental")
    assert not r2.separated and sorted(r2.kept) == ["bass", "drums"]


def test_original_and_instrumental_requests_are_independent(song, tmp_path):
    """원본에서 뽑은 드럼이 있어도 반주에서, 반주에서 뽑은 드럼이 있어도 원본에서 요청하면 분리한다."""
    path, _x = song
    sha = "1" * 64
    run = _runner(Counter())
    stems_a = tmp_path / "a"
    store, entry = _vocal_entry(stems_a, sha, path, run)
    assert I.extract(stems_a, sha, path, "_six", run, names=("drums",), source="original").added == ["drums"]
    r = I.extract(stems_a, sha, path, "_six", run, names=("drums",), source="instrumental")
    assert r.added == ["drums"] and r.kept == []
    stems_b = tmp_path / "b"
    store, entry = _vocal_entry(stems_b, sha, path, run)
    I.extract(stems_b, sha, path, "_six", run, names=("drums",), source="instrumental")
    before = (store.entry(entry.key).path / "drums.wav").read_bytes()
    r = I.extract(stems_b, sha, path, "_six", run, names=("drums",), source="original")
    assert r.added == ["drums"] and r.kept == []
    e = store.entry(entry.key)
    assert (e.path / "drums.wav").read_bytes() == before  # 반주에서 만든 드럼은 그대로
    assert sorted(v.input for v in I.stem_variants(e) if v.name == "drums") == ["instrumental", "original"]


def test_instrumental_request_checks_model(song, tmp_path):
    """반주에서 _withbass로 만든 베이스가 있어도 _six로 요청하면 _six로 분리한다 (모델이 다르면 다른 결과)."""
    path, _x = song
    sha = "2" * 64
    calls: Counter = Counter()
    run = _runner(calls)
    store, entry = _vocal_entry(tmp_path / "s", sha, path, run, model="_half")
    assert I.extract(tmp_path / "s", sha, path, "_withbass", run, names=("bass",), source="instrumental").added == ["bass"]
    r = I.extract(tmp_path / "s", sha, path, "_six", run, names=("bass",), source="instrumental")
    assert r.added == ["bass"] and r.separated
    models = sorted(v.model for v in I.stem_variants(store.entry(entry.key)) if v.name == "bass")
    assert models == ["_six", "_withbass"]


def test_reseparation_keeps_other_stems_and_marks_stale(song, tmp_path):
    """같은 모델로 보컬을 다시 분리해도(사용자의 '다시 분리') 반주에서 만든 악기 파일을 지우지 않고 '오래됨'으로 표시한다.
    다시 만든 베이스가 기본 자리를 쓰면, 그 자리에 있던 다른 입력의 파일은 이름을 바꿔 옮긴다."""
    path, _x = song
    sha = "3" * 64
    stems = tmp_path / "stems"
    run = _runner(Counter())
    store = ModelStore(stems, sha)
    tmp = store.new_tmp()
    run("separate", {"input": path, "out_dir": str(tmp), "model": "_half"})
    entry = store.commit(tmp, identity_for("_half", sha), {"device": "cpu"})
    store.activate(entry.key)
    I.extract(stems, sha, path, "_six", run, names=("drums", "bass"), source="instrumental")
    e = store.entry(entry.key)
    drums = (e.path / "drums.wav").read_bytes()
    inst_bass = (e.path / "bass.wav").read_bytes()  # 보컬 분리(_half)는 베이스를 내지 않아 기본 자리에 들어감
    assert e.manifest.files["bass"]["input"] == "instrumental"

    tmp = store.new_tmp()  # '다시 분리': 같은 키, 이번에는 베이스를 내는 결과(시험용으로 _six 출력을 _half로 적음)
    run("separate", {"input": path, "out_dir": str(tmp), "model": "_six"})  # 기본: 보컬·반주·베이스
    import json

    info = json.loads((tmp / "info.json").read_text(encoding="utf-8"))
    info["model"] = "_half"
    (tmp / "info.json").write_text(json.dumps(info), encoding="utf-8")
    store.commit(tmp, identity_for("_half", sha), {"device": "cpu"})
    e = store.entry(entry.key)
    assert (e.path / "drums.wav").read_bytes() == drums  # 지우지 않음
    moved = [s for s, v in e.manifest.files.items() if s.startswith("bass@")]
    assert moved == ["bass@instrumental-_six"]
    assert (e.path / "bass@instrumental-_six.wav").read_bytes() == inst_bass  # 자리만 옮김
    assert e.manifest.files["bass"].get("input") is None  # 기본 자리 = 이번 보컬 분리의 베이스
    variants = {(v.name, v.input): v for v in I.stem_variants(e)}
    assert file_sha(e.path / "instrumental.wav") != variants[("drums", "instrumental")].input_sha256  # 반주가 바뀜
    assert variants[("drums", "instrumental")].stale and variants[("bass", "instrumental")].stale
    assert not variants[("bass", "original")].stale and "오래됨" in variants[("drums", "instrumental")].model_label
    assert I.find_stems(stems, sha)["bass"].input == "original"  # 오래되지 않은 것을 먼저
    r = I.extract(stems, sha, path, "_six", run, names=("drums",), source="instrumental")  # 오래된 것은 '있음'이 아니다
    assert r.added == ["drums"]
    e = store.entry(entry.key)
    assert (e.path / "drums.wav").read_bytes() == drums  # 오래된 파일도 남긴다 (지우려면 사용자가 고름)
    fresh = I.find_stems(stems, sha)["drums"]
    assert not fresh.stale and fresh.slot == "drums@instrumental-_six"


def file_sha(p):
    import hashlib

    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_bass_pipeline_reads_input_specific_bass(song, tmp_path):
    """베이스 TAB: 같은 폴더의 원본 베이스·반주 베이스는 캐시 입력 id가 다르다 (같은 모델이라도 다른 입력)."""
    from amp.engine.bass_pipeline import bass_input_id, find_bass_stem

    path, _x = song
    sha = "4" * 64
    stems = tmp_path / "stems"
    run = _runner(Counter())
    store, entry = _vocal_entry(stems, sha, path, run)
    key, p_orig = find_bass_stem(stems, sha)
    assert key == entry.key and p_orig.name == "bass.wav"
    id_orig = bass_input_id(stems, sha, key, p_orig)
    assert id_orig == bass_input_id(stems, sha, key)  # 기본 자리는 v0.6 캐시 id와 같다
    I.extract(stems, sha, path, "_six", run, names=("bass",), source="instrumental")
    inst = next(v for v in I.stem_variants(store.entry(key)) if v.input == "instrumental")
    assert bass_input_id(stems, sha, key, inst.path) != id_orig
