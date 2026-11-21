"""STEP 5 모델 비교: 모델별 분리 결과 보관소, 순차 비교 실행, 채보 차이 지표.

실제 htdemucs·RoFormer 가중치는 쓰지 않는다. 분리 단계는 시험용 모델(_identity·_half·_oom·_crash)과 무작위 작은
RoFormer 체크포인트로, 피치·음표 단계는 실제 pYIN으로 돈다.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import threading
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from amp.core.model import AnalysisResult, Note, PitchCurve
from amp.core.tempo import TempoMap
from amp.engine import stems as stem_store
from amp.engine.model_store import ModelStore, identity_for, store_key

SR = 44100


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _in_process(calls: Counter, active: list | None = None, before=None):
    from amp.engine import worker

    lock = threading.Lock()

    def runner(command, args, progress=None, token=None, numba_cache_dir=None):
        calls[command] += 1
        if command == "separate":
            calls["model:" + args["model"]] += 1
            if before:
                before(args, token)
        if active is not None:
            with lock:
                active.append(command)
                calls["max_active"] = max(calls["max_active"], len(active))
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old
            if active is not None:
                with lock:
                    active.remove(command)

    return runner


@pytest.fixture()
def song(tmp_path):
    pytest.importorskip("librosa")
    from amp.devtools.synth import backing, sung

    voice = sung([(1.0, 1.8, 64), (1.9, 2.7, 67), (2.8, 4.0, 69)], 5.0)
    mix = np.stack([voice + 0.3 * backing(5.0)] * 2, 1).astype(np.float32)
    path = tmp_path / "곡 비교.wav"
    sf.write(path, mix, SR)
    return path


def _request(song, tmp_path, **kw):
    from amp.audio.decoder import load_audio
    from amp.engine.cache import file_content_key
    from amp.engine.settings import AnalysisSettings
    from amp.engine.transcription import TranscriptionRequest

    return TranscriptionRequest(load_audio(str(song)), str(song), file_content_key(str(song)),
                                AnalysisSettings(device="cpu"), str(tmp_path / "proj" / "stems"),
                                str(tmp_path / "proj" / "cache"), **kw)


# --- 보관소 -------------------------------------------------------------------------


def test_each_model_gets_its_own_entry_and_is_reused(song, tmp_path):
    """모델별 캐시 생성·재사용: 다른 모델로 분리해도 앞 결과 파일이 그대로고, 같은 모델은 다시 분리하지 않는다."""
    from amp.engine.transcription import TranscriptionPipeline

    calls = Counter()
    req = _request(song, tmp_path)
    out_a = TranscriptionPipeline(_in_process(calls), separation_model="_identity").run(req)
    store = ModelStore(req.stems_dir, req.source_sha256)
    (ea,) = store.entries()
    before = {p.name: _sha(p) for p in ea.path.glob("*.wav")}
    out_b = TranscriptionPipeline(_in_process(calls), separation_model="_half").run(req)
    assert calls["separate"] == 2
    entries = {e.model: e for e in store.entries()}
    assert set(entries) == {"_identity", "_half"} and entries["_half"].active
    assert {p.name: _sha(p) for p in entries["_identity"].path.glob("*.wav")} == before  # 덮어쓰지 않음
    assert out_a.manifest.stems_id != out_b.manifest.stems_id
    # 처음 모델로 돌아가면 다시 분리하지 않고 보관한 결과를 쓴다 (피치도 그 결과의 캐시를 쓴다)
    out = TranscriptionPipeline(_in_process(calls), separation_model="_identity").run(req)
    assert calls["separate"] == 2 and out.reused["separation"] and out.reused["pitch"]
    assert out.manifest.stems_id == out_a.manifest.stems_id
    assert store.active_key() == entries["_identity"].key
    # 위쪽 stems.json은 예전 형식 그대로: 이전 코드의 검증 함수로도 유효하다
    ok, why = stem_store.validate(req.stems_dir, stem_store.read_manifest(req.stems_dir), req.source_sha256)
    assert ok, why


def test_checkpoint_and_config_change_the_key(tmp_path):
    """체크포인트 내용이 바뀌면 다른 키, 같은 내용을 다른 경로로 옮기면 같은 키, 설정 파일이 다르면 다른 키."""
    a = tmp_path / "a.ckpt"
    a.write_bytes(b"weights-1" * 1000)
    a2 = tmp_path / "moved" / "MelBandRoformer.ckpt"
    a2.parent.mkdir()
    shutil.copy(a, a2)
    b = tmp_path / "b.ckpt"
    b.write_bytes(b"weights-2" * 1000)
    c1 = tmp_path / "c1.yaml"
    c1.write_text("model:\n  dim: 384\n", encoding="utf-8")
    c2 = tmp_path / "c2.yaml"
    c2.write_text("model:\n  dim: 256\n", encoding="utf-8")

    def key(ck, cfg=""):
        return store_key(identity_for("mel_roformer", "sha", {"checkpoint": str(ck), "config": str(cfg) if cfg else ""}))

    assert key(a) == key(a2)
    assert key(a) != key(b)
    assert key(a, c1) != key(a, c2) != key(a)
    assert key(a).startswith("mel_roformer-")
    # 모델 버전·옵션도 키에 들어간다 (Demucs 버전이 바뀌면 다른 키)
    ident = identity_for("htdemucs", "sha")
    assert {"demucs", "shifts", "overlap", "split"} <= set(ident)
    assert store_key(ident) != store_key({**ident, "demucs": "9.9"})
    assert store_key(identity_for("htdemucs", "sha")) != store_key(identity_for("htdemucs", "other-source"))


def _flat_stems(tmp_path, sha: str, model="htdemucs"):
    """이전 버전이 만든 평면 결과 (stems/vocals.wav + stems.json)."""
    stems = tmp_path / "proj" / "stems"
    tmp = stems / ".tmp-old"
    tmp.mkdir(parents=True)
    x = (np.random.default_rng(0).standard_normal((SR, 2)) * 0.1).astype(np.float32)
    for name in ("vocals", "instrumental"):
        sf.write(tmp / f"{name}.wav", x, SR, subtype="PCM_24")
    (tmp / "info.json").write_text(f'{{"model": "{model}", "sample_rate": {SR}, "frames": {SR}, "demucs": "4.1.0"}}',
                                   encoding="utf-8")
    m = stem_store.commit(tmp, stems, sha, {"device": "cpu"})
    shutil.rmtree(tmp)
    return stems, m


def test_legacy_flat_result_is_moved_not_overwritten(tmp_path):
    stems, old = _flat_stems(tmp_path, "s" * 64)
    before = {n: _sha(stems / f"{n}.wav") for n in ("vocals", "instrumental")}
    store = ModelStore(stems, "s" * 64)
    key = store.migrate_legacy()
    assert key and key.startswith("htdemucs-")
    assert not (stems / "vocals.wav").exists()
    e = store.entry(key)
    assert e.active and {n: _sha(e.path / f"{n}.wav") for n in before} == before
    m = store.pointer()
    assert m.stems_id == old.stems_id  # 분리 결과 id가 같아 피치 캐시를 그대로 쓴다
    assert stem_store.validate(stems, m, "s" * 64)[0]
    assert store.migrate_legacy() is None  # 한 번만


def test_project_copy_still_loads_entries(song, tmp_path):
    """프로젝트 이동·복사: 목록의 경로가 모두 stems/ 기준 상대 경로라 다른 위치에서도 그대로 읽힌다."""
    from amp.engine.transcription import TranscriptionPipeline

    req = _request(song, tmp_path)
    for model in ("_identity", "_half"):
        TranscriptionPipeline(_in_process(Counter()), separation_model=model).run(req)
    moved = tmp_path / "다른 곳" / "stems"
    shutil.copytree(req.stems_dir, moved)
    shutil.rmtree(tmp_path / "proj")
    store = ModelStore(moved, req.source_sha256)
    assert sorted(e.model for e in store.entries()) == ["_half", "_identity"]
    m = store.pointer()
    assert stem_store.validate(moved, m, req.source_sha256)[0]
    assert stem_store.load(moved, m)["vocals"].frames == req.audio.frames


def test_delete_only_unused_entry(song, tmp_path):
    from amp.engine.transcription import TranscriptionPipeline

    req = _request(song, tmp_path)
    for model in ("_identity", "_half"):
        TranscriptionPipeline(_in_process(Counter()), separation_model=model).run(req)
    store = ModelStore(req.stems_dir, req.source_sha256)
    active = store.active_key()
    with pytest.raises(ValueError):
        store.delete(active)
    other = next(e for e in store.entries() if not e.active)
    store.delete(other.key)
    assert [e.key for e in store.entries()] == [active]
    assert stem_store.validate(req.stems_dir, store.pointer(), req.source_sha256)[0]


# --- 비교 실행 -----------------------------------------------------------------------


def test_compare_runs_models_one_at_a_time_and_reuses(song, tmp_path):
    from amp.engine.model_compare import ModelSpec, run_compare

    calls = Counter()
    active = []
    req = _request(song, tmp_path)
    run = run_compare(req, [ModelSpec("_identity"), ModelSpec("_half")], runner=_in_process(calls, active))
    assert [o.status for o in run.outcomes] == ["done", "done"] and not run.cancelled
    assert calls["max_active"] == 1  # 작업 프로세스(=모델)는 한 번에 하나만
    assert calls["separate"] == 2 and calls["pitch"] == 2  # 피치 캐시는 분리 결과별로 따로
    # 지금 분석에 쓰는 결과는 바꾸지 않는다 (비교만 함)
    assert ModelStore(req.stems_dir, req.source_sha256).pointer() is None
    a, b = (o.output for o in run.outcomes)
    assert a.result.settings == b.result.settings  # 같은 피치·음표 설정
    assert a.result.models["separation"].name == "_identity" and b.result.models["separation"].name == "_half"
    again = run_compare(req, [ModelSpec("_identity"), ModelSpec("_half")], runner=_in_process(calls))
    assert [o.status for o in again.outcomes] == ["reused", "reused"]
    assert calls["separate"] == 2 and calls["pitch"] == 2  # 분리·피치 모두 재사용


def test_compare_cancel_keeps_finished_results(song, tmp_path):
    from amp.core.errors import AnalysisCancelled
    from amp.engine.model_compare import ModelSpec, run_compare

    calls = Counter()

    def before(args, token):
        if args["model"] == "_half":
            token.cancel()
            raise AnalysisCancelled()

    req = _request(song, tmp_path)
    run = run_compare(req, [ModelSpec("_identity"), ModelSpec("_half"), ModelSpec("_identity")],
                      runner=_in_process(calls, before=before), transcribe=False)
    assert run.cancelled
    assert [o.status for o in run.outcomes] == ["done", "cancelled", "cancelled"]
    store = ModelStore(req.stems_dir, req.source_sha256)
    assert [e.model for e in store.entries()] == ["_identity"]
    assert not list((store.root).glob(".tmp-*"))


def test_compare_oom_and_load_failure_do_not_stop_other_models(song, tmp_path):
    """실제 작업 프로세스로: GPU 메모리 부족은 CPU로 넘어가지 않고 'oom', 모델 파일 오류는 'failed', 나머지는 정상."""
    pytest.importorskip("torch")
    from amp.engine.model_compare import ModelSpec, run_compare

    bad = tmp_path / "broken.ckpt"
    bad.write_bytes(b"not a checkpoint")
    req = _request(song, tmp_path)
    run = run_compare(req, [ModelSpec("_oom", device="auto"), ModelSpec("mel_roformer", {"checkpoint": str(bad), "config": ""}),
                            ModelSpec("_identity")], transcribe=False)
    oom, failed, ok = run.outcomes
    assert oom.status == "oom" and oom.error_code == "oom_gpu" and "CPU" in oom.message
    assert failed.status == "failed" and failed.message
    assert ok.status == "done"
    assert [e.model for e in ModelStore(req.stems_dir, req.source_sha256).entries()] == ["_identity"]
    # 사용자가 'CPU로 다시'를 고르면 성공한다 (시험용 _oom은 CPU가 아닌 장치에서만 메모리 부족을 흉내 냄)
    again = run_compare(req, [ModelSpec("_oom", device="cpu")], transcribe=False)
    assert again.outcomes[0].status == "done"


def test_missing_roformer_file_is_reported_per_model(song, tmp_path):
    from amp.engine.model_compare import ModelSpec, run_compare

    calls = Counter()
    req = _request(song, tmp_path)
    run = run_compare(req, [ModelSpec("mel_roformer"), ModelSpec("_identity")], runner=_in_process(calls), transcribe=False)
    assert run.outcomes[0].status == "failed" and "모델 파일" in run.outcomes[0].message
    assert run.outcomes[1].status == "done" and calls["separate"] == 1


# --- 채보 차이 -----------------------------------------------------------------------


def _result(notes, f0=None, conf=None):
    curve = None
    if f0 is not None:
        curve = PitchCurve(0.0, 100.0, np.asarray(f0, float), np.asarray(conf if conf is not None else np.where(np.isfinite(f0), 0.8, 0.0)))
    return AnalysisResult(10.0, TempoMap.constant(120.0), tuple(notes), curve)


def _n(i, s, e, p):
    return Note(f"n{i}", s, e, p)


def test_note_diff_counts_pairs_pitch_timing_only_and_split():
    from amp.compare.model_diff import compare

    a = [_n(1, 0.0, 1.0, 60), _n(2, 1.0, 2.0, 62), _n(3, 2.0, 3.0, 64), _n(4, 4.0, 5.0, 67)]
    b = [_n(1, 0.02, 1.05, 60), _n(2, 1.0, 2.0, 63), _n(3, 2.0, 2.5, 64), _n(4, 2.5, 3.0, 64), _n(5, 6.0, 7.0, 70)]
    d = compare(_result(a), _result(b), "Demucs", "RoFormer")
    assert (d.notes_a, d.notes_b) == (4, 5)
    assert len(d.pairs) == 2
    assert [(p.a.pitch, p.b.pitch) for p in d.pitch_changed] == [(62, 63)]
    st = d.timing_stats()
    assert st["pairs"] == 2 and 0 < st["onset_median_ms"] <= 20 and st["offset_p90_ms"] <= 50
    assert [n.pitch for n in d.only_a] == [67] and [n.pitch for n in d.only_b] == [70]
    assert len(d.splits) == 1 and d.splits[0].single_side == "a" and len(d.splits[0].parts) == 2


def test_curve_diff_finds_octave_and_voicing_differences():
    from amp.compare.model_diff import compare, review_items

    n = 500
    f0a = np.full(n, np.nan)
    f0b = np.full(n, np.nan)
    f0a[0:200] = 220.0
    f0b[0:100] = 440.0  # 0~1초: 한 옥타브 위
    f0b[100:200] = 220.0
    f0b[300:400] = 330.0  # 3~4초: B에서만 유성
    d = compare(_result([], f0a), _result([], f0b), "A", "B", no_vocal_ranges=[(3.0, 4.0)])
    assert len(d.octave) == 1 and d.octave[0][0] == pytest.approx(0.0) and d.octave[0][1] == pytest.approx(1.0)
    assert d.octave[0][2] == pytest.approx(12.0)
    assert d.only_voiced_b == [(3.0, 4.0)] and d.only_voiced_a == []
    assert d.voiced_a == pytest.approx(0.4) and d.voiced_b == pytest.approx(0.6)
    kinds = Counter(it.kind for it in review_items(d))
    assert kinds["model_octave"] == 1 and kinds["model_voiced_b"] == 1


def test_diff_wording_never_claims_accuracy():
    from amp.compare.model_diff import DISCLAIMER, compare, review_items

    a = [_n(1, 0.0, 1.0, 60)]
    b = [_n(1, 0.0, 1.0, 72), _n(2, 2.0, 3.0, 65), _n(3, 3.1, 3.5, 65), _n(4, 3.6, 3.9, 66)]
    d = compare(_result(a), _result(b), "A", "B", no_vocal_ranges=[(2.0, 4.0)])
    text = " ".join(" ".join(r) for r in d.summary_rows()) + " ".join(" ".join(r) for r in d.shared_rows())
    text += " ".join(it.reason for it in review_items(d))
    for word in ("정확도", "더 좋", "우수", "승", "점수"):
        assert word not in text
    assert "정확하다는 뜻이 아닙니다" in DISCLAIMER
    items = review_items(d)
    assert {"model_pitch", "model_only_b", "model_novocal_b"} <= {it.kind for it in items}
    assert all(it.category == "model" and not it.note_ids for it in items)  # 작업본 음표 id를 건드리지 않는다


# --- 코드 검토에서 찾은 문제 ----------------------------------------------------------


def test_audio_cache_token_is_unique_even_when_id_is_reused():
    """느린 재생·스펙트로그램 캐시 키에 id()를 쓰면, 사라진 오디오의 id를 새 오디오가 받아 옛 캐시가 붙었다."""
    import gc

    from amp.audio.data import AudioData

    seen = set()
    for _ in range(50):
        a = AudioData(np.zeros((10, 2), np.float32), SR)
        assert a.token not in seen
        seen.add(a.token)
        del a
        gc.collect()


def test_migrated_legacy_demucs_result_is_reused_by_compare(song, tmp_path):
    """이전 버전 프로젝트의 htdemucs 결과는 옮긴 뒤에도 비교 실행에서 다시 분리하지 않는다."""
    import json

    from amp.engine.model_compare import ModelSpec, run_compare
    from amp.engine.model_store import package_version

    req = _request(song, tmp_path)
    stems = tmp_path / "proj" / "stems"
    tmp = stems / ".tmp-old"
    tmp.mkdir(parents=True)
    a = req.audio
    for name in ("vocals", "instrumental"):
        sf.write(tmp / f"{name}.wav", a.samples * 0.5, a.sample_rate, subtype="PCM_24")
    (tmp / "info.json").write_text(json.dumps({"model": "htdemucs", "sample_rate": a.sample_rate, "frames": a.frames,
                                               "demucs": package_version("demucs")}), encoding="utf-8")
    stem_store.commit(tmp, stems, req.source_sha256, {"device": "cpu"})
    shutil.rmtree(tmp)
    calls = Counter()
    run = run_compare(req, [ModelSpec("htdemucs")], runner=_in_process(calls), transcribe=False)
    assert run.outcomes[0].status == "reused" and calls["separate"] == 0
    assert [e.model for e in ModelStore(stems, req.source_sha256).entries()] == ["htdemucs"]


def test_forced_rerun_of_active_entry_refreshes_pointer(song, tmp_path):
    from amp.engine.model_compare import ModelSpec, run_compare
    from amp.engine.transcription import TranscriptionPipeline

    req = _request(song, tmp_path)
    TranscriptionPipeline(_in_process(Counter()), separation_model="_identity").run(req)
    store = ModelStore(req.stems_dir, req.source_sha256)
    before = store.pointer().created
    import time

    time.sleep(1.1)  # created는 초 단위
    run_compare(req, [ModelSpec("_identity")], runner=_in_process(Counter()), transcribe=False, force=True)
    entry = store.entry(store.active_key())
    assert store.pointer().created == entry.manifest.created != before
    assert store.pointer().stems_id == entry.manifest.stems_id


def test_model_diff_review_keys_are_unique_for_nearby_items():
    from amp.compare.model_diff import compare, review_items

    a = [_n(1, 12.31, 12.36, 60), _n(2, 12.34, 12.40, 62)]
    d = compare(_result(a), _result([]), "A", "B")
    items = [it for it in review_items(d) if it.kind == "model_only_a"]
    assert len(items) == 2 and len({it.key for it in items}) == 2
