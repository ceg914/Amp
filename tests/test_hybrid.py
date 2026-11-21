"""v0.6 Hybrid 보컬 소스: 구간 편집, 결합(샘플 경계·크로스페이드·이득·잘림 방지), 상태 검사, 보관·재사용,
저장·다시 열기, 피치 청크 재사용(전체 계산과 같은 결과), 음표 id 이어받기.

실제 분리 모델은 쓰지 않는다. 보관소 항목은 합성 신호로 직접 만들거나 시험용 분리 모델(_identity·_half)로 만든다.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from amp.engine import hybrid as H
from amp.engine.model_store import ModelStore

SR = 8000
FRAMES = SR * 4  # 4초


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _q24(x: np.ndarray) -> np.ndarray:
    """PCM 24bit로 정확히 표현되는 값으로 맞춘다 (파일에 쓰고 읽은 값과 같게)."""
    return (np.round(np.clip(x, -1, 1 - 2 ** -23) * 2 ** 23) / 2 ** 23).astype(np.float32)


def _entry(store: ModelStore, model: str, vocals: np.ndarray, inst: np.ndarray | None = None, tag: str = "",
           sr: int = SR):
    tmp = store.new_tmp()
    sf.write(tmp / "vocals.wav", vocals, sr, subtype="PCM_24")
    sf.write(tmp / "instrumental.wav", inst if inst is not None else vocals * 0.5, sr, subtype="PCM_24")
    (tmp / "info.json").write_text(json.dumps({"model": model, "sample_rate": sr, "frames": int(vocals.shape[0])}),
                                   encoding="utf-8")
    return store.commit(tmp, {"model": model, "source": store.source_sha256, "tag": tag}, {})


def _sine(freq: float, amp: float = 0.5, phase: float = 0.0, frames: int = FRAMES, ch: int = 2) -> np.ndarray:
    t = np.arange(frames) / SR
    x = amp * np.sin(2 * np.pi * freq * t + phase)
    return _q24(np.stack([x] * ch, 1))


@pytest.fixture()
def two(tmp_path):
    store = ModelStore(tmp_path / "stems", "src")
    a = _entry(store, "htdemucs", _sine(220, 0.5))
    b = _entry(store, "mel_roformer", _sine(330, 0.4, phase=1.0))
    return store, a, b


def _plan_ab(a, b, t: float = 2.0, cf: float = H.DEFAULT_CROSSFADE_S) -> H.HybridPlan:
    p = H.single(SR, FRAMES, a.key, a.model, a.manifest.stems_id)
    p = H.split(p, p.segments[0].segment_id, t)
    p = H.set_source(p, [p.segments[1].segment_id], b.key, b.model, b.manifest.stems_id)
    return H.set_crossfade(p, [p.segments[1].segment_id], cf)


def _read(entry, stem="vocals") -> np.ndarray:
    x, _ = sf.read(str(entry.path / entry.manifest.files[stem]["path"]), dtype="float32", always_2d=True)
    return x


# --- 구간 편집 --------------------------------------------------------------------

def test_edit_operations_keep_segments_contiguous():
    p = H.single(SR, FRAMES, "A", "htdemucs")
    s0 = p.segments[0].segment_id
    p = H.split(p, s0, 1.0)
    p = H.split(p, p.segments[1].segment_id, 3.0)
    assert [(s.start_frame, s.end_frame) for s in p.segments] == [(0, SR), (SR, 3 * SR), (3 * SR, FRAMES)]
    assert p.segments[0].segment_id == s0  # 나눈 왼쪽은 id를 지킨다
    p = H.move_boundary(p, 1, 1.5)
    assert p.segments[0].end_frame == p.segments[1].start_frame == int(1.5 * SR)
    p = H.move_boundary(p, 1, 10.0)  # 다음 구간을 넘어가지 못한다 (최소 길이 남김)
    assert p.segments[1].frames == round(H.MIN_SEGMENT_S * SR)
    p = H.set_source(p, [p.segments[1].segment_id], "B", "mel_roformer")
    assert [s.source_key for s in p.segments] == ["A", "B", "A"]
    m = H.merge(p, [p.segments[1].segment_id, p.segments[2].segment_id])
    assert len(m.segments) == 2 and m.segments[1].source_key == "B" and m.segments[1].end_frame == FRAMES
    m2 = H.merge(p, [p.segments[1].segment_id, p.segments[2].segment_id], keep=p.segments[2].segment_id)
    assert m2.segments[1].source_key == "A"
    with pytest.raises(H.HybridError):
        H.merge(p, [p.segments[0].segment_id, p.segments[2].segment_id])  # 이웃이 아님
    d = H.delete(p, p.segments[1].segment_id)
    assert [(s.start_frame, s.end_frame) for s in d.segments] == [(0, p.segments[1].end_frame),
                                                                (p.segments[2].start_frame, FRAMES)]
    d0 = H.delete(p, p.segments[0].segment_id)
    assert d0.segments[0].start_frame == 0 and d0.segments[0].source_key == "B"
    with pytest.raises(H.HybridError):
        H.delete(H.single(SR, FRAMES, "A", "htdemucs"), H.single(SR, FRAMES, "A", "x").segments[0].segment_id)
    allb = H.set_all(p, "B", "mel_roformer")
    assert {s.source_key for s in allb.segments} == {"B"} and len(allb.segments) == 3
    for q in (p, m, m2, d, d0, allb):
        assert H.validate(q) == []


def test_create_segment_inside_and_across():
    p = H.single(SR, FRAMES, "A", "htdemucs")
    p, sid = H.create(p, 1.0, 2.0, "B", "mel_roformer")
    assert [(s.start_frame, s.end_frame, s.source_key) for s in p.segments] == [
        (0, SR, "A"), (SR, 2 * SR, "B"), (2 * SR, FRAMES, "A")]
    assert p.get(sid).source_key == "B"
    assert p.segments[0].segment_id != p.segments[2].segment_id
    p2, _ = H.create(p, 0.5, 2.5, "C", "external")  # 앞 구간을 자르고 가운데 구간을 덮고 뒤 구간을 자른다
    assert [(s.start_frame, s.source_key) for s in p2.segments] == [(0, "A"), (SR // 2, "C"), (int(2.5 * SR), "A")]
    p3, _ = H.create(p, 1.0 + 0.01, 3.0, "C", "external")  # 1.0~1.01의 10ms 조각은 남기지 않는다
    assert all(s.frames >= round(H.MIN_SEGMENT_S * SR) for s in p3.segments)
    assert H.validate(p3) == []
    with pytest.raises(H.HybridError):
        H.create(p, 1.0, 1.01, "C", "external")


def test_validate_rejects_gaps_overlaps_and_bad_values():
    seg = H.HybridSegment
    gap = H.HybridPlan(SR, FRAMES, (seg("a", 0, 100, "A", "m"), seg("b", 120, FRAMES, "A", "m")))
    over = H.HybridPlan(SR, FRAMES, (seg("a", 0, 130, "A", "m"), seg("b", 120, FRAMES, "A", "m")))
    short = H.HybridPlan(SR, FRAMES, (seg("a", 0, FRAMES - 5, "A", "m"),))
    dup = H.HybridPlan(SR, FRAMES, (seg("a", 0, 100, "A", "m"), seg("a", 100, FRAMES, "A", "m")))
    bad_cf = H.HybridPlan(SR, FRAMES, (seg("a", 0, 100, "A", "m"), seg("b", 100, FRAMES, "A", "m", crossfade_s=2.0)))
    bad_gain = H.HybridPlan(SR, FRAMES, (seg("a", 0, FRAMES, "A", "m", gain_db=40.0),))
    assert any("빈틈" in x for x in H.validate(gap))
    assert any("겹칩니다" in x for x in H.validate(over))
    assert any("끝에서 끝나지" in x for x in H.validate(short))
    assert any("id" in x for x in H.validate(dup))
    assert H.validate(bad_cf) and H.validate(bad_gain)
    with pytest.raises(H.HybridError):
        H.plan_from_dict(H.plan_to_dict(gap))


def test_changed_ranges_cover_only_the_edited_segment():
    p = H.single(SR, FRAMES, "A", "htdemucs")
    p = H.split(p, p.segments[0].segment_id, 1.0)
    p = H.split(p, p.segments[1].segment_id, 2.0)
    q = H.set_source(p, [p.segments[1].segment_id], "B", "mel_roformer")
    r = H.changed_ranges(p, q)
    assert len(r) == 1
    a, b = r[0]
    half = H.DEFAULT_CROSSFADE_S / 2
    assert a == pytest.approx(1.0 - half, abs=2 / SR) and b == pytest.approx(2.0 + half, abs=2 / SR)
    assert H.changed_ranges(p, p) == []
    assert H.changed_ranges(None, p) == [(0.0, p.duration)]


# --- 결합 -------------------------------------------------------------------------

def test_render_combines_two_models_with_exact_boundaries(two):
    store, a, b = two
    before = {f: _sha(e.path / f) for e in (a, b) for f in ("vocals.wav", "instrumental.wav")}
    plan = _plan_ab(a, b, 2.0, cf=0.0)  # 교차 없음: 경계 샘플이 정확한지 본다
    res = H.render(plan, store)
    out, va, vb = _read(res.entry), _read(a), _read(b)
    bnd = 2 * SR
    assert out.shape == va.shape
    assert np.array_equal(out[:bnd], va[:bnd])  # 비트 단위로 같다
    assert np.array_equal(out[bnd:], vb[bnd:])
    inst = _read(res.entry, "instrumental")
    assert np.array_equal(inst[:bnd], _read(a, "instrumental")[:bnd])
    assert res.entry.model == H.HYBRID and res.entry.manifest.frames == FRAMES
    assert res.entry.manifest.sample_rate == SR and res.entry.manifest.channels == 2
    after = {f: _sha(e.path / f) for e in (a, b) for f in ("vocals.wav", "instrumental.wav")}
    assert before == after  # 원본 stem은 바뀌지 않는다
    assert store.active_key() is None  # 만들기만 하고 분석 소스로 바꾸지 않는다


def test_crossfade_is_linear_and_removes_the_click(two):
    store, a, b = two
    va, vb = _read(a), _read(b)
    hard = _read(H.render(_plan_ab(a, b, 2.0, cf=0.0), store).entry)
    soft_plan = _plan_ab(a, b, 2.0, cf=0.03)
    soft = _read(H.render(soft_plan, store).entry)
    bnd, h = 2 * SR, int(round(0.03 * SR / 2))
    assert H.fade_halves(soft_plan) == [0, h]
    lo, hi = bnd - h, bnd + h
    w = ((np.arange(2 * h) + 0.5) / (2 * h))[:, None]
    expect = va[lo:hi].astype(np.float64) * (1 - w) + vb[lo:hi].astype(np.float64) * w
    assert np.max(np.abs(soft[lo:hi] - expect)) < 2 ** -22  # 24bit 반올림 이내
    assert np.array_equal(soft[:lo], va[:lo]) and np.array_equal(soft[hi:], vb[hi:])

    def jump(x):
        return float(np.max(np.abs(np.diff(x[bnd - 50:bnd + 50, 0]))))

    normal = float(np.max(np.abs(np.diff(va[:SR, 0]))))  # 220Hz 사인의 보통 표본 간 변화
    assert jump(hard) > 3 * normal  # 교차가 없으면 경계에서 튄다 (클릭)
    assert jump(soft) <= 1.05 * max(normal, float(np.max(np.abs(np.diff(vb[:SR, 0])))))


def test_crossfade_is_clamped_to_half_of_short_segments(two):
    store, a, b = two
    p = _plan_ab(a, b, 2.0, cf=0.4)
    p = H.split(p, p.segments[1].segment_id, 2.1)  # 100ms 구간
    p = H.set_crossfade(p, None, 0.4)
    halves = H.fade_halves(p)
    assert halves[1] <= int(0.1 * SR) // 2 and halves[2] <= int(0.1 * SR) // 2
    out = _read(H.render(p, store).entry)
    assert np.isfinite(out).all()


def test_gain_and_clipping_protection(two):
    store, a, b = two
    p = _plan_ab(a, b, 2.0, cf=0.0)
    p6 = H.set_gain(p, [p.segments[1].segment_id], -6.0)
    out = _read(H.render(p6, store).entry)
    ratio = np.max(np.abs(out[2 * SR + 100:])) / np.max(np.abs(_read(b)[2 * SR + 100:]))
    assert ratio == pytest.approx(10 ** (-6 / 20), rel=1e-3)
    loud = H.set_gain(p, [p.segments[0].segment_id], 12.0)  # 0.5 × 4 = 2.0 → 잘림
    res = H.render(loud, store)
    out = _read(res.entry)
    assert np.max(np.abs(out)) <= H.PEAK_TARGET + 1e-6
    assert res.peak_scale_db["vocals"] < 0
    assert res.entry.manifest.model_info["peak_scale_db"]["vocals"] == res.peak_scale_db["vocals"]


def test_level_match_brings_models_to_the_quieter_level(two):
    store, a, b = two
    p = dataclasses.replace(_plan_ab(a, b, 2.0, cf=0.0), level_match=True)
    res = H.render(p, store)
    assert res.gains[a.key] < 1.0 and res.gains[b.key] == pytest.approx(1.0)  # 큰 A를 작은 B에 맞춤
    out = _read(res.entry)
    assert np.max(np.abs(out[:SR])) == pytest.approx(np.max(np.abs(out[3 * SR:])), rel=0.02)


def test_mismatched_sample_rate_or_channels_are_rejected(tmp_path):
    store = ModelStore(tmp_path / "stems", "src")
    a = _entry(store, "htdemucs", _sine(220))
    mono = _entry(store, "mel_roformer", _sine(330, ch=1), tag="mono")
    other_sr = _entry(store, "external", _sine(330, frames=FRAMES * 2), sr=SR * 2, tag="sr")
    p = _plan_ab(a, mono)
    with pytest.raises(H.HybridError, match="채널"):
        H.render(p, store)
    p2 = _plan_ab(a, other_sr)
    status = H.check_sources(p2, {e.key: e for e in store.entries()})
    assert status[p2.segments[1].segment_id].state == H.STATUS_CHECK
    with pytest.raises(H.HybridError, match="검증"):
        H.render(p2, store)


def test_missing_or_changed_source_needs_check(two):
    store, a, b = two
    plan = _plan_ab(a, b)
    entries = {e.key: e for e in store.entries()}
    ok = H.check_sources(plan, entries)
    assert all(s.ok for s in ok.values())
    # 체크포인트가 바뀌어 RoFormer의 지금 키가 다르다
    st = H.check_sources(plan, entries, current_keys={"mel_roformer": "mel_roformer-other"})
    assert st[plan.segments[1].segment_id].state == H.STATUS_CHECK and "체크포인트" in st[plan.segments[1].segment_id].reason
    assert st[plan.segments[0].segment_id].ok
    # 같은 키가 다시 만들어짐 (결과 id가 달라짐)
    regen = dataclasses.replace(plan.segments[1], source_stems_id="old-id")
    p2 = dataclasses.replace(plan, segments=(plan.segments[0], regen))
    assert H.check_sources(p2, entries)[regen.segment_id].state == H.STATUS_CHECK
    # 지운 결과
    store.delete(b.key)
    entries = {e.key: e for e in store.entries()}
    st = H.check_sources(plan, entries)
    assert st[plan.segments[1].segment_id].label == "검증 필요"
    with pytest.raises(H.HybridError):
        H.render(plan, store)


def test_same_plan_reuses_render_and_old_renders_are_pruned(two):
    store, a, b = two
    p = _plan_ab(a, b, 2.0)
    r1 = H.render(p, store)
    r1b = H.render(p, store)
    assert r1b.reused and r1b.entry.key == r1.entry.key
    keys = [r1.entry.key]
    for t in (1.0, 1.5, 2.5, 3.0):
        q = H.move_boundary(p, 1, t)
        keys.append(H.render(q, store).entry.key)
    hybrids = [e.key for e in store.entries() if e.model == H.HYBRID]
    assert len(hybrids) == 1 + H.KEEP_RENDERS  # 가장 최근 것 + 이전 2개
    assert keys[-1] in hybrids
    assert {a.key, b.key} <= {e.key for e in store.entries()}  # 모델 분리 결과는 지우지 않는다
    store.activate(keys[-1])
    H.prune(store, keep=set(), keep_renders=0)
    assert [e.key for e in store.entries() if e.model == H.HYBRID] == [keys[-1]]  # 지금 쓰는 것은 남긴다


def test_hybrid_identity_depends_on_every_choice(two):
    store, a, b = two
    p = _plan_ab(a, b)
    k = H.plan_key(p, "src")
    variants = [H.move_boundary(p, 1, 2.5), H.set_gain(p, [p.segments[1].segment_id], 1.0),
                H.set_crossfade(p, [p.segments[1].segment_id], 0.05), dataclasses.replace(p, level_match=True),
                H.set_all(p, a.key, a.model, a.manifest.stems_id)]
    assert len({k, *(H.plan_key(v, "src") for v in variants)}) == 1 + len(variants)
    assert H.plan_key(p, "other-source") != k


def test_plan_roundtrip_and_project_save_reopen(two, tmp_path):
    store, a, b = two
    p = H.set_gain(_plan_ab(a, b, 1.25, cf=0.02), [], 0.0)
    d = H.plan_to_dict(p, H.check_sources(p, {e.key: e for e in store.entries()}))
    seg = d["segments"][1]
    for k in ("segment_id", "start_time", "end_time", "source_model_id", "source_result_id", "crossfade_duration",
              "gain_adjustment_db", "is_user_selected", "status"):
        assert k in seg
    assert seg["start_time"] == pytest.approx(1.25) and seg["status"] == "ok"
    assert H.plan_from_dict(json.loads(json.dumps(d))) == p

    from amp.project import workspace

    wav = tmp_path / "곡.wav"
    sf.write(wav, _sine(220)[:, :1], SR)
    folder = workspace.create_project(str(wav), str(tmp_path / "projects"), "하이브리드")
    project, _ = folder.load_project()
    project.hybrid = p
    folder.save(project)
    assert folder.paths.hybrid_json.exists()
    raw = json.loads(folder.paths.hybrid_json.read_text(encoding="utf-8"))
    assert raw["schema"] == "sonatlas.hybrid" and raw["schema_version"] == 1
    reopened, report = workspace.open_folder(folder.root)[0].load_project()
    assert reopened.hybrid == p and not report.warnings
    project.hybrid = None  # 계획을 지우면 파일은 백업으로 남는다
    folder.save(project)
    assert not folder.paths.hybrid_json.exists()
    assert folder.paths.hybrid_json.with_name("hybrid.json.removed.bak").exists()


def test_corrupt_hybrid_file_opens_project_with_warning(tmp_path):
    from amp.project import workspace

    wav = tmp_path / "a.wav"
    sf.write(wav, _sine(220)[:, :1], SR)
    folder = workspace.create_project(str(wav), str(tmp_path / "p"), "x")
    project, _ = folder.load_project()
    folder.save(project)
    folder.paths.hybrid_json.write_text(json.dumps({"schema": "sonatlas.hybrid", "schema_version": 1,
                                                    "plan": {"sample_rate": SR, "frames": 10, "segments": [
                                                        {"segment_id": "a", "start_frame": 0, "end_frame": 4,
                                                         "source_result_id": "k"}]}}), encoding="utf-8")
    reopened, report = workspace.open_folder(folder.root)[0].load_project()
    assert reopened.hybrid is None and any("Hybrid" in w for w in report.warnings)
    assert folder.paths.hybrid_json.exists()  # 깨진 파일을 지우지 않는다
    newer = {"schema": "sonatlas.hybrid", "schema_version": 99, "plan": {}}
    folder.paths.hybrid_json.write_text(json.dumps(newer), encoding="utf-8")
    reopened, report = workspace.open_folder(folder.root)[0].load_project()
    assert reopened.hybrid is None and any("새로운 버전" in w for w in report.warnings)


# --- 피치 청크 재사용 ------------------------------------------------------------------

def test_pitch_chunk_reuse_matches_full_computation(monkeypatch):
    pytest.importorskip("librosa")
    from amp.devtools.synth import sung
    from amp.engine.dsp import pitch as P

    monkeypatch.setattr(P, "CHUNK_S", 2.0)
    monkeypatch.setattr(P, "PAD_S", 0.5)
    sr = 16000
    a = sung([(0.3, 1.5, 60), (1.7, 3.0, 64), (3.2, 5.5, 67), (5.7, 7.6, 69)], 8.0, sr=sr).astype(np.float32)
    b = (a * 0.7).astype(np.float32)
    mixed = a.copy()
    mixed[int(3.3 * sr):int(4.2 * sr)] = b[int(3.3 * sr):int(4.2 * sr)]  # 한 구간만 다른 소스
    prev = P.track_pitch(a, sr)
    full = P.track_pitch(mixed, sr)
    stats: dict = {}
    part = P.track_pitch(mixed, sr, reuse=[P.ReuseSource(P.to_pitch_sr(a, sr), prev.f0_hz, prev.voiced_prob)],
                         stats=stats)
    assert stats["reused"] >= 2 and stats["computed"] >= 1
    assert np.array_equal(np.isnan(full.f0_hz), np.isnan(part.f0_hz))
    assert np.array_equal(np.nan_to_num(full.f0_hz), np.nan_to_num(part.f0_hz))
    assert np.array_equal(full.voiced_prob, part.voiced_prob)
    assert np.array_equal(full.onset, part.onset) and np.array_equal(full.rms, part.rms)
    # 길이가 다르거나 신호가 다르면 쓰지 않는다
    stats2: dict = {}
    P.track_pitch(mixed, sr, reuse=[P.ReuseSource(P.to_pitch_sr(a[:-160], sr), prev.f0_hz[:-1], prev.voiced_prob[:-1])],
                  stats=stats2)
    assert stats2.get("reused", 0) == 0


@pytest.fixture()
def song(tmp_path):
    pytest.importorskip("librosa")
    from amp.devtools.synth import backing, sung

    voice = sung([(0.5, 1.6, 64), (1.8, 2.9, 67), (3.1, 4.4, 69), (4.6, 5.8, 72)], 6.5)
    mix = np.stack([voice + 0.3 * backing(6.5)] * 2, 1).astype(np.float32)
    path = tmp_path / "곡 하이브리드.wav"
    sf.write(path, mix, 44100)
    return path


def test_pipeline_uses_hybrid_and_reuses_unchanged_pitch_chunks(song, tmp_path, monkeypatch):
    from collections import Counter

    from amp.engine.dsp import pitch as P
    from amp.engine.model_compare import ModelSpec, run_compare
    from amp.engine.transcription import TranscriptionPipeline
    from tests.test_model_compare import _in_process, _request

    monkeypatch.setattr(P, "CHUNK_S", 1.5)
    monkeypatch.setattr(P, "PAD_S", 0.25)
    calls: Counter = Counter()
    runner = _in_process(calls)
    req = _request(song, tmp_path)
    run = run_compare(req, [ModelSpec("_identity"), ModelSpec("_half")], runner=runner)
    ka, kb = (o.key for o in run.outcomes)
    store = ModelStore(req.stems_dir, req.source_sha256)
    ea, eb = store.entry(ka), store.entry(kb)
    plan = H.single(ea.manifest.sample_rate, ea.manifest.frames, ka, ea.model, ea.manifest.stems_id)
    plan, _ = H.create(plan, 3.0, 3.6, kb, eb.model, eb.manifest.stems_id)
    res = H.render(plan, store)
    store.activate(res.entry.key)
    out = TranscriptionPipeline(runner).run(req)
    assert out.manifest.model == H.HYBRID and out.reused["separation"]  # 설정의 모델로 다시 분리하지 않는다
    chunks = out.result.models["pitch"].params["chunks"]
    assert chunks["reused"] >= 2 and chunks["computed"] >= 1
    assert calls["separate"] == 2  # 비교 때 두 번만
    # 처음부터 계산한 피치와 같다
    full = TranscriptionPipeline(runner).run(dataclasses.replace(req, settings=dataclasses.replace(req.settings,
                                                                                                    use_cache=False)))
    assert np.array_equal(np.nan_to_num(full.result.pitch_curve.f0_hz), np.nan_to_num(out.result.pitch_curve.f0_hz))
    assert [(n.start, n.end, n.pitch) for n in full.result.notes] == [(n.start, n.end, n.pitch) for n in out.result.notes]


# --- 음표 id 이어받기 --------------------------------------------------------------------

def _result(notes):
    from amp.core.model import AnalysisResult
    from amp.core.tempo import TempoMap

    return AnalysisResult(duration=10.0, tempo=TempoMap.constant(120), notes=tuple(notes))


def test_reanalysis_carries_ids_of_unchanged_notes():
    from amp.core.model import Note, Project, Score
    from amp.engine.merge import KEEP_EDITS, carry_note_ids, plan_reanalysis

    old = _result([Note("n1", 1.0, 1.5, 60), Note("n2", 2.0, 2.5, 62), Note("n3", 3.0, 3.5, 64)])
    project = Project(analysis=old, score=Score.from_analysis(old))
    # n3을 사용자가 고쳤다 (id 유지, 음높이 변경)
    project.score.replace_notes(["n3"], [project.score.get("n3").with_(pitch=65)])
    new = _result([Note("x1", 1.0, 1.5, 60), Note("x2", 2.0, 2.4, 62), Note("x3", 3.0, 3.5, 64)])
    carried = carry_note_ids(project, new)
    assert [n.id for n in carried.notes] == ["n1", "x2", "x3"]  # 같은 음표만, 고친 음표 id는 쓰지 않음
    plan = plan_reanalysis(project, carried, KEEP_EDITS)
    project.score.replace_notes([n.id for n in plan.removed], plan.added)
    assert "n1" in project.score and project.score.get("n3").pitch == 65
    assert carry_note_ids(Project(), new) is new
