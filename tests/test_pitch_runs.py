"""v0.7.0 Feature A: 피치 모델별 결과 저장·재사용·구간 분석·소스 변경(stale)·비교. 작업 프로세스 명령은 이 프로세스에서 실행."""

from __future__ import annotations

import importlib.util
import io

import numpy as np
import pytest
import soundfile as sf

from amp.compare import pitch_compare as PC
from amp.core import result_meta as RM
from amp.core.errors import AnalysisCancelled
from amp.engine import pitch_runs as PR
from amp.engine.context import CancelToken

HAS_CREPE = importlib.util.find_spec("torchcrepe") is not None and importlib.util.find_spec("torch") is not None
SR = 44100


def runner(calls: list):
    from amp.engine import worker

    def run(command, args, progress=None, token=None, numba_cache_dir=None):
        calls.append((command, args.get("model"), args.get("start")))
        if token is not None and token.cancelled:
            raise AnalysisCancelled("취소")
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    return run


@pytest.fixture
def vocals(tmp_path):
    t = np.arange(int(SR * 3.0)) / SR
    x = np.zeros_like(t)
    m = (t >= 0.5) & (t < 2.5)
    x[m] = sum(0.3 / k * np.sin(2 * np.pi * 220 * k * t[m]) for k in range(1, 5))
    p = tmp_path / "보컬.wav"
    sf.write(p, np.stack([x, x], 1).astype(np.float32), SR)
    return p


def make_input(path, input_id="stems:abc", h="1" * 64):
    return PR.PitchInput(str(path), "vocals", input_id, h, "a" * 64, 3.0, "보컬 (시험)")


@pytest.mark.skipif(not HAS_CREPE, reason="torchcrepe·torch 없음")
def test_runs_are_stored_reused_and_never_overwritten(vocals, tmp_path):
    calls: list = []
    req = PR.PitchRunRequest(make_input(vocals), str(tmp_path / "analysis"), ("pyin", "crepe_tiny"), device="cpu",
                             cache_root=str(tmp_path / "cache"))
    out = PR.run_models(req, runner(calls))
    assert not out.failed and [r.model_id for r in out.runs] == ["pyin", "crepe_tiny"]
    store = PR.PitchRunStore(tmp_path / "analysis")
    assert len(store.ids()) == 2
    for r in store.runs():
        assert r.n == 301 and r.start_frame == 0 and r.meta.status == RM.COMPLETED
        assert r.meta.input_kind == "vocals" and r.meta.input_hash == "1" * 64 and r.meta.source_audio_hash == "a" * 64
        cents = 1200 * np.log2(r.f0_hz[r.voiced] / 220.0)
        assert abs(float(np.median(cents))) < 15
    crepe = next(r for r in store.runs() if r.model_id == "crepe_tiny")
    assert crepe.confidence_kind == "crepe_activation" and crepe.info["device"] == "cpu"
    assert crepe.info["elapsed_s"] > 0 and len(crepe.info["weights_sha256"]) == 64

    calls.clear()
    again = PR.run_models(req, runner(calls))
    assert sorted(again.reused) == sorted(store.ids()) and calls == []  # 같은 결과는 다시 계산하지 않는다
    before = {i: (store.dir / f"{i}.npz").read_bytes() for i in store.ids()}
    forced = PR.run_models(PR.PitchRunRequest(**{**req.__dict__, "models": ("crepe_tiny",), "force": True}),
                           runner(calls))
    assert forced.runs[0].run_id not in before and len(store.ids()) == 3
    assert {i: (store.dir / f"{i}.npz").read_bytes() for i in before} == before  # 이전 결과는 그대로

    other = make_input(vocals, input_id="stems:other", h="2" * 64)
    out2 = PR.run_models(PR.PitchRunRequest(**{**req.__dict__, "inp": other, "models": ("crepe_tiny",)}), runner(calls))
    assert not out2.reused  # 같은 모델이라도 입력이 다르면 다른 결과
    changed = store.mark_stale(other)
    assert set(changed) == set(before) | {forced.runs[0].run_id}
    assert all(store.load(i).meta.status == RM.STALE for i in changed) and len(store.ids()) == 4  # 지우지 않음
    assert store.load(out2.runs[0].run_id).meta.status == RM.COMPLETED


@pytest.mark.skipif(not HAS_CREPE, reason="torchcrepe·torch 없음")
def test_region_run_matches_full_run_timeline(vocals, tmp_path):
    """구간 분석 결과의 프레임이 곡 전체 결과와 같은 시각에 놓인다 (명세 14.2)."""
    calls: list = []
    base = PR.PitchRunRequest(make_input(vocals), str(tmp_path / "a"), ("crepe_tiny",), device="cpu")
    full = PR.run_models(base, runner(calls)).runs[0]
    region = PR.run_models(PR.PitchRunRequest(**{**base.__dict__, "analysis_range": (1.0, 2.0)}), runner(calls)).runs[0]
    assert region.start_frame == 100 and region.n == 101 and region.meta.analysis_range == (1.0, 2.0)
    assert calls[-1] == ("pitch_model", "crepe_tiny", 1.0)
    f_full, v_full, _ = full.window(100, 201)
    assert np.array_equal(v_full, region.voiced)
    assert np.max(np.abs(1200 * np.log2(region.f0_hz[region.voiced] / f_full[region.voiced]))) < 1.0
    assert "(1.0–2.0s)" in region.label
    edge = PR.run_models(PR.PitchRunRequest(**{**base.__dict__, "analysis_range": (0.0, 0.6)}), runner(calls)).runs[0]
    assert edge.start_frame == 0 and abs(np.argmax(edge.voiced) / 100 - 0.5) <= 0.02  # 앞 문맥이 없어도 시각이 맞다


def test_failure_of_one_model_does_not_stop_others(vocals, tmp_path, monkeypatch):
    from amp.engine.dsp import pitch_models as PM

    def missing(model_id, find_spec=None):
        raise PM.PitchModelError("missing_pitch_model", f"{model_id} 가중치 패키지가 없습니다. install_pitch.bat을 실행하세요.")

    monkeypatch.setattr(PM, "weights_path", missing)
    from amp.engine.worker_client import WorkerError

    def failing_runner(command, args, progress=None, token=None, numba_cache_dir=None):
        if command == "pitch_model":
            raise WorkerError("missing_pitch_model", f"{args['model']} 가중치 패키지가 없습니다.")
        return runner([])(command, args, progress, token, numba_cache_dir)

    req = PR.PitchRunRequest(make_input(vocals), str(tmp_path / "a"), ("fcpe", "pyin"), cache_root=str(tmp_path / "c"))
    out = PR.run_models(req, failing_runner)
    assert set(out.failed) == {"fcpe"} and "가중치" in out.failed["fcpe"]
    assert [r.model_id for r in out.runs] == ["pyin"]


def test_cancel_keeps_finished_runs(vocals, tmp_path):
    token = CancelToken()
    calls: list = []

    def run(command, args, progress=None, tok=None, numba_cache_dir=None):
        r = runner(calls)(command, args, progress, tok, numba_cache_dir)
        token.cancel()  # 첫 모델이 끝난 뒤 취소
        return r

    req = PR.PitchRunRequest(make_input(vocals), str(tmp_path / "a"), ("pyin", "crepe_tiny"), cache_root=None)
    out = PR.run_models(req, run, token=token)
    assert out.cancelled and [r.model_id for r in out.runs] == ["pyin"]
    assert len(PR.PitchRunStore(tmp_path / "a").ids()) == 1


def test_precise_mode_picks_installed_external_model():
    assert PR.precise_models({"fcpe", "crepe_tiny"}) == ("pyin", "fcpe")
    assert PR.precise_models({"crepe_full", "fcpe"}) == ("pyin", "crepe_full")
    assert PR.precise_models(set()) == ("pyin",)


def _run(run_id, model, f0, voiced, start=0, inp="stems:x"):
    meta = RM.make_meta(PR.SCHEMA, 1, "a" * 64, input_kind="vocals", input_id=inp, input_hash="1" * 64, model_id=model)
    return PR.PitchRun(run_id, meta, start, np.asarray(f0, dtype=np.float32), np.ones(len(f0), np.float32),
                       np.asarray(voiced, dtype=bool), {"elapsed_s": 1.5, "device": "cpu",
                                                         "confidence_kind": PR.CONFIDENCE_KINDS[model]})


def test_compare_reports_differences_not_accuracy():
    n = 200
    fa = np.full(n, 220.0)
    fb = np.full(n, 221.0)
    fb[50:80] = 440.0  # 0.3초 옥타브 차이 (b는 10프레임부터: 곡 기준 0.6~0.9초)
    va = np.ones(n, bool)
    vb = np.ones(n, bool)
    vb[120:140] = False  # 0.2초 a만 유성 (곡 기준 1.3~1.5초)
    va[180:185] = False  # 50ms b만 유성 (구간 최소 길이보다 짧음)
    a, b = _run("pyin-1", "pyin", fa, va), _run("crepe_tiny-1", "crepe_tiny", fb, vb, start=10)
    d = PC.compare(a, b)
    assert d.start == 0.1 and d.frames == 190  # 겹치는 범위만
    assert d.octave_frames == 30 and d.octave_regions == [(0.6, 0.9, -1200.0)]
    assert d.only_a_regions == [(1.3, 1.5)] and d.only_b_regions == []
    assert d.only_b == 5 and d.both_voiced == 190 - 20 - 5
    assert d.median_abs_cents == pytest.approx(1200 * np.log2(221 / 220), abs=0.06)  # 220 vs 221Hz = 7.85센트
    text = "\n".join(d.summary_lines())
    assert "정확도가 아닙니다" in text and "신뢰도는 비교하지 않습니다" in text
    assert d.resources["pyin-1"]["elapsed_s"] == 1.5
    t, c = PC.cents_series(a, b)
    assert len(t) == 190 and t[0] == pytest.approx(0.1) and np.isnan(c[135 - 10]) \
        and c[70 - 10] == pytest.approx(-1200.0, abs=0.1)
    d2 = PC.compare(a, _run("x", "fcpe", fb, vb, inp="stems:other"))
    assert any("입력 소리가 다릅니다" in s for s in d2.summary_lines())
    assert PC.compare(a, _run("y", "fcpe", fb[:10], vb[:10], start=500)).summary_lines() == ["두 결과가 겹치는 구간이 없습니다."]
