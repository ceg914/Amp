"""피치 모델별 분석 결과 (v0.7.0 Feature A, 명세 4.3·4.4·14.2·14.3). Qt 비의존.

- 공통 인터페이스: 어느 모델이든 결과는 ``PitchRun`` (frame_times = (start_frame + i)/100, frequencies_hz(무성 NaN),
  voiced_mask, confidence, model_metadata=info, analysis_source_id=meta). 신뢰도는 모델마다 뜻이 달라 confidence_kind로 구분한다
- 저장: 프로젝트 ``analysis/pitch_models/<run_id>.npz`` + ``.json`` (json이 완료 표시). 모델·입력·설정·구간이 같은 결과가 있으면
  다시 계산하지 않고 그것을 쓴다. 다시 계산을 고르면 새 run_id로 따로 저장한다. 기존 결과를 덮어쓰지 않는다
- 입력 구분: 입력 종류(보컬·원본·Hybrid 등)·분리 결과 id·입력 파일 해시가 다르면 다른 결과다 (같은 모델이어도)
- 소스가 바뀌면 결과를 지우지 않고 stale로 표시한다
- 여러 모델은 차례로 실행한다 (GPU에 대형 모델을 동시에 올리지 않음). 한 모델이 실패해도 다른 모델은 계속하고, 취소해도 끝난 결과는 남긴다
- pYIN은 보컬 분석과 같은 캐시 키로 읽어, 이미 분석했으면 다시 계산하지 않는다
- 이 결과는 음표에 자동으로 쓰지 않는다 (명세: 기존 방식보다 낫다는 근거 없이 기본 활성화하지 않음)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from amp.core import result_meta as RM
from amp.core.errors import AnalysisCancelled, AnalysisError
from amp.engine.context import CancelToken, ProgressFn

SCHEMA = "sonatlas.pitch_run"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "1"
FRAME_RATE = 100.0
DIR_NAME = "pitch_models"
MODEL_IDS = ("pyin", "crepe_full", "crepe_tiny", "fcpe")
MODEL_LABELS = {"pyin": "pYIN", "crepe_full": "CREPE full", "crepe_tiny": "CREPE tiny", "fcpe": "FCPE"}
CONFIDENCE_KINDS = {"pyin": "pyin_voiced_prob", "crepe_full": "crepe_activation", "crepe_tiny": "crepe_activation",
                    "fcpe": "fcpe_latent_max"}
CONFIDENCE_LABELS = {"pyin_voiced_prob": "pYIN 유성 확률", "crepe_activation": "CREPE 활성값",
                     "fcpe_latent_max": "FCPE 최댓값"}
PRECISE_ORDER = ("crepe_full", "fcpe", "crepe_tiny")  # '정밀 분석'이 고르는 순서 (설치된 것 중 첫째)
MODES = {"basic": "기본 (pYIN)", "precise": "정밀 (외부 모델 + pYIN 비교)", "select": "모델 직접 선택",
         "compare": "여러 모델 비교", "region": "선택 구간 다시 분석"}
REGION_PAD_S = 1.0


@dataclass(frozen=True)
class PitchInput:
    """분석할 소리. kind는 result_meta.INPUT_KINDS, input_id는 분리 결과 id 등(원본이면 '')."""

    path: str
    kind: str
    input_id: str
    input_hash: str
    source_sha256: str
    duration: float
    label: str = ""

    @property
    def cache_input_id(self) -> str:
        """보컬 분석 파이프라인의 pYIN 캐시 키와 같은 표기 (stems:<id> / original)."""
        return self.input_id if self.input_id else "original"


@dataclass
class PitchRun:
    run_id: str
    meta: RM.ResultMeta
    start_frame: int
    f0_hz: np.ndarray
    confidence: np.ndarray
    voiced: np.ndarray
    info: dict = field(default_factory=dict)
    frame_rate: float = FRAME_RATE

    @property
    def model_id(self) -> str:
        return self.meta.model_id

    @property
    def label(self) -> str:
        base = MODEL_LABELS.get(self.model_id, self.model_id)
        if self.meta.analysis_range is not None:
            a, b = self.meta.analysis_range
            base += f" ({a:.1f}–{b:.1f}s)"
        return base

    @property
    def confidence_kind(self) -> str:
        return str(self.info.get("confidence_kind") or CONFIDENCE_KINDS.get(self.model_id, ""))

    @property
    def n(self) -> int:
        return int(len(self.f0_hz))

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.n

    def frame_times(self) -> np.ndarray:
        return (self.start_frame + np.arange(self.n)) / self.frame_rate

    def window(self, g0: int, g1: int):
        """곡 전체 프레임 [g0, g1)에 해당하는 (f0, voiced, confidence). 범위 밖은 무성."""
        f = np.full(g1 - g0, np.nan, dtype=np.float32)
        v = np.zeros(g1 - g0, dtype=bool)
        c = np.zeros(g1 - g0, dtype=np.float32)
        a, b = max(g0, self.start_frame), min(g1, self.end_frame)
        if b > a:
            f[a - g0:b - g0] = self.f0_hz[a - self.start_frame:b - self.start_frame]
            v[a - g0:b - g0] = self.voiced[a - self.start_frame:b - self.start_frame]
            c[a - g0:b - g0] = self.confidence[a - self.start_frame:b - self.start_frame]
        return f, v, c


def settings_for(model_id: str, fmin: float, fmax: float, threshold: float | None, info: dict | None = None) -> dict:
    s = {"fmin": round(float(fmin), 3), "fmax": round(float(fmax), 3), "analysis_version": ANALYSIS_VERSION}
    if model_id != "pyin":
        from amp.engine.dsp import pitch_models as PM

        s.update(threshold=None if threshold is None else round(float(threshold), 6), code_version=PM.CODE_VERSION)
    return s


def run_key(model_id: str, inp: PitchInput, settings: dict, rng: tuple[float, float] | None) -> str:
    ident = {"model": model_id, "kind": inp.kind, "input_id": inp.input_id, "input_hash": inp.input_hash,
             "source": inp.source_sha256, "settings": settings,
             "range": None if rng is None else [round(rng[0], 3), round(rng[1], 3)]}
    digest = hashlib.sha1(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{model_id}-{digest}"


class PitchRunStore:
    """프로젝트 analysis/pitch_models/. 지우기는 사용자가 고를 때만 (remove)."""

    def __init__(self, analysis_dir: str | Path) -> None:
        self.dir = Path(analysis_dir) / DIR_NAME

    def _json(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.json"

    def ids(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json") if (self.dir / f"{p.stem}.npz").is_file())

    def load(self, run_id: str) -> PitchRun | None:
        try:
            d = json.loads(self._json(run_id).read_text(encoding="utf-8"))
            meta = RM.meta_from_dict(d.get("meta"))
            if d.get("schema") != SCHEMA or meta is None or int(d.get("schema_version", 0)) > SCHEMA_VERSION:
                return None
            with np.load(self.dir / f"{run_id}.npz", allow_pickle=False) as z:
                f0 = z["f0_hz"].astype(np.float32)
                conf = z["confidence"].astype(np.float32)
                voiced = z["voiced"].astype(bool)
            if not (len(f0) == len(conf) == len(voiced)):
                return None
            return PitchRun(run_id, meta, int(d.get("start_frame", 0)), f0, conf, voiced, dict(d.get("info") or {}),
                            float(d.get("frame_rate", FRAME_RATE)))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def runs(self) -> list[PitchRun]:
        out = [r for r in (self.load(i) for i in self.ids()) if r is not None]
        out.sort(key=lambda r: (r.meta.created_at, r.run_id))
        return out

    def save(self, run: PitchRun) -> None:
        """배열을 먼저, 목록(json)을 마지막에 바꿔 넣는다 (중간에 멈추면 json이 없어 결과로 보지 않음)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=self.dir)
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, f0_hz=run.f0_hz.astype(np.float32), confidence=run.confidence.astype(np.float32),
                                    voiced=run.voiced.astype(bool))
            os.replace(tmp, self.dir / f"{run.run_id}.npz")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        from amp.core.fileio import atomic_write_json

        atomic_write_json(self._json(run.run_id), {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "meta": run.meta.to_dict(), "run_id": run.run_id,
            "start_frame": int(run.start_frame), "frame_rate": run.frame_rate, "frames": run.n,
            "info": run.info})

    def set_status(self, run_id: str, status: str, reason: str = "") -> None:
        p = self._json(run_id)
        d = json.loads(p.read_text(encoding="utf-8"))
        meta = RM.meta_from_dict(d.get("meta"))
        if meta is None:
            return
        d["meta"] = meta.with_status(status, reason).to_dict()
        from amp.core.fileio import atomic_write_json

        atomic_write_json(p, d)

    def mark_stale(self, current: PitchInput) -> list[str]:
        """지금 분석 소스와 다른 소리로 만든 결과를 stale로 (지우지 않음). 바꾼 run_id 목록."""
        changed = []
        for r in self.runs():
            if r.meta.status == RM.STALE:
                continue
            same = (r.meta.input_kind == current.kind and r.meta.input_id == current.input_id
                    and (r.meta.input_hash or "") == current.input_hash and r.meta.source_audio_hash == current.source_sha256)
            if not same:
                self.set_status(r.run_id, RM.STALE, "분석 소스가 바뀌었습니다 (이 결과는 이전 소리로 만들었습니다).")
                changed.append(r.run_id)
        return changed

    def find(self, key_prefix: str) -> PitchRun | None:
        hits = [r for r in self.runs() if (r.run_id == key_prefix or r.run_id.startswith(key_prefix + "-"))
                and r.meta.status in (RM.COMPLETED, RM.UNCERTAIN)]
        return hits[-1] if hits else None

    def remove(self, run_id: str) -> None:
        for suffix in (".json", ".npz"):
            p = self.dir / f"{run_id}{suffix}"
            if p.exists():
                p.unlink()


@dataclass
class PitchRunRequest:
    inp: PitchInput
    analysis_dir: str
    models: tuple[str, ...]
    device: str = "auto"
    fmin: float = 55.0
    fmax: float = 1400.0
    threshold: float | None = None
    analysis_range: tuple[float, float] | None = None
    force: bool = False
    cache_root: str | None = None
    use_cache: bool = True
    numba_cache_dir: str | None = None


@dataclass
class PitchRunOutcome:
    runs: list[PitchRun] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)  # 다시 계산하지 않은 run_id
    failed: dict = field(default_factory=dict)  # model_id → 오류 메시지
    cancelled: bool = False


def precise_models(available: set[str]) -> tuple[str, ...]:
    ext = next((m for m in PRECISE_ORDER if m in available), None)
    return ("pyin",) + ((ext,) if ext else ())


def _uncertain_reason(voiced: np.ndarray) -> str:
    if len(voiced) == 0:
        return "프레임이 없습니다."
    if float(np.mean(voiced)) < 0.02:
        return "유성 프레임이 거의 없습니다 (2% 미만)."
    return ""


def run_models(req: PitchRunRequest, runner: Callable, progress: ProgressFn | None = None,
               token: CancelToken | None = None) -> PitchRunOutcome:
    """모델을 차례로 실행해 저장한다. 같은 결과가 있으면(force가 아니면) 그것을 쓴다."""
    report = progress or (lambda f, m: None)
    token = token or CancelToken()
    store = PitchRunStore(req.analysis_dir)
    out = PitchRunOutcome()
    n = max(1, len(req.models))
    for k, model_id in enumerate(req.models):
        share = (k / n, (k + 1) / n)

        def sub(f, m, share=share):
            report(share[0] + (share[1] - share[0]) * float(f), m)

        settings = settings_for(model_id, req.fmin, req.fmax, req.threshold)
        key = run_key(model_id, req.inp, settings, req.analysis_range)
        old = None if req.force else store.find(key)
        if old is not None:
            out.runs.append(old)
            out.reused.append(old.run_id)
            sub(1.0, f"{MODEL_LABELS.get(model_id, model_id)}: 저장한 결과 사용")
            continue
        try:
            token.raise_if_cancelled()
            run = _compute(model_id, req, settings, runner, sub, token)
        except AnalysisCancelled:
            out.cancelled = True
            break
        except AnalysisError as e:
            out.failed[model_id] = str(e)
            continue
        except Exception as e:  # noqa: BLE001 - 한 모델의 오류가 다른 모델을 막지 않는다 (13.3)
            out.failed[model_id] = f"{type(e).__name__}: {e}"
            continue
        existing = set(store.ids())
        run_id, i = key, 2
        while run_id in existing:  # 다시 계산: 이전 결과를 덮어쓰지 않고 새 id로
            run_id, i = f"{key}-{i}", i + 1
        run.run_id = run_id
        store.save(run)
        out.runs.append(run)
    return out


def _compute(model_id: str, req: PitchRunRequest, settings: dict, runner, report, token) -> PitchRun:
    inp = req.inp
    n_all = int(inp.duration * FRAME_RATE) + 1
    rng = req.analysis_range
    with tempfile.TemporaryDirectory(prefix="sonatlas-pitchrun-") as scratch:
        out_dir = Path(scratch)
        if model_id == "pyin":
            f0, conf, voiced, info = _pyin(req, runner, report, token, out_dir)
            start = 0
            if rng is not None:  # pYIN은 30초 청크 계산이라 곡 전체 결과에서 구간을 잘라 쓴다 (같은 값)
                g0 = max(0, int(np.floor(rng[0] * FRAME_RATE)))
                g1 = min(len(f0), int(np.ceil(rng[1] * FRAME_RATE)) + 1)
                f0, conf, voiced, start = f0[g0:g1], conf[g0:g1], voiced[g0:g1], g0
        else:
            args = {"input": inp.path, "model": model_id, "device": req.device, "fmin": req.fmin, "fmax": req.fmax,
                    "out_dir": str(out_dir)}
            if req.threshold is not None:
                args["threshold"] = req.threshold
            if rng is not None:
                args.update(start=float(rng[0]), end=float(rng[1]), pad=REGION_PAD_S)
            runner("pitch_model", args, report, token, req.numba_cache_dir)
            with np.load(out_dir / "pitch_model.npz", allow_pickle=False) as z:
                f0 = z["f0_hz"].astype(np.float32)
                conf = z["confidence"].astype(np.float32)
                voiced = z["voiced"].astype(bool)
                start = int(z["start_frame"])
            info = json.loads((out_dir / "info.json").read_text(encoding="utf-8"))
    if abs((start + len(f0)) - n_all) > 3 and rng is None:
        raise AnalysisError(f"{MODEL_LABELS.get(model_id, model_id)} 결과 프레임 수({len(f0)})가 오디오 길이와 맞지 않습니다.")
    reason = _uncertain_reason(voiced)
    meta = RM.make_meta(SCHEMA, SCHEMA_VERSION, inp.source_sha256, input_kind=inp.kind, input_id=inp.input_id,
                        input_hash=inp.input_hash, model_id=model_id,
                        model_version=str(info.get("weights_sha256", "") or info.get("librosa", ""))[:16],
                        analysis_version=ANALYSIS_VERSION, analysis_settings=settings, analysis_range=rng,
                        status=RM.UNCERTAIN if reason else RM.COMPLETED, status_reason=reason)
    info.setdefault("confidence_kind", CONFIDENCE_KINDS.get(model_id, ""))
    info["input_label"] = inp.label
    return PitchRun("", meta, start, f0, conf, voiced, info)


def _pyin(req: PitchRunRequest, runner, report, token, scratch: Path):
    """보컬 분석과 같은 캐시 키(pitch 단계)로 pYIN 프레임을 읽거나 계산한다."""
    import time

    from amp.engine.cache import AnalysisCache
    from amp.engine.dsp.pitch import FMAX_HZ, FMIN_HZ
    from amp.engine.transcription import PITCH_VERSION, TranscriptionPipeline, _load_pitch

    inp = req.inp
    same = abs(req.fmin - FMIN_HZ) < 1e-6 and abs(req.fmax - FMAX_HZ) < 1e-6  # 보컬 분석과 같은 설정 = 같은 캐시
    params = {"tracker": "pyin", "v": PITCH_VERSION, "fmin": FMIN_HZ if same else req.fmin,
              "fmax": FMAX_HZ if same else req.fmax, "input": inp.cache_input_id}
    cache = AnalysisCache(req.cache_root) if req.cache_root else None
    expected = int(inp.duration * 100) + 1
    t0 = time.perf_counter()
    (frames, pmeta), reused = TranscriptionPipeline._cached(
        cache, inp.source_sha256[:32] or "nohash", "pitch", params, req.use_cache, scratch,
        lambda out, p: runner("pitch", {"input": inp.path, "fmin": req.fmin, "fmax": req.fmax, "out_dir": str(out)},
                              p, token, req.numba_cache_dir),
        lambda d: _load_pitch(d, expected), report, [])
    f0 = np.asarray(frames.f0_hz, dtype=np.float32)
    voiced = np.isfinite(f0) & (f0 >= req.fmin) & (f0 <= req.fmax)
    info = {"model_id": "pyin", "model_name": "pYIN", "confidence_kind": "pyin_voiced_prob",
            "librosa": str((pmeta.get("versions") or {}).get("librosa", "")), "device": "cpu",
            "elapsed_s": round(time.perf_counter() - t0, 3), "from_cache": bool(reused), "frame_rate": FRAME_RATE,
            "warnings": []}
    return np.where(voiced, f0, np.nan).astype(np.float32), np.asarray(frames.voiced_prob, dtype=np.float32), voiced, info
