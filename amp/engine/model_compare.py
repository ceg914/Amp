"""분리 모델 비교 실행 (STEP 5). Qt 비의존.

1) 고른 모델을 하나씩 차례로 분리한다. 모델마다 작업 프로세스를 따로 띄우고 끝나면 프로세스가 끝나므로, 그 모델과 텐서가
   차지한 GPU 메모리가 운영체제에 돌아간 뒤 다음 모델을 올린다. 두 모델이 동시에 GPU에 올라가지 않는다.
2) 이미 같은 키(원본·모델·체크포인트·설정)의 결과가 보관소에 있으면 다시 분리하지 않는다.
3) GPU 메모리가 부족하면 CPU로 바꾸지 않고 그 모델을 'oom'으로 표시한다. CPU로 다시 돌릴지는 사용자가 정한다.
4) 한 모델이 실패해도 다른 모델은 계속한다. 취소하면 그때까지 끝난 결과는 보관소에 그대로 남는다.
5) 모델마다 같은 피치 추출(pYIN)·같은 음표 검출 설정으로 채보한다. 피치 캐시는 분리 결과 id별로 따로 쌓인다.

지금 분석에 쓰는 결과(stems/stems.json)는 바꾸지 않는다. 분석 소스를 바꾸는 것은 사용자가 고를 때만이다.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field

from amp.core.errors import AnalysisCancelled, AnalysisError
from amp.engine.context import CancelToken, ProgressFn
from amp.engine.model_store import MODEL_LABELS, ModelStore, identity_for
from amp.engine.transcription import TranscriptionPipeline, TranscriptionRequest
from amp.engine.worker_client import WorkerError, run_worker

log = logging.getLogger(__name__)

STATUS_LABELS = {"reused": "보관한 결과 사용", "done": "분리 완료", "failed": "실패", "oom": "GPU 메모리 부족",
                 "cancelled": "취소됨", "pending": "대기"}


@dataclass(frozen=True)
class ModelSpec:
    model: str  # htdemucs / mel_roformer (시험용 _identity 등)
    model_files: dict = field(default_factory=dict)
    device: str | None = None  # None이면 분석 설정의 장치. 사용자가 'CPU로 다시'를 고르면 "cpu"

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)


@dataclass
class ModelOutcome:
    spec: ModelSpec
    status: str = "pending"
    key: str | None = None
    message: str = ""
    error_code: str = ""
    seconds: float = 0.0  # 분리에 걸린 시간 (재사용이면 확인 시간)
    device: str = ""
    resources: dict = field(default_factory=dict)
    output: object | None = None  # TranscriptionOutput (채보까지 했을 때)

    @property
    def ok(self) -> bool:
        return self.status in ("reused", "done") and self.key is not None


@dataclass
class CompareRun:
    outcomes: list[ModelOutcome]
    cancelled: bool = False

    def available(self) -> list[ModelOutcome]:
        return [o for o in self.outcomes if o.ok]


def run_compare(base: TranscriptionRequest, specs: list[ModelSpec], progress: ProgressFn | None = None,
                token: CancelToken | None = None, runner=run_worker, transcribe: bool = True,
                force: bool = False) -> CompareRun:
    if not base.stems_dir:
        raise AnalysisError("프로젝트를 연 뒤에 모델을 비교할 수 있습니다.")
    token = token or CancelToken()
    report = progress or (lambda f, m: None)
    store = ModelStore(base.stems_dir, base.source_sha256)
    store.migrate_legacy()
    outcomes = [ModelOutcome(s) for s in specs]
    n = max(1, len(specs))
    sep_w = 0.7 if transcribe else 1.0
    run = CompareRun(outcomes)
    pipe = TranscriptionPipeline(runner)
    for i, (spec, oc) in enumerate(zip(specs, outcomes)):
        base_f = sep_w * i / n

        def rep(f, m, base_f=base_f, spec=spec):
            report(base_f + sep_w * min(max(f, 0.0), 1.0) / n, f"[{spec.label}] {m}")

        t0 = time.perf_counter()
        try:
            token.raise_if_cancelled()
            if spec.model == "mel_roformer" and not spec.model_files.get("checkpoint"):
                raise AnalysisError("Mel-Band RoFormer 모델 파일(.ckpt)을 고르지 않았습니다. 분석 설정의 보컬 분리에서 "
                                    "[모델 파일…]로 지정하세요.")
            ident = identity_for(spec.model, base.source_sha256, spec.model_files)
            entry = None if force else store.find(ident)
            if entry is not None:
                oc.status, oc.key = "reused", entry.key
                oc.device = str(entry.manifest.model_info.get("device", ""))
                rep(1.0, "보관한 분리 결과 사용")
            else:
                st = dataclasses.replace(base.settings, device=spec.device or base.settings.device)
                req = dataclasses.replace(base, settings=st, model_files=dict(spec.model_files), cpu_fallback=False)
                pipe.resources = {}
                entry = pipe.separate_into(store, spec.model, ident, req, rep, token)
                if store.active_key() == entry.key:  # 지금 쓰는 결과를 다시 만들었다: 위쪽 목록(만든 시각·크기)도 새로
                    store.activate(entry.key)
                oc.status, oc.key = "done", entry.key
                oc.device = str(entry.manifest.model_info.get("device", ""))
                oc.resources = dict(pipe.resources.get("separation", {}))
        except AnalysisCancelled:
            oc.status, oc.message = "cancelled", "취소됨"
            for rest in outcomes[i + 1:]:
                rest.status, rest.message = "cancelled", "취소되어 실행하지 않음"
            run.cancelled = True
            log.info("모델 비교 취소: 끝난 결과 %d개는 보관소에 남김", len(run.available()))
            return run
        except WorkerError as e:
            oc.status = "oom" if e.code == "oom_gpu" else "failed"
            oc.error_code, oc.message = e.code, str(e)
            log.warning("모델 비교: %s 분리 실패 (%s) %s", spec.model, e.code, e)
        except AnalysisError as e:
            oc.status, oc.message = "failed", str(e)
            log.warning("모델 비교: %s 분리 실패 %s", spec.model, e)
        finally:
            oc.seconds = time.perf_counter() - t0
    if transcribe:
        ok = run.available()
        for j, oc in enumerate(ok):
            base_f = sep_w + (1 - sep_w) * j / max(1, len(ok))

            def rep2(f, m, base_f=base_f, oc=oc):
                report(base_f + (1 - sep_w) * min(max(f, 0.0), 1.0) / max(1, len(ok)), f"[{oc.spec.label}] {m}")

            try:
                oc.output = TranscriptionPipeline(runner).run(dataclasses.replace(base, stems_key=oc.key), rep2, token)
            except AnalysisCancelled:
                run.cancelled = True
                return run
            except AnalysisError as e:  # 채보만 실패: 분리 결과는 남아 있고 재생 비교는 된다
                oc.message = f"채보 실패: {e}"
                log.warning("모델 비교: %s 채보 실패 %s", oc.spec.model, e)
    report(1.0, "모델 비교 완료")
    return run


def transcribe_keys(base: TranscriptionRequest, keys: list[str], progress: ProgressFn | None = None,
                    token: CancelToken | None = None, runner=run_worker) -> dict[str, object]:
    """보관한 분리 결과마다 같은 설정으로 채보한다 (분리는 하지 않음). 키 → TranscriptionOutput 또는 오류 문자열."""
    token = token or CancelToken()
    report = progress or (lambda f, m: None)
    out: dict[str, object] = {}
    for j, key in enumerate(keys):
        def rep(f, m, j=j):
            report((j + min(max(f, 0.0), 1.0)) / max(1, len(keys)), m)

        try:
            out[key] = TranscriptionPipeline(runner).run(dataclasses.replace(base, stems_key=key), rep, token)
        except AnalysisError as e:
            out[key] = str(e)
    return out
