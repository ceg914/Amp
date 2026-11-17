"""보컬 자동 채보 파이프라인 (STEP 2).

    원본 ─(분리, 작업 프로세스)→ stems/vocals.wav, instrumental.wav
          ─(pYIN, 작업 프로세스)→ cache/…/pitch.npz ─(후처리)→ PitchCurve
          ─(BPM, 작업 프로세스)→ cache/…/tempo.json
          ─(음표 검출, 이 프로세스)→ Note 목록 → AnalysisResult

- 분리 결과는 정식 결과로 프로젝트 ``stems/`` 에 두고 stems.json으로 유효성을 관리한다.
- 피치·BPM 원시 결과는 ``cache/`` 에 둔다. 캐시 키는 원본 해시 + 입력(분리 결과 id) + 파라미터다.
- 음표 검출은 빠르므로 캐시하지 않는다. ``redetect_notes`` 로 설정만 바꿔 다시 돌릴 수 있다.

STEP 1의 메모리 기반 단계 인터페이스(``interfaces.py``, 샘플 백엔드)는 그대로 둔다.
무거운 모델은 GUI 밖 작업 프로세스에서 돌려야 해서, 이 파이프라인은 파일을 주고받는다.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from amp.audio.data import AudioData
from amp.core.errors import AnalysisError
from amp.core.model import AnalysisResult, ModelInfo
from amp.core.tempo import TempoMap
from amp.engine import stems as stem_store
from amp.engine.cache import AnalysisCache
from amp.engine.context import CancelToken, ProgressFn
from amp.engine.dsp.pitch import FMAX_HZ, FMIN_HZ, PitchFrames
from amp.engine.features import vocal_presence
from amp.engine.notes import detect_notes
from amp.engine.pitch_post import articulation_envelope, build_curve
from amp.engine.settings import AnalysisSettings
from amp.engine.worker_client import run_worker

log = logging.getLogger(__name__)

PITCH_VERSION = 1
TEMPO_VERSION = 1
NOTES_VERSION = "1"
SEPARATION_MODEL = "htdemucs"

Runner = Callable[..., dict]


@dataclass
class TranscriptionRequest:
    audio: AudioData  # GUI가 이미 디코딩한 원본
    source_path: str  # 원본 파일 (작업 프로세스가 같은 디코더로 다시 읽는다)
    source_sha256: str
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    stems_dir: str | None = None  # 프로젝트 stems/ (없으면 임시 폴더, 저장 안 함)
    cache_root: str | None = None  # 프로젝트 cache/
    force_separation: bool = False  # 사용자가 '다시 분리'를 요청했을 때
    numba_cache_dir: str | None = None


@dataclass
class TranscriptionOutput:
    result: AnalysisResult
    stems: dict[str, AudioData]
    manifest: stem_store.StemManifest | None
    reused: dict[str, bool]


class _Progress:
    def __init__(self, report: ProgressFn, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        self.report = report
        self.spans = {}
        acc = 0.0
        for k, w in weights.items():
            self.spans[k] = (acc / total, w / total)
            acc += w

    def stage(self, name: str) -> ProgressFn:
        base, span = self.spans[name]
        return lambda f, m: self.report(base + span * min(max(f, 0.0), 1.0), m)


class TranscriptionPipeline:
    def __init__(self, runner: Runner = run_worker, separation_model: str = SEPARATION_MODEL) -> None:
        self.runner = runner
        self.separation_model = separation_model  # 분리 모델 교체 지점 (작업 프로세스가 아는 이름)

    def run(self, req: TranscriptionRequest, progress: ProgressFn | None = None,
            token: CancelToken | None = None) -> TranscriptionOutput:
        token = token or CancelToken()
        report = progress or (lambda f, m: None)
        st = req.settings
        separate = st.separation == "htdemucs"
        prog = _Progress(report, {"sep": 0.6, "pitch": 0.3, "tempo": 0.07, "notes": 0.03} if separate
                         else {"pitch": 0.75, "tempo": 0.2, "notes": 0.05})
        warnings: list[str] = []
        reused = {"separation": False, "pitch": False, "tempo": False}
        cache = AnalysisCache(req.cache_root) if req.cache_root else None
        key = req.source_sha256[:32] or "nohash"
        models: dict[str, ModelInfo] = {}

        with tempfile.TemporaryDirectory(prefix="sonatlas-") as scratch:
            scratch = Path(scratch)
            manifest = None
            stems: dict[str, AudioData] = {}
            stems_dir = Path(req.stems_dir) if req.stems_dir else scratch / "stems"
            if separate:
                manifest, reused["separation"] = self._separation(req, stems_dir, prog.stage("sep"), token)
                stems = stem_store.load(stems_dir, manifest)
                for name, a in stems.items():
                    if a.frames != req.audio.frames or a.sample_rate != req.audio.sample_rate:
                        raise AnalysisError(f"분리된 {name}의 길이·샘플레이트가 원본과 다릅니다. 다시 분리해 주세요.")
                presence = vocal_presence(stems["vocals"].mono(), req.audio.mono(), req.audio.sample_rate)
                if not presence["present"]:
                    warnings.append(
                        f"보컬이 거의 없는 음원으로 보입니다 (보컬/전체 에너지 {presence['ratio_db']}dB). "
                        "음표가 적게 나오거나 나오지 않을 수 있습니다."
                    )
                models["separation"] = ModelInfo(manifest.model, str(manifest.model_info.get("demucs", "")),
                                                 params={**manifest.model_info, "presence": presence})
                pitch_input = str(stems_dir / manifest.files["vocals"]["path"])
                tempo_input = str(stems_dir / manifest.files["instrumental"]["path"])
                input_id = f"stems:{manifest.stems_id}"
            else:
                pitch_input = tempo_input = req.source_path
                input_id = "original"
                models["separation"] = ModelInfo("none", params={"note": "분리하지 않고 원본을 분석"})

            token.raise_if_cancelled()
            pitch_params = {"tracker": "pyin", "v": PITCH_VERSION, "fmin": FMIN_HZ, "fmax": FMAX_HZ, "input": input_id}
            pdir, reused["pitch"] = self._cached(cache, key, "pitch", pitch_params, st.use_cache, scratch,
                                                 lambda out, p: self.runner("pitch", {"input": pitch_input, "out_dir": str(out)},
                                                                            p, token, req.numba_cache_dir),
                                                 prog.stage("pitch"))
            frames = PitchFrames.load(pdir / "pitch.npz")
            pmeta = _read_json(pdir / "meta.json")

            token.raise_if_cancelled()
            tempo_params = {"estimator": "librosa", "v": TEMPO_VERSION, "input": input_id}
            tdir, reused["tempo"] = self._cached(cache, key, "tempo", tempo_params, st.use_cache, scratch,
                                                 lambda out, p: self.runner("tempo", {"input": tempo_input, "out_dir": str(out)},
                                                                            p, token, req.numba_cache_dir),
                                                 prog.stage("tempo"))
            tempo = _read_json(tdir / "tempo.json")

        token.raise_if_cancelled()
        notes_report = prog.stage("notes")
        notes_report(0.0, "음표 검출 중")
        curve = build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate, frames.start_time)
        params = st.note_params()
        artic = articulation_envelope(frames.onset, frames.rms, frames.frame_rate)
        notes = detect_notes(curve, artic, params)
        notes_report(1.0, f"음표 {len(notes)}개 검출")
        if tempo.get("reason"):
            warnings.append(tempo["reason"])

        pv = pmeta.get("versions", {})
        models["pitch"] = ModelInfo("pyin", f"librosa {pv.get('librosa', '')}".strip(),
                                    params={k: pmeta.get(k) for k in ("fmin", "fmax", "frame_rate")})
        models["tempo"] = ModelInfo("librosa-beat", str(tempo.get("versions", {}).get("librosa", "")),
                                    params={"reliable": tempo.get("reliable", False)})
        models["notes"] = ModelInfo("segment", NOTES_VERSION, params=dataclasses.asdict(params))
        result = AnalysisResult(
            duration=req.audio.duration,
            tempo=TempoMap.constant(float(tempo["bpm"]), float(tempo.get("first_beat", 0.0))),
            notes=tuple(notes),
            pitch_curve=curve,
            models=models,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            audio_fingerprint=req.source_sha256,
            onset_envelope=artic,  # 음표 재검출에 쓰는 재발음 지표 (온셋 × 음량 하락)
            tempo_info={k: v for k, v in tempo.items() if k != "versions"} | {"algorithm": "librosa beat_track + 박 간격 직선 보정"},
            warnings=tuple(warnings),
            settings=st.to_dict(),
        )
        return TranscriptionOutput(result, stems, manifest, reused)

    # --- 단계 -----------------------------------------------------------------

    def _separation(self, req: TranscriptionRequest, stems_dir: Path, report: ProgressFn, token: CancelToken):
        manifest = stem_store.read_manifest(stems_dir) if stems_dir.exists() else None
        ok, reason = stem_store.validate(stems_dir, manifest, req.source_sha256)
        if ok and manifest is not None and manifest.model == self.separation_model and not req.force_separation:
            report(1.0, "기존 분리 결과 사용")
            return manifest, True
        if manifest is not None and not ok:
            log.info("기존 분리 결과를 쓸 수 없음: %s", reason)
        stems_dir.mkdir(parents=True, exist_ok=True)
        tmp = stems_dir / f".tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            self.runner("separate", {"input": req.source_path, "out_dir": str(tmp), "model": self.separation_model,
                                     "device": req.settings.device}, report, token, req.numba_cache_dir)
            manifest = stem_store.commit(tmp, stems_dir, req.source_sha256, {"device": req.settings.device})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # 성공이면 이미 비었고, 실패·취소면 임시 파일을 지운다
        return manifest, False

    @staticmethod
    def _cached(cache: AnalysisCache | None, key: str, stage: str, params: dict, use_cache: bool, scratch: Path,
                compute: Callable[[Path, ProgressFn], dict], report: ProgressFn) -> tuple[Path, bool]:
        if cache is not None:
            hit = cache.lookup(key, stage, params) if use_cache else None
            if hit is not None:
                report(1.0, "캐시 사용")
                return hit, True
            with cache.writer(key, stage, params) as tmp:
                compute(tmp, report)
            found = cache.lookup(key, stage, params)
            if found is None:
                raise AnalysisError(f"{stage} 캐시를 저장하지 못했습니다.")
            return found, False
        out = scratch / stage
        out.mkdir(parents=True, exist_ok=True)
        compute(out, report)
        return out, False


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def redetect_notes(result: AnalysisResult, settings: AnalysisSettings) -> AnalysisResult:
    """보컬 분리·피치 추출 결과는 그대로 두고 음표 검출만 다시 한다."""
    if result.pitch_curve is None:
        raise AnalysisError("피치 곡선이 없어 음표를 다시 검출할 수 없습니다. 먼저 분석을 실행하세요.")
    params = settings.note_params()
    notes = detect_notes(result.pitch_curve, result.onset_envelope, params)
    models = dict(result.models)
    models["notes"] = ModelInfo("segment", NOTES_VERSION, params=dataclasses.asdict(params))
    merged = dict(result.settings)
    merged.update({k: v for k, v in settings.to_dict().items()
                   if k in ("sensitivity", "min_note_ms", "pitch_change_semitones", "pitch_low", "pitch_high")})
    return dataclasses.replace(result, notes=tuple(notes), models=models, settings=merged,
                               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
