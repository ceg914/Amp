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
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from amp.audio.data import AudioData
from amp.core.errors import AnalysisError
from amp.core.model import AnalysisResult, ModelInfo
from amp.core.tempo import TempoMap
from amp.engine import stems as stem_store
from amp.engine.stem_import import EXTERNAL
from amp.engine.cache import AnalysisCache
from amp.engine.context import CancelToken, ProgressFn
from amp.engine.dsp.pitch import FMAX_HZ, FMIN_HZ, PitchFrames
from amp.engine.features import vocal_presence
from amp.engine.notes import detect_notes
from amp.engine.pitch_post import articulation_envelope, build_curve
from amp.engine.settings import AnalysisSettings
from amp.engine.worker_client import run_worker

log = logging.getLogger(__name__)

PITCH_VERSION = 2  # 2: 온셋 32ms 창·비중심, 20ms RMS 추가 (STEP 2.5)
TEMPO_VERSION = 1
OLD_FEATURES_WARNING = ("이전 버전(STEP 2)에서 추출한 피치·온셋 정보로 음표를 다시 검출했습니다. "
                        "개선된 음표 시작·끝 판정을 모두 반영하려면 전체 분석을 다시 실행하세요.")
NOTES_VERSION = "3"  # 2: 틈 게이트·재발음 절대 임계·미끄러짐 흡수·발음된 꾸밈음 (STEP 2.5), 3: 접근음(끌어올림) 흡수 (STEP 3.5)
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
    # 정렬한 가사의 음절 시작(초). 접근음 흡수가 다음 음절의 음을 대상으로 삼지 않게 한다 (설정에는 저장하지 않음)
    syllable_starts: tuple = ()
    # RoFormer 모델 파일 {"checkpoint", "config"}. 컴퓨터마다 다른 경로라 프로젝트가 아니라 프로그램 설정에 둔다
    model_files: dict = field(default_factory=dict)


@dataclass
class TranscriptionOutput:
    result: AnalysisResult
    stems: dict[str, AudioData]
    manifest: stem_store.StemManifest | None
    reused: dict[str, bool]
    timings: dict[str, float] = field(default_factory=dict)  # 단계 → 초 (캐시 재사용이면 확인 시간만)
    resources: dict[str, dict] = field(default_factory=dict)  # 단계 → 작업 프로세스 최대 메모리 등


_COMMAND_STAGE = {"separate": "separation", "pitch": "pitch", "tempo": "tempo"}


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


def _same_model_file(manifest: stem_store.StemManifest, req: TranscriptionRequest) -> bool:
    """RoFormer 결과는 같은 모델 파일(경로·크기·수정 시각)로 만든 것일 때만 다시 쓴다. 파일을 지정하지 않았으면 있는 결과를 쓴다."""
    if manifest.model != "mel_roformer" or not req.model_files.get("checkpoint"):
        return True
    from amp.engine.dsp.roformer import file_identity

    try:
        now = file_identity(req.model_files["checkpoint"])
    except OSError:
        return True  # 지정한 파일이 없어졌으면 다시 분리할 수도 없다. 있는 결과를 쓴다
    old = manifest.model_info.get("checkpoint_identity") or {}
    same_cfg = os.path.basename(req.model_files.get("config") or "") == manifest.model_info.get("config", "")
    return old == now and same_cfg


class TranscriptionPipeline:
    def __init__(self, runner: Runner = run_worker, separation_model: str | None = None) -> None:
        self.runner = runner
        # 분리 모델을 고정할 때만 준다 (시험용 _identity 등). None이면 설정의 분리 방식(htdemucs / mel_roformer)
        self.separation_model = separation_model
        self.resources: dict[str, dict] = {}

    def _call(self, command: str, args: dict, progress: ProgressFn, token: CancelToken, numba_cache_dir) -> dict:
        out = self.runner(command, args, progress, token, numba_cache_dir)
        if isinstance(out, dict) and isinstance(out.get("resources"), dict):
            self.resources[_COMMAND_STAGE.get(command, command)] = out["resources"]
        return out

    def run(self, req: TranscriptionRequest, progress: ProgressFn | None = None,
            token: CancelToken | None = None) -> TranscriptionOutput:
        token = token or CancelToken()
        report = progress or (lambda f, m: None)
        st = req.settings
        separate = st.separation != "none"
        prog = _Progress(report, {"sep": 0.6, "pitch": 0.3, "tempo": 0.07, "notes": 0.03} if separate
                         else {"pitch": 0.75, "tempo": 0.2, "notes": 0.05})
        warnings: list[str] = []
        reused = {"separation": False, "pitch": False, "tempo": False}
        cache = AnalysisCache(req.cache_root) if req.cache_root else None
        key = req.source_sha256[:32] or "nohash"
        models: dict[str, ModelInfo] = {}
        self.resources = {}
        timings: dict[str, float] = {}
        t_start = clock = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[stage] = now - clock
            clock = now

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
                note = str(manifest.model_info.get("note") or "")
                if note and not reused["separation"]:  # GPU를 못 써 CPU로 분리, GPU 메모리 부족 → CPU 재실행 등
                    warnings.append(note)
                presence = vocal_presence(stems["vocals"].mono(), req.audio.mono(), req.audio.sample_rate)
                if not presence["present"]:
                    warnings.append(
                        f"보컬이 거의 없는 음원으로 보입니다 (보컬/전체 에너지 {presence['ratio_db']}dB). "
                        "음표가 적게 나오거나 나오지 않을 수 있습니다."
                    )
                models["separation"] = ModelInfo(manifest.model, str(manifest.model_info.get("demucs", "")),
                                                 params={**manifest.model_info, "presence": presence})
                lap("separation")
                pitch_input = str(stems_dir / manifest.files["vocals"]["path"])
                tempo_input = str(stems_dir / manifest.files["instrumental"]["path"])
                input_id = f"stems:{manifest.stems_id}"
            else:
                pitch_input = tempo_input = req.source_path
                input_id = "original"
                models["separation"] = ModelInfo("none", params={"note": "분리하지 않고 원본을 분석"})

            token.raise_if_cancelled()
            pitch_params = {"tracker": "pyin", "v": PITCH_VERSION, "fmin": FMIN_HZ, "fmax": FMAX_HZ, "input": input_id}
            expected = int(req.audio.duration * 100) + 1
            (frames, pmeta), reused["pitch"] = self._cached(
                cache, key, "pitch", pitch_params, st.use_cache, scratch,
                lambda out, p: self._call("pitch", {"input": pitch_input, "out_dir": str(out)}, p, token, req.numba_cache_dir),
                lambda d: _load_pitch(d, expected), prog.stage("pitch"), warnings)
            lap("pitch")

            token.raise_if_cancelled()
            tempo_params = {"estimator": "librosa", "v": TEMPO_VERSION, "input": input_id}
            tempo, reused["tempo"] = self._cached(
                cache, key, "tempo", tempo_params, st.use_cache, scratch,
                lambda out, p: self._call("tempo", {"input": tempo_input, "out_dir": str(out)}, p, token, req.numba_cache_dir),
                _load_tempo, prog.stage("tempo"), warnings)
            lap("tempo")

        token.raise_if_cancelled()
        notes_report = prog.stage("notes")
        notes_report(0.0, "음표 검출 중")
        curve = build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate, frames.start_time,
                            rms_fine=frames.rms_fine)
        params = dataclasses.replace(st.note_params(), syllable_starts=tuple(req.syllable_starts))
        artic = articulation_envelope(frames.onset, frames.rms, frames.frame_rate, rms_fine=frames.rms_fine)
        notes = detect_notes(curve, artic, params)
        notes_report(1.0, f"음표 {len(notes)}개 검출")
        lap("notes")
        timings["total"] = time.perf_counter() - t_start
        if tempo.get("reason"):
            warnings.append(tempo["reason"])

        pv = pmeta.get("versions", {})
        models["pitch"] = ModelInfo("pyin", f"librosa {pv.get('librosa', '')}".strip(),
                                    params={**{k: pmeta.get(k) for k in ("fmin", "fmax", "frame_rate")},
                                            "features_version": PITCH_VERSION})
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
        return TranscriptionOutput(result, stems, manifest, reused, timings, dict(self.resources))

    # --- 단계 -----------------------------------------------------------------

    def _separation(self, req: TranscriptionRequest, stems_dir: Path, report: ProgressFn, token: CancelToken):
        manifest = stem_store.read_manifest(stems_dir) if stems_dir.exists() else None
        ok, reason = stem_store.validate(stems_dir, manifest, req.source_sha256)
        model = self.separation_model or req.settings.separation
        reusable = manifest is not None and manifest.model in (model, EXTERNAL) and _same_model_file(manifest, req)
        if ok and reusable and not req.force_separation:
            # 사용자가 가져온 외부 분리 결과(model='external')도 그대로 쓴다. Demucs로 덮어쓰지 않는다
            report(1.0, "가져온 분리 결과 사용" if manifest.model == EXTERNAL else "기존 분리 결과 사용")
            return manifest, True
        if manifest is not None and not ok:
            log.info("기존 분리 결과를 쓸 수 없음: %s", reason)
        stems_dir.mkdir(parents=True, exist_ok=True)
        tmp = stems_dir / f".tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            args = {"input": req.source_path, "out_dir": str(tmp), "model": model, "device": req.settings.device}
            if model == "mel_roformer":
                if not req.model_files.get("checkpoint"):
                    raise AnalysisError("Mel-Band RoFormer 모델 파일(.ckpt)을 고르지 않았습니다. 분석 설정의 보컬 분리에서 "
                                        "[모델 파일…]로 지정하세요.")
                args["model_files"] = dict(req.model_files)
            self._call("separate", args, report, token, req.numba_cache_dir)
            manifest = stem_store.commit(tmp, stems_dir, req.source_sha256, {"device": req.settings.device})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # 성공이면 이미 비었고, 실패·취소면 임시 파일을 지운다
        return manifest, False

    @staticmethod
    def _cached(cache: AnalysisCache | None, key: str, stage: str, params: dict, use_cache: bool, scratch: Path,
                compute: Callable[[Path, ProgressFn], dict], load: Callable[[Path], object], report: ProgressFn,
                warnings: list[str]):
        """(불러온 값, 재사용 여부). 캐시가 깨졌으면 지우고 다시 계산한다.

        새로 계산한 결과도 캐시로 확정하기 전에 load로 검사한다. 검사에 실패하면 캐시에 남지 않는다.
        """
        if cache is not None:
            hit = cache.lookup(key, stage, params) if use_cache else None
            if hit is not None:
                try:
                    value = load(hit)
                    report(1.0, "캐시 사용")
                    return value, True
                except Exception as e:  # noqa: BLE001 - 어떤 손상이든 다시 계산하면 된다
                    log.warning("%s 캐시가 손상되어 다시 계산합니다: %s (%s)", stage, hit, e)
                    warnings.append(f"{stage} 캐시가 손상되어 다시 계산했습니다.")
                    cache.remove(key, stage, params)
            with cache.writer(key, stage, params) as tmp:
                compute(tmp, report)
                value = load(tmp)  # 실패하면 writer가 임시 폴더를 지운다 → 깨진 결과가 캐시에 남지 않는다
            return value, False
        out = scratch / stage
        out.mkdir(parents=True, exist_ok=True)
        compute(out, report)
        return load(out), False


class CacheEntryError(AnalysisError):
    pass


def _load_pitch(d: Path, expected_frames: int) -> tuple[PitchFrames, dict]:
    try:
        frames = PitchFrames.load(d / "pitch.npz")
    except Exception as e:  # noqa: BLE001 - zip·npz 손상
        raise CacheEntryError(f"pitch.npz를 읽을 수 없습니다: {e}") from e
    n = len(frames.f0_hz)
    if not (len(frames.voiced_prob) == len(frames.onset) == len(frames.rms) == n):
        raise CacheEntryError("피치 배열 길이가 서로 다릅니다.")
    if abs(n - expected_frames) > 3:
        raise CacheEntryError(f"피치 프레임 수({n})가 오디오 길이({expected_frames})와 맞지 않습니다.")
    if frames.rms_fine is None or len(frames.rms_fine) != n:
        raise CacheEntryError("20ms RMS가 없거나 길이가 다릅니다.")
    f0 = frames.f0_hz
    if np.any(np.isfinite(f0) & (f0 <= 0)) or np.any(~np.isfinite(frames.voiced_prob)):
        raise CacheEntryError("피치 값이 올바르지 않습니다.")
    return frames, _read_json(d / "meta.json")


def _load_tempo(d: Path) -> dict:
    try:
        data = json.loads((d / "tempo.json").read_text(encoding="utf-8"))
        bpm = float(data["bpm"])
        float(data.get("first_beat", 0.0))
    except Exception as e:  # noqa: BLE001
        raise CacheEntryError(f"tempo.json을 읽을 수 없습니다: {e}") from e
    if not (20.0 <= bpm <= 400.0):
        raise CacheEntryError(f"BPM 값이 올바르지 않습니다: {bpm}")
    return data


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def redetect_notes(result: AnalysisResult, settings: AnalysisSettings, syllable_starts=()) -> AnalysisResult:
    """보컬 분리·피치 추출 결과는 그대로 두고 음표 검출만 다시 한다."""
    if result.pitch_curve is None:
        raise AnalysisError("피치 곡선이 없어 음표를 다시 검출할 수 없습니다. 먼저 분석을 실행하세요.")
    params = dataclasses.replace(settings.note_params(), syllable_starts=tuple(syllable_starts))
    notes = detect_notes(result.pitch_curve, result.onset_envelope, params)
    models = dict(result.models)
    models["notes"] = ModelInfo("segment", NOTES_VERSION, params=dataclasses.asdict(params))
    merged = dict(result.settings)
    merged.update({k: v for k, v in settings.to_dict().items()
                   if k in ("sensitivity", "min_note_ms", "pitch_change_semitones", "pitch_low", "pitch_high",
                            "no_vocal_ranges", "merge_approach")})
    warnings = tuple(w for w in result.warnings if w != OLD_FEATURES_WARNING)
    pitch_info = result.models.get("pitch")
    if pitch_info is not None and int((pitch_info.params or {}).get("features_version", 1)) < PITCH_VERSION:
        warnings += (OLD_FEATURES_WARNING,)
    return dataclasses.replace(result, notes=tuple(notes), models=models, settings=merged, warnings=warnings,
                               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
