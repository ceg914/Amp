"""베이스 TAB 파이프라인 (v0.6.3). Qt 비의존.

    베이스 stem 찾기 (분리 보관소의 htdemucs 결과)
        └ 없으면 htdemucs로 다시 분리해 베이스만 보관 결과에 더한다 (보컬·반주 파일은 바꾸지 않음)
    ─(pYIN, 베이스 음역, 작업 프로세스)→ cache/…/bass_pitch
    ─(음표 검출 + 퉁김 분할, 이 프로세스)→ 베이스 음표

박 맞춤(quantize)과 줄·프렛 배치는 가볍고 템포·옵션에 따라 바뀌므로 호출한 쪽이 한다.
베이스 피치는 보컬 피치와 다른 캐시 단계(bass_pitch)라 보컬 분석 결과·캐시에 영향을 주지 않는다.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from amp.core.errors import AnalysisError
from amp.core.model import Note
from amp.engine import stems as stem_store
from amp.engine.bass import BASS_NOTES_VERSION, BassOptions, detect_bass_notes
from amp.engine.cache import AnalysisCache
from amp.engine.context import CancelToken, ProgressFn
from amp.engine.model_store import ModelStore, identity_for
from amp.engine.transcription import PITCH_VERSION, Runner, TranscriptionPipeline, _load_pitch
from amp.engine.worker_client import run_worker

log = logging.getLogger(__name__)

BASS_MODEL = "htdemucs"  # 베이스를 따로 내는 분리 모델 (RoFormer 보컬 모델은 보컬·반주만 낸다)


@dataclass
class BassRequest:
    source_path: str
    source_sha256: str
    duration: float
    options: BassOptions = field(default_factory=BassOptions)
    stems_dir: str | None = None  # 프로젝트 stems/ (없으면 임시 폴더, 저장 안 함)
    cache_root: str | None = None
    device: str = "auto"
    use_cache: bool = True
    use_original: bool = False  # 베이스만 녹음한 파일처럼 분리 없이 원본을 그대로 분석
    allow_separation: bool = True  # False면 베이스 stem이 없을 때 분리하지 않고 오류
    numba_cache_dir: str | None = None


@dataclass
class BassOutput:
    notes: list[Note]
    curve: object  # PitchCurve
    source: dict  # {"kind": "stem"|"original", "store_key", "path", "separated": bool}
    reused: dict[str, bool]
    pitch_meta: dict


def find_bass_stem(stems_dir: str | Path, source_sha256: str) -> tuple[str, Path] | None:
    """(보관 키, 베이스 파일). 지금 쓰는 분리 결과를 먼저, 그다음 최근 결과."""
    store = ModelStore(stems_dir, source_sha256)
    entries = store.entries()
    entries.sort(key=lambda e: e.manifest.created, reverse=True)
    entries.sort(key=lambda e: not e.active)
    for e in entries:
        p = stem_store.optional_path(e.path, e.manifest, "bass")
        if p is not None:
            return e.key, p
    return None


def bass_input_id(stems_dir: str | Path, source_sha256: str, key: str) -> str:
    e = ModelStore(stems_dir, source_sha256).entry(key)
    if e is None:
        return f"bass:{key}"
    f = e.manifest.files.get("bass", {})
    return f"bass:{e.manifest.stems_id}:{f.get('added', '')}:{f.get('size', '')}"


class BassPipeline:
    def __init__(self, runner: Runner = run_worker, separation_model: str = BASS_MODEL) -> None:
        self.runner = runner
        self.separation_model = separation_model  # 시험에서만 바꾼다 (_withbass)

    def run(self, req: BassRequest, progress: ProgressFn | None = None, token: CancelToken | None = None) -> BassOutput:
        token = token or CancelToken()
        report = progress or (lambda f, m: None)
        cache = AnalysisCache(req.cache_root) if req.cache_root else None
        key = req.source_sha256[:32] or "nohash"
        reused = {"separation": False, "pitch": False}
        warnings: list[str] = []
        with tempfile.TemporaryDirectory(prefix="sonatlas-bass-") as scratch:
            scratch = Path(scratch)
            stems_dir = Path(req.stems_dir) if req.stems_dir else scratch / "stems"
            if req.use_original:
                path, input_id = Path(req.source_path), "original"
                source = {"kind": "original", "path": str(path), "separated": False}
                sep_share = 0.0
            else:
                found = find_bass_stem(stems_dir, req.source_sha256)
                sep_share = 0.0 if found else 0.6
                if found is None:
                    if not req.allow_separation:
                        raise AnalysisError("베이스 분리 결과가 없습니다. 보컬 분리(Demucs)를 설치하거나, "
                                            "베이스만 녹음한 파일이면 '원본 그대로 분석'을 고르세요.")
                    found = self._separate_bass(req, stems_dir, lambda f, m: report(sep_share * f, m), token)
                else:
                    reused["separation"] = True
                    report(0.0, "보관한 베이스 분리 결과 사용")
                store_key, path = found
                input_id = bass_input_id(stems_dir, req.source_sha256, store_key)
                source = {"kind": "stem", "store_key": store_key, "path": str(path),
                          "separated": not reused["separation"]}
            token.raise_if_cancelled()
            fmin, fmax = req.options.hz_range()
            params = {"tracker": "pyin", "v": PITCH_VERSION, "fmin": round(fmin, 3), "fmax": round(fmax, 3),
                      "input": input_id}
            expected = int(req.duration * 100) + 1
            span = 1.0 - sep_share - 0.05

            def pitch_report(f, m):
                report(sep_share + span * f, m)

            (frames, meta), reused["pitch"] = TranscriptionPipeline._cached(
                cache, key, "bass_pitch", params, req.use_cache, scratch,
                lambda out, p: self.runner("pitch", {"input": str(path), "fmin": fmin, "fmax": fmax, "out_dir": str(out)},
                                           p, token, req.numba_cache_dir),
                lambda d: _load_pitch(d, expected), pitch_report, warnings)
        token.raise_if_cancelled()
        report(0.96, "베이스 음표 검출 중")
        notes, curve, _art = detect_bass_notes(frames, req.options)
        report(1.0, f"베이스 음표 {len(notes)}개")
        pmeta = {**{k: meta.get(k) for k in ("fmin", "fmax", "frame_rate")}, "notes_version": BASS_NOTES_VERSION,
                 "warnings": warnings}
        return BassOutput(notes, curve, source, reused, pmeta)

    def _separate_bass(self, req: BassRequest, stems_dir: Path, report: ProgressFn, token: CancelToken):
        """htdemucs로 분리해 베이스만 보관 결과에 더한다. 같은 htdemucs 결과가 있으면 그 폴더에 bass.wav만 넣고,
        없으면 새 보관 결과로 넣는다(지금 쓰는 분리 결과는 바꾸지 않는다)."""
        store = ModelStore(stems_dir, req.source_sha256)
        identity = identity_for(self.separation_model, req.source_sha256)
        existing = store.find(identity)
        tmp = store.new_tmp()
        try:
            self.runner("separate", {"input": req.source_path, "out_dir": str(tmp), "model": self.separation_model,
                                     "device": req.device, "cpu_fallback": True}, report, token, req.numba_cache_dir)
            if not (tmp / "bass.wav").is_file():
                raise AnalysisError("분리 결과에 베이스가 없습니다. Demucs htdemucs로만 베이스를 따로 낼 수 있습니다.")
            if existing is not None:
                entry = store.add_stem(existing.key, "bass", tmp / "bass.wav")
            else:
                entry = store.commit(tmp, identity, {"device": req.device, "purpose": "bass"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                store.root.rmdir()
            except OSError:
                pass
        p = stem_store.optional_path(entry.path, entry.manifest, "bass")
        if p is None:
            raise AnalysisError("베이스 분리 결과를 저장하지 못했습니다.")
        return entry.key, p
