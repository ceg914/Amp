"""분석 결과 공통 머리말과 상태 (v0.7.0, 명세 17.1·17.2·14.3). Qt 비의존.

v0.7.0에서 새로 만드는 분석 결과 파일(피치 모델별 결과, 다중 피치, 성부, 비브라토, 호흡, 발성, 경계)은 모두
``meta`` 항목에 이 머리말을 둔다. 이전 파일(notes·lyrics 등)은 형식을 바꾸지 않는다.

상태:
- not_analyzed: 아직 분석하지 않음 · processing: 분석 중 · completed: 끝남
- uncertain: 끝났지만 근거가 약함(입력이 너무 짧음, 유성 구간이 거의 없음 등)
- stale: 분석에 쓴 소리가 바뀜. 결과를 지우지 않고 오래됨으로 표시한다
- failed: 실패(이유를 status_reason에) · unavailable: 모델·패키지가 없어 할 수 없음

입력 구분 (14.3): source_audio_hash는 프로젝트 원본, input_hash는 실제로 분석한 소리(보컬·반주·Hybrid·악기)의 해시.
input_kind와 input_id(분리 결과 id 등)로 어떤 소리였는지 적는다. 같은 모델이라도 입력이 다르면 다른 결과다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone

NOT_ANALYZED = "not_analyzed"
PROCESSING = "processing"
COMPLETED = "completed"
UNCERTAIN = "uncertain"
STALE = "stale"
FAILED = "failed"
UNAVAILABLE = "unavailable"
STATUSES = (NOT_ANALYZED, PROCESSING, COMPLETED, UNCERTAIN, STALE, FAILED, UNAVAILABLE)
STATUS_LABELS = {NOT_ANALYZED: "분석 안 함", PROCESSING: "분석 중", COMPLETED: "완료", UNCERTAIN: "불확실",
                 STALE: "오래됨", FAILED: "실패", UNAVAILABLE: "사용 불가"}

INPUT_KINDS = ("original", "vocals", "instrumental", "hybrid", "instrument", "external")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def settings_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings or {}, sort_keys=True, ensure_ascii=False, default=str)
                          .encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ResultMeta:
    schema: str
    schema_version: int
    source_audio_id: str  # 프로젝트 원본 id (파일 이름 등 표시용 아님: 원본 해시 앞부분)
    source_audio_hash: str  # 프로젝트 원본 sha256
    input_kind: str = "original"  # 실제 분석 입력의 종류
    input_id: str = ""  # 분리 결과 id·Hybrid id·악기 자리 등 (원본이면 '')
    input_hash: str = ""  # 실제 분석 입력 소리의 sha256 (원본이면 source_audio_hash)
    model_id: str = ""
    model_version: str = ""
    analysis_version: str = ""  # 분석 코드 버전 (엔진 쪽 상수)
    analysis_settings: dict = field(default_factory=dict)
    analysis_range: tuple[float, float] | None = None  # (시작초, 끝초). None = 곡 전체
    created_at: str = ""
    status: str = COMPLETED
    status_reason: str = ""

    @property
    def analysis_source_id(self) -> str:
        """같은 입력 소리인지 비교하는 값 (명세 4.3 analysis_source_id)."""
        return f"{self.input_kind}:{self.input_id}:{(self.input_hash or self.source_audio_hash)[:16]}"

    @property
    def settings_hash(self) -> str:
        return settings_hash(self.analysis_settings)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["analysis_range"] = list(self.analysis_range) if self.analysis_range is not None else None
        return d

    def with_status(self, status: str, reason: str = "") -> ResultMeta:
        if status not in STATUSES:
            raise ValueError(f"알 수 없는 상태: {status}")
        return replace(self, status=status, status_reason=reason)

    def same_source(self, other: ResultMeta) -> bool:
        return self.source_audio_hash == other.source_audio_hash and self.analysis_source_id == other.analysis_source_id


def make_meta(schema: str, schema_version: int, source_audio_hash: str, *, input_kind: str = "original",
              input_id: str = "", input_hash: str = "", model_id: str = "", model_version: str = "",
              analysis_version: str = "", analysis_settings: dict | None = None,
              analysis_range: tuple[float, float] | None = None, status: str = COMPLETED,
              status_reason: str = "") -> ResultMeta:
    if input_kind not in INPUT_KINDS:
        raise ValueError(f"알 수 없는 입력 종류: {input_kind}")
    if status not in STATUSES:
        raise ValueError(f"알 수 없는 상태: {status}")
    rng = None if analysis_range is None else (round(float(analysis_range[0]), 4), round(float(analysis_range[1]), 4))
    return ResultMeta(schema, int(schema_version), source_audio_hash[:16], source_audio_hash, input_kind, input_id,
                      input_hash or (source_audio_hash if input_kind == "original" else ""), model_id, model_version,
                      analysis_version, dict(analysis_settings or {}), rng, now_iso(), status, status_reason)


def meta_from_dict(d: dict | None) -> ResultMeta | None:
    """읽기는 너그럽게: 모르는 키는 버리고, 상태가 이상하면 uncertain으로 둔다. 머리말이 없으면 None."""
    if not isinstance(d, dict) or not d.get("schema"):
        return None
    rng = d.get("analysis_range")
    try:
        rng = (float(rng[0]), float(rng[1])) if rng else None
    except (TypeError, ValueError, IndexError):
        rng = None
    status = str(d.get("status") or COMPLETED)
    reason = str(d.get("status_reason") or "")
    if status not in STATUSES:
        status, reason = UNCERTAIN, f"알 수 없는 상태 값({status})"
    try:
        version = int(d.get("schema_version") or 0)
    except (TypeError, ValueError):
        version = 0
    return ResultMeta(str(d["schema"]), version, str(d.get("source_audio_id") or ""),
                      str(d.get("source_audio_hash") or ""), str(d.get("input_kind") or "original"),
                      str(d.get("input_id") or ""), str(d.get("input_hash") or ""), str(d.get("model_id") or ""),
                      str(d.get("model_version") or ""), str(d.get("analysis_version") or ""),
                      dict(d.get("analysis_settings") or {}), rng, str(d.get("created_at") or ""), status, reason)


def mark_stale_if_changed(meta: ResultMeta, source_audio_hash: str, analysis_source_id: str | None = None,
                          reason: str = "") -> ResultMeta:
    """원본이나 분석 입력이 바뀌었으면 stale로 바꾼 머리말을 돌려준다 (결과는 지우지 않는다)."""
    if meta.status == STALE:
        return meta
    if meta.source_audio_hash and source_audio_hash and meta.source_audio_hash != source_audio_hash:
        return meta.with_status(STALE, reason or "원본 오디오가 분석 뒤 바뀌었습니다.")
    if analysis_source_id is not None and meta.analysis_source_id != analysis_source_id:
        return meta.with_status(STALE, reason or "분석에 쓴 소리(분리 결과 등)가 바뀌었습니다.")
    return meta
