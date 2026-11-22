"""Hybrid 보컬 소스 (v0.6). Qt 비의존.

곡을 시간 구간으로 나누고 구간마다 분리 모델 결과(보관소 항목)를 골라 하나의 보컬 트랙으로 합친다.

- 구간 경계는 샘플 번호(정수)로 저장한다. 구간 목록은 0부터 원본 끝까지 빈틈·겹침 없이 이어져야 하고,
  편집 연산은 모두 이 조건을 지키는 새 계획을 돌려준다(``validate`` 로 다시 확인할 수 있다).
- 합칠 때 원본 stem 파일은 읽기만 한다. 결과는 보관소의 새 항목(``hybrid-<해시>``)이 된다.
  같은 계획이면 같은 키라 다시 만들지 않는다.
- 경계에서는 앞 구간 소리를 줄이고 뒤 구간 소리를 키우는 직선 교차(크로스페이드)를 쓴다. 두 소스가 같은 보컬을
  추정한 것이라 서로 상관이 높아서, 두 가중치의 합이 1인 직선 교차가 음량을 일정하게 지킨다. 같은 크기를 지키는
  등전력 교차(합의 제곱이 1)는 상관 높은 신호에서 경계 가운데가 최대 +3dB 커진다.
- 교차 길이는 경계마다 정한다(구간의 ``crossfade_s`` 는 그 구간 시작 경계의 값). 이웃 경계의 교차 창이 겹치지
  않게 각 쪽 구간 길이의 절반으로 줄인다.
- 한 소스만 쓰는 구간(이득 0dB)은 원본 stem의 표본을 그대로 옮긴다. PCM 24bit → float32 → PCM 24bit가 손실 없이
  돌아오므로 그 구간 결과는 원본 stem과 비트 단위로 같다. 피치 재계산을 청크 단위로 건너뛸 수 있는 근거다.
- 합친 결과의 최대값이 1을 넘으면(이득을 올렸을 때) 전체를 같은 비율로 줄여 잘림을 막고 그 양을 기록한다.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from amp.core.errors import AnalysisError
from amp.core.ids import new_id

log = logging.getLogger(__name__)

HYBRID = "hybrid"
HYBRID_VERSION = "1"
SCHEMA = "sonatlas.hybrid"
DEFAULT_CROSSFADE_S = 0.03
MAX_CROSSFADE_S = 0.5
MIN_SEGMENT_S = 0.05
MAX_GAIN_DB = 12.0
CLIP_LIMIT = 1.0
PEAK_TARGET = 0.999
KEEP_RENDERS = 2  # 지금 쓰는 것 말고 남겨 둘 이전 Hybrid 결과 수 (전후 비교용). 더 오래된 것은 다시 만들 수 있어 지운다

STATUS_OK = "ok"
STATUS_CHECK = "needs_check"
STATUS_LABELS = {STATUS_OK: "확인됨", STATUS_CHECK: "검증 필요"}


class HybridError(AnalysisError):
    pass


@dataclass(frozen=True)
class HybridSegment:
    segment_id: str
    start_frame: int
    end_frame: int
    source_key: str  # 보관소 항목 키 (source_result_id)
    source_model: str  # htdemucs / mel_roformer / external (source_model_id)
    source_stems_id: str = ""  # 고를 때의 결과 id. 같은 키가 다시 만들어지면 달라진다
    crossfade_s: float = DEFAULT_CROSSFADE_S  # 앞 구간과의 경계 교차 길이 (첫 구간은 쓰지 않음)
    gain_db: float = 0.0
    user_selected: bool = True

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class SegmentStatus:
    state: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.state == STATUS_OK

    @property
    def label(self) -> str:
        return STATUS_LABELS.get(self.state, self.state)


@dataclass(frozen=True)
class HybridPlan:
    sample_rate: int
    frames: int
    segments: tuple[HybridSegment, ...]
    level_match: bool = False
    default_crossfade_s: float = DEFAULT_CROSSFADE_S

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def time(self, frame: int) -> float:
        return frame / self.sample_rate

    def frame(self, t: float) -> int:
        return int(min(max(round(t * self.sample_rate), 0), self.frames))

    def span(self, seg: HybridSegment) -> tuple[float, float]:
        return self.time(seg.start_frame), self.time(seg.end_frame)

    def index(self, segment_id: str) -> int:
        for i, s in enumerate(self.segments):
            if s.segment_id == segment_id:
                return i
        raise HybridError(f"구간을 찾을 수 없습니다: {segment_id}")

    def get(self, segment_id: str) -> HybridSegment:
        return self.segments[self.index(segment_id)]

    def at(self, t: float) -> HybridSegment:
        f = self.frame(t)
        for s in self.segments:
            if s.start_frame <= f < s.end_frame:
                return s
        return self.segments[-1]

    def sources(self) -> dict[str, str]:
        """쓰는 소스 키 → 모델."""
        return {s.source_key: s.source_model for s in self.segments}

    def boundaries(self) -> list[int]:
        return [s.start_frame for s in self.segments[1:]]


# --- 만들기·검사 -------------------------------------------------------------------

def single(sample_rate: int, frames: int, key: str, model: str, stems_id: str = "") -> HybridPlan:
    """곡 전체를 한 소스로."""
    if frames <= 0 or sample_rate <= 0:
        raise HybridError("오디오 길이가 0입니다.")
    return HybridPlan(sample_rate, frames, (HybridSegment(new_id(), 0, frames, key, model, stems_id),))


def validate(plan: HybridPlan) -> list[str]:
    """문제 목록 (비었으면 정상). 빈틈·겹침·길이·교차·이득·중복 id."""
    out: list[str] = []
    segs = plan.segments
    if not segs:
        return ["구간이 없습니다."]
    if segs[0].start_frame != 0:
        out.append("첫 구간이 0초에서 시작하지 않습니다.")
    if segs[-1].end_frame != plan.frames:
        out.append("마지막 구간이 곡 끝에서 끝나지 않습니다.")
    ids = [s.segment_id for s in segs]
    if len(set(ids)) != len(ids):
        out.append("구간 id가 겹칩니다.")
    for i, s in enumerate(segs):
        if s.end_frame <= s.start_frame:
            out.append(f"구간 {i + 1}의 길이가 0 이하입니다.")
        if i and segs[i - 1].end_frame != s.start_frame:
            kind = "겹칩니다" if segs[i - 1].end_frame > s.start_frame else "사이에 빈틈이 있습니다"
            out.append(f"구간 {i}과 {i + 1}이 {kind}.")
        if not (0.0 <= s.crossfade_s <= MAX_CROSSFADE_S):
            out.append(f"구간 {i + 1}의 교차 길이가 범위(0~{MAX_CROSSFADE_S * 1000:.0f}ms)를 벗어납니다.")
        if not (-MAX_GAIN_DB <= s.gain_db <= MAX_GAIN_DB) or not np.isfinite(s.gain_db):
            out.append(f"구간 {i + 1}의 음량 조정이 범위(±{MAX_GAIN_DB:.0f}dB)를 벗어납니다.")
        if not s.source_key:
            out.append(f"구간 {i + 1}에 소스가 없습니다.")
    return out


def check_sources(plan: HybridPlan, entries: dict, current_keys: dict[str, str] | None = None) -> dict[str, SegmentStatus]:
    """구간마다 소스 상태. entries: 키 → StoreEntry. current_keys: 모델 → 지금 설정(체크포인트 등)의 키.

    - 결과가 없으면(지워졌거나 원본이 바뀜) 검증 필요
    - 같은 키가 다시 만들어졌으면(결과 id가 다름) 검증 필요
    - 그 모델의 지금 설정 키와 다르면(체크포인트·설정이 바뀜) 검증 필요
    - 샘플레이트·길이가 계획과 다르면 검증 필요
    """
    out: dict[str, SegmentStatus] = {}
    current_keys = current_keys or {}
    for s in plan.segments:
        e = entries.get(s.source_key)
        if e is None:
            out[s.segment_id] = SegmentStatus(STATUS_CHECK, "분리 결과가 없습니다 (지웠거나 원본이 바뀜). 다른 모델을 고르세요.")
        elif e.model == HYBRID:
            out[s.segment_id] = SegmentStatus(STATUS_CHECK, "Hybrid 결과는 구간 소스로 쓸 수 없습니다.")
        elif e.manifest.sample_rate != plan.sample_rate or e.manifest.frames != plan.frames:
            out[s.segment_id] = SegmentStatus(STATUS_CHECK, "분리 결과의 샘플레이트·길이가 다릅니다.")
        elif s.source_stems_id and e.manifest.stems_id != s.source_stems_id:
            out[s.segment_id] = SegmentStatus(STATUS_CHECK, "고른 뒤 분리 결과가 다시 만들어졌습니다. 들어 보고 다시 확인하세요.")
        elif s.source_model in current_keys and current_keys[s.source_model] != s.source_key:
            out[s.segment_id] = SegmentStatus(STATUS_CHECK, "모델 파일(체크포인트·설정)이 바뀌었습니다. 이 구간은 이전 모델 결과입니다.")
        else:
            out[s.segment_id] = SegmentStatus(STATUS_OK)
    return out


# --- 편집 연산 (모두 새 계획을 돌려준다) ----------------------------------------------------

def _min_frames(plan: HybridPlan) -> int:
    return max(1, int(round(MIN_SEGMENT_S * plan.sample_rate)))


def _with(plan: HybridPlan, segs: list[HybridSegment]) -> HybridPlan:
    new = dataclasses.replace(plan, segments=tuple(segs))
    problems = validate(new)
    if problems:
        raise HybridError(" ".join(problems))
    return new


def split(plan: HybridPlan, segment_id: str, t: float) -> HybridPlan:
    i = plan.index(segment_id)
    s = plan.segments[i]
    f = plan.frame(t)
    m = _min_frames(plan)
    if f - s.start_frame < m or s.end_frame - f < m:
        raise HybridError(f"나눌 위치가 구간 끝에 너무 가깝습니다 (양쪽 {MIN_SEGMENT_S * 1000:.0f}ms 이상).")
    left = dataclasses.replace(s, end_frame=f)
    right = dataclasses.replace(s, segment_id=new_id(), start_frame=f, crossfade_s=plan.default_crossfade_s)
    segs = list(plan.segments)
    segs[i:i + 1] = [left, right]
    return _with(plan, segs)


def merge(plan: HybridPlan, segment_ids: list[str], keep: str | None = None) -> HybridPlan:
    """이어진 구간들을 하나로. 소스는 keep(구간 id)의 것, 없으면 맨 앞 구간의 것."""
    idx = sorted(plan.index(sid) for sid in set(segment_ids))
    if len(idx) < 2:
        raise HybridError("합칠 구간을 두 개 이상 고르세요.")
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise HybridError("이웃한 구간만 합칠 수 있습니다.")
    first, last = plan.segments[idx[0]], plan.segments[idx[-1]]
    if keep is not None and plan.index(keep) not in idx:
        raise HybridError("소스로 남길 구간이 합칠 구간 안에 없습니다.")
    src = plan.get(keep) if keep else first
    merged = dataclasses.replace(src, segment_id=first.segment_id, start_frame=first.start_frame,
                                 end_frame=last.end_frame, crossfade_s=first.crossfade_s)
    segs = list(plan.segments)
    segs[idx[0]:idx[-1] + 1] = [merged]
    return _with(plan, segs)


def move_boundary(plan: HybridPlan, index: int, t: float) -> HybridPlan:
    """index번 구간의 시작 경계(= index-1번 구간의 끝)를 옮긴다. 양쪽 구간이 최소 길이를 지키게 막는다."""
    if not (1 <= index < len(plan.segments)):
        raise HybridError("옮길 경계가 없습니다.")
    a, b = plan.segments[index - 1], plan.segments[index]
    m = _min_frames(plan)
    f = min(max(plan.frame(t), a.start_frame + m), b.end_frame - m)
    segs = list(plan.segments)
    segs[index - 1] = dataclasses.replace(a, end_frame=f)
    segs[index] = dataclasses.replace(b, start_frame=f)
    return _with(plan, segs)


def set_source(plan: HybridPlan, segment_ids: list[str], key: str, model: str, stems_id: str = "") -> HybridPlan:
    targets = set(segment_ids)
    for sid in targets:
        plan.index(sid)
    segs = [dataclasses.replace(s, source_key=key, source_model=model, source_stems_id=stems_id, user_selected=True)
            if s.segment_id in targets else s for s in plan.segments]
    return _with(plan, segs)


def set_all(plan: HybridPlan, key: str, model: str, stems_id: str = "") -> HybridPlan:
    """곡 전체를 한 모델로. 구간 경계는 지우지 않는다(나중에 일부만 다시 바꾸기 쉽게)."""
    return set_source(plan, [s.segment_id for s in plan.segments], key, model, stems_id)


def set_gain(plan: HybridPlan, segment_ids: list[str], gain_db: float) -> HybridPlan:
    targets = set(segment_ids)
    segs = [dataclasses.replace(s, gain_db=float(gain_db)) if s.segment_id in targets else s for s in plan.segments]
    return _with(plan, segs)


def set_crossfade(plan: HybridPlan, segment_ids: list[str] | None, seconds: float) -> HybridPlan:
    """구간 시작 경계의 교차 길이. segment_ids가 None이면 모든 경계와 새 경계의 기본값."""
    seconds = float(seconds)
    if segment_ids is None:
        segs = [dataclasses.replace(s, crossfade_s=seconds) for s in plan.segments]
        return _with(dataclasses.replace(plan, default_crossfade_s=seconds), segs)
    targets = set(segment_ids)
    segs = [dataclasses.replace(s, crossfade_s=seconds) if s.segment_id in targets else s for s in plan.segments]
    return _with(plan, segs)


def create(plan: HybridPlan, start: float, end: float, key: str, model: str, stems_id: str = "") -> tuple[HybridPlan, str]:
    """[start, end) 를 새 구간으로 만든다. 걸친 구간은 잘리고, 그 안에 완전히 든 구간은 새 구간이 대신한다."""
    a, b = plan.frame(min(start, end)), plan.frame(max(start, end))
    m = _min_frames(plan)
    if b - a < m:
        raise HybridError(f"구간이 너무 짧습니다 ({MIN_SEGMENT_S * 1000:.0f}ms 이상).")
    # 남는 조각이 최소 길이보다 짧으면 새 구간에 붙인다
    for s in plan.segments:
        if s.start_frame < a < s.end_frame and a - s.start_frame < m:
            a = s.start_frame
        if s.start_frame < b < s.end_frame and s.end_frame - b < m:
            b = s.end_frame
    new = HybridSegment(new_id(), a, b, key, model, stems_id, crossfade_s=plan.default_crossfade_s)
    segs: list[HybridSegment] = []
    for s in plan.segments:
        if s.end_frame <= a or s.start_frame >= b:
            segs.append(s)
            continue
        if s.start_frame < a:
            segs.append(dataclasses.replace(s, end_frame=a))
        if s.end_frame > b:
            segs.append(dataclasses.replace(s, segment_id=new_id() if s.start_frame < a else s.segment_id,
                                            start_frame=b, crossfade_s=plan.default_crossfade_s))
    segs.append(new)
    segs.sort(key=lambda s: s.start_frame)
    return _with(plan, segs), new.segment_id


def delete(plan: HybridPlan, segment_id: str) -> HybridPlan:
    """구간을 지우고 그 시간을 앞 구간(첫 구간이면 뒤 구간)이 맡는다."""
    if len(plan.segments) == 1:
        raise HybridError("구간이 하나뿐이라 지울 수 없습니다. 전체 모델을 바꾸려면 '전체 지정'을 쓰세요.")
    i = plan.index(segment_id)
    segs = list(plan.segments)
    s = segs.pop(i)
    if i > 0:
        segs[i - 1] = dataclasses.replace(segs[i - 1], end_frame=s.end_frame)
    else:
        segs[0] = dataclasses.replace(segs[0], start_frame=0, crossfade_s=s.crossfade_s)
    return _with(plan, segs)


def changed_ranges(old: HybridPlan | None, new: HybridPlan) -> list[tuple[float, float]]:
    """두 계획에서 소리가 달라질 수 있는 시간 범위 (교차 창 포함, 넉넉하게 잡는다). old가 없으면 곡 전체.

    구간 경계만으로 계산한다 (샘플 배열을 만들지 않음: 5분 곡에서 화면 스레드 90ms → 1ms 미만)."""
    if old is None or old.frames != new.frames or old.sample_rate != new.sample_rate \
            or old.level_match != new.level_match:
        return [(0.0, new.duration)]
    oa, na = _assignments(old), _assignments(new)
    spans: list[tuple[int, int]] = []
    edges = sorted({0, new.frames, *(x for a, b, _ in oa + na for x in (a, b))})
    for lo, hi in zip(edges, edges[1:]):
        if _source_at(oa, lo) != _source_at(na, lo):
            spans.append((lo, hi))
    for lo, hi, *_ in _fade_windows(old) ^ _fade_windows(new):  # 한쪽에만 있거나 양옆 소스가 바뀐 교차 창
        spans.append((max(0, lo), min(new.frames, hi)))
    spans.sort()
    merged: list[list[int]] = []
    for lo, hi in spans:
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    sr = new.sample_rate
    return [(a / sr, b / sr) for a, b in merged]


def _assignments(plan: HybridPlan) -> list[tuple[int, int, tuple]]:
    return [(s.start_frame, s.end_frame, (s.source_key, s.source_stems_id, round(s.gain_db, 4)))
            for s in plan.segments]


def _source_at(assign, f: int):
    for a, b, src in assign:
        if a <= f < b:
            return src
    return None


def _fade_windows(plan: HybridPlan) -> set[tuple]:
    halves = fade_halves(plan)
    segs = plan.segments
    assign = _assignments(plan)
    return {(s.start_frame - h, s.start_frame + h, assign[i - 1][2], assign[i][2])
            for i, (s, h) in enumerate(zip(segs, halves)) if i > 0 and h > 0}


# --- 합치기 -----------------------------------------------------------------------

def fade_halves(plan: HybridPlan) -> list[int]:
    """경계마다 교차 창의 반 길이(샘플). [0]은 첫 구간이라 항상 0. 양쪽 구간 길이의 절반을 넘지 않는다."""
    segs = plan.segments
    out = [0]
    for i in range(1, len(segs)):
        want = int(round(segs[i].crossfade_s * plan.sample_rate / 2))
        out.append(max(0, min(want, segs[i - 1].frames // 2, segs[i].frames // 2)))
    return out


def mix(plan: HybridPlan, sources: dict[str, np.ndarray], gains: dict[str, float] | None = None) -> np.ndarray:
    """(frames, ch) float32 소스들을 계획대로 합친다. gains: 소스 키 → 배율 (음량 맞춤). 구간 이득은 따로 곱한다."""
    gains = gains or {}
    segs = plan.segments
    first = sources[segs[0].source_key]
    out = np.empty((plan.frames, first.shape[1]), dtype=np.float32)
    halves = fade_halves(plan)

    def g(s: HybridSegment) -> float:
        return float(gains.get(s.source_key, 1.0)) * float(10 ** (s.gain_db / 20))

    for i, s in enumerate(segs):
        lo = s.start_frame + halves[i]
        hi = s.end_frame - (halves[i + 1] if i + 1 < len(segs) else 0)
        src = sources[s.source_key]
        k = g(s)
        out[lo:hi] = src[lo:hi] if k == 1.0 else (src[lo:hi].astype(np.float64) * k).astype(np.float32)
    for i in range(1, len(segs)):
        h = halves[i]
        if h == 0:
            continue
        a, b = segs[i - 1], segs[i]
        lo, hi = b.start_frame - h, b.start_frame + h
        w = ((np.arange(2 * h, dtype=np.float64) + 0.5) / (2 * h))[:, None]
        pa = sources[a.source_key][lo:hi].astype(np.float64) * g(a)
        pb = sources[b.source_key][lo:hi].astype(np.float64) * g(b)
        out[lo:hi] = (pa * (1.0 - w) + pb * w).astype(np.float32)
    return out


def identity(plan: HybridPlan, source_sha256: str) -> dict:
    return {
        "model": HYBRID, "source": source_sha256, "engine": HYBRID_VERSION, "level_match": plan.level_match,
        "segments": [[s.start_frame, s.end_frame, s.source_key, s.source_stems_id, round(s.gain_db, 4),
                      round(s.crossfade_s, 6)] for s in plan.segments],
    }


def plan_key(plan: HybridPlan, source_sha256: str) -> str:
    from amp.engine.model_store import store_key

    return store_key(identity(plan, source_sha256))


@dataclass
class RenderResult:
    entry: object  # StoreEntry
    reused: bool
    levels_db: dict = field(default_factory=dict)
    gains: dict = field(default_factory=dict)
    peak_scale_db: dict = field(default_factory=dict)
    seconds: float = 0.0


def render(plan: HybridPlan, store, progress: Callable[[float, str], None] | None = None,
           is_cancelled: Callable[[], bool] | None = None, current_keys: dict[str, str] | None = None,
           keep_renders: int = KEEP_RENDERS) -> RenderResult:
    """계획대로 보컬·반주를 합쳐 보관소 항목으로 만든다. 같은 계획이면 있는 항목을 그대로 쓴다.

    원본 stem 파일은 읽기만 한다. 지금 분석에 쓰는 결과(stems.json)는 바꾸지 않는다(사용자가 고를 때 activate).
    """
    import time

    import soundfile as sf

    from amp.audio.loudness import gated_level_db
    from amp.core.errors import AnalysisCancelled

    t0 = time.perf_counter()
    report = progress or (lambda f, m: None)
    problems = validate(plan)
    if problems:
        raise HybridError("Hybrid 구간이 올바르지 않습니다: " + " ".join(problems))
    entries = {e.key: e for e in store.entries()}
    status = check_sources(plan, entries, current_keys)
    bad = [(i, status[s.segment_id]) for i, s in enumerate(plan.segments) if not status[s.segment_id].ok]
    if bad:
        raise HybridError("검증이 필요한 구간이 있어 합치지 않았습니다: " +
                          " / ".join(f"구간 {i + 1}: {st.reason}" for i, st in bad))
    ident = identity(plan, store.source_sha256)
    key = plan_key(plan, store.source_sha256)
    if (hit := entries.get(key)) is not None:
        report(1.0, "같은 구성의 Hybrid 결과 사용")
        info = hit.manifest.model_info
        return RenderResult(hit, True, dict(info.get("levels_db") or {}), dict(info.get("level_gains") or {}),
                            dict(info.get("peak_scale_db") or {}), time.perf_counter() - t0)

    keys = list(dict.fromkeys(s.source_key for s in plan.segments))
    channels = {entries[k].manifest.channels for k in keys}
    if len(channels) != 1:
        raise HybridError("소스마다 채널 수가 달라 합칠 수 없습니다.")
    tmp = store.new_tmp()
    levels: dict[str, float | None] = {}
    gains: dict[str, float] = {}
    peak_scale: dict[str, float] = {}
    try:
        for n, stem in enumerate(("vocals", "instrumental")):
            if is_cancelled and is_cancelled():
                raise AnalysisCancelled("Hybrid 트랙 만들기를 취소했습니다.")
            report(0.05 + 0.45 * n, f"{'보컬' if stem == 'vocals' else '반주'} 읽는 중")
            data: dict[str, np.ndarray] = {}
            for k in keys:
                e = entries[k]
                x, sr = sf.read(str(e.path / e.manifest.files[stem]["path"]), dtype="float32", always_2d=True)
                if sr != plan.sample_rate or x.shape[0] != plan.frames:
                    raise HybridError(f"{e.label}의 {stem} 길이·샘플레이트가 계획과 다릅니다.")
                data[k] = x
            if stem == "vocals":
                levels = {k: gated_level_db(data[k], plan.sample_rate) for k in keys}
                known = [v for v in levels.values() if v is not None]
                target = min(known) if known else None
                gains = {k: (float(10 ** ((target - lv) / 20)) if plan.level_match and lv is not None and target is not None
                             else 1.0) for k, lv in levels.items()}
            out = mix(plan, data, gains)
            del data
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > CLIP_LIMIT:  # 이득을 올려 잘릴 소리: 전체를 같은 비율로 줄인다
                scale = PEAK_TARGET / peak
                out *= np.float32(scale)
                peak_scale[stem] = round(20 * float(np.log10(scale)), 2)
                log.info("Hybrid %s: 최대값 %.3f → 전체 %.2fdB", stem, peak, peak_scale[stem])
            sf.write(str(tmp / f"{stem}.wav"), out, plan.sample_rate, subtype="PCM_24")
            del out
        info = {
            "model": HYBRID, "sample_rate": plan.sample_rate, "frames": plan.frames, "engine": HYBRID_VERSION,
            "segments": [segment_dict(plan, s) for s in plan.segments],
            "sources": {k: {"model": entries[k].model, "label": entries[k].label,
                            "stems_id": entries[k].manifest.stems_id} for k in keys},
            "levels_db": {k: (None if v is None else round(v, 2)) for k, v in levels.items()},
            "level_match": plan.level_match, "level_gains": {k: round(v, 4) for k, v in gains.items()},
            "peak_scale_db": peak_scale, "crossfade": "linear (gains sum to 1)",
        }
        (tmp / "info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        report(0.97, "Hybrid 결과 저장 중")
        entry = store.commit(tmp, ident, {"hybrid": True})
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    prune(store, keep={entry.key}, keep_renders=keep_renders)
    report(1.0, "Hybrid 트랙을 만들었습니다")
    return RenderResult(entry, False, {k: v for k, v in levels.items()}, gains, peak_scale, time.perf_counter() - t0)


def prune(store, keep: set[str] = frozenset(), keep_renders: int = KEEP_RENDERS) -> list[str]:
    """오래된 Hybrid 결과를 지운다. 지금 쓰는 것·keep·최근 keep_renders개는 남긴다. 모델 분리 결과는 건드리지 않는다."""
    active = store.active_key()
    hybrids = [e for e in store.entries() if e.model == HYBRID and e.key != active and e.key not in keep]
    hybrids.sort(key=lambda e: e.manifest.created, reverse=True)
    removed = []
    for e in hybrids[keep_renders:]:
        try:
            store.delete(e.key)
            removed.append(e.key)
        except (OSError, ValueError) as ex:
            log.warning("이전 Hybrid 결과를 지우지 못함 %s: %s", e.key, ex)
    return removed


# --- 저장 형식 ----------------------------------------------------------------------

def segment_dict(plan: HybridPlan, s: HybridSegment, status: SegmentStatus | None = None) -> dict:
    a, b = plan.span(s)
    d = {
        "segment_id": s.segment_id, "start_time": round(a, 6), "end_time": round(b, 6),
        "start_frame": s.start_frame, "end_frame": s.end_frame,
        "source_model_id": s.source_model, "source_result_id": s.source_key, "source_stems_id": s.source_stems_id,
        "crossfade_duration": s.crossfade_s, "gain_adjustment_db": s.gain_db, "is_user_selected": s.user_selected,
    }
    if status is not None:
        d["status"] = status.state
        if status.reason:
            d["status_reason"] = status.reason
    return d


def plan_to_dict(plan: HybridPlan, status: dict[str, SegmentStatus] | None = None) -> dict:
    return {
        "sample_rate": plan.sample_rate, "frames": plan.frames, "duration": round(plan.duration, 6),
        "level_match": plan.level_match, "default_crossfade_s": plan.default_crossfade_s,
        "segments": [segment_dict(plan, s, (status or {}).get(s.segment_id)) for s in plan.segments],
    }


def plan_from_dict(d: dict) -> HybridPlan:
    """저장한 계획을 읽는다. 구간 조건을 어기면 HybridError (불러오는 쪽이 경고로 바꾼다)."""
    try:
        segs = tuple(HybridSegment(
            str(x["segment_id"]), int(x["start_frame"]), int(x["end_frame"]), str(x["source_result_id"]),
            str(x.get("source_model_id", "")), str(x.get("source_stems_id", "")),
            float(x.get("crossfade_duration", DEFAULT_CROSSFADE_S)), float(x.get("gain_adjustment_db", 0.0)),
            bool(x.get("is_user_selected", True))) for x in d["segments"])
        plan = HybridPlan(int(d["sample_rate"]), int(d["frames"]), segs, bool(d.get("level_match", False)),
                          float(d.get("default_crossfade_s", DEFAULT_CROSSFADE_S)))
    except (KeyError, TypeError, ValueError) as e:
        raise HybridError(f"Hybrid 구간 파일 형식이 올바르지 않습니다: {e}") from e
    problems = validate(plan)
    if problems:
        raise HybridError("Hybrid 구간 파일에 문제가 있습니다: " + " ".join(problems))
    return plan


def summary(plan: HybridPlan, labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    parts = []
    for s in plan.segments:
        a, b = plan.span(s)
        parts.append(f"{a:.2f}–{b:.2f}s {labels.get(s.source_key, s.source_model)}")
    return ", ".join(parts)
