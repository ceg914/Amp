"""음절별 가창 길이와 멜리스마 분석 (STEP 4). Qt 비의존 순수 함수.

입력은 이미 있는 데이터뿐이다: 음절 시간(정렬), 음절↔음표 연결(역할 포함), 음표, 피치 곡선, 재발음 지표.
Whisper·Demucs·정렬 모델을 다시 돌리지 않는다. 음표·피치·가사 데이터를 바꾸지 않는다(읽기만 한다).

길이 두 가지
    음절 구간 길이  정렬로 얻은 시작~끝의 차이. 끝 시각은 대부분 추정값(다음 음절 시작·보컬 끊김)이라 '추정'으로 표시한다.
    유성 가창 길이  그 구간 안에서 피치(유성음)가 잡힌 프레임 시간의 합. 무성 자음(ㅅ·ㅎ 등)은 빠지고, 보컬 분리에 남은
                    잔향은 들어갈 수 있다. 믿을 만한 유성 프레임이 없으면 0으로 채우지 않고 '미측정'으로 둔다.

멜리스마 판정 (음절 하나에 이어진 음표 묶음)
    종류는 하나만 고른다(single_note / melisma / repeated_note / pitch_slide / uncertain). 근거는 따로 속성(flags)으로 붙인다.
    접근음 역할 연결은 음표 수에 넣지 않는다. 장식음(ornament)만 덧붙은 음은 단일 음으로 본다.
    사용자 판정(Alignment.melisma)이 있으면 그 판정이 최종이며, 자동 판정은 참고로 함께 보여 준다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

import numpy as np

from amp.core.model import Note, PitchCurve
from amp.core.pitch import note_name
from amp.lyrics.model import LyricsState, MelismaDecision, NoteLink

DETECTOR = "sonatlas-melisma"
DETECTOR_VERSION = "1"

VOICED_CONF = 0.05  # pYIN 유성 확률 하한 ('단순' 민감도와 같은 값). 유성 판정 자체는 pYIN이 f0를 NaN으로 둔 것으로 한다
MIN_SPAN_FRAMES = 2
REARTIC_ONSET = 2.0  # 재발음 지표가 이 이상이면 다시 발음했다고 본다 (STEP 3.5 접근음 판정과 같은 경계)
REARTIC_GAP_S = 0.04  # 두 음표 사이가 이만큼 비었거나 무성이면 다시 발음한 것으로 본다
SLIDE_NOTE_MAX_S = 0.15  # 이보다 짧은 음표가 한 방향으로 이어지면 미끄러지는 소리(포르타멘토)의 조각으로 본다
GAP_FLAG_S = 0.25
LOW_CONF = 0.3
VIBRATO_SPLIT_MAX_S = 0.25
EDGE_TOL_S = 0.05
UNCOVERED_MIN_S = 0.15

KINDS = ("single_note", "melisma", "repeated_note", "pitch_slide", "uncertain")
KIND_LABELS = {"single_note": "단일 음", "melisma": "멜리스마", "repeated_note": "같은 음 반복", "pitch_slide": "음높이 미끄러짐",
               "uncertain": "판정 불확실"}
DIRECTION_LABELS = {"ascending": "상승", "descending": "하강", "mixed": "혼합", "stationary": "같은 높이"}
FLAG_LABELS = {
    "approach": "접근음(끌어올림)은 음표 수에서 뺌", "ornament": "장식음 포함", "shared_note": "다른 음절과 같은 음표를 나눠 씀",
    "new_syllable_inside": "음표 묶음 안에서 다른 음절이 시작됨", "gap": "음표 사이가 0.25초 넘게 비어 있음",
    "low_confidence": "신뢰도 낮은 음표가 있음", "outside_span": "음절 시간 구간 밖의 음표가 있음",
    "possible_oversplit": "같은 높이인데 다시 발음한 흔적이 없음(음표가 잘못 나뉘었을 수 있음)",
    "vibrato_split": "비브라토가 음표 여러 개로 나뉜 것으로 보임", "rearticulated": "음높이가 바뀌는 곳에서 다시 발음함(다른 음절일 수 있음)",
    "uncovered_voicing": "음표가 없는 유성 구간이 있음(음표가 빠졌을 수 있음)", "no_reartic_info": "재발음 지표가 없어 반복 여부를 판단하지 못함",
    "missing_note": "연결된 음표 중 지금 없는 음표가 있음", "user_review": "음표가 바뀌어 사용자 판정을 다시 확인해야 함",
}
SPAN_STATUS_LABELS = {"measured": "검출", "estimated": "추정", "stale": "추정(가사 바뀜)", "unaligned": "미정렬",
                      "invalid": "잘못된 시간"}


# --- 음절 길이 ---------------------------------------------------------------------

@dataclass(frozen=True)
class SyllableDuration:
    syllable_id: str
    start: float | None
    end: float | None
    span: float | None  # 음절 구간 길이 (A)
    span_status: str  # measured / estimated / stale / unaligned / invalid
    voiced: float | None  # 유성 가창 길이 (B). None = 미측정
    voiced_reason: str  # 미측정 이유 또는 측정 범위 설명
    note_union: float | None  # 연결 음표 구간의 합집합 길이 (겹친 부분은 한 번만)
    linked_note_ids: tuple[str, ...] = ()


def _frames(curve: PitchCurve, t0: float, t1: float) -> tuple[int, int]:
    fr = curve.frame_rate
    a = int(np.ceil((t0 - curve.start_time) * fr - 1e-9))
    b = int(np.ceil((t1 - curve.start_time) * fr - 1e-9))
    n = len(curve)
    return max(0, min(n, a)), max(0, min(n, b))


def voiced_mask(curve: PitchCurve, no_vocal_ranges: Iterable[tuple[float, float]] = ()) -> np.ndarray:
    m = np.isfinite(curve.f0_hz) & (curve.confidence >= VOICED_CONF)
    for a, b in no_vocal_ranges:
        i, j = _frames(curve, a, b)
        m[i:j] = False
    return m


def union_length(intervals: Iterable[tuple[float, float]]) -> float:
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def syllable_duration(state: LyricsState, sid: str, notes_by_id: Mapping[str, Note], curve: PitchCurve | None,
                      voiced: np.ndarray | None = None) -> SyllableDuration:
    t = state.alignment.timing(sid)
    linked = tuple(n for n in state.alignment.linked_notes(sid) if n in notes_by_id)
    nu = union_length((notes_by_id[n].start, notes_by_id[n].end) for n in linked) if linked else None
    if not t.aligned:
        return SyllableDuration(sid, None, None, None, "unaligned", None, "음절 시간이 없어(미정렬) 재지 않았습니다.", nu, linked)
    if not t.end > t.start:
        return SyllableDuration(sid, t.start, t.end, None, "invalid", None, "끝이 시작보다 앞이라 재지 않았습니다.", nu, linked)
    span = t.end - t.start
    if t.stale:
        status = "stale"
    elif t.user_edited or (t.detected and t.end_source in ("model", "user", "notes", "legacy")):
        status = "measured"
    else:
        status = "estimated"
    if curve is None or len(curve) == 0:
        return SyllableDuration(sid, t.start, t.end, span, status, None, "피치 분석 결과가 없어 재지 않았습니다.", nu, linked)
    i, j = _frames(curve, t.start, t.end)
    if j - i < MIN_SPAN_FRAMES:
        reason = "피치 곡선 범위 밖입니다." if j <= i else "구간이 피치 프레임 2개보다 짧습니다."
        return SyllableDuration(sid, t.start, t.end, span, status, None, reason, nu, linked)
    mask = voiced if voiced is not None else voiced_mask(curve)
    n_voiced = int(mask[i:j].sum())
    if n_voiced == 0:
        return SyllableDuration(sid, t.start, t.end, span, status, None,
                                "이 구간에서 믿을 만한 유성음을 찾지 못했습니다(0으로 채우지 않음).", nu, linked)
    v = n_voiced / curve.frame_rate
    return SyllableDuration(sid, t.start, t.end, span, status, v,
                            f"피치 프레임 {j - i}개 중 유성 {n_voiced}개 (프레임 {1000 / curve.frame_rate:.0f}ms 단위)", nu, linked)


def all_durations(state: LyricsState, notes_by_id: Mapping[str, Note], curve: PitchCurve | None,
                  no_vocal_ranges: Iterable[tuple[float, float]] = ()) -> dict[str, SyllableDuration]:
    mask = voiced_mask(curve, no_vocal_ranges) if curve is not None and len(curve) else None
    return {s.id: syllable_duration(state, s.id, notes_by_id, curve, mask) for s in state.text.syllables}


# --- 멜리스마 -----------------------------------------------------------------------

@dataclass(frozen=True)
class VocalGroup:
    """음절 하나의 음표 묶음 판정."""

    syllable_id: str
    kind: str  # 자동 판정 (KINDS)
    is_melisma: bool  # 최종 (사용자 판정이 있으면 그것)
    source: str  # auto / user
    confirmed: bool  # 사용자 판정
    needs_review: bool
    note_ids: tuple[str, ...]  # 판정에 쓴 음표 (접근음·장식음 제외, 시간 순)
    approach_ids: tuple[str, ...] = ()
    ornament_ids: tuple[str, ...] = ()
    start: float | None = None
    end: float | None = None
    pitch_sequence: tuple[str, ...] = ()
    pitches: tuple[int, ...] = ()
    pitch_range: int = 0  # 반음
    direction: str = "stationary"
    confidence: float = 0.0  # 규칙 점수 0~1 (확률이 아님)
    flags: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    user_kind: str = ""  # 사용자 판정 종류 (melisma / not_melisma)

    @property
    def duration(self) -> float | None:
        return None if self.start is None or self.end is None else self.end - self.start

    @property
    def note_count(self) -> int:
        return len(self.note_ids)

    @property
    def melisma_id(self) -> str:
        return f"mel-{self.syllable_id}"

    def melisma_record(self) -> dict | None:
        """명세 5.3의 멜리스마 기록. 멜리스마가 아니면 None."""
        if not self.is_melisma:
            return None
        return {
            "melisma_id": self.melisma_id, "syllable_id": self.syllable_id, "linked_note_ids": list(self.note_ids),
            "start_time": self.start, "end_time": self.end, "duration": self.duration, "note_count": self.note_count,
            "pitch_sequence": list(self.pitch_sequence), "pitch_range": self.pitch_range,
            "pitch_direction": self.direction, "confidence": round(self.confidence, 3),
            "detection_source": "user" if self.source == "user" else f"{DETECTOR}/{DETECTOR_VERSION}",
            "is_user_confirmed": self.confirmed,
        }


@dataclass
class _Context:
    state: LyricsState
    notes: Mapping[str, Note]
    curve: PitchCurve | None
    onset: np.ndarray | None
    voiced: np.ndarray | None
    starts: list[tuple[float, str]] = field(default_factory=list)  # (음절 시작, 음절 id) 정렬
    all_notes: list[Note] = field(default_factory=list)


def direction_of(pitches: list[int]) -> str:
    steps = [b - a for a, b in zip(pitches, pitches[1:]) if b != a]
    if not steps:
        return "stationary"
    if all(s > 0 for s in steps):
        return "ascending"
    if all(s < 0 for s in steps):
        return "descending"
    return "mixed"


def _idx(curve: PitchCurve, t: float) -> int:
    return int(round((t - curve.start_time) * curve.frame_rate))


def rearticulated(a: Note, b: Note, ctx: _Context) -> bool | None:
    """a 뒤에 b를 다시 발음했는가. 근거가 없으면 None."""
    if b.start - a.end >= REARTIC_GAP_S:
        return True
    c = ctx.curve
    if c is None or ctx.voiced is None:
        return None
    i = _idx(c, b.start)
    lo, hi = max(0, _idx(c, min(a.end, b.start)) - 2), min(len(c), _idx(c, max(a.end, b.start)) + 3)
    if hi > lo and (~ctx.voiced[lo:hi]).sum() >= int(round(REARTIC_GAP_S * c.frame_rate)):
        return True
    if ctx.onset is None or len(ctx.onset) != len(c):
        return None
    lo, hi = max(0, i - 3), min(len(c), i + 4)
    return bool(hi > lo and float(np.nanmax(ctx.onset[lo:hi])) >= REARTIC_ONSET)


def _continuous(first: Note, last: Note, ctx: _Context) -> bool:
    c = ctx.curve
    if c is None or ctx.voiced is None:
        return False
    i, j = _frames(c, first.start, last.start + 0.03)
    return j > i and ctx.voiced[i:j].mean() >= 0.9


def _uncovered_voicing(sid: str, ctx: _Context) -> bool:
    """음절 구간에서 어느 음표에도 덮이지 않은 유성 시간이 0.15초 넘는가."""
    t = ctx.state.alignment.timing(sid)
    c = ctx.curve
    if c is None or ctx.voiced is None or not t.aligned or not t.end > t.start:
        return False
    i, j = _frames(c, t.start, t.end)
    if j <= i:
        return False
    covered = np.zeros(j - i, dtype=bool)
    k = bisect.bisect_left([n.end for n in ctx.all_notes], t.start)
    for n in ctx.all_notes[k:]:
        if n.start >= t.end:
            break
        a, b = _frames(c, n.start, n.end)
        covered[max(0, a - i):max(0, b - i)] = True
    free = ctx.voiced[i:j] & ~covered
    return free.sum() / c.frame_rate >= UNCOVERED_MIN_S


def _others_start_inside(sid: str, a: float, b: float, ctx: _Context) -> bool:
    lo = bisect.bisect_right(ctx.starts, (a + EDGE_TOL_S, chr(0x10FFFF)))
    hi = bisect.bisect_left(ctx.starts, (b - EDGE_TOL_S, ""))
    return any(s != sid for _t, s in ctx.starts[lo:hi])


def _classify(sid: str, ctx: _Context) -> VocalGroup | None:
    st = ctx.state
    al = st.alignment
    links: tuple[NoteLink, ...] = al.links.get(sid, ())
    present = sorted(((lk, ctx.notes[lk.note_id]) for lk in links if lk.note_id in ctx.notes),
                     key=lambda x: (x[1].start, x[1].id))
    flags: list[str] = []
    if len(present) < len(links):
        flags.append("missing_note")
    if not present:
        return None
    approach = [n for lk, n in present if lk.role == "approach"]
    main = [n for lk, n in present if lk.role != "approach"]
    if approach:
        flags.append("approach")
    ornaments = [n for n in main if n.ornament]
    core = [n for n in main if not n.ornament] or main
    if ornaments and len(core) < len(main):
        flags.append("ornament")
    if any(len(al.note_to_syllables.get(n.id, ())) > 1 for n in core):
        flags.append("shared_note")
    t = al.timing(sid)
    if t.aligned and t.end > t.start:
        tol = 0.15
        if any(n.end < t.start - tol or n.start > t.end + tol for n in core):
            flags.append("outside_span")
    if any(n.confidence is not None and n.confidence < LOW_CONF for n in core):
        flags.append("low_confidence")
    if _uncovered_voicing(sid, ctx):
        flags.append("uncovered_voicing")
    pitches = [n.pitch for n in core]
    kind, conf = "single_note", 0.9
    if len(core) >= 2:
        if _others_start_inside(sid, core[0].start, core[-1].end, ctx):
            flags.append("new_syllable_inside")
        if any(b.start - a.end > GAP_FLAG_S for a, b in zip(core, core[1:])):
            flags.append("gap")
        steps = [b - a for a, b in zip(pitches, pitches[1:])]
        reart = [rearticulated(a, b, ctx) for a, b in zip(core, core[1:])]
        if all(s == 0 for s in steps):
            if all(r is True for r in reart):
                kind, conf = "repeated_note", 0.8
            elif any(r is None for r in reart):
                kind, conf = "uncertain", 0.3
                flags.append("no_reartic_info")
            else:
                kind, conf = "uncertain", 0.35
                flags.append("possible_oversplit")
        elif (len(core) >= 3 and all(abs(s) <= 1 for s in steps) and all(s != 0 for s in steps)
              and all(x * y < 0 for x, y in zip(steps, steps[1:])) and all(n.duration < VIBRATO_SPLIT_MAX_S for n in core)):
            kind, conf = "uncertain", 0.3
            flags.append("vibrato_split")
        elif (all(n.duration < SLIDE_NOTE_MAX_S for n in core[:-1]) and direction_of(pitches) in ("ascending", "descending")
              and not any(r is True for r in reart) and _continuous(core[0], core[-1], ctx)):
            kind, conf = "pitch_slide", 0.6
        else:
            kind, conf = "melisma", 0.85
            if any(r is True and s != 0 for r, s in zip(reart, steps)):
                flags.append("rearticulated")
            doubtful = {"new_syllable_inside", "shared_note", "low_confidence", "outside_span", "gap", "rearticulated"}
            if doubtful.intersection(flags):
                kind, conf = "uncertain", 0.45
    elif "uncovered_voicing" in flags:
        conf = 0.6
    if "low_confidence" in flags:
        conf *= 0.8
    if "missing_note" in flags:
        conf *= 0.8
    if kind == "single_note" and "shared_note" in flags:
        conf = min(conf, 0.7)
    return VocalGroup(
        sid, kind, kind == "melisma", "auto", False, False, tuple(n.id for n in core), tuple(n.id for n in approach),
        tuple(n.id for n in ornaments if n not in core), core[0].start, core[-1].end, tuple(note_name(p) for p in pitches),
        tuple(pitches), max(pitches) - min(pitches), direction_of(pitches), round(conf, 3), tuple(dict.fromkeys(flags)),
        tuple(FLAG_LABELS[f] for f in dict.fromkeys(flags)))


def _apply_user(g: VocalGroup | None, sid: str, d: MelismaDecision, ctx: _Context) -> VocalGroup | None:
    if d.kind == "not_melisma":
        if g is None:
            return None
        return replace(g, is_melisma=False, source="user", confirmed=True, needs_review=d.needs_review, user_kind=d.kind,
                       confidence=1.0)
    notes = sorted((ctx.notes[n] for n in d.note_ids if n in ctx.notes), key=lambda n: (n.start, n.id))
    flags = list(g.flags) if g is not None else []
    review = d.needs_review
    if len(notes) < len(d.note_ids):
        flags.append("missing_note")
        review = True
    if review:
        flags.append("user_review")
    if not notes:
        return replace(g, is_melisma=True, source="user", confirmed=True, needs_review=True, user_kind=d.kind,
                       flags=tuple(dict.fromkeys(flags)), reasons=tuple(FLAG_LABELS[f] for f in dict.fromkeys(flags))) \
            if g is not None else None
    pitches = [n.pitch for n in notes]
    base = g or VocalGroup(sid, "uncertain", True, "user", True, review, ())
    return replace(base, is_melisma=True, source="user", confirmed=True, needs_review=review, user_kind=d.kind,
                   note_ids=tuple(n.id for n in notes), start=notes[0].start, end=notes[-1].end,
                   pitch_sequence=tuple(note_name(p) for p in pitches), pitches=tuple(pitches),
                   pitch_range=max(pitches) - min(pitches), direction=direction_of(pitches), confidence=1.0,
                   flags=tuple(dict.fromkeys(flags)), reasons=tuple(FLAG_LABELS[f] for f in dict.fromkeys(flags)))


def analyze(state: LyricsState, notes: Iterable[Note], curve: PitchCurve | None = None, onset: np.ndarray | None = None,
            no_vocal_ranges: Iterable[tuple[float, float]] = (), ids: Iterable[str] | None = None) -> dict[str, VocalGroup]:
    """음절마다 음표 묶음 판정. 연결 음표가 없는 음절은 결과에 없다."""
    all_notes = sorted(notes, key=lambda n: (n.start, n.id))
    by_id = {n.id: n for n in all_notes}
    voiced = voiced_mask(curve, no_vocal_ranges) if curve is not None and len(curve) else None
    al = state.alignment
    starts = sorted((t.start, sid) for sid, t in al.timings.items() if t.aligned and sid in state.text.index)
    ctx = _Context(state, by_id, curve, None if onset is None else np.asarray(onset, dtype=np.float64), voiced, starts,
                   all_notes)
    out: dict[str, VocalGroup] = {}
    chosen = state.text.order if ids is None else [i for i in ids if i in state.text.index]
    for sid in chosen:
        g = _classify(sid, ctx)
        d = al.melisma.get(sid)
        if d is not None:
            g = _apply_user(g, sid, d, ctx)
        if g is not None:
            out[sid] = g
    return out


def melisma_records(groups: Mapping[str, VocalGroup]) -> list[dict]:
    return [r for g in groups.values() if (r := g.melisma_record()) is not None]


# --- 사용자 판정 (모두 새 LyricsState를 돌려준다) ------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_melisma(state: LyricsState, sid: str, note_ids: Iterable[str], notes_by_id: Mapping[str, Note]) -> LyricsState:
    """음절을 멜리스마로 확정한다. 연결되지 않은 음표는 확정 연결로 함께 잇는다."""
    ids = [n for n in dict.fromkeys(note_ids) if n in notes_by_id]
    if len(ids) < 2:
        raise ValueError("멜리스마는 음표가 2개 이상이어야 합니다.")
    ids.sort(key=lambda n: (notes_by_id[n].start, n))
    al = state.alignment
    links = list(al.links.get(sid, ()))
    have = {lk.note_id for lk in links}
    links += [NoteLink(n, None, "user", True) for n in ids if n not in have]
    links.sort(key=lambda lk: (notes_by_id[lk.note_id].start if lk.note_id in notes_by_id else 0.0, lk.note_id))
    d = MelismaDecision("melisma", tuple(ids), False, "", DETECTOR_VERSION, _now())
    out = replace(state, alignment=al.with_links({sid: links}).with_melisma({sid: d}))
    return out.with_history("멜리스마 지정", f"음표 {len(ids)}개")


def set_not_melisma(state: LyricsState, sid: str) -> LyricsState:
    d = MelismaDecision("not_melisma", (), False, "", DETECTOR_VERSION, _now())
    return replace(state, alignment=state.alignment.with_melisma({sid: d})).with_history("멜리스마 해제", "")


def clear_decision(state: LyricsState, ids: Iterable[str]) -> LyricsState:
    ids = [i for i in ids if i in state.alignment.melisma]
    if not ids:
        return state
    out = replace(state, alignment=state.alignment.with_melisma({i: None for i in ids}))
    return out.with_history("멜리스마 자동 판정으로 되돌림", f"{len(ids)}개")


def mark_reviewed(state: LyricsState, ids: Iterable[str]) -> LyricsState:
    al = state.alignment
    upd = {i: replace(al.melisma[i], needs_review=False, reason="") for i in ids if i in al.melisma and al.melisma[i].needs_review}
    if not upd:
        return state
    return replace(state, alignment=al.with_melisma(upd)).with_history("멜리스마 판정 확인", f"{len(upd)}개")


def add_notes(state: LyricsState, sid: str, note_ids: Iterable[str], notes_by_id: Mapping[str, Note],
              current: VocalGroup | None) -> LyricsState:
    base = list(current.note_ids) if current is not None else []
    d = state.alignment.melisma.get(sid)
    if d is not None and d.kind == "melisma":
        base = list(d.note_ids)
    return set_melisma(state, sid, [*base, *note_ids], notes_by_id)


def remove_notes(state: LyricsState, sid: str, note_ids: Iterable[str], notes_by_id: Mapping[str, Note],
                 current: VocalGroup | None) -> LyricsState:
    """멜리스마에서 음표를 뺀다. 연결은 그대로 둔다(연결 해제는 따로). 2개 미만이 남으면 '멜리스마 아님'으로 확정."""
    drop = set(note_ids)
    d = state.alignment.melisma.get(sid)
    base = list(d.note_ids) if d is not None and d.kind == "melisma" else list(current.note_ids if current else ())
    keep = [n for n in base if n not in drop]
    if len(keep) < 2:
        return set_not_melisma(state, sid)
    return set_melisma(state, sid, keep, notes_by_id)


def remap_decisions(state: LyricsState, mapping: Mapping[str, list[str]], gone: set[str]) -> LyricsState:
    """음표가 바뀐 뒤 사용자 멜리스마 판정의 음표 id를 옮긴다. mapping: 사라진 음표 → 새 음표 후보(시간 순).

    후보가 없거나 하나의 음표가 여러 개로 나뉘었으면 판정은 남기고 '재확인 필요'를 켠다.
    """
    al = state.alignment
    upd: dict[str, MelismaDecision] = {}
    for sid, d in al.melisma.items():
        if not any(n in gone for n in d.note_ids):
            continue
        new: list[str] = []
        review, why = d.needs_review, d.reason
        for n in d.note_ids:
            if n not in gone:
                new.append(n)
                continue
            cands = mapping.get(n) or []
            if len(cands) != 1:
                review = True
                why = "음표가 지워졌거나 나뉘어 멜리스마 음표를 다시 확인해야 합니다."
            new.extend(c for c in cands if c not in new)
        upd[sid] = replace(d, note_ids=tuple(new), needs_review=review, reason=why)
    if not upd:
        return state
    return replace(state, alignment=al.with_melisma(upd))
