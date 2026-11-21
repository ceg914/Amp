"""AI 보조 교정 제안 (v0.6). Qt 비의존.

음표를 직접 바꾸지 않는다. 근거(피치 곡선·재발음 지표·유성 구간·가사 음절)를 보고 "이렇게 고치면 어떨까"를 만든다.
사용자가 [적용]해야만 음표가 바뀌고(실행 취소 가능), 적용한 음표는 사용자 수정 음표가 되어 재분석이 덮어쓰지 않는다.

- 종류: split(나누기) / merge(합치기) / pitch(음높이) / boundary(시작·끝) / missing(빠진 음) / approach(떨어져 나온 접근음)
- 사용자가 고치거나 추가한 음표는 대상으로 삼지 않는다.
- ``score`` 는 근거의 세기를 0~1로 나타낸 규칙 점수다. 확률이 아니다 (보정된 적이 없다).
- 참조 악보는 음높이 제안의 보조 근거로만 쓴다. 악보만 보고 음높이를 바꾸라고 하지 않는다 (가수가 다르게 부를 수 있다).
- 제안을 만든 뒤 대상 음표가 바뀌면(편집·재분석) 그 제안은 outdated가 되고 적용할 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace

import numpy as np

from amp.core.errors import EditError
from amp.core.pitch import note_name as midi_to_name

DETECTOR_VERSION = "1"
TYPES = {"split": "나누기", "merge": "합치기", "pitch": "음높이", "boundary": "시작·끝", "missing": "빠진 음",
         "approach": "접근음 합치기"}
PENDING, ACCEPTED, REJECTED, DISMISSED, OUTDATED = "pending", "accepted", "rejected", "dismissed", "outdated"
STATUS_LABELS = {PENDING: "대기", ACCEPTED: "적용함", REJECTED: "무시함", DISMISSED: "숨김", OUTDATED: "지난 제안"}
SCORE_NOTE = "근거 점수: 근거의 세기를 0~1로 나타낸 규칙 값입니다. 확률이 아닙니다."

# 규칙 값 (원리에서 정함, 실제 곡으로 맞추지 않음)
PITCH_DEV = 0.6  # 반음. 음표 높이와 피치 곡선 중앙값이 이만큼 이상 다르면
PITCH_SPREAD = 0.6  # 반음. 음 가운데의 표준편차가 이보다 크면 음높이 제안을 하지 않는다 (비브라토 ±50c는 표준편차 약 0.35)
MIN_PART_S = 0.12  # 나누기 제안의 각 조각 최소 길이
SPLIT_PITCH_DEV = 0.8
SPLIT_ART_RATIO = 0.6  # 재발음 지표가 자동 분할 기준의 60% 이상이면 나누기 후보
MERGE_GAP_S = 0.03
MERGE_ART_MAX = 1.0
SYL_TOL_S = 0.08
BOUNDARY_MIN_S, BOUNDARY_MAX_S = 0.04, 0.12
TAIL_UNVOICED_S = 0.06
MISSING_MIN_S = 0.15
MISSING_CONF = 0.6
MISSING_STD = 0.5
APPROACH_MAX_S = 0.25


@dataclass(frozen=True)
class Suggestion:
    suggestion_id: str
    suggestion_type: str
    target_note_ids: tuple[str, ...]
    start_time: float
    end_time: float
    proposed_changes: dict
    evidence: tuple[str, ...]
    score: float  # 0~1 근거 점수 (확률 아님)
    fingerprint: str  # 만들 때 대상 음표 상태
    detector_version: str = DETECTOR_VERSION
    status: str = PENDING

    @property
    def type_label(self) -> str:
        return TYPES.get(self.suggestion_type, self.suggestion_type)

    def summary(self) -> str:
        c = self.proposed_changes
        t = self.suggestion_type
        if t == "split":
            return f"{c['at']:.2f}초에서 나누기" + (f" → {_names(c.get('pitches'))}" if c.get("pitches") else "")
        if t == "merge":
            return f"음표 {len(self.target_note_ids)}개 합치기"
        if t == "pitch":
            return f"{midi_to_name(c['from'])} → {midi_to_name(c['pitch'])}"
        if t == "boundary":
            parts = []
            if "start" in c:
                parts.append(f"시작 {c['start']:.2f}초")
            if "end" in c:
                parts.append(f"끝 {c['end']:.2f}초")
            return ", ".join(parts)
        if t == "missing":
            return f"{c['start']:.2f}–{c['end']:.2f}초 {midi_to_name(c['pitch'])} 추가"
        if t == "approach":
            return f"짧은 음을 뒤 음({midi_to_name(c['pitch'])})에 합치기"
        return t


def _names(pitches) -> str:
    return " / ".join(midi_to_name(int(p)) for p in pitches or ())


@dataclass(frozen=True)
class SuggestionState:
    items: tuple[Suggestion, ...] = ()
    created: str = ""
    detector_version: str = DETECTOR_VERSION

    def get(self, sid: str) -> Suggestion | None:
        return next((s for s in self.items if s.suggestion_id == sid), None)

    def with_status(self, ids, status: str) -> "SuggestionState":
        ids = set(ids)
        return replace(self, items=tuple(replace(s, status=status) if s.suggestion_id in ids else s for s in self.items))

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.items:
            out[s.status] = out.get(s.status, 0) + 1
        return out


# --- 근거 ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    fr: float
    start: float
    midi: np.ndarray
    conf: np.ndarray
    art: np.ndarray | None
    split_strength: float
    syllables: list[float] = field(default_factory=list)  # 정렬된 음절 시작
    reference: list = field(default_factory=list)  # (start, end, pitch)
    blocked: list = field(default_factory=list)

    def idx(self, t: float) -> int:
        return int(round((t - self.start) * self.fr))

    def seg(self, a: float, b: float) -> np.ndarray:
        i, j = max(0, self.idx(a)), min(len(self.midi), self.idx(b))
        return self.midi[i:j] if j > i else np.zeros(0)

    def art_max(self, a: float, b: float) -> tuple[float, float]:
        if self.art is None:
            return 0.0, a
        i, j = max(0, self.idx(a)), min(len(self.art), self.idx(b) + 1)
        if j <= i:
            return 0.0, a
        k = i + int(np.argmax(self.art[i:j]))
        return float(self.art[k]), self.start + k / self.fr

    def syllable_near(self, t: float, tol: float = SYL_TOL_S) -> bool:
        return any(abs(s - t) <= tol for s in self.syllables)


def _context(project, split_strength: float | None) -> _Ctx | None:
    a = project.analysis
    if a is None or a.pitch_curve is None or len(a.pitch_curve) == 0:
        return None
    c = a.pitch_curve
    midi = np.asarray(c.midi(), dtype=np.float64)
    conf = np.asarray(c.confidence, dtype=np.float64)
    art = None
    if a.onset_envelope is not None and len(a.onset_envelope) == len(midi):
        art = np.asarray(a.onset_envelope, dtype=np.float64)
    al = project.lyrics.alignment
    syl = sorted(t.start for t in al.timings.values() if t.aligned and not t.stale)
    ref = []
    if project.reference is not None and getattr(project.reference, "alignment", None) is not None:
        try:
            for n, a0, b0 in project.reference.aligned_notes():
                ref.append((a0, b0, n.pitch))
        except Exception:  # noqa: BLE001 - 정렬이 없거나 깨졌으면 보조 근거 없이
            ref = []
    blocked = list((a.settings or {}).get("no_vocal_ranges") or [])
    ss = split_strength if split_strength is not None else 4.0
    return _Ctx(c.frame_rate, c.start_time, midi, conf, art, ss, syl, ref, blocked)


def fingerprint(notes) -> str:
    text = json.dumps(sorted((n.id, round(n.start, 4), round(n.end, 4), n.pitch) for n in notes))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _make(kind: str, targets, a: float, b: float, changes: dict, evidence: list[str], score: float,
          fp_notes=None) -> Suggestion:
    ids = tuple(n.id for n in targets)
    key = json.dumps([kind, sorted(ids), round(a, 2), round(b, 2), changes], sort_keys=True, default=str)
    sid = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Suggestion(sid, kind, ids, round(a, 4), round(b, 4), changes, tuple(evidence),
                      round(float(min(1.0, max(0.0, score))), 3), fingerprint(fp_notes if fp_notes is not None else targets))


# --- 검출기 ---------------------------------------------------------------------------

def detect(project, split_strength: float | None = 4.0) -> list[Suggestion]:
    """프로젝트의 작업본 음표에 대한 제안 목록. 분석 결과(피치 곡선)가 없으면 빈 목록."""
    ctx = _context(project, split_strength)
    if ctx is None:
        return []
    notes = sorted(project.score.notes(), key=lambda n: (n.start, n.pitch))
    auto = [n for n in notes if project.note_status(n) == "original"]
    out: list[Suggestion] = []
    out += _pitch(ctx, auto)
    out += _split(ctx, auto)
    out += _merge(ctx, notes, project)
    out += _approach(ctx, notes, project)
    out += _boundary(ctx, notes, project)
    out += _missing(ctx, notes)
    # 같은 음표를 두 제안이 다르게 바꾸면 둘 다 보여 주되, 같은 id는 하나만
    seen, uniq = set(), []
    for s in out:
        if s.suggestion_id not in seen:
            seen.add(s.suggestion_id)
            uniq.append(s)
    uniq.sort(key=lambda s: (s.start_time, s.suggestion_type))
    return uniq


def _pitch(ctx: _Ctx, notes) -> list[Suggestion]:
    out = []
    for n in notes:
        if n.duration < 0.1:
            continue
        trim = n.duration * 0.2
        seg = ctx.seg(n.start + trim, n.end - trim)
        if seg.size == 0:
            continue
        ok = np.isfinite(seg)
        ratio = float(ok.mean())
        if ratio < 0.6:
            continue
        med = float(np.median(seg[ok]))
        if float(np.std(seg[ok])) > PITCH_SPREAD:  # 음 안에서 높이가 바뀜: 음높이가 아니라 나누기 문제
            continue
        dev = med - n.pitch
        target = int(round(med))
        if abs(dev) < PITCH_DEV or target == n.pitch:
            continue
        ev = [f"피치 곡선 중앙값 {med:.2f} (음표 {n.pitch} = {midi_to_name(n.pitch)}, 차이 {dev * 100:+.0f}센트)",
              f"음 가운데 60% 중 유성 프레임 {ratio:.0%}"]
        score = 0.6 * min(1.0, (abs(dev) - 0.5) / 0.7) + 0.4 * ratio
        for a, b, p in ctx.reference:
            if a < n.end and b > n.start and p == target:
                ev.append(f"참조 악보에서도 이 자리는 {midi_to_name(p)} (보조 근거)")
                break
        out.append(_make("pitch", [n], n.start, n.end, {"from": n.pitch, "pitch": target}, ev, score))
    return out


def _split(ctx: _Ctx, notes) -> list[Suggestion]:
    out = []
    for n in notes:
        if n.duration < 2 * MIN_PART_S:
            continue
        lo, hi = n.start + MIN_PART_S, n.end - MIN_PART_S
        # (1) 음 안에서 높이가 바뀜
        best = None
        step = 1.0 / ctx.fr
        t = lo
        while t <= hi + 1e-9:
            left, right = ctx.seg(n.start + 0.03, t - 0.02), ctx.seg(t + 0.02, n.end - 0.03)
            lf, rf = left[np.isfinite(left)], right[np.isfinite(right)]
            if lf.size >= 5 and rf.size >= 5:
                ml, mr = float(np.median(lf)), float(np.median(rf))
                # 변화점 = 양쪽 중앙값에서 벗어난 정도의 합이 가장 작은 곳
                cost = float(np.abs(lf - ml).sum() + np.abs(rf - mr).sum())
                if abs(mr - ml) >= SPLIT_PITCH_DEV and round(ml) != round(mr) and (best is None or cost < best[0]):
                    best = (cost, t, ml, mr)
            t += step * 2
        if best is not None:
            _cost, at, ml, mr = best
            d = abs(mr - ml)
            ev = [f"앞 {at - n.start:.2f}초 중앙 {ml:.2f}, 뒤 {n.end - at:.2f}초 중앙 {mr:.2f} (차이 {d:.2f}반음)"]
            if ctx.syllable_near(at):
                ev.append("나눌 자리에서 가사 음절이 시작합니다")
            out.append(_make("split", [n], n.start, n.end,
                             {"at": round(at, 3), "pitches": [int(round(ml)), int(round(mr))]}, ev,
                             0.5 * min(1.0, d / 2.0) + 0.3 + (0.2 if ctx.syllable_near(at) else 0.0)))
            continue
        # (2) 같은 높이 안의 재발음 근거 (자동 분할 기준에 조금 못 미침) 또는 음절 시작
        v, at = ctx.art_max(lo, hi)
        thr = ctx.split_strength
        syl_in = [s for s in ctx.syllables if lo <= s <= hi]
        if v >= SPLIT_ART_RATIO * thr or syl_in:
            if syl_in and not v >= SPLIT_ART_RATIO * thr:
                at = syl_in[0]
            ev = []
            if v >= SPLIT_ART_RATIO * thr:
                ev.append(f"재발음 지표 {v:.1f} (자동 분할 기준 {thr:.1f}의 {v / thr:.0%})")
            if syl_in:
                ev.append(f"이 음 안에서 가사 음절 {len(syl_in)}개가 시작합니다")
            score = 0.5 * min(1.0, v / thr) + (0.4 if syl_in else 0.0)
            out.append(_make("split", [n], n.start, n.end, {"at": round(at, 3)}, ev, score))
    return out


def _merge(ctx: _Ctx, notes, project) -> list[Suggestion]:
    out = []
    for a, b in zip(notes, notes[1:]):
        if a.pitch != b.pitch or b.start - a.end > MERGE_GAP_S or b.start < a.end - 1e-6:
            continue
        if project.note_status(a) != "original" or project.note_status(b) != "original":
            continue
        v, _ = ctx.art_max(b.start - 0.04, b.start + 0.04)
        if v >= MERGE_ART_MAX or ctx.syllable_near(b.start):
            continue
        gap = max(0.0, b.start - a.end) * 1000
        ev = [f"같은 높이 {midi_to_name(a.pitch)}, 사이 틈 {gap:.0f}ms",
              f"뒤 음 시작의 재발음 지표 {v:.1f} (자동 분할 기준 {ctx.split_strength:.1f})",
              "뒤 음 시작에 가사 음절 시작 없음" if ctx.syllables else "가사 정렬 없음 (음절 근거를 쓰지 못함)"]
        score = 0.5 + 0.3 * (1 - v / MERGE_ART_MAX) + (0.2 if ctx.syllables else 0.0)
        out.append(_make("merge", [a, b], a.start, b.end, {}, ev, score))
    return out


def _approach(ctx: _Ctx, notes, project) -> list[Suggestion]:
    out = []
    for a, b in zip(notes, notes[1:]):
        if a.duration > APPROACH_MAX_S or b.duration < a.duration or not (1 <= b.pitch - a.pitch <= 4):
            continue
        if b.start - a.end > MERGE_GAP_S or b.start < a.end - 1e-6:
            continue
        if project.note_status(a) != "original" or project.note_status(b) != "original":
            continue
        seg = ctx.seg(a.start, a.end)
        seg = seg[np.isfinite(seg)]
        if seg.size < 4:
            continue
        half = seg.size // 2
        rise = float(np.median(seg[half:]) - np.median(seg[:half]))
        v, _ = ctx.art_max(b.start - 0.03, b.start + 0.04)
        if rise < 0.3 or v >= 2.0 or ctx.syllable_near(b.start):
            continue
        ev = [f"{a.duration * 1000:.0f}ms 짧은 음이 {b.pitch - a.pitch}반음 위 음 바로 앞에 있음",
              f"짧은 음 안에서 {rise:.2f}반음 올라감", f"뒤 음 시작의 재발음 지표 {v:.1f} (다시 부른 근거 약함)"]
        score = 0.4 + 0.3 * min(1.0, rise / 1.5) + 0.3 * (1 - v / 2.0)
        out.append(_make("approach", [a, b], a.start, b.end, {"pitch": b.pitch, "keep": b.id}, ev, score))
    return out


def _boundary(ctx: _Ctx, notes, project) -> list[Suggestion]:
    out = []
    for i, n in enumerate(notes):
        if project.note_status(n) != "original" or n.duration < 0.12:
            continue
        prev_end = notes[i - 1].end if i else -1.0
        nxt_start = notes[i + 1].start if i + 1 < len(notes) else 1e9
        changes, ev, score = {}, [], 0.0
        # 시작: 음표 시작과 40~120ms 떨어진 곳에 자동 분할 기준 이상의 재발음 피크
        for lo, hi in ((n.start - BOUNDARY_MAX_S, n.start - BOUNDARY_MIN_S), (n.start + BOUNDARY_MIN_S, n.start + BOUNDARY_MAX_S)):
            v, at = ctx.art_max(lo, hi)
            here, _ = ctx.art_max(n.start - 0.02, n.start + 0.02)
            if v >= ctx.split_strength and v > 1.5 * max(here, 0.1) and prev_end <= at < n.end - 0.08:
                changes["start"] = round(at, 3)
                ev.append(f"시작에서 {(at - n.start) * 1000:+.0f}ms 떨어진 곳에 재발음 지표 {v:.1f} (지금 시작점 {here:.1f})")
                score = max(score, 0.4 + 0.4 * min(1.0, v / (2 * ctx.split_strength)))
                break
        # 끝: 음 끝쪽이 무성(피치 없음)으로 TAIL_UNVOICED_S 이상
        seg = ctx.seg(n.start, n.end)
        if seg.size:
            voiced = np.isfinite(seg)
            if voiced.any():
                last = int(np.flatnonzero(voiced)[-1])
                tail = (seg.size - 1 - last) / ctx.fr
                new_end = n.start + (last + 1) / ctx.fr
                if tail >= TAIL_UNVOICED_S and new_end - changes.get("start", n.start) >= 0.08 and new_end < nxt_start + 1:
                    changes["end"] = round(new_end, 3)
                    ev.append(f"음 끝 {tail * 1000:.0f}ms에 피치가 없음 (무성·잔향)")
                    score = max(score, 0.3 + 0.5 * min(1.0, tail / 0.2))
        if changes:
            out.append(_make("boundary", [n], n.start, n.end, changes, ev, score))
    return out


def _missing(ctx: _Ctx, notes) -> list[Suggestion]:
    out = []
    ok = np.isfinite(ctx.midi) & (ctx.conf >= MISSING_CONF)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    starts = [n.start for n in notes]
    for i, j in zip(edges[::2], edges[1::2]):
        a, b = ctx.start + i / ctx.fr, ctx.start + j / ctx.fr
        if b - a < MISSING_MIN_S:
            continue
        if any(x < b and y > a for x, y in ctx.blocked):
            continue
        covered = 0.0
        k = max(0, int(np.searchsorted(starts, a)) - 3)
        for n in notes[k:]:
            if n.start >= b:
                break
            covered += max(0.0, min(n.end, b) - max(n.start, a))
        if covered > 0.3 * (b - a):
            continue
        seg = ctx.midi[i:j]
        if float(np.std(seg)) > MISSING_STD:
            continue
        pitch = int(round(float(np.median(seg))))
        mean_conf = float(np.mean(ctx.conf[i:j]))
        ev = [f"{b - a:.2f}초 동안 피치가 안정적 (표준편차 {np.std(seg):.2f}반음, 신뢰도 평균 {mean_conf:.2f})",
              f"이 구간에 음표가 없음 (겹침 {covered / (b - a):.0%})"]
        score = 0.4 + 0.3 * min(1.0, (b - a) / 0.5) + 0.3 * mean_conf
        out.append(_make("missing", [], a, b, {"start": round(a, 3), "end": round(b, 3), "pitch": pitch}, ev, score,
                         fp_notes=[n for n in notes if n.start < b and n.end > a]))
    return out


# --- 상태 관리 ---------------------------------------------------------------------------

def is_outdated(s: Suggestion, project) -> bool:
    score = project.score
    if s.suggestion_type == "missing":
        a, b = s.start_time, s.end_time
        return fingerprint([n for n in score.notes() if n.start < b and n.end > a]) != s.fingerprint
    if any(t not in score for t in s.target_note_ids):
        return True
    return fingerprint([score.get(t) for t in s.target_note_ids]) != s.fingerprint


def refresh(state: SuggestionState, project) -> SuggestionState:
    """대상 음표가 바뀐 대기 제안을 outdated로. 처리한 제안(적용·무시·숨김)은 그대로 둔다."""
    items = tuple(replace(s, status=OUTDATED) if s.status == PENDING and is_outdated(s, project) else s
                  for s in state.items)
    return state if items == state.items else replace(state, items=items)


def merge_detected(old: SuggestionState, fresh: list[Suggestion], created: str = "") -> SuggestionState:
    """다시 찾기: 같은 id의 처리 상태(무시·숨김·적용)는 이어받는다. 사라진 대기 제안은 뺀다."""
    prev = {s.suggestion_id: s for s in old.items}
    items = []
    for s in fresh:
        p = prev.get(s.suggestion_id)
        items.append(replace(s, status=p.status) if p is not None and p.status in (REJECTED, DISMISSED, ACCEPTED) else s)
    kept_ids = {s.suggestion_id for s in items}
    items += [s for s in old.items if s.suggestion_id not in kept_ids and s.status in (ACCEPTED, REJECTED)]
    return SuggestionState(tuple(sorted(items, key=lambda s: (s.start_time, s.suggestion_type))), created)


def plan_apply(project, s: Suggestion):
    """(지울 음표, 넣을 음표). 대상이 바뀌었거나 사용자 음표면 EditError. 음표만 계산하고 바꾸지는 않는다."""
    from amp.core import editing

    if s.status not in (PENDING,):
        raise EditError(f"처리한 제안입니다 ({STATUS_LABELS.get(s.status, s.status)}).")
    if is_outdated(s, project):
        raise EditError("제안을 만든 뒤 음표가 바뀌어 적용할 수 없습니다. [다시 찾기]를 누르세요.")
    score = project.score
    targets = [score.get(t) for t in s.target_note_ids]
    if any(project.note_status(n) != "original" for n in targets):
        raise EditError("사용자가 고친 음표는 제안으로 바꾸지 않습니다.")
    c = s.proposed_changes
    t = s.suggestion_type
    if t == "split":
        left, right = editing.split_note(targets[0], float(c["at"]))
        if c.get("pitches"):
            left, right = left.with_(pitch=int(c["pitches"][0])), right.with_(pitch=int(c["pitches"][1]))
        return targets, [left, right]
    if t == "merge":
        return targets, [editing.merge_notes(targets)]
    if t == "approach":
        merged = editing.merge_notes(targets).with_(pitch=int(c["pitch"]))
        return targets, [merged]
    if t == "pitch":
        return targets, [targets[0].with_(pitch=int(c["pitch"]))]
    if t == "boundary":
        n = targets[0]
        start, end = float(c.get("start", n.start)), float(c.get("end", n.end))
        if end - start < editing.MIN_NOTE_DURATION:
            raise EditError("조정한 음이 너무 짧아집니다.")
        return targets, [n.with_(start=start, end=end)]
    if t == "missing":
        return [], [editing.create_note(float(c["start"]), float(c["end"]), int(c["pitch"]))]
    raise EditError(f"알 수 없는 제안 종류: {t}")


def preview_notes(project, s: Suggestion) -> tuple[list, list]:
    """미리보기용 (사라질 음표, 생길 음표). 적용할 수 없으면 ([], [])."""
    try:
        return plan_apply(project, s)
    except (EditError, KeyError, ValueError):
        return [], []


# --- 저장 ------------------------------------------------------------------------------

def to_dict(state: SuggestionState) -> dict:
    return {"created": state.created, "detector_version": state.detector_version, "score_note": SCORE_NOTE,
            "items": [{"suggestion_id": s.suggestion_id, "suggestion_type": s.suggestion_type,
                       "target_note_ids": list(s.target_note_ids), "start_time": s.start_time, "end_time": s.end_time,
                       "proposed_changes": s.proposed_changes, "evidence": list(s.evidence),
                       "confidence_or_score": s.score, "fingerprint": s.fingerprint,
                       "detector_version": s.detector_version, "status": s.status} for s in state.items]}


def from_dict(d: dict) -> SuggestionState:
    items = []
    for x in d.get("items") or ():
        status = str(x.get("status", PENDING))
        if status not in STATUS_LABELS:
            status = PENDING
        score = float(x.get("confidence_or_score", 0.0))
        if not math.isfinite(score):
            score = 0.0
        items.append(Suggestion(str(x["suggestion_id"]), str(x["suggestion_type"]), tuple(x.get("target_note_ids") or ()),
                                float(x["start_time"]), float(x["end_time"]), dict(x.get("proposed_changes") or {}),
                                tuple(x.get("evidence") or ()), score, str(x.get("fingerprint", "")),
                                str(x.get("detector_version", DETECTOR_VERSION)), status))
    return SuggestionState(tuple(items), str(d.get("created", "")), str(d.get("detector_version", DETECTOR_VERSION)))
