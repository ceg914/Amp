"""두 분리 모델(A·B) 결과로 각각 만든 자동 채보의 차이. Qt 비의존, 읽기 전용.

정답 채보가 없으므로 모든 값은 '두 결과가 어떻게 다른가'이지 '어느 쪽이 맞나'가 아니다. 음표가 더 많다고, 유성 구간이
더 길다고 더 정확하다는 뜻이 아니다. 어느 모델이 낫다는 판정은 하지 않는다. 사용자가 듣고 고른다.

- 음표 대응: 시간 겹침 비율(IoU)이 MATCH_IOU 이상인 쌍을 겹침이 큰 순서로 1:1로 묶는다.
- 같은 음 반복 분할 차이: 한쪽 음표 하나에 다른 쪽의 같은 높이 음표 2개 이상이 각각 절반 넘게 들어가면 묶음으로 본다.
- 유성 프레임: 피치 곡선 f0가 있고 신뢰도가 VOICED_CONF 이상인 프레임 (두 분석 모두 같은 pYIN·같은 100fps 격자).
- 옥타브 차이 의심: 두 곡선이 모두 유성인 프레임에서 차이가 12±OCTAVE_TOL 반음인 구간이 OCTAVE_MIN_S 이상.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amp.compare.review import ReviewItem, _curve_midi, _item

MATCH_IOU = 0.3
PITCH_DIFF = 1  # 반음 (대응한 음표의 높이가 이만큼 이상 다르면 '음높이 다름')
VOICED_CONF = 0.05  # 피치 비교·억양 분석과 같은 기준
ONLY_VOICED_MIN_S = 0.3
OCTAVE_TOL = 1.0
OCTAVE_MIN_S = 0.1
CURVE_DIFF_SEMI = 1.0
CURVE_DIFF_MIN_S = 0.2
NOVOCAL_MANY = 3
DISCLAIMER = ("정답 채보가 없어 두 결과의 '차이'만 보여 줍니다. 음표 수나 유성 구간 길이가 크다고 더 정확하다는 뜻이 아닙니다. "
              "어느 모델이 나은지는 직접 들어 보고 판단하세요.")


@dataclass(frozen=True)
class NotePair:
    a: object
    b: object
    iou: float

    @property
    def onset_diff(self) -> float:
        return float(self.b.start - self.a.start)

    @property
    def offset_diff(self) -> float:
        return float(self.b.end - self.a.end)

    @property
    def pitch_diff(self) -> int:
        return int(self.b.pitch - self.a.pitch)


@dataclass(frozen=True)
class SplitDiff:
    """한쪽은 음표 하나, 다른 쪽은 같은 높이 음표 여러 개 (재발음 분할 차이)."""
    single_side: str  # "a" 또는 "b": 하나로 잡은 쪽
    single: object
    parts: tuple

    @property
    def start(self) -> float:
        return min(self.single.start, self.parts[0].start)

    @property
    def end(self) -> float:
        return max(self.single.end, self.parts[-1].end)


@dataclass
class ModelDiff:
    label_a: str
    label_b: str
    notes_a: int
    notes_b: int
    pairs: list[NotePair] = field(default_factory=list)
    only_a: list = field(default_factory=list)
    only_b: list = field(default_factory=list)
    splits: list[SplitDiff] = field(default_factory=list)
    voiced_a: float = 0.0  # 유성 프레임 비율 (곡 전체)
    voiced_b: float = 0.0
    voiced_both_s: float = 0.0
    only_voiced_a: list[tuple[float, float]] = field(default_factory=list)  # A만 유성인 구간 (ONLY_VOICED_MIN_S 이상)
    only_voiced_b: list[tuple[float, float]] = field(default_factory=list)
    conf_a: float = float("nan")  # 유성 프레임 신뢰도 중앙값
    conf_b: float = float("nan")
    octave: list[tuple[float, float, float]] = field(default_factory=list)  # (시작, 끝, B−A 평균 반음)
    curve_diff: list[tuple[float, float, float]] = field(default_factory=list)  # 옥타브가 아닌 1반음 이상 차이 구간
    novocal_notes: dict = field(default_factory=dict)  # "a"/"b" → 사용자 '보컬 없는 구간' 안 음표 목록
    frame_rate: float = 100.0

    @property
    def pitch_changed(self) -> list[NotePair]:
        return [p for p in self.pairs if abs(p.pitch_diff) >= PITCH_DIFF]

    def timing_stats(self) -> dict:
        """대응한 음표의 시작·끝 차이(ms): 중앙값과 90% 값 (절댓값)."""
        if not self.pairs:
            return {"pairs": 0}
        on = np.abs([p.onset_diff for p in self.pairs]) * 1000
        off = np.abs([p.offset_diff for p in self.pairs]) * 1000
        return {"pairs": len(self.pairs), "onset_median_ms": float(np.median(on)), "onset_p90_ms": float(np.percentile(on, 90)),
                "offset_median_ms": float(np.median(off)), "offset_p90_ms": float(np.percentile(off, 90))}

    def summary_rows(self) -> list[tuple[str, str, str]]:
        """(지표, A, B) — 화면 표와 보고서에 쓴다. 우열 표시는 하지 않는다."""
        def pct(x):
            return f"{100 * x:.1f}%"

        def conf(x):
            return "—" if not np.isfinite(x) else f"{x:.2f}"

        only_a_s = sum(e - s for s, e in self.only_voiced_a)
        only_b_s = sum(e - s for s, e in self.only_voiced_b)
        a_split = sum(1 for s in self.splits if s.single_side == "b")  # A가 여러 개로 나눈 묶음
        b_split = sum(1 for s in self.splits if s.single_side == "a")
        return [
            ("자동 음표 수", str(self.notes_a), str(self.notes_b)),
            ("유성 프레임 비율", pct(self.voiced_a), pct(self.voiced_b)),
            ("이 모델만 유성인 구간", f"{len(self.only_voiced_a)}곳 · {only_a_s:.1f}초", f"{len(self.only_voiced_b)}곳 · {only_b_s:.1f}초"),
            ("피치 신뢰도 중앙값 (유성 프레임)", conf(self.conf_a), conf(self.conf_b)),
            ("이 모델에만 있는 음표", str(len(self.only_a)), str(len(self.only_b))),
            ("같은 음 반복을 여러 개로 나눔", f"{a_split}곳", f"{b_split}곳"),
            ("'보컬 없는 구간' 안의 음표", str(len(self.novocal_notes.get("a", ()))), str(len(self.novocal_notes.get("b", ())))),
        ]

    def shared_rows(self) -> list[tuple[str, str]]:
        st = self.timing_stats()
        rows = [("시간이 대응하는 음표 쌍", f"{len(self.pairs)}쌍 (겹침 IoU {MATCH_IOU:g} 이상)"),
                ("음높이가 다르게 잡힌 쌍", f"{len(self.pitch_changed)}쌍 (±{PITCH_DIFF}반음 이상)")]
        if st["pairs"]:
            rows += [("시작 시각 차이", f"중앙값 {st['onset_median_ms']:.0f}ms · 90% {st['onset_p90_ms']:.0f}ms"),
                     ("끝 시각 차이", f"중앙값 {st['offset_median_ms']:.0f}ms · 90% {st['offset_p90_ms']:.0f}ms")]
        rows += [("둘 다 유성인 시간", f"{self.voiced_both_s:.1f}초"),
                 ("옥타브 차이 의심 구간", f"{len(self.octave)}곳 (12±{OCTAVE_TOL:g}반음, {OCTAVE_MIN_S * 1000:.0f}ms 이상)"),
                 ("피치 곡선 차이 구간", f"{len(self.curve_diff)}곳 ({CURVE_DIFF_SEMI:g}반음 이상, {CURVE_DIFF_MIN_S * 1000:.0f}ms 이상)")]
        return rows


def _iou(a, b) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / (max(a.end, b.end) - min(a.start, b.start))


def _inside(part, whole) -> float:
    inter = min(part.end, whole.end) - max(part.start, whole.start)
    return max(0.0, inter) / max(part.end - part.start, 1e-9)


def _splits(singles: list, many: list, side: str) -> list[SplitDiff]:
    out = []
    for s in singles:
        parts = sorted((m for m in many if m.pitch == s.pitch and _inside(m, s) > 0.5), key=lambda n: n.start)
        if len(parts) >= 2:
            out.append(SplitDiff(side, s, tuple(parts)))
    return out


def match_notes(a: list, b: list) -> tuple[list[NotePair], list, list, list[SplitDiff]]:
    a = sorted(a, key=lambda n: n.start)
    b = sorted(b, key=lambda n: n.start)
    splits = _splits(a, b, "a") + _splits(b, a, "b")
    used_a = {id(s.single) for s in splits if s.single_side == "a"} | {id(p) for s in splits if s.single_side == "b" for p in s.parts}
    used_b = {id(s.single) for s in splits if s.single_side == "b"} | {id(p) for s in splits if s.single_side == "a" for p in s.parts}
    ra = [n for n in a if id(n) not in used_a]
    rb = [n for n in b if id(n) not in used_b]
    starts = np.array([n.start for n in rb])
    longest = max((n.end - n.start for n in rb), default=0.0)
    cands = []
    for i, x in enumerate(ra):
        lo = int(np.searchsorted(starts, x.start - longest, "left")) if len(starts) else 0
        hi = int(np.searchsorted(starts, x.end, "left")) if len(starts) else 0
        for j in range(lo, hi):
            v = _iou(x, rb[j])
            if v >= MATCH_IOU:
                cands.append((v, i, j))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    ta, tb = set(), set()
    pairs = []
    for v, i, j in cands:
        if i in ta or j in tb:
            continue
        ta.add(i)
        tb.add(j)
        pairs.append(NotePair(ra[i], rb[j], float(v)))
    pairs.sort(key=lambda p: p.a.start)
    only_a = [n for i, n in enumerate(ra) if i not in ta]
    only_b = [n for j, n in enumerate(rb) if j not in tb]
    return pairs, only_a, only_b, splits


def _runs(mask: np.ndarray, t0: float, fr: float, min_s: float) -> list[tuple[int, int]]:
    out = []
    if not mask.any():
        return out
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (e - s) / fr >= min_s:
            out.append((int(s), int(e)))
    return out


def compare(result_a, result_b, label_a: str = "A", label_b: str = "B", no_vocal_ranges=()) -> ModelDiff:
    na, nb = list(result_a.notes), list(result_b.notes)
    pairs, only_a, only_b, splits = match_notes(na, nb)
    d = ModelDiff(label_a, label_b, len(na), len(nb), pairs, only_a, only_b, splits)
    ca, cb = result_a.pitch_curve, result_b.pitch_curve
    if ca is not None and cb is not None:
        ma, fa = _curve_midi(ca)
        mb, fb = _curve_midi(cb)
        fr = float(ca.frame_rate)
        d.frame_rate = fr
        off = int(round((cb.start_time - ca.start_time) * fr))  # 같은 격자면 0
        n = min(len(ma), len(mb) + off) if off >= 0 else min(len(ma), len(mb))
        ma, fa = ma[:n], fa[:n]
        mb, fb = (np.concatenate([np.full(off, np.nan), mb])[:n], np.concatenate([np.zeros(off), fb])[:n]) if off > 0 \
            else (mb[:n], fb[:n])
        va = np.isfinite(ma) & (fa >= VOICED_CONF)
        vb = np.isfinite(mb) & (fb >= VOICED_CONF)
        t0 = float(ca.start_time)
        d.voiced_a = float(va.mean()) if n else 0.0
        d.voiced_b = float(vb.mean()) if n else 0.0
        both = va & vb
        d.voiced_both_s = float(both.sum() / fr)
        d.conf_a = float(np.median(fa[va])) if va.any() else float("nan")
        d.conf_b = float(np.median(fb[vb])) if vb.any() else float("nan")
        d.only_voiced_a = [(t0 + s / fr, t0 + e / fr) for s, e in _runs(va & ~vb, t0, fr, ONLY_VOICED_MIN_S)]
        d.only_voiced_b = [(t0 + s / fr, t0 + e / fr) for s, e in _runs(vb & ~va, t0, fr, ONLY_VOICED_MIN_S)]
        diff = np.where(both, mb - ma, np.nan)
        octv = both & (np.abs(np.abs(diff) - 12) <= OCTAVE_TOL)
        d.octave = [(t0 + s / fr, t0 + e / fr, float(np.nanmean(diff[s:e]))) for s, e in _runs(octv, t0, fr, OCTAVE_MIN_S)]
        other = both & ~octv & (np.abs(diff) >= CURVE_DIFF_SEMI)
        d.curve_diff = [(t0 + s / fr, t0 + e / fr, float(np.nanmedian(diff[s:e]))) for s, e in _runs(other, t0, fr, CURVE_DIFF_MIN_S)]
    for side, notes in (("a", na), ("b", nb)):
        d.novocal_notes[side] = [x for x in notes if any(min(x.end, e) - max(x.start, s) > 0.5 * (x.end - x.start)
                                                        for s, e in no_vocal_ranges)]
    return d


def review_items(d: ModelDiff, max_items: int = 400) -> list[ReviewItem]:
    """검토 목록의 '모델 비교' 항목. 음표 id는 비교용 자동 음표(작업본이 아님)라 note_ids에 넣지 않는다."""
    A, B = d.label_a, d.label_b
    out: list[ReviewItem] = []
    for side, notes, here, there in (("a", d.only_a, A, B), ("b", d.only_b, B, A)):
        for n in notes:
            out.append(_item(f"model_only_{side}", "model", n.start, n.end,
                             f"{here} 결과에만 있는 음표 (MIDI {n.pitch}, {(n.end - n.start) * 1000:.0f}ms). {there} 결과에는 "
                             f"시간이 겹치는(IoU {MATCH_IOU:g} 이상) 음표가 없음", "medium",
                             basis=(side, n.pitch, round(n.start, 2), round(n.end, 2))))
    for p in d.pitch_changed:
        s, e = min(p.a.start, p.b.start), max(p.a.end, p.b.end)
        out.append(_item("model_pitch", "model", s, e,
                         f"같은 시간의 음표 높이가 다름: {A} MIDI {p.a.pitch}, {B} MIDI {p.b.pitch} ({p.pitch_diff:+d}반음)",
                         "high" if abs(p.pitch_diff) >= 12 else "medium", basis=(p.a.pitch, p.b.pitch, round(s, 2))))
    for s, e, m in d.octave:
        out.append(_item("model_octave", "model", s, e,
                         f"두 피치 곡선이 약 한 옥타브 다름 ({B}−{A} 평균 {m:+.1f}반음, {(e - s) * 1000:.0f}ms) — 옥타브 오류 의심",
                         "high", basis=(round(s, 2), round(e, 2))))
    for s, e, m in d.curve_diff:
        out.append(_item("model_curve", "model", s, e,
                         f"두 피치 곡선 차이 {m:+.1f}반음 ({(e - s) * 1000:.0f}ms, 기준 {CURVE_DIFF_SEMI:g}반음)", "low",
                         basis=(round(s, 2), round(e, 2))))
    for side, spans, here, there in (("a", d.only_voiced_a, A, B), ("b", d.only_voiced_b, B, A)):
        for s, e in spans:
            out.append(_item(f"model_voiced_{side}", "model", s, e,
                             f"{here}에서만 유성으로 잡힌 구간 {e - s:.2f}초 ({there}는 무성 또는 신뢰도 {VOICED_CONF:g} 미만) — "
                             "반주가 섞였거나 보컬이 빠졌을 수 있음", "medium", basis=(side, round(s, 2), round(e, 2))))
    for sp in d.splits:
        one = A if sp.single_side == "a" else B
        many = B if sp.single_side == "a" else A
        out.append(_item("model_repeat", "model", sp.start, sp.end,
                         f"같은 높이(MIDI {sp.single.pitch})를 {one}은 음표 1개, {many}는 {len(sp.parts)}개로 잡음 — "
                         f"재발음이 {one}에서 나뉘지 않았거나 {many}에서 과하게 나뉘었을 수 있음", "medium",
                         basis=(sp.single_side, sp.single.pitch, round(sp.start, 2), len(sp.parts))))
    for side, label in (("a", A), ("b", B)):
        notes = d.novocal_notes.get(side, [])
        if len(notes) >= NOVOCAL_MANY:
            s, e = min(n.start for n in notes), max(n.end for n in notes)
            out.append(_item(f"model_novocal_{side}", "model", s, e,
                             f"사용자가 지정한 '보컬 없는 구간' 안에 {label} 기반 음표가 {len(notes)}개 (기준 {NOVOCAL_MANY}개) — "
                             "반주가 보컬로 분리됐을 수 있음", "medium", basis=(side, len(notes))))
    out.sort(key=lambda x: (x.start, x.kind))
    return out[:max_items]

