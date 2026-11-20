"""참조 악보 음높이 ↔ 실제 피치 곡선 음정 비교 (가창 분석 보조). Qt 비의존.

악보와 다른 음정을 가수의 실수로 판정하지 않는다. 편곡·창법으로 일부러 다르게 부를 수 있다.
전곡 점수 하나 대신 음표별 결과를 먼저 낸다.

- 오차(cent) = 1200 × log2(실제 Hz / 기준 Hz), 기준 = 평균율 A4 440Hz.
- 유효 프레임만 쓴다: 피치가 있고(유성) 신뢰도가 MIN_CONF 이상이며, 목표에서 ±FAR_CENTS 안.
  그보다 먼 프레임은 '다른 음'(옥타브 오류·다른 음 부름)으로 따로 센다.
- 음 시작 ATTACK_S(최대 길이의 25%)는 오차 통계에서 빼고, 목표 도달 시간으로 따로 잰다(끌어올림을 오류로 치지 않음).
- 비브라토(3.5~9Hz, 폭 20cent 이상)가 있으면 오차는 중심선(약 1주기 이동 평균) 기준으로 잰다. 흔들림 폭은 따로 보고한다.
- 정렬하지 않은(확정 안 함) 악보는 계산은 하되 결과에 '정렬 미확정'을 붙인다. 정렬 범위 밖 음표는 계산하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amp.compare.reference import ReferenceScore

# 피치 곡선은 이미 에너지·유성 판정을 거친 프레임만 값이 있다. pYIN 유성 확률은 곡에 따라 낮게 나와(실제 곡 '조각 나비'에서
# 곡선 프레임의 중앙값 0.18) 0.5 같은 높은 기준은 프레임 대부분을 버린다. 앱 음표 검출과 비슷한 낮은 기준을 쓴다
MIN_CONF = 0.05
LOW_MEAN_CONF = 0.1
FAR_CENTS = 600.0
ATTACK_S = 0.08
REACH_CENTS = 50.0
REACH_HOLD_S = 0.05
SHORT_S = 0.12
QUANTILE = 90
SUSTAIN_CENTS = 25.0
SUSTAIN_MIN_NOTES = 3


@dataclass(frozen=True)
class NoteIntonation:
    ref_id: str
    pitch: int
    start: float  # 음원 시간
    end: float
    status: str  # scored / uncertain / missing / not_scored
    median_cents: float | None = None
    abs_q_cents: float | None = None  # |오차|의 QUANTILE 분위수
    within50: float | None = None  # ±50cent 안 프레임 비율
    valid_ratio: float = 0.0  # 음표 안 유효 프레임 비율
    far_ratio: float = 0.0  # 목표에서 6반음 넘게 벗어난 프레임 비율
    reach_s: float | None = None  # 목표(±50cent)에 50ms 이상 머물기 시작한 시각 − 음표 시작
    vibrato_cents: float | None = None
    vibrato_hz: float | None = None
    onset_diff: float | None = None  # 겹치는 자동 채보 음표 시작 − 악보 음표 시작 (음표 경계 차이)
    offset_diff: float | None = None
    short: bool = False  # 0.12초 미만 (접근음일 수 있음)
    notes: tuple[str, ...] = ()


@dataclass
class IntonationReport:
    notes: list[NoteIntonation] = field(default_factory=list)
    histogram: list[int] = field(default_factory=list)  # −100~+100cent, 10cent 칸 21개 (양 끝은 그 밖 포함)
    within50_frames: float | None = None
    scored: int = 0
    uncertain: int = 0
    missing: int = 0
    not_scored: int = 0
    sustained: list[tuple[float, float, str, float]] = field(default_factory=list)  # (시작, 끝, high/low, 평균 cent)
    extra: list[tuple[float, float]] = field(default_factory=list)  # 악보에 없는 유성 구간 (추가 의심)
    confirmed: bool = False
    caveat: str = ("악보와 다른 음정은 편곡·창법으로 일부러 다르게 불렀을 수 있습니다. 실수로 판정하지 않습니다. "
                   "음정 기준은 평균율(A4 = 440Hz)입니다.")

    def note(self, ref_id: str) -> NoteIntonation | None:
        return next((n for n in self.notes if n.ref_id == ref_id), None)


def _vibrato(c: np.ndarray, fr: float) -> tuple[float, float] | None:
    if c.size < max(10, int(0.3 * fr)):
        return None
    x = np.arange(c.size)
    r = c - np.polyval(np.polyfit(x, c, 1), x)
    r = np.convolve(r, np.ones(3) / 3, mode="same")
    sig = r[np.abs(r) > 5]
    if sig.size < 6:
        return None
    rate = np.count_nonzero(np.diff(np.sign(sig)) != 0) / 2.0 / (c.size / fr)
    extent = float((np.percentile(r, 95) - np.percentile(r, 5)) / 2.0)
    if 3.5 <= rate <= 9.0 and extent >= 20.0:
        return float(rate), extent
    return None


def analyse(ref: ReferenceScore, curve, transcribed=(), duration: float | None = None) -> IntonationReport:
    rep = IntonationReport(confirmed=ref.alignment.confirmed)
    fr = curve.frame_rate
    f0 = np.asarray(curve.f0_hz, dtype=float)
    conf = np.asarray(curve.confidence, dtype=float)
    n_frames = len(f0)
    t_end = curve.start_time + n_frames / fr
    all_cents: list[np.ndarray] = []
    covered = np.zeros(n_frames, dtype=bool)
    trans = sorted(transcribed, key=lambda n: n.start)
    for rn, a, b in ref.aligned_notes():
        if b <= a or a < curve.start_time - 1e-6 or b > t_end + 1e-6 or (duration is not None and a > duration):
            rep.notes.append(NoteIntonation(rn.id, rn.pitch, a, b, "not_scored", notes=("정렬 범위 밖",)))
            continue
        i0 = max(0, int(round((a - curve.start_time) * fr)))
        i1 = min(n_frames, int(round((b - curve.start_time) * fr)))
        covered[i0:i1] = True
        seg = f0[i0:i1]
        cseg = conf[i0:i1]
        ref_hz = 440.0 * 2 ** ((rn.pitch - 69) / 12)
        voiced = np.isfinite(seg) & (seg > 0) & (cseg >= MIN_CONF)
        cents = np.full(seg.shape, np.nan)
        cents[voiced] = 1200 * np.log2(seg[voiced] / ref_hz)
        near = voiced & (np.abs(np.nan_to_num(cents, nan=1e9)) <= FAR_CENTS)
        far_ratio = float((voiced & ~near).sum() / max(1, seg.size))
        dur = b - a
        short = dur < SHORT_S
        flags = []
        if short:
            flags.append("짧은 음표: 접근음·장식음일 수 있어 오차를 그대로 해석하지 마세요")
        valid_ratio = float(near.sum() / max(1, seg.size))
        if valid_ratio < 0.2:
            status = "missing"
            flags.append("이 음표 동안 목표 근처 유성 피치가 거의 없습니다 (음표 누락·다른 음 의심)")
            rep.notes.append(NoteIntonation(rn.id, rn.pitch, a, b, status, valid_ratio=valid_ratio, far_ratio=far_ratio,
                                            short=short, notes=tuple(flags)))
            rep.missing += 1
            continue
        # 목표 도달 시간: 첫 유성 프레임부터 ±50cent 안에 50ms 머문 시점
        reach = None
        hold = max(1, int(round(REACH_HOLD_S * fr)))
        okc = near & (np.abs(np.nan_to_num(cents, nan=1e9)) <= REACH_CENTS + 0.5)
        run = 0
        for k in range(len(okc)):
            run = run + 1 if okc[k] else 0
            if run >= hold:
                reach = (k - hold + 1) / fr
                break
        skip = min(int(round(ATTACK_S * fr)), int(len(seg) * 0.25))
        body = cents[skip:]
        body_ok = near[skip:]
        vals = body[body_ok]
        if vals.size < 3:
            vals = cents[near]
        vib = _vibrato(vals, fr) if vals.size else None
        center = vals
        if vib is not None:  # 비브라토는 오차가 아니다: 약 1주기 이동 평균(중심선)으로 잰다
            w = max(3, int(round(fr / vib[0])))
            center = np.convolve(vals, np.ones(w) / w, mode="valid") if vals.size >= w else vals
            flags.append(f"비브라토 {vib[0]:.1f}Hz, 폭 ±{vib[1]:.0f}cent (오차는 중심선 기준)")
        med = float(np.median(center))
        absq = float(np.percentile(np.abs(center), QUANTILE))
        within = float(np.mean(np.abs(center) <= 50.0 + 0.5))  # 경계(정확히 50cent)는 안으로
        all_cents.append(center)
        mean_conf = float(np.mean(cseg[near])) if near.any() else 0.0
        uncertain = valid_ratio < 0.5 or mean_conf < LOW_MEAN_CONF or not ref.alignment.confirmed
        if not ref.alignment.confirmed:
            flags.append("정렬을 확정하지 않아 결과가 바뀔 수 있습니다")
        if valid_ratio < 0.5 or mean_conf < LOW_MEAN_CONF:
            flags.append(f"피치 신뢰 프레임 {valid_ratio * 100:.0f}% · 평균 신뢰도 {mean_conf:.2f} (불확실)")
        # 음표 경계: 겹치는 자동 채보 음표
        onset_diff = offset_diff = None
        best = None
        for tn in trans:
            if tn.end <= a:
                continue
            if tn.start >= b:
                break
            ov = min(b, tn.end) - max(a, tn.start)
            if best is None or ov > best[0]:
                best = (ov, tn)
        if best is not None:
            onset_diff = best[1].start - a
            offset_diff = best[1].end - b
        status = "uncertain" if uncertain else "scored"
        rep.notes.append(NoteIntonation(
            rn.id, rn.pitch, a, b, status, median_cents=round(med, 1), abs_q_cents=round(absq, 1),
            within50=round(within, 3), valid_ratio=round(valid_ratio, 3), far_ratio=round(far_ratio, 3),
            reach_s=None if reach is None else round(reach, 3),
            vibrato_cents=None if vib is None else round(vib[1], 1), vibrato_hz=None if vib is None else round(vib[0], 2),
            onset_diff=None if onset_diff is None else round(onset_diff, 3),
            offset_diff=None if offset_diff is None else round(offset_diff, 3), short=short, notes=tuple(flags)))
        if uncertain:
            rep.uncertain += 1
        else:
            rep.scored += 1
    rep.not_scored = sum(1 for n in rep.notes if n.status == "not_scored")
    if all_cents:
        allc = np.concatenate(all_cents)
        hist, _ = np.histogram(np.clip(allc, -105, 105), bins=np.arange(-105, 115, 10))
        rep.histogram = [int(x) for x in hist]
        rep.within50_frames = round(float(np.mean(np.abs(allc) <= 50.5)), 3)
    # 계속 높게·낮게 부른 구간: 연속한 음표 SUSTAIN_MIN_NOTES개 이상이 같은 방향으로 ±25cent 넘음
    run: list[NoteIntonation] = []
    sign = 0
    for n in sorted((x for x in rep.notes if x.median_cents is not None), key=lambda x: x.start) + [None]:
        s = 0 if n is None else (1 if n.median_cents > SUSTAIN_CENTS else -1 if n.median_cents < -SUSTAIN_CENTS else 0)
        if n is not None and s != 0 and s == sign:
            run.append(n)
            continue
        if len(run) >= SUSTAIN_MIN_NOTES:
            rep.sustained.append((run[0].start, run[-1].end, "high" if sign > 0 else "low",
                                  round(float(np.mean([x.median_cents for x in run])), 1)))
        run, sign = ([n], s) if n is not None and s != 0 else ([], 0)
    # 악보에 없는 유성 구간 (정렬 범위 안, 0.25초 이상)
    if rep.notes:
        lo = min(n.start for n in rep.notes if n.status != "not_scored") if any(n.status != "not_scored" for n in rep.notes) else None
        hi = max(n.end for n in rep.notes if n.status != "not_scored") if lo is not None else None
        if lo is not None:
            voiced_all = np.isfinite(f0) & (conf >= MIN_CONF)
            i_lo, i_hi = int((lo - curve.start_time) * fr), int((hi - curve.start_time) * fr)
            extra = voiced_all & ~covered
            k = max(0, i_lo)
            while k < min(i_hi, n_frames):
                if extra[k]:
                    j = k
                    while j < min(i_hi, n_frames) and extra[j]:
                        j += 1
                    if (j - k) / fr >= 0.25:
                        rep.extra.append((curve.start_time + k / fr, curve.start_time + j / fr))
                    k = j
                else:
                    k += 1
    return rep
