"""비브라토 정밀 분석 (v0.7.0 Feature D, 명세 7장). Qt 비의존, 피치 곡선을 바꾸지 않는다.

방법
1. 유성 구간마다 피치를 센트로 바꾸고, 큰 도약(50ms 안에 150센트 넘게)·옥타브 오류 근처를 빼서 조각으로 나눈다
2. 느린 흐름(2Hz 아래, 음높이가 서서히 이동하는 것)과 진동(3~9Hz 띠)을 영위상 필터로 나눈다
3. 진동의 극값(봉우리·골)을 차례로 찾는다. 이웃 극값 사이 = 반주기. 반주기마다 속도 = 1 / (2 × 반주기 시간),
   폭(봉우리-골 차이) = 봉우리 간 폭(peak-to-peak). 한쪽 진폭 = 봉우리 간 폭 / 2
4. 반주기 속도가 대역 안이고 봉우리 간 폭이 기준 이상인 반주기가 4개(2주기) 넘게 이어지면 비브라토 구간
5. 2주기가 안 되거나 주기가 불규칙하면(변동계수 0.35 초과) '불확실'. 폭이 너무 크거나(한쪽 150센트 초과)
   두 높이에 머물렀다 오가는 네모꼴이면(흔들림 RMS가 한쪽 진폭의 0.85배 넘음) '트릴 의심'

폭 정의: extent_cents = 한쪽 진폭(중심에서 위 또는 아래로 벗어난 양), extent_pp_cents = 봉우리 간 폭(최고 − 최저).
두 값을 모두 적는다 (기존 음표 속성 Vibrato.extent_cents도 한쪽 진폭이다).
대역(기본 3~9Hz)과 폭 기준은 설정으로 바꿀 수 있다. 정답 범위로 취급하지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ANALYSIS_VERSION = "1"


@dataclass(frozen=True)
class VibratoSettings:
    min_rate_hz: float = 3.0
    max_rate_hz: float = 9.0
    min_extent_pp_cents: float = 20.0  # 봉우리 간 폭 하한 (한쪽 10센트)
    min_cycles: float = 2.0
    max_period_cv: float = 0.35
    trill_extent_cents: float = 150.0  # 한쪽 진폭이 이보다 크면 트릴·음 바뀜 의심
    trill_squareness: float = 0.85  # 흔들림 RMS / (백분위 5~95 폭의 절반): 사인 0.72, 두 높이를 오가는 트릴(네모꼴)은 1에 가깝다
    jump_cents: float = 150.0  # 50ms 안에 이보다 크게 뛰면 그 근처는 비브라토로 보지 않는다
    min_run_s: float = 0.35

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VibratoSegment:
    start: float
    end: float
    rate_hz: float
    extent_cents: float  # 한쪽 진폭 평균
    extent_pp_cents: float  # 봉우리 간 폭 평균
    max_extent_cents: float  # 한쪽 진폭 최대
    center_midi: float  # 느린 흐름(중심) 평균
    regularity: float  # 1 − 반주기 길이 변동계수 (1에 가까울수록 고름)
    rate_slope_hz_per_s: float  # 속도 변화
    extent_slope_cents_per_s: float  # 폭 변화
    onset_delay_s: float | None  # 유성 구간(또는 음) 시작부터 비브라토 시작까지
    cycles: float
    status: str = "completed"  # completed / uncertain
    reason: str = ""
    cycle_times: list = field(default_factory=list)  # 반주기 가운데 시각
    cycle_rates: list = field(default_factory=list)
    cycle_extents: list = field(default_factory=list)  # 한쪽 진폭

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        for k in ("start", "end", "rate_hz", "extent_cents", "extent_pp_cents", "max_extent_cents", "center_midi",
                  "regularity", "rate_slope_hz_per_s", "extent_slope_cents_per_s", "cycles"):
            d[k] = round(float(d[k]), 3)
        d["cycle_times"] = [round(float(x), 3) for x in self.cycle_times]
        d["cycle_rates"] = [round(float(x), 2) for x in self.cycle_rates]
        d["cycle_extents"] = [round(float(x), 1) for x in self.cycle_extents]
        return d


def _butter(order: int, lo: float | None, hi: float | None, fs: float):
    from scipy.signal import butter

    if lo and hi:
        return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    if hi:
        return butter(order, hi, btype="lowpass", fs=fs, output="sos")
    return butter(order, lo, btype="highpass", fs=fs, output="sos")


def _filt(sos, x: np.ndarray) -> np.ndarray:
    from scipy.signal import sosfiltfilt

    pad = min(len(x) - 1, 3 * 12)
    return sosfiltfilt(sos, x, padlen=max(0, pad))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _extrema(x: np.ndarray) -> list[int]:
    """번갈아 나오는 봉우리·골 위치 (평평한 곳은 가운데)."""
    d = np.sign(np.diff(x))
    for i in range(1, len(d)):  # 평평한 곳은 앞 기울기를 잇는다
        if d[i] == 0:
            d[i] = d[i - 1]
    idx = np.flatnonzero(np.diff(d) != 0) + 1
    return idx.tolist()


def _parabolic(x: np.ndarray, i: int) -> tuple[float, float]:
    if 0 < i < len(x) - 1:
        a, b, c = x[i - 1], x[i], x[i + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            d = 0.5 * (a - c) / den
            return i + d, b - 0.25 * (a - c) * d
    return float(i), float(x[i])


def analyze_curve(f0_hz: np.ndarray, frame_rate: float = 100.0, start_time: float = 0.0,
                  settings: VibratoSettings | None = None, onsets: list[float] | None = None) -> list[VibratoSegment]:
    """f0 곡선(무성 NaN) → 비브라토 구간들. onsets(음 시작 시각들)를 주면 시작 지연을 그 음 시작부터 잰다."""
    st = settings or VibratoSettings()
    f0 = np.asarray(f0_hz, dtype=np.float64)
    fr = float(frame_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(f0 / 440.0) + 6900.0
    ok = np.isfinite(cents)
    # 큰 도약 근처(앞뒤 60ms)는 뺀다: 음 바뀜·옥타브 오류·성부 교차
    jump = np.zeros(len(cents), dtype=bool)
    w = max(1, int(round(0.05 * fr)))
    for k in range(1, w + 1):
        dd = np.abs(cents[k:] - cents[:-k])
        big = np.where(np.isfinite(dd), dd > st.jump_cents, False)
        jump[k:] |= big
        jump[:-k] |= big
    jump = np.convolve(jump.astype(float), np.ones(2 * w + 1), mode="same") > 0
    usable = ok & ~jump
    out: list[VibratoSegment] = []
    band = _butter(2, st.min_rate_hz * 0.8, min(st.max_rate_hz * 1.2, 0.45 * fr), fr)
    slow = _butter(2, None, min(2.0, st.min_rate_hz * 0.6), fr)
    voiced_runs = runs(ok)
    for a, b in runs(usable):
        if (b - a) / fr < st.min_run_s:
            continue
        x = cents[a:b]
        trend = _filt(slow, x)
        osc = _filt(band, x - trend)
        ext = _extrema(osc)
        if len(ext) < 3:
            continue
        pos = [_parabolic(osc, i) for i in ext]
        half = []  # (가운데 프레임, 반주기 길이 프레임, 봉우리 간 폭)
        for (p0, v0), (p1, v1) in zip(pos, pos[1:]):
            dt = p1 - p0
            if dt <= 0:
                continue
            half.append(((p0 + p1) / 2.0, dt, abs(v1 - v0)))
        good = [(m, dt, pp) for m, dt, pp in half
                if st.min_rate_hz <= fr / (2 * dt) <= st.max_rate_hz and pp >= st.min_extent_pp_cents]
        # 조건을 만족하는 반주기가 이어지는 묶음
        groups, cur = [], []
        for h in half:
            if h in good and (not cur or abs(h[0] - cur[-1][0] - (cur[-1][1] + h[1]) / 2) < 0.75 * (cur[-1][1] + h[1])):
                cur.append(h)
            else:
                if cur:
                    groups.append(cur)
                cur = [h] if h in good else []
        if cur:
            groups.append(cur)
        for g in groups:
            cyc = len(g) / 2.0
            if cyc < 1.0:
                continue
            m = np.array([h[0] for h in g])
            dts = np.array([h[1] for h in g])
            pps = np.array([h[2] for h in g])
            rates = fr / (2 * dts)
            s_fr = m[0] - dts[0] / 2
            e_fr = m[-1] + dts[-1] / 2
            t_mid = start_time + (a + m) / fr
            cv = float(np.std(dts) / np.mean(dts)) if len(dts) > 1 else 1.0
            one = pps / 2.0
            seg_start = start_time + (a + s_fr) / fr
            seg_end = start_time + (a + e_fr) / fr
            status, reason = "completed", ""
            if cyc < st.min_cycles:
                status, reason = "uncertain", f"주기가 {cyc:.1f}개뿐입니다 (기준 {st.min_cycles:.0f}개)."
            elif cv > st.max_period_cv:
                status, reason = "uncertain", f"주기가 고르지 않습니다 (변동계수 {cv:.2f})."
            i0s, i1s = int(max(0, np.floor(s_fr))), int(min(len(x), np.ceil(e_fr)))
            raw = (x - trend)[i0s:i1s]
            half_range = 0.5 * float(np.percentile(raw, 95) - np.percentile(raw, 5)) if len(raw) else 0.0
            square = float(np.sqrt(np.mean((raw - raw.mean()) ** 2)) / max(half_range, 1e-9)) if len(raw) else 0.0
            if float(np.median(one)) > st.trill_extent_cents or square > st.trill_squareness:
                status, reason = "uncertain", "폭이 매우 크거나 두 높이를 오가는 모양이라 트릴·음 바뀜일 수 있습니다."
            rs = np.polyfit(t_mid, rates, 1)[0] if len(g) >= 4 else 0.0
            es = np.polyfit(t_mid, one, 1)[0] if len(g) >= 4 else 0.0
            i0, i1 = a + int(max(0, np.floor(s_fr))), a + int(min(len(x), np.ceil(e_fr)))
            center = float(np.mean(trend[i0 - a:i1 - a])) / 100.0
            run_start = next((ra for ra, rb in voiced_runs if ra <= i0 < rb), a)
            ref = start_time + run_start / fr
            if onsets:
                prev = [o for o in onsets if o <= seg_start + 1e-6]
                if prev:
                    ref = max(ref, prev[-1])
            out.append(VibratoSegment(
                seg_start, seg_end, float(np.mean(rates)), float(np.mean(one)), float(np.mean(pps)), float(np.max(one)),
                center, float(max(0.0, 1.0 - cv)), float(rs), float(es), max(0.0, seg_start - ref), cyc, status, reason,
                t_mid.tolist(), rates.tolist(), one.tolist()))
    return out


def summary(segments: list[VibratoSegment]) -> dict:
    ok = [s for s in segments if s.status == "completed"]
    if not ok:
        return {"segments": len(segments), "completed": 0}
    dur = np.array([s.duration for s in ok])
    w = dur / dur.sum()
    return {"segments": len(segments), "completed": len(ok), "total_s": round(float(dur.sum()), 2),
            "rate_hz": round(float(np.sum(w * [s.rate_hz for s in ok])), 2),
            "extent_cents": round(float(np.sum(w * [s.extent_cents for s in ok])), 1),
            "extent_pp_cents": round(float(np.sum(w * [s.extent_pp_cents for s in ok])), 1),
            "onset_delay_s_median": round(float(np.median([s.onset_delay_s for s in ok if s.onset_delay_s is not None]
                                                          or [0.0])), 3)}
