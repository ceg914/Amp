"""다중 기본주파수 후보 (v0.7.0 Feature B). numpy만 쓴다 (작업 프로세스에서 실행).

단성 피치 모델을 여러 번 돌리는 방식이 아니다. 한 프레임의 스펙트럼에서 여러 목소리의 기본주파수 후보를 함께 찾는다.

1. 16kHz, 창 2048(128ms) 한 창, 홉 160(100fps, 중심 = i×10ms: pYIN과 같은 격자)
2. 스펙트럼 평탄화: 1/3옥타브 폭으로 고른 스펙트럼 윤곽을 ν=0.67 제곱으로 나눠 포먼트·음색 차이를 줄인다
3. 배음 합(harmonic salience): 후보 f0(10센트 간격)마다 1~H배 위치의 크기를 가중 합한다.
   가중치 (f0 + 27) / (h·f0 + 320) (Klapuri 2006 방식)
4. 반복 추정·제거: 가장 큰 후보를 고르고, 그 배음을 스펙트럼에서 뺀 뒤(이웃 배음 평균보다 큰 몫은 남김 = 스펙트럼 매끄러움,
   옥타브 위 목소리의 배음이 겹친 경우를 위해) 다시 찾는다. 최대 max_voices개
5. 후보마다 배음 근거(배음 크기로 가중한, 배음 사이보다 뚜렷한 배음의 비율)와 홀수/짝수 배음 비. 홀수 배음이 거의 없으면 실제 목소리의
   한 옥타브 아래(서브하모닉)일 가능성이 커 근거를 낮춘다
6. 후보 유효: 프레임 음량이 곡 최대보다 50dB 넘게 작으면 무음. 첫 후보 대비 배음 합 비율·배음 근거가 기준 이상

결과는 프레임마다 최대 max_voices개 (f0, 배음 합, 배음 근거). 목소리 잇기는 amp/engine/voices.py에서 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

PITCH_SR = 16000
HOP = 160
N_FFT = 2048
FMIN = 70.0
FMAX = 1100.0
CENTS_STEP = 10.0
N_HARM = 10
ALPHA, BETA = 27.0, 320.0
WHITEN_NU = 0.67
SILENCE_DB = -50.0  # 곡 최대 프레임 음량 대비
REL_SALIENCE = 0.25  # 첫 후보 대비 배음 합 비율 하한 (둘째·셋째 후보)
REFINE_MAX_CENTS = 100.0  # 봉우리로 다듬을 때 격자 값에서 옮길 수 있는 최대 폭
SAME_CENTS = 80.0  # 이미 고른 후보와 이만큼 가까우면 같은 목소리의 잔여로 본다
REL_HARMONIC = 0.65  # 이미 고른 후보의 2·3·4배 높이에 있는 후보는 더 커야 한다 (남은 배음을 목소리로 오인하지 않게)
MIN_EVIDENCE = 0.6
CODE_VERSION = 1


@dataclass
class Candidates:
    """프레임 격자 (i/100초). 배열 모양 (T, K). 없는 후보는 f0 NaN."""

    f0_hz: np.ndarray
    salience: np.ndarray
    evidence: np.ndarray
    odd_even: np.ndarray
    frame_db: np.ndarray  # (T,) 곡 최대 대비 dB
    start_frame: int = 0
    frame_rate: float = PITCH_SR / HOP

    @property
    def n(self) -> int:
        return int(self.f0_hz.shape[0])

    @property
    def k(self) -> int:
        return int(self.f0_hz.shape[1])

    def save(self, path) -> None:
        np.savez_compressed(path, f0_hz=self.f0_hz.astype(np.float32), salience=self.salience.astype(np.float32),
                            evidence=self.evidence.astype(np.float32), odd_even=self.odd_even.astype(np.float32),
                            frame_db=self.frame_db.astype(np.float32), start_frame=np.int64(self.start_frame),
                            frame_rate=np.float64(self.frame_rate))

    @classmethod
    def load(cls, path) -> "Candidates":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["f0_hz"].astype(np.float64), z["salience"].astype(np.float64),
                       z["evidence"].astype(np.float64), z["odd_even"].astype(np.float64),
                       z["frame_db"].astype(np.float64), int(z["start_frame"]), float(z["frame_rate"]))


def stft_mag(y16: np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    """(F, T) 크기 스펙트럼. 프레임 i 중심 = i×HOP (앞뒤 n_fft/2 0 채움), T = len//HOP + 1."""
    y = np.asarray(y16, dtype=np.float32)
    n = 1 + len(y) // HOP
    yp = np.pad(y, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(yp, n_fft)[::HOP][:n]
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n_fft // 2 + 1, n), dtype=np.float32)
    for a in range(0, n, 2048):  # 메모리를 아끼려고 나눠 계산
        out[:, a:a + 2048] = np.abs(np.fft.rfft(frames[a:a + 2048] * win, axis=1)).T
    return out


def whiten(mag: np.ndarray, sr: int = PITCH_SR) -> np.ndarray:
    """주파수 방향으로 1/3옥타브 폭의 평균 윤곽을 구해 ν 제곱으로 나눈다 (윤곽이 큰 곳의 크기를 줄임)."""
    F = mag.shape[0]
    df = sr / (2 * (F - 1))
    freqs = np.arange(F) * df
    cs = np.cumsum(np.r_[np.zeros((1, mag.shape[1])), mag.astype(np.float64) ** 2], axis=0)
    half = np.maximum(1, np.round(freqs * (2 ** (1 / 6) - 1) / df)).astype(int)
    lo = np.clip(np.arange(F) - half, 0, F)
    hi = np.clip(np.arange(F) + half + 1, 0, F)
    env = np.sqrt((cs[hi] - cs[lo]) / (hi - lo)[:, None])
    floor = np.percentile(env, 5) + 1e-9
    return (mag / np.maximum(env, floor) ** WHITEN_NU).astype(np.float32)


def f0_grid(fmin: float = FMIN, fmax: float = FMAX, step: float = CENTS_STEP) -> np.ndarray:
    n = int(np.floor(1200 * np.log2(fmax / fmin) / step)) + 1
    return fmin * 2.0 ** (np.arange(n) * step / 1200.0)


def _maxfilt(x: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return x
    out = x.copy()
    for s in range(1, r + 1):
        out[s:] = np.maximum(out[s:], x[:-s])
        out[:-s] = np.maximum(out[:-s], x[s:])
    return out


def salience(Y: np.ndarray, grid: np.ndarray, n_harm: int = N_HARM, sr: int = PITCH_SR) -> np.ndarray:
    """(G, T) 배음 합. 배음 위치 주변(낮은 배음 ±1칸, 높은 배음 ±2칸)의 최댓값을 쓴다 (비브라토·창 번짐)."""
    F = Y.shape[0]
    df = sr / (2 * (F - 1))
    narrow, wide = _maxfilt(Y, 1), _maxfilt(Y, 2)
    S = np.zeros((len(grid), Y.shape[1]), dtype=np.float32)
    for h in range(1, n_harm + 1):
        fh = grid * h
        ok = fh < 0.47 * sr
        if not ok.any():
            break
        idx = np.round(fh[ok] / df).astype(int)
        w = ((grid[ok] + ALPHA) / (fh[ok] + BETA)).astype(np.float32)
        src = narrow if h <= 4 else wide
        S[ok] += w[:, None] * src[idx]
    return S


def harmonic_amps(Y: np.ndarray, f0: np.ndarray, n_harm: int = 8, sr: int = PITCH_SR) -> np.ndarray:
    """(T, n_harm) 프레임별 f0의 h배 위치 최대 크기 (±1칸). f0 NaN이면 0."""
    F, T = Y.shape
    df = sr / (2 * (F - 1))
    out = np.zeros((T, n_harm), dtype=np.float32)
    ok = np.isfinite(f0)
    tt = np.flatnonzero(ok)
    for h in range(1, n_harm + 1):
        idx = np.round(f0[ok] * h / df).astype(int)
        valid = idx < F - 1
        vals = np.zeros(len(tt), dtype=np.float32)
        i = np.clip(idx, 1, F - 2)
        vals[valid] = np.maximum(np.maximum(Y[i - 1, tt], Y[i, tt]), Y[i + 1, tt])[valid]
        out[tt, h - 1] = vals
    return out


def refine_by_peaks(Y: np.ndarray, f0: np.ndarray, floor: np.ndarray, n_harm: int = 4, sr: int = PITCH_SR) -> np.ndarray:
    """배음 1~n_harm의 실제 봉우리 위치(로그 크기 포물선 보간)로 f0를 다듬는다. 배음 크기로 가중 평균하고,
    원래 값과 50센트 넘게 다르면 원래 값을 둔다 (배음 합 격자는 칸 폭만큼 평평해 순음에서 치우칠 수 있어서)."""
    F, T = Y.shape
    df = sr / (2 * (F - 1))
    out = f0.copy()
    tt = np.flatnonzero(np.isfinite(f0))
    if len(tt) == 0:
        return out
    num = np.zeros(len(tt))
    den = np.zeros(len(tt))
    logY = np.log(np.maximum(Y, 1e-12))
    for h in range(1, n_harm + 1):
        c = np.round(f0[tt] * h / df).astype(int)
        ok = (c >= 3) & (c < F - 3)
        win = np.stack([Y[np.clip(c + o, 0, F - 1), tt] for o in (-2, -1, 0, 1, 2)])  # (5, n)
        k = np.argmax(win, axis=0) - 2
        p = np.clip(c + k, 1, F - 2)
        a, b, cc = logY[p - 1, tt], logY[p, tt], logY[p + 1, tt]
        den_p = a - 2 * b + cc
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(np.abs(den_p) > 1e-9, 0.5 * (a - cc) / den_p, 0.0)
        pos = p + np.clip(d, -0.5, 0.5)
        amp = Y[p, tt]
        use = ok & (amp > 2 * floor[tt])
        num += np.where(use, amp * pos * df / h, 0.0)
        den += np.where(use, amp, 0.0)
    est = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        good = np.isfinite(est) & (np.abs(1200 * np.log2(est / f0[tt])) <= REFINE_MAX_CENTS)
    out[tt[good]] = est[good]
    return out


def _hann_mag(x: np.ndarray) -> np.ndarray:
    """한 창(Hann)의 주파수 응답 크기 (칸 단위 거리 x, 0에서 1). 주엽 폭 ±2칸."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = np.abs(x) < 2.0
    xm = x[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sinc(xm) / (1.0 - xm ** 2)
    v[np.isclose(np.abs(xm), 1.0)] = 0.5
    out[m] = np.abs(v)
    return out


def _cancel(Y: np.ndarray, f0: np.ndarray, n_harm: int = N_HARM, sr: int = PITCH_SR) -> np.ndarray:
    """f0의 배음 몫을 뺀다: 배음 크기 × 창 응답 모양(정확한 소수 칸 위치 기준, ±2칸). 배음 크기는 앞뒤 배음 평균을
    넘지 않게(매끄러움) 잡아, 옥타브 위 등 다른 목소리의 배음이 겹친 몫은 남긴다."""
    F, T = Y.shape
    df = sr / (2 * (F - 1))
    amps = harmonic_amps(Y, f0, n_harm, sr)
    pad = np.pad(amps, ((0, 0), (1, 1)), mode="edge")
    smooth = np.minimum(amps, (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0)
    smooth[:, 0] = amps[:, 0]  # 기본음은 그대로 뺀다
    out = Y.copy()
    tt = np.flatnonzero(np.isfinite(f0))
    for h in range(1, n_harm + 1):
        pos = f0[tt] * h / df
        base = np.round(pos).astype(int)
        valid = (base >= 3) & (base < F - 3)
        tv, pv, bv = tt[valid], pos[valid], base[valid]
        a = smooth[tv, h - 1] * (1.0 + 0.15 * (h - 1) / n_harm)  # 높은 배음은 비브라토로 번져 조금 더 뺀다
        for off in (-2, -1, 0, 1, 2):
            ii = bv + off
            out[ii, tv] = np.maximum(0.0, out[ii, tv] - a * _hann_mag(ii - pv).astype(np.float32) / 1.0)
    return out


def _refine(S: np.ndarray, k: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """격자 최댓값 주변 포물선 보간으로 센트 단위를 다듬는다."""
    G = S.shape[0]
    t = np.arange(S.shape[1])
    k = np.clip(k, 1, G - 2)
    a, b, c = S[k - 1, t], S[k, t], S[k + 1, t]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    d = np.clip(d, -0.5, 0.5)
    return grid[0] * 2.0 ** ((k + d) * CENTS_STEP / 1200.0)


OCT_DOWN_SAL = 0.5  # f/2의 배음 합이 f의 이만큼 이상이고
OCT_DOWN_ODD = 0.45  # f/2의 홀수 배음(f/2, 3f/2, 5f/2)이 짝수 배음의 이만큼 이상이고
OCT_DOWN_FUND = 0.3  # f/2 위치의 크기가 f 위치 크기의 이만큼 이상이면 f/2를 고른다


def _prefer_octave_down(Y, S, grid, cand, peak, floor, fmin):
    """옥타브 위 오류 고치기: 고른 f의 한 옥타브 아래 f/2에 실제 홀수 배음이 있으면 f/2가 기본음이다
    (f의 배음은 모두 f/2의 짝수 배음이라, 홀수 배음이 뚜렷해야만 아래 목소리가 있다고 본다)."""
    half = cand / 2.0
    ok = half >= fmin
    if not ok.any():
        return cand, peak
    T = len(cand)
    gi = np.clip(np.round(1200 * np.log2(np.where(ok, half, fmin) / grid[0]) / CENTS_STEP).astype(int), 0, len(grid) - 1)
    lo = np.clip(gi - 2, 0, len(grid) - 1)
    hi = np.clip(gi + 3, 0, len(grid))
    sal_half = np.array([S[lo[t]:hi[t], t].max() if hi[t] > lo[t] else 0.0 for t in range(T)])
    amps = harmonic_amps(Y, np.where(ok, half, np.nan), 6)
    _e, odd_even = evidence_of(amps, floor)
    own = harmonic_amps(Y, cand, 1)[:, 0]
    odd_present = (amps[:, 0] > 2 * floor) & (amps[:, 2] > 2 * floor) & (amps[:, 0] >= OCT_DOWN_FUND * own)
    switch = ok & (sal_half >= OCT_DOWN_SAL * peak) & (odd_even >= OCT_DOWN_ODD) & odd_present
    return np.where(switch, half, cand), np.where(switch, np.maximum(sal_half, peak), peak)


def evidence_of(amps: np.ndarray, floor: np.ndarray, between: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(배음 근거, 홀/짝 비).
    배음 근거 = 배음 크기로 가중한, '배음 사이(h±0.5배)보다 뚜렷하고 바닥의 2배를 넘는 배음'의 비율 (0~1).
    순음(사인)은 기본음 하나만 있어도 1에 가깝고, 잡음은 배음 위치와 사이가 비슷해 낮다.
    홀/짝 = (1·3·5배) / (2·4·6배). 작으면 실제 목소리의 한 옥타브 아래(서브하모닉)일 가능성이 크다."""
    if between is None:
        between = np.zeros_like(amps)
    present = (amps > 2.0 * floor[:, None]) & (amps > 1.5 * between)
    ev = (amps * present).sum(axis=1) / np.maximum(amps.sum(axis=1), 1e-12)
    odd = amps[:, 0] + amps[:, 2] + amps[:, 4]
    even = amps[:, 1] + amps[:, 3] + amps[:, 5]
    return ev, odd / (even + 1e-9)


def between_amps(Y: np.ndarray, f0: np.ndarray, n_harm: int = 8, sr: int = PITCH_SR) -> np.ndarray:
    """(T, n_harm) 배음 h의 양옆 (h-0.5)·(h+0.5)배 위치 크기의 평균."""
    lo = harmonic_amps(Y, f0 * 0.5, 2 * n_harm + 1, sr)  # (k·f0/2), k = 1..2H+1
    idx = np.arange(1, n_harm + 1)
    return 0.5 * (lo[:, 2 * idx - 2] + lo[:, 2 * idx])  # (2h-1)·f0/2, (2h+1)·f0/2


def candidates(y16: np.ndarray, max_voices: int = 2, fmin: float = FMIN, fmax: float = FMAX,
               progress: Callable[[float, str], None] | None = None,
               is_cancelled: Callable[[], bool] | None = None) -> Candidates:
    report = progress or (lambda f, m: None)
    max_voices = int(max(1, min(3, max_voices)))
    report(0.02, "스펙트럼 계산 중")
    mag = stft_mag(y16)
    T = mag.shape[1]
    energy = np.sqrt(np.mean(mag.astype(np.float64) ** 2, axis=0))
    db = 20 * np.log10(np.maximum(energy, 1e-12))
    db = db - (db.max() if T else 0.0)
    loud = db > SILENCE_DB
    Y = whiten(mag)
    floor = np.median(Y, axis=0) + 1e-9  # 프레임별 바닥 (배음 근거 기준)
    grid = f0_grid(fmin, fmax)
    f0 = np.full((T, max_voices), np.nan)
    sal = np.zeros((T, max_voices))
    ev = np.zeros((T, max_voices))
    oe = np.full((T, max_voices), np.nan)
    first = None
    for k in range(max_voices):
        if is_cancelled and is_cancelled():
            from amp.core.errors import AnalysisCancelled

            raise AnalysisCancelled("다중 피치 분석을 취소했습니다.")
        report(0.1 + 0.8 * k / max_voices, f"후보 {k + 1} 찾는 중")
        S = salience(Y, grid)
        best = np.argmax(S, axis=0)
        peak = S[best, np.arange(T)]
        cand = _refine(S, best, grid)
        cand, peak = _prefer_octave_down(Y, S, grid, cand, peak, floor, fmin)
        cand = refine_by_peaks(Y, cand, floor)
        amps = harmonic_amps(Y, cand, 8)
        e, r = evidence_of(amps, floor, between_amps(Y, cand, 8))
        sub = r < 0.25  # 홀수 배음이 거의 없음: 실제 목소리의 한 옥타브 아래(서브하모닉)일 가능성
        if first is None:
            first = np.maximum(peak, 1e-9)
            keep = loud & (e >= MIN_EVIDENCE) & ~sub
        else:
            keep = loud & (peak >= REL_SALIENCE * first) & (e >= MIN_EVIDENCE) & ~sub
            for j in range(k):
                prev = f0[:, j]
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = 1200 * np.log2(cand / prev)
                keep &= ~(np.abs(c) < SAME_CENTS)  # 이미 고른 후보와 같은 높이 (제거 뒤 남은 봉우리)
                on_harm = np.zeros(T, dtype=bool)
                for m in (1200, 1902, 2400):  # 2·3·4배 위
                    on_harm |= np.abs(c - m) < 40
                keep &= ~on_harm | (peak >= REL_HARMONIC * first)
        f0[keep, k] = cand[keep]
        sal[keep, k] = peak[keep]
        ev[keep, k] = e[keep]
        oe[keep, k] = r[keep]
        Y = _cancel(Y, np.where(np.isfinite(f0[:, k]), f0[:, k], cand), N_HARM)
    report(1.0, "다중 피치 후보 완료")
    return Candidates(f0, sal, ev, oe, db)
