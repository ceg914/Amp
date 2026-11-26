"""프레임별 음향 지표 (v0.7.0 Feature E·F 공통). numpy만 쓴다 (작업 프로세스에서 계산).

16kHz, 창 512(32ms) 한 창, 홉 160(100fps, 중심 = i×10ms: pYIN과 같은 격자).
- rms_db: dBFS
- centroid_hz: 스펙트럼 중심 (밝기)
- flatness: 스펙트럼 평탄도 (기하평균/산술평균, 0~1. 잡음일수록 1에 가깝다)
- hf_ratio: 2kHz 위 에너지 비율
- hnr_db: f0가 있는 프레임만. 배음 위치(±1칸, 창 2048로 따로 계산) 에너지 / 나머지 에너지 (Harmonic-to-Noise 근사)
- h1_h2_db: 첫째·둘째 배음 크기 차이 (배음 에너지 분포)
음원 분리를 거친 보컬은 음색·배음이 바뀌었을 수 있다. 이 지표만으로 흉성·두성·믹스보이스·성대 상태를 정하지 않는다.
"""

from __future__ import annotations

import numpy as np

PITCH_SR = 16000
HOP = 160
WIN = 512
VERSION = 1


def _frames(y: np.ndarray, win: int) -> np.ndarray:
    n = 1 + len(y) // HOP
    yp = np.pad(np.asarray(y, dtype=np.float32), (win // 2, win // 2))
    return np.lib.stride_tricks.sliding_window_view(yp, win)[::HOP][:n]


def frame_features(y16: np.ndarray, f0_hz: np.ndarray | None = None) -> dict:
    fr = _frames(y16, WIN)
    n = fr.shape[0]
    win = np.hanning(WIN).astype(np.float32)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("rms_db", "centroid_hz", "flatness", "hf_ratio",
                                                             "hnr_db", "h1_h2_db")}
    freqs = np.fft.rfftfreq(WIN, 1 / PITCH_SR)
    hf = freqs >= 2000
    for a in range(0, n, 4096):
        x = fr[a:a + 4096].astype(np.float64)
        rms = np.sqrt(np.mean(x ** 2, axis=1))
        out["rms_db"][a:a + 4096] = 20 * np.log10(np.maximum(rms, 1e-10))
        P = np.abs(np.fft.rfft(x * win, axis=1)) ** 2 + 1e-18
        tot = P.sum(axis=1)
        out["centroid_hz"][a:a + 4096] = (P * freqs).sum(axis=1) / tot
        out["flatness"][a:a + 4096] = np.exp(np.mean(np.log(P[:, 1:]), axis=1)) / np.mean(P[:, 1:], axis=1)
        out["hf_ratio"][a:a + 4096] = P[:, hf].sum(axis=1) / tot
    if f0_hz is not None:
        _harmonic_metrics(y16, np.asarray(f0_hz, dtype=np.float64)[:n], out)
    return out


def _harmonic_metrics(y16, f0, out) -> None:
    N = 2048
    fr = _frames(y16, N)
    n = min(fr.shape[0], len(f0))
    win = np.hanning(N)
    df = PITCH_SR / N
    idx = np.flatnonzero(np.isfinite(f0[:n]) & (f0[:n] > 0))
    for a in range(0, len(idx), 2048):
        ii = idx[a:a + 2048]
        P = np.abs(np.fft.rfft(fr[ii].astype(np.float64) * win, axis=1)) ** 2 + 1e-18
        F = P.shape[1]
        harm = np.zeros(len(ii))
        amp = {}
        for h in range(1, 21):
            b = np.round(f0[ii] * h / df).astype(int)
            ok = b < F - 2
            e = np.zeros(len(ii))
            for o in (-1, 0, 1):
                bb = np.clip(b + o, 0, F - 1)
                e += np.where(ok, P[np.arange(len(ii)), bb], 0.0)
            harm += e
            if h <= 2:
                amp[h] = np.where(ok, P[np.arange(len(ii)), np.clip(b, 0, F - 1)], np.nan)
        lim = np.round(np.minimum(21 * f0[ii], 0.47 * PITCH_SR) / df).astype(int)
        tot = np.array([P[k, :lim[k]].sum() for k in range(len(ii))])
        noise = np.maximum(tot - harm, 1e-18)
        out["hnr_db"][ii] = 10 * np.log10(np.maximum(harm, 1e-18) / noise)
        out["h1_h2_db"][ii] = 10 * np.log10(amp[1] / amp[2])
