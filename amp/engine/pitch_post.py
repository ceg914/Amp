"""pYIN 결과 후처리 (Qt 비의존, numpy·scipy).

1. 에너지 게이트: 보컬 트랙이 아주 조용한 프레임(분리 잔여음, 호흡 끝)은 무성으로 본다.
2. 짧은 유성 조각 제거: 40ms 미만으로 튀어나온 유성 구간은 잡음으로 본다.
3. 옥타브 오류 보정: 주변 유성 프레임의 이동 중앙값과 약 ±12반음 차이 나는 프레임을 되돌린다.
   구간 전체가 앞뒤 구간과 한 옥타브 어긋난 경우도 구간 단위로 보정한다.
4. 튐 제거: 5프레임 중앙값에서 1.5반음 넘게 벗어난 프레임을 중앙값으로 바꾼다.
5. 무성 경계를 넘지 않는 3프레임 중앙값 평활.

무성 프레임은 끝까지 NaN으로 남긴다. 화면에서 곡선을 끊어 그리는 기준이 된다.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from amp.core.model import PitchCurve
from amp.core.pitch import hz_to_midi, midi_to_hz

MIN_ISLAND_S = 0.04
OCTAVE_WINDOW_S = 0.35
OCTAVE_TOL = 1.5  # 반음
SPIKE_TOL = 1.5  # 반음
ENERGY_FLOOR_DB = -60.0  # 절대 하한 (dBFS)
ENERGY_REL_DB = -40.0  # 곡 안 최대 RMS 대비 하한


def voiced_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 구간의 [시작, 끝) 목록."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def energy_gate(rms: np.ndarray) -> np.ndarray:
    """소리가 충분히 큰 프레임이면 True."""
    rms = np.asarray(rms, dtype=np.float64)
    if rms.size == 0 or rms.max() <= 0:
        return np.zeros(rms.shape, dtype=bool)
    floor = max(10 ** (ENERGY_FLOOR_DB / 20), rms.max() * 10 ** (ENERGY_REL_DB / 20))
    return rms >= floor


def _fix_octaves(midi: np.ndarray, frame_rate: float) -> np.ndarray:
    voiced = np.isfinite(midi)
    if voiced.sum() < 3:
        return midi
    idx = np.flatnonzero(voiced)
    vals = midi[idx]
    win = max(3, int(OCTAVE_WINDOW_S * frame_rate) | 1)
    ref = median_filter(vals, size=win, mode="nearest")
    d = vals - ref
    k = np.round(d / 12.0)
    fix = (k != 0) & (np.abs(d - 12.0 * k) < OCTAVE_TOL)
    vals = vals - 12.0 * k * fix
    out = midi.copy()
    out[idx] = vals
    # 구간 단위: 짧은 구간의 중앙값이 앞뒤 구간과 한 옥타브 어긋나면 옮긴다
    runs = voiced_runs(np.isfinite(out))
    meds = [float(np.median(out[a:b])) for a, b in runs]
    for i, (a, b) in enumerate(runs):
        neighbors = [meds[j] for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        if not neighbors or (b - a) > 1.0 * frame_rate:
            continue
        ref_n = float(np.median(neighbors))
        k = round((meds[i] - ref_n) / 12.0)
        if k != 0 and abs(meds[i] - ref_n - 12.0 * k) < OCTAVE_TOL:
            out[a:b] -= 12.0 * k
    return out


def clean_midi(midi: np.ndarray, frame_rate: float, gate: np.ndarray | None = None) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float64).copy()
    if gate is not None:
        midi[~gate] = np.nan
    min_island = max(1, int(round(MIN_ISLAND_S * frame_rate)))
    for a, b in voiced_runs(np.isfinite(midi)):
        if b - a < min_island:
            midi[a:b] = np.nan
    midi = _fix_octaves(midi, frame_rate)
    for a, b in voiced_runs(np.isfinite(midi)):
        seg = midi[a:b]
        if seg.size >= 5:
            med = median_filter(seg, size=5, mode="nearest")
            spike = np.abs(seg - med) > SPIKE_TOL
            seg = np.where(spike, med, seg)
        if seg.size >= 3:
            seg = median_filter(seg, size=3, mode="nearest")
        midi[a:b] = seg
    return midi


def build_curve(f0_hz: np.ndarray, voiced_prob: np.ndarray, rms: np.ndarray | None,
                frame_rate: float, start_time: float = 0.0) -> PitchCurve:
    """작업 프로세스가 낸 원시 결과를 정리해 PitchCurve로 만든다."""
    midi = hz_to_midi(np.asarray(f0_hz, dtype=np.float64))
    gate = energy_gate(rms) if rms is not None and len(rms) == len(midi) else None
    clean = clean_midi(midi, frame_rate, gate)
    conf = np.where(np.isfinite(clean), np.asarray(voiced_prob, dtype=np.float32), 0.0).astype(np.float32)
    f0 = np.where(np.isfinite(clean), midi_to_hz(np.nan_to_num(clean)), np.nan).astype(np.float32)
    return PitchCurve(start_time, frame_rate, f0, conf)
