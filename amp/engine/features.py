"""가벼운 신호 특징 (numpy만 사용)."""

from __future__ import annotations

import numpy as np

NO_VOCALS_RATIO_DB = -25.0  # 보컬 에너지가 믹스보다 이만큼 이상 작으면 보컬이 없다고 본다
NO_VOCALS_ACTIVE = 0.03  # 보컬이 들리는 구간이 전체의 3% 미만이면 보컬이 없다고 본다


def _frame_rms(x: np.ndarray, sr: int, frame_s: float = 0.05) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    k = len(x) // n
    if k == 0:
        return np.array([float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))]) if len(x) else np.zeros(1)
    blocks = x[: k * n].astype(np.float64).reshape(k, n)
    return np.sqrt((blocks ** 2).mean(axis=1))


def vocal_presence(vocals: np.ndarray, mix: np.ndarray, sr: int) -> dict:
    """분리된 보컬이 실제로 있는지 대략 판단한다. (모노 배열 입력)"""
    mix_e = float(np.mean(mix.astype(np.float64) ** 2))
    voc_e = float(np.mean(vocals.astype(np.float64) ** 2))
    ratio_db = 10 * np.log10((voc_e + 1e-12) / (mix_e + 1e-12))
    vr = _frame_rms(vocals, sr)
    mr = _frame_rms(mix, sr)
    ref = float(mr.max()) if mr.size else 0.0
    active = float(np.mean(vr > ref * 10 ** (-30 / 20))) if ref > 0 else 0.0
    present = ratio_db > NO_VOCALS_RATIO_DB and active >= NO_VOCALS_ACTIVE
    return {"ratio_db": round(float(ratio_db), 1), "active_fraction": round(active, 3), "present": bool(present)}
