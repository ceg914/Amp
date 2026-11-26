"""악기 분리 결과 비교 (v0.7.0 명세 15.2). Qt 비의존.

보컬 유사 성분(근사 지표): 1초 구간마다 악기 = g·기준 보컬 + 나머지로 최소제곱 분해해, 보컬이 있는 구간에서
g²·|보컬|² / |악기|² 를 dB로 낸다. 정답(깨끗한 악기 음원)과 비교한 값이 아니다.
- 기준 보컬이 실제 보컬과 다르면(분리 잔여물) 값도 달라진다
- 값이 낮아졌다고 악기 소리가 더 잘 보존되었다는 뜻은 아니다 (악기 소리가 함께 줄었을 수도 있음)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOTE = ("근사 지표: 기준 보컬과 닮은 성분의 비중입니다. 정답 음원과 비교한 값이 아니며, 낮다고 악기 소리 보존이 "
        "더 좋다는 뜻은 아닙니다.")
ACTIVE_DB = 30.0  # 가장 큰 1초 구간보다 이만큼 안쪽인 구간만 '보컬이 있는 구간'


@dataclass(frozen=True)
class VocalShare:
    share_db: float | None  # 보컬 유사 성분 비중 (dB). 보컬 있는 구간이 없으면 None
    level_dbfs: float | None  # 보컬 있는 구간에서 악기 음량
    seconds: int  # 계산에 쓴 1초 구간 수
    note: str = NOTE

    def to_dict(self) -> dict:
        return {"vocal_share_db": self.share_db, "level_dbfs": self.level_dbfs, "seconds": self.seconds, "note": self.note}


def _mono(path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def vocal_share(stem: np.ndarray, vocals: np.ndarray, sr: int, window_s: float = 1.0) -> VocalShare:
    w = max(1, int(sr * window_s))
    n = min(len(stem), len(vocals)) // w
    if n == 0:
        return VocalShare(None, None, 0)
    v = vocals[:n * w].reshape(n, w).astype(np.float64)
    x = stem[:n * w].reshape(n, w).astype(np.float64)
    vv = np.einsum("ij,ij->i", v, v)
    xx = np.einsum("ij,ij->i", x, x)
    xv = np.einsum("ij,ij->i", x, v)
    rms = np.sqrt(vv / w)
    if not np.any(rms > 0):
        return VocalShare(None, None, 0)
    active = (20 * np.log10(rms + 1e-12) > 20 * np.log10(rms.max() + 1e-12) - ACTIVE_DB) & (vv > 0)
    if not active.any():
        return VocalShare(None, None, 0)
    g = xv[active] / vv[active]
    leak = float(np.sum(g * g * vv[active]))
    tot = float(np.sum(xx[active]))
    if tot <= 0:
        return VocalShare(None, None, int(active.sum()))
    share = 10 * np.log10(leak / tot + 1e-12)
    level = 10 * np.log10(tot / (active.sum() * w) + 1e-12)
    return VocalShare(round(float(share), 1), round(float(level), 1), int(active.sum()))


def vocal_share_files(stem_path, vocals_path) -> VocalShare:
    x, sr = _mono(stem_path)
    v, sr2 = _mono(vocals_path)
    if sr2 != sr:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, sr2)
        v = resample_poly(v, sr // g, sr2 // g).astype(np.float32)
    return vocal_share(x, v, sr)
