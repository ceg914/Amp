"""비교 청취용 음량 맞춤. 분리 WAV 파일은 바꾸지 않고, 재생할 때 곱할 음량만 계산한다. Qt 비의존.

음량 = 0.4초 블록 RMS를 무음 블록을 빼고 평균한 dBFS. 절대 기준 ABS_GATE_DB보다 작은 블록과, 가장 큰 블록보다
REL_GATE_DB 넘게 작은 블록은 뺀다. 방송용 LUFS(K-가중)는 아니다. 두 모델의 크기를 비슷하게 맞춰 '큰 쪽이 좋게
들리는' 편향을 줄이려는 것이고, 소리를 키우지 않고 큰 쪽을 작은 쪽에 맞춰 줄이므로 잘림(클리핑)이 생기지 않는다.
"""

from __future__ import annotations

import numpy as np

BLOCK_S = 0.4
ABS_GATE_DB = -60.0
REL_GATE_DB = -20.0


def gated_level_db(samples: np.ndarray, sr: int, start: float | None = None, end: float | None = None) -> float | None:
    """(frames, ch) 또는 (frames,) → dBFS. 소리가 거의 없으면 None."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    a = 0 if start is None else max(0, int(start * sr))
    b = x.shape[0] if end is None else min(x.shape[0], int(end * sr))
    x = x[a:b]
    n = max(1, int(BLOCK_S * sr))
    k = x.shape[0] // n
    if k == 0:
        if x.size == 0:
            return None
        blocks = np.array([np.mean(x.astype(np.float64) ** 2)])
    else:
        blocks = np.mean(x[:k * n].astype(np.float64).reshape(k, n) ** 2, axis=1)
    db = 10 * np.log10(blocks + 1e-20)
    keep = db > ABS_GATE_DB
    if not keep.any():
        return None
    keep &= db > db[keep].max() + REL_GATE_DB
    return float(10 * np.log10(np.mean(blocks[keep])))


def matching_gains(level_a: float | None, level_b: float | None) -> tuple[float, float]:
    """(A 배율, B 배율). 큰 쪽을 작은 쪽에 맞춰 줄인다 (1을 넘지 않음). 한쪽을 모르면 (1, 1)."""
    if level_a is None or level_b is None:
        return 1.0, 1.0
    target = min(level_a, level_b)
    return float(10 ** ((target - level_a) / 20)), float(10 ** ((target - level_b) / 20))
