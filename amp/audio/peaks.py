"""여러 해상도의 파형 최솟값/최댓값 요약.

긴 오디오를 화면에 그릴 때 원본 샘플을 매번 훑지 않도록,
불러올 때 한 번 버킷별 min/max를 계산해 둔다.
화면을 그릴 때는 확대 배율에 맞는 레벨을 골라 픽셀 열마다 다시 줄인다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASE_BUCKET = 32  # 가장 세밀한 레벨에서 버킷 하나가 묶는 샘플 수
LEVEL_FACTOR = 4
MIN_BUCKETS = 64  # 버킷이 이보다 적어지면 더 거친 레벨을 만들지 않는다


@dataclass(frozen=True)
class PeakLevel:
    bucket: int  # 버킷 하나의 샘플 수
    mins: np.ndarray
    maxs: np.ndarray


class WaveformPeaks:
    def __init__(self, sample_rate: int, frames: int, levels: list[PeakLevel]) -> None:
        if not levels:
            raise ValueError("레벨이 하나 이상 필요합니다.")
        self.sample_rate = sample_rate
        self.frames = frames
        self.levels = levels

    @classmethod
    def build(cls, mono: np.ndarray, sample_rate: int) -> "WaveformPeaks":
        mono = np.asarray(mono, dtype=np.float32)
        n = mono.shape[0]
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return cls(sample_rate, 0, [PeakLevel(BASE_BUCKET, empty, empty)])
        n_buckets = -(-n // BASE_BUCKET)
        padded = np.empty(n_buckets * BASE_BUCKET, dtype=np.float32)
        padded[:n] = mono
        padded[n:] = mono[-1]  # 마지막 값으로 채워 min/max가 왜곡되지 않게 한다
        blocks = padded.reshape(n_buckets, BASE_BUCKET)
        levels = [PeakLevel(BASE_BUCKET, blocks.min(axis=1), blocks.max(axis=1))]
        while levels[-1].mins.shape[0] // LEVEL_FACTOR >= MIN_BUCKETS:
            prev = levels[-1]
            m = prev.mins.shape[0]
            k = -(-m // LEVEL_FACTOR)
            pad = k * LEVEL_FACTOR - m
            mins = np.concatenate([prev.mins, np.repeat(prev.mins[-1:], pad)]).reshape(k, LEVEL_FACTOR)
            maxs = np.concatenate([prev.maxs, np.repeat(prev.maxs[-1:], pad)]).reshape(k, LEVEL_FACTOR)
            levels.append(PeakLevel(prev.bucket * LEVEL_FACTOR, mins.min(axis=1), maxs.max(axis=1)))
        return cls(sample_rate, n, levels)

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    def level_for(self, samples_per_pixel: float) -> PeakLevel:
        """픽셀당 샘플 수보다 작거나 같은 버킷 중 가장 거친 레벨."""
        chosen = self.levels[0]
        for lv in self.levels:
            if lv.bucket <= samples_per_pixel:
                chosen = lv
            else:
                break
        return chosen

    def columns(self, t0: float, t1: float, width: int) -> tuple[np.ndarray, np.ndarray]:
        """[t0, t1) 을 width 개 픽셀 열로 나눴을 때 열마다 (min, max).

        오디오 범위 밖의 열은 NaN이다.
        """
        width = int(width)
        mins = np.full(width, np.nan, dtype=np.float32)
        maxs = np.full(width, np.nan, dtype=np.float32)
        if width <= 0 or t1 <= t0 or self.frames == 0:
            return mins, maxs
        spp = (t1 - t0) * self.sample_rate / width
        lv = self.level_for(spp)
        n_b = lv.mins.shape[0]
        edges = (t0 * self.sample_rate + np.arange(width + 1) * spp) / lv.bucket
        starts = np.floor(edges[:-1]).astype(np.int64)
        ends = np.maximum(np.ceil(edges[1:]).astype(np.int64), starts + 1)
        valid = (ends > 0) & (starts < n_b) & (starts * lv.bucket < self.frames)
        if not valid.any():
            return mins, maxs
        s = np.clip(starts[valid], 0, n_b - 1)
        last_in = np.clip(ends[valid], 1, n_b) - 1  # 열이 걸치는 마지막 버킷
        # reduceat은 [s_i, s_{i+1}) 을 줄인다. 다음 열과 반쯤 겹치는 버킷(last_in)은 따로 더한다.
        col_min = np.minimum(np.minimum.reduceat(lv.mins, s), lv.mins[last_in])
        col_max = np.maximum(np.maximum.reduceat(lv.maxs, s), lv.maxs[last_in])
        # 마지막 열은 reduceat이 배열 끝까지 줄이므로 직접 계산한다
        a, b = int(s[-1]), int(last_in[-1]) + 1
        col_min[-1] = lv.mins[a:b].min()
        col_max[-1] = lv.maxs[a:b].max()
        mins[valid] = col_min
        maxs[valid] = col_max
        return mins, maxs
