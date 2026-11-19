"""메모리에 올린 오디오 데이터."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioData:
    """``samples`` 는 (프레임 수, 채널 수) 모양의 float32 배열, 값 범위 -1~1."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        s = np.asarray(self.samples, dtype=np.float32)
        if s.ndim == 1:
            s = s[:, None]
        if s.ndim != 2 or s.shape[1] < 1:
            raise ValueError("samples는 (frames, channels) 모양이어야 합니다.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate는 0보다 커야 합니다.")
        s.flags.writeable = False  # 여러 스레드가 함께 읽으므로 실수로 바꾸지 못하게 한다
        object.__setattr__(self, "samples", s)

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def mono(self) -> np.ndarray:
        if self.channels == 1:
            return self.samples[:, 0]
        # mean(axis=1)은 행마다 줄이는 방식이라 느리다(5분 스테레오 약 0.22초). 열을 차례로 더해 나누면 값은 같고 약 7배 빠르다
        out = self.samples[:, 0].astype(np.float32, copy=True)
        for c in range(1, self.channels):
            out += self.samples[:, c]
        out /= np.float32(self.channels)
        return out
