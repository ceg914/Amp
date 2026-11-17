"""재생 장치가 요구하는 형식으로 PCM 바이트를 만든다.

장치가 원본 샘플레이트·채널 수를 지원하지 않을 수 있어서
출력 프레임 번호 기준으로 필요한 부분만 선형 보간·채널 변환·양자화한다.
상태가 없으므로 같은 위치를 다시 읽어도 같은 결과가 나온다(탐색이 쉽다).
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class SampleFormat(Enum):
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"

    @property
    def bytes_per_sample(self) -> int:
        return 2 if self is SampleFormat.INT16 else 4


class PcmConverter:
    def __init__(
        self,
        samples: np.ndarray,
        in_rate: int,
        out_rate: int,
        out_channels: int,
        sample_format: SampleFormat = SampleFormat.INT16,
        gain: float = 1.0,
    ) -> None:
        if samples.ndim != 2:
            raise ValueError("samples는 (frames, channels) 모양이어야 합니다.")
        if in_rate <= 0 or out_rate <= 0 or out_channels <= 0:
            raise ValueError("샘플레이트와 채널 수는 0보다 커야 합니다.")
        self.samples = samples
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.out_channels = int(out_channels)
        self.sample_format = sample_format
        self.gain = float(gain)
        self._ratio = self.in_rate / self.out_rate
        self.total_out_frames = int(np.floor((samples.shape[0] - 1) / self._ratio)) + 1 if samples.shape[0] else 0

    @property
    def bytes_per_frame(self) -> int:
        return self.out_channels * self.sample_format.bytes_per_sample

    def out_frame_at(self, seconds: float) -> int:
        return int(min(max(round(seconds * self.out_rate), 0), self.total_out_frames))

    def seconds_at(self, out_frame: int) -> float:
        return out_frame / self.out_rate

    def read(self, start: int, count: int) -> bytes:
        """출력 프레임 [start, start+count) 를 바이트로. 끝을 넘으면 짧게 돌려준다."""
        start = max(0, int(start))
        count = max(0, min(int(count), self.total_out_frames - start))
        if count == 0:
            return b""
        block = self._resample(start, count)
        block = self._map_channels(block)
        if self.gain != 1.0:
            block = block * self.gain
        return self._quantize(block).tobytes()

    def _resample(self, start: int, count: int) -> np.ndarray:
        src = self.samples
        if self.in_rate == self.out_rate:
            return src[start : start + count].astype(np.float32, copy=False)
        pos = (start + np.arange(count)) * self._ratio
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, src.shape[0] - 1)
        frac = (pos - i0).astype(np.float32)[:, None]
        return src[i0] * (1.0 - frac) + src[i1] * frac

    def _map_channels(self, block: np.ndarray) -> np.ndarray:
        in_ch = block.shape[1]
        out_ch = self.out_channels
        if in_ch == out_ch:
            return block
        if in_ch == 1:
            return np.repeat(block, out_ch, axis=1)
        if out_ch == 1:
            return block.mean(axis=1, keepdims=True)
        out = np.zeros((block.shape[0], out_ch), dtype=np.float32)
        n = min(in_ch, out_ch)
        out[:, :n] = block[:, :n]
        return out

    def _quantize(self, block: np.ndarray) -> np.ndarray:
        block = np.clip(block, -1.0, 1.0)
        if self.sample_format is SampleFormat.INT16:
            return (block * 32767.0).astype("<i2")
        if self.sample_format is SampleFormat.INT32:
            return (block.astype(np.float64) * 2147483647.0).astype("<i4")
        return block.astype("<f4")
