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


class PcmMixer:
    """여러 트랙(원본·보컬·반주)을 트랙별 음량·음소거·솔로에 따라 섞어 PCM으로 만든다.

    - 솔로가 하나라도 켜져 있으면 솔로 트랙만 들린다. 아니면 음소거하지 않은 트랙이 모두 들린다.
    - 상태는 재생 중에도 바꿀 수 있다. 오디오 스레드는 블록을 읽을 때마다 현재 상태를 쓴다.
    - 모든 트랙은 같은 샘플레이트·프레임 수여야 한다(분리 결과가 원본과 같은 시간축을 쓰므로 성립).
    - 음량·음소거·솔로가 바뀌면 트랙마다 RAMP_S 동안 음량을 직선으로 옮긴다. 모델 A/B 전환(솔로 A → 솔로 B)은
      A가 줄고 B가 커지는 짧은 크로스페이드가 되어 딸깍 소리가 나지 않는다. 재생 위치는 그대로다.
    """

    RAMP_S = 0.012

    def __init__(
        self,
        tracks: dict[str, np.ndarray],
        in_rate: int,
        out_rate: int,
        out_channels: int,
        sample_format: SampleFormat = SampleFormat.INT16,
    ) -> None:
        if not tracks:
            raise ValueError("트랙이 하나 이상 필요합니다.")
        lengths = {t.shape[0] for t in tracks.values()}
        if len(lengths) != 1:
            raise ValueError("모든 트랙의 길이가 같아야 합니다.")
        self.in_rate = int(in_rate)
        self._converters = {
            name: PcmConverter(samples, in_rate, out_rate, out_channels, sample_format)
            for name, samples in tracks.items()
        }
        first = next(iter(self._converters.values()))
        self.out_rate = first.out_rate
        self.out_channels = first.out_channels
        self.sample_format = sample_format
        self.total_out_frames = first.total_out_frames
        self.state = {name: {"gain": 1.0, "mute": False, "solo": False} for name in tracks}
        self._applied: dict[str, float] | None = None  # 마지막 블록 끝의 실제 음량 (처음 읽을 때 목표값으로 시작)

    @property
    def bytes_per_frame(self) -> int:
        return self.out_channels * self.sample_format.bytes_per_sample

    def out_frame_at(self, seconds: float) -> int:
        return next(iter(self._converters.values())).out_frame_at(seconds)

    def seconds_at(self, out_frame: int) -> float:
        return out_frame / self.out_rate

    def set_state(self, name: str, *, gain: float | None = None, mute: bool | None = None, solo: bool | None = None) -> None:
        if name not in self.state:
            return
        s = dict(self.state[name])
        if gain is not None:
            s["gain"] = float(max(0.0, gain))
        if mute is not None:
            s["mute"] = bool(mute)
        if solo is not None:
            s["solo"] = bool(solo)
        self.state = {**self.state, name: s}  # 통째로 바꿔 오디오 스레드가 반쯤 바뀐 상태를 보지 않게 한다

    def set_states(self, changes: dict[str, dict]) -> None:
        """여러 트랙 상태를 한 번에 바꾼다 (A/B 전환: 오디오 스레드가 '둘 다 솔로 아님' 같은 중간 상태를 보지 않게)."""
        new = dict(self.state)
        for name, ch in changes.items():
            if name not in new:
                continue
            s = dict(new[name])
            if ch.get("gain") is not None:
                s["gain"] = float(max(0.0, ch["gain"]))
            for k in ("mute", "solo"):
                if ch.get(k) is not None:
                    s[k] = bool(ch[k])
            new[name] = s
        self.state = new

    def add_track(self, name: str, samples: np.ndarray, state: dict | None = None) -> bool:
        """재생 중에 트랙을 더한다 (v0.6.5 악기 듣기). 길이가 다르면 더하지 않고 False.

        목록을 통째로 바꿔 오디오 스레드가 반쯤 바뀐 상태를 보지 않게 한다. 새 트랙은 음량 0에서 RAMP_S 동안 올라온다."""
        first = next(iter(self._converters.values()))
        if samples.shape[0] != first.samples.shape[0]:
            return False
        conv = PcmConverter(samples, self.in_rate, first.out_rate, first.out_channels, self.sample_format)
        st = {"gain": 1.0, "mute": False, "solo": False, **(state or {})}
        self.state = {**self.state, name: {"gain": float(st["gain"]), "mute": bool(st["mute"]), "solo": bool(st["solo"])}}
        self._converters = {**self._converters, name: conv}
        return True

    def remove_track(self, name: str) -> None:
        """재생 중에 트랙을 뺀다. 마지막 트랙은 빼지 않는다 (호출한 쪽이 먼저 음량을 0으로 내려 두면 딸깍 소리가 없다)."""
        if name in self._converters and len(self._converters) > 1:
            self._converters = {k: v for k, v in self._converters.items() if k != name}
            self.state = {k: v for k, v in self.state.items() if k != name}

    def audible(self) -> list[tuple[str, float]]:
        state = self.state
        solo = [n for n, s in state.items() if s["solo"]]
        names = solo if solo else [n for n, s in state.items() if not s["mute"]]
        return [(n, state[n]["gain"]) for n in names if state[n]["gain"] > 0]

    def targets(self, converters: dict | None = None) -> dict[str, float]:
        """트랙별 목표 음량 (들리지 않으면 0)."""
        out = dict.fromkeys(self._converters if converters is None else converters, 0.0)
        out.update((n, g) for n, g in self.audible() if n in out)
        return out

    def read_float(self, start: int, count: int) -> np.ndarray:
        """출력 프레임 [start, start+count)를 섞은 float32 (count', out_channels). 끝을 넘으면 짧다."""
        start = max(0, int(start))
        count = max(0, min(int(count), self.total_out_frames - start))
        mix = np.zeros((count, self.out_channels), dtype=np.float32)
        if count == 0:
            return mix
        converters = self._converters  # 이 블록 동안 쓸 목록 (재생 중 트랙을 더하거나 빼도 안전)
        targets = self.targets(converters)
        first_block = self._applied is None
        applied = dict(targets) if first_block else self._applied
        ramp = max(1, int(self.RAMP_S * self.out_rate))
        done: dict[str, float] = {}
        for name, conv in converters.items():
            g1 = targets[name]
            g0 = applied.get(name, g1 if first_block else 0.0)  # 재생 중에 더한 트랙은 0에서 올라온다
            if abs(g1 - g0) < 1e-6:
                done[name] = g1
                if g1 > 0:
                    mix += conv._map_channels(conv._resample(start, count)) * g1
                continue
            step = (g1 - g0) / ramp  # 블록이 짧으면 다음 블록에서 이어서 옮긴다
            env = g0 + step * np.arange(1, count + 1, dtype=np.float32)
            env = np.minimum(env, g1) if g1 > g0 else np.maximum(env, g1)
            mix += conv._map_channels(conv._resample(start, count)) * env[:, None].astype(np.float32)
            done[name] = float(env[-1])
        self._applied = done
        return mix

    def quantize(self, block: np.ndarray) -> bytes:
        return self._converters[next(iter(self._converters))]._quantize(block).tobytes()

    def read(self, start: int, count: int) -> bytes:
        mix = self.read_float(start, count)
        return self.quantize(mix) if mix.shape[0] else b""
