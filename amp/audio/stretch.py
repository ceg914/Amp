"""음높이를 유지하는 시간 늘리기 (느린 재생용). Qt 비의존.

샘플레이트를 바꾸는 방식은 음높이도 내려가므로 쓰지 않는다. 위상 보코더(STFT → 프레임 간격을 바꿔 다시 합성)에
Laroche·Dolson의 identity phase locking을 더해 위상 흐림(울리는 소리)을 줄였다.

- 시간 대응이 선형이다: 출력 샘플 s ↔ 원본 샘플 s × speed. 그래서 재생 위치를 원본 시간으로 정확히 바꿀 수 있다.
- 출력 길이는 round(입력 길이 / speed)로 맞춘다.
- 여러 트랙·채널을 한 배열로 쌓아 프레임마다 한 번에 계산한다(파이썬 반복은 출력 프레임 수만큼).
- 곡 전체가 아니라 사용자가 고른 구간만 처리하고, 결과는 StretchCache에 둔다.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

SPEEDS = (0.5, 0.75, 0.9, 1.0)
MIN_SPEED, MAX_SPEED = 0.5, 1.0


def _params(sr: int) -> tuple[int, int]:
    n_fft = 2048 if sr <= 48000 else 4096
    return n_fft, n_fft // 4


def time_stretch_many(signals: list[np.ndarray], speed: float, sr: int) -> list[np.ndarray]:
    """같은 길이의 (frames, channels) 배열들을 speed배로 늘린다. speed<1이면 느리고 길어진다."""
    if not signals:
        return []
    speed = float(speed)
    if not (0.25 <= speed <= 4.0):
        raise ValueError("speed는 0.25~4 사이여야 합니다.")
    n = signals[0].shape[0]
    if any(s.shape[0] != n for s in signals):
        raise ValueError("모든 신호의 길이가 같아야 합니다.")
    if abs(speed - 1.0) < 1e-9 or n == 0:
        return [np.array(s, dtype=np.float32, copy=True) for s in signals]
    widths = [s.shape[1] for s in signals]
    x = np.concatenate([np.asarray(s, dtype=np.float32) for s in signals], axis=1).T  # (C, n)
    y = _stretch_rows(x, speed, sr)
    out, k = [], 0
    for w in widths:
        out.append(np.ascontiguousarray(y[k:k + w].T, dtype=np.float32))
        k += w
    return out


def time_stretch(x: np.ndarray, speed: float, sr: int) -> np.ndarray:
    return time_stretch_many([x], speed, sr)[0]


def _stretch_rows(x: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """(C, n) → (C, round(n / speed))."""
    n_fft, hop = _params(sr)
    c, n = x.shape
    out_len = int(round(n / speed))
    pad = n_fft // 2
    mode = "reflect" if n > pad else "constant"
    xp = np.pad(x, ((0, 0), (pad, pad + n_fft)), mode=mode)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float64)  # 주기 hann
    n_frames = 1 + (xp.shape[1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[:, idx] * window, axis=-1)  # (C, T, F)
    n_bins = spec.shape[-1]
    spec = np.concatenate([spec, np.zeros((c, 1, n_bins), dtype=spec.dtype)], axis=1)  # 끝 보간용
    mag_all, ph_all = np.abs(spec), np.angle(spec)
    omega = 2 * np.pi * hop * np.arange(n_bins) / n_fft  # 프레임당 기대 위상 증가
    n_out_frames = int(np.ceil((out_len + pad) / hop)) + 1
    steps = np.minimum(np.arange(n_out_frames) * speed, n_frames - 1)
    out_spec = np.empty((c, n_out_frames, n_bins), dtype=np.complex128)
    phase = ph_all[:, 0, :].copy()
    bins = np.arange(n_bins)
    for t, step in enumerate(steps):
        i = int(step)
        a = step - i
        mag = (1 - a) * mag_all[:, i, :] + a * mag_all[:, i + 1, :]
        if t > 0:
            dphi = ph_all[:, i + 1, :] - ph_all[:, i, :] - omega
            dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
            phase = phase + omega + dphi
        # identity phase locking: 각 봉우리 주변 칸은 봉우리 위상과의 원래 차이를 유지한다
        peaks = np.zeros_like(mag, dtype=bool)
        peaks[:, 1:-1] = (mag[:, 1:-1] > mag[:, :-2]) & (mag[:, 1:-1] >= mag[:, 2:])
        if peaks.any():
            prev = np.where(peaks, bins, -1)
            prev = np.maximum.accumulate(prev, axis=1)
            nxt = np.where(peaks, bins, n_bins * 4)
            nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
            use_prev = (prev >= 0) & ((nxt >= n_bins) | (bins - prev <= nxt - bins))
            owner = np.where(use_prev, prev, np.where(nxt < n_bins, nxt, bins))
            rows = np.arange(c)[:, None]
            ph_in = ph_all[:, i, :]
            phase = phase[rows, owner] + (ph_in - ph_in[rows, owner])
        out_spec[:, t, :] = mag * np.exp(1j * phase)
    frames = np.fft.irfft(out_spec, n=n_fft, axis=-1) * window  # (C, T', n_fft)
    total = (n_out_frames - 1) * hop + n_fft
    y = np.zeros((c, total))
    norm = np.zeros(total)
    w2 = window ** 2
    for t in range(n_out_frames):
        s = t * hop
        y[:, s:s + n_fft] += frames[:, t, :]
        norm[s:s + n_fft] += w2
    y /= np.maximum(norm, 1e-8)
    y = y[:, pad:pad + out_len]
    if y.shape[1] < out_len:
        y = np.pad(y, ((0, 0), (0, out_len - y.shape[1])))
    return y.astype(np.float32)


class StretchCache:
    """(트랙 식별자, 구간, 속도) → 늘린 오디오. 메모리 상한을 넘으면 오래 안 쓴 것부터 버린다(다시 만들 수 있음)."""

    def __init__(self, max_bytes: int = 384 * 2**20) -> None:
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(track_id: str, start_frame: int, end_frame: int, speed: float) -> tuple:
        return (track_id, int(start_frame), int(end_frame), round(float(speed), 4))

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            v = self._items.get(key)
            if v is not None:
                self._items.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return v

    def put(self, key: tuple, value: np.ndarray) -> None:
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = value
            self._bytes += value.nbytes
            while self._bytes > self.max_bytes and len(self._items) > 1:
                _k, v = self._items.popitem(last=False)
                self._bytes -= v.nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def stretch_tracks(self, tracks: dict[str, tuple[str, np.ndarray]], start: int, end: int, speed: float,
                       sr: int) -> dict[str, np.ndarray]:
        """tracks: 이름 → (식별자, (frames, ch) 전체 배열). 구간 [start, end)를 늘린 결과(캐시 사용)."""
        out: dict[str, np.ndarray] = {}
        todo: list[tuple[str, tuple, np.ndarray]] = []
        for name, (tid, arr) in tracks.items():
            k = self.key(tid, start, end, speed)
            hit = self.get(k)
            if hit is not None:
                out[name] = hit
            else:
                todo.append((name, k, arr[start:end]))
        if todo:
            done = time_stretch_many([a for _n, _k, a in todo], speed, sr)
            for (name, k, _a), y in zip(todo, done):
                self.put(k, y)
                out[name] = y
        return out
