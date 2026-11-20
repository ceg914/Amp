"""스펙트로그램 타일 계산과 표시 매핑. Qt 비의존.

- STFT(hann, center): 프레임 k의 중심 시각 = k × hop / sr. 표시할 때 프레임 k는 [t_k − hop/2, t_k + hop/2]를 덮는다.
  그래서 파형·피치 곡선·음표와 반 프레임 넘게 어긋나지 않는다(시험으로 확인).
- 확대 수준에 맞는 hop을 고른다(화면 1px에 1~2프레임). 곡 전체를 고해상도로 한 번에 만들지 않는다.
- 시간축을 TILE_FRAMES 프레임 단위 타일로 나눠 필요한 타일만 계산하고 캐시한다(크기 상한, 다시 만들 수 있음).
- 세기는 dBFS(가득 찬 사인파 = 0dB)로 고정한다. 타일마다 자동 범위를 쓰지 않으므로 트랙·모델끼리 같은 기준으로 비교된다.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

TILE_FRAMES = 256
RESOLUTIONS = {"time": 1024, "balanced": 2048, "frequency": 4096}  # 시간 우선 / 균형 / 주파수 우선 (n_fft)
RESOLUTION_LABELS = {"time": "시간 우선 (1024)", "balanced": "균형 (2048)", "frequency": "주파수 우선 (4096)"}


@dataclass(frozen=True)
class SpecParams:
    n_fft: int = 2048
    hop: int = 512

    @property
    def window(self) -> np.ndarray:
        return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)).astype(np.float32)


def choose_hop(n_fft: int, sr: int, px_per_sec: float) -> int:
    """화면 1px에 프레임이 1개 이상 되도록 hop을 n_fft/4 × 2^k에서 고른다 (너무 촘촘하면 계산 낭비)."""
    base = max(1, n_fft // 4)
    hop = base
    target = sr / max(px_per_sec, 1e-3)  # 1px에 해당하는 샘플 수
    while hop * 2 <= target:
        hop *= 2
    return hop


def frame_time(k: int | np.ndarray, hop: int, sr: int):
    return np.asarray(k) * hop / sr


def compute_tile(mono: np.ndarray, sr: int, params: SpecParams, tile: int) -> np.ndarray:
    """dBFS (bins, TILE_FRAMES) float16. 신호 밖 프레임은 −200dB."""
    n_fft, hop = params.n_fft, params.hop
    k0 = tile * TILE_FRAMES
    centers = (k0 + np.arange(TILE_FRAMES)) * hop
    starts = centers - n_fft // 2
    n = mono.shape[0]
    out = np.full((n_fft // 2 + 1, TILE_FRAMES), -200.0, np.float32)
    valid = centers < n + n_fft // 2
    if not valid.any():
        return out.astype(np.float16)
    idx = starts[valid][:, None] + np.arange(n_fft)[None, :]
    frames = np.where((idx >= 0) & (idx < n), mono[np.clip(idx, 0, n - 1)], 0.0).astype(np.float32)
    w = params.window
    spec = np.abs(np.fft.rfft(frames * w, axis=1))
    ref = w.sum() / 2.0  # 진폭 1 사인파의 봉우리 크기
    db = 20.0 * np.log10(np.maximum(spec / ref, 1e-10))
    out[:, valid] = db.T
    return out.astype(np.float16)


def bin_hz(params: SpecParams, sr: int) -> np.ndarray:
    return np.arange(params.n_fft // 2 + 1) * sr / params.n_fft


def row_frequencies(rows: int, fmin: float, fmax: float, log_axis: bool) -> np.ndarray:
    """화면 행(위 = 0) → 주파수. 로그 축이면 같은 높이 = 같은 음정 간격."""
    y = (np.arange(rows) + 0.5) / max(rows, 1)  # 0 = 위
    frac = 1.0 - y
    if log_axis:
        return fmin * (fmax / fmin) ** frac
    return fmin + (fmax - fmin) * frac


def freq_to_y(hz, rows: int, fmin: float, fmax: float, log_axis: bool):
    hz = np.asarray(hz, dtype=float)
    if log_axis:
        frac = np.log(np.maximum(hz, 1e-6) / fmin) / np.log(fmax / fmin)
    else:
        frac = (hz - fmin) / (fmax - fmin)
    return (1.0 - frac) * rows


def rows_from_bins(db: np.ndarray, params: SpecParams, sr: int, rows: int, fmin: float, fmax: float,
                   log_axis: bool) -> np.ndarray:
    """(bins, frames) → (rows, frames). 행 하나가 여러 칸을 덮으면 최댓값(가는 배음이 사라지지 않게), 아니면 보간."""
    hz = row_frequencies(rows + 1, fmin, fmax, log_axis)  # 행 경계
    edges = hz * params.n_fft / sr
    centers = 0.5 * (edges[:-1] + edges[1:])
    lo = np.floor(np.minimum(edges[:-1], edges[1:])).astype(int)
    hi = np.ceil(np.maximum(edges[:-1], edges[1:])).astype(int)
    nb = db.shape[0]
    out = np.empty((rows, db.shape[1]), np.float32)
    dbf = db.astype(np.float32)
    for r in range(rows):
        a, b = max(lo[r], 0), min(hi[r], nb - 1)
        if b - a >= 2:
            out[r] = dbf[a:b + 1].max(axis=0)
        else:
            c = min(max(centers[r], 0.0), nb - 1.0)
            i = int(c)
            j = min(i + 1, nb - 1)
            f = c - i
            out[r] = dbf[i] * (1 - f) + dbf[j] * f
    return out


# Obsidian Violet 위 색 단계: 바탕 → 경계선 → 바이올렛 → 민트 → 밝은 글자색 (밝기가 계속 커진다)
_STOPS = [(0.0, (0x12, 0x11, 0x16)), (0.3, (0x3B, 0x35, 0x46)), (0.62, (0xAD, 0x96, 0xF5)),
          (0.85, (0x66, 0xC8, 0xBE)), (1.0, (0xF0, 0xEC, 0xF6))]


def colormap(n: int = 256) -> np.ndarray:
    """(n, 3) uint8."""
    x = np.linspace(0, 1, n)
    out = np.zeros((n, 3))
    for (x0, c0), (x1, c1) in zip(_STOPS[:-1], _STOPS[1:]):
        m = (x >= x0) & (x <= x1)
        f = ((x[m] - x0) / (x1 - x0))[:, None]
        out[m] = np.array(c0) * (1 - f) + np.array(c1) * f
    return out.round().astype(np.uint8)


def to_indices(db_rows: np.ndarray, db_floor: float, db_ceil: float) -> np.ndarray:
    """dB → 색 번호 0~255 (같은 기준이면 같은 색)."""
    t = (db_rows - db_floor) / max(db_ceil - db_floor, 1e-6)
    return (np.clip(t, 0.0, 1.0) * 255).astype(np.uint8)


class TileCache:
    """(트랙 id, n_fft, hop, 타일 번호) → dB 타일. 바이트 상한을 넘으면 오래 안 쓴 것부터 버린다."""

    def __init__(self, max_bytes: int = 96 * 2**20) -> None:
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            v = self._items.get(key)
            if v is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key: tuple, tile: np.ndarray) -> None:
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = tile
            self._bytes += tile.nbytes
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

    def __len__(self) -> int:
        return len(self._items)


def tiles_for_range(t0: float, t1: float, hop: int, sr: int) -> range:
    k0 = max(0, int(np.floor(t0 * sr / hop)) - 1)
    k1 = int(np.ceil(t1 * sr / hop)) + 1
    return range(k0 // TILE_FRAMES, k1 // TILE_FRAMES + 1)
