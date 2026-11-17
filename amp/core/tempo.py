"""템포 맵: 초 단위 시간과 박(beat) 위치를 서로 변환한다.

음표 시간은 항상 초 단위로 저장한다. 템포 맵은 격자 표시, 스냅,
이후 MIDI 내보내기에서 초 ↔ 박 변환에만 쓴다.
템포가 바뀌는 곡을 위해 구간별 고정 템포(piecewise constant)를 지원한다.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from amp.core.errors import EditError

MIN_BPM = 20.0
MAX_BPM = 400.0


@dataclass(frozen=True)
class TempoPoint:
    """``time`` 초부터 ``bpm`` 템포가 적용된다."""

    time: float
    bpm: float


@dataclass(frozen=True)
class TempoMap:
    """구간별 템포. ``points[0].time`` 이 0번째 박의 위치다."""

    points: tuple[TempoPoint, ...]
    beats_per_bar: int = 4
    beat_unit: int = 4
    _beats: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _times: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise EditError("템포 맵에는 최소 한 개의 템포 지점이 필요합니다.")
        pts = tuple(sorted(self.points, key=lambda p: p.time))
        for p in pts:
            if not (MIN_BPM <= p.bpm <= MAX_BPM) or not math.isfinite(p.bpm):
                raise EditError(f"BPM은 {MIN_BPM:g}~{MAX_BPM:g} 사이여야 합니다: {p.bpm}")
        if self.beats_per_bar < 1:
            raise EditError("마디당 박 수는 1 이상이어야 합니다.")
        beats = [0.0]
        for prev, cur in zip(pts, pts[1:]):
            beats.append(beats[-1] + (cur.time - prev.time) * prev.bpm / 60.0)
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "_beats", tuple(beats))
        object.__setattr__(self, "_times", tuple(p.time for p in pts))

    @classmethod
    def constant(cls, bpm: float, first_beat: float = 0.0, beats_per_bar: int = 4) -> "TempoMap":
        return cls((TempoPoint(first_beat, bpm),), beats_per_bar=beats_per_bar)

    @property
    def first_beat_time(self) -> float:
        return self.points[0].time

    @property
    def initial_bpm(self) -> float:
        return self.points[0].bpm

    def with_initial_bpm(self, bpm: float) -> "TempoMap":
        """첫 구간 템포만 바꾼 새 템포 맵 (툴바 BPM 입력용)."""
        pts = (TempoPoint(self.points[0].time, bpm),) + self.points[1:]
        return TempoMap(pts, self.beats_per_bar, self.beat_unit)

    def beat_at(self, t: float) -> float:
        """시간 t(초)의 박 위치. 첫 박 이전은 음수."""
        k = max(0, bisect.bisect_right(self._times, t) - 1)
        p = self.points[k]
        return self._beats[k] + (t - p.time) * p.bpm / 60.0

    def time_at(self, beat: float) -> float:
        """박 위치를 시간(초)으로."""
        k = max(0, bisect.bisect_right(self._beats, beat) - 1)
        p = self.points[k]
        return p.time + (beat - self._beats[k]) * 60.0 / p.bpm

    def seconds_per_beat_at(self, t: float) -> float:
        k = max(0, bisect.bisect_right(self._times, t) - 1)
        return 60.0 / self.points[k].bpm

    def snap(self, t: float, step_beats: float) -> float:
        """가장 가까운 격자 위치로 맞춘다. step_beats <= 0 이면 그대로 둔다."""
        if step_beats <= 0:
            return t
        b = self.beat_at(t)
        return self.time_at(round(b / step_beats) * step_beats)

    def grid_beats(self, t0: float, t1: float, step_beats: float) -> list[float]:
        """[t0, t1] 구간에 들어가는 격자 박 위치 목록."""
        if step_beats <= 0 or t1 < t0:
            return []
        b0 = math.ceil(self.beat_at(t0) / step_beats - 1e-9)
        b1 = math.floor(self.beat_at(t1) / step_beats + 1e-9)
        return [k * step_beats for k in range(b0, b1 + 1)]

    def bar_beat(self, t: float) -> tuple[int, float]:
        """(마디 번호(1부터), 마디 안 박 위치(1부터)) — 화면 표시용."""
        b = self.beat_at(t)
        bar = math.floor(b / self.beats_per_bar)
        return bar + 1, b - bar * self.beats_per_bar + 1.0
