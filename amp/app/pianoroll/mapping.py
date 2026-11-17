"""피아노 롤 좌표 변환 (시간·음높이 ↔ 장면 좌표). Qt 비의존."""

from __future__ import annotations

import math

PITCH_MIN = 0
PITCH_MAX = 127
MIN_ROW_HEIGHT = 6.0
MAX_ROW_HEIGHT = 40.0


class PianoRollMapper:
    """x = 시간 × px_per_sec, y = (PITCH_MAX - 음높이) × row_height.

    높은 음일수록 위(작은 y)에 그린다. 정수 음높이 p는 [y(p), y(p)+row_height) 행을 차지한다.
    """

    def __init__(self, px_per_sec: float = 100.0, row_height: float = 14.0) -> None:
        self.px_per_sec = px_per_sec
        self.row_height = row_height

    @property
    def scene_height(self) -> float:
        return (PITCH_MAX - PITCH_MIN + 1) * self.row_height

    def time_to_x(self, t: float) -> float:
        return t * self.px_per_sec

    def x_to_time(self, x: float) -> float:
        return x / self.px_per_sec

    def pitch_to_y(self, pitch: float) -> float:
        """정수 음높이 행의 위쪽 y."""
        return (PITCH_MAX - pitch) * self.row_height

    def pitch_center_y(self, pitch: float) -> float:
        """실수 음높이(피치 곡선)의 y. 정수 음높이는 행 가운데에 온다."""
        return (PITCH_MAX - pitch) * self.row_height + self.row_height / 2

    def y_to_pitch(self, y: float) -> int:
        p = PITCH_MAX - math.floor(y / self.row_height)
        return max(PITCH_MIN, min(PITCH_MAX, p))
