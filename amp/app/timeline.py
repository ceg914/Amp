"""파형·눈금자·피아노 롤이 공유하는 가로축(시간) 상태.

가로 스크롤과 확대 배율의 주인은 피아노 롤 뷰다.
다른 위젯은 ``changed`` 시그널을 받아 다시 그리고,
확대·스크롤이 필요하면 ``zoomRequested``/``scrollRequested`` 로 요청만 보낸다.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

MIN_PX_PER_SEC = 4.0
MAX_PX_PER_SEC = 2000.0


class TimelineState(QObject):
    changed = Signal()
    zoomRequested = Signal(float, float)  # 배율, 기준 시간(초)
    scrollRequested = Signal(float)  # 가로 스크롤 이동량(px)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.px_per_sec = 100.0
        self.scroll_x = 0.0  # 화면 왼쪽 끝의 장면 x 좌표

    def update_view(self, px_per_sec: float, scroll_x: float) -> None:
        if px_per_sec != self.px_per_sec or scroll_x != self.scroll_x:
            self.px_per_sec = px_per_sec
            self.scroll_x = scroll_x
            self.changed.emit()

    def time_to_x(self, t: float) -> float:
        return t * self.px_per_sec - self.scroll_x

    def x_to_time(self, x: float) -> float:
        return (x + self.scroll_x) / self.px_per_sec
