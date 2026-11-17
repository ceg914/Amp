"""시간 눈금자: 마디 번호와 초 단위 눈금. 클릭하면 재생 위치를 옮긴다."""

from __future__ import annotations

import math

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from amp.app import theme
from amp.app.timeline import TimelineState
from amp.core.tempo import TempoMap

_SECOND_STEPS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300)


class TimeRuler(QWidget):
    seekRequested = Signal(float)

    def __init__(self, timeline: TimelineState, tempo_getter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = timeline
        self._tempo_getter = tempo_getter  # () -> TempoMap
        self._playhead = 0.0
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        timeline.changed.connect(self.update)

    def set_playhead(self, t: float) -> None:
        old = self.timeline.time_to_x(self._playhead)
        self._playhead = t
        new = self.timeline.time_to_x(t)
        self.update(int(old) - 2, 0, 5, self.height())
        self.update(int(new) - 2, 0, 5, self.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), theme.WAVE_BG)
        tl = self.timeline
        t0, t1 = max(0.0, tl.x_to_time(0)), tl.x_to_time(w)
        tempo: TempoMap = self._tempo_getter()
        half = h / 2

        # 위쪽: 마디 번호
        beat_px = tl.px_per_sec * 60.0 / tempo.initial_bpm
        bar_step = float(tempo.beats_per_bar)
        while beat_px * bar_step < 40:
            bar_step *= 2
        p.setPen(QPen(theme.GRID_BAR, 1))
        for b in tempo.grid_beats(t0, t1, bar_step):
            x = tl.time_to_x(tempo.time_at(b))
            p.drawLine(QLineF(x, 0, x, half))
            p.setPen(theme.TEXT)
            p.drawText(QRectF(x + 3, 0, 60, half), Qt.AlignmentFlag.AlignVCenter, str(int(b // tempo.beats_per_bar) + 1))
            p.setPen(QPen(theme.GRID_BAR, 1))

        # 아래쪽: 초 눈금
        step = next((s for s in _SECOND_STEPS if s * tl.px_per_sec >= 70), _SECOND_STEPS[-1])
        p.setPen(QPen(theme.GRID_BEAT, 1))
        p.drawLine(QLineF(0, half, w, half))
        k0 = math.floor(t0 / step)
        k1 = math.ceil(t1 / step)
        for k in range(k0, k1 + 1):
            t = k * step
            x = tl.time_to_x(t)
            p.setPen(QPen(theme.GRID_BEAT, 1))
            p.drawLine(QLineF(x, h - 6, x, h))
            p.setPen(theme.TEXT_DIM)
            p.drawText(QRectF(x + 3, half, 70, half), Qt.AlignmentFlag.AlignVCenter, _fmt(t, step))

        x = tl.time_to_x(self._playhead)
        p.setPen(QPen(theme.PLAYHEAD, 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, h))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.seekRequested.emit(max(0.0, self.timeline.x_to_time(event.position().x())))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.seekRequested.emit(max(0.0, self.timeline.x_to_time(event.position().x())))

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.timeline.zoomRequested.emit(1.25 if delta > 0 else 0.8, self.timeline.x_to_time(event.position().x()))
        else:
            self.timeline.scrollRequested.emit(-float(delta))
        event.accept()


def _fmt(t: float, step: float) -> str:
    m, s = divmod(t, 60)
    if step < 1:
        return f"{int(m)}:{s:04.1f}"
    return f"{int(m)}:{int(round(s)):02d}"
