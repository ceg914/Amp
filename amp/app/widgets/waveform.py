"""파형 표시. 화면 폭만큼만 요약값을 계산하고 결과를 픽스맵으로 캐시한다."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from amp.app import theme
from amp.app.timeline import TimelineState
from amp.audio.peaks import WaveformPeaks


class WaveformView(QWidget):
    seekRequested = Signal(float)

    def __init__(self, timeline: TimelineState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = timeline
        self._peaks: WaveformPeaks | None = None
        self._playhead = 0.0
        self._cache: QPixmap | None = None
        self._cache_key: tuple | None = None
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(90)
        timeline.changed.connect(self.update)

    def set_peaks(self, peaks: WaveformPeaks | None) -> None:
        self._peaks = peaks
        self._cache_key = None
        self.update()

    def set_playhead(self, t: float) -> None:
        old_x = self.timeline.time_to_x(self._playhead)
        self._playhead = t
        new_x = self.timeline.time_to_x(t)
        h = self.height()
        self.update(int(old_x) - 2, 0, 5, h)
        self.update(int(new_x) - 2, 0, 5, h)

    def _render(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        w, h = self.width(), self.height()
        key = (w, h, dpr, self.timeline.px_per_sec, self.timeline.scroll_x, id(self._peaks))
        if self._cache is not None and self._cache_key == key:
            return self._cache
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(theme.WAVE_BG)
        p = QPainter(pm)
        mid = h / 2
        p.setPen(QPen(theme.GRID_BEAT, 1))
        p.drawLine(QLineF(0, mid, w, mid))
        if self._peaks is not None and w > 0:
            cols = int(w * dpr)
            t0 = self.timeline.x_to_time(0)
            t1 = self.timeline.x_to_time(w)
            mins, maxs = self._peaks.columns(t0, t1, cols)
            valid = np.flatnonzero(~np.isnan(mins))
            if valid.size:
                xs = valid / dpr
                y_top = mid - maxs[valid] * (mid - 2)
                y_bot = mid - mins[valid] * (mid - 2)
                p.setPen(QPen(theme.WAVE, 1.0 / dpr))
                p.drawLines([QLineF(x, a, x, b + 0.5) for x, a, b in zip(xs.tolist(), y_top.tolist(), y_bot.tolist())])
        else:
            p.setPen(theme.TEXT_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "오디오 파일을 열면 파형이 표시됩니다 (Ctrl+O)")
        p.end()
        self._cache, self._cache_key = pm, key
        return pm

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.drawPixmap(0, 0, self._render())
        x = self.timeline.time_to_x(self._playhead)
        p.setPen(QPen(theme.PLAYHEAD, 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._peaks is not None:
            self.seekRequested.emit(max(0.0, self.timeline.x_to_time(event.position().x())))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton and self._peaks is not None:
            self.seekRequested.emit(max(0.0, self.timeline.x_to_time(event.position().x())))

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.timeline.zoomRequested.emit(1.25 if delta > 0 else 0.8, self.timeline.x_to_time(event.position().x()))
        else:
            self.timeline.scrollRequested.emit(-float(delta))
        event.accept()
