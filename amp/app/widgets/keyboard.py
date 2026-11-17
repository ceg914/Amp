"""피아노 롤 왼쪽 건반. 피아노 롤의 세로 스크롤·확대를 그대로 따라간다."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsView, QSizePolicy, QWidget

from amp.app import theme
from amp.app.pianoroll.mapping import PITCH_MAX, PITCH_MIN, PianoRollMapper
from amp.core.pitch import is_black_key, note_name

KEYBOARD_WIDTH = 64


class PianoKeyboard(QWidget):
    def __init__(self, view: QGraphicsView, mapper: PianoRollMapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = view
        self._mapper = mapper
        self.setFixedWidth(KEYBOARD_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        view.verticalScrollBar().valueChanged.connect(self.update)
        view.verticalScrollBar().rangeChanged.connect(lambda *_: self.update())

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        m = self._mapper
        # 피아노 롤 뷰는 테두리가 없으므로(NoFrame) 뷰포트 위쪽 = 이 위젯 위쪽이다
        top = self._view.mapToScene(0, 0).y()
        p.fillRect(self.rect(), theme.BG)
        p_top = min(PITCH_MAX, m.y_to_pitch(top))
        p_bot = max(PITCH_MIN, m.y_to_pitch(top + self.height()))
        font = p.font()
        font.setPixelSize(max(8, min(11, int(m.row_height - 2))))
        p.setFont(font)
        for pitch in range(p_bot, p_top + 1):
            y = m.pitch_to_y(pitch) - top
            r = QRectF(0, y, self.width(), m.row_height)
            black = is_black_key(pitch)
            p.fillRect(r, QColor(25, 25, 28) if black else QColor(225, 225, 228))
            p.setPen(QPen(QColor(120, 120, 125), 1))
            p.drawLine(r.bottomLeft(), r.bottomRight())
            if pitch % 12 == 0 and m.row_height >= 8:
                p.setPen(QColor(40, 40, 45))
                p.drawText(r.adjusted(0, 0, -4, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, note_name(pitch))
