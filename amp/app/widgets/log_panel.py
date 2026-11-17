"""하단 로그 패널 (명세 6.6: 자세한 로그는 필요할 때 펼친다)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPlainTextEdit

from amp.app import theme


class _Bridge(QObject):
    line = Signal(str)


class QtLogHandler(logging.Handler):
    """다른 스레드의 로그도 시그널(큐 연결)로 UI 스레드에 넘긴다."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.bridge = _Bridge()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.line.emit(self.format(record))
        except RuntimeError:  # 창이 닫히는 중
            pass


class LogPanel(QPlainTextEdit):
    MAX_LINES = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setFont(theme.mono_font(12))
        self.handler = QtLogHandler()
        self.handler.bridge.line.connect(self.appendPlainText)

    def attach(self) -> None:
        logging.getLogger().addHandler(self.handler)

    def detach(self) -> None:
        logging.getLogger().removeHandler(self.handler)
