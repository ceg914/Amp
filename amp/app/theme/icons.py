"""Lucide 아이콘 (ISC 라이선스, icons/LICENSE-lucide.txt).

SVG의 stroke="currentColor"를 토큰 색으로 바꿔 그린다. Qt SVG 렌더러는 currentColor를 모른다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from amp.app.theme import tokens as T

ICON_DIR = Path(__file__).with_name("icons")


@lru_cache(maxsize=None)
def _svg(name: str) -> str | None:
    p = ICON_DIR / f"{name}.svg"
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _pixmap(name: str, color: str, size: int, dpr: float = 2.0) -> QPixmap:
    text = _svg(name)
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.GlobalColor.transparent)
    if text is None:
        return pm
    renderer = QSvgRenderer(QByteArray(text.replace("currentColor", color).encode("utf-8")))
    painter = QPainter(pm)
    renderer.render(painter, QRectF(0, 0, pm.width(), pm.height()))
    painter.end()
    pm.setDevicePixelRatio(dpr)
    return pm


@lru_cache(maxsize=None)
def icon(name: str, color: str = T.TEXT, size: int = 18) -> QIcon:
    ic = QIcon()
    ic.addPixmap(_pixmap(name, color, size), QIcon.Mode.Normal)
    ic.addPixmap(_pixmap(name, T.DISABLED, size), QIcon.Mode.Disabled)
    ic.addPixmap(_pixmap(name, T.ACCENT, size), QIcon.Mode.Normal, QIcon.State.On)
    return ic


ICON_SIZE = QSize(18, 18)
