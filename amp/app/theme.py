"""화면 색상과 어두운 테마."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BG = QColor(30, 32, 36)
ROW_WHITE = QColor(40, 43, 48)
ROW_BLACK = QColor(33, 35, 40)
ROW_C_LINE = QColor(75, 80, 90)
GRID_SUB = QColor(48, 51, 57)
GRID_BEAT = QColor(60, 64, 72)
GRID_BAR = QColor(95, 100, 112)
PLAYHEAD = QColor(255, 80, 80)
WAVE = QColor(90, 160, 220)
WAVE_BG = QColor(24, 26, 30)
PITCH_CURVE = QColor(255, 220, 90, 220)
TEXT = QColor(220, 222, 228)
TEXT_DIM = QColor(150, 155, 165)

# 음표 상태별 색 (원본 그대로 / 수정됨 / 사용자 추가)
NOTE_FILL = {
    "original": QColor(70, 130, 200),
    "modified": QColor(220, 140, 60),
    "user": QColor(90, 180, 110),
}
NOTE_SELECTED_BORDER = QColor(255, 255, 255)
NOTE_BORDER = QColor(15, 18, 22)


def apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(45, 47, 52))
    p.setColor(QPalette.ColorRole.WindowText, TEXT)
    p.setColor(QPalette.ColorRole.Base, QColor(30, 32, 36))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 47, 52))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(60, 63, 70))
    p.setColor(QPalette.ColorRole.ToolTipText, TEXT)
    p.setColor(QPalette.ColorRole.Text, TEXT)
    p.setColor(QPalette.ColorRole.Button, QColor(55, 58, 64))
    p.setColor(QPalette.ColorRole.ButtonText, TEXT)
    p.setColor(QPalette.ColorRole.Highlight, QColor(70, 130, 200))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, TEXT_DIM)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, TEXT_DIM)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, TEXT_DIM)
    app.setPalette(p)
