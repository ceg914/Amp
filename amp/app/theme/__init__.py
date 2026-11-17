"""화면 테마: 토큰 → Qt 팔레트·스타일시트·글꼴·아이콘.

예전 코드가 쓰던 상수(BG, NOTE_FILL 등)도 여기서 토큰을 바탕으로 제공한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from amp.app.theme import tokens as T

log = logging.getLogger(__name__)
FONT_DIR = Path(__file__).with_name("fonts")

# --- 피아노 롤·파형에서 쓰는 색 ---
BG = QColor(T.BG_PRIMARY)
ROW_WHITE = QColor(T.ROLL_ROW_WHITE)
ROW_BLACK = QColor(T.ROLL_ROW_BLACK)
ROW_C_LINE = QColor(T.ROLL_OCTAVE)
GRID_SUB = QColor(T.ROLL_GRID_SUB)
GRID_BEAT = QColor(T.ROLL_GRID_BEAT)
GRID_BAR = QColor(T.ROLL_GRID_BAR)
PLAYHEAD = QColor(T.PLAYHEAD)
WAVE = QColor(T.TEXT_2)
WAVE_BG = QColor(T.WAVE_BG)
PITCH_CURVE = QColor(T.ACCENT_2)
TEXT = QColor(T.TEXT)
TEXT_DIM = QColor(T.TEXT_2)
ACCENT = QColor(T.ACCENT)

# 음표는 모두 바이올렛. 상태(원본/수정/추가)는 채움·테두리 모양으로 구분한다(색만으로 구분하지 않음).
NOTE_FILL = {"original": QColor(T.ACCENT), "modified": QColor(T.ACCENT), "user": QColor(T.ACCENT)}
NOTE_SELECTED_BORDER = QColor(T.TEXT)
NOTE_BORDER = QColor("#5B4A99")

_font_families: dict[str, str] = {}


def load_fonts() -> dict[str, str]:
    """번들 글꼴을 등록하고, 실제로 쓸 글꼴 이름을 돌려준다 {"ui": ..., "mono": ...}."""
    if _font_families:
        return _font_families
    for f in sorted(FONT_DIR.glob("*.[ot]tf")):
        if QFontDatabase.addApplicationFont(str(f)) < 0:
            log.warning("글꼴을 불러오지 못했습니다: %s", f.name)
    families = set(QFontDatabase.families())

    def pick(primary: str, fallbacks: tuple[str, ...]) -> str:
        for name in (primary, *fallbacks):
            if name in families:
                return name
        return QFont().defaultFamily()

    _font_families["ui"] = pick(T.FONT_UI, T.FONT_UI_FALLBACKS)
    _font_families["mono"] = pick(T.FONT_MONO, T.FONT_MONO_FALLBACKS)
    return _font_families


def mono_font(px: int = T.SIZE_MONO, weight: QFont.Weight = QFont.Weight.Medium) -> QFont:
    f = QFont(load_fonts()["mono"])
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def ui_font(px: int = T.SIZE_BODY, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont(load_fonts()["ui"])
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def stylesheet() -> str:
    fam = load_fonts()
    return f"""
* {{ font-family: "{fam['ui']}"; font-size: {T.SIZE_BODY}px; color: {T.TEXT}; }}
QMainWindow, QDialog {{ background: {T.BG_PRIMARY}; }}
QWidget#panel, QDockWidget > QWidget, QFrame#panel {{ background: {T.BG_PANEL}; }}
QLabel#panelTitle {{ font-size: {T.SIZE_PANEL_TITLE}px; font-weight: 600; color: {T.TEXT}; }}
QLabel#hint, QLabel[role="secondary"] {{ color: {T.TEXT_2}; font-size: {T.SIZE_SMALL + 1}px; }}
QLabel#mono, QLabel[role="mono"] {{ font-family: "{fam['mono']}"; font-size: {T.SIZE_MONO + 1}px; }}
QLabel#timecode {{ font-family: "{fam['mono']}"; font-size: 14px; font-weight: 500; padding: 0 8px; }}
QToolBar {{ background: {T.BG_PANEL}; border: none; border-bottom: 1px solid {T.BORDER}; spacing: 4px; padding: 4px; }}
QToolBar::separator {{ background: {T.BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton, QPushButton {{
  background: {T.BG_RAISED}; border: 1px solid {T.BORDER}; border-radius: {T.RADIUS_BUTTON}px; padding: 4px 10px;
}}
QToolButton:hover, QPushButton:hover {{ background: {T.RAISED_HOVER}; }}
QToolButton:pressed, QPushButton:pressed {{ background: {T.BG_PANEL}; }}
QToolButton:checked {{ background: {T.BG_PANEL}; border-color: {T.ACCENT}; color: {T.ACCENT}; }}
QToolButton:focus, QPushButton:focus {{ border-color: {T.ACCENT}; }}
QToolButton:disabled, QPushButton:disabled {{ color: {T.DISABLED}; border-color: {T.BG_RAISED}; }}
QPushButton[primary="true"], QToolButton[primary="true"] {{
  background: {T.ACCENT}; color: {T.BG_PRIMARY}; border-color: {T.ACCENT}; font-weight: 600;
}}
QPushButton[primary="true"]:hover, QToolButton[primary="true"]:hover {{ background: {T.ACCENT_HOVER}; }}
QPushButton[primary="true"]:pressed, QToolButton[primary="true"]:pressed {{ background: {T.ACCENT_PRESSED}; }}
QPushButton[primary="true"]:disabled, QToolButton[primary="true"]:disabled {{ background: {T.BG_RAISED}; color: {T.DISABLED}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
  background: {T.BG_RAISED}; border: 1px solid {T.BORDER}; border-radius: {T.RADIUS_BUTTON}px; padding: 3px 6px;
  selection-background-color: {T.ACCENT}; selection-color: {T.BG_PRIMARY};
}}
QSpinBox, QDoubleSpinBox {{ font-family: "{fam['mono']}"; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {T.ACCENT}; }}
QComboBox QAbstractItemView {{ background: {T.BG_RAISED}; border: 1px solid {T.BORDER}; selection-background-color: {T.ACCENT}; selection-color: {T.BG_PRIMARY}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {T.BORDER}; background: {T.BG_RAISED}; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {T.ACCENT}; border-color: {T.ACCENT}; }}
QGroupBox {{ border: 1px solid {T.BORDER}; border-radius: 8px; margin-top: 14px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; font-weight: 600; font-size: {T.SIZE_PANEL_TITLE}px; }}
QSlider::groove:horizontal {{ height: 4px; background: {T.BG_RAISED}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; border-radius: 6px; background: {T.TEXT_2}; }}
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:focus {{ background: {T.ACCENT}; }}
QSlider::sub-page:horizontal {{ background: {T.ACCENT}; border-radius: 2px; }}
QProgressBar {{ background: {T.BG_RAISED}; border: none; border-radius: 3px; height: 6px; }}
QProgressBar::chunk {{ background: {T.ACCENT}; border-radius: 3px; }}
QMenuBar {{ background: {T.BG_PANEL}; border-bottom: 1px solid {T.BORDER}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {T.BG_RAISED}; }}
QMenu {{ background: {T.BG_PANEL}; border: 1px solid {T.BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 22px; border-radius: 4px; }}
QMenu::item:disabled {{ color: {T.DISABLED}; }}
QMenu::separator {{ height: 1px; background: {T.BORDER}; margin: 4px 8px; }}
QStatusBar {{ background: {T.BG_PANEL}; border-top: 1px solid {T.BORDER}; }}
QStatusBar QLabel {{ color: {T.TEXT_2}; }}
QSplitter::handle {{ background: {T.BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {T.BG_PANEL}; padding: 4px 8px; border-bottom: 1px solid {T.BORDER}; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 0; }}
QScrollBar::handle {{ background: {T.BG_RAISED}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {T.BORDER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {T.BG_RAISED}; color: {T.TEXT}; border: 1px solid {T.BORDER}; padding: 4px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 6px 10px; color: {T.TEXT_2}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {T.TEXT}; border-bottom-color: {T.ACCENT}; }}
QScrollArea {{ border: none; background: transparent; }}
QFrame#trackRow {{ background: {T.BG_RAISED}; border: 1px solid {T.BORDER}; border-radius: 8px; }}
QFrame#trackRow QToolButton {{ padding: 0px; background: {T.BG_PANEL}; }}
QFrame#trackRow QToolButton:checked {{ background: {T.BG_PRIMARY}; border-color: {T.ACCENT}; }}
QFrame#trackRow[selected="true"] {{ border-color: {T.ACCENT}; }}
QFrame#trackRow[available="false"] QLabel {{ color: {T.DISABLED}; }}
QLabel[state="ok"] {{ color: {T.SUCCESS}; }}
QLabel[state="warn"] {{ color: {T.WARNING}; }}
QLabel[state="error"] {{ color: {T.ERROR}; }}
QLabel[state="muted"] {{ color: {T.DISABLED}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: T.BG_PRIMARY, QPalette.ColorRole.WindowText: T.TEXT,
        QPalette.ColorRole.Base: T.BG_RAISED, QPalette.ColorRole.AlternateBase: T.BG_PANEL,
        QPalette.ColorRole.ToolTipBase: T.BG_RAISED, QPalette.ColorRole.ToolTipText: T.TEXT,
        QPalette.ColorRole.Text: T.TEXT, QPalette.ColorRole.Button: T.BG_RAISED,
        QPalette.ColorRole.ButtonText: T.TEXT, QPalette.ColorRole.Highlight: T.ACCENT,
        QPalette.ColorRole.HighlightedText: T.BG_PRIMARY, QPalette.ColorRole.PlaceholderText: T.TEXT_2,
        QPalette.ColorRole.Link: T.ACCENT,
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(T.DISABLED))
    app.setPalette(p)
    app.setFont(ui_font())
    app.setStyleSheet(stylesheet())


def apply_dark_palette(app: QApplication) -> None:
    """예전 이름 (STEP 1 코드·테스트 호환)."""
    apply_theme(app)
