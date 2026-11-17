"""좌측 프로젝트·트랙 패널 (명세 6.2).

프로젝트 정보(이름·원본 저장 방식·원본 상태)와 Original / Vocals / Instrumental 트랙 세 줄을 보여 준다.
트랙마다 솔로(개별 재생)·음소거·음량·내보내기·파일 위치 열기·분리 상태가 있다.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from amp.app.theme import tokens as T
from amp.app.theme.icons import icon

TRACK_LABELS = {"original": "Original", "vocals": "Vocals", "instrumental": "Instrumental"}
TRACK_HINTS = {"original": "원본", "vocals": "보컬", "instrumental": "반주"}


class _Swatch(QWidget):
    def __init__(self, color: str, parent=None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(4, 34)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self._color)


def _tool(name: str, tip: str, checkable: bool = False) -> QToolButton:
    b = QToolButton()
    b.setIcon(icon(name, size=16))
    b.setIconSize(QSize(16, 16))
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(False)
    b.setFixedSize(28, 26)
    return b


class TrackRow(QFrame):
    soloToggled = Signal(str, bool)
    muteToggled = Signal(str, bool)
    volumeChanged = Signal(str, float)
    exportRequested = Signal(str)
    revealRequested = Signal(str)
    selected = Signal(str)

    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.name = name
        self.setObjectName("trackRow")
        self.setProperty("selected", False)
        self.setProperty("available", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        title = QLabel(TRACK_LABELS[name])
        title.setObjectName("panelTitle")
        self.info = QLabel()
        self.info.setProperty("role", "secondary")
        self.info.setWordWrap(True)
        self.status = QLabel()
        self.status.setProperty("role", "secondary")

        self.solo = _tool("headphones", f"{TRACK_HINTS[name]}만 듣기 (솔로)", checkable=True)
        self.mute = _tool("volume-x", "음소거", checkable=True)
        self.export = _tool("download", f"{TRACK_HINTS[name]} 내보내기 (WAV/MP3)")
        self.reveal = _tool("folder-open", "파일 위치 열기")
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 150)
        self.volume.setValue(100)
        self.volume.setToolTip("음량 100%")
        self.volume.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.solo.toggled.connect(lambda on: self.soloToggled.emit(self.name, on))
        self.mute.toggled.connect(lambda on: self.muteToggled.emit(self.name, on))
        self.volume.valueChanged.connect(self._on_volume)
        self.export.clicked.connect(lambda: self.exportRequested.emit(self.name))
        self.reveal.clicked.connect(lambda: self.revealRequested.emit(self.name))

        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.status)
        controls = QHBoxLayout()
        controls.setSpacing(4)
        for w in (self.solo, self.mute):
            controls.addWidget(w)
        controls.addWidget(self.volume, 1)
        for w in (self.export, self.reveal):
            controls.addWidget(w)
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        body.addLayout(head)
        body.addWidget(self.info)
        body.addLayout(controls)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(_Swatch(T.TRACK_COLORS[name]))
        outer.addLayout(body, 1)

    def _on_volume(self, v: int) -> None:
        self.volume.setToolTip(f"음량 {v}%")
        self.volumeChanged.emit(self.name, v / 100.0)

    def set_available(self, available: bool, info: str, status: str, state: str) -> None:
        """state: ok / warn / error / muted (상태는 색과 함께 문구로도 보여 준다)."""
        self.setProperty("available", available)
        for w in (self.solo, self.mute, self.volume, self.export):
            w.setEnabled(available)
        self.reveal.setEnabled(available)
        self.info.setText(info)
        self.status.setText(status)
        self.status.setProperty("state", state)
        self._repolish()

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self._repolish()

    def set_controls(self, solo: bool, mute: bool, gain: float) -> None:
        for w, v in ((self.solo, solo), (self.mute, mute)):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self.volume.blockSignals(True)
        self.volume.setValue(int(round(gain * 100)))
        self.volume.blockSignals(False)

    def _repolish(self) -> None:
        for w in (self, self.status, self.info):
            w.style().unpolish(w)
            w.style().polish(w)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.property("available"):
            self.selected.emit(self.name)
        super().mousePressEvent(event)


class TrackPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setMinimumWidth(240)
        self.project_name = QLabel("프로젝트 없음")
        self.project_name.setObjectName("panelTitle")
        self.project_name.setWordWrap(True)
        self.project_info = QLabel("오디오를 열면 프로젝트 폴더가 만들어집니다.")
        self.project_info.setProperty("role", "secondary")
        self.project_info.setWordWrap(True)
        self.original_state = QLabel()
        self.original_state.setWordWrap(True)
        self.open_folder = QToolButton()
        self.open_folder.setText(" 폴더 열기")
        self.open_folder.setIcon(icon("folder-open", size=16))
        self.open_folder.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.relink = QToolButton()
        self.relink.setText(" 원본 다시 연결")
        self.relink.setIcon(icon("link", size=16))
        self.relink.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.relink.hide()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        header = QLabel("프로젝트")
        header.setProperty("role", "secondary")
        lay.addWidget(header)
        lay.addWidget(self.project_name)
        lay.addWidget(self.project_info)
        lay.addWidget(self.original_state)
        btns = QHBoxLayout()
        btns.addWidget(self.open_folder)
        btns.addWidget(self.relink)
        btns.addStretch(1)
        lay.addLayout(btns)
        tracks_title = QLabel("트랙")
        tracks_title.setProperty("role", "secondary")
        lay.addSpacing(6)
        lay.addWidget(tracks_title)
        self.rows = {name: TrackRow(name) for name in TRACK_LABELS}
        for row in self.rows.values():
            lay.addWidget(row)
            row.selected.connect(self.select)
        lay.addStretch(1)
        self.selected_track = "original"
        self.select("original")

    def select(self, name: str) -> None:
        self.selected_track = name
        for n, row in self.rows.items():
            row.set_selected(n == name)
