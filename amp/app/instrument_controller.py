"""악기 소리 (v0.6.4): 오른쪽 '악기' 탭의 '악기 소리'. 드럼·베이스·기타·피아노·나머지를 분리해 파일로 내보낸다.

악보는 만들지 않는다(베이스만 '베이스 TAB'). 분리 결과는 프로젝트 stems/models/에 남고, 보컬 분석 소스는 바꾸지 않는다.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from amp.engine import instruments as I

# 단추 → (분리 모델, 저장할 악기)
KINDS = {"basic": ("htdemucs", I.MODEL_STEMS["htdemucs"]), "six": ("htdemucs_6s", I.MODEL_STEMS["htdemucs_6s"])}


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


class InstrumentPanel(QWidget):
    extractRequested = Signal(str)  # KINDS 키
    exportRequested = Signal(str)  # 악기 이름
    revealRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        intro = QLabel("악기 소리를 따로 뽑아 파일(WAV·MP3)로 저장합니다. 악보는 만들지 않습니다. "
                       "분리한 파일은 프로젝트의 stems/models/에 남고, 보컬 분석에 쓰는 결과는 바뀌지 않습니다.")
        intro.setWordWrap(True)
        intro.setProperty("role", "secondary")
        lay.addWidget(intro)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.status: dict[str, QLabel] = {}
        self.export: dict[str, QPushButton] = {}
        self.reveal: dict[str, QPushButton] = {}
        for r, (name, label) in enumerate(I.INSTRUMENTS.items()):
            title = QLabel(label)
            title.setObjectName("panelTitle")
            st = QLabel("없음")
            st.setProperty("role", "secondary")
            st.setWordWrap(True)
            ex = QPushButton("내보내기…")
            ex.setToolTip(f"{label} 소리를 WAV(24bit)나 MP3(320kbps)로 저장")
            rv = QPushButton("위치")
            rv.setToolTip("분리한 파일이 있는 폴더 열기")
            ex.clicked.connect(lambda _c=False, n=name: self.exportRequested.emit(n))
            rv.clicked.connect(lambda _c=False, n=name: self.revealRequested.emit(n))
            grid.addWidget(title, r, 0)
            grid.addWidget(st, r, 1)
            grid.addWidget(ex, r, 2)
            grid.addWidget(rv, r, 3)
            self.status[name], self.export[name], self.reveal[name] = st, ex, rv
        lay.addLayout(grid)
        self.basic = QPushButton("드럼·베이스·나머지 분리 (Demucs htdemucs)")
        self.six = QPushButton("기타·피아노까지 분리 (Demucs htdemucs_6s)")
        self.basic.clicked.connect(lambda: self.extractRequested.emit("basic"))
        self.six.clicked.connect(lambda: self.extractRequested.emit("six"))
        lay.addWidget(self.basic)
        lay.addWidget(self.six)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setProperty("role", "secondary")
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.message)
        note = QLabel(f"분리는 곡 전체를 한 번 더 처리합니다. 이미 있는 악기 파일은 다시 만들지 않습니다. "
                      f"악기 하나는 1분에 약 {I.MB_PER_MIN:.0f}MB(WAV 24bit)를 씁니다. "
                      "'나머지'는 보컬과 위 악기들을 뺀 소리라 분리 모델에 따라 내용이 다릅니다. "
                      "htdemucs_6s의 피아노 분리는 Demucs 설명서에도 아직 잘 되지 않는다고 적혀 있습니다.")
        note.setWordWrap(True)
        note.setProperty("role", "secondary")
        lay.addWidget(note)
        lay.addStretch(1)


class InstrumentController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        w = self.w = window
        self.panel = InstrumentPanel()
        w.instrument_tabs.insertTab(0, self.panel, "악기 소리")
        w.instrument_tabs.setCurrentIndex(0)
        self.found: dict[str, I.StemFile] = {}
        self._installed: bool | None = None
        self.panel.extractRequested.connect(self.extract)
        self.panel.exportRequested.connect(self.export)
        self.panel.revealRequested.connect(self.reveal)
        w.doc.projectReset.connect(self.refresh)
        w.right_tabs.currentChanged.connect(lambda _i: self._refresh_if_shown())
        w.instrument_tabs.currentChanged.connect(lambda _i: self._refresh_if_shown())
        self.refresh()

    def installed(self) -> bool:
        if self._installed is None:
            from amp.app.lite import separation_installed

            self._installed = separation_installed()
        return self._installed

    def _refresh_if_shown(self) -> None:
        if self.w.side_panel() is self.panel:
            self.refresh()

    def refresh(self) -> None:
        folder = self.w.session.folder
        self.found = {}
        if folder is not None:
            try:
                self.found = I.find_stems(folder.paths.stems, folder.meta.original.sha256)
            except OSError:
                self.found = {}
        for name in I.INSTRUMENTS:
            f = self.found.get(name)
            if f is None:
                self.panel.status[name].setText("없음" if folder is not None else "프로젝트 없음")
            else:
                try:
                    import soundfile as sf

                    dur = f" · {_fmt_time(sf.info(str(f.path)).duration)}"
                except Exception:  # noqa: BLE001
                    dur = ""
                self.panel.status[name].setText(f"있음 · {f.model_label}{dur}")
            self.panel.status[name].setToolTip(str(f.path) if f else "")
            self.panel.export[name].setEnabled(f is not None)
            self.panel.reveal[name].setEnabled(f is not None)
        self.update_actions()

    def update_actions(self) -> None:
        s = self.w.session
        can = s.folder is not None and s.original is not None and not s.busy and self.installed()
        self.panel.basic.setEnabled(can)
        self.panel.six.setEnabled(can)
        if not self.installed():
            from amp.core.install_hints import script

            self.panel.basic.setToolTip(f"보컬 분리(PyTorch·Demucs)를 설치해야 합니다: {script('separation')}")
            self.panel.six.setToolTip(self.panel.basic.toolTip())
        busy = s.busy
        for name in I.INSTRUMENTS:
            self.panel.export[name].setEnabled(name in self.found and not busy)

    # --- 분리 --------------------------------------------------------------------

    def extract(self, kind: str) -> None:
        w = self.w
        if w.session.busy:
            w.statusBar().showMessage("다른 분석이 끝난 뒤에 분리하세요.", 5000)
            return
        model, names = KINDS[kind]
        handle = w.session.extract_instruments(model, device=w.inspector.analysis.settings().device, names=names)
        if handle is None:
            return
        self.panel.message.setText(f"분리 중… ({', '.join(I.INSTRUMENTS[n] for n in names)})")
        handle.finished.connect(self._on_done)
        handle.failed.connect(lambda msg, _tb: self.panel.message.setText(f"분리하지 못했습니다: {msg}"))
        handle.cancelled.connect(lambda: self.panel.message.setText("분리를 취소했습니다."))
        handle.ended.connect(self.refresh)
        self.update_actions()

    def _on_done(self, result) -> None:
        names = lambda ns: "·".join(I.INSTRUMENTS[n] for n in ns)  # noqa: E731
        from amp.engine.model_store import MODEL_LABELS

        model = MODEL_LABELS.get(result.model, result.model)
        if result.separated:
            text = f"저장했습니다: {names(result.added)} ({model})."
            if result.kept:
                text += f" {names(result.kept)}은(는) 이미 있어 그대로 두었습니다."
        else:
            text = f"이미 분리해 둔 소리가 있어 다시 분리하지 않았습니다 ({names(result.kept)}, {model})."
        self.panel.message.setText(text)
        self.w.statusBar().showMessage(text, 8000)
        self.refresh()

    # --- 내보내기 ------------------------------------------------------------------

    def export(self, name: str) -> None:
        w = self.w
        f = self.found.get(name)
        folder = w.session.folder
        if f is None or folder is None:
            return
        if w.session.busy:
            w.statusBar().showMessage("분석이 끝난 뒤에 내보내세요.", 5000)
            return
        from amp.project.export import ExportOptions

        folder.paths.exports.mkdir(parents=True, exist_ok=True)
        label = I.INSTRUMENTS[name]
        path, chosen = QFileDialog.getSaveFileName(
            w, f"{label} 소리 내보내기", os.path.join(str(folder.paths.exports), f"{folder.name}_{name}.wav"),
            "WAV 24-bit (*.wav);;MP3 320kbps (*.mp3)")
        if not path:
            return
        fmt = "mp3" if path.lower().endswith(".mp3") or (not path.lower().endswith(".wav") and "MP3" in chosen) else "wav"
        if not path.lower().endswith("." + fmt):
            path += "." + fmt
        w.session.export_file(str(f.path), path, ExportOptions(fmt=fmt, wav_subtype="PCM_24", mp3_bitrate=320),
                              f"{label} 내보내기")

    def reveal(self, name: str) -> None:
        f = self.found.get(name)
        if f is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(f.path.parent)))
