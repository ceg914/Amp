"""악기 소리 (v0.6.4): 오른쪽 '악기' 탭의 '악기 소리'. 드럼·베이스·기타·피아노·나머지를 분리해 듣고 파일로 내보낸다.

악보는 만들지 않는다(베이스만 '베이스 TAB'). 분리 결과는 프로젝트 stems/models/에 남고, 보컬 분석 소스는 바꾸지 않는다.

듣기 (v0.6.5): [듣기]를 켠 악기만 재생기에 트랙으로 올린다. 이 트랙은 늘 음소거 + 솔로라서
- 켠 동안: 솔로라 그 악기(여러 개 켜면 그 악기들, 왼쪽 트랙 패널에서 솔로한 트랙도 함께)만 들린다
- 모델 비교 A/B처럼 다른 쪽이 솔로를 끄면: 음소거라 들리지 않는다. 기존 재생(원본·보컬·반주)은 바뀌지 않는다
끄면 트랙을 내려 메모리를 돌려준다. 재생 위치·반복·속도는 그대로다.
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from amp.engine import instruments as I

TRACK_PREFIX = "inst_"  # 재생기 트랙 이름 (inst_drums 등)
# 단추 → (분리 모델, 저장할 악기)
KINDS = {"basic": ("htdemucs", I.MODEL_STEMS["htdemucs"]), "six": ("htdemucs_6s", I.MODEL_STEMS["htdemucs_6s"])}


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


class InstrumentPanel(QWidget):
    extractRequested = Signal(str)  # KINDS 키
    exportRequested = Signal(str)  # 악기 이름
    revealRequested = Signal(str)
    listenToggled = Signal(str, bool)

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
        self.listen: dict[str, QPushButton] = {}
        for r, (name, label) in enumerate(I.INSTRUMENTS.items()):
            title = QLabel(label)
            title.setObjectName("panelTitle")
            st = QLabel("없음")
            st.setProperty("role", "secondary")
            st.setWordWrap(True)
            ls = QPushButton("듣기")
            ls.setCheckable(True)
            ls.setToolTip(f"{label}만 듣기. 여러 악기를 켜면 켠 악기만 함께 들립니다")
            ls.toggled.connect(lambda on, n=name: self.listenToggled.emit(n, on))
            ex = QPushButton("내보내기…")
            ex.setToolTip(f"{label} 소리를 WAV(24bit)나 MP3(320kbps)로 저장")
            rv = QPushButton("위치")
            rv.setToolTip("분리한 파일이 있는 폴더 열기")
            ex.clicked.connect(lambda _c=False, n=name: self.exportRequested.emit(n))
            rv.clicked.connect(lambda _c=False, n=name: self.revealRequested.emit(n))
            for b in (ls, ex, rv):  # 글자가 잘리지 않게 단추는 글자 너비로, 남는 폭은 상태 글이 쓴다
                b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            grid.addWidget(title, r, 0)
            grid.addWidget(st, r, 1)
            grid.addWidget(ls, r, 2)
            grid.addWidget(ex, r, 3)
            grid.addWidget(rv, r, 4)
            self.status[name], self.export[name], self.reveal[name], self.listen[name] = st, ex, rv, ls
        lay.addLayout(grid)
        self.listen_info = QLabel("")
        self.listen_info.setWordWrap(True)
        self.listen_info.setProperty("role", "secondary")
        self.stop_listen = QPushButton("악기 듣기 모두 끄기")
        lay.addWidget(self.listen_info)
        lay.addWidget(self.stop_listen)
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
        self.panel.listenToggled.connect(self.listen)
        self.panel.stop_listen.clicked.connect(self.stop_listening)
        self._loading: set[str] = set()
        self._fading: set[str] = set()  # 끄는 중 (음량을 내린 뒤 트랙을 내린다)
        w.doc.projectReset.connect(self.stop_listening)
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
            self.panel.listen[name].setEnabled(f is not None and self.w.session.audio is not None)
            if f is None and self.listening(name):
                self.listen(name, False)
        self._update_listen_info()
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

    # --- 듣기 --------------------------------------------------------------------

    def listening(self, name: str) -> bool:
        return name in self._loading or (TRACK_PREFIX + name in self.w.player.tracks and name not in self._fading)

    def _set_checked(self, name: str, on: bool) -> None:
        b = self.panel.listen[name]
        if b.isChecked() != on:
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)

    def _update_listen_info(self) -> None:
        on = [I.INSTRUMENTS[n] for n in I.INSTRUMENTS if self.listening(n)]
        bass = getattr(self.w, "bass", None)
        if bass is not None:  # 베이스 TAB 탭의 [베이스만 듣기]도 같은 상태로
            b = bass.panel.listen
            b.blockSignals(True)
            b.setChecked(self.listening("bass"))
            b.blockSignals(False)
            b.setEnabled("bass" in self.found and self.w.session.audio is not None)
        self.panel.listen_info.setText(f"듣는 중: {'·'.join(on)} (다른 트랙은 왼쪽에서 솔로한 것만 함께 들림)" if on else "")
        self.panel.stop_listen.setEnabled(bool(on))

    def listen(self, name: str, on: bool) -> None:
        """name 악기만 듣기를 켜고 끈다. 켜면 분리 파일을 작업 스레드에서 읽어 재생기 트랙으로 올린다."""
        w = self.w
        track = TRACK_PREFIX + name
        self._set_checked(name, on)
        if not on:
            self._loading.discard(name)
            w.player.set_track_state(track, mute=True, solo=False)  # 재생 중이면 믹서가 12ms 동안 줄인다
            if track in w.player.tracks:
                from amp.app.player import PlaybackState

                if w.player.state is PlaybackState.PLAYING:
                    self._fading.add(name)
                    QTimer.singleShot(80, lambda n=name: self._unload(n))
                else:
                    w.player.set_track(track, None)
            self._update_listen_info()
            return
        self._fading.discard(name)
        f = self.found.get(name)
        if f is None or w.session.audio is None:
            self._set_checked(name, False)
            return
        self._loading.add(name)
        self._update_listen_info()
        path = str(f.path)

        def job(progress, token):
            import soundfile as sf

            from amp.audio.data import AudioData

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return AudioData(np.ascontiguousarray(data), sr)

        handle = w.runner.submit(f"{I.INSTRUMENTS[name]} 소리 불러오기", job)
        handle.finished.connect(lambda audio, n=name: self._on_loaded(n, audio))
        handle.failed.connect(lambda msg, _tb, n=name: self._on_load_failed(n, msg))

    def _unload(self, name: str) -> None:
        if name not in self._fading:  # 그사이 다시 켰다
            return
        self._fading.discard(name)
        track = TRACK_PREFIX + name
        if track in self.w.player.tracks:
            self.w.player.set_track(track, None)
        self._update_listen_info()

    def _on_loaded(self, name: str, audio) -> None:
        w = self.w
        if name not in self._loading:  # 불러오는 사이에 껐거나 프로젝트를 바꿨다
            return
        self._loading.discard(name)
        track = TRACK_PREFIX + name
        w.player.set_track_state(track, mute=True, solo=True)  # 솔로일 때만 들리고, 솔로가 풀리면 음소거
        if track in w.player.tracks:  # 끄는 중이던 같은 트랙이 아직 있으면 그대로 다시 쓴다
            self._update_listen_info()
            return
        w.player.set_track(track, audio)
        if track not in w.player.tracks:  # 길이·샘플레이트가 원본과 다르면 재생기가 받지 않는다
            self._set_checked(name, False)
        self._update_listen_info()

    def _on_load_failed(self, name: str, msg: str) -> None:
        self._loading.discard(name)
        self._set_checked(name, False)
        self._update_listen_info()
        self.panel.message.setText(f"{I.INSTRUMENTS[name]} 소리를 불러오지 못했습니다: {msg}")

    def stop_listening(self) -> None:
        for name in I.INSTRUMENTS:
            if self.listening(name) or self.panel.listen[name].isChecked():
                self.listen(name, False)
        self._update_listen_info()

    def reveal(self, name: str) -> None:
        f = self.found.get(name)
        if f is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(f.path.parent)))
