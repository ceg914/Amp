"""악기 소리 (v0.6.4): 오른쪽 '악기' 탭의 '악기 소리'. 드럼·베이스·기타·피아노·나머지를 분리해 파일로 내보낸다.
악보는 만들지 않는다(베이스만 '베이스 TAB'). 분리 결과는 프로젝트 stems/models/에 남고, 보컬 분석 소스는 바꾸지 않는다.

악기 트랙 (v0.6.6): 분리한 악기는 왼쪽 트랙 패널에 Original·Vocals·Instrumental과 같은 줄로 나온다
(헤드폰 = 솔로, 스피커 = 음소거, 음량, 내보내기, 파일 위치, 누르면 파형).
- 악기 소리는 반주에 이미 들어 있으므로 처음에는 음소거다. 음소거를 풀면 지금 소리에 더해지고, 솔로면 그것만 들린다
- 들리거나(음소거 해제·솔로) 파형을 보는 동안만 파일을 메모리에 올리고, 다시 음소거하면 내린다
- 재생 중에 바꿔도 출력 장치를 다시 열지 않는다(믹서에서 트랙을 더하고 뺀다). 음소거·솔로·음량은 프로젝트에 저장한다
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.engine import instruments as I

TRACK_PREFIX = "inst_"  # 재생기 트랙 이름 (inst_drums 등)
# 단추 → (분리 모델, 저장할 악기)
KINDS = {"basic": ("htdemucs", I.MODEL_STEMS["htdemucs"]), "six": ("htdemucs_6s", I.MODEL_STEMS["htdemucs_6s"])}


def _ident(v) -> str:
    return f"{v.key}/{v.slot or v.name}"


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


class InstrumentPanel(QWidget):
    compareRequested = Signal(str)  # v0.7.0 결과 비교: play / measure / spectrogram
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
            for b in (ex, rv):  # 글자가 잘리지 않게 단추는 글자 너비로, 남는 폭은 상태 글이 쓴다
                b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            grid.addWidget(title, r, 0)
            grid.addWidget(st, r, 1)
            grid.addWidget(ex, r, 2)
            grid.addWidget(rv, r, 3)
            self.status[name], self.export[name], self.reveal[name] = st, ex, rv
        lay.addLayout(grid)
        hint = QLabel("듣기: 분리한 악기는 왼쪽 트랙 패널에 줄로 나옵니다. 헤드폰 = 그 악기만, 스피커(음소거 풀기) = 지금 소리에 더하기. "
                      "반주에 이미 들어 있어 처음에는 음소거입니다.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        form = QFormLayout()
        self.source = QComboBox()
        self.source.setToolTip("반주: 보컬 분석에서 이미 보컬을 뺀 소리를 분리해 악기 파일에 보컬이 덜 섞입니다. "
                               "원본: 보컬이 든 원곡을 그대로 분리합니다.")
        form.addRow("분리할 소리", self.source)
        lay.addLayout(form)
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
        note = QLabel(f"분리는 곡 전체를 한 번 더 처리합니다. 같은 소리(반주·원본)를 같은 모델로 분리한 악기 파일이 있으면 다시 만들지 않습니다. "
                      f"악기 하나는 1분에 약 {I.MB_PER_MIN:.0f}MB(WAV 24bit)를 씁니다. "
                      "'나머지'는 보컬과 위 악기들을 뺀 소리라 분리 모델에 따라 내용이 다릅니다. "
                      "반주에서 분리하면 결과는 보컬 분석에 쓰는 분리 결과 폴더에 더합니다. 원본에서 만든 악기와 반주에서 만든 악기는 따로 보관합니다. "
                      "htdemucs_6s의 피아노 분리는 Demucs 설명서에도 아직 잘 되지 않는다고 적혀 있습니다.")
        note.setWordWrap(True)
        note.setProperty("role", "secondary")
        lay.addWidget(note)
        # v0.7.0 결과 비교 (명세 15.2): 원본·반주·모델별 결과를 골라 듣고(A/B), 스펙트로그램으로 보고, 근사 지표를 잰다
        cmp_title = QLabel("결과 비교")
        cmp_title.setObjectName("panelTitle")
        lay.addWidget(cmp_title)
        self.compare = QTreeWidget()
        self.compare.setHeaderLabels(["악기", "만든 방법", "보컬 유사 성분 (근사)"])
        self.compare.setRootIsDecorated(False)
        self.compare.setMinimumHeight(120)
        lay.addWidget(self.compare)
        crow = QHBoxLayout()
        self.compare_play = QPushButton("이 결과로 듣기")
        self.compare_play.setToolTip("고른 결과를 왼쪽 트랙 패널의 그 악기 줄에서 재생합니다 (음소거·솔로·음량은 그대로). "
                                     "다른 결과를 골라 다시 누르면 A/B로 바꿔 들을 수 있습니다.")
        self.compare_spec = QPushButton("스펙트로그램")
        self.compare_spec.setToolTip("고른 결과를 가운데 스펙트로그램으로 봅니다.")
        self.compare_measure = QPushButton("보컬 유사 성분 재기")
        self.compare_measure.setToolTip("지금 보컬 분석에 쓰는 분리 결과의 보컬을 기준으로, 악기 파일마다 보컬과 닮은 성분의 비중을 잽니다.")
        for b, k in ((self.compare_play, "play"), (self.compare_spec, "spectrogram"), (self.compare_measure, "measure")):
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _c=False, k=k: self.compareRequested.emit(k))
            crow.addWidget(b)
        crow.addStretch(1)
        lay.addLayout(crow)
        from amp.engine.stem_compare import NOTE as CMP_NOTE

        cmp_note = QLabel(CMP_NOTE + " 같은 소리는 반주·원본 줄과 함께 켜면 겹쳐 들립니다.")
        cmp_note.setWordWrap(True)
        cmp_note.setProperty("role", "secondary")
        lay.addWidget(cmp_note)
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
        self.panel.compareRequested.connect(self.on_compare)
        self.choice: dict[str, str] = {}  # v0.7.0 악기 → 듣기로 고른 결과 (slot·키) (프로젝트를 바꾸면 비움)
        self.shares: dict[str, dict] = {}  # 파일 경로 → 보컬 유사 성분 (잰 값, 메모리에만)
        self.variants: dict[str, list] = {}
        self._loading: set[str] = set()
        self._cache: dict = {}  # 악기 → AudioData (들리거나 파형을 보는 동안만)
        self._peaks: dict = {}  # 악기 → WaveformPeaks
        self._durations: dict[str, str] = {}
        self._gen = 0  # 프로젝트가 바뀌면 늘린다 (그 전에 시작한 불러오기 결과는 버림)
        self._source_chosen = False  # 사용자가 분리할 소리를 직접 골랐으면 그대로 둔다
        self.panel.source.activated.connect(lambda _i: setattr(self, "_source_chosen", True))
        self._wire_rows()
        w.doc.projectReset.connect(self._on_project_reset)
        w.session.tracksChanged.connect(self.refresh)  # 보컬 분석·분리 결과 가져오기 뒤 반주 선택지를 맞춘다
        w.right_tabs.currentChanged.connect(lambda _i: self._refresh_if_shown())
        w.instrument_tabs.currentChanged.connect(lambda _i: self._refresh_if_shown())
        self.refresh()

    def installed(self) -> bool:
        if self._installed is None:
            from amp.app.lite import separation_installed

            self._installed = separation_installed()
        return self._installed

    def _on_project_reset(self) -> None:
        self._gen += 1
        self._source_chosen = False
        self.choice = {}
        self.shares = {}
        self.reset_tracks()
        self.refresh()

    def _refresh_if_shown(self) -> None:
        if self.w.side_panel() is self.panel:
            self.refresh()

    def refresh(self) -> None:
        folder = self.w.session.folder
        self.found = {}
        self.variants = {}
        if folder is not None:
            try:
                self.found = I.find_stems(folder.paths.stems, folder.meta.original.sha256)
                self.variants = I.find_all_stems(folder.paths.stems, folder.meta.original.sha256)
            except OSError:
                self.found = {}
        for name, ident in list(self.choice.items()):  # 듣기로 고른 결과가 아직 있으면 그것
            v = next((x for x in self.variants.get(name, ()) if _ident(x) == ident), None)
            if v is None:
                self.choice.pop(name, None)
            else:
                self.found[name] = v
        self._fill_compare()
        self._fill_sources()
        self._durations = {}
        for name in I.INSTRUMENTS:
            f = self.found.get(name)
            if f is None:
                self.panel.status[name].setText("없음" if folder is not None else "프로젝트 없음")
            else:
                try:
                    import soundfile as sf

                    self._durations[name] = f"{sf.info(str(f.path)).duration:.1f}s · "
                    dur = f" · {_fmt_time(sf.info(str(f.path)).duration)}"
                except Exception:  # noqa: BLE001
                    dur = ""
                self.panel.status[name].setText(f"있음 · {f.model_label}{dur}")
            self.panel.status[name].setToolTip(str(f.path) if f else "")
            self.panel.export[name].setEnabled(f is not None)
            self.panel.reveal[name].setEnabled(f is not None)
        self._refresh_rows()
        self.update_actions()

    def _fill_sources(self) -> None:
        """분리할 소리 목록: 지금 보컬 분석에 쓰는 분리 결과의 반주(있으면 기본) / 원본."""
        combo = self.panel.source
        prev = combo.currentData()
        folder = self.w.session.folder
        entry = I.instrumental_source(folder.paths.stems, folder.meta.original.sha256) if folder is not None else None
        combo.blockSignals(True)
        combo.clear()
        if entry is not None:
            combo.addItem(f"보컬 분석의 반주 ({entry.label})", "instrumental")
        else:
            combo.addItem("보컬 분석의 반주 (보컬 분석을 먼저 하세요)", "instrumental")
            combo.model().item(0).setEnabled(False)
        combo.addItem("원본 (보컬이 더 섞일 수 있음)", "original")
        want = prev if prev in ("instrumental", "original") and self._source_chosen else "instrumental"
        if entry is None:
            want = "original"
        combo.setCurrentIndex(max(0, combo.findData(want)))
        combo.blockSignals(False)

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

    # --- 결과 비교 (v0.7.0 명세 15.2) ------------------------------------------------------

    def _fill_compare(self) -> None:
        tree = self.panel.compare
        keep = self._chosen_item()
        tree.clear()
        for name, vs in self.variants.items():
            for v in vs:
                playing = self.found.get(name) is not None and _ident(self.found[name]) == _ident(v)
                sh = self.shares.get(str(v.path))
                share = "재지 않음" if sh is None else ("보컬 구간 없음" if sh["vocal_share_db"] is None
                                                       else f"{sh['vocal_share_db']:+.1f} dB")
                it = QTreeWidgetItem([("▶ " if playing else "") + I.INSTRUMENTS[name], v.model_label, share])
                it.setData(0, Qt.ItemDataRole.UserRole, (name, _ident(v)))
                it.setToolTip(1, str(v.path))
                tree.addTopLevelItem(it)
                if keep == (name, _ident(v)):
                    tree.setCurrentItem(it)
        for i in range(3):
            tree.resizeColumnToContents(i)
        has = tree.topLevelItemCount() > 0
        for b in (self.panel.compare_play, self.panel.compare_spec, self.panel.compare_measure):
            b.setEnabled(has)

    def _chosen_item(self):
        it = self.panel.compare.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _variant(self, name: str, ident: str):
        return next((v for v in self.variants.get(name, ()) if _ident(v) == ident), None)

    def on_compare(self, kind: str) -> None:
        chosen = self._chosen_item()
        if kind == "measure":
            self.measure_shares()
            return
        if chosen is None:
            self.panel.message.setText("비교할 결과를 목록에서 고르세요.")
            return
        name, ident = chosen
        if kind == "play":
            self.choose_variant(name, ident)
        elif kind == "spectrogram":
            self.show_spectrogram(name, ident)

    def choose_variant(self, name: str, ident: str) -> bool:
        """그 악기 줄에서 들을 결과를 바꾼다. 음소거·솔로·음량은 바꾸지 않는다."""
        v = self._variant(name, ident)
        if v is None:
            return False
        cur = self.found.get(name)
        self.choice[name] = ident
        if cur is not None and _ident(cur) == ident:
            return False
        self.found[name] = v
        track = TRACK_PREFIX + name
        self._cache.pop(name, None)
        self._peaks.pop(name, None)
        self._gen += 1  # 불러오던 이전 파일 결과는 버린다
        self._loading.discard(name)
        if track in self.w.player.tracks:
            self.w.player.set_track(track, None)
        self.sync(track)
        self._fill_compare()
        self.panel.message.setText(f"{I.INSTRUMENTS[name]} 줄을 '{v.model_label}' 결과로 바꿨습니다. 믹서 설정은 그대로입니다.")
        return True

    def show_spectrogram(self, name: str, ident: str) -> None:
        v = self._variant(name, ident)
        if v is None:
            return
        w = self.w
        path, label = str(v.path), f"{I.INSTRUMENTS[name]} · {v.model_label}"

        def job(progress, token):
            import soundfile as sf

            from amp.audio.data import AudioData

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return AudioData(np.ascontiguousarray(data), sr)

        def done(audio):
            if getattr(w.listening, "lane_mode", "") not in ("spec", "both", "spec_curve"):
                w.listening.set_lane_mode("both")
            w.spectrogram.set_source(label, audio)
            self.panel.message.setText(f"스펙트로그램: {label}")

        handle = w.runner.submit(f"{I.INSTRUMENTS[name]} 스펙트로그램 불러오기", job)
        handle.finished.connect(done)
        handle.failed.connect(lambda msg, _tb: self.panel.message.setText(f"불러오지 못했습니다: {msg}"))

    def reference_vocals(self):
        """보컬 유사 성분의 기준: 지금 보컬 분석에 쓰는 분리 결과의 보컬 파일."""
        folder = self.w.session.folder
        if folder is None:
            return None
        from amp.engine.model_store import ModelStore

        store = ModelStore(folder.paths.stems, folder.meta.original.sha256)
        key = store.active_key()
        e = store.entry(key) if key else None
        if e is None or "vocals" not in e.manifest.files:
            return None
        return e.path / e.manifest.files["vocals"]["path"]

    def measure_shares(self) -> None:
        ref = self.reference_vocals()
        if ref is None:
            self.panel.message.setText("기준 보컬이 없습니다. 보컬 분석(분리)을 먼저 하세요.")
            return
        paths = [str(v.path) for vs in self.variants.values() for v in vs]
        if not paths:
            return

        def job(progress, token):
            from amp.engine.stem_compare import vocal_share_files

            out = {}
            for i, p in enumerate(paths):
                token.raise_if_cancelled()
                progress(i / len(paths), f"보컬 유사 성분 재는 중 {i + 1}/{len(paths)}")
                out[p] = vocal_share_files(p, ref).to_dict()
            return out

        def done(out):
            self.shares.update(out)
            self._fill_compare()
            self.panel.message.setText(f"악기 파일 {len(out)}개의 보컬 유사 성분을 쟀습니다 (기준: {ref.parent.name}의 보컬). "
                                       "근사 지표입니다.")

        handle = self.w.runner.submit("보컬 유사 성분 재기", job)
        handle.finished.connect(done)
        handle.failed.connect(lambda msg, _tb: self.panel.message.setText(f"재지 못했습니다: {msg}"))

    # --- 분리 --------------------------------------------------------------------

    def extract(self, kind: str) -> None:
        w = self.w
        if w.session.busy:
            w.statusBar().showMessage("다른 분석이 끝난 뒤에 분리하세요.", 5000)
            return
        model, names = KINDS[kind]
        source = self.panel.source.currentData() or "original"
        handle = w.session.extract_instruments(model, device=w.inspector.analysis.settings().device, names=names,
                                               source=source)
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
        if getattr(result, "source_model", ""):
            model += f", {MODEL_LABELS.get(result.source_model, result.source_model)} 반주에서"
        else:
            model += ", 원본에서"
        if result.separated:
            text = f"저장했습니다: {names(result.added)} ({model})."
            if result.kept:
                text += f" {names(result.kept)}은(는) 같은 소리·같은 모델로 이미 있어 그대로 두었습니다."
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

    # --- 왼쪽 트랙 패널의 악기 트랙 (v0.6.6) -------------------------------------------

    def _wire_rows(self) -> None:
        for track, row in self.w.tracks.inst_rows.items():
            row.soloToggled.connect(lambda t, on: self.set_track_state(t, solo=on))
            row.muteToggled.connect(lambda t, on: self.set_track_state(t, mute=on))
            row.volumeChanged.connect(lambda t, g: self.w.player.set_track_state(t, gain=g))
            row.exportRequested.connect(lambda t: self.export(t[len(TRACK_PREFIX):]))
            row.revealRequested.connect(lambda t: self.reveal(t[len(TRACK_PREFIX):]))
            row.selected.connect(lambda t: (self.w._show_track_waveform(t), self.sync(t)))
        for row in self.w.tracks.rows.values():  # 다른 줄을 고르면 파형만 보던 악기 소리를 내린다
            row.selected.connect(lambda _t: self.sync_all())

    def reset_tracks(self) -> None:
        """프로젝트가 바뀜: 악기 트랙을 내리고, 상태를 기본(음소거)으로. 저장해 둔 재생 상태는 오디오를 불러온 뒤 덮어쓴다."""
        w = self.w
        self._cache.clear()
        self._peaks.clear()
        self._loading.clear()
        for track in self.w.tracks.inst_rows:
            if track in w.player.tracks:
                w.player.set_track(track, None)
            w.player.set_track_state(track, gain=1.0, mute=True, solo=False)

    def set_track_state(self, track: str, **kw) -> None:
        self.w.player.set_track_state(track, **kw)
        self.sync(track)

    def set_solo(self, name: str, on: bool) -> None:
        """name 악기만 듣기 (트랙 줄의 헤드폰과 같음). 베이스 TAB 탭의 [베이스만 듣기]가 쓴다."""
        row = self.w.tracks.inst_rows.get(TRACK_PREFIX + name)
        if row is not None:
            row.solo.setChecked(on)  # → soloToggled → set_track_state

    def audible(self, track: str) -> bool:
        st = self.w.player.track_state(track)
        return st["solo"] or (not st["mute"] and st["gain"] > 0)

    def sync_all(self) -> None:
        for track in self.w.tracks.inst_rows:
            self.sync(track)

    def sync(self, track: str) -> None:
        """재생기 트랙을 상태에 맞춘다: 들려야 하면 올리고(필요하면 파일을 읽음), 아니면 내린다. 줄 표시도 맞춘다."""
        w = self.w
        name = track[len(TRACK_PREFIX):]
        st = w.player.track_state(track)
        row = w.tracks.inst_rows[track]
        row.set_controls(st["solo"], st["mute"], st["gain"])
        want = self.audible(track) and name in self.found and w.session.audio is not None
        selected = w.tracks.selected_track == track and row.isVisible()
        if want or selected:
            self._ensure(name)
        if not want and track in w.player.tracks:
            from amp.app.player import PlaybackState

            if w.player.state is PlaybackState.PLAYING:  # 믹서가 12ms 동안 음량을 내린 뒤 내린다
                QTimer.singleShot(80, lambda t=track: self._unload_if_quiet(t))
            else:
                w.player.set_track(track, None)
        if not want and not selected:
            self._cache.pop(name, None)  # 메모리를 돌려준다 (파형 요약은 작아서 둔다)
        self._sync_bass_button()
        if hasattr(self.w, "_sync_group_mutes"):
            self.w._sync_group_mutes()

    def _unload_if_quiet(self, track: str) -> None:
        if not self.audible(track) and track in self.w.player.tracks:
            self.w.player.set_track(track, None)

    def _ensure(self, name: str) -> None:
        """악기 소리를 메모리에 올리고(작업 스레드), 들려야 하면 재생기 트랙으로, 고른 줄이면 파형으로 보인다."""
        w = self.w
        track = TRACK_PREFIX + name
        if name in self._cache:
            self._apply(name)
            return
        if name in self._loading or name not in self.found:
            return
        self._loading.add(name)
        path, gen = str(self.found[name].path), self._gen

        def job(progress, token):
            import soundfile as sf

            from amp.audio.data import AudioData
            from amp.audio.peaks import WaveformPeaks

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = AudioData(np.ascontiguousarray(data), sr)
            return audio, WaveformPeaks.build(audio.mono(), sr)

        handle = w.runner.submit(f"{I.INSTRUMENTS[name]} 소리 불러오기", job)
        handle.finished.connect(lambda r, n=name, g=gen: self._on_loaded(n, g, *r))
        handle.failed.connect(lambda msg, _tb, n=name: self._on_load_failed(n, msg))
        w.tracks.inst_rows[track].status.setText("불러오는 중…")

    def _on_loaded(self, name: str, gen: int, audio, peaks) -> None:
        self._loading.discard(name)
        if gen != self._gen:  # 그사이 프로젝트가 바뀌었다
            return
        self._cache[name] = audio
        self._peaks[name] = peaks
        self.w.tracks.inst_rows[TRACK_PREFIX + name].status.setText("✓ 분리됨")
        self._apply(name)

    def _apply(self, name: str) -> None:
        w = self.w
        track = TRACK_PREFIX + name
        audio = self._cache.get(name)
        if audio is None:
            return
        if self.audible(track) and track not in w.player.tracks:
            w.player.set_track(track, audio)
            if track not in w.player.tracks:  # 길이·샘플레이트가 원본과 다르면 재생기가 받지 않는다
                self.panel.message.setText(f"{I.INSTRUMENTS[name]} 소리의 길이가 원본과 달라 재생하지 못했습니다.")
        if w.tracks.selected_track == track:
            w._show_track_waveform(track)
        if not self.audible(track) and w.tracks.selected_track != track:
            self._cache.pop(name, None)

    def _on_load_failed(self, name: str, msg: str) -> None:
        self._loading.discard(name)
        self.w.tracks.inst_rows[TRACK_PREFIX + name].status.setText("⚠ 읽기 실패")
        self.panel.message.setText(f"{I.INSTRUMENTS[name]} 소리를 불러오지 못했습니다: {msg}")

    def peaks(self, track: str):
        return self._peaks.get(track[len(TRACK_PREFIX):])

    def audio(self, track: str):
        return self._cache.get(track[len(TRACK_PREFIX):])

    def _sync_bass_button(self) -> None:
        bass = getattr(self.w, "bass", None)
        if bass is None:
            return
        b = bass.panel.listen
        b.blockSignals(True)
        b.setChecked(self.w.player.track_state(TRACK_PREFIX + "bass")["solo"])
        b.blockSignals(False)
        b.setEnabled("bass" in self.found and self.w.session.audio is not None)

    def _refresh_rows(self) -> None:
        w = self.w
        tp = w.tracks
        tp.set_instruments(TRACK_PREFIX + n for n in self.found)
        for name in I.INSTRUMENTS:
            track = TRACK_PREFIX + name
            row = tp.inst_rows[track]
            f = self.found.get(name)
            if f is None:
                if track in w.player.tracks:
                    w.player.set_track(track, None)
                self._cache.pop(name, None)
                continue
            try:
                rel = os.path.relpath(f.path, w.session.folder.paths.root)
            except (ValueError, AttributeError):
                rel = str(f.path)
            row.set_available(True, f"{rel}\n{self._durations.get(name, '')}{f.model_label}",
                              "불러오는 중…" if name in self._loading else "✓ 분리됨", "ok")
            row.setToolTip(str(f.path))
        if tp.selected_track in tp.inst_rows and not tp.inst_rows[tp.selected_track].isVisible():
            tp.select("original")
        self.sync_all()

    def reveal(self, name: str) -> None:
        f = self.found.get(name)
        if f is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(f.path.parent)))
