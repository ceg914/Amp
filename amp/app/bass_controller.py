"""베이스 TAB 화면 (v0.6.3): 오른쪽 '베이스 TAB' 탭, 베이스 편집 모드(두 번째 피아노 롤), GP5·텍스트 내보내기.

- 베이스 음표는 보컬과 따로 둔 문서(ProjectDocument)에 담는다. 보컬 음표·가사·검토 상태는 건드리지 않는다.
- 베이스 편집 모드에서는 가운데 피아노 롤을 베이스 피아노 롤로 바꾼다. 편집 방법은 보컬과 같고(끌기·더블클릭·Delete·
  Ctrl+E·Ctrl+M), 실행 취소(Ctrl+Z)는 지금 보이는 쪽의 기록을 쓴다.
- 박 위치는 보컬 프로젝트의 템포(BPM·첫 박)를 함께 쓴다. 템포를 고치면 TAB 미리보기도 바로 바뀐다.
- 다시 분석해도 고친 음표·추가한 음표는 지킨다 ('수정 유지', 보컬 재분석과 같은 규칙).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from amp.app.document import ProjectDocument
from amp.app.pianoroll.view import PianoRollView
from amp.app.widgets.keyboard import PianoKeyboard
from amp.core.model import AnalysisResult, ModelInfo, Project, Score
from amp.engine.bass import (BASS_NOTES_VERSION, GRIDS, OPEN_PREFS, TUNINGS, BassOptions, BassState, quantize,
                             summary, text_tab)

EDIT_SHORTCUT = "Ctrl+Shift+B"
ANALYZE_SHORTCUT = "F8"


class BassPanel(QWidget):
    optionsChanged = Signal()
    analyzeRequested = Signal()
    editModeToggled = Signal(bool)
    exportGp5Requested = Signal()
    exportTextRequested = Signal()
    exportAudioRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        intro = QLabel("분리한 베이스(Demucs htdemucs)에서 베이스 음을 찾아 TAB 악보로 만듭니다. 박 위치는 보컬 분석의 "
                       "BPM·첫 박을 씁니다.")
        intro.setWordWrap(True)
        intro.setProperty("role", "secondary")
        lay.addWidget(intro)
        form = QFormLayout()
        self.tuning = QComboBox()
        for k, (label, _m) in TUNINGS.items():
            self.tuning.addItem(label, k)
        self.max_fret = QSpinBox()
        self.max_fret.setRange(5, 24)
        self.open_strings = QComboBox()
        for k, (label, _w) in OPEN_PREFS.items():
            self.open_strings.addItem(label, k)
        self.grid = QComboBox()
        for v, label in GRIDS.items():
            self.grid.addItem(label, v)
        form.addRow("조율", self.tuning)
        form.addRow("최대 프렛", self.max_fret)
        form.addRow("개방현", self.open_strings)
        form.addRow("박자 격자", self.grid)
        lay.addLayout(form)
        self.use_original = QCheckBox("분리하지 않고 원본 그대로 분석 (베이스만 녹음한 파일)")
        lay.addWidget(self.use_original)
        self.analyze = QPushButton(f"베이스 분석 ({ANALYZE_SHORTCUT})")
        self.analyze.setProperty("variant", "primary")
        lay.addWidget(self.analyze)
        self.status = QLabel("아직 베이스를 분석하지 않았습니다.")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "secondary")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.status)
        self.edit = QPushButton(f"피아노 롤에서 베이스 편집 ({EDIT_SHORTCUT})")
        self.edit.setCheckable(True)
        lay.addWidget(self.edit)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        try:
            from amp.app.theme import mono_font

            self.preview.setFont(mono_font())
        except Exception:  # noqa: BLE001 - 글꼴이 없으면 기본 글꼴
            pass
        self.preview.setPlaceholderText("베이스를 분석하면 TAB 미리보기가 여기에 나옵니다.")
        lay.addWidget(self.preview, 1)
        row = QHBoxLayout()
        self.export_gp5 = QPushButton("GP5 내보내기…")
        self.export_text = QPushButton("텍스트 TAB 내보내기…")
        row.addWidget(self.export_gp5)
        row.addWidget(self.export_text)
        lay.addLayout(row)
        self.export_audio = QPushButton("베이스 소리 내보내기 (WAV·MP3)…")
        self.export_audio.setToolTip("분리해 둔 베이스(Demucs htdemucs)를 파일로 저장합니다. 베이스 분석을 한 번 하면 생깁니다.")
        lay.addWidget(self.export_audio)
        note = QLabel("GP5는 Guitar Pro·TuxGuitar·MuseScore에서 열립니다. 템포는 처음 BPM 하나만 씁니다. 자동 채보라 "
                      "줄·프렛 배치와 박 위치는 직접 확인하세요.")
        note.setWordWrap(True)
        note.setProperty("role", "secondary")
        lay.addWidget(note)
        self.set_options(BassOptions())
        for w in (self.tuning, self.open_strings, self.grid):
            w.currentIndexChanged.connect(lambda _i: self.optionsChanged.emit())
        self.max_fret.valueChanged.connect(lambda _v: self.optionsChanged.emit())
        self.analyze.clicked.connect(self.analyzeRequested.emit)
        self.edit.toggled.connect(self.editModeToggled.emit)
        self.export_gp5.clicked.connect(self.exportGp5Requested.emit)
        self.export_text.clicked.connect(self.exportTextRequested.emit)
        self.export_audio.clicked.connect(self.exportAudioRequested.emit)
        self.set_has_notes(False)

    def options(self) -> BassOptions:
        return BassOptions(tuning=str(self.tuning.currentData()), max_fret=int(self.max_fret.value()),
                           open_strings=str(self.open_strings.currentData()), grid=float(self.grid.currentData()))

    def set_options(self, o: BassOptions) -> None:
        widgets = (self.tuning, self.max_fret, self.open_strings, self.grid)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.tuning.setCurrentIndex(max(0, self.tuning.findData(o.tuning)))
            self.max_fret.setValue(int(o.max_fret))
            self.open_strings.setCurrentIndex(max(0, self.open_strings.findData(o.open_strings)))
            self.grid.setCurrentIndex(max(0, next((i for i in range(self.grid.count())
                                                   if abs(float(self.grid.itemData(i)) - o.grid) < 1e-9), 0)))
        finally:
            for w in widgets:
                w.blockSignals(False)

    def set_has_notes(self, has: bool) -> None:
        self.export_gp5.setEnabled(has)
        self.export_text.setEnabled(has)


class BassController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        w = self.w = window
        self.doc = ProjectDocument(self)
        self.view = PianoRollView(self.doc, w.timeline)
        self.view.timeline_owner = False
        self.keyboard = PianoKeyboard(self.view, self.view.mapper)
        self.view.hide()
        self.keyboard.hide()
        w.center_grid.addWidget(self.keyboard, 4, 0)
        w.center_grid.addWidget(self.view, 4, 1)
        self.panel = BassPanel()
        w.instrument_tabs.addTab(self.panel, "베이스 TAB")
        self.mode_label = QLabel("베이스 편집 중")
        self.mode_label.setProperty("role", "accent")
        self.mode_label.hide()
        w.statusBar().addPermanentWidget(self.mode_label)
        self.active = False
        self._loading = False
        self._source: dict = {}
        self._handle = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self.refresh_preview)
        w.undo_group.addStack(self.doc.undo_stack)

        w.doc.projectReset.connect(self.load_from_project)
        w.doc.tempoChanged.connect(self._on_main_tempo)
        self.doc.notesReplaced.connect(self._on_bass_edited)
        self.doc.analysisChanged.connect(self._on_bass_edited)
        self.panel.optionsChanged.connect(self._on_options)
        self.panel.analyzeRequested.connect(self.analyze)
        self.panel.editModeToggled.connect(self.set_active)
        self.panel.exportGp5Requested.connect(self.export_gp5)
        self.panel.exportTextRequested.connect(self.export_text)
        self.panel.exportAudioRequested.connect(self.export_audio)
        self.view.editFailed.connect(w._show_status_error)
        self.view.notePreviewRequested.connect(w.listening.preview_pitch)
        self.view.selectionInfoChanged.connect(lambda s: self.active and w.selection_label.setText(s))
        self.view.hoverInfoChanged.connect(lambda s: self.active and w.hover_label.setText(s))
        self.view.roll.selectionChanged.connect(w._update_actions)
        self.load_from_project()

    # --- 상태 --------------------------------------------------------------------

    @property
    def state(self) -> BassState | None:
        return self.w.doc.project.bass

    def tab(self):
        """지금 작업본으로 만든 TAB (보컬 프로젝트 템포 기준)."""
        return quantize(self.doc.score.notes(), self.w.doc.score.tempo, self.panel.options())

    def load_from_project(self) -> None:
        st = self.state
        main = self.w.doc.project
        proj = Project(audio=main.audio, analysis=st.analysis if st else None,
                       score=Score(main.score.tempo, list(st.notes) if st else []))
        self._loading = True
        try:
            self.doc.reset(proj, None)
        finally:
            self._loading = False
        self._source = dict(st.source) if st else {}
        self.panel.set_options(st.options if st else BassOptions())
        audio = getattr(self.w.session, "audio", None)
        if audio is not None:
            self.view.roll.set_audio_duration(audio.duration)
        self._update_status()
        self.refresh_preview()

    def set_audio_duration(self, duration: float) -> None:
        self.view.roll.set_audio_duration(duration)

    def _store(self) -> None:
        """베이스 작업본을 프로젝트에 넣고 저장할 변경으로 표시한다 (보컬 실행 취소 기록과는 따로)."""
        self.w.doc.project.bass = BassState(self.panel.options(), self.doc.project.analysis,
                                            tuple(sorted(self.doc.score.notes(), key=lambda n: n.start)),
                                            dict(self._source))
        self.w.doc.mark_modified()

    def _on_bass_edited(self, *_args) -> None:
        if self._loading:
            return
        self._store()
        self._update_status()
        self._preview_timer.start()

    def _on_options(self) -> None:
        if self.state is not None or self.doc.score.notes():
            self._store()
        self._preview_timer.start()

    def _on_main_tempo(self) -> None:
        self.doc._apply_tempo(self.w.doc.score.tempo)  # 같은 템포를 쓴다 (베이스 실행 취소 기록에 넣지 않음)
        self._preview_timer.start()

    def _update_status(self) -> None:
        notes = self.doc.score.notes()
        self.panel.set_has_notes(bool(notes))
        for act in (getattr(self.w, "act_bass_gp5", None), getattr(self.w, "act_bass_text", None)):
            if act is not None:
                act.setEnabled(bool(notes))
        a = self.doc.project.analysis
        if a is None and not notes:
            self.panel.status.setText("아직 베이스를 분석하지 않았습니다.")
            return
        statuses = [self.doc.note_status(n) for n in notes]
        edited = sum(1 for s in statuses if s != "original")
        src = self._source
        where = {"original": "원본 그대로"}.get(src.get("kind", ""), f"분리 결과 {src.get('store_key', '')}".strip())
        text = f"베이스 음표 {len(notes)}개 (고치거나 추가한 음표 {edited}개)"
        if a is not None:
            text += f" · {where} · 분석 {a.created_at[:16].replace('T', ' ')}"
        self.panel.status.setText(text)

    def refresh_preview(self) -> None:
        notes = self.doc.score.notes()
        if not notes:
            self.panel.preview.setPlainText("")
            return
        tab = self.tab()
        s = summary(tab)
        head = f"마디 {s['bars']}개 · 개방현 {s['open_strings']}번"
        if s["fret_range"]:
            head += f" · 프렛 {s['fret_range'][0]}~{s['fret_range'][1]}"
        if s["unplayable"]:
            head += f" · 칠 수 없는 음 {s['unplayable']}개 (조율·최대 프렛 밖)"
        bar = self.panel.preview.verticalScrollBar().value()
        self.panel.preview.setPlainText(head + "\n" + text_tab(tab, bars_per_line=2))
        self.panel.preview.verticalScrollBar().setValue(bar)

    # --- 분석 --------------------------------------------------------------------

    def analyze(self) -> None:
        w = self.w
        if w.session.busy:
            w.statusBar().showMessage("다른 분석이 끝난 뒤에 베이스를 분석하세요.", 5000)
            return
        from amp.app.lite import separation_installed

        opts = self.panel.options()
        handle = w.session.run_bass(opts, device=w.inspector.analysis.settings().device,
                                    use_original=self.panel.use_original.isChecked(),
                                    allow_separation=separation_installed())
        if handle is None:
            return
        self._handle = handle
        handle.finished.connect(lambda out, o=opts: self._on_result(out, o))
        handle.ended.connect(self._on_ended)
        self.panel.analyze.setEnabled(False)

    def _on_ended(self) -> None:
        self._handle = None
        self.panel.analyze.setEnabled(True)

    def _on_result(self, out, opts: BassOptions) -> None:
        w = self.w
        if w.session.audio is None:
            return
        result = AnalysisResult(
            duration=w.session.audio.duration, tempo=w.doc.score.tempo, notes=tuple(out.notes),
            pitch_curve=out.curve,
            models={"bass": ModelInfo("bass-pyin", BASS_NOTES_VERSION,
                                      params={**out.pitch_meta, "source": out.source, "reused": out.reused})},
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            audio_fingerprint=w.session.folder.meta.original.sha256 if w.session.folder else "",
            warnings=tuple(out.pitch_meta.get("warnings") or ()), settings=opts.to_dict())
        self._source = dict(out.source)
        kept = 0
        if self.doc.project.analysis is None and not self.doc.score.notes():
            self._loading = True
            try:
                self.doc.set_analysis(result)
            finally:
                self._loading = False
        else:
            plan = self.doc.apply_analysis(result, "keep_edits", "베이스 다시 분석")  # 고친 음표는 지킨다
            kept = len(getattr(plan, "kept", ()) or ())
        self._store()
        self._update_status()
        self.refresh_preview()
        msg = f"베이스 음표 {len(out.notes)}개를 찾았습니다."
        if out.source.get("separated"):
            msg += " 베이스를 새로 분리해 분리 결과에 더했습니다(보컬·반주 파일은 그대로)."
        if kept:
            msg += f" 고친 음표 {kept}개는 그대로 두었습니다."
        w.statusBar().showMessage(msg, 8000)
        if not self.active and out.notes:
            w.show_side_panel(self.panel)

    # --- 편집 모드 ----------------------------------------------------------------

    def set_active(self, on: bool) -> None:
        on = bool(on)
        w = self.w
        if on == self.active:
            self._sync_toggles()
            return
        self.active = on
        w.view.setVisible(not on)
        w.keyboard.setVisible(not on)
        self.view.setVisible(on)
        self.keyboard.setVisible(on)
        w.view.timeline_owner = not on
        self.view.timeline_owner = on
        shown = self.view if on else w.view
        shown.match_timeline()
        shown.set_playhead(w.current_time())
        if on and self.doc.score.notes():
            notes = self.doc.score.notes()
            self.view.center_on_pitch(sum(n.pitch for n in notes) / len(notes))
        w.undo_group.setActiveStack(self.doc.undo_stack if on else w.doc.undo_stack)
        self.mode_label.setVisible(on)
        shown.setFocus()
        self._sync_toggles()
        w._update_actions()

    def _sync_toggles(self) -> None:
        for b in (self.panel.edit, getattr(self.w, "act_bass_edit", None)):
            if b is not None and b.isChecked() != self.active:
                b.blockSignals(True)
                b.setChecked(self.active)
                b.blockSignals(False)

    # --- 내보내기 ------------------------------------------------------------------

    def _export_path(self, suffix: str, filt: str) -> str | None:
        w = self.w
        folder = w.session.folder
        base = os.path.splitext(folder.meta.original.file_name)[0] if folder else "bass"
        start_dir = str(folder.paths.exports) if folder else ""
        if folder:
            folder.paths.exports.mkdir(parents=True, exist_ok=True)
        path, _f = QFileDialog.getSaveFileName(w, "베이스 TAB 내보내기", os.path.join(start_dir, f"{base}-bass{suffix}"),
                                               filt)
        if not path:
            return None
        if not path.lower().endswith(suffix):
            path += suffix
        return path

    def _title(self) -> str:
        folder = self.w.session.folder
        return os.path.splitext(folder.meta.original.file_name)[0] if folder else ""

    def export_gp5(self) -> None:
        if not self.doc.score.notes():
            return
        path = self._export_path(".gp5", "Guitar Pro 5 (*.gp5)")
        if not path:
            return
        from amp.export.gp5 import write_gp5

        tab = self.tab()
        try:
            write_gp5(tab, path, self._title(), "")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.w, "GP5로 내보내지 못했습니다", str(e))
            return
        extra = f" 칠 수 없는 음 {tab.unplayable}개는 쉼표로 넣었습니다." if tab.unplayable else ""
        self.w.statusBar().showMessage(f"GP5로 내보냈습니다: {path}.{extra}", 8000)

    def stem_path(self) -> str | None:
        """분리해 둔 베이스 파일 (지금 쓰는 분리 결과 먼저). 없으면 None."""
        folder = self.w.session.folder
        if folder is None:
            return None
        from amp.engine.bass_pipeline import find_bass_stem

        found = find_bass_stem(folder.paths.stems, folder.meta.original.sha256)
        return str(found[1]) if found else None

    def export_audio(self) -> None:
        """베이스 stem을 WAV(24bit, 분리 결과 그대로)나 MP3(320kbps)로 내보낸다. 분리를 새로 하지는 않는다."""
        w = self.w
        folder = w.session.folder
        if folder is None:
            return
        src = self.stem_path()
        if src is None:
            original = self._source.get("kind") == "original"
            QMessageBox.information(
                w, "베이스 소리가 없습니다",
                ("원본을 그대로 분석해서 따로 뽑은 베이스가 없습니다. '분리하지 않고 원본 그대로 분석'을 끄고 " if original
                 else "아직 베이스를 분리하지 않았습니다. ")
                + "베이스 분석(F8)을 실행하면 베이스를 분리해 두고, 그 뒤에 내보낼 수 있습니다.")
            return
        if w.session.busy:
            w.statusBar().showMessage("분석이 끝난 뒤에 내보내세요.", 5000)
            return
        from amp.project.export import ExportOptions

        folder.paths.exports.mkdir(parents=True, exist_ok=True)
        path, chosen = QFileDialog.getSaveFileName(
            w, "베이스 소리 내보내기", os.path.join(str(folder.paths.exports), f"{folder.name}_bass.wav"),
            "WAV 24-bit (*.wav);;MP3 320kbps (*.mp3)")
        if not path:
            return
        fmt = "mp3" if path.lower().endswith(".mp3") or (not path.lower().endswith(".wav") and "MP3" in chosen) else "wav"
        if not path.lower().endswith("." + fmt):
            path += "." + fmt
        w.session.export_file(src, path, ExportOptions(fmt=fmt, wav_subtype="PCM_24", mp3_bitrate=320), "베이스 내보내기")

    def export_text(self) -> None:
        if not self.doc.score.notes():
            return
        path = self._export_path(".txt", "텍스트 (*.txt)")
        if not path:
            return
        from amp.export.tab_text import write_text_tab

        try:
            write_text_tab(self.tab(), path, self._title())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.w, "텍스트 TAB으로 내보내지 못했습니다", str(e))
            return
        self.w.statusBar().showMessage(f"텍스트 TAB으로 내보냈습니다: {path}", 8000)
