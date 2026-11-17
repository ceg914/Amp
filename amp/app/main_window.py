"""메인 창: 메뉴·툴바·위젯 배치와 작업 흐름(열기·저장·분석)을 묶는다."""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStyle,
    QToolButton,
    QWidget,
)

from amp import APP_DISPLAY_NAME, __version__
from amp.app.autosave import AutosaveManager, RecoveryEntry
from amp.app.document import ProjectDocument
from amp.app.pianoroll.view import PianoRollView
from amp.app.player import AudioPlayer, PlaybackState
from amp.app.timeline import TimelineState
from amp.app.widgets.keyboard import KEYBOARD_WIDTH, PianoKeyboard
from amp.app.widgets.ruler import TimeRuler
from amp.app.widgets.waveform import WaveformView
from amp.app.workers import TaskHandle, TaskRunner
from amp.audio.data import AudioData
from amp.audio.decoder import SUPPORTED_EXTENSIONS, file_fingerprint, load_audio
from amp.audio.peaks import WaveformPeaks
from amp.core.errors import EditError, ProjectFormatError
from amp.core.model import AudioRef
from amp.core.tempo import MAX_BPM, MIN_BPM
from amp.engine.pipeline import AnalysisPipeline, PipelineConfig
from amp.engine.registry import default_registry
from amp.storage.project_file import PROJECT_EXTENSION, load_project, save_project

log = logging.getLogger(__name__)

AUDIO_FILTER = "오디오 파일 (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"
PROJECT_FILTER = f"Amp 프로젝트 (*{PROJECT_EXTENSION})"
SNAP_CHOICES = (("스냅 끔", 0.0), ("1/4", 1.0), ("1/8", 0.5), ("1/16", 0.25), ("1/32", 0.125))

HELP_TEXT = """<h3>조작 방법</h3>
<ul>
<li>오디오 열기 <b>Ctrl+O</b>, 재생/일시정지 <b>Space</b>, 정지 <b>Esc</b>(재생 시작 위치로), 처음으로 <b>Home</b></li>
<li>눈금자·파형 클릭/드래그: 재생 위치 이동</li>
<li>음표 클릭: 선택, Ctrl+클릭: 선택 추가/해제, 빈 곳 드래그: 범위 선택</li>
<li>음표 드래그: 이동, 음표 양 끝 드래그: 길이 변경, 드래그 중 <b>Alt</b>: 스냅 해제</li>
<li>빈 곳 더블클릭: 음표 추가, 오른쪽 클릭: 분할·병합·삭제</li>
<li>↑↓: 반음 이동(Shift: 옥타브), ←→: 스냅 단위 이동, <b>Delete</b>: 삭제</li>
<li><b>Ctrl+E</b>: 재생 위치에서 선택 음표 분할, <b>Ctrl+M</b>: 선택 음표 병합</li>
<li>Ctrl+휠: 가로 확대, Ctrl+Shift+휠: 세로 확대, Shift+휠: 가로 스크롤</li>
</ul>
<p>음표 색: 파랑 = 분석 원본 그대로, 주황 = 수정됨, 초록 = 직접 추가</p>
<h3>현재 버전에서 동작하지 않는 것</h3>
<p>보컬/반주 분리, 실제 음높이·BPM·가사 분석, 가사 입력·편집, 발음 기호 생성,
MIDI·JSON 내보내기는 아직 구현하지 않았습니다.
'샘플 분석'은 오디오 길이만 보고 고정 패턴을 만드는 테스트용 기능입니다.</p>"""


class MainWindow(QMainWindow):
    def __init__(self, recovery_dir: str | None = None, autosave_interval_ms: int = 60_000) -> None:
        super().__init__()
        self.doc = ProjectDocument(self)
        self.timeline = TimelineState(self)
        self.runner = TaskRunner(self)
        self.player = AudioPlayer(self)
        self.registry = default_registry()
        self.audio: AudioData | None = None
        self.peaks: WaveformPeaks | None = None
        self._time = 0.0  # 재생 장치가 없을 때도 쓰는 재생 위치
        self._load_seq = 0
        self._analysis: TaskHandle | None = None
        self._tracked: list[TaskHandle] = []
        self._player_error_shown = False
        self.settings = QSettings()
        self.autosave = AutosaveManager(self.doc, recovery_dir, autosave_interval_ms, self)

        self._build_central()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._connect()
        self._update_title()
        self._update_actions()
        self._sync_tempo_widgets()
        self.resize(1280, 800)
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    # --- 화면 구성 ----------------------------------------------------------------

    def _build_central(self) -> None:
        self.view = PianoRollView(self.doc, self.timeline)
        self.ruler = TimeRuler(self.timeline, lambda: self.doc.score.tempo)
        self.waveform = WaveformView(self.timeline)
        self.keyboard = PianoKeyboard(self.view, self.view.mapper)

        corner = QLabel("시간")
        corner.setFixedWidth(KEYBOARD_WIDTH)
        corner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wave_label = QLabel("파형")
        wave_label.setFixedWidth(KEYBOARD_WIDTH)
        wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        central = QWidget()
        grid = QGridLayout(central)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(corner, 0, 0)
        grid.addWidget(self.ruler, 0, 1)
        grid.addWidget(wave_label, 1, 0)
        grid.addWidget(self.waveform, 1, 1)
        grid.addWidget(self.keyboard, 2, 0)
        grid.addWidget(self.view, 2, 1)
        grid.setRowStretch(2, 1)
        grid.setColumnStretch(1, 1)
        self.setCentralWidget(central)

    def _action(self, text: str, slot, shortcut=None, checkable: bool = False, checked: bool = False) -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            if isinstance(shortcut, (list, tuple)):
                a.setShortcuts([QKeySequence(s) for s in shortcut])
            else:
                a.setShortcut(QKeySequence(shortcut))
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
            a.toggled.connect(slot)
        else:
            a.triggered.connect(slot)
        return a

    def _build_actions(self) -> None:
        A = self._action
        self.act_open_audio = A("오디오 열기…", self.open_audio_dialog, QKeySequence.StandardKey.Open)
        self.act_open_project = A("프로젝트 열기…", self.open_project_dialog, "Ctrl+Shift+O")
        self.act_save = A("저장", self.save, QKeySequence.StandardKey.Save)
        self.act_save_as = A("다른 이름으로 저장…", self.save_as, "Ctrl+Shift+S")
        self.act_relink = A("오디오 파일 다시 연결…", self.relink_audio_dialog)
        self.act_quit = A("종료", self.close, "Ctrl+Q")

        self.act_undo = self.doc.undo_stack.createUndoAction(self, "실행 취소")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = self.doc.undo_stack.createRedoAction(self, "다시 실행")
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.act_delete = A("삭제", self.delete_selected, [QKeySequence.StandardKey.Delete, "Backspace"])
        self.act_split = A("재생 위치에서 분할", self.split_selected_at_playhead, "Ctrl+E")
        self.act_merge = A("선택한 음표 병합", self.merge_selected, "Ctrl+M")
        self.act_select_all = A("모두 선택", self.select_all, QKeySequence.StandardKey.SelectAll)
        self.act_revert = A("분석 원본으로 되돌리기", self.revert_to_analysis)

        self.act_analyze = A("샘플 분석 실행 (실제 AI 분석 아님)", self.run_sample_analysis, "F5")
        self.act_cancel_analysis = A("분석 취소", self.cancel_analysis)

        self.act_play = A("재생", self.toggle_play, "Space")
        self.act_stop = A("정지", self.stop, "Esc")
        self.act_home = A("처음으로", lambda: self.seek(0.0), "Home")

        self.act_zoom_in = A("가로 확대", lambda: self.view.zoom_time(1.25), ["Ctrl+=", "Ctrl++"])
        self.act_zoom_out = A("가로 축소", lambda: self.view.zoom_time(0.8), "Ctrl+-")
        self.act_zoom_fit = A("전체 보기", self.view.zoom_to_fit, "Ctrl+0")
        self.act_show_curve = A("피치 곡선 표시", self.view.roll.curve_item.setVisible, checkable=True, checked=True)
        self.act_follow = A("재생 위치 따라가기", self._set_follow, checkable=True, checked=True)
        self.act_help = A("조작법 및 구현 현황", self.show_help, "F1")
        self.act_about = A("정보", self.show_about)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("파일(&F)")
        for a in (self.act_open_audio, self.act_open_project, None, self.act_save, self.act_save_as, None,
                  self.act_relink, None, self.act_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("편집(&E)")
        for a in (self.act_undo, self.act_redo, None, self.act_delete, self.act_split, self.act_merge, None,
                  self.act_select_all, None, self.act_revert):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("재생(&P)")
        for a in (self.act_play, self.act_stop, self.act_home):
            m.addAction(a)
        m = mb.addMenu("분석(&A)")
        m.addAction(self.act_analyze)
        m.addAction(self.act_cancel_analysis)
        m = mb.addMenu("보기(&V)")
        for a in (self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit, None, self.act_show_curve, self.act_follow):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("도움말(&H)")
        m.addAction(self.act_help)
        m.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("재생")
        tb.setObjectName("transport")
        tb.setMovable(False)
        icon = self.style().standardIcon
        self.act_home.setIcon(icon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.act_play.setIcon(icon(QStyle.StandardPixmap.SP_MediaPlay))
        self.act_stop.setIcon(icon(QStyle.StandardPixmap.SP_MediaStop))
        tb.addAction(self.act_home)
        tb.addAction(self.act_play)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(190)
        self.time_label.setStyleSheet("font-family: Consolas, 'D2Coding', monospace; font-size: 14px; padding: 0 8px;")
        tb.addWidget(self.time_label)
        tb.addSeparator()
        tb.addWidget(QLabel(" BPM "))
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(MIN_BPM, MAX_BPM)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setKeyboardTracking(False)
        self.bpm_spin.valueChanged.connect(self._on_bpm_changed)
        tb.addWidget(self.bpm_spin)
        tb.addWidget(QLabel("  격자 "))
        self.snap_combo = QComboBox()
        for label, value in SNAP_CHOICES:
            self.snap_combo.addItem(label, value)
        self.snap_combo.setCurrentIndex(3)
        self.snap_combo.currentIndexChanged.connect(lambda _i: self.view.set_snap_beats(float(self.snap_combo.currentData())))
        tb.addWidget(self.snap_combo)
        tb.addSeparator()
        tb.addAction(self.act_analyze)
        tb.addAction(self.act_zoom_fit)
        self._update_time_label(0.0)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.selection_label = QLabel()
        self.analysis_label = QLabel()
        legend = QLabel(
            "<span style='color:#4682c8'>■</span> 원본 "
            "<span style='color:#dc8c3c'>■</span> 수정 "
            "<span style='color:#5ab46e'>■</span> 추가"
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.progress_label = QLabel()
        self.progress_label.hide()
        self.cancel_button = QToolButton()
        self.cancel_button.setText("취소")
        self.cancel_button.clicked.connect(self._cancel_tracked)
        self.cancel_button.hide()
        sb.addWidget(self.selection_label, 1)
        sb.addPermanentWidget(self.progress_label)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_button)
        sb.addPermanentWidget(legend)
        sb.addPermanentWidget(self.analysis_label)

    def _connect(self) -> None:
        self.player.positionChanged.connect(self._on_position)
        self.player.stateChanged.connect(self._on_player_state)
        self.player.errorOccurred.connect(self._on_player_error)
        self.ruler.seekRequested.connect(self.seek)
        self.waveform.seekRequested.connect(self.seek)
        self.view.selectionInfoChanged.connect(self.selection_label.setText)
        self.view.roll.selectionChanged.connect(self._update_actions)
        self.view.editFailed.connect(self._show_status_error)
        self.doc.dirtyChanged.connect(self._update_title)
        self.doc.filePathChanged.connect(self._update_title)
        self.doc.projectReset.connect(self._on_project_reset)
        self.doc.tempoChanged.connect(self._sync_tempo_widgets)

    # --- 상태 갱신 ----------------------------------------------------------------

    def _update_title(self, *_args) -> None:
        name = os.path.basename(self.doc.file_path) if self.doc.file_path else "새 프로젝트"
        if self.doc.project.audio and not self.doc.file_path:
            name += f" ({os.path.basename(self.doc.project.audio.path)})"
        self.setWindowTitle(f"{name}[*] - {APP_DISPLAY_NAME}")
        self.setWindowModified(self.doc.is_dirty)

    def _update_actions(self) -> None:
        n_sel = len(self.view.selected_ids())
        has_audio = self.audio is not None
        running = self._analysis is not None
        self.act_delete.setEnabled(n_sel > 0)
        self.act_split.setEnabled(n_sel > 0)
        self.act_merge.setEnabled(n_sel >= 2)
        self.act_revert.setEnabled(self.doc.project.analysis is not None)
        self.act_analyze.setEnabled(has_audio and not running)
        self.act_cancel_analysis.setEnabled(running)
        self.act_play.setEnabled(has_audio)
        self.act_stop.setEnabled(has_audio)
        self.act_relink.setEnabled(self.doc.project.audio is not None)

    def _update_analysis_label(self) -> None:
        a = self.doc.project.analysis
        if a is None:
            self.analysis_label.setText("분석 결과 없음")
            self.analysis_label.setToolTip("")
        elif a.is_placeholder:
            self.analysis_label.setText("<b style='color:#e0a040'>샘플 데이터 (실제 분석 아님)</b>")
            self.analysis_label.setToolTip("\n".join(f"{k}: {m.name} v{m.version}" for k, m in a.models.items()))
        else:
            self.analysis_label.setText("분석 결과: " + ", ".join(sorted({m.name for m in a.models.values()})))

    def _sync_tempo_widgets(self) -> None:
        self.bpm_spin.blockSignals(True)
        self.bpm_spin.setValue(self.doc.score.tempo.initial_bpm)
        self.bpm_spin.blockSignals(False)
        self.ruler.update()
        self._update_time_label(self.current_time())

    def _on_project_reset(self) -> None:
        self._update_title()
        self._update_actions()
        self._update_analysis_label()
        self._sync_tempo_widgets()
        self.view.center_on_content()

    def _on_bpm_changed(self, value: float) -> None:
        try:
            self.doc.set_bpm(value)
        except EditError as e:
            self.statusBar().showMessage(str(e), 5000)
            self._sync_tempo_widgets()

    def _set_follow(self, on: bool) -> None:
        self.view.follow_playhead = on

    # --- 재생 -----------------------------------------------------------------

    def current_time(self) -> float:
        return self.player.position() if self.player.has_audio else self._time

    def toggle_play(self) -> None:
        if self.audio is None:
            return
        self.player.toggle()

    def stop(self) -> None:
        self.player.stop()

    def seek(self, t: float) -> None:
        if self.player.has_audio:
            self.player.seek(t)
        else:
            self._on_position(max(0.0, t))

    def _on_position(self, t: float) -> None:
        self._time = t
        self.view.set_playhead(t)
        self.waveform.set_playhead(t)
        self.ruler.set_playhead(t)
        self._update_time_label(t)

    def _update_time_label(self, t: float) -> None:
        m, s = divmod(max(t, 0.0), 60)
        bar, beat = self.doc.score.tempo.bar_beat(t)
        self.time_label.setText(f"{int(m):02d}:{s:06.3f}   {bar}.{beat:.2f}")

    def _on_player_state(self, state: PlaybackState) -> None:
        playing = state is PlaybackState.PLAYING
        self.act_play.setText("일시정지" if playing else "재생")
        pix = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.act_play.setIcon(self.style().standardIcon(pix))

    def _on_player_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        if not self._player_error_shown:
            self._player_error_shown = True
            QMessageBox.warning(self, "재생 오류", message)

    # --- 백그라운드 작업 표시 ----------------------------------------------------------

    def _track(self, handle: TaskHandle, label: str) -> None:
        self._tracked.append(handle)
        handle.progress.connect(lambda f, msg, h=handle: self._on_task_progress(h, f, msg))
        handle.ended.connect(lambda h=handle: self._on_task_ended(h))
        self.progress.setValue(0)
        self.progress_label.setText(label)
        for w in (self.progress, self.progress_label, self.cancel_button):
            w.show()

    def _on_task_progress(self, handle: TaskHandle, fraction: float, message: str) -> None:
        if self._tracked and self._tracked[-1] is handle:
            self.progress.setValue(int(fraction * 1000))
            if message:
                self.progress_label.setText(message)

    def _on_task_ended(self, handle: TaskHandle) -> None:
        if handle in self._tracked:
            self._tracked.remove(handle)
        if not self._tracked:
            for w in (self.progress, self.progress_label, self.cancel_button):
                w.hide()

    def _cancel_tracked(self) -> None:
        if self._tracked:
            self._tracked[-1].cancel()

    def _show_status_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _show_error(self, title: str, message: str, details: str = "") -> None:
        box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
        if details:
            box.setDetailedText(details)
        box.exec()

    # --- 오디오 ---------------------------------------------------------------

    def open_audio_dialog(self) -> None:
        if not self.maybe_save():
            return
        start = self.settings.value("paths/last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "오디오 열기", start, AUDIO_FILTER)
        if path:
            self.settings.setValue("paths/last_dir", os.path.dirname(path))
            self.load_audio(path, new_project=True)

    def load_audio(self, path: str, new_project: bool) -> TaskHandle:
        """오디오를 백그라운드에서 디코딩한다. new_project=False면 열린 프로젝트에 연결한다."""
        self._load_seq += 1
        seq = self._load_seq
        self.player.stop()

        def job(progress, token):
            audio = load_audio(path, progress, lambda: token.cancelled)
            progress(1.0, "파형 계산 중")
            peaks = WaveformPeaks.build(audio.mono(), audio.sample_rate)
            return audio, peaks, file_fingerprint(path)

        handle = self.runner.submit("오디오 불러오기", job)
        def loaded(result) -> None:
            if seq == self._load_seq:  # 더 나중에 시작한 불러오기가 있으면 결과를 버린다
                self._on_audio_loaded(path, result, new_project)

        handle.finished.connect(loaded)
        handle.failed.connect(lambda msg, tb: self._show_error("오디오를 불러오지 못했습니다", msg, tb))
        handle.cancelled.connect(lambda: self.statusBar().showMessage("불러오기를 취소했습니다.", 5000))
        self._track(handle, "오디오 불러오는 중")
        return handle

    def _on_audio_loaded(self, path: str, result, new_project: bool) -> None:
        audio, peaks, fingerprint = result
        ref = AudioRef(path, fingerprint, audio.duration, audio.sample_rate, audio.channels)
        old = self.doc.project.audio
        self.audio, self.peaks = audio, peaks
        self.player.set_audio(audio)
        self.waveform.set_peaks(peaks)
        self.view.roll.set_audio_duration(audio.duration)
        if new_project:
            self.doc.set_audio(ref)
            self.view.zoom_to_fit()
        else:
            if old is not None and old.fingerprint and old.fingerprint != fingerprint:
                QMessageBox.warning(
                    self, "오디오 파일이 다릅니다",
                    "프로젝트를 저장할 때와 오디오 파일 내용이 다릅니다.\n음표 위치가 맞지 않을 수 있습니다.",
                )
            if old is None or old.path != path or old.fingerprint != fingerprint:
                self.doc.relink_audio(ref)
        self._on_position(0.0)
        self._update_actions()
        self.statusBar().showMessage(
            f"{os.path.basename(path)} — {audio.duration:.1f}초, {audio.sample_rate}Hz, {audio.channels}채널", 8000
        )

    def relink_audio_dialog(self) -> None:
        start = os.path.dirname(self.doc.project.audio.path) if self.doc.project.audio else ""
        path, _ = QFileDialog.getOpenFileName(self, "오디오 파일 다시 연결", start, AUDIO_FILTER)
        if path:
            self.load_audio(path, new_project=False)

    def _clear_audio(self) -> None:
        self._load_seq += 1
        self.audio = self.peaks = None
        self.player.set_audio(None)
        self.waveform.set_peaks(None)
        self.view.roll.set_audio_duration(0.0)

    # --- 프로젝트 ----------------------------------------------------------------

    def open_project_dialog(self) -> None:
        if not self.maybe_save():
            return
        start = self.settings.value("paths/last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", start, PROJECT_FILTER)
        if path:
            self.settings.setValue("paths/last_dir", os.path.dirname(path))
            self.open_project(path)

    def open_project(self, path: str, *, as_recovery_of: str | None = None) -> bool:
        try:
            project = load_project(path)
        except ProjectFormatError as e:
            self._show_error("프로젝트를 열 수 없습니다", str(e))
            return False
        self.cancel_analysis()
        self._clear_audio()
        if as_recovery_of is None:
            self.doc.reset(project, path)
        else:  # 복구본은 원래 경로로 저장되도록 하고, 저장 전까지 변경 상태로 둔다
            self.doc.reset(project, as_recovery_of or None, dirty=True)
        self._load_project_audio()
        return True

    def _load_project_audio(self) -> None:
        ref = self.doc.project.audio
        if ref is None:
            return
        if os.path.exists(ref.path):
            self.load_audio(ref.path, new_project=False)
            return
        answer = QMessageBox.question(
            self, "오디오 파일 없음",
            f"프로젝트의 오디오 파일을 찾을 수 없습니다.\n{ref.path}\n\n파일 위치를 직접 지정하시겠습니까?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.relink_audio_dialog()

    def save(self) -> bool:
        if not self.doc.file_path:
            return self.save_as()
        return self._write(self.doc.file_path)

    def save_as(self) -> bool:
        start = self.doc.file_path or ""
        if not start and self.doc.project.audio:
            start = os.path.splitext(self.doc.project.audio.path)[0] + PROJECT_EXTENSION
        path, _ = QFileDialog.getSaveFileName(self, "다른 이름으로 저장", start, PROJECT_FILTER)
        if not path:
            return False
        if not path.lower().endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        return self._write(path)

    def _write(self, path: str) -> bool:
        try:
            save_project(self.doc.project, path)
        except OSError as e:
            self._show_error("저장하지 못했습니다", f"{path}\n{e}")
            return False
        self.doc.mark_saved(path)
        self.autosave.discard()
        self.statusBar().showMessage(f"저장했습니다: {path}", 5000)
        return True

    def maybe_save(self) -> bool:
        """저장하지 않은 변경이 있으면 묻는다. 계속 진행해도 되면 True."""
        if not self.doc.is_dirty:
            return True
        answer = QMessageBox.question(
            self, "저장하지 않은 변경",
            "저장하지 않은 변경 사항이 있습니다. 저장하시겠습니까?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    # --- 복구 -----------------------------------------------------------------

    def check_recovery(self) -> None:
        entries = AutosaveManager.find_orphans(self.autosave.directory)
        if not entries:
            return
        latest = entries[0]
        target = latest.original_path or "저장하지 않은 프로젝트"
        answer = QMessageBox.question(
            self, "작업 복구",
            f"이전 실행이 비정상 종료된 것 같습니다.\n자동 저장본({latest.saved_at}, {target})을 복구하시겠습니까?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.restore(latest)
            others = entries[1:]
        else:
            others = entries
        for e in others:
            AutosaveManager.remove_entry(e)

    def restore(self, entry: RecoveryEntry) -> None:
        ok = self.open_project(entry.project_path, as_recovery_of=entry.original_path or "")
        AutosaveManager.remove_entry(entry)
        if ok:
            self.statusBar().showMessage("자동 저장본을 복구했습니다. 확인한 뒤 저장해 주세요.", 10000)

    # --- 편집 -----------------------------------------------------------------

    def _guard(self, fn) -> None:
        try:
            fn()
        except EditError as e:
            self.statusBar().showMessage(str(e), 5000)

    def delete_selected(self) -> None:
        self._guard(lambda: self.doc.delete_notes(self.view.selected_ids()))

    def split_selected_at_playhead(self) -> None:
        self._guard(lambda: self.doc.split_notes_at(self.view.selected_ids(), self.current_time()))

    def merge_selected(self) -> None:
        self._guard(lambda: self.doc.merge_notes(self.view.selected_ids()))

    def select_all(self) -> None:
        self.view.roll.select_ids(list(self.view.roll.items_by_id))

    def revert_to_analysis(self) -> None:
        self._guard(self.doc.revert_to_analysis)

    # --- 분석 -----------------------------------------------------------------

    def run_sample_analysis(self) -> None:
        if self.audio is None or self._analysis is not None:
            return
        if self.doc.project.analysis is not None or len(self.doc.score) > 0:
            answer = QMessageBox.question(
                self, "분석 실행",
                "현재 음표를 새 분석 결과로 바꿉니다. 실행 취소 기록도 지워집니다.\n계속하시겠습니까?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        audio = self.audio
        fingerprint = self.doc.project.audio.fingerprint if self.doc.project.audio else ""
        pipeline = AnalysisPipeline(self.registry, PipelineConfig())
        handle = self.runner.submit("분석", lambda progress, token: pipeline.run(audio, progress, token, fingerprint))
        self._analysis = handle

        def done(result) -> None:
            if self.audio is not audio:
                return  # 분석 도중 다른 오디오를 열었다
            self.doc.set_analysis(result)
            note = " — 샘플 데이터이며 실제 분석 결과가 아닙니다." if result.is_placeholder else ""
            self.statusBar().showMessage(f"음표 {len(result.notes)}개를 만들었습니다{note}", 10000)

        def ended() -> None:
            if self._analysis is handle:
                self._analysis = None
            self._update_actions()

        handle.finished.connect(done)
        handle.failed.connect(lambda msg, tb: self._show_error("분석 실패", msg, tb))
        handle.cancelled.connect(lambda: self.statusBar().showMessage("분석을 취소했습니다.", 5000))
        handle.ended.connect(ended)
        self._track(handle, "분석 중")
        self._update_actions()

    def cancel_analysis(self) -> None:
        if self._analysis is not None:
            self._analysis.cancel()

    # --- 기타 -----------------------------------------------------------------

    def show_help(self) -> None:
        QMessageBox.information(self, "조작법 및 구현 현황", HELP_TEXT)

    def show_about(self) -> None:
        QMessageBox.about(self, "정보", f"<b>{APP_DISPLAY_NAME}</b> {__version__}<br>Python + PySide6 보컬 채보 편집기 (개발 중)")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self.maybe_save():
            event.ignore()
            return
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.player.shutdown()
        if not self.runner.shutdown(5000):
            log.warning("백그라운드 작업이 5초 안에 끝나지 않았습니다.")
        self.autosave.shutdown()
        event.accept()
