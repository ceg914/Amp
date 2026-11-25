"""메인 창 (SONATLAS_UI_SPEC 5장 레이아웃).

    ┌ 메뉴 / 도구막대: 프로젝트명·열기·저장·폴더 / 재생 / 타임코드 / BPM / 분석 실행 ┐
    │ 좌: 프로젝트·트랙 │ 중앙: 눈금자·파형·피아노 롤 │ 우: 선택 음표·분석 설정 │
    └ 상태 표시줄: 진행률·취소·메시지 (로그는 아래 도크로 펼침) ┘

작업 흐름(프로젝트 생성·열기·저장·분석·내보내기)은 ProjectSession이 맡고, 이 클래스는 화면과 연결만 한다.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence, QUndoGroup
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QToolButton,
    QWidget,
)

from amp import APP_DISPLAY_NAME, __version__
from amp.core.install_hints import script as install_script
from amp.app import theme
from amp.app.autosave import AutosaveManager, RecoveryEntry
from amp.app.dialogs import (
    ExportDialog,
    NewProjectDialog,
    NoVocalRangesDialog,
    PitchRegionsDialog,
    ReanalysisDialog,
    RoformerModelDialog,
    StemImportDialog,
)
from amp.app.document import ProjectDocument
from amp.app.compare_controller import CompareController
from amp.app.export_controller import ExportController
from amp.app.hybrid_controller import HybridController
from amp.app.lite import lite_mode, separation_installed
from amp.app.model_compare_controller import ModelCompareController
from amp.app.quality_controller import QualityController
from amp.app.suggestions_controller import SuggestionsController
from amp.app.listening_controller import ListeningController
from amp.app.review_controller import ReviewController
from amp.app.lyrics_controller import LyricsController
from amp.app.vocal_controller import VocalDetailController
from amp.app.pianoroll.view import PianoRollView
from amp.app.player import AudioPlayer, PlaybackState
from amp.app.session import ProjectSession
from amp.app.theme import tokens as T
from amp.app.theme.icons import ICON_SIZE, icon
from amp.app.timeline import TimelineState
from amp.app.widgets.inspector import Inspector
from amp.app.widgets.keyboard import KEYBOARD_WIDTH, PianoKeyboard
from amp.app.widgets.log_panel import LogPanel
from amp.app.widgets.lyric_lane import LyricLane
from amp.app.widgets.lyrics_panel import LyricsPanel
from amp.app.widgets.ruler import TimeRuler
from amp.app.widgets.track_panel import TRACK_HINTS, TRACK_LABELS, TrackPanel
from amp.app.widgets.spectrogram import SpectrogramView
from amp.app.widgets.waveform import WaveformView
from amp.app.workers import TaskHandle, TaskRunner
from amp.audio.decoder import SUPPORTED_EXTENSIONS
from amp.core.errors import AmpError, EditError, ProjectFormatError
from amp.core.model import AnalysisResult
from amp.core.tempo import MAX_BPM, MIN_BPM, TempoMap
from amp.engine.merge import KEEP_EDITS, REPLACE_ALL, plan_reanalysis, summarize
from amp.engine.settings import SEPARATION_LABELS, AnalysisSettings, normalize_ranges
from amp.project.layout import PROJECT_FILE, is_project_folder
from amp.storage.project_file import PROJECT_EXTENSION, load_project

log = logging.getLogger(__name__)
STALL_TICK_MS = 100
STALL_LOG_MS = 500

AUDIO_FILTER = "오디오 파일 (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"
OPEN_FILTER = f"SONATLAS 프로젝트 ({PROJECT_FILE});;이전 형식 프로젝트 (*{PROJECT_EXTENSION})"
SNAP_CHOICES = (("스냅 끔", 0.0), ("1/4", 1.0), ("1/8", 0.5), ("1/16", 0.25), ("1/32", 0.125))
SETTINGS_KEY = "analysis/settings"
MODEL_FILES_KEY = "separation/roformer"  # 컴퓨터마다 다른 모델 파일 경로 (프로젝트에는 저장하지 않음)

HELP_TEXT = """<h3>조작 방법</h3>
<ul>
<li>새 프로젝트(오디오 열기) <b>Ctrl+O</b>, 프로젝트 열기 <b>Ctrl+Shift+O</b>, 저장 <b>Ctrl+S</b></li>
<li>재생/일시정지 <b>Space</b>, 정지 <b>Esc</b>(재생 시작 위치로), 처음으로 <b>Home</b></li>
<li>눈금자·파형 클릭/드래그: 재생 위치 이동. 왼쪽 트랙 줄을 누르면 그 트랙의 파형을 보여 줍니다.</li>
<li>트랙: 헤드폰 = 그 트랙만 듣기(솔로), 스피커 = 음소거, 막대 = 음량</li>
<li>음표 클릭: 선택, Ctrl+클릭: 선택 추가/해제, 빈 곳 드래그: 범위 선택</li>
<li>음표 드래그: 이동, 음표 양 끝 드래그: 길이 변경, 드래그 중 <b>Alt</b>: 스냅 해제</li>
<li>빈 곳 더블클릭: 음표 추가, 오른쪽 클릭: 분할·병합·삭제</li>
<li>↑↓: 반음 이동(Shift: 옥타브), ←→: 스냅 단위 이동, <b>Delete</b>: 삭제</li>
<li><b>Ctrl+E</b>: 재생 위치에서 선택 음표 분할, <b>Ctrl+M</b>: 선택 음표 병합</li>
<li>Ctrl+휠: 가로 확대, Ctrl+Shift+휠: 세로 확대, Shift+휠: 가로 스크롤</li>
</ul>
<h3>가사 (오른쪽 '가사' 탭, 피아노 롤 아래 가사 레인)</h3>
<ul>
<li>가사 인식 <b>F6</b> (Whisper, 기본으로 정렬·음표 연결까지 이어서 함), 시간 정렬 <b>F7</b>, 바뀐 부분만 다시 정렬 <b>Shift+F7</b>, 가사 표시 켜기·끄기 <b>Ctrl+L</b></li>
<li>가사 탭 편집기에서 고치고 [적용]: 바뀐 줄만 다시 나누고, 그대로인 음절의 시간·음표 연결은 남깁니다</li>
<li>가사 레인: 음절 클릭 = 그 위치로 이동, 막대 끌기 = 시간 이동, 양 끝 끌기 = 경계 조절(맞닿은 이웃도 함께),
더블클릭 = 그 음절부터 재생, 오른쪽 클릭 = 음표 연결·나누기·합치기·구간 반복 재생(<b>Ctrl+R</b>)</li>
<li>막대 테두리: 실선 = 모델 검출, 점선 = 추정, 굵은 선 = 직접 지정, 주황 ↻ = 다시 정렬 필요, ! = 정렬 신뢰도 낮음</li>
<li>음표 아래 글자: 그 음표에서 시작하는 음절('사랑' = 한 음표에 두 음절), '–'와 민트 선 = 같은 음절이 이어지는 멜리스마</li>
</ul>
<p>음표 구분: 꽉 찬 바이올렛 = 분석 원본 그대로, 점선 테두리 + 삼각 표식 = 수정함,
옅은 채움 + '+' = 직접 추가, 점 테두리 = 장식음. 민트 곡선 = 실제 피치.</p>
<h3>청취 도구 (STEP 5, 위쪽 두 번째 도구막대·재생 메뉴)</h3>
<ul>
<li>반복 구간: 파형을 끌어 고르기, 선택 음표 구간 반복(<b>Ctrl+Shift+R</b>), 가사 음절 '구간 반복', 시작·끝 직접 입력.
음표·음절로 정할 때는 '여유'만큼 앞뒤를 넓힙니다. 반복 켜기·끄기 <b>Alt+L</b>, 시작·끝 = 재생 위치 <b>Alt+[</b> / <b>Alt+]</b></li>
<li>속도 0.5·0.75·0.9·1배: 음높이를 유지합니다. 느리게 <b>Alt+,</b> 빠르게 <b>Alt+.</b> 정상 <b>Alt+/</b>.
반복 구간(없으면 재생 위치부터 60초)만 미리 늘리며, 화면의 재생 위치는 원본 시간입니다.</li>
<li>음표 미리 듣기: 음표를 누르거나 위아래로 옮기면 그 음높이를 내장 합성음으로 들려줍니다(가수 음색이 아님).
선택 음표 소리 듣기 <b>Alt+P</b>, 음표 소리를 보컬과 함께 재생 <b>Alt+M</b></li>
<li>보기 &gt; 파형·스펙트로그램: 숨기기·파형만·스펙트로그램만·둘 다·스펙트로그램 + 피치 곡선. 아래 가장자리를 끌어 높이 조절.
트랙 패널에서 고른 트랙(원본·보컬·반주)을 보여 줍니다.</li>
</ul>
<h3>보컬 분리 모델 비교 (STEP 5, 분석 &gt; 모델 비교 모드 <b>Ctrl+Shift+M</b>)</h3>
<ul>
<li>분석 &gt; 분리 모델 비교 실행: htdemucs·Mel-Band RoFormer를 하나씩 차례로 분리하고 같은 설정으로 채보합니다.
결과는 모델별로 stems/models/에 따로 남고, 같은 모델·설정이면 다시 계산하지 않습니다. GPU 메모리가 부족하면 CPU로 바꿀지 묻습니다.</li>
<li>A/B 전환 <b>Ctrl+B</b>: 재생 위치·반복 구간·속도를 유지한 채 짧게 크로스페이드합니다. '청취용 음량 맞춤'은 재생 음량만 바꿉니다.</li>
<li>스펙트로그램: A만·B만·좌우(가르는 선을 끌어 옮김)·위아래·차이. 두 쪽 모두 같은 dB 범위를 씁니다.</li>
<li>피아노 롤: A 음표 = 바이올렛 점선(행 위쪽), B 음표 = 민트 점점선(행 아래쪽). 비교용 그림이라 편집되지 않습니다.</li>
<li>'모델 차이' 탭과 검토 탭의 '모델 비교' 항목은 차이만 보여 줍니다. 정답 채보가 없으므로 어느 쪽이 정확하다고 판정하지 않습니다.</li>
<li>[A/B를 분석에 사용…]: 바뀔 항목을 먼저 보여 주고, 사용자 편집은 '수정 유지'로 지킵니다. 다른 모델 결과 파일은 지우지 않습니다.</li>
</ul>
<h3>Hybrid 보컬 (v0.6, 분석 &gt; Hybrid 편집 모드 <b>Ctrl+Shift+H</b>)</h3>
<ul>
<li>구간마다 분리 모델을 골라 하나의 보컬로 합칩니다. 파형 위 Hybrid 레인에서 경계를 끌어 옮기고, 오른쪽 클릭으로
나누기·합치기·모델 지정·지우기를 합니다. 모델 이름이 구간 안에 쓰이고, 검증이 필요한 구간은 빗금과 '!'로 표시합니다.</li>
<li>경계는 샘플 단위로 맞추고 기본 30ms 직선 교차(크로스페이드)로 잇습니다. 원본 분리 파일은 바꾸지 않습니다.</li>
<li>[전후 비교 듣기]: A = 지금 분석 소스, B = 새 Hybrid (Ctrl+B 전환). [분석 소스로 사용]: 바뀐 범위 근처의 피치만 다시
계산하고, 바뀔 항목을 보여 준 뒤 사용자 편집을 지키며 반영합니다.</li>
</ul>
<h3>베이스 TAB (v0.6.3, 오른쪽 '베이스 TAB' 탭)</h3>
<ul>
<li>조율·최대 프렛·개방현·박자 격자를 고르고 <b>베이스 분석</b>(<b>F8</b>). 박 위치는 보컬 분석의 BPM·첫 박을 씁니다.
Demucs htdemucs의 베이스 소리를 쓰며, 없으면 한 번 다시 분리해 베이스만 더합니다(보컬·반주 파일은 그대로).</li>
<li><b>베이스 편집 모드</b>(<b>Ctrl+Shift+B</b>): 가운데 피아노 롤이 베이스로 바뀝니다. 편집·단축키는 보컬과 같고, 실행 취소는 보이는 쪽 기록만 되돌립니다.
다시 분석해도 고친 음표는 남습니다.</li>
<li>미리보기 TAB은 지금 음표로 만듭니다. 내보내기 &gt; 베이스 TAB 내보내기(GP5·텍스트). GP5에는 처음 BPM 하나만 들어갑니다.</li>
<li>내보내기 &gt; 베이스 소리 내보내기: 분리해 둔 베이스를 WAV(24bit)·MP3(320kbps)로 저장합니다.</li>
</ul>
<h3>아직 없는 기능</h3>
<p>자음·모음 단위 시간은 아직 구현하지 않았습니다. 내보내기 &gt; 채보·가사 내보내기에서 USTX·UFData·MIDI·MusicXML·JSON·CSV·LRC로 저장합니다(조옮김·곡 음역 표시).
'샘플 데이터 생성'은 오디오 길이만 보고 고정 패턴을 만드는 시험용 기능입니다.</p>"""


class MainWindow(QMainWindow):
    def __init__(self, recovery_dir: str | None = None, autosave_interval_ms: int = 60_000,
                 projects_root: str | None = None) -> None:
        super().__init__()
        self.settings = QSettings()
        self.doc = ProjectDocument(self)
        self.timeline = TimelineState(self)
        self.runner = TaskRunner(self, max_threads=3)
        self.player = AudioPlayer(self)
        self.session = ProjectSession(self.doc, self.player, self.runner, self.settings, projects_root, self)
        self.autosave = AutosaveManager(self.doc, recovery_dir, autosave_interval_ms, self)
        # 화면 멈춤 감시 (진단용): 이벤트 루프가 0.5초 넘게 멈추면 그때 돌던 작업과 함께 로그 파일에 남긴다.
        # 실제 곡 F6 실행 중 한 번 0.75초 멈춤이 있었고 원인을 찾지 못해, 사용자 PC 로그로 잡으려는 것
        self._stall_clock = QElapsedTimer()
        self._stall_clock.start()
        self.stalls: list[tuple[int, str]] = []
        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(STALL_TICK_MS)
        self._stall_timer.timeout.connect(self._check_stall)
        self._stall_timer.start()
        self._time = 0.0
        self._tracked: list[TaskHandle] = []
        self._player_error_shown = False
        self._pending_ui: dict = {}
        self._pending_playback: dict = {}

        self._build_central()
        self._build_actions()
        self.lyrics = LyricsController(self)
        self.vocal = VocalDetailController(self)
        self.listening = ListeningController(self)
        self.review = ReviewController(self)
        self.compare = CompareController(self)
        self.exporter = ExportController(self)
        self.model_compare = ModelCompareController(self)
        self.hybrid = HybridController(self, self.model_compare)
        self.hybrid.lane_label.setFixedWidth(KEYBOARD_WIDTH)
        self.hybrid.lane_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hybrid.lane_label.setProperty("role", "secondary")
        self.center_grid.addWidget(self.hybrid.lane_label, 1, 0)
        self.center_grid.addWidget(self.hybrid.lane, 1, 1)
        self.hybrid.lane.hide()
        self.hybrid.lane_label.hide()
        self.right_tabs.addTab(self.review.panel, "검토")
        self.suggest = SuggestionsController(self)
        self.right_tabs.addTab(self.suggest.panel, "교정 제안")
        self.quality = QualityController(self)
        self.right_tabs.addTab(self.quality.panel, "품질")
        for seq, step, text in (("Ctrl+Alt+[", -1, "이전 교정 제안"), ("Ctrl+Alt+]", 1, "다음 교정 제안")):
            act = QAction(text, self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(lambda _c=False, s=step: self.suggest.go(s))
            self.addAction(act)
        self.right_tabs.addTab(self.compare.panel, "악보 비교")
        self._build_menus()
        self._build_toolbar()
        self.listening.build_toolbar()
        self._build_statusbar()
        self._build_log_dock()
        from amp.app.bass_controller import BassController

        self.bass = BassController(self)  # v0.6.3 베이스 TAB (오른쪽 탭·베이스 편집 모드)
        self._connect()
        self.inspector.analysis.set_settings(AnalysisSettings.from_dict(self._stored_settings()))
        self._set_model_files(self._stored_model_files())
        self.lite = lite_mode()
        if self.lite:
            self._apply_lite()
        self._update_title()
        self._update_actions()
        self._sync_tempo_widgets()
        self._refresh_tracks()
        self.resize(1440, 860)
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        sizes = self.settings.value("window/splitter")
        if sizes:
            self.splitter.setSizes([int(x) for x in sizes])

    # --- 화면 구성 ----------------------------------------------------------------

    def _build_central(self) -> None:
        self.view = PianoRollView(self.doc, self.timeline)
        self.ruler = TimeRuler(self.timeline, lambda: self.doc.score.tempo)
        self.waveform = WaveformView(self.timeline)
        self.spectrogram = SpectrogramView(self.timeline)
        self.keyboard = PianoKeyboard(self.view, self.view.mapper)
        corner = QLabel("시간")
        corner.setFixedWidth(KEYBOARD_WIDTH)
        corner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        corner.setProperty("role", "secondary")
        self.wave_label = QLabel("파형")
        self.wave_label.setFixedWidth(KEYBOARD_WIDTH)
        self.wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_label.setProperty("role", "secondary")

        center = QWidget()
        grid = QGridLayout(center)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(corner, 0, 0)
        grid.addWidget(self.ruler, 0, 1)
        # 1행: v0.6 Hybrid 구간 레인 (컨트롤러를 만든 뒤 넣는다)
        self.center_grid = grid
        grid.addWidget(self.wave_label, 2, 0)
        grid.addWidget(self.waveform, 2, 1)
        # STEP 5: 스펙트로그램 레인 (보기 > 파형·스펙트로그램). 같은 시간축, 아래 가장자리를 끌어 높이 조절
        self.spec_label = QLabel("스펙트로\n그램")
        self.spec_label.setFixedWidth(KEYBOARD_WIDTH)
        self.spec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spec_label.setProperty("role", "secondary")
        self.spec_label.setWordWrap(True)
        grid.addWidget(self.spec_label, 3, 0)
        grid.addWidget(self.spectrogram, 3, 1)
        self.spectrogram.setVisible(False)
        self.spec_label.setVisible(False)
        grid.addWidget(self.keyboard, 4, 0)
        grid.addWidget(self.view, 4, 1)
        # 가사 레인 (명세 5장: 피아노 롤 아래 '가사·음절 정렬 영역'). 파형·피아노 롤과 같은 시간축
        self.lyric_label = QLabel("가사")
        self.lyric_label.setFixedWidth(KEYBOARD_WIDTH)
        self.lyric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lyric_label.setProperty("role", "secondary")
        self.lyric_lane = LyricLane(self.doc, self.timeline)
        grid.addWidget(self.lyric_label, 5, 0)
        grid.addWidget(self.lyric_lane, 5, 1)
        grid.setRowStretch(4, 1)
        grid.setColumnStretch(1, 1)

        self.tracks = TrackPanel()
        self.inspector = Inspector()
        self.lyrics_panel = LyricsPanel()
        self.right_tabs = QTabWidget()
        self.right_tabs.setDocumentMode(True)
        self.right_tabs.addTab(self.inspector, "속성·분석")
        self.right_tabs.addTab(self.lyrics_panel, "가사")
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.tracks)
        self.splitter.addWidget(center)
        self.splitter.addWidget(self.right_tabs)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([250, 900, 380])
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

    def _action(self, text: str, slot, shortcut=None, checkable: bool = False, checked: bool = False,
                icon_name: str | None = None) -> QAction:
        a = QAction(text, self)
        if icon_name:
            a.setIcon(icon(icon_name))
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
        self.act_new = A("새 프로젝트 (오디오 열기)…", self.open_audio_dialog, QKeySequence.StandardKey.Open, icon_name="file-audio")
        self.act_open_project = A("프로젝트 열기…", self.open_project_dialog, "Ctrl+Shift+O", icon_name="folder-open")
        self.act_save = A("프로젝트 저장", self.save, QKeySequence.StandardKey.Save, icon_name="save")
        self.act_save_as = A("다른 이름으로 저장…", self.save_as, "Ctrl+Shift+S")
        self.act_open_folder = A("프로젝트 폴더 열기", self.open_project_folder, icon_name="folder")
        self.act_relink = A("원본 다시 연결…", self.relink_audio_dialog, icon_name="link")
        self.act_change_root = A("기본 프로젝트 위치 변경…", self.change_projects_root)
        self.act_quit = A("종료", self.close, "Ctrl+Q")

        # v0.6.3: 보컬 문서와 베이스 문서의 실행 취소 기록을 묶고, 보이는 피아노 롤 쪽 기록을 쓴다
        self.undo_group = QUndoGroup(self)
        self.undo_group.addStack(self.doc.undo_stack)
        self.undo_group.setActiveStack(self.doc.undo_stack)
        self.act_undo = self.undo_group.createUndoAction(self, "실행 취소")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.setIcon(icon("undo-2"))
        self.act_redo = self.undo_group.createRedoAction(self, "다시 실행")
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.act_redo.setIcon(icon("redo-2"))
        self.act_delete = A("삭제", self.delete_selected, [QKeySequence.StandardKey.Delete, "Backspace"], icon_name="trash-2")
        self.act_split = A("재생 위치에서 분할", self.split_selected_at_playhead, "Ctrl+E", icon_name="scissors")
        self.act_merge = A("선택한 음표 병합", self.merge_selected, "Ctrl+M", icon_name="merge")
        self.act_select_all = A("모두 선택", self.select_all, QKeySequence.StandardKey.SelectAll)
        self.act_revert = A("분석 원본으로 되돌리기", self.revert_to_analysis, icon_name="rotate-ccw")

        self.act_analyze = A("보컬 분석 실행", self.run_analysis, "F5", icon_name="wand-sparkles")
        self.act_redetect = A("음표만 다시 검출", self.redetect_notes, "Shift+F5", icon_name="refresh-cw")
        self.act_reseparate = A("보컬 분리 다시 실행", self.reseparate)
        self.act_import_stems = A("외부 분리 결과 가져오기…", self.import_stems_dialog, icon_name="file-audio")
        self.act_cancel_analysis = A("분석 취소", self.session.cancel_analysis, icon_name="x")
        self.act_clear_cache = A("캐시 비우기", self.clear_cache)
        self.act_compare_runs = A("최근 두 실행 비교 보고서", self.compare_recent_runs)
        self.act_no_vocal = A("보컬 없는 구간…", self.edit_no_vocal_ranges)
        self.act_pitch_regions = A("피치 다시 찾기 구간 (겹친 목소리)…", self.edit_pitch_regions)
        self.act_probe = A("GPU 확인", self.probe_gpu, icon_name="cpu")
        self.act_sample = A("샘플 데이터 생성 (시험용, 실제 분석 아님)", self.run_sample_analysis)

        self.act_export = A("오디오 내보내기…", lambda: self.export_dialog([]), "Ctrl+Shift+E", icon_name="download")

        self.act_play = A("재생", self.toggle_play, "Space", icon_name="play")
        self.act_stop = A("정지", self.stop, "Esc", icon_name="square")
        self.act_home = A("처음으로", lambda: self.seek(0.0), "Home", icon_name="skip-back")

        self.act_zoom_in = A("가로 확대", lambda: self.roll_view().zoom_time(1.25), ["Ctrl+=", "Ctrl++"], icon_name="zoom-in")
        self.act_zoom_out = A("가로 축소", lambda: self.roll_view().zoom_time(0.8), "Ctrl+-", icon_name="zoom-out")
        self.act_zoom_fit = A("전체 보기", lambda: self.roll_view().zoom_to_fit(), "Ctrl+0", icon_name="maximize-2")
        self.act_bass_analyze = A("베이스 분석 (TAB)", lambda: self.bass.analyze(), "F8", icon_name="wand-sparkles")
        self.act_bass_edit = A("베이스 편집 모드", lambda on: self.bass.set_active(on), "Ctrl+Shift+B", checkable=True)
        self.act_bass_gp5 = A("베이스 TAB 내보내기 (GP5)…", lambda: self.bass.export_gp5())
        self.act_bass_text = A("베이스 TAB 내보내기 (텍스트)…", lambda: self.bass.export_text())
        self.act_bass_audio = A("베이스 소리 내보내기 (WAV·MP3)…", lambda: self.bass.export_audio())
        self.act_show_notes = A("음표 표시", self.view.roll.set_notes_visible, checkable=True, checked=True)
        self.act_show_curve = A("피치 곡선 표시", self.view.roll.curve_item.setVisible, checkable=True, checked=True)
        self.act_follow = A("재생 위치 따라가기", self._set_follow, checkable=True, checked=True)
        self.act_help = A("조작법 및 구현 현황", self.show_help, "F1", icon_name="info")
        self.act_about = A("정보", self.show_about)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("프로젝트(&P)")
        self.menu_project = m
        m.addAction(self.act_new)
        m.addAction(self.act_open_project)
        self.recent_menu = m.addMenu("최근 프로젝트")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        m.addSeparator()
        for a in (self.act_save, self.act_save_as, None, self.act_open_folder, self.act_relink, self.act_change_root,
                  None, self.act_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("편집(&E)")
        for a in (self.act_undo, self.act_redo, None, self.act_delete, self.act_split, self.act_merge, None,
                  self.act_select_all, None, self.act_revert):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("재생(&L)")
        for a in (self.act_play, self.act_stop, self.act_home):
            m.addAction(a)
        self.menu_play = m
        m = mb.addMenu("분석(&A)")
        for a in (self.act_analyze, self.act_redetect, self.act_reseparate, self.act_import_stems, self.act_cancel_analysis, None,
                  self.act_no_vocal, self.act_pitch_regions, self.act_compare_runs, self.act_probe, self.act_clear_cache,
                  None, self.act_sample, None, self.act_bass_analyze, self.act_bass_edit):
            m.addSeparator() if a is None else m.addAction(a)
        self.model_compare.build_menu(m)
        self.hybrid.build_menu(m)
        self.lyrics.build_menu(mb)
        m = mb.addMenu("내보내기(&X)")
        m.addAction(self.act_export)
        m.addAction(self.exporter.act)
        m.addSeparator()
        m.addAction(self.act_bass_gp5)
        m.addAction(self.act_bass_text)
        m.addAction(self.act_bass_audio)
        m = mb.addMenu("보기(&V)")
        for a in (self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit, None, self.act_show_notes,
                  self.act_show_curve, self.act_follow):
            m.addSeparator() if a is None else m.addAction(a)
        self.listening.build_menu(self.menu_play, m)
        self.compare.build_menu(self.menu_project, m)
        m.addSeparator()
        self.vocal.build_menu(m, self.lyrics.menu)
        self.view_menu = m
        m = mb.addMenu("도움말(&H)")
        m.addAction(self.act_help)
        m.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("도구")
        tb.setObjectName("main")
        tb.setMovable(False)
        tb.setIconSize(ICON_SIZE)
        self.project_label = QLabel("프로젝트 없음")
        self.project_label.setObjectName("panelTitle")
        self.project_label.setMinimumWidth(140)
        tb.addWidget(self.project_label)
        for a in (self.act_new, self.act_open_project, self.act_save, self.act_open_folder):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_home, self.act_play, self.act_stop):
            tb.addAction(a)
        self.time_label = QLabel()
        self.time_label.setObjectName("timecode")
        self.time_label.setMinimumWidth(300)
        tb.addWidget(self.time_label)
        tb.addSeparator()
        tb.addWidget(QLabel(" BPM "))
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(MIN_BPM, MAX_BPM)
        self.bpm_spin.setDecimals(2)
        self.bpm_spin.setKeyboardTracking(False)
        self.bpm_spin.valueChanged.connect(self._on_bpm_changed)
        tb.addWidget(self.bpm_spin)
        self.bpm_state = QLabel()
        self.bpm_state.setProperty("role", "secondary")
        tb.addWidget(self.bpm_state)
        self.bpm_double = QToolButton()
        self.bpm_double.setText("×2")
        self.bpm_double.setToolTip("BPM 두 배 (추정이 절반으로 나왔을 때)")
        self.bpm_double.clicked.connect(lambda: self._scale_bpm(2.0))
        self.bpm_half = QToolButton()
        self.bpm_half.setText("½")
        self.bpm_half.setToolTip("BPM 절반 (추정이 두 배로 나왔을 때)")
        self.bpm_half.clicked.connect(lambda: self._scale_bpm(0.5))
        tb.addWidget(self.bpm_double)
        tb.addWidget(self.bpm_half)
        tb.addWidget(QLabel(" 첫 박 "))
        self.beat_offset = QDoubleSpinBox()
        self.beat_offset.setRange(0.0, 10.0)
        self.beat_offset.setDecimals(3)
        self.beat_offset.setSingleStep(0.01)
        self.beat_offset.setSuffix(" s")
        self.beat_offset.setKeyboardTracking(False)
        self.beat_offset.setToolTip("박자선 위치를 맞춥니다 (마디 첫 박은 자동으로 판정하지 않습니다)")
        self.beat_offset.valueChanged.connect(self._on_offset_changed)
        tb.addWidget(self.beat_offset)
        sig = QLabel(" 4/4 ")
        sig.setToolTip("박자는 4/4로 가정합니다")
        tb.addWidget(sig)
        self.snap_combo = QComboBox()
        self.snap_combo.setToolTip("격자 (스냅 단위)")
        for label, value in SNAP_CHOICES:
            self.snap_combo.addItem(label, value)
        self.snap_combo.setCurrentIndex(3)
        self.snap_combo.currentIndexChanged.connect(lambda _i: self.view.set_snap_beats(float(self.snap_combo.currentData())))
        tb.addWidget(self.snap_combo)
        tb.addSeparator()
        self.analyze_button = QToolButton()
        self.analyze_button.setDefaultAction(self.act_analyze)
        self.analyze_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.analyze_button.setProperty("primary", True)
        self.analyze_button.setIcon(icon("wand-sparkles", T.BG_PRIMARY))
        tb.addWidget(self.analyze_button)
        self.compare_button = QToolButton()
        self.compare_button.setDefaultAction(self.model_compare.act_mode)
        self.compare_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._compare_tb = tb.addWidget(self.compare_button)
        self.hybrid_button = QToolButton()
        self.hybrid_button.setDefaultAction(self.hybrid.act_mode)
        self.hybrid_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._hybrid_tb = tb.addWidget(self.hybrid_button)
        self._update_time_label(0.0)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.selection_label = QLabel()
        self.hover_label = QLabel()
        self.hover_label.setFont(theme.mono_font(12))
        self.analysis_label = QLabel()
        legend = QLabel("음표: ■ 원본  ▧ 수정(점선·삼각)  □ 추가(+)  ┄ 장식음   피치 곡선: ━")
        legend.setToolTip("원본·수정·추가는 테두리와 표식으로도 구분합니다")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(200)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.progress_label = QLabel()
        self.progress_label.hide()
        self.cancel_button = QToolButton()
        self.cancel_button.setText("취소")
        self.cancel_button.setIcon(icon("x", size=14))
        self.cancel_button.clicked.connect(self._cancel_tracked)
        self.cancel_button.hide()
        self.log_button = QToolButton()
        self.log_button.setText("로그")
        self.log_button.setCheckable(True)
        sb.addWidget(self.selection_label, 1)
        sb.addWidget(self.hover_label)
        sb.addPermanentWidget(self.progress_label)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_button)
        sb.addPermanentWidget(legend)
        sb.addPermanentWidget(self.analysis_label)
        sb.addPermanentWidget(self.log_button)

    def _build_log_dock(self) -> None:
        self.log_panel = LogPanel()
        self.log_panel.attach()
        self.log_dock = QDockWidget("로그", self)
        self.log_dock.setObjectName("log")
        self.log_dock.setWidget(self.log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        self.log_button.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(self.log_button.setChecked)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.log_dock.toggleViewAction())

    def _connect(self) -> None:
        self.player.positionChanged.connect(self._on_position)
        self.player.stateChanged.connect(self._on_player_state)
        self.player.errorOccurred.connect(self._on_player_error)
        self.ruler.seekRequested.connect(self.seek)
        self.waveform.seekRequested.connect(self.seek)
        self.view.selectionInfoChanged.connect(self.selection_label.setText)
        self.view.hoverInfoChanged.connect(self.hover_label.setText)
        self.view.roll.selectionChanged.connect(self._on_selection_changed)
        self.view.editFailed.connect(self._show_status_error)
        self.doc.dirtyChanged.connect(self._update_title)
        self.doc.filePathChanged.connect(self._update_title)
        self.doc.projectReset.connect(self._on_project_reset)
        self.doc.analysisChanged.connect(self._on_analysis_changed)
        self.doc.tempoChanged.connect(self._sync_tempo_widgets)
        self.doc.notesReplaced.connect(lambda *_: self._on_selection_changed())
        self.doc.notesReplaced.connect(lambda *_: self.listening.notes_changed())
        self.doc.analysisChanged.connect(self.listening.notes_changed)
        self.doc.analysisChanged.connect(lambda: self.listening.set_lane_mode(self.listening.lane_mode))
        self.doc.projectReset.connect(self.listening.notes_changed)
        self.waveform.regionSelected.connect(lambda a, b: self.listening.set_region(a, b, enable=True))
        self.spectrogram.seekRequested.connect(self.seek)
        self.view.notePreviewRequested.connect(self.listening.preview_pitch)
        self.view.roll.selectionChanged.connect(self.compare.show_selected)
        self.session.tracksChanged.connect(self.review.schedule)

        s = self.session
        s.opened.connect(self._on_session_opened)
        s.audioReady.connect(self._on_audio_ready)
        s.tracksChanged.connect(self._refresh_tracks)
        s.originalChanged.connect(self._refresh_project_panel)
        s.message.connect(lambda text, ms: self.statusBar().showMessage(text, ms))
        s.error.connect(self._show_error)
        s.notice.connect(lambda title, text: QMessageBox.information(self, title, text))
        s.analysisReady.connect(self._on_analysis_ready)
        s.stemsImported.connect(self._on_stems_imported)
        s.busyChanged.connect(lambda _b: self._update_actions())
        s.taskStarted.connect(self._track)

        for row in self.tracks.rows.values():
            row.soloToggled.connect(lambda name, on: self.player.set_track_state(name, solo=on))
            row.muteToggled.connect(lambda name, on: self.player.set_track_state(name, mute=on))
            row.volumeChanged.connect(lambda name, g: self.player.set_track_state(name, gain=g))
            row.exportRequested.connect(lambda name: self.export_dialog([name]))
            row.revealRequested.connect(self.reveal_track)
            row.selected.connect(self._show_track_waveform)
        self.tracks.open_folder.clicked.connect(self.open_project_folder)
        self.tracks.relink.clicked.connect(self.relink_audio_dialog)

        a = self.inspector.analysis
        a.runRequested.connect(self.run_analysis)
        a.redetectRequested.connect(self.redetect_notes)
        a.reseparateRequested.connect(self.reseparate)
        a.clearCacheRequested.connect(self.clear_cache)
        a.probeRequested.connect(self.probe_gpu)
        a.noVocalEditRequested.connect(self.edit_no_vocal_ranges)
        a.pitchRegionsEditRequested.connect(self.edit_pitch_regions)
        a.modelFilesRequested.connect(self.choose_model_files)
        a.settingsChanged.connect(self._on_analysis_settings_changed)
        self.view.noVocalRangeRequested.connect(self.mark_no_vocal)

    # --- 상태 갱신 ----------------------------------------------------------------

    def _stored_settings(self) -> dict:
        v = self.settings.value(SETTINGS_KEY)
        return v if isinstance(v, dict) else {}

    def _update_title(self, *_args) -> None:
        folder = self.session.folder
        name = folder.name if folder else "프로젝트 없음"
        self.setWindowTitle(f"{name}[*] - {APP_DISPLAY_NAME}" + (" (가벼운 모드)" if getattr(self, "lite", False) else ""))
        self.setWindowModified(self.doc.is_dirty)
        self.project_label.setText(f"{name}{' •' if self.doc.is_dirty else ''}")
        self.project_label.setToolTip("저장하지 않은 변경이 있습니다" if self.doc.is_dirty else "저장됨")

    def roll_view(self):
        """지금 보이는 피아노 롤 (보컬 또는 v0.6.3 베이스 편집 모드의 베이스)."""
        bass = getattr(self, "bass", None)
        return bass.view if bass is not None and bass.active else self.view

    def roll_doc(self):
        bass = getattr(self, "bass", None)
        return bass.doc if bass is not None and bass.active else self.doc

    def _update_actions(self) -> None:
        n_sel = len(self.roll_view().selected_ids())
        has_audio = self.session.audio is not None
        is_open = self.session.is_open
        busy = self.session.busy
        has_pitch = self.doc.project.analysis is not None and self.doc.project.analysis.pitch_curve is not None
        self.act_delete.setEnabled(n_sel > 0)
        self.act_split.setEnabled(n_sel > 0)
        self.act_merge.setEnabled(n_sel >= 2)
        self.act_revert.setEnabled(self.doc.project.analysis is not None)
        self.act_analyze.setEnabled(has_audio and is_open and not busy)
        self.act_reseparate.setEnabled(has_audio and is_open and not busy)
        self.act_import_stems.setEnabled(has_audio and is_open and not busy)
        self.model_compare.update_actions(has_audio, is_open, busy)
        self.hybrid.update_actions(has_audio, is_open, busy)
        self.act_redetect.setEnabled(has_pitch and not busy)
        self.act_sample.setEnabled(has_audio and not busy)
        self.act_cancel_analysis.setEnabled(busy)
        self.act_play.setEnabled(has_audio)
        self.act_stop.setEnabled(has_audio)
        for a in (self.act_save, self.act_save_as, self.act_open_folder, self.act_clear_cache, self.act_export,
                  self.act_compare_runs, self.act_no_vocal, self.act_pitch_regions):
            a.setEnabled(is_open)
        self.act_relink.setEnabled(is_open and self.session.folder.meta.original.mode == "link")
        panel = self.inspector.analysis
        panel.run.setEnabled(self.act_analyze.isEnabled())
        panel.redetect.setEnabled(self.act_redetect.isEnabled())
        panel.reseparate.setEnabled(self.act_reseparate.isEnabled())
        panel.clear_cache.setEnabled(is_open)
        panel.set_cache_size(self.session.cache_size())
        if hasattr(self, "lyrics"):
            self.lyrics.update_actions()
        if hasattr(self, "bass"):
            self.act_bass_analyze.setEnabled(has_audio and is_open and not busy)
            self.bass.panel.analyze.setEnabled(self.act_bass_analyze.isEnabled())
            has_bass = bool(self.bass.doc.score.notes())
            self.act_bass_gp5.setEnabled(has_bass)
            self.act_bass_text.setEnabled(has_bass)
            self.act_bass_audio.setEnabled(is_open and not busy)
            self.bass.panel.export_audio.setEnabled(is_open and not busy)

    def _update_analysis_label(self) -> None:
        a = self.doc.project.analysis
        if a is None:
            self.analysis_label.setText("분석 결과 없음")
            self.analysis_label.setToolTip("")
        elif a.is_placeholder:
            self.analysis_label.setText(f"<b style='color:{T.WARNING}'>⚠ 샘플 데이터 (실제 분석 아님)</b>")
            self.analysis_label.setToolTip("\n".join(f"{k}: {m.name} {m.version}" for k, m in a.models.items()))
        else:
            sep = a.models.get("separation")
            self.analysis_label.setText(f"분석: {sep.name if sep else '-'} · pYIN · 음표 {len(a.notes)}개")
            self.analysis_label.setToolTip("\n".join(f"{k}: {m.name} {m.version}" for k, m in a.models.items())
                                           + ("\n\n" + "\n".join(a.warnings) if a.warnings else ""))

    def _sync_tempo_widgets(self) -> None:
        tempo = self.doc.score.tempo
        for w, v in ((self.bpm_spin, tempo.initial_bpm), (self.beat_offset, tempo.first_beat_time)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        a = self.doc.project.analysis
        if a is None:
            self.bpm_state.setText("기본값")
            self.bpm_state.setToolTip("")
        elif tempo == a.tempo:
            reliable = a.tempo_info.get("reliable")
            cands = ", ".join(f"{c:.1f}" for c in a.tempo_candidates) or "-"
            self.bpm_state.setText("추정" if not a.is_placeholder else "샘플")
            self.bpm_state.setToolTip(f"자동 추정값입니다. 정확하다고 판정하지 않았습니다.\n후보(원래·2배·절반): {cands}"
                                      + ("" if reliable else "\n박을 충분히 찾지 못했습니다."))
        else:
            self.bpm_state.setText("수동")
            self.bpm_state.setToolTip("사용자가 정한 템포입니다.")
        self.ruler.update()
        self._update_time_label(self.current_time())

    def _on_project_reset(self) -> None:
        self._update_title()
        self._update_actions()
        self._update_analysis_label()
        self._sync_tempo_widgets()
        self.view.center_on_content()
        self._on_selection_changed()

    def _on_analysis_changed(self) -> None:
        self._update_actions()
        self._update_analysis_label()
        self._sync_tempo_widgets()
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        self._update_actions()
        notes = []
        for nid in self.view.selected_ids():
            if nid in self.doc.score:
                notes.append(self.doc.score.get(nid))
        self.inspector.note_info.show_notes(notes, self.doc.project)

    def _on_bpm_changed(self, value: float) -> None:
        self._guard(lambda: self.doc.set_bpm(value))
        self._sync_tempo_widgets()

    def _on_offset_changed(self, value: float) -> None:
        t = self.doc.score.tempo
        self._guard(lambda: self.doc.set_tempo(TempoMap.constant(t.initial_bpm, value, t.beats_per_bar)))

    def _scale_bpm(self, factor: float) -> None:
        bpm = self.doc.score.tempo.initial_bpm * factor
        if MIN_BPM <= bpm <= MAX_BPM:
            self._guard(lambda: self.doc.set_bpm(bpm))
            self._sync_tempo_widgets()

    def _set_follow(self, on: bool) -> None:
        self.view.follow_playhead = on

    # --- 프로젝트·트랙 패널 ----------------------------------------------------------------

    def _refresh_project_panel(self) -> None:
        f = self.session.folder
        p = self.tracks
        if f is None:
            p.project_name.setText("프로젝트 없음")
            p.project_info.setText("오디오를 열면 프로젝트 폴더가 만들어집니다.")
            p.original_state.setText("")
            p.relink.hide()
            return
        o = f.meta.original
        p.project_name.setText(f.name)
        p.project_info.setText(f"{f.root}\n원본: {'프로젝트에 복사' if o.mode == 'copy' else '외부 파일 연결'} · {o.file_name}")
        st = self.session.original
        state = st.state if st else "missing"
        text = {"ok": "✓ 원본 확인됨", "changed": "⚠ 원본이 바뀌었습니다", "missing": "✕ 원본을 찾을 수 없습니다"}[state]
        p.original_state.setText(text)
        p.original_state.setProperty("state", {"ok": "ok", "changed": "warn", "missing": "error"}[state])
        p.original_state.style().unpolish(p.original_state)
        p.original_state.style().polish(p.original_state)
        p.relink.setVisible(o.mode == "link" or state != "ok")
        p.relink.setEnabled(True)
        self._update_actions()

    def _refresh_tracks(self) -> None:
        s = self.session
        for name, row in self.tracks.rows.items():
            audio = s.track_audio(name)
            if name == "original":
                if audio is not None:
                    o = s.folder.meta.original if s.folder else None
                    info = f"{o.file_name if o else ''}\n{audio.duration:.1f}s · {audio.sample_rate}Hz · {audio.channels}ch"
                    row.set_available(True, info, "✓ 원본", "ok")
                else:
                    row.set_available(False, "오디오 없음", "—", "muted")
            elif audio is not None:
                m = s.manifest
                rel = f"stems/{m.files[name]['path']}" if m is not None and name in m.files else f"stems/{name}.wav"
                if m is not None and m.model == "mel_roformer":
                    dev = str(m.model_info.get("device", ""))
                    info = (f"{rel}\n{audio.duration:.1f}s · RoFormer ({m.model_info.get('checkpoint', '')})"
                            f" · {dev.upper()}")
                    row.set_available(True, info, "✓ 분리됨", "ok")
                elif m is not None and m.model == "external":
                    src = m.model_info.get("tool") or m.model_info.get(f"{name}_file") or m.model_info.get("vocals_file", "")
                    if name == "instrumental" and m.model_info.get("instrumental") == "original_minus_vocals":
                        src = "원본 − 가져온 보컬"
                    info = f"{rel}\n{audio.duration:.1f}s · 가져옴: {src}"
                    row.set_available(True, info, "✓ 가져옴", "ok")
                else:
                    dev = m.model_info.get("device", "") if m else ""
                    info = f"{rel}\n{audio.duration:.1f}s · {m.model if m else ''} · {dev.upper()}"
                    row.set_available(True, info, "✓ 분리됨", "ok")
            else:
                reason = s.stems_reason or "분리 전"
                row.set_available(False, reason + " — 보컬 분석을 실행하면 만들어집니다", "분리 전" if reason == "분리 전" else "⚠ 무효",
                                  "muted" if reason == "분리 전" else "warn")
            st = self.player.track_state(name)
            row.set_controls(st["solo"], st["mute"], st["gain"])
        if self.tracks.selected_track not in self.player.tracks:
            self.tracks.select("original")
        self._show_track_waveform(self.tracks.selected_track)
        self._update_actions()

    def _show_track_waveform(self, name: str) -> None:
        self.tracks.select(name)
        peaks = self.session.peaks.get(name)
        from PySide6.QtGui import QColor

        self.waveform.set_peaks(peaks, QColor(T.TRACK_COLORS[name]), f"{TRACK_LABELS[name]}" if peaks else "")
        self.wave_label.setText(TRACK_HINTS[name])
        self.spectrogram.set_source(TRACK_LABELS[name], self.session.track_audio(name))  # 파형과 같은 트랙

    def reveal_track(self, name: str) -> None:
        path = self.session.track_path(name)
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    # --- 재생 -----------------------------------------------------------------

    def current_time(self) -> float:
        return self.player.position() if self.player.has_audio else self._time

    def toggle_play(self) -> None:
        if self.session.audio is not None:
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
        self.roll_view().set_playhead(t)
        self.waveform.set_playhead(t)
        if self.spectrogram.isVisible():
            self.spectrogram.set_playhead(t)
        self.ruler.set_playhead(t)
        self.lyrics.on_position(t)
        self._update_time_label(t)

    def _update_time_label(self, t: float) -> None:
        def fmt(x: float) -> str:
            m, s = divmod(max(x, 0.0), 60)
            return f"{int(m):02d}:{s:06.3f}"

        bar, beat = self.doc.score.tempo.bar_beat(t)
        total = self.session.audio.duration if self.session.audio else 0.0
        self.time_label.setText(f"{fmt(t)} / {fmt(total)}   {bar}.{beat:.2f}")

    def _on_player_state(self, state: PlaybackState) -> None:
        playing = state is PlaybackState.PLAYING
        self.act_play.setText("일시정지" if playing else "재생")
        self.act_play.setIcon(icon("pause" if playing else "play"))

    def _on_player_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        if not self._player_error_shown:
            self._player_error_shown = True
            QMessageBox.warning(self, "재생 오류", message)

    def _check_stall(self) -> None:
        late = int(self._stall_clock.restart()) - STALL_TICK_MS
        if late >= STALL_LOG_MS:
            what = self.progress_label.text() if self._tracked else "작업 없음"
            self.stalls.append((late, what))
            del self.stalls[:-20]
            log.warning("화면 이벤트 루프가 %dms 멈췄습니다 (진행 중: %s)", late, what)

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
        self._update_actions()

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

    # --- 프로젝트 ---------------------------------------------------------------

    def open_audio_dialog(self) -> None:
        if not self.maybe_save():
            return
        start = self.settings.value("paths/last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "오디오 열기 (새 프로젝트)", start, AUDIO_FILTER)
        if not path:
            return
        self.settings.setValue("paths/last_dir", os.path.dirname(path))
        existing = self.session.existing_projects_for(path)
        dlg = NewProjectDialog(path, self.session.projects_root, existing, self)
        if dlg.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        choice = dlg.choice()
        if choice.open_existing:
            self.session.open_folder(choice.open_existing)
            return
        self.create_project_from_audio(path, choice.name, choice.mode, choice.parent)

    def create_project_from_audio(self, path: str, name: str | None = None, mode: str = "copy",
                                  parent: str | None = None) -> TaskHandle:
        return self.session.create_from_audio(path, name, mode, parent)

    def open_project_dialog(self) -> None:
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", self.session.projects_root, OPEN_FILTER)
        if path:
            self.open_project(path)

    def open_project(self, path: str) -> bool:
        """project.json 또는 프로젝트 폴더를 연다. .ampproj(이전 형식)는 새 폴더로 변환한다."""
        p = Path(path)
        if p.suffix.lower() == PROJECT_EXTENSION:
            answer = QMessageBox.question(
                self, "이전 형식 프로젝트",
                f"{p.name}은(는) 이전 형식입니다. SONATLAS 프로젝트 폴더로 변환해 엽니다.\n"
                "원래 파일은 그대로 두고, 새 폴더의 backup/에 사본도 남깁니다. 계속할까요?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
            return self.session.import_legacy(str(p))
        root = p.parent if p.name == PROJECT_FILE else p
        if not is_project_folder(root):
            self._show_error("프로젝트를 열 수 없습니다", f"SONATLAS 프로젝트 폴더가 아닙니다: {root}")
            return False
        return self.session.open_folder(str(root))

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        items = self.session.recent()
        if not items:
            a = self.recent_menu.addAction("(없음)")
            a.setEnabled(False)
        for p in items:
            self.recent_menu.addAction(p, lambda p=p: self.maybe_save() and self.open_project(p))

    def _on_session_opened(self) -> None:
        self.listening.project_closed()  # 이전 프로젝트의 미리 듣기·늘린 오디오·음표 소리 트랙 정리
        f = self.session.folder
        self._pending_ui = dict(f.meta.ui) if f else {}
        self._pending_playback = dict(f.meta.playback) if f else {}
        panel = self.inspector.analysis
        if f and f.meta.analysis_settings:
            panel.set_settings(AnalysisSettings.from_dict(f.meta.analysis_settings))
        else:  # 보컬 없는 구간은 곡마다 다르므로 이전 프로젝트 것을 가져오지 않는다
            panel.set_settings(dataclasses.replace(panel.settings(), no_vocal_ranges=(), pitch_regions=()))
        self.view.roll.set_no_vocal_ranges(panel.settings().no_vocal_ranges)
        self._sync_stem_mask(panel.settings())
        self._apply_ui_state(self._pending_ui)
        self._refresh_project_panel()
        self._refresh_tracks()
        self._update_title()

    def reveal_time(self, t: float) -> None:
        """t가 화면 밖이면 가로로 스크롤해 보이게 한다."""
        a, b = self.view.visible_time_range()
        if not (a <= t <= b):
            bar = self.view.horizontalScrollBar()
            bar.setValue(int(self.view.mapper.time_to_x(t) - self.view.viewport().width() / 3))

    def _on_audio_ready(self) -> None:
        audio = self.session.audio
        self.view.roll.set_audio_duration(audio.duration)
        self.bass.set_audio_duration(audio.duration)
        self.lyric_lane.duration = audio.duration
        for name, st in (self._pending_playback.get("tracks") or {}).items():
            self.player.set_track_state(name, gain=float(st.get("gain", 1.0)), mute=bool(st.get("mute", False)),
                                        solo=bool(st.get("solo", False)))
        if not self._pending_ui:
            self.view.zoom_to_fit()
        else:
            self._apply_ui_state(self._pending_ui)
        self.seek(float(self._pending_playback.get("position", 0.0) or 0.0))
        self._refresh_tracks()
        self._update_actions()

    def _ui_state(self) -> dict:
        return {
            "px_per_sec": self.view.mapper.px_per_sec,
            "row_height": self.view.mapper.row_height,
            "scroll_x": self.view.horizontalScrollBar().value(),
            "scroll_y": self.view.verticalScrollBar().value(),
            "show_notes": self.act_show_notes.isChecked(),
            "show_pitch": self.act_show_curve.isChecked(),
            "snap_beats": float(self.snap_combo.currentData()),
            "follow": self.act_follow.isChecked(),
            "waveform_track": self.tracks.selected_track,
            "splitter": self.splitter.sizes(),
            "show_lyrics": self.lyrics.act_show.isChecked(),
            "right_tab": self.right_tabs.currentIndex(),
        }

    def _playback_state(self) -> dict:
        return {"position": self.current_time(),
                "tracks": {n: self.player.track_state(n) for n in TRACK_LABELS}}

    def _apply_ui_state(self, ui: dict) -> None:
        if not ui:
            return
        try:
            if "px_per_sec" in ui:
                self.view.zoom_time(float(ui["px_per_sec"]) / self.view.mapper.px_per_sec, 0.0)
            if "row_height" in ui:
                self.view.zoom_pitch(float(ui["row_height"]) / self.view.mapper.row_height)
            self.act_show_notes.setChecked(bool(ui.get("show_notes", True)))
            self.act_show_curve.setChecked(bool(ui.get("show_pitch", True)))
            self.act_follow.setChecked(bool(ui.get("follow", True)))
            self.lyrics.act_show.setChecked(bool(ui.get("show_lyrics", True)))
            self.right_tabs.setCurrentIndex(int(ui.get("right_tab", 0)))
            idx = self.snap_combo.findData(float(ui.get("snap_beats", 0.25)))
            if idx >= 0:
                self.snap_combo.setCurrentIndex(idx)
            if ui.get("splitter"):
                self.splitter.setSizes([int(x) for x in ui["splitter"]])
            QTimer.singleShot(0, lambda: (self.view.horizontalScrollBar().setValue(int(ui.get("scroll_x", 0))),
                                          self.view.verticalScrollBar().setValue(int(ui.get("scroll_y", 0)))))
        except (TypeError, ValueError):
            log.warning("작업 공간 상태를 복원하지 못했습니다: %s", ui)

    def save(self) -> bool:
        args = (self._ui_state(), self._playback_state(), self.inspector.analysis.settings().to_dict())
        ok = self.session.save(*args)
        if not ok and self.session.last_conflict:
            answer = QMessageBox.question(
                self, "저장 충돌",
                "프로젝트를 연 뒤에 다른 곳(다른 프로그램이나 공유 폴더의 다른 컴퓨터)에서 바뀐 파일이 있습니다:\n"
                + ", ".join(self.session.last_conflict)
                + "\n\n덮어쓰면 그쪽에서 바꾼 내용이 사라집니다. 덮어쓸까요?\n"
                "'아니요'를 누르면 저장하지 않습니다. 지금 작업은 '다른 이름으로 저장'으로 따로 남길 수 있습니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                ok = self.session.save(*args, force=True)
            else:
                self.statusBar().showMessage("저장하지 않았습니다 (저장 충돌).", 8000)
        if ok:
            self.autosave.discard()
        return ok

    def save_as(self) -> None:
        if not self.session.is_open:
            return
        parent = QFileDialog.getExistingDirectory(self, "새 프로젝트를 만들 위치", self.session.projects_root)
        if not parent:
            return
        self.session.save_as(parent, f"{self.session.folder.name} 사본", self._ui_state(), self._playback_state())

    def open_project_folder(self) -> None:
        if self.session.folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session.folder.root)))

    def relink_audio_dialog(self) -> None:
        f = self.session.folder
        if f is None:
            return
        start = os.path.dirname(f.meta.original.path) if f.meta.original.mode == "link" else str(f.root)
        path, _ = QFileDialog.getOpenFileName(self, "원본 다시 연결", start, AUDIO_FILTER)
        if path:
            self.session.relink(path)

    def change_projects_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "기본 프로젝트 위치", self.session.projects_root)
        if d:
            self.session.set_projects_root(d)
            self.statusBar().showMessage(f"새 프로젝트는 {d}에 만듭니다.", 6000)

    def maybe_save(self) -> bool:
        if not self.doc.is_dirty:
            return True
        answer = QMessageBox.question(
            self, "저장하지 않은 변경", "저장하지 않은 변경 사항이 있습니다. 저장하시겠습니까?",
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
        target = latest.original_path or "저장하지 않은 작업"
        answer = QMessageBox.question(
            self, "작업 복구",
            f"이전 실행이 비정상 종료된 것 같습니다.\n자동 저장본({latest.saved_at}, {target})을 복구하시겠습니까?")
        if answer == QMessageBox.StandardButton.Yes:
            self.restore(latest)
            others = entries[1:]
        else:
            others = entries
        for e in others:
            AutosaveManager.remove_entry(e)

    def restore(self, entry: RecoveryEntry) -> None:
        try:
            recovered = load_project(entry.project_path)
        except ProjectFormatError as e:
            self._show_error("복구하지 못했습니다", str(e))
            AutosaveManager.remove_entry(entry)
            return
        folder = entry.original_path
        if folder and is_project_folder(folder) and self.session.open_folder(folder):
            recovered.audio = self.doc.project.audio
            self.doc.reset(recovered, folder, dirty=True)
        else:
            self.doc.reset(recovered, None, dirty=True)
        AutosaveManager.remove_entry(entry)
        self.statusBar().showMessage("자동 저장본을 복구했습니다. 확인한 뒤 저장해 주세요.", 10000)

    # --- 편집 -----------------------------------------------------------------

    def _guard(self, fn) -> None:
        try:
            fn()
        except EditError as e:
            self.statusBar().showMessage(str(e), 5000)

    def delete_selected(self) -> None:
        self._guard(lambda: self.roll_doc().delete_notes(self.roll_view().selected_ids()))

    def split_selected_at_playhead(self) -> None:
        self._guard(lambda: self.roll_doc().split_notes_at(self.roll_view().selected_ids(), self.current_time()))

    def merge_selected(self) -> None:
        self._guard(lambda: self.roll_doc().merge_notes(self.roll_view().selected_ids()))

    def select_all(self) -> None:
        v = self.roll_view()
        v.roll.select_ids(list(v.roll.items_by_id))

    def revert_to_analysis(self) -> None:
        self._guard(self.doc.revert_to_analysis)

    # --- 분석 -----------------------------------------------------------------

    def run_analysis(self) -> None:
        s = self.inspector.analysis.settings()
        if self._ensure_model_files(s, force=False):
            self.session.run_analysis(s)

    def reseparate(self) -> None:
        s = self.inspector.analysis.settings()
        if s.separation == "none":
            QMessageBox.information(self, "보컬 분리 다시 실행", "분석 설정의 보컬 분리가 '분리 안 함'입니다. "
                                    "Demucs나 RoFormer를 고른 뒤 다시 실행하세요.")
            return
        if not self._ensure_model_files(s, force=True):
            return
        label = SEPARATION_LABELS.get(s.separation, s.separation)
        text = f"기존 분리 결과(stems/)를 쓰지 않고 {label}(으)로 다시 분리합니다. 새 결과가 완성된 뒤에 기존 파일을 바꿉니다."
        if s.separation == "mel_roformer":
            text += " RoFormer는 Demucs보다 오래 걸립니다(GPU 권장)."
        manifest = self.session.manifest
        if manifest is not None and manifest.model == "external":
            text += " 지금 쓰는 결과는 다른 프로그램에서 가져온 것이라, 바꾸기 전에 backup/ 폴더에 복사해 둡니다."
        answer = QMessageBox.question(self, "보컬 분리 다시 실행", text + " 계속할까요?")
        if answer == QMessageBox.StandardButton.Yes:
            self.session.run_analysis(s, force_separation=True)

    # --- RoFormer 모델 파일 ------------------------------------------------------

    def _stored_model_files(self) -> dict:
        v = self.settings.value(MODEL_FILES_KEY)
        return {k: str(v.get(k) or "") for k in ("checkpoint", "config")} if isinstance(v, dict) else {}

    def _set_model_files(self, files: dict, status: str = "") -> None:
        files = {k: str(files.get(k) or "") for k in ("checkpoint", "config")} if files else {}
        self.session.model_files = files
        self.settings.setValue(MODEL_FILES_KEY, files)
        ckpt = files.get("checkpoint", "")
        if ckpt and not status and not os.path.isfile(ckpt):
            status = "⚠ 파일이 없습니다. 다시 고르세요."
        self.inspector.analysis.set_model_files(files, status)

    def choose_model_files(self) -> bool:
        dlg = RoformerModelDialog(self.session.model_files, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        files = dlg.values()
        self._set_model_files(files, "확인 중… (모델 파일을 읽는 데 몇 초 걸립니다)")
        self._check_model_files(files)
        return True

    def _check_model_files(self, files: dict) -> None:
        handle = self.session.check_model_files(files)
        panel = self.inspector.analysis

        def done(info: dict) -> None:
            if self.session.model_files != files:
                return
            c = info.get("config") or {}
            parts = [f"✓ 확인됨: 파라미터 {info.get('params', 0) / 1e6:.1f}M", "스테레오" if c.get("stereo") else "모노",
                     f"대역 {c.get('num_bands')}개", f"{int(c.get('chunk_size') or 0) / 44100:.0f}초 조각"]
            panel.set_model_files(files, " · ".join(parts + list(info.get("notes") or [])))

        def failed(message: str, _detail: str) -> None:
            if self.session.model_files == files:
                panel.set_model_files(files, f"⚠ {message}")

        handle.finished.connect(done)
        handle.failed.connect(failed)

    def _ensure_model_files(self, s: AnalysisSettings, force: bool) -> bool:
        """RoFormer로 새로 분리해야 하는데 모델 파일이 없으면 고르게 한다. 고르지 않으면 False."""
        if s.separation != "mel_roformer" or self.session.model_files.get("checkpoint"):
            return True
        m = self.session.manifest
        if not force and m is not None and m.model in ("mel_roformer", "external"):
            return True  # 있는 분리 결과를 쓰므로 모델 파일이 필요 없다
        return self.choose_model_files()

    def import_stems_dialog(self) -> None:
        folder = self.session.folder
        if folder is None or self.session.audio is None or self.session.busy:
            return
        start = str(self.settings.value("import_stems_dir", "") or "")
        dlg = StemImportDialog(start, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        vocals, inst, tool = dlg.values()
        self.settings.setValue("import_stems_dir", os.path.dirname(vocals))
        self.session.import_stems(vocals, inst, tool)

    def _on_stems_imported(self, rep) -> None:
        self._update_actions()
        text = rep.summary()
        if rep.warnings:
            text += "\n\n주의:\n" + "\n".join(f"• {w}" for w in rep.warnings)
        settings = self.inspector.analysis.settings()
        if settings.separation == "none":
            text += ("\n\n분석 설정의 보컬 분리가 '분리 안 함'이라 지금은 가져온 보컬이 분석에 쓰이지 않습니다. "
                     "Demucs나 RoFormer로 바꾸면 분리를 다시 하지 않고 가져온 결과를 씁니다.")
        if self.doc.project.analysis is None and settings.separation != "none":
            QMessageBox.information(self, "분리 결과를 가져왔습니다",
                                    text + "\n\n보컬 분석 실행(F5)을 하면 가져온 보컬로 피치·음표를 계산합니다.")
            return
        if settings.separation == "none":
            QMessageBox.information(self, "분리 결과를 가져왔습니다", text)
            return
        answer = QMessageBox.question(
            self, "분리 결과를 가져왔습니다",
            text + "\n\n지금 음표는 이전 보컬로 계산한 것입니다. 새 보컬로 피치·음표를 다시 분석할까요? "
            "(직접 고친 음표를 어떻게 할지는 다음 창에서 고릅니다)")
        if answer == QMessageBox.StandardButton.Yes:
            self.run_analysis()

    def redetect_notes(self) -> None:
        self.session.redetect(self.inspector.analysis.settings())

    # --- 보컬 없는 구간 ----------------------------------------------------------

    def _on_analysis_settings_changed(self, s: AnalysisSettings) -> None:
        if hasattr(self, "review"):
            self.review.schedule()
        # 프로그램 기본 설정에는 곡마다 다른 보컬 없는 구간을 넣지 않는다 (프로젝트 저장 때 project.json에 남는다)
        self.settings.setValue(SETTINGS_KEY, {**s.to_dict(), "no_vocal_ranges": [], "pitch_regions": []})
        self.view.roll.set_no_vocal_ranges(s.no_vocal_ranges)
        self._sync_stem_mask(s)

    def _sync_stem_mask(self, s: AnalysisSettings) -> None:
        """보컬 없는 구간을 보컬·반주 트랙(재생·파형·내보내기)에도 적용한다. 분석 결과에는 영향이 없다."""
        self.session.set_no_vocal_mask(s.no_vocal_ranges if s.mute_no_vocal else ())

    def _apply_no_vocal(self, ranges) -> None:
        panel = self.inspector.analysis
        if tuple(ranges) == panel.settings().no_vocal_ranges:
            return
        panel.set_no_vocal_ranges(ranges)
        analysis = self.doc.project.analysis
        if analysis is not None and analysis.pitch_curve is not None and not self.session.busy:
            self.redetect_notes()  # 직접 고친 음표는 재분석 반영 규칙(수정 유지)대로 남는다
        else:
            self.statusBar().showMessage("보컬 없는 구간을 바꿨습니다. 다음 분석부터 적용됩니다.", 8000)

    def mark_no_vocal(self, start: float, end: float) -> None:
        current = self.inspector.analysis.settings().no_vocal_ranges
        self._apply_no_vocal(normalize_ranges([*current, (start, end)]))

    def edit_no_vocal_ranges(self) -> None:
        audio = self.session.audio
        dlg = NoVocalRangesDialog(self.inspector.analysis.settings().no_vocal_ranges,
                                  audio.duration if audio is not None else 0.0, self.player.position(), self)
        if dlg.exec() == NoVocalRangesDialog.DialogCode.Accepted:
            self._apply_no_vocal(dlg.ranges())

    def _apply_lite(self) -> None:
        """가벼운 모드 (크롬북 등): 메모리·디스크를 많이 쓰는 기능의 메뉴·단추를 숨긴다. 프로젝트 형식은 같다."""
        for a in (self.model_compare.act_mode, self.model_compare.act_run, self.model_compare.act_ab,
                  self.hybrid.act_mode, self.act_probe, self._compare_tb, self._hybrid_tb):
            a.setVisible(False)
        installed = separation_installed()
        if not installed:
            self.act_reseparate.setVisible(False)
        self.inspector.analysis.set_lite(True, installed)
        self.lyrics_panel.set_lite(True)
        log.info("가벼운 모드로 실행 (보컬 분리 설치: %s)", "예" if installed else "아니오")

    def edit_pitch_regions(self) -> None:
        """v0.6.1: 여러 소리가 겹쳐 음표가 빠진 구간을 목소리 음역을 정해 다시 찾는다. 바꾸면 분석을 다시 돌린다
        (분리·곡 전체 피치는 캐시를 쓰고 지정 구간만 새로 계산). 직접 고친 음표는 '수정 유지'로 남는다."""
        audio = self.session.audio
        panel = self.inspector.analysis
        dlg = PitchRegionsDialog(panel.settings().pitch_regions, audio.duration if audio is not None else 0.0,
                                 self.player.loop_region, self.player.position(), self)
        if dlg.exec() != PitchRegionsDialog.DialogCode.Accepted or tuple(dlg.regions()) == panel.settings().pitch_regions:
            return
        panel.set_pitch_regions(dlg.regions())
        if self.doc.project.analysis is not None and not self.session.busy and self.session.folder is not None:
            self.statusBar().showMessage("피치 다시 찾기 구간을 적용해 다시 분석합니다 (분리·곡 전체 피치는 캐시 사용).", 8000)
            self.run_analysis()
        else:
            self.statusBar().showMessage("피치 다시 찾기 구간을 바꿨습니다. 다음 분석부터 적용됩니다.", 8000)

    def run_sample_analysis(self) -> None:
        self.session.run_sample()

    def _on_analysis_ready(self, result: AnalysisResult) -> None:
        summary = summarize(self.doc.project)
        mode = KEEP_EDITS
        if summary.has_edits:
            preview = plan_reanalysis(self.doc.project, result, KEEP_EDITS)
            dlg = ReanalysisDialog(summary, len(result.notes), len(preview.kept), len(preview.dropped), self)
            if dlg.exec() != ReanalysisDialog.DialogCode.Accepted or dlg.mode is None:
                self.statusBar().showMessage("새 분석 결과를 반영하지 않았습니다.", 6000)
                return
            mode = dlg.mode
        elif self.doc.project.analysis is None and len(self.doc.score) == 0:
            mode = REPLACE_ALL
        plan = self.doc.apply_analysis(result, mode)
        note = " — 샘플 데이터이며 실제 분석 결과가 아닙니다." if result.is_placeholder else ""
        kept = f", 사용자 편집 {len(plan.kept)}개 유지" if plan.kept else ""
        self.statusBar().showMessage(f"음표 {len(plan.added)}개를 반영했습니다{kept}{note}", 10000)
        if result.warnings:
            QMessageBox.warning(self, "분석 안내", "\n".join(result.warnings))
        self.view.center_on_content()

    def compare_recent_runs(self) -> None:
        try:
            path, cmp = self.session.compare_recent_runs()
        except AmpError as e:
            QMessageBox.information(self, "실행 비교", str(e))
            return
        QMessageBox.information(
            self, "실행 비교",
            f"{cmp.a.run_id} → {cmp.b.run_id}\n"
            f"음표 {len(cmp.a.notes)}개 → {len(cmp.b.notes)}개 (일치 {len(cmp.matched)}, 음높이만 바뀜 "
            f"{len(cmp.pitch_changed)}, 사라짐 {len(cmp.only_a)}, 새로 생김 {len(cmp.only_b)})\n\n"
            f"보고서: {path}")

    def clear_cache(self) -> None:
        answer = QMessageBox.question(self, "캐시 비우기",
                                      "cache/ 폴더의 중간 결과를 지웁니다. 원본·분리 결과·분석 결과·편집 내용·내보낸 파일은 "
                                      "그대로입니다. 다음 분석은 피치 추출을 다시 계산합니다. 계속할까요?")
        if answer == QMessageBox.StandardButton.Yes:
            self.session.clear_cache()
            self._update_actions()

    def probe_gpu(self) -> None:
        handle = self.session.probe()
        label = self.inspector.analysis.device_info

        def done(info: dict) -> None:
            if info.get("error"):
                label.setText(f"⚠ {info['error']}  {install_script('separation')} 으로 설치할 수 있습니다.")
            elif info.get("cuda"):
                label.setText(f"✓ GPU 사용 가능: {info.get('gpu')} (CUDA {info.get('cuda_version')}, PyTorch {info.get('torch')})")
            elif info.get("hint"):
                label.setText(f"⚠ {info['hint']}")
            else:
                label.setText(f"CPU로 실행합니다. CUDA를 쓸 수 없습니다 (PyTorch {info.get('torch')}).")

        handle.finished.connect(done)

    # --- 내보내기 -----------------------------------------------------------------

    def export_dialog(self, preselect: list[str]) -> None:
        f = self.session.folder
        if f is None:
            return
        available = {n: self.session.track_audio(n) is not None for n in TRACK_LABELS}
        dlg = ExportDialog(f.name, available, str(f.paths.exports), preselect or ["vocals", "instrumental"], self,
                           no_vocal_ranges=len(self.session._applied_mask))
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        jobs = dlg.jobs()
        if not jobs:
            return
        existing = [j.path for j in jobs if os.path.exists(j.path)]
        if existing:
            answer = QMessageBox.question(self, "덮어쓰기 확인", "같은 이름의 파일이 있습니다. 덮어쓸까요?\n"
                                          + "\n".join(Path(p).name for p in existing))
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.session.export([(j.track, j.path) for j in jobs], dlg.options())

    # --- 기타 -----------------------------------------------------------------

    def show_help(self) -> None:
        QMessageBox.information(self, "조작법 및 구현 현황", HELP_TEXT)

    def show_about(self) -> None:
        QMessageBox.about(self, "정보", f"<b>{APP_DISPLAY_NAME}</b> {__version__}<br>"
                          "Python + PySide6 보컬 분리·자동 채보 편집기 (개발 중)<br><br>"
                          "글꼴: Pretendard, JetBrains Mono (SIL OFL 1.1) · 아이콘: Lucide (ISC)<br>"
                          "보컬 분리: Demucs htdemucs (MIT), Mel-Band RoFormer 계산 코드(MIT, 가중치 미포함) · 피치·BPM: librosa (ISC)<br>"
                          "가사 인식: OpenAI Whisper (MIT) · 가사 정렬: MMS_FA 가중치 (CC-BY-NC 4.0, 비상업적 이용)")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self.maybe_save():
            event.ignore()
            return
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.splitter.sizes())
        f = self.session.folder
        if f is not None:  # 작업 공간·재생 위치만 기록 (음표 내용은 저장 여부를 따른다)
            try:
                f.meta.ui = self._ui_state()
                f.meta.playback = self._playback_state()
                f.write_meta()
            except Exception:  # noqa: BLE001 - 닫기를 막지 않는다
                log.exception("작업 공간 상태 저장 실패")
        self.listening.shutdown()
        self.player.shutdown()
        if not self.runner.shutdown(5000):
            log.warning("백그라운드 작업이 5초 안에 끝나지 않았습니다.")
        self.session.close()
        self.autosave.shutdown()
        self.log_panel.detach()
        event.accept()
