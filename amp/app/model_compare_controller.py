"""모델 비교 모드 (STEP 5): 보컬 분리 모델 A/B 비교·청취·채보 차이·분석 소스 선택.

- 비교 모드는 기본 편집 화면과 따로 켠다(분석 > 모델 비교 모드, Ctrl+Shift+M). 켜면 왼쪽 트랙 패널 자리에 A/B 패널,
  오른쪽에 '모델 차이' 탭이 나오고, 스펙트로그램·피아노 롤에 A/B가 겹쳐 그려진다. 끄면 모두 원래대로.
- A/B 청취: A·B 분리 결과를 재생 트랙 cmp_a·cmp_b로 올리고 한쪽만 솔로로 둔다. 전환은 같은 재생 스트림 안에서 솔로만
  바꾸므로 시간 위치·반복 구간·느린 재생이 그대로고, 믹서가 12ms 크로스페이드한다.
- 청취용 음량 맞춤은 재생 음량만 바꾼다. 분리 WAV는 읽기만 한다.
- 채보 차이는 '차이'만 보여 준다. 정답 채보가 없으므로 정확도나 우수 모델을 표시하지 않는다.
- 분석 소스 선택은 사용자가 고를 때만 한다. 바뀔 항목을 먼저 보여 주고, 사용자 편집은 '수정 유지'로 지킨다.
  다른 모델의 분리 결과 파일은 지우지 않는다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.app import theme
from amp.app.player import COMPARE_TRACKS
from amp.app.widgets import shrinkable_combos
from amp.app.widgets.spectrogram import COMPARE_MODES
from amp.audio import loudness
from amp.compare import model_diff
from amp.compare import source_switch
from amp.engine.model_compare import STATUS_LABELS, ModelSpec

log = logging.getLogger(__name__)

CMP_A, CMP_B = COMPARE_TRACKS
STATE_FILE = "model_compare.json"  # analysis/ 안. A/B로 고른 결과 키만 적는다 (분리 결과는 stems/models/)
TRACK_KINDS = {"vocals": "보컬", "instrumental": "반주"}


def _size(n: int) -> str:
    return f"{n / 2**20:.0f}MB"


class CompareRunDialog(QDialog):
    """현재 모델만 / htdemucs / Mel-Band RoFormer / 두 모델 비교."""

    def __init__(self, current: str, has_ckpt: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("분리 모델 비교 실행")
        lay = QVBoxLayout(self)
        self.group = QButtonGroup(self)
        self.options = {}
        cur_label = {"htdemucs": "htdemucs", "mel_roformer": "Mel-Band RoFormer"}.get(current, "")
        for key, text in (("current", f"현재 모델만 분리 ({cur_label or '분리 안 함으로 설정됨'})"), ("htdemucs", "htdemucs 실행"),
                          ("mel_roformer", "Mel-Band RoFormer 실행"), ("both", "두 모델 비교 실행 (htdemucs → RoFormer 차례로)")):
            b = QRadioButton(text)
            self.group.addButton(b)
            self.options[key] = b
            lay.addWidget(b)
        if not cur_label:
            self.options["current"].setEnabled(False)
        if not has_ckpt:
            for k in ("mel_roformer", "both"):
                self.options[k].setEnabled(False)
            note = QLabel("RoFormer 모델 파일(.ckpt)을 고르지 않았습니다. 분석 설정의 보컬 분리에서 [모델 파일…]로 지정하세요.")
            note.setWordWrap(True)
            note.setProperty("role", "secondary")
            lay.addWidget(note)
        (self.options["both"] if has_ckpt else self.options["htdemucs"]).setChecked(True)
        self.reuse = QCheckBox("보관한 결과가 있으면 다시 계산하지 않고 쓰기")
        self.reuse.setChecked(True)
        lay.addWidget(self.reuse)
        info = QLabel("모델은 하나씩 차례로 실행합니다(두 모델을 GPU에 함께 올리지 않음). 한 모델이 끝나면 그 작업 프로세스를 "
                      "닫아 GPU 메모리를 돌려준 뒤 다음 모델을 실행합니다. GPU 메모리가 부족하면 CPU로 바꾸지 않고 묻습니다. "
                      "취소해도 이미 끝난 결과는 남습니다. 지금 분석에 쓰는 결과와 사용자 편집은 바꾸지 않습니다.")
        info.setWordWrap(True)
        info.setProperty("role", "secondary")
        lay.addWidget(info)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("실행")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.current = current

    def choice(self) -> str:
        return next(k for k, b in self.options.items() if b.isChecked())

    def specs(self, model_files: dict) -> list[ModelSpec]:
        c = self.choice()
        names = {"current": [self.current], "htdemucs": ["htdemucs"], "mel_roformer": ["mel_roformer"],
                 "both": ["htdemucs", "mel_roformer"]}[c]
        return [ModelSpec(n, dict(model_files) if n == "mel_roformer" else {}) for n in names]


class SourceSwitchDialog(QDialog):
    def __init__(self, label: str, lines: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("분석 소스 바꾸기")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        title = QLabel(f"'{label}' 보컬을 분석 소스로 씁니다")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        body = QLabel("\n".join(["바뀔 수 있는 항목:", *(f"• {x}" for x in lines), "",
                                 "피치 추출·음표 검출·가사 인식·가사 시간 정렬·발음/멜리스마 분석은 이후 이 보컬을 씁니다.",
                                 "다른 모델의 분리 결과 파일은 지우지 않습니다. 음표·가사 반영은 편집 > 실행 취소로 되돌릴 수 있습니다."]))
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(body)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("이 결과로 바꾸기 (수정 유지)")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class ModelComparePanel(QWidget):
    """비교 모드 왼쪽 패널: 보관한 분리 결과, A/B 선택, 청취, 표시, 분석 소스."""

    actionRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setMinimumWidth(260)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        title = QLabel("모델 비교")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        hint = QLabel("같은 곡을 두 분리 모델로 나눈 결과를 같은 시간축에서 듣고 봅니다. 어느 쪽이 나은지는 직접 들어 보고 "
                      "판단하세요. 이 화면은 사용자 편집을 바꾸지 않습니다.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)

        lay.addWidget(self._section("보관한 분리 결과"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("모델", "상태", "장치", "크기"))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(110)
        lay.addWidget(self.table)
        g = QGridLayout()
        self.buttons: dict[str, QPushButton] = {}
        for i, (key, text, tip) in enumerate((
                ("set_a", "A로", "고른 결과를 A로"), ("set_b", "B로", "고른 결과를 B로"),
                ("reveal", "파일 위치", "고른 결과의 WAV가 있는 폴더 열기"),
                ("delete", "지우기…", "고른 분리 결과를 지웁니다 (지금 분석에 쓰는 결과는 지울 수 없음)"))):
            b = self._button(key, text, tip)
            g.addWidget(b, i // 2, i % 2)  # 2열: 좁은 패널(250px)에서도 가로 스크롤 없이
        lay.addLayout(g)
        row = QHBoxLayout()
        row.addWidget(self._button("run", "비교 실행…", "htdemucs·RoFormer 분리와 모델별 채보"), 1)
        row.addWidget(self._button("transcribe", "모델별 채보", "보관한 A·B 결과를 같은 설정으로 채보 (분리는 안 함)"), 1)
        lay.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "secondary")
        lay.addWidget(self.status)

        lay.addWidget(self._section("A / B"))
        form = QGridLayout()
        self.combo_a = QComboBox()
        self.combo_b = QComboBox()
        tag_a = QLabel("A")
        tag_a.setStyleSheet(f"color: {theme.MODEL_A.name()}; font-weight: 700;")
        tag_b = QLabel("B")
        tag_b.setStyleSheet(f"color: {theme.MODEL_B.name()}; font-weight: 700;")
        form.addWidget(tag_a, 0, 0)
        form.addWidget(self.combo_a, 0, 1)
        form.addWidget(tag_b, 1, 0)
        form.addWidget(self.combo_b, 1, 1)
        form.addWidget(QLabel("트랙"), 2, 0)
        self.track = QComboBox()
        for k, v in TRACK_KINDS.items():
            self.track.addItem(v, k)
        form.addWidget(self.track, 2, 1)
        form.setColumnStretch(1, 1)
        lay.addLayout(form)
        self.paths = QLabel("")
        self.paths.setWordWrap(True)
        self.paths.setProperty("role", "secondary")
        self.paths.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.paths)

        lay.addWidget(self._section("청취"))
        row = QHBoxLayout()
        row.addWidget(self._button("listen_a", "A 듣기", "A만 들림 (재생 위치 유지)"), 1)
        row.addWidget(self._button("listen_b", "B 듣기", "B만 들림 (재생 위치 유지)"), 1)
        row.addWidget(self._button("toggle", "A/B 전환", "Ctrl+B"), 1)
        lay.addLayout(row)
        self.now = QLabel("듣는 쪽: —")
        lay.addWidget(self.now)
        self.level_mode = QComboBox()
        self.level_mode.addItem("원본 음량", "raw")
        self.level_mode.addItem("청취용 음량 맞춤", "match")
        self.level_mode.setToolTip("큰 소리가 더 좋게 들리는 편향을 줄입니다. 재생 음량만 바꾸고 분리 WAV는 바꾸지 않습니다.")
        lay.addWidget(self.level_mode)
        self.level_info = QLabel("")
        self.level_info.setWordWrap(True)
        self.level_info.setProperty("role", "secondary")
        lay.addWidget(self.level_info)
        vol = QGridLayout()
        self.vol_a = self._slider()
        self.vol_b = self._slider()
        self.vol_a_label = QLabel("0 dB")
        self.vol_b_label = QLabel("0 dB")
        vol.addWidget(QLabel("A 음량"), 0, 0)
        vol.addWidget(self.vol_a, 0, 1)
        vol.addWidget(self.vol_a_label, 0, 2)
        vol.addWidget(QLabel("B 음량"), 1, 0)
        vol.addWidget(self.vol_b, 1, 1)
        vol.addWidget(self.vol_b_label, 1, 2)
        vol.setColumnStretch(1, 1)
        lay.addLayout(vol)
        lay.addWidget(self._button("goto_region", "반복 구간으로 이동", "반복 구간 시작으로 이동"))

        lay.addWidget(self._section("표시"))
        self.spec_mode = QComboBox()
        for k, v in COMPARE_MODES.items():
            if k != "off":
                self.spec_mode.addItem(f"스펙트로그램: {v}", k)
        lay.addWidget(self.spec_mode)
        grid = QGridLayout()
        self.layer_boxes = {}
        for i, (key, text) in enumerate((("a_notes", "A 음표"), ("b_notes", "B 음표"), ("a_curve", "A 피치 곡선"),
                                         ("b_curve", "B 피치 곡선"), ("user_notes", "작업본 음표"))):
            cb = QCheckBox(text)
            cb.setChecked(True)
            self.layer_boxes[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        lay.addLayout(grid)
        legend = QLabel("A = 바이올렛 점선(행 위쪽 절반), B = 민트 점점선(행 아래쪽 절반). 작업본 음표(채운 막대)는 편집할 수 "
                        "있고, A/B 음표는 비교용 그림이라 편집되지 않습니다.")
        legend.setWordWrap(True)
        legend.setProperty("role", "secondary")
        lay.addWidget(legend)

        lay.addWidget(self._section("분석 소스"))
        self.source = QLabel("—")
        self.source.setWordWrap(True)
        lay.addWidget(self.source)
        lay.addWidget(self._button("use_a", "A를 분석에 사용…", "바뀔 항목을 먼저 보여 줍니다"))
        lay.addWidget(self._button("use_b", "B를 분석에 사용…", "바뀔 항목을 먼저 보여 줍니다"))
        lay.addWidget(self._button("exit", "비교 모드 끝내기", "기본 편집 화면으로"))
        lay.addStretch(1)
        shrinkable_combos(body)

    def _section(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "secondary")
        lab.setStyleSheet("font-weight: 600; margin-top: 8px;")
        return lab

    def _button(self, key: str, text: str, tip: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda _c=False, k=key: self.actionRequested.emit(k))
        self.buttons[key] = b
        return b

    @staticmethod
    def _slider() -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(-24, 6)
        s.setValue(0)
        s.setToolTip("비교용 재생 음량 (dB). 파일은 바꾸지 않습니다")
        return s

    def selected_key(self) -> str | None:
        r = self.table.currentRow()
        if r < 0:
            return None
        it = self.table.item(r, 0)
        return None if it is None else it.data(Qt.ItemDataRole.UserRole)


class ModelDiffPanel(QWidget):
    """오른쪽 '모델 차이' 탭: 지표 표 (A, B 나란히). 우열 표시 없음."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.header = QLabel("모델별 채보 결과가 아직 없습니다. 왼쪽 패널에서 [비교 실행…] 또는 [모델별 채보]를 누르세요.")
        self.header.setWordWrap(True)
        lay.addWidget(self.header)
        note = QLabel(model_diff.DISCLAIMER)
        note.setWordWrap(True)
        note.setProperty("role", "secondary")
        lay.addWidget(note)
        self.table = self._table(("지표", "A", "B"))
        lay.addWidget(self.table, 2)
        self.shared = self._table(("두 결과 사이", "값"))
        lay.addWidget(self.shared, 2)
        self.region = QLabel("")
        self.region.setWordWrap(True)
        self.region.setProperty("role", "secondary")
        lay.addWidget(self.region)
        hint = QLabel("검토 탭의 '모델 비교' 항목을 고르면 그 시각으로 이동합니다. [구간 반복]으로 그 구간을 A/B 전환하며 들어 "
                      "보세요. 검토 상태(확인 완료·무시 등)는 프로젝트에 저장됩니다.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)

    @staticmethod
    def _table(headers) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(headers)):
            t.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        return t

    def set_diff(self, d: model_diff.ModelDiff | None, region_text: str = "") -> None:
        if d is None:
            self.table.setRowCount(0)
            self.shared.setRowCount(0)
            self.region.setText("")
            return
        self.header.setText(f"A: {d.label_a}\nB: {d.label_b}")
        for t, rows in ((self.table, d.summary_rows()), (self.shared, d.shared_rows())):
            t.setRowCount(len(rows))
            for r, vals in enumerate(rows):
                for c, v in enumerate(vals):
                    cell = QTableWidgetItem(v)
                    if c:
                        cell.setFont(theme.mono_font())
                    t.setItem(r, c, cell)
        self.region.setText(region_text)


class ModelCompareController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.panel = ModelComparePanel()
        self.diff_panel = ModelDiffPanel()
        self.active = False
        self.keys = {"a": None, "b": None}
        self.outputs: dict[str, object] = {}  # 결과 키 → TranscriptionOutput (이번 실행 동안)
        self.audio: dict[tuple, object] = {}  # (결과 키, 트랙 종류) → AudioData. 지금 A/B의 고른 트랙만 메모리에 둔다
        self._saved_states: dict[str, dict] = {}  # 비교 모드 전 트랙 음량·음소거·솔로 (나올 때 되돌림)
        self.levels: dict[tuple, float | None] = {}  # (키, 트랙) → 청취 음량 dB
        self.listening = "a"
        self.diff: model_diff.ModelDiff | None = None
        self._saved_lane = None
        self._entries = []
        self.act_mode = QAction("모델 비교 모드", window)
        self.act_mode.setCheckable(True)
        self.act_mode.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self.act_mode.setToolTip("보컬 분리 모델 A/B 비교 화면 (Ctrl+Shift+M)")
        self.act_mode.toggled.connect(self.set_active)
        self.act_run = QAction("분리 모델 비교 실행…", window)
        self.act_run.triggered.connect(self.run_dialog)
        self.act_ab = QAction("A/B 전환", window)
        self.act_ab.setShortcut(QKeySequence("Ctrl+B"))
        self.act_ab.triggered.connect(self.toggle_ab)
        window.addAction(self.act_ab)
        p = self.panel
        p.actionRequested.connect(self.on_action)
        p.combo_a.currentIndexChanged.connect(lambda _i: self._on_combo("a"))
        p.combo_b.currentIndexChanged.connect(lambda _i: self._on_combo("b"))
        p.track.currentIndexChanged.connect(lambda _i: self._load_tracks())
        p.level_mode.currentIndexChanged.connect(lambda _i: self._apply_gains())
        p.vol_a.valueChanged.connect(lambda _v: self._apply_gains())
        p.vol_b.valueChanged.connect(lambda _v: self._apply_gains())
        p.spec_mode.currentIndexChanged.connect(lambda _i: self._apply_spec_mode())
        for key, cb in p.layer_boxes.items():
            cb.toggled.connect(lambda on, k=key: self._on_show(k, on))
        window.session.audioReady.connect(self._on_project_changed)
        window.doc.projectReset.connect(self._on_project_changed)
        window.player.playbackChanged.connect(self._update_region_text)

    # --- 모드 -----------------------------------------------------------------------

    def set_active(self, on: bool) -> None:
        w = self.w
        if on and w.session.folder is None:
            w.statusBar().showMessage("프로젝트를 먼저 여세요.", 4000)
            self.act_mode.blockSignals(True)
            self.act_mode.setChecked(False)
            self.act_mode.blockSignals(False)
            return
        if on == self.active:
            return
        self.active = on
        if self.act_mode.isChecked() != on:
            self.act_mode.blockSignals(True)
            self.act_mode.setChecked(on)
            self.act_mode.blockSignals(False)
        if on:
            self._saved_states = {n: w.player.track_state(n) for n in w.player.tracks}
            w.splitter.replaceWidget(0, self.panel)
            self.panel.show()
            w.tracks.hide()
            w.right_tabs.addTab(self.diff_panel, "모델 차이")
            self._saved_lane = w.listening.lane_mode
            if w.listening.lane_mode in ("hidden", "wave"):
                w.listening.set_lane_mode("both")
            self.refresh_entries()
            self._restore_keys()
            self._load_tracks()
            self._update_overlays()
            w.statusBar().showMessage("모델 비교 모드: A/B 전환 Ctrl+B. 끝내려면 Ctrl+Shift+M", 6000)
        else:
            self._remove_tracks()
            w.spectrogram.set_compare(None, None)
            w.spectrogram.set_compare_mode("off")
            w.view.roll.set_model_layers(None)
            w.view.roll.set_notes_visible(w.act_show_notes.isChecked())
            w.review.set_extra_items([])
            i = w.right_tabs.indexOf(self.diff_panel)
            if i >= 0:
                w.right_tabs.removeTab(i)
            w.splitter.replaceWidget(0, w.tracks)
            w.tracks.show()
            self.panel.hide()
            if self._saved_lane is not None:
                w.listening.set_lane_mode(self._saved_lane)
            self._saved_lane = None
        w._update_actions()

    def _on_project_changed(self, *_a) -> None:
        self.outputs.clear()
        self.audio.clear()
        self.levels.clear()
        self.diff = None
        self.keys = {"a": None, "b": None}
        if self.active:
            self.set_active(False)

    # --- 보관소 ---------------------------------------------------------------------

    def store(self):
        return self.w.session.model_store()

    def refresh_entries(self) -> None:
        store = self.store()
        try:
            if store is not None and store.migrate_legacy() is not None:
                self.w.session.load_stems()  # 예전 평면 결과를 옮겼다: 세션이 가진 목록(파일 경로)을 새로 읽는다
            self._entries = store.entries() if store is not None else []
        except OSError as e:
            log.warning("분리 결과 목록을 읽지 못함: %s", e)
            self._entries = []
        p = self.panel
        t = p.table
        t.setRowCount(len(self._entries))
        for r, e in enumerate(self._entries):
            info = e.manifest.model_info
            status = "분석에 사용 중" if e.active else ("채보 있음" if e.key in self.outputs else "보관")
            vals = (e.label, status, str(info.get("device", "")).upper() or "—", _size(e.size_bytes))
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, e.key)
                    cell.setToolTip(f"{e.key}\n만든 때 {e.manifest.created}\nstems/models/{e.key}/")
                t.setItem(r, c, cell)
        for role, combo in (("a", p.combo_a), ("b", p.combo_b)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("—", None)
            for e in self._entries:
                combo.addItem(f"{e.label} · {e.key[-10:]}", e.key)
            i = combo.findData(self.keys[role])
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)
        active = next((e for e in self._entries if e.active), None)
        p.source.setText(f"지금 분석에 쓰는 결과: {active.label}" if active else "지금 분석에 쓰는 분리 결과: 없음 (분리 전 또는 원본)")
        self._update_paths()

    def entry(self, key: str | None):
        return next((e for e in self._entries if e.key == key), None)

    def label(self, key: str | None) -> str:
        e = self.entry(key)
        return e.label if e is not None else "—"

    def _state_path(self):
        f = self.w.session.folder
        return None if f is None else f.paths.analysis / STATE_FILE

    def _save_keys(self) -> None:
        path = self._state_path()
        if path is None:
            return
        try:
            from amp.core.fileio import atomic_write_json

            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {"schema": "sonatlas.model_compare", "schema_version": 1, "a": self.keys["a"],
                                     "b": self.keys["b"], "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        except OSError as e:
            log.warning("모델 비교 선택을 저장하지 못함: %s", e)

    def _restore_keys(self) -> None:
        if self.keys["a"] or self.keys["b"]:
            return
        path = self._state_path()
        keys = {e.key for e in self._entries}
        try:
            d = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.exists() else {}
        except (OSError, ValueError):
            d = {}
        for role in ("a", "b"):
            k = d.get(role)
            self.keys[role] = k if k in keys else None
        if not self.keys["a"] and not self.keys["b"] and self._entries:  # 처음: 보관한 결과 앞의 둘
            ordered = sorted(self._entries, key=lambda e: (e.model != "htdemucs", e.manifest.created))
            self.keys["a"] = ordered[0].key
            self.keys["b"] = ordered[1].key if len(ordered) > 1 else None
        self.refresh_entries()

    def _on_combo(self, role: str) -> None:
        combo = self.panel.combo_a if role == "a" else self.panel.combo_b
        self.keys[role] = combo.currentData()
        self._save_keys()
        self._load_tracks()
        self._update_overlays()

    def _update_paths(self) -> None:
        lines = []
        for role in ("a", "b"):
            e = self.entry(self.keys[role])
            if e is not None:
                lines.append(f"{role.upper()}: stems/models/{e.key}/")
        self.panel.paths.setText("\n".join(lines))

    # --- 청취 -----------------------------------------------------------------------

    def _load_tracks(self) -> None:
        """A·B의 고른 트랙(보컬/반주)을 재생 트랙 cmp_a·cmp_b로 올린다. WAV는 읽기만 한다."""
        if not self.active:
            return
        import numpy as np
        import soundfile as sf

        from amp.audio.data import AudioData

        w = self.w
        kind = self.panel.track.currentData()
        loaded = {}
        for role in ("a", "b"):
            key = self.keys[role]
            e = self.entry(key)
            if e is None or kind not in e.manifest.files:
                loaded[role] = None
                continue
            if (key, kind) not in self.audio:
                try:  # 읽기 전용으로 연다. 분리 WAV는 바꾸지 않는다
                    data, sr = sf.read(str(e.path / e.manifest.files[kind]["path"]), dtype="float32", always_2d=True)
                    self.audio[(key, kind)] = AudioData(np.ascontiguousarray(data), sr)
                except Exception as ex:  # noqa: BLE001 - 파일 손상·삭제
                    w._show_error("분리 결과를 읽지 못했습니다", f"{e.label}: {ex}")
                    loaded[role] = None
                    continue
            loaded[role] = self.audio[(key, kind)]
        keep = {(self.keys["a"], kind), (self.keys["b"], kind)}
        for k in [k for k in self.audio if k not in keep]:
            self.audio.pop(k)  # A/B의 고른 트랙이 아닌 것은 메모리에서 내린다
        for role, name in (("a", CMP_A), ("b", CMP_B)):
            if w.player.tracks.get(name) is not loaded[role]:
                w.player.set_track(name, loaded[role])
        w.spectrogram.set_compare(
            (f"{self.label(self.keys['a'])} {TRACK_KINDS[kind]}", loaded["a"]) if loaded["a"] is not None else None,
            (f"{self.label(self.keys['b'])} {TRACK_KINDS[kind]}", loaded["b"]) if loaded["b"] is not None else None)
        self._apply_spec_mode()
        self._measure_levels()
        self._apply_gains()
        self._update_paths()

    def _measure_levels(self) -> None:
        kind = self.panel.track.currentData()
        for role in ("a", "b"):
            key = self.keys[role]
            if key is None or (key, kind) in self.levels or (key, kind) not in self.audio:
                continue
            a = self.audio[(key, kind)]
            self.levels[(key, kind)] = loudness.gated_level_db(a.samples, a.sample_rate)

    def _remove_tracks(self) -> None:
        w = self.w
        restore = {n: dict(self._saved_states.get(n, {"gain": 1.0, "mute": False, "solo": False}))
                   for n in w.player.tracks if n not in (CMP_A, CMP_B)}
        w.player.set_track_states(restore)  # 비교 모드 전 음량·음소거·솔로로
        for name in (CMP_A, CMP_B):
            if name in w.player.tracks:
                w.player.set_track(name, None)
        self.audio.clear()

    def gains(self) -> dict[str, float]:
        """재생 배율: 사용자 음량(dB) × 청취용 음량 맞춤."""
        p = self.panel
        kind = p.track.currentData()
        la = self.levels.get((self.keys["a"], kind))
        lb = self.levels.get((self.keys["b"], kind))
        ga, gb = loudness.matching_gains(la, lb) if p.level_mode.currentData() == "match" else (1.0, 1.0)
        ua, ub = 10 ** (p.vol_a.value() / 20), 10 ** (p.vol_b.value() / 20)
        p.vol_a_label.setText(f"{p.vol_a.value():+d} dB")
        p.vol_b_label.setText(f"{p.vol_b.value():+d} dB")
        if la is not None and lb is not None:
            import math

            txt = f"측정 음량 A {la:.1f}dB · B {lb:.1f}dB (무음 블록 제외 RMS, LUFS 아님)"
            if p.level_mode.currentData() == "match":
                txt += f" → 재생 보정 A {20 * math.log10(ga):+.1f}dB · B {20 * math.log10(gb):+.1f}dB"
            p.level_info.setText(txt)
        else:
            p.level_info.setText("")
        return {CMP_A: ga * ua, CMP_B: gb * ub}

    def _apply_gains(self) -> None:
        if not self.active:
            return
        g = self.gains()
        self.listen(self.listening, gains=g)

    def listen(self, role: str, gains: dict | None = None) -> None:
        """A 또는 B만 들리게 한다. 재생 중이어도 위치·반복·속도는 그대로, 믹서가 짧게 크로스페이드."""
        w = self.w
        if not self.active:
            return
        self.listening = role
        g = gains or self.gains()
        names = set(w.player.tracks)
        target = CMP_A if role == "a" else CMP_B
        changes = {n: {"solo": False} for n in names}
        if target in names:
            changes[target] = {"solo": True, "mute": False, "gain": g[target]}
        other = CMP_B if role == "a" else CMP_A
        if other in names:
            changes[other] = {"solo": False, "gain": g[other]}
        w.player.set_track_states(changes)
        label = self.label(self.keys[role])
        missing = "" if target in names else " (결과 없음 — 다른 트랙이 들립니다)"
        self.panel.now.setText(f"듣는 쪽: {role.upper()} — {label}{missing}")
        self.panel.now.setStyleSheet(f"color: {(theme.MODEL_A if role == 'a' else theme.MODEL_B).name()}; font-weight: 600;")

    def toggle_ab(self) -> None:
        if self.active:
            self.listen("b" if self.listening == "a" else "a")

    # --- 표시 -----------------------------------------------------------------------

    def _apply_spec_mode(self) -> None:
        if self.active:
            self.w.spectrogram.set_compare_mode(self.panel.spec_mode.currentData() or "side")

    def _on_show(self, key: str, on: bool) -> None:
        if key == "user_notes":
            self.w.view.roll.set_notes_visible(on)
        else:
            self.w.view.roll.set_model_visibility(**{key: on})

    def _update_overlays(self) -> None:
        if not self.active:
            return
        w = self.w
        layers = {}
        for role in ("a", "b"):
            out = self.outputs.get(self.keys[role])
            if out is None:
                continue
            res = out.result
            layers[role] = {"notes": [(n.start, n.end, n.pitch) for n in res.notes], "curve": res.pitch_curve,
                            "label": self.label(self.keys[role])}
        w.view.roll.set_model_layers(layers)
        for key, cb in self.panel.layer_boxes.items():
            self._on_show(key, cb.isChecked())
        a, b = (self.outputs.get(self.keys[r]) for r in ("a", "b"))
        if a is not None and b is not None:
            ranges = tuple(w.inspector.analysis.settings().no_vocal_ranges)
            self.diff = model_diff.compare(a.result, b.result, self.label(self.keys["a"]), self.label(self.keys["b"]), ranges)
            w.review.set_extra_items(model_diff.review_items(self.diff))
        else:
            self.diff = None
            w.review.set_extra_items([])
        self._update_region_text()
        self.refresh_entries()

    def _update_region_text(self, *_a) -> None:
        d = self.diff
        region = self.w.player.loop_region
        text = ""
        if d is not None and region is not None:
            a, b = region

            def inside(s, e):
                return e > a and s < b

            text = (f"반복 구간 {a:.2f}~{b:.2f}초 안: A에만 있는 음표 {sum(inside(n.start, n.end) for n in d.only_a)}개, "
                    f"B에만 있는 음표 {sum(inside(n.start, n.end) for n in d.only_b)}개, 음높이 다른 쌍 "
                    f"{sum(inside(p.a.start, p.a.end) for p in d.pitch_changed)}개, 옥타브 차이 의심 "
                    f"{sum(inside(s, e) for s, e, _m in d.octave)}곳")
        self.diff_panel.set_diff(d, text)

    # --- 실행 -----------------------------------------------------------------------

    def run_dialog(self) -> None:
        w = self.w
        if w.session.folder is None or w.session.audio is None:
            w.statusBar().showMessage("프로젝트를 먼저 여세요.", 4000)
            return
        settings = w.inspector.analysis.settings()
        dlg = CompareRunDialog(settings.separation, bool(w.session.model_files.get("checkpoint")), w)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.run(dlg.specs(w.session.model_files), force=not dlg.reuse.isChecked())

    def run(self, specs: list[ModelSpec], force: bool = False):
        w = self.w
        if not self.active:
            self.set_active(True)
        handle = w.session.run_model_compare(w.inspector.analysis.settings(), specs, force)
        if handle is None:
            return None
        self.panel.status.setText("비교 실행 중… (분석 > 분석 취소로 멈출 수 있음)")
        handle.finished.connect(self._on_run_done)
        handle.cancelled.connect(lambda: self.panel.status.setText("취소했습니다. 끝난 분리 결과는 남아 있습니다."))
        handle.failed.connect(lambda msg, _d: self.panel.status.setText(f"실패: {msg}"))
        return handle

    def _on_run_done(self, run) -> None:
        lines = []
        oom = []
        for oc in run.outcomes:
            dev = f" · {oc.device.upper()}" if oc.device else ""
            extra = f" · {oc.seconds:.0f}초" if oc.status == "done" else ""
            lines.append(f"{oc.spec.label}: {STATUS_LABELS.get(oc.status, oc.status)}{dev}{extra}"
                         + (f" — {oc.message.splitlines()[0]}" if oc.message and oc.status not in ("done", "reused") else ""))
            if oc.status == "oom":
                oom.append(oc.spec)
            if oc.output is not None and oc.key:
                self.outputs[oc.key] = oc.output
        if run.cancelled:
            lines.append("취소됨 — 이미 끝난 결과는 보관소에 남았습니다.")
        self.panel.status.setText("\n".join(lines))
        ok = [oc.key for oc in run.available()]
        if ok:
            self.keys["a"] = ok[0]
            if len(ok) > 1:
                self.keys["b"] = ok[1]
            self._save_keys()
        self.refresh_entries()
        self._load_tracks()
        self._update_overlays()
        if oom:  # 완료 처리 중에는 세션이 아직 '작업 중'이라 새 실행을 받지 않는다: 다음 이벤트 루프에서 묻는다
            QTimer.singleShot(0, lambda specs=list(oom): self._ask_cpu_retry(specs))

    def _ask_cpu_retry(self, oom: list[ModelSpec]) -> None:
        names = ", ".join(s.label for s in oom)
        ans = QMessageBox.question(
            self.w, "GPU 메모리 부족",
            f"GPU 메모리가 부족해 {names} 분리를 마치지 못했습니다. 자동으로 CPU로 바꾸지 않았습니다.\n\n"
            "다른 프로그램(게임·브라우저 등)을 닫고 다시 실행하거나, CPU로 다시 실행할 수 있습니다. "
            "CPU는 훨씬 오래 걸립니다(곡 길이에 따라 수 분~수십 분). CPU로 다시 실행할까요?")
        if ans == QMessageBox.StandardButton.Yes:
            from dataclasses import replace

            self.run([replace(s, device="cpu") for s in oom])

    def transcribe(self) -> None:
        w = self.w
        keys = [k for k in (self.keys["a"], self.keys["b"]) if k]
        if not keys:
            w.statusBar().showMessage("A·B로 쓸 분리 결과를 먼저 고르세요.", 4000)
            return
        handle = w.session.transcribe_entries(w.inspector.analysis.settings(), keys)
        if handle is None:
            return
        self.panel.status.setText("모델별 채보 중…")

        def done(outs: dict) -> None:
            errs = []
            for k, v in outs.items():
                if isinstance(v, str):
                    errs.append(f"{self.label(k)}: {v}")
                else:
                    self.outputs[k] = v
            self.panel.status.setText("모델별 채보 완료" + ("\n" + "\n".join(errs) if errs else ""))
            self._update_overlays()

        handle.finished.connect(done)
        return handle

    # --- 동작 -----------------------------------------------------------------------

    def on_action(self, key: str) -> None:
        w = self.w
        p = self.panel
        if key == "run":
            self.run_dialog()
        elif key == "transcribe":
            self.transcribe()
        elif key in ("set_a", "set_b"):
            k = p.selected_key()
            if k is None:
                w.statusBar().showMessage("목록에서 분리 결과를 먼저 고르세요.", 3000)
                return
            role = key[-1]
            combo = p.combo_a if role == "a" else p.combo_b
            combo.setCurrentIndex(max(0, combo.findData(k)))
        elif key == "reveal":
            e = self.entry(p.selected_key() or self.keys["a"])
            if e is not None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(e.path)))
        elif key == "delete":
            self.delete_selected()
        elif key == "listen_a":
            self.listen("a")
        elif key == "listen_b":
            self.listen("b")
        elif key == "toggle":
            self.toggle_ab()
        elif key == "goto_region":
            region = w.player.loop_region
            if region is None:
                w.statusBar().showMessage("반복 구간이 없습니다. 파형을 끌어 구간을 고르세요.", 4000)
                return
            w.seek(region[0])
            w.reveal_time(region[0])
        elif key in ("use_a", "use_b"):
            self.use_as_source(key[-1])
        elif key == "exit":
            self.set_active(False)

    def delete_selected(self) -> None:
        w = self.w
        e = self.entry(self.panel.selected_key())
        if e is None:
            w.statusBar().showMessage("지울 분리 결과를 목록에서 고르세요.", 3000)
            return
        if e.active:
            QMessageBox.information(w, "지울 수 없음", "지금 분석에 쓰는 분리 결과는 지울 수 없습니다. 다른 결과를 분석 소스로 먼저 고르세요.")
            return
        ans = QMessageBox.question(w, "분리 결과 지우기",
                                   f"{e.label} 분리 결과({_size(e.size_bytes)}, stems/models/{e.key}/)를 지웁니다. "
                                   "원본·다른 모델 결과·음표·가사는 그대로입니다. 다시 쓰려면 다시 분리해야 합니다. 지울까요?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return
        for role in ("a", "b"):
            if self.keys[role] == e.key:
                self.keys[role] = None
        for k in [k for k in self.audio if k[0] == e.key]:
            self.audio.pop(k)
        self.outputs.pop(e.key, None)
        self._load_tracks()  # 재생 트랙에서 먼저 내린다
        try:
            self.store().delete(e.key)
        except (OSError, ValueError) as ex:
            w._show_error("분리 결과를 지우지 못했습니다", str(ex))
        self._save_keys()
        self.refresh_entries()
        self._update_overlays()

    def use_as_source(self, role: str) -> None:
        """A 또는 B 분리 결과를 실제 분석 소스로. 바뀔 항목을 먼저 보여 주고, 사용자 편집은 수정 유지로 지킨다."""
        w = self.w
        key = self.keys[role]
        e = self.entry(key)
        if e is None:
            w.statusBar().showMessage(f"{role.upper()}에 고른 분리 결과가 없습니다.", 4000)
            return
        out = self.outputs.get(key)
        if out is None:
            QMessageBox.information(w, "모델별 채보 필요",
                                    "이 결과로 만든 채보가 아직 없습니다. [모델별 채보]를 먼저 실행하세요. "
                                    "(바뀔 항목을 미리 보여 주려면 새 채보 결과가 필요합니다.)")
            return
        pv = source_switch.preview(w.doc.project, out.result)
        dlg = SourceSwitchDialog(e.label, pv.lines(), w)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            w.session.activate_stems(key)
        except Exception as ex:  # noqa: BLE001
            w._show_error("분석 소스를 바꾸지 못했습니다", str(ex))
            return
        if e.model in ("htdemucs", "mel_roformer"):  # 다음 '보컬 분석 실행'이 이 결과를 그대로 쓰게 설정도 맞춘다
            from dataclasses import replace

            s = w.inspector.analysis.settings()
            if s.separation != e.model:
                w.inspector.analysis.set_settings(replace(s, separation=e.model))
        from amp.engine.merge import KEEP_EDITS

        plan = w.doc.apply_analysis(out.result, KEEP_EDITS, f"분석 소스 변경: {e.label}")
        w.session._record_run(out.result, f"source:{e.model}", out.timings, out.reused, out.resources)
        self.refresh_entries()
        msg = (f"분석 소스를 {e.label}로 바꿨습니다. 자동 음표 {len(plan.added)}개 반영, 사용자 편집 {len(plan.kept)}개 유지.")
        if pv.unlinked or pv.melisma_review:
            msg += (f" 음표 없는 음절 {len(pv.unlinked)}개·멜리스마 재확인 {len(pv.melisma_review)}개는 검토 탭에서 확인하세요.")
        w.statusBar().showMessage(msg, 12000)
        log.info(msg)

    def build_menu(self, analysis_menu) -> None:
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.act_mode)
        analysis_menu.addAction(self.act_run)
        analysis_menu.addAction(self.act_ab)

    def update_actions(self, has_audio: bool, is_open: bool, busy: bool) -> None:
        self.act_mode.setEnabled(is_open and has_audio)
        self.act_run.setEnabled(is_open and has_audio and not busy)
        self.act_ab.setEnabled(self.active)
        for k in ("run", "transcribe", "use_a", "use_b", "delete"):
            self.panel.buttons[k].setEnabled(not busy)

