"""우측 속성·분석 설정 패널 (명세 6.5).

- 선택 음표: 음 이름·MIDI·주파수(음표 기준 / 실제 곡선 중앙값)·시작·끝·길이·박 위치·신뢰도·상태·비브라토
- 분석 설정: 민감도 프리셋, 최소 음표 길이, 음높이 변화 기준, 음역, 보컬 분리, 처리 장치, 캐시 사용
- 실행 버튼: 보컬 분석 실행, 음표만 다시 검출, 보컬 분리 다시 실행 / 캐시 크기·비우기 / GPU 확인
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from amp.app import theme
from amp.core.model import Note, Project
from amp.core.pitch import midi_to_hz, note_name
from amp.engine.settings import (
    DEVICE_LABELS,
    DEVICES,
    SENSITIVITIES,
    SENSITIVITY_LABELS,
    SEPARATION_LABELS,
    SEPARATIONS,
    AnalysisSettings,
)

STATUS_TEXT = {"original": "자동 분석 (원본 그대로)", "modified": "자동 분석 후 수정", "user": "사용자 추가"}


def _mono_label() -> QLabel:
    lab = QLabel("—")
    lab.setFont(theme.mono_font())
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


class NoteInfo(QGroupBox):
    FIELDS = (("name", "음계"), ("midi", "MIDI"), ("freq", "주파수"), ("measured", "실제 피치"), ("start", "시작"),
              ("end", "종료"), ("length", "길이"), ("beat", "박 위치"), ("conf", "신뢰도"), ("status", "구분"),
              ("vibrato", "비브라토"))

    def __init__(self, parent=None) -> None:
        super().__init__("선택 음표", parent)
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.values: dict[str, QLabel] = {}
        for key, label in self.FIELDS:
            v = _mono_label() if key not in ("status",) else QLabel("—")
            self.values[key] = v
            form.addRow(label, v)
        self.show_notes([], None)

    def show_notes(self, notes: list[Note], project: Project | None) -> None:
        v = self.values
        if not notes:
            for lab in v.values():
                lab.setText("—")
            v["status"].setText("음표를 선택하세요")
            return
        if len(notes) > 1:
            for lab in v.values():
                lab.setText("—")
            start = min(n.start for n in notes)
            end = max(n.end for n in notes)
            v["start"].setText(f"{start:.3f} s")
            v["end"].setText(f"{end:.3f} s")
            v["length"].setText(f"{(end - start) * 1000:.0f} ms")
            v["status"].setText(f"음표 {len(notes)}개 선택")
            return
        n = notes[0]
        v["name"].setText(note_name(n.pitch) + ("  (장식음)" if n.ornament else ""))
        v["midi"].setText(str(n.pitch))
        v["freq"].setText(f"{float(midi_to_hz(n.pitch)):.2f} Hz")
        v["measured"].setText(self._measured(n, project))
        v["start"].setText(f"{n.start:.3f} s")
        v["end"].setText(f"{n.end:.3f} s")
        v["length"].setText(f"{n.duration * 1000:.0f} ms")
        if project is not None:
            bar, beat = project.score.tempo.bar_beat(n.start)
            v["beat"].setText(f"{bar}마디 {beat:.2f}박")
            v["status"].setText(STATUS_TEXT.get(project.note_status(n), ""))
        v["conf"].setText(f"{n.confidence:.2f}" if n.confidence is not None else "—")
        v["vibrato"].setText(f"{n.vibrato.rate_hz:.1f} Hz ±{n.vibrato.extent_cents:.0f}c" if n.vibrato else "없음")

    @staticmethod
    def _measured(n: Note, project: Project | None) -> str:
        curve = project.analysis.pitch_curve if project and project.analysis else None
        if curve is None:
            return "곡선 없음"
        i0, i1 = curve.frame_range(n.start, n.end)
        f = curve.f0_hz[i0:i1]
        f = f[np.isfinite(f)]
        if f.size == 0:
            return "무성"
        hz = float(np.median(f))
        cents = 1200 * np.log2(hz / float(midi_to_hz(n.pitch)))
        return f"{hz:.2f} Hz ({cents:+.0f}c)"


class AnalysisPanel(QGroupBox):
    runRequested = Signal()
    redetectRequested = Signal()
    reseparateRequested = Signal()
    clearCacheRequested = Signal()
    probeRequested = Signal()
    noVocalEditRequested = Signal()
    settingsChanged = Signal(object)  # AnalysisSettings

    def __init__(self, parent=None) -> None:
        super().__init__("분석 설정", parent)
        self._updating = False
        self._no_vocal: tuple = ()  # 보컬 없는 구간 (프로젝트별). 패널에는 개수만 보이고 편집은 대화상자에서
        self.sensitivity = QComboBox()
        for s in SENSITIVITIES:
            self.sensitivity.addItem(SENSITIVITY_LABELS[s], s)
        self.min_note = QSpinBox()
        self.min_note.setRange(20, 2000)
        self.min_note.setSuffix(" ms")
        self.change = QDoubleSpinBox()
        self.change.setRange(0.3, 12.0)
        self.change.setSingleStep(0.1)
        self.change.setDecimals(1)
        self.change.setSuffix(" 반음")
        self.low = self._note_combo()
        self.high = self._note_combo()
        self.separation = QComboBox()
        for s in SEPARATIONS:
            self.separation.addItem(SEPARATION_LABELS[s], s)
        self.device = QComboBox()
        for d in DEVICES:
            self.device.addItem(DEVICE_LABELS[d], d)
        self.use_cache = QCheckBox("이전 분석 캐시 재사용")
        self.merge_approach = QCheckBox("음 시작의 끌어올림을 한 음으로 합치기")
        self.merge_approach.setToolTip(
            "목표 음보다 1~2반음 아래에서 미끄러져 올라가며 시작한 짧은 구간을 따로 음표로 만들지 않고 목표 음에 합칩니다.\n"
            "재발음이 있거나 한 높이에 0.12초 넘게 머문 음은 합치지 않습니다. 피치 곡선은 바뀌지 않습니다.\n"
            "끄면 STEP 3까지와 같은 음표가 나옵니다. 바꾼 뒤 '음표만 다시 검출'로 비교할 수 있습니다.")
        self.device_info = QLabel("GPU 상태를 확인하지 않았습니다.")
        self.device_info.setProperty("role", "secondary")
        self.device_info.setWordWrap(True)
        self.probe = QPushButton("GPU 확인")
        self.cache_info = QLabel("캐시 —")
        self.cache_info.setProperty("role", "secondary")
        self.clear_cache = QPushButton("캐시 비우기")
        self.no_vocal_info = QLabel("없음")
        self.no_vocal_info.setProperty("role", "secondary")
        self.no_vocal_edit = QPushButton("편집…")
        self.no_vocal_edit.setToolTip("보컬이 없는 구간(전주·간주 등)을 지정하면 그 구간에서는 음표를 만들지 않습니다.")
        self.run = QPushButton("보컬 분석 실행")
        self.run.setProperty("primary", True)
        self.redetect = QPushButton("음표만 다시 검출")
        self.reseparate = QPushButton("분리 다시 실행")
        self.redetect.setToolTip("보컬 분리·피치 추출 결과를 그대로 쓰고 음표 검출만 다시 합니다.")
        self.reseparate.setToolTip("기존 분리 결과를 쓰지 않고 htdemucs를 다시 실행합니다.")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("음표 분리 민감도", self.sensitivity)
        form.addRow("최소 음표 길이", self.min_note)
        form.addRow("음높이 변화 기준", self.change)
        rng = QHBoxLayout()
        rng.addWidget(self.low)
        rng.addWidget(QLabel("~"))
        rng.addWidget(self.high)
        form.addRow("분석 음역", rng)
        form.addRow("보컬 분리", self.separation)
        form.addRow("처리 장치", self.device)
        nv = QHBoxLayout()
        nv.addWidget(self.no_vocal_info, 1)
        nv.addWidget(self.no_vocal_edit)
        form.addRow("보컬 없는 구간", nv)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.merge_approach)
        lay.addWidget(self.use_cache)
        dev = QHBoxLayout()
        dev.addWidget(self.device_info, 1)
        dev.addWidget(self.probe)
        lay.addLayout(dev)
        cache = QHBoxLayout()
        cache.addWidget(self.cache_info, 1)
        cache.addWidget(self.clear_cache)
        lay.addLayout(cache)
        lay.addWidget(self.run)
        row = QHBoxLayout()
        row.addWidget(self.redetect)
        row.addWidget(self.reseparate)
        lay.addLayout(row)

        self.sensitivity.currentIndexChanged.connect(self._on_preset)
        for w in (self.min_note, self.change):
            w.valueChanged.connect(self._emit)
        for w in (self.low, self.high, self.separation, self.device):
            w.currentIndexChanged.connect(self._emit)
        self.use_cache.toggled.connect(self._emit)
        self.merge_approach.toggled.connect(self._emit)
        self.run.clicked.connect(self.runRequested)
        self.redetect.clicked.connect(self.redetectRequested)
        self.reseparate.clicked.connect(self.reseparateRequested)
        self.clear_cache.clicked.connect(self.clearCacheRequested)
        self.probe.clicked.connect(self.probeRequested)
        self.no_vocal_edit.clicked.connect(self.noVocalEditRequested)
        for combo in (self.sensitivity, self.low, self.high, self.separation, self.device):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
        self.set_settings(AnalysisSettings())

    @staticmethod
    def _note_combo() -> QComboBox:
        c = QComboBox()
        for m in range(24, 97):
            c.addItem(f"{note_name(m)} ({m})", m)
        return c

    def _on_preset(self) -> None:
        if self._updating:
            return
        s = AnalysisSettings.for_sensitivity(self.sensitivity.currentData(), self.settings())
        self.set_settings(s)
        self.settingsChanged.emit(s)

    def _emit(self, *_args) -> None:
        if not self._updating:
            try:
                self.settingsChanged.emit(self.settings())
            except Exception:  # noqa: BLE001 - 잘못된 음역 조합은 저장하지 않는다
                pass

    def settings(self) -> AnalysisSettings:
        low, high = self.low.currentData(), self.high.currentData()
        if low >= high:
            high = low + 1
        return AnalysisSettings(
            sensitivity=self.sensitivity.currentData(), min_note_ms=float(self.min_note.value()),
            pitch_change_semitones=float(self.change.value()), pitch_low=int(low), pitch_high=int(high),
            device=self.device.currentData(), use_cache=self.use_cache.isChecked(),
            separation=self.separation.currentData(), no_vocal_ranges=self._no_vocal,
            merge_approach=self.merge_approach.isChecked(),
        )

    def set_settings(self, s: AnalysisSettings) -> None:
        self._updating = True
        try:
            self.sensitivity.setCurrentIndex(max(0, self.sensitivity.findData(s.sensitivity)))
            self.min_note.setValue(int(s.min_note_ms))
            self.change.setValue(s.pitch_change_semitones)
            self.low.setCurrentIndex(max(0, self.low.findData(s.pitch_low)))
            self.high.setCurrentIndex(max(0, self.high.findData(s.pitch_high)))
            self.separation.setCurrentIndex(max(0, self.separation.findData(s.separation)))
            self.device.setCurrentIndex(max(0, self.device.findData(s.device)))
            self.use_cache.setChecked(s.use_cache)
            self.merge_approach.setChecked(s.merge_approach)
            self._set_no_vocal(s.no_vocal_ranges)
        finally:
            self._updating = False

    def _set_no_vocal(self, ranges) -> None:
        self._no_vocal = tuple(ranges)
        total = sum(b - a for a, b in self._no_vocal)
        self.no_vocal_info.setText(f"{len(self._no_vocal)}개 · 합계 {total:.1f}초" if self._no_vocal else "없음")

    def set_no_vocal_ranges(self, ranges) -> None:
        """구간을 바꾸고 settingsChanged를 보낸다."""
        self._set_no_vocal(ranges)
        self._emit()

    def set_cache_size(self, nbytes: int | None) -> None:
        self.cache_info.setText("캐시 —" if nbytes is None else f"캐시 {nbytes / 1e6:.1f} MB")


class Inspector(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(370)
        inner = QWidget()
        inner.setObjectName("panel")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        self.note_info = NoteInfo()
        self.analysis = AnalysisPanel()
        lay.addWidget(self.note_info)
        lay.addWidget(self.analysis)
        lay.addStretch(1)
        self.setWidget(inner)
