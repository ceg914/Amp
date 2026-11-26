"""v0.7.0 '노래 분석 → 보컬 정밀' 탭. 안쪽에 '피치 모델' 등 묶음을 둔다 (명세 16.3: 탭이 많으면 안쪽 묶음).

피치 모델 (Feature A):
- 모드: 기본(pYIN) · 정밀(설치된 외부 모델 중 하나 + pYIN) · 모델 직접 선택 · 여러 모델 비교 · 선택 구간(반복 구간) 다시 분석
- 모델 상태는 모델 등록 정보로 보여 준다 (설치 안 됨이면 고를 수 없고 설치 방법을 적음). 등록 정보는 작업 스레드에서 읽는다
- 결과는 프로젝트 analysis/pitch_models/에 모델별로 따로 남고, 고른 결과를 피아노 롤에 겹쳐 그린다(편집되지 않는 그림)
- 두 결과 비교: 유성 판정 차이·센트 차이·옥타브 차이 구간·한 모델에서만 유성인 구간. 정답이 없으면 정확도라고 하지 않는다
- 음표·가사는 바꾸지 않는다
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.app.theme import tokens
from amp.engine import pitch_runs as PR

TAB_TITLE = "보컬 정밀"


def _secondary(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("role", "secondary")
    return lab


class PitchModelPage(QWidget):
    runRequested = Signal(bool)  # force
    overlayChanged = Signal()
    compareRequested = Signal()
    regionActivated = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(_secondary("외부 피치 모델로 보컬 피치를 따로 분석해 pYIN과 비교합니다. 결과는 모델마다 따로 저장하고, "
                                 "음표·가사에는 자동으로 쓰지 않습니다."))
        form = QFormLayout()
        self.source = QLabel("—")
        self.source.setWordWrap(True)
        form.addRow("분석할 소리", self.source)
        self.mode = QComboBox()
        for key, label in PR.MODES.items():
            self.mode.addItem(label, key)
        form.addRow("방식", self.mode)
        self.range_label = QLabel("")
        self.range_label.setWordWrap(True)
        form.addRow("구간", self.range_label)
        lay.addLayout(form)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.models: dict[str, QCheckBox] = {}
        self.model_state: dict[str, QLabel] = {}
        for r, mid in enumerate(PR.MODEL_IDS):
            cb = QCheckBox(PR.MODEL_LABELS[mid])
            st = _secondary("확인 중…")
            grid.addWidget(cb, r, 0)
            grid.addWidget(st, r, 1)
            self.models[mid], self.model_state[mid] = cb, st
        lay.addLayout(grid)
        row = QHBoxLayout()
        self.run = QPushButton("분석")
        self.rerun = QPushButton("다시 계산")
        self.rerun.setToolTip("같은 설정의 결과가 있어도 새로 계산해 따로 저장합니다 (이전 결과는 그대로).")
        for b in (self.run, self.rerun):
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.run.clicked.connect(lambda: self.runRequested.emit(False))
        self.rerun.clicked.connect(lambda: self.runRequested.emit(True))
        self.message = _secondary("")
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.message)

        title = QLabel("저장한 결과 (체크하면 피아노 롤에 겹쳐 그림)")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(["모델", "상태", "장치·시간", "입력"])
        self.runs.setRootIsDecorated(False)
        self.runs.itemChanged.connect(lambda *_a: self.overlayChanged.emit())
        lay.addWidget(self.runs, 2)

        title = QLabel("두 결과 비교")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        crow = QHBoxLayout()
        self.cmp_a, self.cmp_b = QComboBox(), QComboBox()
        self.compare = QPushButton("비교")
        self.compare.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        crow.addWidget(self.cmp_a, 1)
        crow.addWidget(self.cmp_b, 1)
        crow.addWidget(self.compare)
        lay.addLayout(crow)
        self.compare.clicked.connect(self.compareRequested.emit)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(150)
        lay.addWidget(self.summary)
        self.regions = QListWidget()
        self.regions.setToolTip("누르면 그 구간으로 이동해 반복 구간으로 정합니다.")
        self.regions.itemActivated.connect(self._region_clicked)
        self.regions.itemClicked.connect(self._region_clicked)
        lay.addWidget(self.regions, 1)

    def _region_clicked(self, item: QListWidgetItem) -> None:
        a, b = item.data(Qt.ItemDataRole.UserRole)
        self.regionActivated.emit(float(a), float(b))

    def checked_overlays(self) -> list[str]:
        out = []
        for i in range(self.runs.topLevelItemCount()):
            it = self.runs.topLevelItem(i)
            if it.checkState(0) == Qt.CheckState.Checked:
                out.append(str(it.data(0, Qt.ItemDataRole.UserRole)))
        return out


class VocalPrecisionPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.sections = QTabWidget()
        self.sections.setDocumentMode(True)
        self.pitch = PitchModelPage()
        self.sections.addTab(self.pitch, "피치 모델")
        from amp.app.voice_controller import MultiVoicePage

        self.voices = MultiVoicePage()
        self.sections.addTab(self.voices, "다중 보컬")
        from amp.app.acoustics_controller import AcousticPage

        self.acoustics = AcousticPage()
        self.sections.addTab(self.acoustics, "음향 분석")
        lay.addWidget(self.sections)


class VocalPrecisionController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        w = self.w = window
        self.panel = VocalPrecisionPanel()
        w.song_tabs.insertTab(1, self.panel, TAB_TITLE)
        self.page = p = self.panel.pitch
        self._layers: dict[str, dict] = {}
        from amp.app.voice_controller import VoiceController

        self.voices = VoiceController(w, self.panel.voices)
        from amp.app.acoustics_controller import AcousticsController

        self.acoustics = AcousticsController(w, self.panel.acoustics)
        self.statuses: list = []
        self.usable: set[str] = {"pyin"}
        self.runs: list[PR.PitchRun] = []
        self._scan_handle = None
        self._shown_once = False
        p.mode.currentIndexChanged.connect(lambda _i: self._apply_mode())
        p.runRequested.connect(self.run)
        p.overlayChanged.connect(self.update_overlays)
        p.compareRequested.connect(self.compare)
        p.regionActivated.connect(self._go_region)
        w.doc.projectReset.connect(self._on_project_reset)
        w.session.tracksChanged.connect(self.refresh)
        w.right_tabs.currentChanged.connect(lambda _i: self._refresh_if_shown())
        w.song_tabs.currentChanged.connect(lambda _i: self._refresh_if_shown())
        w.player.playbackChanged.connect(self._update_range)  # 반복 구간이 바뀌면 '구간' 표시를 맞춘다
        w.session.busyChanged.connect(lambda _b: self.update_actions())
        self._apply_mode()

    # --- 상태 -------------------------------------------------------------------

    def _on_project_reset(self) -> None:
        self.runs = []
        self.page.summary.clear()
        self.page.regions.clear()
        self.voices.reset()
        self.acoustics.reset()
        self.refresh()

    def set_layers(self, group: str, layers: dict) -> None:
        """피아노 롤 겹쳐 보기: 묶음(피치 모델 'pitch', 성부 'voices')마다 층을 바꾸고 합쳐 그린다."""
        self._layers[group] = dict(layers)
        merged = {}
        for g in self._layers.values():
            merged.update(g)
        self.w.view.roll.overlay_curves.set_layers(merged)

    def _refresh_if_shown(self) -> None:
        if self.w.side_panel() is self.panel:
            if not self._shown_once:
                self._shown_once = True
                self.scan_models()
            self.refresh()

    def scan_models(self) -> None:
        """모델 등록 정보를 작업 스레드에서 읽는다 (패키지 메타데이터를 읽느라 화면 스레드에서는 100ms를 넘을 수 있음)."""
        session = self.w.session

        def job(_progress, _token):
            return session.model_registry(save=False)

        self._scan_handle = handle = self.w.runner.submit("모델 등록 정보", job)
        handle.finished.connect(self._on_scanned)

    def _on_scanned(self, statuses) -> None:
        from amp.core.install_hints import script

        self.statuses = list(statuses or [])
        by = {s.model_id: s for s in self.statuses}
        self.usable = {m for m in PR.MODEL_IDS if m in by and by[m].usable}
        for mid, cb in self.page.models.items():
            st = by.get(mid)
            ok = mid in self.usable
            cb.setEnabled(ok)
            if not ok:
                cb.setChecked(False)
            if st is None:
                text = "알 수 없음"
            elif ok:
                text = f"{st.state_label} · {st.version or ''}".strip(" ·")
                if st.experimental:
                    text += " · 실험"
            else:
                text = f"{st.state_label}" + (f" — {script('pitch')}로 설치" if mid != "pyin" else "")
            self.page.model_state[mid].setText(text)
            self.page.model_state[mid].setToolTip(st.reason if st else "")
        self._apply_mode()

    def _apply_mode(self) -> None:
        p = self.page
        mode = p.mode.currentData()
        pick = mode in ("select", "compare", "region")
        for mid, cb in p.models.items():
            cb.setEnabled(pick and mid in self.usable)
        if mode == "basic":
            self._set_checked({"pyin"})
        elif mode == "precise":
            self._set_checked(set(PR.precise_models(self.usable)))
        elif mode == "compare" and not any(cb.isChecked() for cb in p.models.values()):
            self._set_checked(set(m for m in PR.MODEL_IDS if m in self.usable))
        self._update_range()
        self.update_actions()

    def _set_checked(self, ids: set[str]) -> None:
        for mid, cb in self.page.models.items():
            cb.setChecked(mid in ids)

    def selected_models(self) -> tuple[str, ...]:
        return tuple(m for m in PR.MODEL_IDS if self.page.models[m].isChecked() and m in self.usable)

    def analysis_range(self):
        if self.page.mode.currentData() != "region":
            return None
        r = self.w.player.loop_region
        return (float(r[0]), float(r[1])) if r else None

    def _update_range(self) -> None:
        p = self.page
        if p.mode.currentData() != "region":
            p.range_label.setText("곡 전체")
            return
        r = self.w.player.loop_region
        p.range_label.setText(f"반복 구간 {r[0]:.2f}–{r[1]:.2f}s (앞뒤 {PR.REGION_PAD_S:.0f}초 문맥 포함)" if r
                              else "반복 구간을 먼저 정하세요 (파형·눈금자에서 끌기)")
        self.update_actions()

    def update_actions(self) -> None:
        s = self.w.session
        ok = s.folder is not None and s.audio is not None and not s.busy and bool(self.selected_models())
        if self.page.mode.currentData() == "region" and self.analysis_range() is None:
            ok = False
        self.page.run.setEnabled(ok)
        self.page.rerun.setEnabled(ok)
        self.page.compare.setEnabled(self.page.cmp_a.count() >= 2)

    def refresh(self) -> None:
        s = self.w.session
        src = s.pitch_source()
        self.page.source.setText(src["label"] if src else "프로젝트·오디오 없음")
        folder = s.folder
        self.runs = PR.PitchRunStore(folder.paths.analysis).runs() if folder is not None else []
        self._fill_runs()
        self._update_range()
        self.voices.refresh()
        self.acoustics.refresh()

    def _fill_runs(self) -> None:
        from amp.core.result_meta import STATUS_LABELS

        p = self.page
        keep = set(p.checked_overlays())
        prev_a, prev_b = p.cmp_a.currentData(), p.cmp_b.currentData()
        p.runs.blockSignals(True)
        p.runs.clear()
        for r in reversed(self.runs):
            dev = str(r.info.get("device", "")).upper()
            el = r.info.get("elapsed_s")
            cached = " (캐시)" if r.info.get("from_cache") else ""
            it = QTreeWidgetItem([r.label, STATUS_LABELS.get(r.meta.status, r.meta.status),
                                  f"{dev} {el:.1f}s{cached}" if el is not None else dev,
                                  str(r.info.get("input_label", r.meta.input_kind))])
            it.setData(0, Qt.ItemDataRole.UserRole, r.run_id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Checked if r.run_id in keep else Qt.CheckState.Unchecked)
            it.setToolTip(1, r.meta.status_reason or "")
            it.setForeground(0, _color(r.model_id))
            p.runs.addTopLevelItem(it)
        p.runs.blockSignals(False)
        for combo, prev in ((p.cmp_a, prev_a), (p.cmp_b, prev_b)):
            combo.blockSignals(True)
            combo.clear()
            for r in reversed(self.runs):
                combo.addItem(f"{r.label} · {r.meta.created_at[5:16].replace('T', ' ')}", r.run_id)
            i = combo.findData(prev)
            combo.setCurrentIndex(i if i >= 0 else (1 if combo is p.cmp_b and combo.count() > 1 else 0))
            combo.blockSignals(False)
        self.update_overlays()
        self.update_actions()

    # --- 실행 ---------------------------------------------------------------------

    def run(self, force: bool = False) -> None:
        w = self.w
        models = self.selected_models()
        if not models:
            return
        if w.session.busy:
            w.statusBar().showMessage("다른 분석이 끝난 뒤에 실행하세요.", 5000)
            return
        dev = w.inspector.analysis.settings().device
        handle = w.session.run_pitch_models(models, self.analysis_range(), force, dev)
        if handle is None:
            return
        self.page.message.setText(f"분석 중… ({', '.join(PR.MODEL_LABELS[m] for m in models)})")
        handle.finished.connect(self._on_done)
        handle.failed.connect(lambda msg, _tb: self.page.message.setText(f"분석하지 못했습니다: {msg}"))
        handle.cancelled.connect(lambda: self.page.message.setText("분석을 취소했습니다. 끝난 모델의 결과는 남겼습니다."))
        handle.ended.connect(self.refresh)
        self.update_actions()

    def _on_done(self, out) -> None:
        parts = []
        new = [r for r in out.runs if r.run_id not in out.reused]
        if new:
            parts.append("저장: " + ", ".join(f"{r.label} {r.info.get('elapsed_s', 0):.1f}s ({str(r.info.get('device', '')).upper()})"
                                             for r in new))
        if out.reused:
            parts.append(f"이미 있는 결과 사용 {len(out.reused)}개")
        for mid, msg in out.failed.items():
            parts.append(f"{PR.MODEL_LABELS.get(mid, mid)} 실패: {msg}")
        warns = [x for r in new for x in r.info.get("warnings", [])]
        parts.extend(warns)
        if out.cancelled:
            parts.append("취소: 끝난 모델의 결과만 남겼습니다.")
        text = " · ".join(parts) or "결과 없음"
        self.page.message.setText(text)
        self.w.statusBar().showMessage(text, 8000)
        self.refresh()
        ids = [r.run_id for r in out.runs]  # 방금 결과를 겹쳐 보기에 켠다
        self._check_runs(ids)

    def _check_runs(self, ids) -> None:
        p = self.page
        p.runs.blockSignals(True)
        for i in range(p.runs.topLevelItemCount()):
            it = p.runs.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) in ids:
                it.setCheckState(0, Qt.CheckState.Checked)
        p.runs.blockSignals(False)
        self.update_overlays()

    # --- 겹쳐 보기·비교 ------------------------------------------------------------

    def update_overlays(self) -> None:
        ids = set(self.page.checked_overlays())
        layers = {}
        for r in self.runs:
            if r.run_id in ids:
                with np.errstate(divide="ignore", invalid="ignore"):
                    midi = np.where(r.voiced, 69.0 + 12.0 * np.log2(r.f0_hz / 440.0), np.nan)
                layers[r.run_id] = (r.start_frame, r.frame_rate, midi, _color(r.model_id))
        self.set_layers("pitch", layers)

    def run_by_id(self, run_id) -> PR.PitchRun | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    def compare(self) -> None:
        from amp.compare import pitch_compare as PC

        p = self.page
        a, b = self.run_by_id(p.cmp_a.currentData()), self.run_by_id(p.cmp_b.currentData())
        if a is None or b is None:
            return
        if a.run_id == b.run_id:
            p.summary.setPlainText("서로 다른 두 결과를 고르세요.")
            return
        d = PC.compare(a, b)
        res = [f"{(a if k == a.run_id else b).label}: " + ", ".join(f"{n} {v}" for n, v in v.items())
               for k, v in d.resources.items() if v]
        p.summary.setPlainText("\n".join(d.summary_lines() + (["처리: " + " / ".join(res)] if res else [])))
        p.regions.clear()
        for kind, regs in (("옥타브 차이", d.octave_regions), (f"{a.label}만 유성", d.only_a_regions),
                           (f"{b.label}만 유성", d.only_b_regions)):
            for reg in regs:
                extra = f" ({reg[2]:+.0f}센트)" if len(reg) > 2 else ""
                it = QListWidgetItem(f"{kind} {reg[0]:.2f}–{reg[1]:.2f}s{extra}")
                it.setData(Qt.ItemDataRole.UserRole, (reg[0], reg[1]))
                p.regions.addItem(it)
        self._check_runs([a.run_id, b.run_id])
        self.last_diff = d

    def _go_region(self, a: float, b: float) -> None:
        w = self.w
        w.listening.set_region(a, b, pad=True)
        w.seek(max(0.0, a - w.listening.padding))


def _color(model_id: str):
    from PySide6.QtGui import QColor

    return QColor(tokens.PITCH_MODEL_COLORS.get(model_id, tokens.TEXT_2))
