"""Hybrid 보컬 편집 (v0.6): 구간마다 분리 모델을 골라 하나의 보컬 트랙으로 합치고, 전후를 들어 본 뒤 분석 소스로 쓴다.

- 구간 계획은 프로젝트(analysis/hybrid.json)에 저장하고 편집은 실행 취소할 수 있다(문서의 set_hybrid).
- [Hybrid 트랙 만들기]는 작업 스레드에서 합친다. 원본 stem은 읽기만 한다. 같은 계획이면 다시 만들지 않는다.
- [전후 비교 듣기]는 모델 비교 모드의 A/B에 지금 분석 소스(A)와 새 Hybrid(B)를 올린다.
- [분석 소스로 사용]은 Hybrid로 채보한 뒤(바뀌지 않은 피치 청크는 다시 계산하지 않음) 바뀔 항목을 보여 주고,
  확인하면 사용자 편집을 지키는 방식(수정 유지)으로 반영한다. 음표·가사를 묻지 않고 바꾸지 않는다.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.app.widgets import shrinkable_combos
from amp.app.widgets.hybrid_lane import SHORT_LABELS, HybridLane
from amp.core.errors import AmpError
from amp.engine import hybrid as H

log = logging.getLogger(__name__)


class HybridPanel(QWidget):
    """왼쪽 'Hybrid 구간' 탭: 소스 상태, 구간 목록, 고른 구간 편집, 만들기·듣기·적용."""

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
        title = QLabel("Hybrid 보컬")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        hint = QLabel("구간마다 더 잘 들리는 분리 모델을 골라 하나의 보컬로 합칩니다. 원본 분리 파일은 바꾸지 않습니다. "
                      "어느 쪽이 나은지는 직접 들어 보고 판단하세요.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        self.source = QLabel()
        self.source.setWordWrap(True)
        lay.addWidget(self.source)
        self.stale = QLabel()
        self.stale.setWordWrap(True)
        self.stale.setProperty("role", "warning")
        lay.addWidget(self.stale)
        self.buttons: dict[str, QPushButton] = {}
        lay.addWidget(self._button("start", "현재 결과로 시작", "지금 분석에 쓰는 분리 결과 하나로 곡 전체 구간을 만듭니다"))

        lay.addWidget(self._section("구간"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("시작", "끝", "모델", "음량", "교차", "상태"))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c in range(6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(120)
        lay.addWidget(self.table)
        g = QGridLayout()
        for i, (k, t, tip) in enumerate((
                ("split", "재생 위치에서 나누기", "고른 구간을 재생선 위치에서 둘로 나눕니다"),
                ("merge", "고른 구간 합치기", "이웃한 구간들을 하나로 (앞 구간의 모델)"),
                ("delete", "구간 지우기", "구간을 지우고 그 시간을 앞 구간에 붙입니다"),
                ("from_loop", "반복 구간 → 새 구간", "파형에서 고른 반복 구간을 새 구간으로 만듭니다"),
                ("loop", "구간 반복 듣기", "고른 구간을 반복 재생합니다"),
                ("set_all", "전체를 이 모델로", "모든 구간의 모델을 아래에서 고른 모델로 바꿉니다 (경계는 남김)"))):
            g.addWidget(self._button(k, t, tip), i // 2, i % 2)
        lay.addLayout(g)

        lay.addWidget(self._section("고른 구간"))
        form = QGridLayout()
        self.model = QComboBox()
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for sp in (self.start, self.end):
            sp.setDecimals(3)
            sp.setSingleStep(0.01)
            sp.setSuffix(" s")
            sp.setKeyboardTracking(False)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(-H.MAX_GAIN_DB, H.MAX_GAIN_DB)
        self.gain.setSingleStep(0.5)
        self.gain.setDecimals(1)
        self.gain.setSuffix(" dB")
        self.gain.setKeyboardTracking(False)
        self.crossfade = QSpinBox()
        self.crossfade.setRange(0, int(H.MAX_CROSSFADE_S * 1000))
        self.crossfade.setSuffix(" ms")
        self.crossfade.setKeyboardTracking(False)
        self.crossfade.setToolTip("이 구간 시작 경계의 교차(크로스페이드) 길이. 이웃 구간 길이의 절반을 넘지 않게 줄여 씁니다")
        for r, (label, wdg) in enumerate((("모델", self.model), ("시작", self.start), ("끝", self.end),
                                          ("음량 조정", self.gain), ("시작 경계 교차", self.crossfade))):
            form.addWidget(QLabel(label), r, 0)
            form.addWidget(wdg, r, 1)
        lay.addLayout(form)
        self.level_match = QCheckBox("모델 간 음량 맞춤 (큰 쪽을 작은 쪽에 맞춤)")
        self.level_match.setToolTip("모델마다 보컬 크기가 다르면 경계에서 크기가 튈 수 있습니다. 측정 음량(무음 블록 제외 RMS)으로 맞춥니다")
        lay.addWidget(self.level_match)
        cf = QHBoxLayout()
        cf.addWidget(QLabel("모든 경계 교차"))
        self.crossfade_all = QSpinBox()
        self.crossfade_all.setRange(0, int(H.MAX_CROSSFADE_S * 1000))
        self.crossfade_all.setSuffix(" ms")
        self.crossfade_all.setValue(int(H.DEFAULT_CROSSFADE_S * 1000))
        cf.addWidget(self.crossfade_all)
        cf.addWidget(self._button("crossfade_all", "적용"))
        lay.addLayout(cf)

        lay.addWidget(self._section("만들기·듣기·적용"))
        self.changed = QLabel()
        self.changed.setWordWrap(True)
        self.changed.setProperty("role", "secondary")
        lay.addWidget(self.changed)
        g2 = QGridLayout()
        for i, (k, t, tip) in enumerate((
                ("render", "Hybrid 트랙 만들기", "계획대로 보컬·반주를 합쳐 새 결과로 보관합니다"),
                ("compare", "전후 비교 듣기", "모델 비교의 A에 지금 분석 소스, B에 새 Hybrid를 올립니다 (Ctrl+B로 전환)"),
                ("use", "분석 소스로 사용", "Hybrid로 채보해 바뀔 항목을 보여 주고, 확인하면 반영합니다"),
                ("clear", "계획 지우기", "구간 계획만 지웁니다. 만든 Hybrid 결과·분석은 그대로입니다"))):
            g2.addWidget(self._button(k, t, tip), i // 2, i % 2)
        lay.addLayout(g2)
        self.result = QLabel()
        self.result.setWordWrap(True)
        self.result.setProperty("role", "secondary")
        lay.addWidget(self.result)
        lay.addStretch(1)
        shrinkable_combos(self)

    def _section(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "section")
        return lab

    def _button(self, key: str, text: str, tip: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda _c=False, k=key: self.actionRequested.emit(k))
        self.buttons[key] = b
        return b

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        out = []
        for r in rows:
            it = self.table.item(r, 0)
            if it is not None:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out


class HybridController(QObject):
    def __init__(self, window, compare) -> None:
        super().__init__(window)
        self.w = window
        self.compare = compare
        self.panel = HybridPanel()
        self.lane = HybridLane(window.timeline)
        self.lane_label = QLabel("Hybrid")
        self.active = False
        self.status: dict[str, H.SegmentStatus] = {}
        self._entries = []
        self._syncing = False
        self.last_render: H.RenderResult | None = None
        self.act_mode = QAction("Hybrid 편집 모드", window)
        self.act_mode.setCheckable(True)
        self.act_mode.setShortcut(QKeySequence("Ctrl+Shift+H"))
        self.act_mode.setToolTip("구간마다 분리 모델을 골라 하나의 보컬로 합칩니다 (Ctrl+Shift+H)")
        self.act_mode.toggled.connect(self.set_active)
        p = self.panel
        p.actionRequested.connect(self.on_action)
        p.table.itemSelectionChanged.connect(self._on_table_selection)
        p.model.activated.connect(lambda _i: self._apply_model())
        p.start.editingFinished.connect(lambda: self._apply_time("start"))
        p.end.editingFinished.connect(lambda: self._apply_time("end"))
        p.gain.editingFinished.connect(self._apply_gain)
        p.crossfade.editingFinished.connect(self._apply_crossfade)
        p.level_match.toggled.connect(self._apply_level_match)
        lane = self.lane
        lane.segmentsSelected.connect(self._on_lane_selection)
        lane.boundaryMoved.connect(self._on_boundary_moved)
        lane.loopRequested.connect(self._loop)
        lane.actionRequested.connect(self._on_lane_action)
        lane.sourceRequested.connect(lambda ids, key: self.set_model(ids, key))
        window.doc.hybridChanged.connect(self.refresh)
        window.doc.projectReset.connect(self._on_project_reset)
        window.session.audioReady.connect(lambda *_: self.refresh())

    # --- 모드 -------------------------------------------------------------------------

    @property
    def plan(self) -> H.HybridPlan | None:
        return self.w.doc.project.hybrid

    def set_active(self, on: bool) -> None:
        w = self.w
        if on and w.session.folder is None:
            w.statusBar().showMessage("프로젝트를 먼저 여세요.", 4000)
            self._set_checked(False)
            return
        if on == self.active:
            return
        self.active = on
        self._set_checked(on)
        tabs = self.compare.left_tabs
        if on:
            if not self.compare.active:
                self.compare.set_active(True)
            if tabs.indexOf(self.panel) < 0:
                tabs.addTab(self.panel, "Hybrid 구간")
            tabs.setCurrentWidget(self.panel)
            self.refresh()
            w.statusBar().showMessage("Hybrid 편집 모드: 레인의 경계를 끌거나 오른쪽 클릭으로 나누기·모델 지정", 6000)
        else:
            i = tabs.indexOf(self.panel)
            if i >= 0:
                tabs.removeTab(i)
        self._update_lane_visible()
        w._update_actions()

    def _set_checked(self, on: bool) -> None:
        if self.act_mode.isChecked() != on:
            self.act_mode.blockSignals(True)
            self.act_mode.setChecked(on)
            self.act_mode.blockSignals(False)

    def on_compare_closed(self) -> None:
        """모델 비교 모드를 끄면 Hybrid 편집 모드도 끈다 (왼쪽 탭이 사라지므로)."""
        if self.active:
            self.set_active(False)

    def _update_lane_visible(self) -> None:
        on = self.active or self.plan is not None
        self.lane.setVisible(on)
        self.lane_label.setVisible(on)

    def _on_project_reset(self) -> None:
        self.last_render = None
        if self.active and self.w.session.folder is None:
            self.set_active(False)
        self.refresh()

    # --- 상태 -------------------------------------------------------------------------

    def store(self):
        return self.w.session.model_store()

    def entry(self, key: str | None):
        return next((e for e in self._entries if e.key == key), None)

    def model_entries(self):
        return [e for e in self._entries if e.model != H.HYBRID]

    def active_entry(self):
        return next((e for e in self._entries if e.active), None)

    def active_plan(self) -> H.HybridPlan | None:
        """지금 분석에 쓰는 결과를 구간 계획으로 (Hybrid면 그 구간, 모델 결과면 곡 전체 한 구간)."""
        e = self.active_entry()
        if e is None:
            return None
        info = e.manifest.model_info
        if e.model == H.HYBRID and info.get("segments"):
            try:
                return H.plan_from_dict({"sample_rate": e.manifest.sample_rate, "frames": e.manifest.frames,
                                         "segments": info["segments"], "level_match": info.get("level_match", False)})
            except H.HybridError:
                return None
        return H.single(e.manifest.sample_rate, e.manifest.frames, e.key, e.model, e.manifest.stems_id)

    def refresh(self) -> None:
        store = self.store()
        try:
            self._entries = store.entries() if store is not None else []
        except OSError as e:
            log.warning("분리 결과 목록을 읽지 못함: %s", e)
            self._entries = []
        plan = self.plan
        try:
            current = self.w.session.current_store_keys()
        except Exception as e:  # noqa: BLE001 - 체크포인트 해시 실패 등: 상태만 덜 정확해진다
            log.warning("현재 모델 키를 구하지 못함: %s", e)
            current = {}
        self.status = H.check_sources(plan, {e.key: e for e in self._entries}, current) if plan is not None else {}
        labels = {e.key: e.label for e in self._entries}
        sources = [(e.key, f"{e.label} · {e.key[-10:]}") for e in self.model_entries()]
        changed: list = []
        ae = self.active_entry()
        if plan is not None and not (ae is not None and store is not None
                                     and H.plan_key(plan, store.source_sha256) == ae.key):
            ap = self.active_plan()
            changed = H.changed_ranges(ap, plan) if ap is not None else [(0.0, plan.duration)]
        self.lane.changed = changed
        self.lane.set_plan(plan, self.status, labels, sources)
        self._fill_panel(plan, labels, changed)
        self._update_lane_visible()

    def _fill_panel(self, plan, labels: dict, changed: list) -> None:
        p = self.panel
        self._syncing = True
        try:
            ae = self.active_entry()
            p.source.setText(f"지금 분석 소스: {ae.label}" if ae else "지금 분석 소스: 분리 결과 없음")
            p.stale.setText(self._stale_text(ae))
            p.stale.setVisible(bool(p.stale.text()))
            p.buttons["start"].setVisible(plan is None)
            sel = set(self.lane.selected)
            t = p.table
            t.setRowCount(0 if plan is None else len(plan.segments))
            if plan is not None:
                for r, s in enumerate(plan.segments):
                    a, b = plan.span(s)
                    st = self.status.get(s.segment_id)
                    vals = (f"{a:.3f}", f"{b:.3f}", labels.get(s.source_key, SHORT_LABELS.get(s.source_model, s.source_model)),
                            f"{s.gain_db:+.1f}dB", "—" if r == 0 else f"{s.crossfade_s * 1000:.0f}ms",
                            st.label if st else "")
                    for c, v in enumerate(vals):
                        cell = QTableWidgetItem(v)
                        if c == 0:
                            cell.setData(Qt.ItemDataRole.UserRole, s.segment_id)
                        if c == 5 and st is not None and not st.ok:
                            cell.setToolTip(st.reason)
                            cell.setText("! " + st.label)
                        t.setItem(r, c, cell)
                    if s.segment_id in sel:
                        for c in range(t.columnCount()):
                            t.item(r, c).setSelected(True)
                p.level_match.setChecked(plan.level_match)
            p.model.clear()
            for e in self.model_entries():
                p.model.addItem(f"{e.label} · {e.key[-10:]}", e.key)
            self._fill_editor()
            if plan is None:
                p.changed.setText("")
            elif changed:
                span = ", ".join(f"{a:.2f}–{b:.2f}s" for a, b in changed[:4]) + (" …" if len(changed) > 4 else "")
                p.changed.setText(f"지금 분석 소스와 달라지는 범위: {span}. 분석 소스로 쓰면 이 범위 근처의 피치만 다시 "
                                  "계산합니다 (나머지는 같은 신호라 이전 결과를 그대로 씀).")
            else:
                p.changed.setText("이 계획은 지금 분석 소스와 같습니다.")
            bad = [s for s in (plan.segments if plan else ()) if not self.status.get(s.segment_id, H.SegmentStatus("ok")).ok]
            for k in ("render", "compare", "use"):
                p.buttons[k].setEnabled(plan is not None and not bad and not self.w.session.busy)
            for k in ("split", "merge", "delete", "from_loop", "loop", "set_all", "clear", "crossfade_all"):
                p.buttons[k].setEnabled(plan is not None)
            if bad:
                p.result.setText(f"검증 필요 구간 {len(bad)}개: 다른 모델을 고르거나 들어 보고 다시 지정해야 만들 수 있습니다.")
        finally:
            self._syncing = False

    def _stale_text(self, active_entry) -> str:
        a = self.w.doc.project.analysis
        if a is None or active_entry is None or "separation" not in a.models:
            return ""
        used = a.models["separation"].params.get("store_key")
        if used and used != active_entry.key:
            return "현재 음표·피치는 다른 보컬 소스로 분석한 결과입니다. 다시 분석하면 지금 소스 기준으로 바뀝니다."
        return ""

    def _fill_editor(self) -> None:
        p = self.panel
        plan = self.plan
        ids = self.lane.selected
        enabled = plan is not None and bool(ids)
        for wdg in (p.model, p.start, p.end, p.gain, p.crossfade):
            wdg.setEnabled(enabled)
        if not enabled:
            return
        segs = [plan.get(i) for i in ids if i in {s.segment_id for s in plan.segments}]
        if not segs:
            return
        first, last = segs[0], segs[-1]
        for sp in (p.start, p.end):
            sp.blockSignals(True)
            sp.setRange(0.0, plan.duration)
        p.start.setValue(plan.time(first.start_frame))
        p.end.setValue(plan.time(last.end_frame))
        p.start.setEnabled(len(segs) == 1 and plan.index(first.segment_id) > 0)
        p.end.setEnabled(len(segs) == 1 and plan.index(last.segment_id) < len(plan.segments) - 1)
        for sp in (p.start, p.end):
            sp.blockSignals(False)
        p.model.setCurrentIndex(max(0, p.model.findData(first.source_key)))
        p.gain.blockSignals(True)
        p.gain.setValue(first.gain_db)
        p.gain.blockSignals(False)
        p.crossfade.blockSignals(True)
        p.crossfade.setValue(int(round(first.crossfade_s * 1000)))
        p.crossfade.setEnabled(plan.index(first.segment_id) > 0)
        p.crossfade.blockSignals(False)

    # --- 선택 ---------------------------------------------------------------------------

    def _on_table_selection(self) -> None:
        if self._syncing:
            return
        ids = self.panel.selected_ids()
        self.lane.select(ids)
        self._fill_editor()

    def _on_lane_selection(self, ids: list) -> None:
        self._syncing = True
        try:
            t = self.panel.table
            t.clearSelection()
            for r in range(t.rowCount()):
                it = t.item(r, 0)
                if it is not None and it.data(Qt.ItemDataRole.UserRole) in ids:
                    for c in range(t.columnCount()):
                        t.item(r, c).setSelected(True)
        finally:
            self._syncing = False
        self._fill_editor()
        if ids and self.plan is not None:
            a, _b = self.plan.span(self.plan.get(ids[0]))
            self.w.reveal_time(a)

    # --- 편집 ---------------------------------------------------------------------------

    def _push(self, new: H.HybridPlan, text: str) -> bool:
        return self.w.doc.set_hybrid(new, text)

    def _try(self, fn, text: str) -> bool:
        try:
            new = fn()
        except (H.HybridError, AmpError) as e:
            self.w.statusBar().showMessage(str(e), 6000)
            return False
        return self._push(new, text)

    def _selected_or_at_playhead(self) -> list[str]:
        if self.lane.selected:
            return list(self.lane.selected)
        if self.plan is None:
            return []
        return [self.plan.at(self.w.player.position()).segment_id]

    def start_plan(self) -> None:
        if self.plan is not None:
            return
        self.refresh()
        e = self.active_entry()
        if e is not None and e.model == H.HYBRID:
            plan = self.active_plan()
        else:
            e = e if e is not None and e.model != H.HYBRID else next(iter(self.model_entries()), None)
            if e is None:
                QMessageBox.information(self.w, "분리 결과 없음",
                                        "Hybrid로 합칠 분리 결과가 없습니다. 보컬 분석을 실행하거나 모델 비교로 결과를 만드세요.")
                return
            plan = H.single(e.manifest.sample_rate, e.manifest.frames, e.key, e.model, e.manifest.stems_id)
        if plan is not None:
            self._push(plan, "Hybrid 구간 시작")

    def set_model(self, ids: list[str], key: str) -> None:
        e = self.entry(key)
        if e is None or self.plan is None or not ids:
            return
        self._try(lambda: H.set_source(self.plan, ids, key, e.model, e.manifest.stems_id),
                  f"Hybrid 구간 모델: {SHORT_LABELS.get(e.model, e.model)}")

    def _apply_model(self) -> None:
        if self._syncing:
            return
        key = self.panel.model.currentData()
        if key:
            self.set_model(list(self.lane.selected), key)

    def _apply_time(self, which: str) -> None:
        if self._syncing or self.plan is None or len(self.lane.selected) != 1:
            return
        i = self.plan.index(self.lane.selected[0])
        if which == "start":
            self._try(lambda: H.move_boundary(self.plan, i, self.panel.start.value()), "Hybrid 구간 시작 이동")
        else:
            self._try(lambda: H.move_boundary(self.plan, i + 1, self.panel.end.value()), "Hybrid 구간 끝 이동")

    def _apply_gain(self) -> None:
        if self._syncing or self.plan is None or not self.lane.selected:
            return
        self._try(lambda: H.set_gain(self.plan, list(self.lane.selected), self.panel.gain.value()), "Hybrid 구간 음량")

    def _apply_crossfade(self) -> None:
        if self._syncing or self.plan is None or not self.lane.selected:
            return
        self._try(lambda: H.set_crossfade(self.plan, list(self.lane.selected), self.panel.crossfade.value() / 1000),
                  "Hybrid 경계 교차")

    def _apply_level_match(self, on: bool) -> None:
        if self._syncing or self.plan is None or self.plan.level_match == on:
            return
        import dataclasses

        self._push(dataclasses.replace(self.plan, level_match=bool(on)), "Hybrid 음량 맞춤")

    def _on_boundary_moved(self, index: int, t: float) -> None:
        if self.plan is not None:
            self._try(lambda: H.move_boundary(self.plan, index, t), "Hybrid 경계 이동")

    def _on_lane_action(self, kind: str, ids: list, t: float) -> None:
        plan = self.plan
        if plan is None:
            return
        if kind == "split" and ids:
            self._try(lambda: H.split(plan, ids[0], t), "Hybrid 구간 나누기")
        elif kind == "merge":
            self._try(lambda: H.merge(plan, ids), "Hybrid 구간 합치기")
        elif kind == "merge_next" and ids:
            i = plan.index(ids[0])
            if i + 1 < len(plan.segments):
                self._try(lambda: H.merge(plan, [ids[0], plan.segments[i + 1].segment_id]), "Hybrid 구간 합치기")
        elif kind == "delete" and ids:
            self._try(lambda: H.delete(plan, ids[0]), "Hybrid 구간 지우기")

    def _loop(self, a: float, b: float) -> None:
        self.w.listening.loop_range(a, b)

    def on_action(self, key: str) -> None:
        w = self.w
        plan = self.plan
        if key == "start":
            self.start_plan()
            return
        if plan is None:
            w.statusBar().showMessage("먼저 [현재 결과로 시작]을 누르세요.", 4000)
            return
        if key == "split":
            t = w.player.position()
            sid = plan.at(t).segment_id
            self._try(lambda: H.split(plan, sid, t), "Hybrid 구간 나누기")
        elif key == "merge":
            self._try(lambda: H.merge(plan, self._selected_or_at_playhead()), "Hybrid 구간 합치기")
        elif key == "delete":
            ids = self._selected_or_at_playhead()
            if ids:
                self._try(lambda: H.delete(plan, ids[0]), "Hybrid 구간 지우기")
        elif key == "from_loop":
            region = w.player.loop_region
            key_ = self.panel.model.currentData()
            e = self.entry(key_)
            if region is None or e is None:
                w.statusBar().showMessage("파형을 끌어 반복 구간을 고르고, 위에서 모델을 고르세요.", 5000)
                return

            def make():
                new, sid = H.create(plan, region[0], region[1], e.key, e.model, e.manifest.stems_id)
                self.lane.selected = [sid]
                return new

            self._try(make, "Hybrid 새 구간")
        elif key == "loop":
            ids = self._selected_or_at_playhead()
            if ids:
                self._loop(*plan.span(plan.get(ids[0])))
        elif key == "set_all":
            e = self.entry(self.panel.model.currentData())
            if e is not None:
                self._try(lambda: H.set_all(plan, e.key, e.model, e.manifest.stems_id),
                          f"Hybrid 전체 모델: {SHORT_LABELS.get(e.model, e.model)}")
        elif key == "crossfade_all":
            self._try(lambda: H.set_crossfade(plan, None, self.panel.crossfade_all.value() / 1000), "Hybrid 모든 경계 교차")
        elif key == "clear":
            ans = QMessageBox.question(w, "Hybrid 계획 지우기",
                                       "구간 계획만 지웁니다. 만든 Hybrid 결과 파일, 분석 결과, 음표·가사는 그대로입니다. "
                                       "실행 취소(Ctrl+Z)로 되돌릴 수 있습니다. 지울까요?")
            if ans == QMessageBox.StandardButton.Yes:
                self._push(None, "Hybrid 계획 지우기")
        elif key == "render":
            self.render()
        elif key == "compare":
            self.render(then=self._compare_rendered)
        elif key == "use":
            self.render(then=self._use_rendered)

    # --- 만들기·듣기·적용 ------------------------------------------------------------------

    def render(self, then=None) -> None:
        w = self.w
        plan = self.plan
        if plan is None:
            return
        handle = w.session.render_hybrid(plan, w.session.current_store_keys())
        if handle is None:
            w.statusBar().showMessage("다른 작업이 끝난 뒤 다시 시도하세요.", 4000)
            return

        def done(res: H.RenderResult) -> None:
            self.last_render = res
            info = res.entry.manifest.model_info
            txt = f"{'같은 구성의 결과를 다시 씀' if res.reused else f'만듦 ({res.seconds:.1f}초)'}: stems/models/{res.entry.key}/"
            if info.get("peak_scale_db"):
                txt += " · 잘림 방지로 전체 " + ", ".join(f"{k} {v:+.1f}dB" for k, v in info["peak_scale_db"].items())
            self.panel.result.setText(txt)
            self.compare.refresh_entries()
            self.refresh()
            w.statusBar().showMessage(f"Hybrid 트랙: {txt}", 8000)
            if then is not None:  # 이 슬롯 안에서는 세션이 아직 바쁨 상태다: 작업이 끝난 뒤에 이어 간다
                QTimer.singleShot(0, lambda k=res.entry.key: then(k))

        handle.finished.connect(done)

    def _compare_rendered(self, key: str) -> None:
        a = self.active_entry()
        if not self.compare.active:
            self.compare.set_active(True)
        self.compare.set_pair(a.key if a is not None and a.key != key else None, key)
        self.compare.listen("b")
        self.w.statusBar().showMessage("A = 지금 분석 소스, B = 새 Hybrid. Ctrl+B로 전환해 들어 보세요.", 8000)

    def _use_rendered(self, key: str) -> None:
        w = self.w
        e = next((x for x in (self.store().entries() if self.store() else ()) if x.key == key), None)
        if e is not None and e.active and self._stale_text(e) == "":
            w.statusBar().showMessage("이미 이 Hybrid 결과로 분석했습니다.", 5000)
            return
        settings = w.inspector.analysis.settings()
        handle = w.session.transcribe_entries(settings, [key])
        if handle is None:
            w.statusBar().showMessage("다른 작업이 끝난 뒤 다시 시도하세요.", 4000)
            return

        def done(outs: dict) -> None:
            out = outs.get(key)
            if isinstance(out, str) or out is None:
                w._show_error("Hybrid 채보에 실패했습니다", str(out))
                return
            chunks = out.result.models.get("pitch").params.get("chunks") if "pitch" in out.result.models else None
            lines = []
            if chunks:
                lines.append(f"피치: 청크 {chunks['computed']}개 새로 계산, {chunks['reused']}개는 같은 신호라 이전 결과 사용")
            if self.compare.apply_source(key, out, lines):
                self.refresh()

        handle.finished.connect(done)

    # --- 메뉴 -------------------------------------------------------------------------------

    def build_menu(self, analysis_menu) -> None:
        analysis_menu.addAction(self.act_mode)

    def update_actions(self, has_audio: bool, is_open: bool, busy: bool) -> None:
        self.act_mode.setEnabled(is_open and has_audio)
        plan = self.plan
        bad = any(not st.ok for st in self.status.values())
        for k in ("render", "compare", "use"):
            self.panel.buttons[k].setEnabled(plan is not None and not bad and not busy)
