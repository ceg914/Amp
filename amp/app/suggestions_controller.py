"""교정 제안 탭 (v0.6): 근거와 함께 보여 주고, 사용자가 고른 것만 적용한다.

- [다시 찾기]: 지금 음표·피치 곡선·가사 정렬로 제안을 새로 만든다. 무시·숨김·적용 상태는 이어받는다.
- 행을 고르면 그 시각으로 이동하고, 미리보기를 켜 두면 피아노 롤에 바뀔 모양을 점선으로 그린다(편집되지 않는 그림).
- [적용]은 고른 제안을 한 번에 적용하고 Ctrl+Z 한 번으로 되돌린다. [무시]는 다시 찾아도 나오지 않게, [숨김]은 목록에서만 뺀다.
- '근거 점수'는 규칙 값이다. 확률이 아니다.
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.compare import suggestions as S

log = logging.getLogger(__name__)


class SuggestionsPanel(QWidget):
    actionRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        self.counts = QLabel()
        self.counts.setWordWrap(True)
        top.addWidget(self.counts, 1)
        self.buttons: dict[str, QPushButton] = {}
        top.addWidget(self._button("detect", "다시 찾기", "지금 음표·피치 곡선·가사 정렬로 제안을 새로 만듭니다"))
        lay.addLayout(top)
        f = QHBoxLayout()
        self.type_filter = QComboBox()
        self.type_filter.addItem("모든 종류", "")
        for k, v in S.TYPES.items():
            self.type_filter.addItem(v, k)
        self.show_done = QCheckBox("처리한 제안도 보기")
        self.preview = QCheckBox("미리보기")
        self.preview.setChecked(True)
        self.preview.setToolTip("고른 제안이 바꿀 모양을 피아노 롤에 점선으로 그립니다 (편집되지 않음)")
        f.addWidget(self.type_filter, 1)
        f.addWidget(self.show_done)
        f.addWidget(self.preview)
        lay.addLayout(f)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("종류", "시각", "내용", "근거 점수", "상태"))
        self.table.horizontalHeaderItem(3).setToolTip(S.SCORE_NOTE)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c in range(5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        self.evidence = QLabel()
        self.evidence.setWordWrap(True)
        self.evidence.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.evidence.setMinimumHeight(60)
        lay.addWidget(self.evidence)
        g = QGridLayout()
        for i, (k, t, tip) in enumerate((
                ("prev", "◀ 이전", "앞 제안으로 (Ctrl+Alt+[)"), ("next", "다음 ▶", "다음 제안으로 (Ctrl+Alt+])"),
                ("apply", "적용", "고른 제안을 적용합니다. Ctrl+Z 한 번으로 모두 되돌립니다"),
                ("reject", "무시", "이 제안을 무시합니다 (다시 찾아도 나오지 않음)"),
                ("dismiss", "숨김", "목록에서만 숨깁니다"), ("restore", "대기로 되돌리기", "무시·숨김을 취소합니다"))):
            g.addWidget(self._button(k, t, tip), i // 2, i % 2)
        lay.addLayout(g)
        note = QLabel(S.SCORE_NOTE + " 사용자가 고친 음표는 제안하지 않고, 적용한 음표는 재분석이 덮어쓰지 않습니다.")
        note.setWordWrap(True)
        note.setProperty("role", "secondary")
        lay.addWidget(note)

    def _button(self, key: str, text: str, tip: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda _c=False, k=key: self.actionRequested.emit(k))
        self.buttons[key] = b
        return b

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows if self.table.item(r, 0)]


class SuggestionsController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.panel = SuggestionsPanel()
        self.rows: list[S.Suggestion] = []
        self.last_detect_ms = 0.0
        p = self.panel
        p.actionRequested.connect(self.on_action)
        p.table.itemSelectionChanged.connect(self._on_selection)
        p.type_filter.currentIndexChanged.connect(lambda _i: self.refresh())
        p.show_done.toggled.connect(lambda _on: self.refresh())
        p.preview.toggled.connect(lambda _on: self._update_preview())
        doc = window.doc
        doc.suggestionsChanged.connect(self.refresh)
        doc.projectReset.connect(self.refresh)
        doc.notesReplaced.connect(lambda *_: self.refresh())
        doc.analysisChanged.connect(self.refresh)

    @property
    def state(self) -> S.SuggestionState:
        return self.w.doc.project.suggestions or S.SuggestionState()

    def status_of(self, s: S.Suggestion) -> str:
        """표시할 상태: 대기 제안의 대상이 바뀌었으면 '지난 제안'."""
        if s.status == S.PENDING and S.is_outdated(s, self.w.doc.project):
            return S.OUTDATED
        return s.status

    # --- 찾기 -------------------------------------------------------------------------

    def detect(self) -> int:
        w = self.w
        if w.doc.project.analysis is None:
            w.statusBar().showMessage("분석 결과가 있어야 교정 제안을 찾을 수 있습니다.", 4000)
            return 0
        settings = w.inspector.analysis.settings()
        t0 = time.perf_counter()
        fresh = S.detect(w.doc.project, settings.note_params().split_strength or 4.0)
        self.last_detect_ms = (time.perf_counter() - t0) * 1000
        from datetime import datetime, timezone

        new = S.merge_detected(self.state, fresh, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        w.doc.set_suggestions(new, "교정 제안 찾기")
        n = sum(1 for s in new.items if s.status == S.PENDING)
        w.statusBar().showMessage(f"교정 제안 {n}개 ({self.last_detect_ms:.0f}ms). 음표는 바꾸지 않았습니다.", 6000)
        log.info("교정 제안 %d개, %.0fms", n, self.last_detect_ms)
        return n

    # --- 목록 -------------------------------------------------------------------------

    def refresh(self) -> None:
        p = self.panel
        kind = p.type_filter.currentData()
        show_done = p.show_done.isChecked()
        keep = set(p.selected_ids())
        rows = []
        counts: dict[str, int] = {}
        for s in self.state.items:
            st = self.status_of(s)
            counts[st] = counts.get(st, 0) + 1
            if kind and s.suggestion_type != kind:
                continue
            if not show_done and st != S.PENDING:
                continue
            rows.append((s, st))
        self.rows = [s for s, _ in rows]
        t = p.table
        t.blockSignals(True)
        t.setUpdatesEnabled(False)
        try:
            t.setRowCount(len(rows))
            for r, (s, st) in enumerate(rows):
                vals = (s.type_label, f"{s.start_time:.2f}", s.summary(), f"{s.score:.2f}", S.STATUS_LABELS.get(st, st))
                for c, v in enumerate(vals):
                    cell = QTableWidgetItem(v)
                    if c == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, s.suggestion_id)
                    if c == 2:
                        cell.setToolTip("\n".join(s.evidence))
                    t.setItem(r, c, cell)
                if s.suggestion_id in keep:
                    for c in range(t.columnCount()):
                        t.item(r, c).setSelected(True)
        finally:
            t.setUpdatesEnabled(True)
            t.blockSignals(False)
        parts = [f"{S.STATUS_LABELS[k]} {v}" for k, v in counts.items() if v]
        p.counts.setText("교정 제안: " + (" · ".join(parts) if parts else "없음 — [다시 찾기]를 누르세요"))
        self._on_selection()

    def _on_selection(self) -> None:
        ids = self.panel.selected_ids()
        s = self.state.get(ids[0]) if ids else None
        if s is None:
            self.panel.evidence.setText("")
        else:
            st = self.status_of(s)
            lines = [f"<b>{s.type_label}</b> {s.summary()} · 근거 점수 {s.score:.2f} (확률 아님) · "
                     f"{S.STATUS_LABELS.get(st, st)}"] + [f"• {e}" for e in s.evidence]
            if st == S.OUTDATED:
                lines.append("• 제안을 만든 뒤 대상 음표가 바뀌었습니다. 적용할 수 없습니다.")
            self.panel.evidence.setText("<br>".join(lines))
        self._update_preview()

    def _update_preview(self) -> None:
        roll = self.w.view.roll
        ids = self.panel.selected_ids()
        if not self.panel.preview.isChecked() or not ids:
            roll.set_preview()
            return
        removed, added = [], []
        for sid in ids[:20]:
            s = self.state.get(sid)
            if s is not None and self.status_of(s) == S.PENDING:
                r, a = S.preview_notes(self.w.doc.project, s)
                removed += r
                added += a
        roll.set_preview(removed, added)

    def go(self, step: int) -> None:
        t = self.panel.table
        if t.rowCount() == 0:
            return
        rows = sorted({i.row() for i in t.selectedIndexes()})
        r = (rows[0] + step) % t.rowCount() if rows else (0 if step > 0 else t.rowCount() - 1)
        t.clearSelection()
        t.selectRow(r)
        s = self.rows[r]
        self.w.seek(s.start_time)
        self.w.reveal_time(s.start_time)

    # --- 동작 -------------------------------------------------------------------------

    def on_action(self, key: str) -> None:
        w = self.w
        ids = self.panel.selected_ids()
        if key == "detect":
            self.detect()
        elif key == "prev":
            self.go(-1)
        elif key == "next":
            self.go(1)
        elif key == "apply":
            if not ids:
                w.statusBar().showMessage("적용할 제안을 고르세요.", 3000)
                return
            applied, skipped = w.doc.apply_suggestions(ids)
            msg = f"교정 제안 {applied}개 적용 (Ctrl+Z로 되돌림)."
            if skipped:
                msg += f" 적용하지 못함 {len(skipped)}개: " + "; ".join(skipped[:3])
            w.statusBar().showMessage(msg, 10000)
        elif key in ("reject", "dismiss", "restore"):
            if not ids:
                return
            status = {"reject": S.REJECTED, "dismiss": S.DISMISSED, "restore": S.PENDING}[key]
            text = {"reject": "교정 제안 무시", "dismiss": "교정 제안 숨김", "restore": "교정 제안 되돌리기"}[key]
            w.doc.set_suggestions(self.state.with_status(ids, status), text)
