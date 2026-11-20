"""분석 오류 검토 패널과 컨트롤러 (STEP 5).

목록은 자동 분석 결과를 읽기만 해서 만든다. 음표 삭제·연결 변경·시간 수정은 하지 않는다.
항목을 고르면 그 시각으로 이동하고 관련 음표·음절을 강조한다. 검토 상태 표시는 실행 취소할 수 있고 프로젝트에 저장된다.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.app import theme
from amp.compare import review as R

COLUMNS = ("상태", "심각도", "유형", "시작", "끝", "대상", "사유")
STATUS_MARK = {"open": "●", "checked": "✓", "fixed": "✔", "ignored": "–", "recheck": "↻"}


def _fmt_t(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):d}:{s:05.2f}"


class ReviewPanel(QWidget):
    rowActivated = Signal(int)
    actionRequested = Signal(str)  # prev / next / loop / zoom / notes / lyrics / checked / fixed / ignored / open / refresh

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        hint = QLabel("자동 분석에서 의심스러운 곳입니다. 목록은 읽기만 하며 음표·가사를 바꾸지 않습니다. "
                      "항목을 고르면 그 위치로 이동합니다. 사유의 숫자는 판단 기준입니다(정확도 점수가 아님).")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        filt = QGridLayout()
        self.category = QComboBox()
        self.category.addItem("전체", "")
        for k, v in R.CATEGORIES.items():
            self.category.addItem(v, k)
        self.severity = QComboBox()
        self.severity.addItem("모든 심각도", "")
        for k, v in R.SEVERITIES.items():
            self.severity.addItem(v, k)
        self.unreviewed = QCheckBox("미검토만")
        filt.addWidget(self.category, 0, 0)
        filt.addWidget(self.severity, 0, 1)
        filt.addWidget(self.unreviewed, 0, 2)
        lay.addLayout(filt)
        self.summary = QLabel("—")
        self.summary.setProperty("role", "secondary")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        for c in range(len(COLUMNS) - 1):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.setWordWrap(False)
        self.table.itemSelectionChanged.connect(self._on_select)
        lay.addWidget(self.table, 1)
        nav = QGridLayout()
        self.buttons = {}
        for i, (key, text) in enumerate((("prev", "◀ 이전"), ("next", "다음 ▶"), ("loop", "구간 반복"), ("zoom", "구간 확대"),
                                         ("notes", "음표 편집"), ("lyrics", "가사 정렬"), ("checked", "확인 완료"),
                                         ("fixed", "수정 완료"), ("ignored", "무시"), ("open", "검토 필요로"),
                                         ("refresh", "다시 검사"))):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, k=key: self.actionRequested.emit(k))
            nav.addWidget(b, i // 4, i % 4)
            self.buttons[key] = b
        lay.addLayout(nav)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.detail)
        self._rows: list[tuple[R.ReviewItem, str]] = []

    def set_rows(self, rows: list[tuple[R.ReviewItem, str]], summary: str) -> None:
        keep = self.current_key()
        self._rows = rows
        self.summary.setText(summary)
        t = self.table
        t.blockSignals(True)
        # 채우는 동안 열 너비 '내용에 맞춤'을 끈다. 켜 둔 채 셀을 넣으면 셀마다 전체 열을 다시 재서
        # 실제 곡(검토 항목 약 630개, 셀 4,387개)에서 69초 동안 화면이 멈췄다. 다 채운 뒤 한 번만 맞춘다
        hh = t.horizontalHeader()
        modes = [hh.sectionResizeMode(c) for c in range(t.columnCount())]
        for c in range(t.columnCount()):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        t.setUpdatesEnabled(False)
        t.setRowCount(len(rows))
        mono = theme.mono_font()
        warn = theme.LYRIC_WARN
        for r, (it, status) in enumerate(rows):
            vals = (f"{STATUS_MARK[status]} {R.STATUSES[status]}", R.SEVERITIES[it.severity], it.label,
                    _fmt_t(it.start) if it.end > 0 else "—", _fmt_t(it.end) if it.end > 0 else "—",
                    _targets(it), it.reason)
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c in (3, 4):
                    cell.setFont(mono)
                if c == 1 and it.severity == "high":
                    cell.setForeground(warn)
                if c == 0 and status in ("open", "recheck"):
                    cell.setForeground(theme.ACCENT if status == "open" else warn)
                cell.setToolTip(it.reason)
                t.setItem(r, c, cell)
        for c, m in enumerate(modes):
            hh.setSectionResizeMode(c, m)
        t.setUpdatesEnabled(True)
        t.blockSignals(False)
        if keep is not None:
            for r, (it, _s) in enumerate(rows):
                if it.key == keep:
                    t.selectRow(r)
                    break

    def current_row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def current_key(self) -> str | None:
        r = self.current_row()
        return self._rows[r][0].key if 0 <= r < len(self._rows) else None

    def selected_items(self) -> list[R.ReviewItem]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()}) if self.table.selectionModel() else []
        return [self._rows[r][0] for r in rows if r < len(self._rows)]

    def select_row(self, r: int) -> None:
        if 0 <= r < self.table.rowCount():
            self.table.selectRow(r)
            self.table.scrollToItem(self.table.item(r, 0))

    def _on_select(self) -> None:
        r = self.current_row()
        if 0 <= r < len(self._rows):
            it, status = self._rows[r]
            self.detail.setText(f"<b>{it.label}</b> · {R.CATEGORIES[it.category]} · 심각도 {R.SEVERITIES[it.severity]} · "
                                f"{R.STATUSES[status]}<br>{it.reason}")
            self.rowActivated.emit(r)


def _targets(it: R.ReviewItem) -> str:
    parts = []
    if it.note_ids:
        parts.append(f"음표 {len(it.note_ids)}")
    if it.syllable_ids:
        parts.append(f"음절 {len(it.syllable_ids)}")
    return " · ".join(parts) or "구간"


class ReviewController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.panel = ReviewPanel()
        self.items: list[R.ReviewItem] = []
        self.extra_items: list[R.ReviewItem] = []  # 다른 기능(모델 비교 등)이 넣는 항목
        self.leak_items: list[R.ReviewItem] = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._seq = 0
        self.panel.rowActivated.connect(self._on_row)
        self.panel.actionRequested.connect(self.on_action)
        for w in (self.panel.category, self.panel.severity):
            w.currentIndexChanged.connect(self._show)
        self.panel.unreviewed.toggled.connect(self._show)
        doc = window.doc
        for sig in (doc.notesReplaced, doc.analysisChanged, doc.lyricsChanged, doc.projectReset):
            sig.connect(self.schedule)
        doc.reviewChanged.connect(self._show)

    def schedule(self, *_a) -> None:
        self._timer.start()

    # --- 계산 -----------------------------------------------------------------------

    def refresh(self) -> None:
        """규칙을 다시 돌린다 (작업 스레드). 반주 섞임 검사는 분리 결과가 있을 때만."""
        w = self.w
        doc = w.doc
        notes = list(doc.score.notes())
        analysis = doc.project.analysis
        lyrics = doc.lyrics
        groups = dict(getattr(w.vocal, "groups", {}) or {})
        settings = w.inspector.analysis.settings()
        ranges = tuple(settings.no_vocal_ranges)
        original = frozenset(n.id for n in analysis.notes) if analysis is not None else frozenset()
        split = float(settings.note_params().split_strength or 4.0)
        stems = dict(getattr(w.session, "raw_stems", {}) or {})
        self._seq += 1
        seq = self._seq

        def job(progress, token):
            items = R.note_rules(notes, analysis, ranges, split, original)
            items += R.lyric_rules(lyrics, {n.id: n for n in notes})
            items += R.vocal_rules(lyrics, groups)
            v, i = stems.get("vocals"), stems.get("instrumental")
            if v is not None and i is not None:
                items += R.separation_leak_rules(notes, v.mono(), i.mono(), v.sample_rate)
            return items

        handle = w.runner.submit("분석 오류 검토", job)

        def done(items) -> None:
            if seq != self._seq:
                return
            self.items = items
            self._show()

        handle.finished.connect(done)

    def all_items(self) -> list[R.ReviewItem]:
        return self.items + self.extra_items

    def set_extra_items(self, items: list[R.ReviewItem]) -> None:
        self.extra_items = list(items)
        self._show()

    def _filtered(self) -> list[tuple[R.ReviewItem, str]]:
        p = self.panel
        cat, sev, only = p.category.currentData(), p.severity.currentData(), p.unreviewed.isChecked()
        rows = R.merge_states(self.all_items(), self.w.doc.project.review)
        return [(it, st) for it, st in rows if (not cat or it.category == cat) and (not sev or it.severity == sev)
                and (not only or st in ("open", "recheck"))]

    def _show(self, *_a) -> None:
        rows = self._filtered()
        allrows = R.merge_states(self.all_items(), self.w.doc.project.review)
        n_open = sum(1 for _i, s in allrows if s in ("open", "recheck"))
        by_cat = {}
        for it, _s in allrows:
            by_cat[it.category] = by_cat.get(it.category, 0) + 1
        cats = " · ".join(f"{R.CATEGORIES[k]} {v}" for k, v in by_cat.items()) or "없음"
        self.panel.set_rows(rows, f"전체 {len(allrows)}개 (미검토 {n_open}) — {cats}. 표시 {len(rows)}개")

    # --- 이동·동작 ------------------------------------------------------------------

    def _on_row(self, r: int) -> None:
        rows = self.panel._rows
        if not (0 <= r < len(rows)):
            return
        it = rows[r][0]
        w = self.w
        if it.end > 0:
            w.reveal_time(it.start)
        ids = [n for n in it.note_ids if n in w.doc.score]
        if ids:
            w.view.roll.select_ids(ids)
        if it.syllable_ids:
            w.lyric_lane.select(list(it.syllable_ids), emit=False)

    def on_action(self, key: str) -> None:
        p = self.panel
        w = self.w
        r = p.current_row()
        if key in ("prev", "next"):
            p.select_row(max(0, r - 1) if key == "prev" else min(p.table.rowCount() - 1, r + 1))
            return
        if key == "refresh":
            self.refresh()
            return
        items = p.selected_items()
        if not items:
            w.statusBar().showMessage("검토 항목을 먼저 고르세요.", 3000)
            return
        it = items[0]
        if key in ("checked", "fixed", "ignored", "open"):
            w.doc.set_review_state(R.mark(w.doc.project.review, items, key), f"검토: {R.STATUSES[key]}")
            return
        if key == "loop" and it.end > it.start:
            w.listening.set_region(it.start - 0.3, it.end + 0.3, enable=True, play=True)
        elif key == "zoom" and it.end > it.start:
            span = max(it.end - it.start + 2.0, 1.0)
            target = w.view.viewport().width() / span
            w.view.zoom_time(target / w.view.mapper.px_per_sec)
            w.reveal_time(max(0.0, it.start - 1.0))
        elif key == "notes":
            ids = [n for n in it.note_ids if n in w.doc.score]
            if ids:
                w.view.roll.select_ids(ids)
            w.view.setFocus()
            w.right_tabs.setCurrentWidget(w.inspector)
        elif key == "lyrics":
            if it.syllable_ids:
                w.lyric_lane.select(list(it.syllable_ids))
            w.right_tabs.setCurrentWidget(w.lyrics_panel)
