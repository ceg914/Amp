"""Hybrid 구간 레인 (v0.6): 곡의 어느 구간이 어떤 분리 모델 결과를 쓰는지 시간 막대로 보여 주고 고친다.

파형·피아노 롤과 같은 ``TimelineState`` 를 쓴다. 계획은 컨트롤러가 넣어 주고, 레인은 편집 요청만 보낸다
(끌기 중에는 레인 안에서만 미리 보여 주고, 놓을 때 한 번만 요청하므로 실행 취소 기록이 한 단계다).

조작
- 클릭: 구간 선택 / Ctrl+클릭: 선택 추가·빼기 / Shift+클릭: 범위 선택
- 경계(세로선) 끌기: 두 구간의 경계 이동. 끄는 동안 지금 시각·처음 시각·차이(ms)를 보여 준다
- 더블클릭: 그 구간 반복 재생 / 오른쪽 클릭: 나누기·합치기·모델 지정·지우기 메뉴
- 휠: 가로 스크롤 / Ctrl+휠: 가로 확대

모양(색만으로 구분하지 않음): 구간 안에 모델 이름을 쓴다. 검증 필요 구간은 빗금 + 주황 점선 테두리 + '!'.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QMenu, QSizePolicy, QToolTip, QWidget

from amp.app import theme
from amp.app.timeline import TimelineState
from amp.engine import hybrid as H

EDGE_PX = 5.0
SHORT_LABELS = {"htdemucs": "Demucs", "mel_roformer": "RoFormer", "external": "외부"}


def model_color(model: str) -> QColor:
    return {"htdemucs": theme.MODEL_A, "mel_roformer": theme.MODEL_B}.get(model, theme.TEXT_DIM)


class HybridLane(QWidget):
    segmentsSelected = Signal(list)
    boundaryMoved = Signal(int, float)  # 경계 번호(오른쪽 구간 index), 새 시각
    loopRequested = Signal(float, float)
    seekRequested = Signal(float)
    actionRequested = Signal(str, list, float)  # 종류, 구간 id 목록, 시각
    sourceRequested = Signal(list, str)  # 구간 id 목록, 소스 키

    def __init__(self, timeline: TimelineState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = timeline
        self.plan: H.HybridPlan | None = None
        self.status: dict[str, H.SegmentStatus] = {}
        self.labels: dict[str, str] = {}  # 소스 키 → 표시 이름
        self.sources: list[tuple[str, str]] = []  # 메뉴에 넣을 (키, 이름)
        self.selected: list[str] = []
        self.changed: list[tuple[float, float]] = []  # 마지막으로 만든 결과와 달라진 범위 (표시용)
        self._drag_index: int | None = None
        self._drag_from = 0.0
        self._drag_t = 0.0
        self._anchor: str | None = None
        self.setMinimumHeight(30)
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setToolTip("Hybrid 구간: 경계를 끌어 옮기고, 오른쪽 클릭으로 나누기·합치기·모델 지정")
        timeline.changed.connect(self.update)

    # --- 상태 ------------------------------------------------------------------------

    def set_plan(self, plan: H.HybridPlan | None, status: dict | None = None, labels: dict | None = None,
                 sources: list | None = None) -> None:
        self.plan = plan
        self.status = dict(status or {})
        self.labels = dict(labels or {})
        if sources is not None:
            self.sources = list(sources)
        ids = {s.segment_id for s in plan.segments} if plan is not None else set()
        self.selected = [s for s in self.selected if s in ids]
        self._drag_index = None
        self.update()

    def select(self, ids: list[str]) -> None:
        self.selected = list(ids)
        self.update()

    def _bounds(self) -> list[tuple[int, float]]:
        """(경계 번호, 시각). 끄는 중이면 그 경계는 끄는 위치."""
        if self.plan is None:
            return []
        out = []
        for i, s in enumerate(self.plan.segments[1:], start=1):
            t = self._drag_t if self._drag_index == i else self.plan.time(s.start_frame)
            out.append((i, t))
        return out

    def _spans(self) -> list[tuple[H.HybridSegment, float, float]]:
        if self.plan is None:
            return []
        b = dict(self._bounds())
        out = []
        segs = self.plan.segments
        for i, s in enumerate(segs):
            a = 0.0 if i == 0 else b[i]
            e = self.plan.duration if i == len(segs) - 1 else b[i + 1]
            out.append((s, a, e))
        return out

    def segment_at(self, x: float) -> H.HybridSegment | None:
        t = self.timeline.x_to_time(x)
        for s, a, b in self._spans():
            if a <= t < b:
                return s
        return None

    def boundary_at(self, x: float) -> int | None:
        best, best_d = None, EDGE_PX
        for i, t in self._bounds():
            d = abs(self.timeline.time_to_x(t) - x)
            if d <= best_d:
                best, best_d = i, d
        return best

    # --- 그리기 ------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.SURFACE)
        p.setPen(QPen(theme.BORDER, 1))
        p.drawLine(0, 0, self.width(), 0)
        if self.plan is None:
            p.setPen(theme.TEXT_DIM)
            p.setFont(theme.ui_font(11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Hybrid 구간 없음 — Hybrid 편집 모드에서 [현재 결과로 시작]을 누르세요")
            return
        h = self.height()
        top, bot = 4.0, h - 4.0
        w = self.width()
        sel = set(self.selected)
        p.setFont(theme.ui_font(11))
        fm = p.fontMetrics()
        for lo, hi in self.changed:  # 아직 만들지 않은 변경 범위: 아래 줄로 표시
            xa, xb = self.timeline.time_to_x(lo), self.timeline.time_to_x(hi)
            if xb >= 0 and xa <= w:
                p.fillRect(QRectF(xa, bot - 2, max(2.0, xb - xa), 2), theme.LYRIC_WARN)
        for s, a, b in self._spans():
            xa, xb = self.timeline.time_to_x(a), self.timeline.time_to_x(b)
            if xb < -2 or xa > w + 2:
                continue
            rect = QRectF(xa + 1, top, max(1.0, xb - xa - 2), bot - top - 3)
            c = QColor(model_color(s.source_model))
            c.setAlpha(70 if s.segment_id not in sel else 120)
            p.fillRect(rect, c)
            st = self.status.get(s.segment_id)
            bad = st is not None and not st.ok
            if bad:
                p.fillRect(rect, QBrush(theme.LYRIC_WARN, Qt.BrushStyle.BDiagPattern))
                pen = QPen(theme.LYRIC_WARN, 1.5, Qt.PenStyle.DashLine)
            elif s.segment_id in sel:
                pen = QPen(theme.TEXT, 2)
            else:
                pen = QPen(model_color(s.source_model), 1)
            p.setPen(pen)
            p.drawRect(rect)
            name = SHORT_LABELS.get(s.source_model, s.source_model)
            extra = []
            if abs(s.gain_db) > 1e-6:
                extra.append(f"{s.gain_db:+.1f}dB")
            text = ("! " if bad else "") + name + (f" {' '.join(extra)}" if extra else "")
            if rect.width() > fm.horizontalAdvance(text) + 8:
                p.setPen(theme.LYRIC_WARN if bad else theme.TEXT)
                p.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            elif rect.width() > 14:
                p.setPen(theme.LYRIC_WARN if bad else theme.TEXT)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "!" if bad else name[:1])
        for i, t in self._bounds():  # 경계 손잡이
            x = self.timeline.time_to_x(t)
            if -2 <= x <= w + 2:
                p.setPen(QPen(theme.TEXT if self._drag_index == i else theme.TEXT_DIM, 2))
                p.drawLine(QPointF(x, 1), QPointF(x, h - 1))

    # --- 마우스 ----------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.plan is None:
            return
        x = event.position().x()
        if event.button() == Qt.MouseButton.LeftButton:
            b = self.boundary_at(x)
            if b is not None:
                self._drag_index = b
                self._drag_from = self._drag_t = self.plan.time(self.plan.segments[b].start_frame)
                return
            s = self.segment_at(x)
            if s is None:
                return
            mods = event.modifiers()
            ids = [x.segment_id for x in self.plan.segments]
            if mods & Qt.KeyboardModifier.ShiftModifier and self._anchor in ids:
                i, j = sorted((ids.index(self._anchor), ids.index(s.segment_id)))
                self.selected = ids[i:j + 1]
            elif mods & Qt.KeyboardModifier.ControlModifier:
                self.selected = [x for x in self.selected if x != s.segment_id] if s.segment_id in self.selected \
                    else self.selected + [s.segment_id]
                self._anchor = s.segment_id
            else:
                self.selected = [s.segment_id]
                self._anchor = s.segment_id
            self.segmentsSelected.emit(list(self.selected))
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        if self._drag_index is not None and self.plan is not None:
            i = self._drag_index
            segs = self.plan.segments
            lo = self.plan.time(segs[i - 1].start_frame) + H.MIN_SEGMENT_S
            hi = self.plan.time(segs[i].end_frame) - H.MIN_SEGMENT_S
            self._drag_t = min(max(self.timeline.x_to_time(x), lo), hi)
            delta = (self._drag_t - self._drag_from) * 1000
            QToolTip.showText(event.globalPosition().toPoint(),
                              f"경계 {self._drag_t:.3f}s (처음 {self._drag_from:.3f}s, {delta:+.0f}ms)", self)
            self.update()
            return
        self.setCursor(Qt.CursorShape.SizeHorCursor if self.boundary_at(x) is not None else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_index is None:
            return
        i, t, t0 = self._drag_index, self._drag_t, self._drag_from
        self._drag_index = None
        QToolTip.hideText()
        self.update()
        if abs(t - t0) > 1e-6:
            self.boundaryMoved.emit(i, t)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        s = self.segment_at(event.position().x())
        if s is not None and self.plan is not None:
            self.loopRequested.emit(*self.plan.span(s))

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if self.plan is None:
            return
        x = float(event.pos().x())
        s = self.segment_at(x)
        if s is None:
            return
        if s.segment_id not in self.selected:
            self.selected = [s.segment_id]
            self.segmentsSelected.emit(list(self.selected))
            self.update()
        t = self.timeline.x_to_time(x)
        ids = list(self.selected)
        menu = QMenu(self)
        menu.addAction(f"여기서 나누기 ({t:.2f}s)", lambda: self.actionRequested.emit("split", [s.segment_id], t))
        merge = menu.addAction("고른 구간 합치기", lambda: self.actionRequested.emit("merge", ids, t))
        merge.setEnabled(len(ids) >= 2)
        nxt = menu.addAction("다음 구간과 합치기", lambda: self.actionRequested.emit("merge_next", [s.segment_id], t))
        nxt.setEnabled(s.segment_id != self.plan.segments[-1].segment_id)
        sub = menu.addMenu("모델 지정")
        for key, label in self.sources:
            a = sub.addAction(label, lambda k=key: self.sourceRequested.emit(ids, k))
            a.setCheckable(True)
            a.setChecked(all(self.plan.get(i).source_key == key for i in ids))
        sub.setEnabled(bool(self.sources))
        menu.addSeparator()
        menu.addAction("이 구간 반복 듣기", lambda: self.loopRequested.emit(*self.plan.span(s)))
        dele = menu.addAction("구간 지우기 (앞 구간에 붙임)", lambda: self.actionRequested.emit("delete", [s.segment_id], t))
        dele.setEnabled(len(self.plan.segments) > 1)
        menu.exec(event.globalPos())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.timeline.zoomRequested.emit(1.25 if delta > 0 else 0.8, self.timeline.x_to_time(event.position().x()))
        else:
            self.timeline.scrollRequested.emit(-float(delta))
        event.accept()
