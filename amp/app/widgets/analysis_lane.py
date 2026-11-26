"""분석 레인 (v0.7.0 명세 16.1): 비브라토 · 호흡/프레이즈 · 발성 경계를 피아노 롤과 같은 시간축에 줄로 보여 준다.

- 세 줄은 따로 켜고 끈다. 처음에는 모두 꺼져 있고, 켠 줄이 없으면 레인 자체를 숨긴다
- 읽기 전용이다: 클릭하면 그 시각으로 옮기고, 마우스를 올리면 값을 보여 준다. 고치는 일은 오른쪽 '음향 분석' 탭에서 한다
- 모양으로도 구분한다 (색만으로 구분하지 않음): 비브라토 '불확실'은 빗금, 호흡 '확정'은 채움·'후보'는 테두리·'아님'은 ×,
  경계는 표식 모양이 종류마다 다르다 (▲ 소리 시작 · │ 유성 시작 · ◆ 목표 음높이 도달 · ▼ 유성 끝)
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from amp.app import theme
from amp.app.timeline import TimelineState

ROWS = (("vibrato", "비브라토"), ("breath", "호흡·프레이즈"), ("boundary", "발성 경계"))
ROW_H = 20
MARKS = {"acoustic_onset": "▲ 소리 시작", "voiced_onset": "│ 유성 시작", "stable_pitch_onset": "◆ 목표 음높이 도달",
         "voiced_end": "▼ 유성 끝"}


class AnalysisLane(QWidget):
    seekRequested = Signal(float)

    def __init__(self, timeline: TimelineState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = timeline
        self.visible_rows: list[str] = []
        self.vibrato: list = []
        self.breaths: list = []
        self.phrases: list = []
        self.boundaries: list = []
        self.stale = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        timeline.changed.connect(self.update)
        self._resize()

    # --- 데이터 ------------------------------------------------------------------

    def set_result(self, res) -> None:
        """acoustics.run.AcousticResult 또는 None."""
        if res is None:
            self.vibrato, self.breaths, self.phrases, self.boundaries, self.stale = [], [], [], [], False
        else:
            from amp.core import result_meta as RM

            self.vibrato, self.breaths, self.phrases = list(res.vibrato), list(res.breaths), list(res.phrases)
            self.boundaries = list(res.boundaries)
            self.stale = any(m.status == RM.STALE for m in (res.meta or {}).values())
        self.update()

    def set_rows(self, rows) -> None:
        self.visible_rows = [k for k, _l in ROWS if k in set(rows)]
        self._resize()
        self.setVisible(bool(self.visible_rows))
        self.update()

    def _resize(self) -> None:
        self.setFixedHeight(max(1, len(self.visible_rows)) * ROW_H + 2)

    def _row_rect(self, key: str) -> QRectF | None:
        if key not in self.visible_rows:
            return None
        i = self.visible_rows.index(key)
        return QRectF(0, 1 + i * ROW_H, self.width(), ROW_H)

    # --- 그리기 ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), theme.SURFACE)
        p.setFont(theme.ui_font(10))
        tl = self.timeline
        for key, label in ROWS:
            r = self._row_rect(key)
            if r is None:
                continue
            p.setPen(QPen(theme.BORDER, 1))
            p.drawLine(QPointF(0, r.bottom()), QPointF(self.width(), r.bottom()))
            if key == "vibrato":
                for s in self.vibrato:
                    xa, xb = tl.time_to_x(s.start), tl.time_to_x(s.end)
                    if xb < 0 or xa > self.width():
                        continue
                    box = QRectF(xa, r.top() + 4, max(2.0, xb - xa), r.height() - 8)
                    color = QColor(theme.ACCENT)
                    if s.status == "completed":
                        p.fillRect(box, color)
                    else:
                        p.fillRect(box, QBrush(color, Qt.BrushStyle.BDiagPattern))
                        p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                        p.drawRect(box)
                    if box.width() > 60:
                        p.setPen(theme.BG)
                        p.drawText(box.adjusted(3, 0, -3, 0), Qt.AlignmentFlag.AlignVCenter,
                                   f"{s.rate_hz:.1f}Hz ±{s.extent_cents:.0f}c")
            elif key == "breath":
                for ph in self.phrases:
                    xa, xb = tl.time_to_x(ph.start), tl.time_to_x(ph.end)
                    if xb < 0 or xa > self.width():
                        continue
                    p.fillRect(QRectF(xa, r.bottom() - 4, max(2.0, xb - xa), 2), theme.TEXT_DIM)
                for b in self.breaths:
                    xa, xb = tl.time_to_x(b.start), tl.time_to_x(b.end)
                    if xb < 0 or xa > self.width():
                        continue
                    box = QRectF(xa, r.top() + 3, max(3.0, xb - xa), r.height() - 9)
                    c = QColor(theme.PITCH_CURVE)
                    if b.status == "confirmed":
                        p.fillRect(box, c)
                    elif b.status == "rejected":
                        p.setPen(QPen(theme.TEXT_DIM, 1))
                        p.drawLine(box.topLeft(), box.bottomRight())
                        p.drawLine(box.bottomLeft(), box.topRight())
                    else:
                        p.setPen(QPen(c, 1.2))
                        p.drawRect(box)
            elif key == "boundary":
                for nb in self.boundaries:
                    for field in MARKS:
                        t = getattr(nb, field)
                        if t is None:
                            continue
                        x = tl.time_to_x(t)
                        if x < -6 or x > self.width() + 6:
                            continue
                        self._mark(p, field, x, r)
            p.setPen(theme.TEXT_DIM)
            p.drawText(QRectF(4, r.top(), 120, r.height()), Qt.AlignmentFlag.AlignVCenter,
                       label + (" (오래됨)" if self.stale else ""))
        p.end()

    def _mark(self, p: QPainter, field: str, x: float, r: QRectF) -> None:
        mid = r.center().y()
        c = QColor(theme.LYRIC_WARN if field in ("acoustic_onset", "voiced_end") else theme.ACCENT)
        p.setPen(QPen(c, 1.2))
        p.setBrush(c)
        if field == "acoustic_onset":
            p.drawPolygon(QPolygonF([QPointF(x, mid - 5), QPointF(x - 4, mid + 3), QPointF(x + 4, mid + 3)]))
        elif field == "voiced_onset":
            p.drawLine(QPointF(x, r.top() + 3), QPointF(x, r.bottom() - 3))
        elif field == "stable_pitch_onset":
            p.drawPolygon(QPolygonF([QPointF(x, mid - 4), QPointF(x + 4, mid), QPointF(x, mid + 4), QPointF(x - 4, mid)]))
        else:
            p.drawPolygon(QPolygonF([QPointF(x - 4, mid - 3), QPointF(x + 4, mid - 3), QPointF(x, mid + 5)]))
        p.setBrush(Qt.BrushStyle.NoBrush)

    # --- 조작 ------------------------------------------------------------------

    def describe(self, x: float, y: float) -> str:
        """그 자리의 값 (풍선 도움말·시험용)."""
        t = self.timeline.x_to_time(x)
        tol = 4.0 / max(1e-6, self.timeline.px_per_sec)
        row = next((k for k in self.visible_rows if self._row_rect(k).contains(QPointF(x, y))), None)
        if row == "vibrato":
            s = next((s for s in self.vibrato if s.start <= t <= s.end), None)
            if s:
                return (f"비브라토 {s.start:.2f}~{s.end:.2f}s · {s.rate_hz:.2f}Hz · 한쪽 {s.extent_cents:.0f}센트 "
                        f"(봉우리 간 {s.extent_pp_cents:.0f}) · {'확인됨' if s.status == 'completed' else '불확실'}")
        elif row == "breath":
            b = next((b for b in self.breaths if b.start - tol <= t <= b.end + tol), None)
            if b:
                st = {"candidate": "후보", "confirmed": "확정", "rejected": "아님"}.get(b.status, b.status)
                return f"호흡 {st} {b.start:.2f}~{b.end:.2f}s · {b.level_db:.0f}dB · 점수 {b.score:.2f}"
            ph = next((ph for ph in self.phrases if ph.start <= t <= ph.end), None)
            if ph:
                return f"프레이즈 {ph.start:.2f}~{ph.end:.2f}s · 유성 {ph.voiced_s:.2f}s"
        elif row == "boundary":
            best = None
            for nb in self.boundaries:
                for field in MARKS:
                    v = getattr(nb, field)
                    if v is not None and abs(v - t) <= tol and (best is None or abs(v - t) < best[0]):
                        best = (abs(v - t), field, v)
            if best:
                return f"{MARKS[best[1]]} {best[2]:.3f}s (음표에 자동으로 적용하지 않음)"
        return ""

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        text = self.describe(event.position().x(), event.position().y())
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.seekRequested.emit(max(0.0, self.timeline.x_to_time(event.position().x())))

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.timeline.zoomRequested.emit(1.25 if delta > 0 else 0.8, self.timeline.x_to_time(event.position().x()))
        else:
            self.timeline.scrollRequested.emit(-float(delta))
        event.accept()
