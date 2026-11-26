"""피아노 롤 장면에 올라가는 그래픽 항목."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QStyle, QStyleOptionGraphicsItem

from amp.app import theme
from amp.app.pianoroll.mapping import PITCH_MAX, PianoRollMapper
from amp.core.model import Note, PitchCurve

Z_NOTE = 10
Z_CURVE = 20
Z_PLAYHEAD = 30


class NoteItem(QGraphicsRectItem):
    """음표 하나. 모델을 직접 바꾸지 않고, 드래그 중에는 미리보기 위치만 바꾼다.

    가사 글자(label)는 피치 곡선에 가리지 않도록 장면의 drawForeground에서 그린다.
    """

    def __init__(self, note: Note, status: str, mapper: PianoRollMapper) -> None:
        super().__init__()
        self.note = note
        self.status = status
        self.label = ""
        self.label_pron = ""  # STEP 4: 이 음표가 주 음표인 음절의 발음 표기
        self.label_ipa = ""
        self._mapper = mapper
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(Z_NOTE)
        self.setAcceptHoverEvents(False)
        self.sync(note, status)

    def model_rect(self, note: Note | None = None) -> QRectF:
        n = note or self.note
        m = self._mapper
        x0 = m.time_to_x(n.start)
        x1 = m.time_to_x(n.end)
        return QRectF(x0, m.pitch_to_y(n.pitch) + 1, max(x1 - x0, 2.0), m.row_height - 2)

    def sync(self, note: Note | None = None, status: str | None = None) -> None:
        if note is not None:
            self.note = note
        if status is not None:
            self.status = status
        self.setRect(self.model_rect())
        self.setToolTip(self._tooltip())

    def show_preview(self, note: Note) -> None:
        self.setRect(self.model_rect(note))

    def set_label(self, label: str) -> None:
        if label != self.label:
            self.label = label
            if self.scene() is not None:
                self.scene().update(self.sceneBoundingRect())

    def _tooltip(self) -> str:
        from amp.core.pitch import note_name

        n = self.note
        parts = [
            f"{note_name(n.pitch)} (MIDI {n.pitch})",
            f"{n.start:.3f}s ~ {n.end:.3f}s ({n.duration * 1000:.0f}ms)",
            {"original": "분석 원본", "modified": "수정됨", "user": "사용자 추가"}.get(self.status, ""),
        ]
        if n.confidence is not None:
            parts.append(f"신뢰도 {n.confidence:.2f}")
        if n.vibrato is not None:
            parts.append(f"비브라토 {n.vibrato.rate_hz:.1f}Hz ±{n.vibrato.extent_cents:.0f}c")
        return "\n".join(parts)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        """음표는 모두 바이올렛. 상태는 색만이 아니라 채움·테두리·표식으로 구분한다.

        원본: 꽉 찬 채움 / 수정: 점선 테두리 + 오른쪽 위 삼각 표식 / 추가: 옅은 채움 + 굵은 테두리 + '+' 표식
        장식음: 점 테두리 / 선택: 밝은 테두리 + 양 끝 크기 조절 손잡이
        """
        r = self.rect()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        fill = QColor(theme.ACCENT)
        fill.setAlpha({"original": 235, "modified": 205, "user": 110}.get(self.status, 235))
        if selected:
            fill = fill.lighter(115)
        if selected:
            pen = QPen(theme.NOTE_SELECTED_BORDER, 2.0)
        elif self.status == "user":
            pen = QPen(theme.ACCENT, 1.6)
        elif self.status == "modified":
            pen = QPen(theme.TEXT, 1.2, Qt.PenStyle.DashLine)
        else:
            pen = QPen(theme.NOTE_BORDER, 1.0)
        if self.note.ornament and not selected:
            pen.setStyle(Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        radius = min(3.0, r.height() / 3)
        painter.drawRoundedRect(r, radius, radius)
        h = r.height()
        if r.width() > 10 and h >= 7:
            mark = QPen(theme.TEXT, 1.2)
            mark.setCosmetic(True)
            painter.setPen(mark)
            s = min(5.0, h / 2)
            if self.status == "modified":
                painter.setBrush(QBrush(theme.TEXT))
                tri = QPolygonF([QPointF(r.right() - s - 1, r.top() + 1), QPointF(r.right() - 1, r.top() + 1),
                                 QPointF(r.right() - 1, r.top() + 1 + s)])
                painter.drawPolygon(tri)
            elif self.status == "user":
                cx, cy = r.left() + 2 + s / 2, r.top() + h / 2
                painter.drawLine(QPointF(cx - s / 2, cy), QPointF(cx + s / 2, cy))
                painter.drawLine(QPointF(cx, cy - s / 2), QPointF(cx, cy + s / 2))
        if selected and r.width() > 8:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(theme.TEXT))
            hw = 3.0
            painter.drawRect(QRectF(r.left(), r.top() + h * 0.2, hw, h * 0.6))
            painter.drawRect(QRectF(r.right() - hw, r.top() + h * 0.2, hw, h * 0.6))


class PitchCurveItem(QGraphicsItem):
    """분석된 f0 곡선. 화면에 드러난 구간만 골라 그린다.

    무성 프레임(NaN)에서는 선을 끊는다. 유성이지만 신뢰도가 MIN_CONFIDENCE 미만인 프레임은
    숨기지 않고 흐린 점선으로 그린다(불확실 구간).
    """

    MIN_CONFIDENCE = 0.3

    def __init__(self, mapper: PianoRollMapper) -> None:
        super().__init__()
        self._mapper = mapper
        self._curve: PitchCurve | None = None
        self._midi: np.ndarray | None = None
        self._sure: np.ndarray | None = None
        self._bounds = QRectF()
        self.setZValue(Z_CURVE)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def set_curve(self, curve: PitchCurve | None) -> None:
        self._curve = curve
        if curve is None:
            self._midi = self._sure = None
        else:
            self._midi = curve.midi().astype(np.float64)
            self._sure = np.asarray(curve.confidence) >= self.MIN_CONFIDENCE
        self.relayout()

    def uncertain_fraction(self) -> float:
        """유성 프레임 중 불확실(흐리게 그리는) 프레임 비율. 테스트·상태 표시용."""
        if self._midi is None or self._sure is None:
            return 0.0
        voiced = np.isfinite(self._midi)
        return float((voiced & ~self._sure).sum() / max(1, voiced.sum()))

    def relayout(self) -> None:
        self.prepareGeometryChange()
        if self._curve is None:
            self._bounds = QRectF()
        else:
            m = self._mapper
            self._bounds = QRectF(
                m.time_to_x(self._curve.start_time), 0, m.time_to_x(self._curve.end_time - self._curve.start_time), m.scene_height
            )
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        if self._curve is None or self._midi is None:
            return
        m = self._mapper
        exposed = option.exposedRect if option.exposedRect.isValid() else self._bounds
        t0 = m.x_to_time(exposed.left()) - 0.05
        t1 = m.x_to_time(exposed.right()) + 0.05
        i0, i1 = self._curve.frame_range(t0, t1)
        if i1 - i0 < 2:
            return
        midi = self._midi[i0:i1]
        sure = self._sure[i0:i1]
        # 화면 1픽셀에 프레임이 여러 개 몰리면 간격을 두고 건너뛴다
        frames_per_px = self._curve.frame_rate / m.px_per_sec
        step = max(1, int(frames_per_px / 2))
        idx = np.arange(0, midi.shape[0], step)
        xs = m.time_to_x(self._curve.start_time) + (i0 + idx) / self._curve.frame_rate * m.px_per_sec
        ys = (PITCH_MAX - midi[idx]) * m.row_height + m.row_height / 2
        paths = {True: QPainterPath(), False: QPainterPath()}
        prev = None  # 직전 점 (x, y). 두 점 사이 선은 뒤 점의 신뢰도로 그린다
        for x, y, ok in zip(xs.tolist(), ys.tolist(), sure[idx].tolist()):
            if y != y:  # NaN = 무성 구간
                prev = None
                continue
            path = paths[bool(ok)]
            if prev is None:
                path.moveTo(x, y)
            else:
                path.moveTo(*prev)
                path.lineTo(x, y)
            prev = (x, y)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        faint = QColor(theme.PITCH_CURVE)
        faint.setAlphaF(0.45)
        pen = QPen(faint, 1.4, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(paths[False])
        pen = QPen(theme.PITCH_CURVE, 1.8)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(paths[True])


class OverlayCurvesItem(QGraphicsItem):
    """v0.7.0 피치 모델 결과 겹쳐 보기. 편집할 수 없는 그림이다. 층마다 (시작 프레임, 프레임률, MIDI 배열, 색, 이름).
    분석 곡선보다 가늘게, 무성(NaN)에서 끊어 그린다."""

    def __init__(self, mapper: PianoRollMapper) -> None:
        super().__init__()
        self._mapper = mapper
        self._layers: dict[str, tuple] = {}
        self._bounds = QRectF()
        self.setZValue(Z_CURVE - 1)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def set_layers(self, layers: dict[str, tuple]) -> None:
        self._layers = {k: (int(s), float(fr), np.asarray(midi, dtype=np.float64), QColor(c))
                        for k, (s, fr, midi, c) in layers.items()}
        self.relayout()

    def layer_keys(self) -> list[str]:
        return list(self._layers)

    def relayout(self) -> None:
        self.prepareGeometryChange()
        if not self._layers:
            self._bounds = QRectF()
        else:
            m = self._mapper
            t0 = min(s / fr for s, fr, _mi, _c in self._layers.values())
            t1 = max((s + len(mi)) / fr for s, fr, mi, _c in self._layers.values())
            self._bounds = QRectF(m.time_to_x(t0), 0, m.time_to_x(t1 - t0), m.scene_height)
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        if not self._layers:
            return
        m = self._mapper
        exposed = option.exposedRect if option.exposedRect.isValid() else self._bounds
        t0 = m.x_to_time(exposed.left()) - 0.05
        t1 = m.x_to_time(exposed.right()) + 0.05
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for start, fr, midi, color in self._layers.values():
            i0 = max(0, int(t0 * fr) - start)
            i1 = min(len(midi), int(t1 * fr) - start + 2)
            if i1 - i0 < 2:
                continue
            step = max(1, int(fr / m.px_per_sec / 2))
            idx = np.arange(i0, i1, step)
            xs = m.time_to_x((start + idx) / fr)
            ys = (PITCH_MAX - midi[idx]) * m.row_height + m.row_height / 2
            path = QPainterPath()
            pen_down = False
            for x, y in zip(np.atleast_1d(xs).tolist(), ys.tolist()):
                if y != y:
                    pen_down = False
                    continue
                if pen_down:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
                    pen_down = True
            pen = QPen(color, 1.2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)


class PlayheadItem(QGraphicsItem):
    def __init__(self, mapper: PianoRollMapper) -> None:
        super().__init__()
        self._mapper = mapper
        self.setZValue(Z_PLAYHEAD)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-1.5, 0, 3, self._mapper.scene_height)

    def set_time(self, t: float) -> None:
        self.setPos(QPointF(self._mapper.time_to_x(t), 0))

    def relayout(self, t: float) -> None:
        self.prepareGeometryChange()
        self.set_time(t)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pen = QPen(theme.PLAYHEAD, 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(0, 0), QPointF(0, self._mapper.scene_height))
