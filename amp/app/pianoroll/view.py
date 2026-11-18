"""피아노 롤 뷰: 마우스·키보드 편집과 확대/스크롤을 담당한다.

조작 방법
- 클릭: 음표 선택 / Ctrl+클릭: 선택 토글 / 빈 곳 드래그: 범위 선택
- 음표 몸통 드래그: 이동 (시간·음높이), 좌우 끝 드래그: 길이 변경
- 드래그 중 Alt: 스냅 일시 해제
- 빈 곳 더블클릭: 음표 추가
- 오른쪽 클릭: 분할·병합·삭제 메뉴
- 방향키: ↑↓ 반음(Shift: 옥타브), ←→ 스냅 단위 이동
- 휠: 세로 스크롤 / Shift+휠: 가로 스크롤 / Ctrl+휠: 가로 확대 / Ctrl+Shift+휠: 세로 확대
"""

from __future__ import annotations

import math
from enum import Enum, auto

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QFrame, QGraphicsView, QMenu

from amp.app.document import ProjectDocument
from amp.app.pianoroll.items import NoteItem
from amp.app.pianoroll.mapping import MAX_ROW_HEIGHT, MIN_ROW_HEIGHT, PITCH_MAX, PianoRollMapper
from amp.app.pianoroll.scene import PianoRollScene
from amp.app.timeline import MAX_PX_PER_SEC, MIN_PX_PER_SEC, TimelineState
from amp.core import editing
from amp.core.errors import EditError
from amp.core.model import Note

EDGE_PX = 6.0  # 음표 끝에서 이 거리 안을 잡으면 길이 변경
DRAG_THRESHOLD_PX = 3.0
TIME_DEADZONE_PX = 5.0  # 세로로만 옮길 때 시간이 스냅으로 튀지 않게 하는 여유


class _Drag(Enum):
    NONE = auto()
    MOVE = auto()
    RESIZE_START = auto()
    RESIZE_END = auto()


class PianoRollView(QGraphicsView):
    editFailed = Signal(str)
    selectionInfoChanged = Signal(str)
    hoverInfoChanged = Signal(str)  # 마우스 위치의 음높이·시간
    noVocalRangeRequested = Signal(float, float)  # 선택 음표 구간 (시작초, 끝초)

    def __init__(self, doc: ProjectDocument, timeline: TimelineState, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.timeline = timeline
        self.mapper = PianoRollMapper(timeline.px_per_sec, 16.0)
        self.roll = PianoRollScene(doc, self.mapper, self)
        self.setScene(self.roll)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.follow_playhead = True

        self._drag = _Drag.NONE
        self._press_scene = QPointF()
        self._press_view = QPointF()
        self._anchor: Note | None = None
        self._originals: list[Note] = []
        self._pending: tuple | None = None  # 놓았을 때 반영할 (종류, 값...)
        self._click_only_select: str | None = None

        self.horizontalScrollBar().valueChanged.connect(self._sync_timeline)
        timeline.zoomRequested.connect(self._on_zoom_requested)
        timeline.scrollRequested.connect(self._on_scroll_requested)
        self.roll.selectionChanged.connect(self._emit_selection_info)

    # --- 공용 ----------------------------------------------------------------

    @property
    def snap_beats(self) -> float:
        return self.roll.snap_beats

    def set_snap_beats(self, value: float) -> None:
        self.roll.snap_beats = value
        self.roll.invalidate(self.roll.sceneRect())

    def selected_ids(self) -> list[str]:
        return self.roll.selected_ids()

    def visible_time_range(self) -> tuple[float, float]:
        r = self.mapToScene(self.viewport().rect()).boundingRect()
        return self.mapper.x_to_time(r.left()), self.mapper.x_to_time(r.right())

    def set_playhead(self, t: float) -> None:
        self.roll.set_playhead(t)
        if self.follow_playhead:
            t0, t1 = self.visible_time_range()
            if t > t1 - (t1 - t0) * 0.05 or t < t0:
                x = self.mapper.time_to_x(t) - self.viewport().width() * 0.1
                self.horizontalScrollBar().setValue(int(max(0.0, x)))

    def center_on_pitch(self, pitch: float) -> None:
        y = self.mapper.pitch_center_y(pitch) - self.viewport().height() / 2
        self.verticalScrollBar().setValue(int(max(0.0, y)))

    def center_on_content(self) -> None:
        notes = self.doc.score.notes()
        if notes:
            pitches = sorted(n.pitch for n in notes)
            self.center_on_pitch(pitches[len(pitches) // 2])
        else:
            self.center_on_pitch(60)

    def zoom_time(self, factor: float, anchor_time: float | None = None) -> None:
        if anchor_time is None:
            t0, t1 = self.visible_time_range()
            anchor_time = (t0 + t1) / 2
        anchor_vx = self.mapper.time_to_x(anchor_time) - self.horizontalScrollBar().value()
        pps = min(max(self.mapper.px_per_sec * factor, MIN_PX_PER_SEC), MAX_PX_PER_SEC)
        if math.isclose(pps, self.mapper.px_per_sec):
            return
        self.mapper.px_per_sec = pps
        self.roll.relayout()
        self.horizontalScrollBar().setValue(int(round(self.mapper.time_to_x(anchor_time) - anchor_vx)))
        self._sync_timeline()

    def zoom_to_fit(self) -> None:
        dur = self.roll.content_duration()
        width = max(self.viewport().width(), 100)
        self.zoom_time((width / dur) / self.mapper.px_per_sec, 0.0)
        self.horizontalScrollBar().setValue(0)

    def zoom_pitch(self, factor: float, anchor_y_view: float | None = None) -> None:
        if anchor_y_view is None:
            anchor_y_view = self.viewport().height() / 2
        scene_y = self.verticalScrollBar().value() + anchor_y_view
        pitch_f = PITCH_MAX - scene_y / self.mapper.row_height
        rh = min(max(self.mapper.row_height * factor, MIN_ROW_HEIGHT), MAX_ROW_HEIGHT)
        if math.isclose(rh, self.mapper.row_height):
            return
        self.mapper.row_height = rh
        self.roll.relayout()
        self.verticalScrollBar().setValue(int(round((PITCH_MAX - pitch_f) * rh - anchor_y_view)))

    # --- 내부: 타임라인 동기화 ------------------------------------------------------

    def _sync_timeline(self, *_args) -> None:
        self.timeline.update_view(self.mapper.px_per_sec, float(self.horizontalScrollBar().value()))

    def _on_scroll_requested(self, dx: float) -> None:
        bar = self.horizontalScrollBar()
        bar.setValue(int(bar.value() + dx))

    def _on_zoom_requested(self, factor: float, anchor_time: float) -> None:
        self.zoom_time(factor, anchor_time)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._sync_timeline()

    def _emit_selection_info(self) -> None:
        from amp.core.pitch import note_name

        # 항목을 지우는 도중에도 호출되므로 문서 대신 항목이 가진 음표를 쓴다
        items = [i for i in self.roll.selectedItems() if isinstance(i, NoteItem)]
        if not items:
            self.selectionInfoChanged.emit("")
        elif len(items) == 1:
            n = items[0].note
            self.selectionInfoChanged.emit(
                f"{note_name(n.pitch)}  {n.start:.3f}s ~ {n.end:.3f}s  ({n.duration * 1000:.0f}ms)"
            )
        else:
            self.selectionInfoChanged.emit(f"음표 {len(items)}개 선택")

    # --- 마우스 ----------------------------------------------------------------

    def _note_item_at(self, view_pos) -> NoteItem | None:
        for item in self.items(view_pos.toPoint() if hasattr(view_pos, "toPoint") else view_pos):
            if isinstance(item, NoteItem):
                return item
        return None

    def _hit_edge(self, item: NoteItem, scene_pos: QPointF) -> _Drag:
        r = item.rect()
        edge = min(EDGE_PX, r.width() / 4)
        if scene_pos.x() <= r.left() + edge:
            return _Drag.RESIZE_START
        if scene_pos.x() >= r.right() - edge:
            return _Drag.RESIZE_END
        return _Drag.MOVE

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        item = self._note_item_at(event.position())
        if item is None:
            super().mousePressEvent(event)  # 범위 선택 시작 / 선택 해제
            return
        self.setFocus()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl:
            item.setSelected(not item.isSelected())
            event.accept()
            return
        self._click_only_select = None
        if not item.isSelected():
            self.roll.clearSelection()
            item.setSelected(True)
        elif len(self.roll.selected_ids()) > 1:
            self._click_only_select = item.note.id  # 드래그 없이 놓으면 이 음표만 선택
        scene_pos = self.mapToScene(event.position().toPoint())
        self._drag = self._hit_edge(item, scene_pos)
        self._press_scene = scene_pos
        self._press_view = event.position()
        self._anchor = item.note
        self._originals = [self.doc.score.get(i) for i in self.roll.selected_ids()]
        self._pending = None
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is _Drag.NONE:
            self._update_cursor(event.position())
            self._emit_hover(event.position())
            super().mouseMoveEvent(event)
            return
        moved = event.position() - self._press_view
        if abs(moved.x()) < DRAG_THRESHOLD_PX and abs(moved.y()) < DRAG_THRESHOLD_PX and self._pending is None:
            return
        self._click_only_select = None
        scene_pos = self.mapToScene(event.position().toPoint())
        no_snap = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        self._preview(scene_pos, no_snap)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is _Drag.NONE or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        pending, ids = self._pending, [n.id for n in self._originals]
        click_only = self._click_only_select
        self._drag = _Drag.NONE
        self._pending = None
        self._click_only_select = None
        try:
            changed = False
            if pending is not None:
                kind = pending[0]
                if kind == "move":
                    changed = self.doc.move_notes(ids, pending[1], pending[2])
                else:
                    changed = self.doc.resize_notes(ids, pending[1], pending[2])
            if not changed:
                self._restore_preview()
        except EditError as e:
            self._restore_preview()
            self.editFailed.emit(str(e))
        if click_only:
            self.roll.select_ids([click_only])
        event.accept()

    def _snap(self, t: float, no_snap: bool) -> float:
        if no_snap or self.snap_beats <= 0:
            return t
        return self.doc.score.tempo.snap(t, self.snap_beats)

    def _preview(self, scene_pos: QPointF, no_snap: bool) -> None:
        anchor = self._anchor
        if anchor is None:
            return
        m = self.mapper
        dx = scene_pos.x() - self._press_scene.x()
        raw_dt = dx / m.px_per_sec
        if self._drag is _Drag.MOVE:
            dpitch = m.y_to_pitch(scene_pos.y()) - m.y_to_pitch(self._press_scene.y())
            if abs(dx) < TIME_DEADZONE_PX:
                dt = 0.0
            else:
                dt = self._snap(anchor.start + raw_dt, no_snap) - anchor.start
            dt, dpitch = editing.clamp_move(self._originals, dt, dpitch)
            preview = editing.move_notes(self._originals, dt, dpitch)
            self._pending = ("move", dt, dpitch)
        else:
            edge = "start" if self._drag is _Drag.RESIZE_START else "end"
            base = anchor.start if edge == "start" else anchor.end
            delta = self._snap(base + raw_dt, no_snap) - base
            delta = editing.clamp_resize(self._originals, edge, delta)
            preview = editing.resize_notes(self._originals, edge, delta)
            self._pending = ("resize", edge, delta)
        for note in preview:
            item = self.roll.items_by_id.get(note.id)
            if item is not None:
                item.show_preview(note)

    def _restore_preview(self) -> None:
        for note in self._originals:
            item = self.roll.items_by_id.get(note.id)
            if item is not None:
                item.sync()

    def _emit_hover(self, view_pos) -> None:
        from amp.core.pitch import midi_to_hz, note_name

        sp = self.mapToScene(view_pos.toPoint())
        t = max(0.0, self.mapper.x_to_time(sp.x()))
        p = self.mapper.y_to_pitch(sp.y())
        bar, beat = self.doc.score.tempo.bar_beat(t)
        self.hoverInfoChanged.emit(f"{note_name(p)} ({p}, {float(midi_to_hz(p)):.1f} Hz)  {t:.3f}s  {bar}.{beat:.2f}")

    def _update_cursor(self, view_pos) -> None:
        item = self._note_item_at(view_pos)
        if item is None:
            self.viewport().unsetCursor()
            return
        kind = self._hit_edge(item, self.mapToScene(view_pos.toPoint()))
        shape = Qt.CursorShape.SizeHorCursor if kind is not _Drag.MOVE else Qt.CursorShape.SizeAllCursor
        self.viewport().setCursor(shape)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._note_item_at(event.position()) is not None:
            super().mouseDoubleClickEvent(event)
            return
        self.add_note_at(self.mapToScene(event.position().toPoint()))
        event.accept()

    def add_note_at(self, scene_pos: QPointF) -> None:
        tempo = self.doc.score.tempo
        t = max(0.0, self.mapper.x_to_time(scene_pos.x()))
        step = self.snap_beats if self.snap_beats > 0 else 0.5
        if self.snap_beats > 0:
            start_beat = math.floor(tempo.beat_at(t) / step) * step
            start = max(0.0, tempo.time_at(start_beat))
        else:
            start = t
        end = tempo.time_at(tempo.beat_at(start) + step)
        try:
            self.doc.add_note(start, end, self.mapper.y_to_pitch(scene_pos.y()))
        except EditError as e:
            self.editFailed.emit(str(e))

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self._note_item_at(QPointF(event.pos()))
        scene_pos = self.mapToScene(event.pos())
        menu = QMenu(self)
        if item is not None:
            if not item.isSelected():
                self.roll.clearSelection()
                item.setSelected(True)
            at = self.mapper.x_to_time(scene_pos.x())
            snapped = self._snap(at, False)
            n = item.note
            if n.start + editing.MIN_NOTE_DURATION <= snapped <= n.end - editing.MIN_NOTE_DURATION:
                at = snapped
            split = QAction("여기서 분할", menu)
            split.triggered.connect(lambda: self._guard(lambda: self.doc.split_note(n.id, at)))
            menu.addAction(split)
            merge = QAction("선택한 음표 병합", menu)
            merge.setEnabled(len(self.selected_ids()) >= 2)
            merge.triggered.connect(lambda: self._guard(lambda: self.doc.merge_notes(self.selected_ids())))
            menu.addAction(merge)
            menu.addSeparator()
            span = self._selected_span()
            if span is not None:
                nv = QAction("선택 구간을 보컬 없음으로 지정", menu)
                nv.setToolTip("이 구간에서는 음표를 만들지 않고 음표를 다시 검출합니다. 직접 고친 음표는 남습니다.")
                nv.triggered.connect(lambda: self.noVocalRangeRequested.emit(span[0], span[1]))
                menu.addAction(nv)
            delete = QAction("삭제", menu)
            delete.triggered.connect(lambda: self._guard(lambda: self.doc.delete_notes(self.selected_ids())))
            menu.addAction(delete)
        else:
            add = QAction("여기에 음표 추가", menu)
            add.triggered.connect(lambda: self.add_note_at(scene_pos))
            menu.addAction(add)
        menu.exec(event.globalPos())

    def _selected_span(self) -> tuple[float, float] | None:
        notes = [self.roll.items_by_id[i].note for i in self.selected_ids() if i in self.roll.items_by_id]
        if not notes:
            return None
        return min(n.start for n in notes), max(n.end for n in notes)

    def _guard(self, fn) -> None:
        try:
            fn()
        except EditError as e:
            self.editFailed.emit(str(e))

    # --- 휠·키보드 ---------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if ctrl and shift:
            self.zoom_pitch(1.15 if delta > 0 else 1 / 1.15, event.position().y())
        elif ctrl:
            anchor_time = self.mapper.x_to_time(self.mapToScene(event.position().toPoint()).x())
            self.zoom_time(1.25 if delta > 0 else 0.8, anchor_time)
        elif shift:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - int(delta))
        else:
            super().wheelEvent(event)
            return
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        ids = self.selected_ids()
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if ids and key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = 12 if shift else 1
            self._guard(lambda: self.doc.move_notes(ids, 0.0, step if key == Qt.Key.Key_Up else -step))
            event.accept()
            return
        if ids and key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            tempo = self.doc.score.tempo
            anchor = min((self.doc.score.get(i) for i in ids), key=lambda n: n.start)
            step = self.snap_beats if self.snap_beats > 0 else 0.25
            sign = 1 if key == Qt.Key.Key_Right else -1
            dt = tempo.time_at(tempo.beat_at(anchor.start) + sign * step) - anchor.start
            self._guard(lambda: self.doc.move_notes(ids, dt, 0))
            event.accept()
            return
        super().keyPressEvent(event)
