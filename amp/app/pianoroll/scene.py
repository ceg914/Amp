"""피아노 롤 장면: 문서 시그널을 받아 음표 항목을 만들고 고친다."""

from __future__ import annotations

import math
from contextlib import contextmanager

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsScene

from amp.app import theme
from amp.app.document import ProjectDocument
from amp.app.pianoroll.items import NoteItem, PitchCurveItem, PlayheadItem
from amp.app.pianoroll.mapping import PITCH_MAX, PITCH_MIN, PianoRollMapper
from amp.core.model import Note
from amp.core.pitch import is_black_key

MIN_SCENE_SECONDS = 30.0
TAIL_SECONDS = 4.0
MIN_GRID_PX = 7.0  # 격자선 간격이 이보다 좁으면 그리지 않는다


class PianoRollScene(QGraphicsScene):
    def __init__(self, doc: ProjectDocument, mapper: PianoRollMapper, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.mapper = mapper
        self.audio_duration = 0.0
        self.snap_beats = 0.25
        self.items_by_id: dict[str, NoteItem] = {}
        self.curve_item = PitchCurveItem(mapper)
        self.addItem(self.curve_item)
        self.playhead = PlayheadItem(mapper)
        self.addItem(self.playhead)
        self._playhead_time = 0.0
        self.no_vocal_ranges: tuple = ()  # 보컬 없는 구간 (초). 배경에 빗금 띠로 그린다
        self.lyrics_visible = True
        self.highlight_notes: set[str] = set()  # 선택한 음절에 연결된 음표 (민트 테두리)
        self.current_notes: set[str] = set()  # 재생 중인 음절에 연결된 음표 (글자 강조)
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)

        doc.projectReset.connect(self.rebuild)
        doc.analysisChanged.connect(self.rebuild)
        doc.notesReplaced.connect(self._on_notes_replaced)
        doc.tempoChanged.connect(self._on_tempo_changed)
        doc.selectionRequested.connect(self.select_ids)
        doc.lyricsChanged.connect(self._on_lyrics_changed)
        self.rebuild()

    # --- 크기 ---------------------------------------------------------------

    def content_duration(self) -> float:
        p = self.doc.project
        candidates = [self.audio_duration, p.score.end_time()]
        if p.analysis is not None:
            candidates.append(p.analysis.duration)
        if p.audio is not None:
            candidates.append(p.audio.duration)
        return max(max(candidates) + TAIL_SECONDS, MIN_SCENE_SECONDS)

    def update_scene_rect(self) -> None:
        w = self.mapper.time_to_x(self.content_duration())
        self.setSceneRect(QRectF(0, 0, w, self.mapper.scene_height))

    def set_audio_duration(self, seconds: float) -> None:
        self.audio_duration = seconds
        self.update_scene_rect()

    # --- 문서 동기화 ----------------------------------------------------------

    def rebuild(self) -> None:
        with self._batched_selection():
            for item in self.items_by_id.values():
                self.removeItem(item)
            self.items_by_id.clear()
            for note in self.doc.score.notes():
                self._add_item(note)
        analysis = self.doc.project.analysis
        self.curve_item.set_curve(analysis.pitch_curve if analysis else None)
        self._refresh_labels()
        self.update_scene_rect()
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

    def _on_tempo_changed(self) -> None:
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

    def _add_item(self, note: Note) -> NoteItem:
        item = NoteItem(note, self.doc.note_status(note), self.mapper)
        item.setVisible(getattr(self, "notes_visible", True))
        self.items_by_id[note.id] = item
        self.addItem(item)
        return item

    def _on_notes_replaced(self, removed_ids: list, added: list) -> None:
        added_ids = {n.id for n in added}
        with self._batched_selection():  # 선택된 항목을 지울 때마다 시그널이 나가지 않게
            for rid in removed_ids:
                if rid in added_ids:
                    continue  # 같은 id는 아래에서 갱신
                item = self.items_by_id.pop(rid, None)
                if item is not None:
                    self.removeItem(item)
            for note in added:
                item = self.items_by_id.get(note.id)
                if item is None:
                    self._add_item(note)
                else:
                    item.sync(note, self.doc.note_status(note))
        self._refresh_labels()
        self.update_scene_rect()

    def _on_lyrics_changed(self) -> None:
        self._refresh_labels()
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)

    def _refresh_labels(self) -> None:
        """음절이 시작하는 음표에는 글자를(여러 음절이 한 음표면 이어 붙임: '사랑'),
        멜리스마로 이어지는 음표에는 '–'를 붙인다. 연결은 가사 쪽(alignment)에서 읽는다."""
        ly = self.doc.lyrics
        text, al = ly.text, ly.alignment
        first_of: dict[str, str] = {}
        self.melisma_chains: list[list[str]] = []
        for sid, links in al.links.items():
            if sid not in text.index:
                continue
            notes = sorted((self.items_by_id[lk.note_id].note for lk in links if lk.note_id in self.items_by_id),
                           key=lambda n: (n.start, n.id))
            if notes:
                first_of[sid] = notes[0].id
                if len(notes) > 1:
                    self.melisma_chains.append([n.id for n in notes])
        for nid, item in self.items_by_id.items():
            sids = sorted((x for x in al.note_to_syllables.get(nid, ()) if x in text.index), key=text.order.get)
            firsts = [x for x in sids if first_of.get(x) == nid]
            if firsts:
                item.set_label("".join(text.syllable(x).text for x in firsts))
            elif sids:
                item.set_label("–")
            else:
                item.set_label("")

    def set_lyrics_visible(self, visible: bool) -> None:
        self.lyrics_visible = visible
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)

    def set_highlights(self, notes: set[str] | None = None, current: set[str] | None = None) -> None:
        if notes is not None:
            self.highlight_notes = set(notes)
        if current is not None:
            if set(current) == self.current_notes and notes is None:
                return
            self.current_notes = set(current)
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)

    def set_no_vocal_ranges(self, ranges) -> None:
        self.no_vocal_ranges = tuple(ranges)
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

    def relayout(self) -> None:
        """확대 배율이 바뀌었을 때 모든 항목 위치를 다시 계산한다."""
        for item in self.items_by_id.values():
            item.sync()
        self.curve_item.relayout()
        self.playhead.relayout(self._playhead_time)
        self.update_scene_rect()
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

    # --- 선택 ---------------------------------------------------------------

    def selected_ids(self) -> list[str]:
        return [i.note.id for i in self.selectedItems() if isinstance(i, NoteItem)]

    def select_ids(self, ids: list) -> None:
        # 항목마다 selectionChanged가 나가면 음표가 많을 때 O(n²)이 되므로 한 번만 내보낸다
        with self._batched_selection():
            self.clearSelection()
            for nid in ids:
                item = self.items_by_id.get(nid)
                if item is not None:
                    item.setSelected(True)

    @contextmanager
    def _batched_selection(self):
        before = self.selected_ids()
        blocked = self.blockSignals(True)
        try:
            yield
        finally:
            self.blockSignals(blocked)
        if not blocked and set(self.selected_ids()) != set(before):
            self.selectionChanged.emit()

    def set_notes_visible(self, visible: bool) -> None:
        """음표 레이어 표시/숨김 (피치 곡선과 따로)."""
        self.notes_visible = visible
        for item in self.items_by_id.values():
            item.setVisible(visible)
        self.update()

    def set_playhead(self, t: float) -> None:
        self._playhead_time = t
        self.playhead.set_time(t)

    # --- 배경 격자 -------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        m = self.mapper
        painter.fillRect(rect, theme.BG)
        p_top = min(PITCH_MAX, m.y_to_pitch(rect.top()))
        p_bot = max(PITCH_MIN, m.y_to_pitch(rect.bottom()))
        left, right = rect.left(), rect.right()
        for p in range(p_bot, p_top + 1):
            y = m.pitch_to_y(p)
            painter.fillRect(QRectF(left, y, rect.width(), m.row_height), theme.ROW_BLACK if is_black_key(p) else theme.ROW_WHITE)
        c_pen = QPen(theme.ROW_C_LINE, 1)
        c_pen.setCosmetic(True)
        painter.setPen(c_pen)
        c_lines = [QLineF(left, m.pitch_to_y(p) + m.row_height, right, m.pitch_to_y(p) + m.row_height)
                   for p in range(p_bot, p_top + 1) if p % 12 == 0]
        if c_lines:
            painter.drawLines(c_lines)
        for a, b in self.no_vocal_ranges:  # 보컬 없는 구간: 어둡게 + 빗금
            x0, x1 = max(left, m.time_to_x(a)), min(right, m.time_to_x(b))
            if x1 > x0:
                band = QRectF(x0, rect.top(), x1 - x0, rect.height())
                painter.fillRect(band, theme.NO_VOCAL_SHADE)
                painter.fillRect(band, QBrush(theme.NO_VOCAL_HATCH, Qt.BrushStyle.BDiagPattern))

        tempo = self.doc.score.tempo
        t0 = max(0.0, m.x_to_time(left))
        t1 = m.x_to_time(right)
        if t1 <= t0:
            return
        beat_px = min(m.px_per_sec * 60.0 / p.bpm for p in tempo.points)  # 가장 빠른 템포 기준
        bpb = tempo.beats_per_bar
        sub = self.snap_beats if 0 < self.snap_beats < 1 else 0.0
        top, bottom = rect.top(), rect.bottom()
        layers = []
        if sub and beat_px * sub >= MIN_GRID_PX:
            layers.append((sub, theme.GRID_SUB))
        if beat_px >= MIN_GRID_PX:
            layers.append((1.0, theme.GRID_BEAT))
        bar_step = float(bpb)
        while beat_px * bar_step < MIN_GRID_PX:
            bar_step *= 2
        layers.append((bar_step, theme.GRID_BAR))
        for step, color in layers:
            pen = QPen(color, 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            lines = []
            for b in tempo.grid_beats(t0, t1, step):
                if step < bar_step and math.isclose(b % bar_step, 0.0, abs_tol=1e-6):
                    continue
                x = m.time_to_x(tempo.time_at(b))
                lines.append(QLineF(x, top, x, bottom))
            if lines:
                painter.drawLines(lines)

    # --- 가사 글자 ---------------------------------------------------------------

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        """가사는 음표 아래에 쓴다 (명세: 음표 아래 가사 표시). 멜리스마는 민트 연결선으로 잇는다."""
        if not self.lyrics_visible:
            return
        grow = QRectF(rect).adjusted(-200, -40, 200, 40)
        visible = [i for i in self.items(grow) if isinstance(i, NoteItem) and i.isVisible()]
        if not visible:
            return
        pen = QPen(theme.LYRIC_CURRENT, 1.4)
        pen.setCosmetic(True)
        painter.setPen(pen)
        on_screen = {i.note.id for i in visible}
        for chain in getattr(self, "melisma_chains", ()):
            if not on_screen.intersection(chain):
                continue
            pts = [self.items_by_id[n].rect() for n in chain if n in self.items_by_id]
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(QPointF(a.right(), a.center().y()), QPointF(b.left(), b.center().y()))
        for item in visible:
            if item.note.id in self.highlight_notes:
                hp = QPen(theme.LYRIC_CURRENT, 2.2)
                hp.setCosmetic(True)
                painter.setPen(hp)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(item.rect().adjusted(-2, -2, 2, 2), 3, 3)
        font = theme.ui_font()
        font.setPixelSize(max(10, min(13, int(self.mapper.row_height - 2))))
        painter.setFont(font)
        fm = painter.fontMetrics()
        for item in visible:
            if not item.label:
                continue
            r = item.rect()
            w = max(fm.horizontalAdvance(item.label) + 6, 12)
            box = QRectF(r.left(), r.bottom() + 1, w, fm.height() + 1)
            current = item.note.id in self.current_notes
            bg = QColor(theme.LYRIC_CURRENT if current else theme.SURFACE_RAISED)
            bg.setAlpha(235 if current else 215)
            painter.fillRect(box, bg)
            painter.setPen(theme.BG if current else theme.TEXT)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, item.label)
