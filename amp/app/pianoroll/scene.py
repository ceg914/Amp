"""피아노 롤 장면: 문서 시그널을 받아 음표 항목을 만들고 고친다."""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsScene

from amp.app import theme
from amp.app.document import ProjectDocument
from amp.app.pianoroll.items import NoteItem, PitchCurveItem, PlayheadItem
from amp.app.pianoroll.mapping import PITCH_MAX, PITCH_MIN, PianoRollMapper
from amp.core.model import Note
from amp.core.pitch import is_black_key
from amp.lyrics.pronunciation import marked_text as pron_marked

IPA_PAD = 4.0  # IPA 줄 아래 여백(px)

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
        self.loop_region: tuple[float, float] | None = None  # 반복 재생 구간 (초). 배경에 옅은 바이올렛 띠
        # STEP 5: 참조 악보 음표 [(음원 시작, 끝, 음높이, 표시: "" / high / low / missing / uncertain)]. 편집할 수 없는 그림
        self.reference_notes: list[tuple[float, float, int, str]] = []
        self.reference_mode = "both"  # auto / ref / both / diff
        self.reference_preview = False  # 정렬을 확정하지 않음 (점선을 더 촘촘히)
        # STEP 5 모델 비교: "a"/"b" → {"notes": [(시작, 끝, 음높이)], "curve": PitchCurve | None, "label": str}
        # 비교용 자동 결과는 편집할 수 없는 그림이다. A는 행 위쪽 절반 바이올렛 점선, B는 아래쪽 절반 민트 점점선
        self.model_layers: dict[str, dict] = {}
        self.model_show = {"a_notes": True, "b_notes": True, "a_curve": True, "b_curve": True}
        self.lyrics_visible = True
        self.highlight_notes: set[str] = set()  # 선택한 음절에 연결된 음표 (민트 테두리)
        self.current_notes: set[str] = set()  # 재생 중인 음절에 연결된 음표 (글자 강조)
        # STEP 4: 가사 표시 방식(lyrics / pron / lyrics_pron / lyrics_ipa / all)과 멜리스마 강조.
        # vocal_groups는 VocalDetailController가 넣는 음절별 판정(amp.lyrics.vocal.VocalGroup). 비어 있으면 연결만 보고 그린다
        self.display_mode = "lyrics"
        self.melisma_emphasis = True
        self.vocal_groups: dict = {}
        self.melisma_chains: list[tuple[list[str], str]] = []  # (음표 id, 모양: user / auto / uncertain)
        self.labels_hidden = 0  # 마지막으로 그릴 때 겹쳐서 생략한 글상자 수 (시험·상태 표시용)
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
        """음절의 주 음표에는 글자를(여러 음절이 한 음표면 이어 붙임: '사랑'), 멜리스마로 이어지는 음표에는 '–',
        주 음표로 올라가는 접근음에는 '·'을 붙인다. 연결은 가사 쪽(alignment)에서 읽는다."""
        from amp.lyrics.linking import main_note_id

        ly = self.doc.lyrics
        text, al = ly.text, ly.alignment
        first_of: dict[str, str] = {}
        approach: set[str] = set()
        chains: list[tuple[list[str], str]] = []
        notes_by_id = {nid: item.note for nid, item in self.items_by_id.items()}
        for sid, links in al.links.items():
            if sid not in text.index:
                continue
            notes = sorted((notes_by_id[lk.note_id] for lk in links if lk.note_id in notes_by_id),
                           key=lambda n: (n.start, n.id))
            if notes:
                first_of[sid] = main_note_id(links, notes_by_id) or notes[0].id
                approach.update(lk.note_id for lk in links if lk.role == "approach" and lk.note_id != first_of[sid])
                if len(notes) > 1 and not self.vocal_groups:
                    chains.append(([n.id for n in notes], "auto"))
        for sid, g in self.vocal_groups.items():  # STEP 4: 멜리스마 판정 결과로 잇는다
            ids = [n for n in g.note_ids if n in notes_by_id]
            if len(ids) < 2:
                continue
            if g.is_melisma:
                chains.append((ids, "user" if g.source == "user" else "auto"))
            elif g.kind == "uncertain" and not g.user_kind:
                chains.append((ids, "uncertain"))
        self.melisma_chains = chains
        pron = ly.pronunciation
        for nid, item in self.items_by_id.items():
            sids = sorted((x for x in al.note_to_syllables.get(nid, ()) if x in text.index), key=text.order.get)
            firsts = [x for x in sids if first_of.get(x) == nid]
            if firsts:
                item.set_label("".join(text.syllable(x).text for x in firsts))
                ps = [pron.get(x) for x in firsts]
                item.label_pron = "".join(pron_marked(e) for e in ps)  # 상태 기호(↻ ✓ ?)를 붙인다
                item.label_ipa = " ".join(pron_marked(e, ipa=True) for e in ps)
            else:
                item.label_pron = item.label_ipa = ""
                if nid in approach:
                    item.set_label("·")
                elif sids:
                    item.set_label("–")
                else:
                    item.set_label("")

    def set_display(self, mode: str | None = None, melisma_emphasis: bool | None = None) -> None:
        if mode is not None:
            self.display_mode = mode
        if melisma_emphasis is not None:
            self.melisma_emphasis = melisma_emphasis
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)

    def set_vocal_groups(self, groups: dict) -> None:
        self.vocal_groups = dict(groups)
        self._refresh_labels()
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)

    def label_lines(self, item: NoteItem) -> list[tuple[str, str]]:
        """음표 아래 글상자 줄 [(종류, 글)]. 이어지는 음표('–')·접근음('·')은 한 줄만."""
        if not item.label:
            return []
        if not item.label_pron and not item.label_ipa:
            return [("lyric", item.label)]
        mode = self.display_mode
        lines = []
        if mode in ("lyrics", "lyrics_pron", "lyrics_ipa", "all"):
            lines.append(("lyric", item.label))
        if mode in ("pron", "lyrics_pron", "all"):
            lines.append(("pron", f"[{item.label_pron}]"))
        if mode in ("lyrics_ipa", "all"):
            lines.append(("ipa", f"/{item.label_ipa}/"))
        return lines

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

    def set_reference(self, notes, mode: str | None = None, preview: bool | None = None) -> None:
        self.reference_notes = list(notes)
        if mode is not None:
            self.reference_mode = mode
        if preview is not None:
            self.reference_preview = preview
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

    def _draw_reference(self, painter: QPainter, rect: QRectF) -> None:
        if not self.reference_notes or self.reference_mode == "auto":
            return
        m = self.mapper
        left, right = rect.left(), rect.right()
        t0, t1 = m.x_to_time(left), m.x_to_time(right)
        diff = self.reference_mode == "diff"
        fill = QColor(240, 236, 246, 40)
        edge = QPen(QColor(240, 236, 246, 170), 1.2, Qt.PenStyle.DotLine if self.reference_preview else Qt.PenStyle.DashLine)
        warn = QPen(theme.LYRIC_WARN, 2.0)
        marks = {"high": "▲", "low": "▼", "missing": "✕", "uncertain": "?"}
        for a, b, pitch, flag in self.reference_notes:
            if b < t0 or a > t1:
                continue
            r = QRectF(m.time_to_x(a), m.pitch_to_y(pitch) + 1, max(m.time_to_x(b) - m.time_to_x(a), 2.0), m.row_height - 2)
            painter.fillRect(r, fill)
            painter.setPen(warn if diff and flag in ("high", "low", "missing") else edge)
            painter.drawRect(r)
            if diff and flag in marks and r.width() >= 10:
                painter.setPen(theme.LYRIC_WARN if flag != "uncertain" else theme.TEXT_DIM)
                painter.drawText(QRectF(r.left() + 1, r.top() - 1, 12, r.height() + 2), Qt.AlignmentFlag.AlignCenter, marks[flag])

    def set_model_layers(self, layers: dict[str, dict] | None) -> None:
        self.model_layers = dict(layers or {})
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer | QGraphicsScene.SceneLayer.ForegroundLayer)

    def set_model_visibility(self, **flags: bool) -> None:
        for k, v in flags.items():
            if k in self.model_show:
                self.model_show[k] = bool(v)
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer | QGraphicsScene.SceneLayer.ForegroundLayer)

    def _model_pen(self, role: str, width: float = 1.4) -> QPen:
        pen = QPen(theme.MODEL_A if role == "a" else theme.MODEL_B, width,
                   Qt.PenStyle.DashLine if role == "a" else Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        return pen

    def _draw_model_notes(self, painter: QPainter, rect: QRectF) -> None:
        m = self.mapper
        t0, t1 = m.x_to_time(rect.left()), m.x_to_time(rect.right())
        half = m.row_height / 2
        painter.setFont(theme.mono_font(8))
        for role in ("a", "b"):
            layer = self.model_layers.get(role)
            if not layer or not self.model_show.get(f"{role}_notes", True):
                continue
            color = theme.MODEL_A if role == "a" else theme.MODEL_B
            fill = QColor(color)
            fill.setAlpha(34)
            pen = self._model_pen(role)
            for a, b, pitch in layer.get("notes", ()):
                if b < t0 or a > t1:
                    continue
                y = m.pitch_to_y(pitch) + (0.5 if role == "a" else half)
                r = QRectF(m.time_to_x(a), y, max(m.time_to_x(b) - m.time_to_x(a), 2.0), half - 0.5)
                painter.fillRect(r, fill)
                painter.setPen(pen)
                painter.drawRect(r)
                if r.width() >= 14 and m.row_height >= 12:
                    painter.setPen(color)
                    painter.drawText(QRectF(r.left() + 1, r.top() - 1, 10, half + 1), Qt.AlignmentFlag.AlignCenter, role.upper())

    def _draw_model_curves(self, painter: QPainter, rect: QRectF) -> None:
        m = self.mapper
        t0, t1 = m.x_to_time(rect.left()), m.x_to_time(rect.right())
        for role in ("a", "b"):
            layer = self.model_layers.get(role)
            c = layer.get("curve") if layer else None
            if c is None or not self.model_show.get(f"{role}_curve", True):
                continue
            i0 = max(0, int((t0 - c.start_time) * c.frame_rate) - 1)
            i1 = min(len(c.f0_hz), int((t1 - c.start_time) * c.frame_rate) + 2)
            if i1 <= i0:
                continue
            f = np.asarray(c.f0_hz[i0:i1], dtype=float)
            conf = np.asarray(c.confidence[i0:i1], dtype=float)
            ok = np.isfinite(f) & (f > 0) & (conf >= 0.05)
            midi = 69 + 12 * np.log2(np.where(ok, f, 440.0) / 440.0)
            xs = (c.start_time + np.arange(i0, i1) / c.frame_rate) * m.px_per_sec
            ys = (PITCH_MAX - midi) * m.row_height + m.row_height / 2
            path = QPainterPath()
            down = False
            for x, y, good in zip(xs.tolist(), ys.tolist(), ok.tolist()):
                if not good:
                    down = False
                    continue
                if down:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
                    down = True
            painter.setPen(self._model_pen(role, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

    def set_loop_region(self, region) -> None:
        self.loop_region = tuple(region) if region else None
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

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
        if self.loop_region is not None:  # 반복 구간: 옅은 바이올렛 + 양 끝 점선
            a, b = self.loop_region
            x0, x1 = max(left, m.time_to_x(a)), min(right, m.time_to_x(b))
            if x1 > x0:
                painter.fillRect(QRectF(x0, rect.top(), x1 - x0, rect.height()), QColor(173, 150, 245, 26))
            pen = QPen(theme.ACCENT, 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for t in (a, b):
                x = m.time_to_x(t)
                if left <= x <= right:
                    painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))

        self._draw_reference(painter, rect)  # STEP 5: 참조 악보 (격자 위, 자동 채보 음표 아래)
        if self.model_layers:
            self._draw_model_notes(painter, rect)  # STEP 5: 모델 비교용 A/B 자동 음표 (작업본 음표 아래)
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
        """가사·발음·IPA는 음표 아래 글상자로 쓴다. 화면에 보이는 음표만 그리고, 앞 글상자와 겹치면 생략한다.
        멜리스마는 음표를 민트 선으로 잇고 아래에 괄호선을 긋는다(사용자 확정: 굵은 실선, 자동: 실선, 판정 불확실: 점선)."""
        if self.model_layers:
            self._draw_model_curves(painter, rect)  # A/B 피치 곡선은 음표 위에 그린다 (가려지지 않게)
        if not self.lyrics_visible:
            return
        grow = QRectF(rect).adjusted(-200, -40, 200, 40)
        visible = [i for i in self.items(grow) if isinstance(i, NoteItem) and i.isVisible()]
        if not visible:
            return
        on_screen = {i.note.id for i in visible}
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
        line_h = fm.height() + 1
        n_lines = {"lyrics": 1, "pron": 1, "lyrics_pron": 2, "lyrics_ipa": 2, "all": 3}.get(self.display_mode, 1)
        for chain, style in self.melisma_chains:
            if not on_screen.intersection(chain) or (style == "uncertain" and not self.melisma_emphasis):
                continue
            pts = [self.items_by_id[n].rect() for n in chain if n in self.items_by_id]
            if len(pts) < 2:
                continue
            width = {"user": 2.0, "auto": 1.4, "uncertain": 1.0}[style] if self.melisma_emphasis else 1.2
            pen = QPen(theme.LYRIC_CURRENT if style != "uncertain" else theme.TEXT_DIM, width)
            if style == "uncertain":
                pen.setStyle(Qt.PenStyle.DotLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(QPointF(a.right(), a.center().y()), QPointF(b.left(), b.center().y()))
            if self.melisma_emphasis:  # 괄호선: 시작~끝 음표 아래, 글상자 밑
                y = max(r.bottom() for r in pts) + line_h * n_lines + 4
                x0, x1 = min(r.left() for r in pts), max(r.right() for r in pts)
                painter.drawLine(QPointF(x0, y), QPointF(x1, y))
                painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y))
                painter.drawLine(QPointF(x1, y - 4), QPointF(x1, y))
        colors = {"lyric": theme.TEXT, "pron": theme.ACCENT, "ipa": theme.TEXT_DIM}
        active: list[QRectF] = []  # 이미 그린 글상자 중 오른쪽 끝이 아직 지나지 않은 것
        hidden = 0
        for item in sorted(visible, key=lambda i: i.rect().left()):
            lines = self.label_lines(item)
            if not lines:
                continue
            r = item.rect()
            w = max(max(fm.horizontalAdvance(t) for _k, t in lines) + 6, 12)
            # IPA 줄 아래 여백: p͈ 같은 아래 결합 기호가 p 꼬리와 겹치거나 글상자 끝에 잘리지 않게
            pad = IPA_PAD if lines[-1][0] == "ipa" else 0.0
            box = QRectF(r.left(), r.bottom() + 1, w, line_h * len(lines) + pad)
            active = [x for x in active if x.right() > box.left()]
            if any(x.intersects(box) for x in active):
                hidden += 1  # 앞 글상자와 겹친다 (확대하면 보인다). 음높이가 가까운 이웃 음표의 글상자도 본다
                continue
            active.append(box)
            current = item.note.id in self.current_notes
            bg = QColor(theme.LYRIC_CURRENT if current else theme.SURFACE_RAISED)
            bg.setAlpha(235 if current else 215)
            painter.fillRect(box, bg)
            for k, (kind, t) in enumerate(lines):
                painter.setPen(theme.BG if current else colors[kind])
                align = Qt.AlignmentFlag.AlignHCenter | (Qt.AlignmentFlag.AlignTop if kind == "ipa" else Qt.AlignmentFlag.AlignVCenter)
                painter.drawText(QRectF(box.left(), box.top() + k * line_h, box.width(), line_h + (pad if kind == "ipa" else 0)),
                                 align, t)
        self.labels_hidden = hidden
