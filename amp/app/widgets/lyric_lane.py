"""가사 레인: 피아노 롤 아래에서 음절을 시간 막대로 보여 주고 직접 고친다.

파형·피아노 롤과 같은 ``TimelineState`` (가로축)를 쓰므로 확대·스크롤·재생선이 함께 움직인다.

조작
- 클릭: 음절 선택 + 그 위치로 이동 / Ctrl+클릭: 선택 추가·빼기 / Shift+클릭: 범위 선택
- 막대 끌기: 음절 시간 이동 / 양 끝 끌기: 시작·끝 조절. 이웃 음절과 맞닿은 경계는 둘을 함께 움직인다(경계 조절)
- 끌 때 가까운 음표 경계(6px 안)에 붙는다. Alt를 누르면 붙지 않는다
- 더블클릭: 그 음절부터 재생 / 오른쪽 클릭: 나누기·합치기·음표 연결·반복 재생 메뉴
- 휠: 가로 스크롤 / Ctrl+휠: 가로 확대

막대 모양(색만으로 구분하지 않음)
- 실선 테두리: 모델이 시작을 검출 / 점선: 추정값(인식 단어 시각을 나눔 등) / 굵은 테두리: 사용자 지정
- 주황 점선 + '↻': 텍스트가 바뀌어 다시 정렬 필요 / 주황 '!': 정렬 신뢰도 낮음
- 흐린 글자 + 밑줄 없음: 음표 연결 없음 / 아래 점: 멜리스마 판정(민트 = 멜리스마, 회색 = 판정 불확실)
"""

from __future__ import annotations

import bisect
from enum import Enum, auto

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from amp.app import theme
from amp.app.document import ProjectDocument
from amp.app.timeline import TimelineState
from amp.lyrics import linking
from amp.lyrics import pronunciation as pron
from amp.lyrics.ops import MIN_SYLLABLE_S

EDGE_PX = 5.0
SNAP_PX = 6.0
TOUCH_S = 0.015  # 이웃 음절과 이만큼 가까우면 맞닿은 것으로 본다
LOW_CONF = 0.1  # 실제 두 곡: 0.1 미만 음절의 오차율 8~9% vs 나머지 1~2% (약한 지표)


class _Drag(Enum):
    NONE = auto()
    MOVE = auto()
    START = auto()
    END = auto()


class LyricLane(QWidget):
    syllablesSelected = Signal(list)  # 선택한 음절 id 목록
    seekRequested = Signal(float)
    playFromRequested = Signal(float)
    loopRequested = Signal(float, float)
    editFailed = Signal(str)
    # 메뉴 요청 (메인 창이 처리): 종류, 음절 id 목록
    actionRequested = Signal(str, list)

    def __init__(self, doc: ProjectDocument, timeline: TimelineState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.timeline = timeline
        self.selected: list[str] = []
        self.current: str | None = None
        self.duration = 0.0
        self.note_edges: list[float] = []  # 음표 경계 (붙이기용)
        self._playhead = 0.0
        self._drag = _Drag.NONE
        self._press_x = 0.0
        self._press_sid: str | None = None
        self._orig: dict[str, tuple[float, float]] = {}
        self._preview: dict[str, tuple[float, float]] = {}
        self._rows: list[tuple[str, float, float]] = []  # (id, 시작, 끝) 시작 순
        self._starts: list[float] = []
        # STEP 4: 음절 아래 둘째 줄(none / pron / ipa), 음절 구간 길이 표시, 멜리스마 판정
        self.sub_mode = "none"
        self.show_lengths = False
        self.vocal_groups: dict = {}
        self.setMinimumHeight(44)
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setToolTip("")
        timeline.changed.connect(self.update)
        doc.lyricsChanged.connect(self.refresh)
        doc.projectReset.connect(self.refresh)
        doc.notesReplaced.connect(lambda *_: self._refresh_note_edges())
        doc.analysisChanged.connect(self._refresh_note_edges)
        self.refresh()

    # --- 데이터 -----------------------------------------------------------------

    def refresh(self) -> None:
        ly = self.doc.lyrics
        rows = []
        for s in ly.text.syllables:
            t = ly.alignment.timing(s.id)
            if t.aligned:
                rows.append((s.id, t.start, t.end))
        rows.sort(key=lambda r: (r[1], ly.text.order[r[0]]))
        self._rows = rows
        self._starts = [r[1] for r in rows]
        live = set(ly.text.index)
        self.selected = [s for s in self.selected if s in live]
        if self.current not in live:
            self.current = None
        self._refresh_note_edges()
        self.update()

    def _refresh_note_edges(self) -> None:
        edges = set()
        for n in self.doc.score.notes():
            edges.add(round(n.start, 4))
            edges.add(round(n.end, 4))
        self.note_edges = sorted(edges)

    def select(self, ids: list[str], emit: bool = True) -> None:
        self.selected = list(dict.fromkeys(ids))
        self.update()
        if emit:
            self.syllablesSelected.emit(list(self.selected))

    def set_playhead(self, t: float) -> str | None:
        """재생 위치를 받고, 그 시각의 음절 id를 돌려준다(강조용)."""
        old_x = self.timeline.time_to_x(self._playhead)
        self._playhead = t
        sid = self.syllable_at_time(t)
        if sid != self.current:
            self.current = sid
            self.update()
        else:
            x = self.timeline.time_to_x(t)
            self.update(int(min(old_x, x)) - 3, 0, int(abs(x - old_x)) + 6, self.height())
        return sid

    def syllable_at_time(self, t: float) -> str | None:
        i = bisect.bisect_right(self._starts, t) - 1
        for j in (i, i - 1):
            if 0 <= j < len(self._rows) and self._rows[j][1] <= t < self._rows[j][2]:
                return self._rows[j][0]
        return None

    def _times(self, sid: str) -> tuple[float, float]:
        if sid in self._preview:
            return self._preview[sid]
        t = self.doc.lyrics.alignment.timing(sid)
        return t.start, t.end

    def _hit(self, x: float) -> tuple[str | None, _Drag]:
        t = self.timeline.x_to_time(x)
        best = None
        for sid, a, b in self._rows:
            xa, xb = self.timeline.time_to_x(a), self.timeline.time_to_x(b)
            if xa - EDGE_PX <= x <= xb + EDGE_PX:
                if abs(x - xa) <= EDGE_PX and (xb - xa) > 2 * EDGE_PX:
                    return sid, _Drag.START
                if abs(x - xb) <= EDGE_PX and (xb - xa) > 2 * EDGE_PX:
                    best = (sid, _Drag.END)
                    continue
                if a <= t <= b:
                    return sid, _Drag.MOVE
        return best if best else (None, _Drag.NONE)

    def _neighbors(self, sid: str) -> tuple[str | None, str | None]:
        order = self.doc.lyrics.text.order
        ids = [r[0] for r in sorted(self._rows, key=lambda r: order[r[0]])]
        i = ids.index(sid)
        return (ids[i - 1] if i > 0 else None), (ids[i + 1] if i + 1 < len(ids) else None)

    # --- 그리기 -----------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.SURFACE)
        p.setPen(QPen(theme.BORDER, 1))
        p.drawLine(0, 0, self.width(), 0)
        ly = self.doc.lyrics
        text, al = ly.text, ly.alignment
        if text.is_empty:
            p.setPen(theme.TEXT_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "가사 없음 — 오른쪽 '가사' 탭에서 입력하거나 자동 인식하세요")
            return
        font = theme.ui_font(13)
        small = theme.ui_font(10)
        mono = theme.mono_font(9)
        p.setFont(small)
        small_fm = p.fontMetrics()
        p.setFont(font)
        fm = p.fontMetrics()
        h = self.height()
        top, bot = 8.0, h - 10.0
        w = self.width()
        selected = set(self.selected)
        for sid, a0, b0 in self._rows:
            a, b = self._times(sid)
            xa, xb = self.timeline.time_to_x(a), self.timeline.time_to_x(b)
            if xb < -50 or xa > w + 50:
                continue
            t = al.timing(sid)
            syl = text.syllable(sid)
            linked = bool(al.links.get(sid))
            r = QRectF(xa, top, max(xb - xa, 3.0), bot - top)
            if sid == self.current:
                fill = QColor(theme.LYRIC_CURRENT)
            elif sid in selected:
                fill = QColor(theme.ACCENT)
            else:
                fill = QColor(theme.SURFACE_RAISED)
            p.setBrush(fill)
            if t.stale:
                pen = QPen(theme.LYRIC_WARN, 1.2, Qt.PenStyle.DashLine)
            elif t.user_edited or t.start_source in ("user", "notes"):
                pen = QPen(theme.TEXT, 2.0)
            elif t.start_source == "model":
                pen = QPen(theme.ACCENT, 1.0)
            else:
                pen = QPen(theme.TEXT_DIM, 1.0, Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawRoundedRect(r, 3, 3)
            label = syl.text
            if t.stale:
                label += "↻"
            elif t.confidence is not None and t.start_source == "model" and t.confidence < LOW_CONF:
                label += "!"
            dark = sid == self.current or sid in selected
            p.setPen(theme.BG if dark else (theme.TEXT if linked else theme.LYRIC_UNLINKED))
            sub = self._sub_text(sid) if self.sub_mode != "none" else ""
            main_r = QRectF(r.left(), r.top(), r.width(), r.height() * (0.55 if sub or self.show_lengths else 1.0))
            if r.width() >= fm.horizontalAdvance(label[:1]) + 2:
                p.drawText(main_r, Qt.AlignmentFlag.AlignCenter,
                           fm.elidedText(label, Qt.TextElideMode.ElideRight, int(r.width())))
            if sub or self.show_lengths:
                p.setFont(small)
                sub_r = QRectF(r.left() + 1, main_r.bottom() - 1, r.width() - 2, r.bottom() - main_r.bottom())
                if sub and r.width() >= 10:
                    e = ly.pronunciation.get(sid)
                    uncertain = e is not None and (e.status != "valid" or e.stale) and not (e.user_edited and not e.stale)
                    p.setPen(theme.BG if dark else (theme.LYRIC_WARN if uncertain else theme.ACCENT))
                    p.drawText(sub_r, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                               small_fm.elidedText(sub, Qt.TextElideMode.ElideRight, int(sub_r.width())))
                if self.show_lengths and t.aligned and r.width() >= 26:
                    p.setFont(mono)
                    p.setPen(theme.BG if dark else theme.TEXT_DIM)
                    p.drawText(sub_r, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, f"{(t.end - t.start) * 1000:.0f}")
                p.setFont(font)
            g = self.vocal_groups.get(sid)
            if self.vocal_groups:  # 멜리스마 표식: 판정 결과 기준 (확정·자동 멜리스마는 민트, 판정 불확실은 회색)
                n_dots = len(g.note_ids) if g is not None and (g.is_melisma or (g.kind == "uncertain" and not g.user_kind)) else 0
                color = theme.LYRIC_CURRENT if g is not None and g.is_melisma else theme.TEXT_DIM
            else:  # 판정 결과가 없으면 STEP 3처럼 연결 수로
                n_dots = len(al.links.get(sid, ())) if len(al.links.get(sid, ())) > 1 else 0
                color = theme.LYRIC_CURRENT
            if n_dots > 1:
                # 모양으로도 구분 (UI 명세 2.5): 판정 불확실 = 빈 원, 사용자 확정 멜리스마 = 사각형, 자동 멜리스마 = 채운 원
                hollow = g is not None and not g.is_melisma
                square = g is not None and g.is_melisma and g.source == "user"
                p.setPen(QPen(color, 1.0) if hollow else Qt.PenStyle.NoPen)
                p.setBrush(Qt.BrushStyle.NoBrush if hollow else color)
                for k in range(min(n_dots, 4)):
                    c = QPointF(xa + 4 + k * 5, bot + 4)
                    if square:
                        p.drawRect(QRectF(c.x() - 1.7, c.y() - 1.7, 3.4, 3.4))
                    else:
                        p.drawEllipse(c, 1.6, 1.6)
        n_un = len(ly.unaligned())
        if n_un:
            msg = f"미정렬 {n_un}"
            p.setPen(theme.LYRIC_WARN)
            p.drawText(QRectF(w - fm.horizontalAdvance(msg) - 10, 0, fm.horizontalAdvance(msg) + 8, 14),
                       Qt.AlignmentFlag.AlignRight, msg)
        x = self.timeline.time_to_x(self._playhead)
        p.setPen(QPen(theme.PLAYHEAD, 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, h))

    # --- 마우스 ---------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        sid, kind = self._hit(x)
        if event.button() == Qt.MouseButton.RightButton:
            if sid and sid not in self.selected:
                self.select([sid])
            self._context_menu(event.globalPosition().toPoint(), sid)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        mods = event.modifiers()
        if sid is None:
            self.select([])
            self.seekRequested.emit(max(0.0, self.timeline.x_to_time(x)))
            return
        if mods & Qt.KeyboardModifier.ControlModifier:
            sel = [s for s in self.selected if s != sid] if sid in self.selected else self.selected + [sid]
            self.select(sel)
            return
        if mods & Qt.KeyboardModifier.ShiftModifier and self.selected:
            order = self.doc.lyrics.text.order
            a, b = sorted((order[self.selected[0]], order[sid]))
            self.select([s.id for s in self.doc.lyrics.text.syllables[a:b + 1]])
            return
        if sid not in self.selected or kind != _Drag.MOVE:
            self.select([sid])
        self.seekRequested.emit(self._times(sid)[0])
        self._drag = kind
        self._press_x = x
        self._press_sid = sid
        ids = self.selected if kind == _Drag.MOVE else [sid]
        self._orig = {s: self._times(s) for s in ids}
        if kind in (_Drag.START, _Drag.END):  # 맞닿은 이웃도 함께 (경계 조절)
            prev, nxt = self._neighbors(sid)
            a, b = self._orig[sid]
            if kind == _Drag.START and prev and abs(self._times(prev)[1] - a) < TOUCH_S:
                self._orig[prev] = self._times(prev)
            if kind == _Drag.END and nxt and abs(self._times(nxt)[0] - b) < TOUCH_S:
                self._orig[nxt] = self._times(nxt)

    def _snap(self, t: float, no_snap: bool) -> float:
        if no_snap or not self.note_edges:
            return t
        i = bisect.bisect_left(self.note_edges, t)
        best = t
        for j in (i - 1, i):
            if 0 <= j < len(self.note_edges):
                e = self.note_edges[j]
                if abs(self.timeline.time_to_x(e) - self.timeline.time_to_x(t)) <= SNAP_PX:
                    best = e
        return best

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        if self._drag == _Drag.NONE:
            sid, kind = self._hit(x)
            self.setCursor(Qt.CursorShape.SizeHorCursor if kind in (_Drag.START, _Drag.END) else Qt.CursorShape.ArrowCursor)
            if sid:
                self.setToolTip(self._tooltip(sid))
            return
        no_snap = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        dt = (x - self._press_x) / self.timeline.px_per_sec
        sid = self._press_sid
        a0, b0 = self._orig[sid]
        prev: dict[str, tuple[float, float]] = {}
        if self._drag == _Drag.MOVE:
            lo = -min(a for a, _ in self._orig.values())
            dt = max(dt, lo)
            if self.duration > 0:
                dt = min(dt, self.duration - max(b for _, b in self._orig.values()))
            snapped = self._snap(a0 + dt, no_snap) - a0
            dt = snapped
            for s, (a, b) in self._orig.items():
                prev[s] = (a + dt, b + dt)
        elif self._drag == _Drag.START:
            na = min(max(0.0, self._snap(a0 + dt, no_snap)), b0 - MIN_SYLLABLE_S)
            prev[sid] = (na, b0)
            for s, (a, b) in self._orig.items():
                if s != sid:
                    na = max(na, a + MIN_SYLLABLE_S)
                    prev[sid] = (na, b0)
                    prev[s] = (a, na)
        elif self._drag == _Drag.END:
            nb = max(self._snap(b0 + dt, no_snap), a0 + MIN_SYLLABLE_S)
            if self.duration > 0:
                nb = min(nb, self.duration)
            prev[sid] = (a0, nb)
            for s, (a, b) in self._orig.items():
                if s != sid:
                    nb = min(nb, b - MIN_SYLLABLE_S)
                    prev[sid] = (a0, nb)
                    prev[s] = (nb, b)
        self._preview = prev
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag == _Drag.NONE:
            return
        changes = {s: v for s, v in self._preview.items() if abs(v[0] - self._orig[s][0]) > 1e-4
                   or abs(v[1] - self._orig[s][1]) > 1e-4}
        self._drag = _Drag.NONE
        self._preview = {}
        if changes:
            from amp.core.errors import EditError
            from amp.lyrics import ops

            try:
                new = ops.set_times(self.doc.lyrics, changes, duration=self.duration or None,
                                    action="음절 경계 조절" if len(changes) > 1 and len(self._orig) == 2 else "음절 시간 수정")
                self.doc.push_lyrics(new, "음절 시간 수정")
            except EditError as e:
                self.editFailed.emit(str(e))
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        sid, _ = self._hit(event.position().x())
        if sid:
            self.playFromRequested.emit(self._times(sid)[0])

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.timeline.zoomRequested.emit(1.25 if delta > 0 else 0.8, self.timeline.x_to_time(event.position().x()))
        else:
            self.timeline.scrollRequested.emit(-float(delta))
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) and self.selected:
            order = self.doc.lyrics.text.order
            rows = sorted((r[0] for r in self._rows), key=order.get)
            if rows:
                cur = self.selected[-1]
                i = rows.index(cur) if cur in rows else 0
                i = max(0, min(len(rows) - 1, i + (1 if event.key() == Qt.Key.Key_Right else -1)))
                self.select([rows[i]])
                self.seekRequested.emit(self._times(rows[i])[0])
            return
        super().keyPressEvent(event)

    def _tooltip(self, sid: str) -> str:
        from amp.lyrics.model import SOURCE_LABELS

        ly = self.doc.lyrics
        s = ly.text.syllable(sid)
        t = ly.alignment.timing(sid)
        parts = [f"'{s.text}'  {t.start:.3f}s ~ {t.end:.3f}s ({(t.end - t.start) * 1000:.0f}ms)",
                 f"시작: {SOURCE_LABELS.get(t.start_source, t.start_source)} · 끝: {SOURCE_LABELS.get(t.end_source, t.end_source)}"]
        if t.confidence is not None:
            parts.append(f"정렬 신뢰도 {t.confidence:.2f}")
        if t.stale:
            parts.append("가사가 바뀌어 다시 정렬이 필요합니다")
        kind = linking.relation_kind(ly, sid)
        parts.append({"none": "음표 연결 없음", "1:1": "음표 1개", "1:N": "음표 여러 개",
                      "N:1": "다른 음절과 한 음표를 나눠 씀", "N:M": "여러 음표·여러 음절"}[kind])
        if s.note:
            parts.append(s.note)
        return "\n".join(parts)

    def _context_menu(self, pos, sid: str | None) -> None:
        menu = QMenu(self)
        sel = list(self.selected)
        ly = self.doc.lyrics

        def add(text: str, kind: str, enabled: bool = True) -> None:
            act = menu.addAction(text)
            act.setEnabled(enabled)
            act.triggered.connect(lambda: self.actionRequested.emit(kind, list(sel)))

        one = len(sel) == 1
        add("이 구간 반복 재생", "loop", bool(sel))
        menu.addSeparator()
        add("선택한 음표로 연결 바꾸기", "link_replace", one)
        add("선택한 음표를 연결에 더하기", "link_add", one)
        add("음표와 순서대로 연결 (선택 음절 ↔ 선택 음표)", "link_sequential", bool(sel))
        n_links = len(ly.alignment.links.get(sel[0], ())) if one else 0
        first_is_approach = one and any(lk.role == "approach" for lk in ly.alignment.links.get(sel[0], ()))
        add("접근음 표시 해제 (첫 음표에도 가사 표시)" if first_is_approach else "첫 연결 음표를 접근음으로 (가사는 다음 음표에)",
            "toggle_approach", n_links >= 2)
        add("연결 확정", "confirm", bool(sel))
        add("연결 지우기", "unlink", bool(sel))
        add("연결한 음표의 시간 가져오기", "times_from_notes", bool(sel))
        menu.addSeparator()
        can_split = one and len(ly.text.syllable(sel[0]).text) >= 2 if one else False
        add("음절 나누기", "split", can_split)
        add("음절 합치기", "merge", len(sel) >= 2)
        add("선택 음절만 다시 정렬", "realign", bool(sel))
        add("시간 지우기 (미정렬로)", "clear_times", bool(sel))
        menu.addSeparator()
        decided = one and sel[0] in ly.alignment.melisma
        add("멜리스마로 확정 (선택 음표가 2개 이상이면 그 음표로)", "mel_set", one)
        add("멜리스마 아님으로 확정", "mel_not", one)
        add("선택 음표를 멜리스마에 추가", "mel_add", one)
        add("선택 음표를 멜리스마에서 빼기", "mel_remove", one)
        add("멜리스마 자동 판정으로 되돌리기", "mel_clear", bool(sel) and (decided or len(sel) > 1))
        menu.addSeparator()
        add("발음 확정 (자동 재생성에서 보호)", "pron_confirm", bool(sel))
        add("기본 발음으로 되돌리기", "pron_reset", bool(sel))
        add("발음 다시 생성 (자동 발음만)", "pron_regen", bool(sel))
        menu.exec(pos)

    def set_display(self, sub_mode: str | None = None, show_lengths: bool | None = None) -> None:
        if sub_mode is not None:
            self.sub_mode = sub_mode
        if show_lengths is not None:
            self.show_lengths = show_lengths
        self.setFixedHeight(68 if self.sub_mode != "none" or self.show_lengths else 52)
        self.update()

    def set_vocal_groups(self, groups: dict) -> None:
        self.vocal_groups = dict(groups)
        self.update()

    def _sub_text(self, sid: str) -> str:
        e = self.doc.lyrics.pronunciation.get(sid)
        if e is None:
            return ""
        return pron.marked_text(e, ipa=self.sub_mode == "ipa")  # 상태 기호(↻ ✓ ?)를 색과 함께 붙인다
