"""참조 악보 비교 패널과 컨트롤러 (STEP 5): 악보 가져오기, 정렬(오프셋·템포·자동·확정), 음정 비교, 피아노 롤 표시.

가져온 악보는 자동 채보·사용자 음표와 따로 둔다(덮어쓰지 않음). 정렬은 확정 전까지 미리보기(점선)다.
음정 비교는 가창 분석 보조이며 악보와 다른 음정을 실수로 판정하지 않는다.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone

from PySide6.QtCore import QObject, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from amp.app import theme
from amp.compare import intonation as INT
from amp.compare.align import auto_align
from amp.compare.reference import ScoreAlignment
from amp.compare.score_io import build_reference, guess_vocal_part, read_score, sha256_bytes
from amp.core.pitch import note_name

MODES = {"auto": "자동 채보만", "ref": "참조 악보만", "both": "둘 다", "diff": "차이 강조"}
SCORE_FILTER = "악보 (*.mid *.midi *.musicxml *.xml *.mxl);;MIDI (*.mid *.midi);;MusicXML (*.musicxml *.xml *.mxl)"


class Histogram(QWidget):
    """음정 오차 분포 (−100~+100cent, 10cent 칸). 막대 높이 = 프레임 수 비율."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.bins: list[int] = []
        self.setMinimumHeight(70)

    def set_bins(self, bins) -> None:
        self.bins = list(bins)
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.SURFACE_RAISED)
        if not self.bins or max(self.bins) == 0:
            p.setPen(theme.TEXT_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "분포 없음")
            return
        w, h = self.width(), self.height() - 14
        n = len(self.bins)
        bw = w / n
        top = max(self.bins)
        for i, v in enumerate(self.bins):
            bh = h * v / top
            center = i == n // 2
            p.fillRect(QRectF(i * bw + 1, h - bh, bw - 2, bh), theme.PITCH_CURVE if center else theme.ACCENT)
        p.setPen(theme.TEXT_DIM)
        p.setFont(theme.mono_font(9))
        for label, frac in (("−100", 0.0), ("0", 0.5), ("+100", 1.0)):
            x = frac * (w - 30)
            p.drawText(QRectF(x, h, 30, 14), Qt.AlignmentFlag.AlignCenter, label)


class ComparePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.import_btn = QPushButton("참조 악보 가져오기…")
        self.import_btn.setProperty("primary", True)
        lay.addWidget(self.import_btn)
        self.info = QLabel("가져온 악보가 없습니다. MIDI(.mid)·MusicXML(.musicxml/.xml/.mxl)을 가져와 자동 채보·실제 피치와 비교합니다.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        g = QGridLayout()
        self.part = QComboBox()
        g.addWidget(QLabel("성부"), 0, 0)
        g.addWidget(self.part, 0, 1, 1, 2)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-600, 600)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.05)
        self.offset.setSuffix(" s")
        self.offset.setKeyboardTracking(False)
        self.offset.setFont(theme.mono_font())
        self.tempo = QDoubleSpinBox()
        self.tempo.setRange(50, 200)
        self.tempo.setDecimals(1)
        self.tempo.setSuffix(" %")
        self.tempo.setKeyboardTracking(False)
        self.tempo.setFont(theme.mono_font())
        self.tempo.setToolTip("악보 템포 배율 (100% = 악보에 적힌 빠르기). 바꾸면 자동 정렬의 구간 대응은 지워집니다.")
        g.addWidget(QLabel("시작 오프셋"), 1, 0)
        g.addWidget(self.offset, 1, 1, 1, 2)
        g.addWidget(QLabel("템포"), 2, 0)
        g.addWidget(self.tempo, 2, 1, 1, 2)
        self.auto_btn = QPushButton("자동 정렬")
        self.confirm_btn = QPushButton("정렬 확정")
        self.clear_btn = QPushButton("악보 지우기")
        g.addWidget(self.auto_btn, 3, 0)
        g.addWidget(self.confirm_btn, 3, 1)
        g.addWidget(self.clear_btn, 3, 2)
        self.mode = QComboBox()
        for k, v in MODES.items():
            self.mode.addItem(v, k)
        self.mode.setCurrentIndex(2)
        g.addWidget(QLabel("표시"), 4, 0)
        g.addWidget(self.mode, 4, 1, 1, 2)
        lay.addLayout(g)
        self.align_info = QLabel("")
        self.align_info.setWordWrap(True)
        self.align_info.setProperty("role", "secondary")
        lay.addWidget(self.align_info)
        title = QLabel("음정 비교 (가창 분석 보조)")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        self.summary = QLabel("—")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.hist = Histogram()
        lay.addWidget(self.hist)
        self.caveat = QLabel(INT.IntonationReport().caveat)
        self.caveat.setWordWrap(True)
        self.caveat.setProperty("role", "secondary")
        lay.addWidget(self.caveat)
        lay.addWidget(QLabel("오차가 큰 음표 (누르면 이동)"))
        self.worst = QListWidget()
        self.worst.setMinimumHeight(120)
        lay.addWidget(self.worst)
        self.detail = QLabel("선택한 음표와 겹치는 악보 음표의 통계가 여기에 나옵니다.")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.detail)
        lay.addStretch(1)


class CompareController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.panel = ComparePanel()
        self.report: INT.IntonationReport | None = None
        self.mode = "both"
        self._seq = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.recompute)
        p = self.panel
        p.import_btn.clicked.connect(self.import_dialog)
        p.auto_btn.clicked.connect(self.auto_align)
        p.confirm_btn.clicked.connect(self.confirm)
        p.clear_btn.clicked.connect(self.clear)
        p.offset.valueChanged.connect(self._on_manual)
        p.tempo.valueChanged.connect(self._on_manual)
        p.mode.currentIndexChanged.connect(lambda _i: self.set_mode(p.mode.currentData()))
        p.part.activated.connect(self._on_part)
        p.worst.itemClicked.connect(self._on_worst)
        doc = window.doc
        doc.referenceChanged.connect(self._on_reference_changed)
        doc.projectReset.connect(self._on_reference_changed)
        doc.analysisChanged.connect(self.schedule)
        doc.notesReplaced.connect(lambda *_: self.schedule())
        self.act_import = QAction("참조 악보 가져오기 (MIDI·MusicXML)…", window)
        self.act_import.triggered.connect(self.import_dialog)
        self.mode_group = QActionGroup(window)
        self.mode_actions = {}
        for k, v in MODES.items():
            a = QAction(v, window, checkable=True)
            a.triggered.connect(lambda _c=False, k=k: self.set_mode(k))
            self.mode_group.addAction(a)
            self.mode_actions[k] = a
        self.mode_actions["both"].setChecked(True)
        self._sync_panel()

    def build_menu(self, project_menu, view_menu) -> None:
        project_menu.insertAction(project_menu.actions()[3] if len(project_menu.actions()) > 3 else None, self.act_import)
        m = view_menu.addMenu("참조 악보 표시")
        for a in self.mode_actions.values():
            m.addAction(a)

    @property
    def ref(self):
        return self.w.doc.project.reference

    # --- 가져오기 --------------------------------------------------------------------

    def import_dialog(self, path: str | None = None) -> None:
        w = self.w
        folder = w.session.folder
        if folder is None:
            w.statusBar().showMessage("프로젝트를 먼저 여세요.", 4000)
            return
        if not path:
            start = str(w.settings.value("import_score_dir", "") or "")
            path, _ = QFileDialog.getOpenFileName(w, "참조 악보 가져오기", start, SCORE_FILTER)
            if not path:
                return
        w.settings.setValue("import_score_dir", os.path.dirname(path))
        handle = w.runner.submit("악보 읽기", lambda p, t: read_score(path))
        handle.finished.connect(lambda r: self._choose_part(path, *r))

    def _choose_part(self, path: str, parsed, data: bytes) -> None:
        from amp.app.dialogs import ScoreImportDialog

        w = self.w
        dlg = ScoreImportDialog(os.path.basename(path), parsed.parts, guess_vocal_part(parsed), parsed.warnings, w)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.part_id():
            return
        folder = w.session.folder
        folder.paths.scores.mkdir(exist_ok=True)
        sha = sha256_bytes(data)
        name = os.path.basename(path)
        dst = folder.paths.scores / name
        if dst.exists() and sha256_bytes(dst.read_bytes()) != sha:
            stem, ext = os.path.splitext(name)
            dst = folder.paths.scores / f"{stem}-{sha[:8]}{ext}"
        if not dst.exists():
            shutil.copy2(path, dst)
        ref = build_reference(parsed, dlg.part_id(), name, sha, stored_path=f"scores/{dst.name}",
                              imported=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        w.doc.set_reference(ref, "참조 악보 가져오기")
        if w.doc.project.analysis is not None and w.doc.project.analysis.pitch_curve is not None:
            self.auto_align()  # 미리보기로 시도. 확정은 사용자가 한다

    def _on_part(self, _i: int) -> None:
        ref = self.ref
        pid = self.panel.part.currentData()
        if ref is None or pid == ref.part_id or not ref.stored_path:
            return
        path = self.w.session.folder.root / ref.stored_path
        try:
            parsed, data = read_score(str(path))
        except Exception as e:  # noqa: BLE001
            self.w._show_error("성부 바꾸기 실패", str(e))
            return
        new = build_reference(parsed, pid, ref.source_name, ref.source_sha256, ref.stored_path, ref.imported)
        self.w.doc.set_reference(new, "참조 악보 성부 바꾸기")

    def clear(self) -> None:
        if self.ref is not None:
            self.w.doc.set_reference(None, "참조 악보 지우기")

    # --- 정렬 ------------------------------------------------------------------------

    def auto_align(self) -> None:
        ref = self.ref
        an = self.w.doc.project.analysis
        if ref is None or an is None or an.pitch_curve is None:
            self.w.statusBar().showMessage("자동 정렬에는 참조 악보와 피치 분석 결과가 필요합니다.", 4000)
            return
        notes, curve, dur = list(ref.notes), an.pitch_curve, an.duration
        self.panel.align_info.setText("자동 정렬 중…")
        handle = self.w.runner.submit("악보 자동 정렬", lambda p, t: auto_align(notes, curve, dur))

        def done(al: ScoreAlignment) -> None:
            cur = self.ref
            if cur is None or cur.source_sha256 != ref.source_sha256 or cur.part_id != ref.part_id:
                return
            self.w.doc.set_reference(cur.with_alignment(al), "악보 자동 정렬 (미리보기)")

        handle.finished.connect(done)

    def confirm(self) -> None:
        ref = self.ref
        if ref is None:
            return
        from dataclasses import replace

        self.w.doc.set_reference(ref.with_alignment(replace(ref.alignment, confirmed=True)), "악보 정렬 확정")

    def _on_manual(self, *_a) -> None:
        ref = self.ref
        if ref is None or self._syncing:
            return
        p = self.panel
        al = ref.alignment
        scale = 100.0 / p.tempo.value()
        if abs(scale - al.scale) > 1e-6:
            new = ScoreAlignment(offset=p.offset.value(), scale=scale, method="manual")
        else:
            new = al.nudged(p.offset.value() - al.offset)
        self.w.doc.set_reference(ref.with_alignment(new), "악보 정렬 수동 조절")

    # --- 표시·계산 ---------------------------------------------------------------------

    _syncing = False

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            return
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        i = self.panel.mode.findData(mode)
        if self.panel.mode.currentIndex() != i:
            self.panel.mode.blockSignals(True)
            self.panel.mode.setCurrentIndex(i)
            self.panel.mode.blockSignals(False)
        roll = self.w.view.roll
        roll.set_notes_visible(mode != "ref" and self.w.act_show_notes.isChecked())
        self._push_roll()

    def schedule(self) -> None:
        self._timer.start()

    def _on_reference_changed(self) -> None:
        self._sync_panel()
        self.schedule()

    def recompute(self) -> None:
        ref = self.ref
        an = self.w.doc.project.analysis
        if ref is None or an is None or an.pitch_curve is None:
            self.report = None
            self._show_report()
            return
        notes = list(self.w.doc.score.notes())
        curve, dur = an.pitch_curve, an.duration
        self._seq += 1
        seq = self._seq
        handle = self.w.runner.submit("음정 비교", lambda p, t: INT.analyse(ref, curve, notes, dur))

        def done(rep) -> None:
            if seq == self._seq:
                self.report = rep
                self._show_report()

        handle.finished.connect(done)

    def flags(self) -> dict[str, str]:
        out = {}
        if self.report is None:
            return out
        for n in self.report.notes:
            if n.status == "missing":
                out[n.ref_id] = "missing"
            elif n.status == "uncertain":
                out[n.ref_id] = "uncertain"
            elif n.median_cents is not None and n.median_cents > 50:
                out[n.ref_id] = "high"
            elif n.median_cents is not None and n.median_cents < -50:
                out[n.ref_id] = "low"
        return out

    def _push_roll(self) -> None:
        ref = self.ref
        roll = self.w.view.roll
        if ref is None:
            roll.set_reference([], self.mode, False)
            return
        fl = self.flags()
        items = [(a, b, n.pitch, fl.get(n.id, "")) for n, a, b in ref.aligned_notes()]
        roll.set_reference(items, self.mode, not ref.alignment.confirmed)

    def _sync_panel(self) -> None:
        p = self.panel
        ref = self.ref
        has = ref is not None
        for wdg in (p.part, p.offset, p.tempo, p.auto_btn, p.confirm_btn, p.clear_btn):
            wdg.setEnabled(has)
        self._syncing = True
        try:
            p.part.clear()
            if not has:
                p.info.setText("가져온 악보가 없습니다. MIDI(.mid)·MusicXML(.musicxml/.xml/.mxl)을 가져와 자동 채보·실제 피치와 비교합니다.")
                p.align_info.setText("")
                return
            for part in ref.parts:
                p.part.addItem(f"{part.name} ({part.notes})", part.id)
            p.part.setCurrentIndex(max(0, p.part.findData(ref.part_id)))
            lyr = "가사 있음" if ref.has_lyrics else "가사 없음"
            p.info.setText(f"<b>{ref.source_name}</b> · {ref.kind.upper()} · 음표 {len(ref.notes)}개 · {lyr}"
                           + ("<br>" + "<br>".join(ref.warnings) if ref.warnings else ""))
            al = ref.alignment
            p.offset.setValue(al.offset)
            p.tempo.setValue(100.0 / al.scale if al.scale else 100.0)
            state = "확정" if al.confirmed else "미리보기 (확정 전)"
            q = al.quality or {}
            text = f"정렬: {state} · {'자동' if al.method == 'auto' else '수동'}"
            if al.warp:
                text += f" · 구간 대응점 {len(al.warp)}개"
            if "match" in q:
                text += f" · 악보 음과 맞는 프레임 {q['match'] * 100:.0f}% (우연 수준 {q.get('match_null', 0) * 100:.0f}%)"
            if q.get("uncertain"):
                text += f"<br>⚠ {q.get('reason', '정렬이 불확실합니다.')}"
            p.align_info.setText(text)
            p.confirm_btn.setEnabled(not al.confirmed)
        finally:
            self._syncing = False
        self._push_roll()

    def _show_report(self) -> None:
        p = self.panel
        rep = self.report
        p.worst.clear()
        if rep is None:
            p.summary.setText("—" if self.ref is None else "피치 분석 결과가 있어야 음정을 비교합니다.")
            p.hist.set_bins([])
            self._push_roll()
            return
        w50 = "—" if rep.within50_frames is None else f"{rep.within50_frames * 100:.0f}%"
        lines = [f"계산한 음표 {rep.scored}개 · 불확실 {rep.uncertain} · 누락 의심 {rep.missing} · 정렬 범위 밖 {rep.not_scored}",
                 f"±50cent 안 프레임 {w50} (유효 프레임 기준)"]
        for a, b, kind, c in rep.sustained[:5]:
            lines.append(f"{'계속 높게' if kind == 'high' else '계속 낮게'} 부른 구간 {a:.1f}~{b:.1f}초 (평균 {c:+.0f}cent)")
        if rep.extra:
            lines.append(f"악보에 없는 유성 구간(추가 의심) {len(rep.extra)}곳")
        if not rep.confirmed:
            lines.append("정렬을 확정하지 않아 모든 결과가 '불확실'입니다.")
        p.summary.setText("<br>".join(lines))
        p.hist.set_bins(rep.histogram)
        ranked = sorted((n for n in rep.notes if n.median_cents is not None), key=lambda n: -abs(n.median_cents))[:15]
        for n in ranked:
            it = QListWidgetItem(f"{n.start:7.2f}s  {note_name(n.pitch):>4}  {n.median_cents:+6.0f}c  "
                                 f"(90%: {n.abs_q_cents:.0f}c)" + ("  불확실" if n.status == "uncertain" else ""))
            it.setFont(theme.mono_font())
            it.setData(256, n.ref_id)
            p.worst.addItem(it)
        for n in rep.notes:
            if n.status == "missing":
                it = QListWidgetItem(f"{n.start:7.2f}s  {note_name(n.pitch):>4}  누락 의심")
                it.setFont(theme.mono_font())
                it.setForeground(QColor(theme.LYRIC_WARN))
                it.setData(256, n.ref_id)
                p.worst.addItem(it)
        self._push_roll()
        self.show_selected()

    def _on_worst(self, item) -> None:
        rep = self.report
        n = rep.note(item.data(256)) if rep else None
        if n is not None:
            self.w.reveal_time(n.start)
            self.w.seek(n.start)
            self._detail(n)

    def show_selected(self) -> None:
        rep = self.report
        if rep is None:
            return
        ids = self.w.view.roll.selected_ids()
        sel = [self.w.doc.score.get(i) for i in ids if i in self.w.doc.score]
        if not sel:
            return
        a = min(n.start for n in sel)
        b = max(n.end for n in sel)
        best = max((n for n in rep.notes if n.end > a and n.start < b), default=None,
                   key=lambda n: min(n.end, b) - max(n.start, a))
        if best is not None:
            self._detail(best)

    def _detail(self, n) -> None:
        st = {"scored": "계산함", "uncertain": "불확실", "missing": "누락 의심", "not_scored": "계산 안 함"}[n.status]

        def f(v, unit="", fmt="{:+.0f}"):
            return "—" if v is None else fmt.format(v) + unit

        lines = [f"<b>악보 음 {note_name(n.pitch)} (MIDI {n.pitch})</b> · {n.start:.2f}~{n.end:.2f}초 · {st}",
                 f"중앙 오차 {f(n.median_cents, 'cent')} · |오차| 90% {f(n.abs_q_cents, 'cent', '{:.0f}')} · "
                 f"±50cent 안 {f(None if n.within50 is None else n.within50 * 100, '%', '{:.0f}')}",
                 f"목표 도달 {f(None if n.reach_s is None else n.reach_s * 1000, 'ms', '{:.0f}')} · "
                 f"음표 시작 차이 {f(None if n.onset_diff is None else n.onset_diff * 1000, 'ms')} · "
                 f"끝 차이 {f(None if n.offset_diff is None else n.offset_diff * 1000, 'ms')}",
                 f"유효 프레임 {n.valid_ratio * 100:.0f}% · 다른 음(±6반음 밖) {n.far_ratio * 100:.0f}%"]
        if n.vibrato_cents is not None:
            lines.append(f"비브라토 {n.vibrato_hz:.1f}Hz, 폭 ±{n.vibrato_cents:.0f}cent")
        lines += list(n.notes)
        self.panel.detail.setText("<br>".join(lines))
