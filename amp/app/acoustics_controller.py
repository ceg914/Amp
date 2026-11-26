"""v0.7.0 '보컬 정밀 → 음향 분석' (Feature C·D·E·F·G 화면).

- 분석: 지금 분석 소스의 pYIN 피치(보컬 분석 캐시) + 음향 지표 → 비브라토·호흡·프레이즈·음높이 변화 사건·발성 경계
- 목록을 누르면 그 구간을 반복 구간으로 정하고 이동한다 (반복 재생으로 확인)
- 호흡 후보: 확정·제외 (다시 분석해도 남음). 프레이즈는 그 결과로 다시 나눈다
- 발성 경계: 음표에 자동으로 적용하지 않는다. [교정 제안으로 보내기]를 눌러야 '교정 제안' 탭에 대기 제안으로 들어간다
- 다중 보컬 결과가 있으면 성부 사이 음정·주선율/화음 후보(추정)를 보여 준다
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.engine.acoustics import breath as B
from amp.engine.acoustics import onsets as O
from amp.engine.acoustics import phonation as P


def _secondary(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("role", "secondary")
    return lab


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setHeaderLabels(headers)
    t.setRootIsDecorated(False)
    t.setAlternatingRowColors(True)
    return t


class AcousticPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(_secondary("비브라토·호흡·프레이즈·음높이 변화·발성 경계를 피치 곡선과 음향 지표로 분석합니다. "
                                 "음표·가사·피치 곡선은 바꾸지 않습니다. 분리한 보컬은 음색·배음이 바뀌었을 수 있어, "
                                 "이 지표만으로 흉성·두성 등 발성 유형을 정하지 않습니다."))
        row = QHBoxLayout()
        self.run = QPushButton("분석")
        self.run.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row.addWidget(self.run)
        row.addStretch(1)
        lay.addLayout(row)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.vibrato = _tree(["시작", "끝", "속도 Hz", "폭(한쪽)", "봉우리 간", "주기", "상태"])
        self.tabs.addTab(self.vibrato, "비브라토")
        bw = QWidget()
        bl = QVBoxLayout(bw)
        bl.setContentsMargins(0, 0, 0, 0)
        self.breaths = _tree(["호흡 후보", "시작", "끝", "음량 dB", "평탄도", "점수", "상태"])
        bl.addWidget(self.breaths, 1)
        brow = QHBoxLayout()
        self.b_confirm = QPushButton("확정")
        self.b_reject = QPushButton("제외")
        self.b_reset = QPushButton("자동으로")
        for b in (self.b_confirm, self.b_reject, self.b_reset):
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            brow.addWidget(b)
        brow.addStretch(1)
        bl.addLayout(brow)
        self.phrases = _tree(["프레이즈", "시작", "끝", "길이", "앞 쉼", "음절", "초당 음절", "호흡"])
        bl.addWidget(self.phrases, 1)
        self.tabs.addTab(bw, "호흡·프레이즈")
        ew = QWidget()
        el = QVBoxLayout(ew)
        el.setContentsMargins(0, 0, 0, 0)
        self.kind = QComboBox()
        self.kind.addItem("모든 종류", "")
        for k, label in P.KINDS.items():
            self.kind.addItem(label, k)
        el.addWidget(self.kind)
        self.events = _tree(["종류", "시작", "끝", "이동(센트)", "안정화(초)", "근거"])
        el.addWidget(self.events, 1)
        self.tabs.addTab(ew, "음높이 변화")
        gw = QWidget()
        gl = QVBoxLayout(gw)
        gl.setContentsMargins(0, 0, 0, 0)
        self.bounds = _tree(["음표", *[O.FIELD_LABELS[f] for f in O.FIELDS]])
        gl.addWidget(self.bounds, 1)
        grow = QHBoxLayout()
        self.start_field = QComboBox()
        for f in ("voiced_onset", "stable_pitch_onset", "acoustic_onset"):
            self.start_field.addItem(f"시작 기준: {O.FIELD_LABELS[f]}", f)
        self.to_suggest = QPushButton("교정 제안으로 보내기")
        self.to_suggest.setToolTip("음표를 바로 바꾸지 않습니다. '교정 제안' 탭에 대기 제안으로 넣습니다.")
        grow.addWidget(self.start_field)
        grow.addWidget(self.to_suggest)
        grow.addStretch(1)
        gl.addLayout(grow)
        self.tabs.addTab(gw, "발성 경계")
        self.harmony = QPlainTextEdit()
        self.harmony.setReadOnly(True)
        self.tabs.addTab(self.harmony, "화음 관계")
        lay.addWidget(self.tabs, 1)


class AcousticsController(QObject):
    def __init__(self, window, page: AcousticPage) -> None:
        super().__init__(window)
        self.w = window
        self.page = p = page
        self.res = None
        p.run.clicked.connect(self.run)
        p.kind.currentIndexChanged.connect(lambda _i: self._fill_events())
        for tree in (p.vibrato, p.breaths, p.phrases, p.events):
            tree.itemClicked.connect(self._go)
        p.b_confirm.clicked.connect(lambda: self._breath_status("confirmed"))
        p.b_reject.clicked.connect(lambda: self._breath_status("rejected"))
        p.b_reset.clicked.connect(lambda: self._breath_status(None))
        p.to_suggest.clicked.connect(self.send_suggestions)
        window.session.busyChanged.connect(lambda _b: self.update_actions())

    def reset(self) -> None:
        self.res = None
        self.refresh()

    def refresh(self) -> None:
        from amp.engine.acoustics import run as AR

        folder = self.w.session.folder
        loaded = False
        if folder is not None and self.res is None:
            self.res = AR.load(folder.paths.analysis)
            loaded = self.res is not None
        self._fill()
        self.update_actions()
        vocal = getattr(self.w, "vocal", None)
        if loaded and vocal is not None and self.vibrato_spans():
            vocal.refresh()

    def update_actions(self) -> None:
        s = self.w.session
        self.page.run.setEnabled(s.folder is not None and s.audio is not None and not s.busy)
        has = self.res is not None
        for b in (self.page.b_confirm, self.page.b_reject, self.page.b_reset):
            b.setEnabled(has and not s.busy)
        self.page.to_suggest.setEnabled(has and bool(self.res and self.res.boundaries) and not s.busy)

    def snapshot(self):
        from amp.engine.acoustics.run import Snapshot

        w = self.w
        notes = list(w.doc.score.notes())
        starts = list(w.session.syllable_starts()) or None
        mel = []
        for sid, g in (getattr(w.vocal, "groups", {}) or {}).items():
            if getattr(g, "is_melisma", False) and g.start is not None and g.end is not None:
                mel.append((float(g.start), float(g.end), sid))
        return Snapshot(notes, starts, mel)

    def run(self) -> None:
        handle = self.w.session.run_acoustics(self.snapshot())
        if handle is None:
            return
        self.page.summary.setPlainText("분석 중…")
        handle.finished.connect(self._on_done)
        handle.failed.connect(lambda msg, _tb: self.page.summary.setPlainText(f"분석하지 못했습니다: {msg}"))
        self.update_actions()

    def _on_done(self, res) -> None:
        self.res = res
        self._fill()
        self.update_actions()
        vocal = getattr(self.w, "vocal", None)
        if vocal is not None:
            vocal.refresh()  # 측정한 비브라토 구간을 멜리스마 판정에 넘긴다

    def vibrato_spans(self) -> list[tuple[float, float]]:
        """멜리스마 판정에 넘길 비브라토 구간. 소스가 바뀐(stale) 결과나 '불확실' 구간은 쓰지 않는다."""
        from amp.core import result_meta as RM

        r = self.res
        if r is None:
            return []
        m = r.meta.get("vibrato")
        if m is not None and m.status == RM.STALE:
            return []
        return [(s.start, s.end) for s in r.vibrato if s.status == "completed"]

    # --- 표시 -----------------------------------------------------------------------

    def _fill(self) -> None:
        if hasattr(self.w, "sync_analysis_lane"):
            self.w.sync_analysis_lane()  # v0.7.0 분석 레인도 같은 결과로
        p = self.page
        for t in (p.vibrato, p.breaths, p.phrases, p.events, p.bounds):
            t.clear()
        r = self.res
        if r is None:
            p.summary.setPlainText("아직 분석하지 않았습니다.")
            p.harmony.setPlainText("")
            return
        from amp.core.result_meta import STATUS_LABELS

        vs, bs = r.vibrato_summary, r.breath_summary
        lines = [f"비브라토 {vs.get('completed', 0)}구간 (불확실 포함 {vs.get('segments', 0)})"
                 + (f" · 평균 {vs['rate_hz']}Hz, 한쪽 {vs['extent_cents']}센트 (봉우리 간 {vs['extent_pp_cents']})"
                    if vs.get("rate_hz") else ""),
                 f"호흡 후보 {bs.get('breath_candidates', 0)} · 확정 {bs.get('breath_confirmed', 0)} · 프레이즈 {bs.get('phrases', 0)}"
                 + (f" · 초당 음절(중앙) {bs['syllables_per_s_median']}" if bs.get("syllables_per_s_median") else ""),
                 f"음높이 변화 사건 {sum(1 for e in r.events if e.kind != 'voicing')} · 발성 경계 {len(r.boundaries)}음"]
        stale = [k for k, m in r.meta.items() if m.status != "completed"]
        if stale:
            lines.append("상태: " + ", ".join(f"{k} {STATUS_LABELS.get(r.meta[k].status, r.meta[k].status)}" for k in stale))
        p.summary.setPlainText("\n".join(lines))
        for s in r.vibrato:
            it = QTreeWidgetItem([f"{s.start:.2f}", f"{s.end:.2f}", f"{s.rate_hz:.2f}", f"{s.extent_cents:.0f}",
                                  f"{s.extent_pp_cents:.0f}", f"{s.cycles:.1f}", "완료" if s.status == "completed" else "불확실"])
            it.setData(0, Qt.ItemDataRole.UserRole, (s.start, s.end))
            it.setToolTip(6, s.reason)
            p.vibrato.addTopLevelItem(it)
        labels = {"candidate": "후보", "confirmed": "확정", "rejected": "제외"}
        for b in r.breaths:
            it = QTreeWidgetItem([b.breath_id, f"{b.start:.2f}", f"{b.end:.2f}", f"{b.level_db:.0f}",
                                  f"{b.flatness:.2f}", f"{b.score:.2f}", labels.get(b.status, b.status)])
            it.setData(0, Qt.ItemDataRole.UserRole, (b.start, b.end))
            it.setData(1, Qt.ItemDataRole.UserRole, b.breath_id)
            p.breaths.addTopLevelItem(it)
        for ph in r.phrases:
            it = QTreeWidgetItem([ph.phrase_id, f"{ph.start:.2f}", f"{ph.end:.2f}", f"{ph.duration:.2f}",
                                  "" if ph.gap_before_s is None else f"{ph.gap_before_s:.2f}",
                                  "" if ph.syllables is None else str(ph.syllables),
                                  "" if ph.syllables_per_s is None else f"{ph.syllables_per_s:.2f}", ",".join(ph.breath_ids)])
            it.setData(0, Qt.ItemDataRole.UserRole, (ph.start, ph.end))
            p.phrases.addTopLevelItem(it)
        self._fill_events()
        f = lambda v: "미측정" if v is None else f"{v:.3f}"  # noqa: E731
        for nb in r.boundaries:
            p.bounds.addTopLevelItem(QTreeWidgetItem([f"{nb.note_start:.2f}s", *[f(getattr(nb, k)) for k in O.FIELDS]]))
        self._fill_harmony()

    def _fill_events(self) -> None:
        p = self.page
        p.events.clear()
        if self.res is None:
            return
        want = p.kind.currentData()
        for e in self.res.events:
            if (want and e.kind != want) or (not want and e.kind == "voicing"):
                continue
            it = QTreeWidgetItem([P.KINDS.get(e.kind, e.kind), f"{e.start:.2f}", f"{e.end:.2f}",
                                  "" if e.delta_cents is None else f"{e.delta_cents:+.0f}",
                                  "" if e.settle_s is None else f"{e.settle_s:.2f}", e.evidence])
            it.setData(0, Qt.ItemDataRole.UserRole, (e.start, max(e.end, e.start + 0.2)))
            p.events.addTopLevelItem(it)

    def _fill_harmony(self) -> None:
        from amp.engine.acoustics import harmony as H

        vs = self.w.precision.voices.vs
        if vs is None or len(vs.voices) < 2:
            self.page.harmony.setPlainText("다중 보컬 분석 결과가 있으면 성부 사이 음정과 주선율·화음 후보(추정)를 보여 줍니다.")
            return
        ov = H.overlaps(vs)
        roles = H.estimate_roles(vs, list(self.w.session.syllable_starts()) or None)
        names = {v.voice_track_id: v.display_name for v in vs.voices}
        from amp.engine.voices import ROLES

        lines = ["동시 발성 구간의 음정 (초):"] + [f"  {k}: {v}" for k, v in H.interval_summary(ov).items()]
        lines.append("역할 추정 (근거 점수, 확정 아님 — 가장 큰 소리를 주선율로 가정하지 않음):")
        for r in roles:
            lines.append(f"  {names.get(r.voice_track_id, r.voice_track_id)}: {ROLES.get(r.role, r.role)}"
                         f"{' (사용자 지정)' if r.user_set else ''} · 점수 {r.score:.2f} · " + "; ".join(r.evidence))
        self.page.harmony.setPlainText("\n".join(lines))

    # --- 동작 -----------------------------------------------------------------------

    def _go(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        span = item.data(0, Qt.ItemDataRole.UserRole)
        if not span:
            return
        a, b = span
        lc = self.w.listening
        lc.set_region(float(a), float(b), pad=True)
        self.w.seek(max(0.0, float(a) - lc.padding))

    def _breath_status(self, status) -> None:
        from amp.engine.acoustics import run as AR

        folder = self.w.session.folder
        items = self.page.breaths.selectedItems()
        if folder is None or self.res is None or not items:
            return
        ov = dict(self.res.overrides or {})
        st = dict(ov.get("breath_status") or {})
        for it in items:
            bid = it.data(1, Qt.ItemDataRole.UserRole)
            if status is None:
                st.pop(bid, None)
            else:
                st[bid] = status
        ov["breath_status"] = st
        res = AR.save_overrides(folder.paths.analysis, ov)
        if res is not None:
            res.meta = self.res.meta
            self.res = res
            self._fill()

    def send_suggestions(self) -> None:
        from datetime import datetime, timezone

        from amp.compare import suggestions as S

        w = self.w
        if self.res is None:
            return
        new = O.suggestions(w.doc.project, self.res.boundaries, self.page.start_field.currentData())
        old = w.doc.project.suggestions or S.SuggestionState()
        have = {s.suggestion_id for s in old.items}
        add = [s for s in new if s.suggestion_id not in have]
        if not add:
            w.statusBar().showMessage("새로 보낼 경계 제안이 없습니다 (차이가 작거나 이미 보냈거나 직접 고친 음표).", 6000)
            return
        state = S.SuggestionState(tuple(sorted(old.items + tuple(add), key=lambda s: (s.start_time, s.suggestion_type))),
                                  datetime.now(timezone.utc).isoformat(timespec="seconds"))
        w.doc.set_suggestions(state, "발성 경계 제안 추가")
        w.statusBar().showMessage(f"교정 제안 {len(add)}개를 더했습니다. 음표는 바꾸지 않았습니다.", 6000)
        w.show_side_panel(w.suggest.panel)
