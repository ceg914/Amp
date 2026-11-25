"""품질 탭 (v0.6): 정답 없는 지표, 검토 대시보드(숫자를 누르면 검토 목록을 그 종류로 거름), 참조 악보 비교, Hybrid 출처,
모델별 같은 지표. 정답이 없으면 정확도라고 하지 않는다.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from amp.compare import quality as Q
from amp.compare import review as R

log = logging.getLogger(__name__)


class QualityPanel(QScrollArea):
    countClicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("panel")
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(10, 10, 10, 10)
        self.setWidget(body)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setProperty("role", "secondary")
        self.lay.addWidget(self.note)
        self.lay.addWidget(self._section("검토 현황 (숫자를 누르면 검토 탭에서 그 항목만)"))
        self.grid = QGridLayout()
        self.lay.addLayout(self.grid)
        self.count_buttons: dict[str, QPushButton] = {}
        self.lay.addWidget(self._section("지표 (정답 없이 계산)"))
        self.indicators = QLabel()
        self.indicators.setWordWrap(True)
        self.lay.addWidget(self.indicators)
        self.lay.addWidget(self._section("참조 악보 비교"))
        self.reference = QLabel()
        self.reference.setWordWrap(True)
        self.lay.addWidget(self.reference)
        self.lay.addWidget(self._section("보컬 소스"))
        self.sources = QLabel()
        self.sources.setWordWrap(True)
        self.lay.addWidget(self.sources)
        self.lay.addWidget(self._section("모델별 지표 (모델 비교 결과가 있을 때)"))
        self.models = QLabel()
        self.models.setWordWrap(True)
        self.lay.addWidget(self.models)
        self.refresh_btn = QPushButton("다시 계산")
        self.lay.addWidget(self.refresh_btn)
        self.lay.addStretch(1)

    def _section(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "section")
        return lab

    def set_counts(self, counts: dict) -> None:
        for i, (key, (label, n, _kinds)) in enumerate(counts.items()):
            b = self.count_buttons.get(key)
            if b is None:
                b = QPushButton()
                b.clicked.connect(lambda _c=False, k=key: self.countClicked.emit(k))
                self.count_buttons[key] = b
                self.grid.addWidget(b, i // 2, i % 2)
            b.setText(f"{label}  {n}")
            b.setToolTip("검토 탭에서 이 항목만 보기" if _kinds else "검토 탭 전체 보기")


class QualityController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.panel = QualityPanel()
        self.report: Q.QualityReport | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self.refresh)
        self.panel.countClicked.connect(self.open_count)
        self.panel.refresh_btn.clicked.connect(self.refresh)
        doc = window.doc
        for sig in (doc.notesReplaced, doc.analysisChanged, doc.lyricsChanged, doc.projectReset, doc.reviewChanged,
                    doc.suggestionsChanged, doc.hybridChanged, doc.referenceChanged):
            sig.connect(lambda *_: self._timer.start())

    def _review_rows(self):
        rv = self.w.review
        return R.merge_states(rv.all_items(), self.w.doc.project.review)

    def refresh(self) -> None:
        w = self.w
        labels = {}
        try:
            store = w.session.model_store()
            labels = {e.key: e.label for e in store.entries()} if store is not None else {}
        except OSError:
            pass
        rep = Q.assess(w.doc.project, self._review_rows(), labels)
        self.report = rep
        p = self.panel
        p.note.setText(" ".join(rep.notes))
        p.set_counts(rep.counts)
        p.indicators.setText("<br>".join(f"{k}: <b>{v}</b>" for k, v in rep.indicators.items()) or "분석 결과가 없습니다.")
        if rep.reference:
            r = rep.reference
            lines = [f"정답 음표 {r['ref_notes']}개 · 추정 {r['est_notes']}개"]
            for k, v in r.items():
                if k.startswith("onset±"):
                    lines.append(f"시작 {k[5:]}: P {v['precision']:.2f} · R {v['recall']:.2f} · F1 {v['f1']:.2f} · 일치 {v['matched']} · "
                                 f"시작 오차 {v['onset_mae_ms']}ms · 끝 오차 {v['offset_mae_ms']}ms")
            pc = r.get("pitch_curve_vs_reference")
            if pc:
                lines.append(f"피치 곡선 vs 악보 음높이: 중앙 오차 {pc['median_abs_cents']}센트, ±50센트 안 "
                             f"{pc['within_50c']:.0%} ({pc['notes_with_pitch']}음)")
            lines.append(f"<i>{r['caveat']}</i>")
            p.reference.setText("<br>".join(lines))
        else:
            p.reference.setText("참조 악보가 없거나 정렬하지 않았습니다. 정확도를 계산하지 않습니다.")
        if rep.sources:
            p.sources.setText("Hybrid: " + "<br>".join(f"{a:.2f}–{b:.2f}s {m}" for a, b, m in rep.sources))
        else:
            sep = rep.indicators.get("분리 소스", "—")
            p.sources.setText(f"곡 전체 한 소스: {sep}")
        outs = getattr(w.model_compare, "outputs", {}) or {}
        rows = Q.model_table(outs, {k: w.model_compare.label(k) for k in outs})
        p.models.setText("<br>".join(
            f"{r['model']}: 음표 {r['notes']} · 유성 {r.get('voiced', '—')} · 신뢰도 중앙값 {r.get('conf_median', '—')} · "
            f"채보 {r['seconds']}초" + (f" · GPU 최대 {r['gpu_peak_mb']}MB" if r.get("gpu_peak_mb") else "")
            for r in rows) or "모델 비교 결과가 없습니다 (분석 > 분리 모델 비교 실행).")

    def open_count(self, key: str) -> None:
        if self.report is None:
            return
        label, _n, kinds = self.report.counts.get(key, ("", 0, ()))
        w = self.w
        w.review.set_kind_filter(kinds or None, label)
        w.show_side_panel(w.review.panel)
