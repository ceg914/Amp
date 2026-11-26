"""분석 선택 창 (v0.7.0 명세 16.4)과 차례 실행.

창에서 분석할 기능을 고르면 앞 단계(의존 관계)를 더하고, 할 수 없는 기능은 이유와 함께 뺀 계획을 바로 보여 준다
(engine.analysis_plan). '실행'하면 기존 기능을 차례로 부른다: 보컬 분석 → 다중 피치 → 음향 분석 → 가사 인식 → 가사 정렬.
한 단계가 끝나야(작업이 끝나고 결과가 바뀌어야) 다음으로 가고, 실패·취소되면 거기서 멈춘다. 이미 반영된 결과와 사용자 편집은 그대로 둔다.
'전체'를 골라도 실험 기능과 설치되지 않은 외부 모델은 넣지 않는다. 사용자가 저장한 분석 설정은 바꾸지 않는다.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from amp.engine import analysis_plan as AP


def context_for(window, statuses=None) -> AP.Context:
    if statuses is None:
        try:
            statuses = window.session.model_registry(save=False)
        except Exception:  # noqa: BLE001 - 등록 정보를 못 읽어도 신호 처리 기능은 고를 수 있게
            statuses = []
    usable = {s.model_id for s in statuses if s.usable} | {"pyin"}
    return AP.Context(usable, not window.doc.lyrics.text.is_empty, window.doc.project.analysis is not None,
                      bool(getattr(window, "lite", False)))


class AnalysisDialog(QDialog):
    def __init__(self, window, ctx: AP.Context) -> None:
        super().__init__(window)
        self.ctx = ctx
        self.setWindowTitle("분석 선택")
        self.resize(720, 560)
        lay = QVBoxLayout(self)
        hint = QLabel("할 분석을 고르세요. 먼저 필요한 분석은 함께 하고, 지금 할 수 없는 분석은 이유를 보여 줍니다. "
                      "'권장'은 실험 기능과 설치되지 않은 모델을 빼고 고릅니다. 이미 고친 음표·가사·발음은 바꾸지 않습니다.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["분석", "할 수 있는지", "먼저 필요", "참고"])
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.items: dict[str, QTreeWidgetItem] = {}
        for f in AP.FEATURES:
            ok, why = AP.available(f, ctx)
            needs = ", ".join("가사" if d == "lyrics" else AP.BY_KEY[d].label for d in f.needs) or "—"
            if f.uses:
                needs += " (있으면 좋음: " + ", ".join(AP.BY_KEY[d].label for d in f.uses) + ")"
            note = ("실험 · " if f.experimental else "") + ("자동 · " if f.step == "auto" else "") + f.note
            it = QTreeWidgetItem([f.label, "가능" if ok else f"불가: {why}", needs, note.strip(" ·")])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Unchecked)
            if not ok:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            it.setData(0, Qt.ItemDataRole.UserRole, f.key)
            self.tree.addTopLevelItem(it)
            self.items[f.key] = it
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.btn_default = QPushButton("권장 (전체)")
        self.btn_none = QPushButton("모두 끄기")
        row.addWidget(self.btn_default)
        row.addWidget(self.btn_none)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel("실행 계획"))
        self.plan_text = QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setMaximumHeight(170)
        lay.addWidget(self.plan_text)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("실행")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        lay.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.btn_default.clicked.connect(lambda: self.set_selected(AP.defaults(ctx)))
        self.btn_none.clicked.connect(lambda: self.set_selected(()))
        self.tree.itemChanged.connect(lambda _i, _c: self._update())
        self.set_selected(AP.defaults(ctx))

    def selected(self) -> list[str]:
        return [k for k, it in self.items.items() if it.checkState(0) == Qt.CheckState.Checked]

    def set_selected(self, keys) -> None:
        keys = set(keys)
        self.tree.blockSignals(True)
        for k, it in self.items.items():
            it.setCheckState(0, Qt.CheckState.Checked if k in keys and it.flags() & Qt.ItemFlag.ItemIsEnabled
                             else Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self._update()

    def plan(self) -> AP.Plan:
        return AP.resolve(self.selected(), self.ctx)

    def _update(self) -> None:
        plan = self.plan()
        self.plan_text.setPlainText(plan.summary())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(plan.steps or plan.chosen))


class AnalysisChain(QObject):
    """계획의 단계를 차례로 실행한다. 단계마다 '시작했는가'와 '결과가 바뀌었는가'를 보고 다음으로 간다."""

    finished = Signal(str)  # 요약

    POLL_MS = 150

    def __init__(self, window, plan: AP.Plan) -> None:
        super().__init__(window)
        self.w = window
        self.plan = plan
        self.steps = list(plan.steps)
        # 가사 인식과 정렬을 함께 고르면 인식 단계의 '인식한 뒤 바로 정렬'로 이어서 한다 (정렬을 두 번 하지 않음)
        self.recognize_then_align = "asr" in self.steps and "align" in self.steps
        if self.recognize_then_align:
            self.steps.remove("align")
        self.done: list[str] = []
        self.failed: str = ""
        self.current: str | None = None
        self._before = None
        self._saw_busy = False
        self._restore_align_after: bool | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)

    # --- 단계별 시작·결과 확인 -------------------------------------------------------

    def _snapshot(self, step: str):
        w = self.w
        if step == "vocal":
            return w.doc.project.analysis
        if step == "multipitch":
            return w.precision.voices.vs
        if step == "acoustics":
            return w.precision.acoustics.res
        if step == "asr":
            return (w.doc.lyrics.recognition, w.doc.lyrics.alignment)
        if step == "align":
            return w.doc.lyrics.alignment
        return None

    def _start(self, step: str) -> bool:
        w = self.w
        if step == "vocal":
            s = w.inspector.analysis.settings()
            if not self.plan.separation:
                s = replace(s, separation="none")  # 이번 실행만 (저장한 설정은 그대로)
            if not w._ensure_model_files(s, force=False):
                return False
            return w.session.run_analysis(s) is not None
        if step == "multipitch":
            w.precision.voices.run()
        elif step == "acoustics":
            w.precision.acoustics.run()
        elif step == "asr":
            box = w.lyrics_panel.align_after
            self._restore_align_after = box.isChecked()
            box.setChecked(self.recognize_then_align)
            w.lyrics.recognize()
        elif step == "align":
            w.lyrics.align("all")
        return w.session.busy

    def start(self) -> None:
        self._next()

    def _next(self) -> None:
        if self._restore_align_after is not None and self.current == "asr":
            self.w.lyrics_panel.align_after.setChecked(self._restore_align_after)
            self._restore_align_after = None
        if not self.steps:
            self._end()
            return
        step = self.steps.pop(0)
        self.current = step
        self._before = self._snapshot(step)
        self.w.statusBar().showMessage(f"분석 선택: {AP.STEP_LABELS[step]} 중… (남은 단계 {len(self.steps)})", 0)
        if self.w.session.busy:
            self.failed = f"{AP.STEP_LABELS[step]}: 다른 작업이 돌고 있어 시작하지 못했습니다."
            self._end()
            return
        started = self._start(step)
        if not started and self._snapshot(step) is self._before:
            self.failed = f"{AP.STEP_LABELS[step]}: 시작하지 못했습니다."
            self._end()
            return
        self._saw_busy = self.w.session.busy
        self._timer.start()

    def _poll(self) -> None:
        w = self.w
        if w.session.busy:
            self._saw_busy = True
            return
        if self.current == "asr" and getattr(w.lyrics, "chain", None) is not None:
            return  # 인식 → 정렬 → 연결 연속 작업이 아직 끝나지 않았다
        self._timer.stop()
        changed = self._snapshot(self.current) is not self._before and self._snapshot(self.current) != self._before
        if not changed:
            self.failed = f"{AP.STEP_LABELS[self.current]}: 결과가 바뀌지 않았습니다 (실패·취소했거나 반영하지 않음)."
            self._end()
            return
        self.done.append(self.current)
        self._next()

    def _end(self) -> None:
        self._timer.stop()
        if self._restore_align_after is not None:
            self.w.lyrics_panel.align_after.setChecked(self._restore_align_after)
            self._restore_align_after = None
        parts = [f"끝난 단계: {', '.join(AP.STEP_LABELS[s] for s in self.done) or '없음'}"]
        if self.failed:
            parts.append("멈춤 — " + self.failed)
        if self.plan.skipped:
            parts.append("뺀 분석: " + ", ".join(AP.BY_KEY[k].label for k in self.plan.skipped))
        text = " · ".join(parts)
        self.w.statusBar().showMessage(text, 15000)
        self.current = None
        self.finished.emit(text)
