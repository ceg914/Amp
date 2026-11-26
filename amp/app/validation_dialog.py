"""검증 자료 창 (v0.7.0 명세 19). 분석 → 검증 자료 만들기….

구간(10~30초)과 자료 종류를 고르고, 직접 확인해 고친 항목을 '확정'해 저장한다. 저장한 자료로 지금 결과를 비교한다.
자료는 프로젝트 validation/에만 두고 학습·전송에 쓰지 않는다. 지우기는 묻고 나서 한다.
"""

from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from amp.engine import validation as V


class ValidationDialog(QDialog):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.setWindowTitle("검증 자료")
        self.resize(640, 620)
        lay = QVBoxLayout(self)
        hint = QLabel("직접 듣고 고친 결과를 구간(10~30초)마다 '정답'으로 저장해, 나중에 다시 분석한 결과와 비교합니다. "
                      + V.POLICY + " 규칙을 고치는 데 본 자료는 독립 검증으로 쓰지 마세요.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        form = QFormLayout()
        dur = window.session.audio.duration if window.session.audio is not None else 0.0
        region = window.player.loop_region if window.player.loop_enabled else None
        a, b = region if region and V.MIN_S <= region[1] - region[0] <= V.MAX_S else \
            (max(0.0, window.current_time()), min(dur, max(0.0, window.current_time()) + 20.0))
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for s, v in ((self.start, a), (self.end, b)):
            s.setRange(0.0, max(dur, 0.0))
            s.setDecimals(2)
            s.setSuffix(" 초")
            s.setValue(v)
        row = QHBoxLayout()
        row.addWidget(self.start)
        row.addWidget(QLabel("~"))
        row.addWidget(self.end)
        form.addRow("구간", row)
        self.category = QComboBox()
        for k, label in V.CATEGORIES.items():
            self.category.addItem(label, k)
        form.addRow("자료 종류", self.category)
        self.dev = QCheckBox("이 자료를 보고 규칙을 고친 적이 있음 (개발용으로 표시)")
        form.addRow("", self.dev)
        self.note = QLineEdit()
        form.addRow("메모", self.note)
        lay.addLayout(form)
        grid = QGridLayout()
        self.items: dict[str, QCheckBox] = {}
        for i, (k, label) in enumerate(V.ITEMS.items()):
            cb = QCheckBox(label)
            self.items[k] = cb
            grid.addWidget(cb, i // 2, i % 2)
        lay.addLayout(grid)
        self.btn_save = QPushButton("지금 결과를 이 구간의 검증 자료로 저장")
        self.btn_save.setProperty("variant", "primary")
        lay.addWidget(self.btn_save)
        lay.addWidget(QLabel("저장한 검증 자료"))
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        brow = QHBoxLayout()
        self.btn_compare = QPushButton("지금 결과와 비교")
        self.btn_delete = QPushButton("지우기…")
        brow.addWidget(self.btn_compare)
        brow.addWidget(self.btn_delete)
        brow.addStretch(1)
        lay.addLayout(brow)
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setMaximumHeight(170)
        lay.addWidget(self.result)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.button(QDialogButtonBox.StandardButton.Close).setText("닫기")
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_compare.clicked.connect(self.compare_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self._reload()

    @property
    def root(self):
        return self.w.session.folder.root

    def _sources(self) -> dict:
        w = self.w
        prec = getattr(w, "precision", None)
        return {"pitch_curve": w.doc.project.analysis.pitch_curve if w.doc.project.analysis is not None else None,
                "voices": prec.voices.vs if prec is not None else None,
                "acoustics": prec.acoustics.res if prec is not None else None, "lyrics": w.doc.lyrics}

    def _reload(self) -> None:
        self.list.clear()
        for vs in V.load_all(self.root):
            it = QListWidgetItem(vs.label + (" · 개발용" if vs.used_for_development else "") + (f" · {vs.note}" if vs.note else ""))
            it.setData(256, vs.set_id)
            self.list.addItem(it)
        has = self.list.count() > 0
        self.btn_compare.setEnabled(has)
        self.btn_delete.setEnabled(has)

    def save_current(self) -> V.ValidationSet | None:
        chosen = [k for k, cb in self.items.items() if cb.isChecked()]
        src = self.w.session.folder.meta.original.sha256 if self.w.session.folder else ""
        try:
            vs = V.snapshot(self.start.value(), self.end.value(), self.category.currentData(), chosen,
                            note=self.note.text().strip(), source={"original_sha256": src}, **self._sources())
        except V.ValidationError as e:
            self.result.setPlainText(str(e))
            return None
        vs.used_for_development = self.dev.isChecked()
        V.save(self.root, vs)
        self.result.setPlainText(f"저장했습니다: {vs.label}")
        self._reload()
        return vs

    def _selected(self) -> V.ValidationSet | None:
        it = self.list.currentItem() or (self.list.item(0) if self.list.count() else None)
        if it is None:
            return None
        sid = it.data(256)
        return next((v for v in V.load_all(self.root) if v.set_id == sid), None)

    def compare_selected(self) -> dict | None:
        vs = self._selected()
        if vs is None:
            return None
        s = self._sources()
        res = V.compare(vs, pitch_curve=s["pitch_curve"], acoustics=s["acoustics"], lyrics=s["lyrics"])
        self.result.setPlainText(json.dumps(res, ensure_ascii=False, indent=1))
        return res

    def delete_selected(self) -> bool:
        vs = self._selected()
        if vs is None:
            return False
        ans = QMessageBox.question(self, "검증 자료 지우기", f"이 검증 자료를 지울까요? 되돌릴 수 없습니다.\n{vs.label}")
        if ans != QMessageBox.StandardButton.Yes:
            return False
        V.delete(self.root, vs.set_id)
        self._reload()
        return True
