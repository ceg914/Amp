"""프로젝트 사용자 발음 사전 창 (v0.7.0 Feature H).

이 창은 항목 목록을 고칠 뿐 저장하지 않는다. '적용'을 누르면 바뀐 항목 목록을 돌려주고, 호출한 쪽이 실행 취소 가능한
가사 편집으로 문서에 넣는다(→ 프로젝트 lyrics.json). 파일 가져오기·내보내기는 다른 프로젝트와 사전을 나눌 때 쓴다.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from amp.lyrics.model import PronDictEntry
from amp.lyrics.pronunciation import dictionary as D

COLUMNS = ("원문", "범위", "발음", "IPA", "언어")
LANGS = (("", "자동"), ("ko", "한국어"), ("ja", "일본어"), ("en", "영어"))
HINT = ("이 프로젝트에서만 쓰는 사전입니다. 자동 발음을 만들 때 먼저 쓰며, 음절에서 직접 고친 발음은 바꾸지 않습니다. "
        "단어 항목은 발음 음절 수가 가사 음절 수와 같을 때만 적용됩니다 (음절을 '.'으로 나눠 적을 수 있음). "
        "사전 값도 가사 글자에서 정한 발음이며 실제 부른 발음을 들은 것이 아닙니다.")


class PronDictionaryDialog(QDialog):
    def __init__(self, entries, parent=None, start_dir: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("발음 사전 (이 프로젝트)")
        self.setMinimumWidth(620)
        self.start_dir = start_dir
        self._d = D.PronDictionary.of(entries)
        lay = QVBoxLayout(self)
        hint = QLabel(HINT)
        hint.setWordWrap(True)
        hint.setObjectName("secondary")
        lay.addWidget(hint)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table)

        add = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("원문 (예: 마음, love)")
        self.scope_combo = QComboBox()
        for k in D.SCOPES:
            self.scope_combo.addItem(D.SCOPE_LABELS[k], k)
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("발음 (예: 마흠, 러.브)")
        self.ipa_edit = QLineEdit()
        self.ipa_edit.setPlaceholderText("IPA (선택)")
        self.lang_combo = QComboBox()
        for k, label in LANGS:
            self.lang_combo.addItem(label, k)
        self.btn_add = QPushButton("추가·바꾸기")
        for w_ in (self.key_edit, self.scope_combo, self.text_edit, self.ipa_edit, self.lang_combo, self.btn_add):
            add.addWidget(w_)
        lay.addLayout(add)

        row = QHBoxLayout()
        self.btn_remove = QPushButton("고른 항목 빼기")
        self.btn_import = QPushButton("파일에서 가져오기…")
        self.btn_export = QPushButton("파일로 내보내기…")
        for b in (self.btn_remove, self.btn_import, self.btn_export):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("적용")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        lay.addWidget(self.buttons)

        self.btn_add.clicked.connect(self._add)
        self.btn_remove.clicked.connect(self._remove)
        self.btn_import.clicked.connect(self._import)
        self.btn_export.clicked.connect(self._export)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.table.itemSelectionChanged.connect(self._load_selected)
        self._fill()

    def entries(self) -> tuple[PronDictEntry, ...]:
        return self._d.as_tuple()

    def _fill(self) -> None:
        self.table.setRowCount(0)
        for e in self._d.entries.values():
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate((e.key, D.SCOPE_LABELS.get(e.scope, e.scope), e.text, e.ipa, dict(LANGS).get(e.language, e.language))):
                it = QTableWidgetItem(v)
                it.setData(256, (e.scope, e.key))
                self.table.setItem(r, c, it)
        self.btn_remove.setEnabled(bool(self._d))
        self.btn_export.setEnabled(bool(self._d))

    def _load_selected(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if len(rows) != 1:
            return
        scope, key = self.table.item(rows.pop(), 0).data(256)
        e = self._d.entries.get((scope, key))
        if e is None:
            return
        self.key_edit.setText(e.key)
        self.scope_combo.setCurrentIndex(max(0, self.scope_combo.findData(e.scope)))
        self.text_edit.setText(e.text)
        self.ipa_edit.setText(e.ipa)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(e.language)))

    def add_entry(self, key: str, text: str, scope: str = "word", ipa: str = "", language: str = "") -> bool:
        key, text = key.strip(), text.strip()
        if not key or not text or scope not in D.SCOPES:
            return False
        self._d = self._d.add(PronDictEntry(key, text, ipa.strip(), language, scope))
        self._fill()
        return True

    def _add(self) -> None:
        if not self.add_entry(self.key_edit.text(), self.text_edit.text(), self.scope_combo.currentData(),
                              self.ipa_edit.text(), self.lang_combo.currentData() or ""):
            QMessageBox.information(self, "발음 사전", "원문과 발음을 모두 적어 주세요.")

    def remove_rows(self, rows) -> int:
        keys = [self.table.item(r, 0).data(256) for r in sorted(set(rows)) if self.table.item(r, 0) is not None]
        for scope, key in keys:
            self._d = self._d.remove(scope, key)
        self._fill()
        return len(keys)

    def _remove(self) -> None:
        self.remove_rows(i.row() for i in self.table.selectedItems())

    def import_file(self, path: str) -> int:
        """사전 파일 항목을 더한다(같은 키는 파일 값으로). 더한 수. 형식이 다르면 0."""
        other = D.load(path)
        self._d = self._d.merged(other)
        self._fill()
        return len(other)

    def _import(self) -> None:
        path, _f = QFileDialog.getOpenFileName(self, "발음 사전 가져오기", self.start_dir,
                                               f"SONATLAS 발음 사전 (*{D.FILE_SUFFIX} *.json)")
        if not path:
            return
        n = self.import_file(path)
        if not n:
            QMessageBox.warning(self, "발음 사전", "읽을 수 있는 사전 항목이 없습니다 (SONATLAS 발음 사전 파일이 아니거나 비어 있음).")

    def _export(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "발음 사전 내보내기",
                                               os.path.join(self.start_dir, "pronunciation" + D.FILE_SUFFIX),
                                               f"SONATLAS 발음 사전 (*{D.FILE_SUFFIX})")
        if not path:
            return
        try:
            D.save(path, self._d)
        except OSError as e:
            QMessageBox.warning(self, "발음 사전", f"저장하지 못했습니다: {e}")
