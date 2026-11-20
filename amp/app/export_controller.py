"""채보·가사 내보내기 대화상자와 컨트롤러 (STEP 5): USTX, MIDI, MusicXML, JSON, CSV, LRC.

기존 오디오 내보내기(WAV/MP3)는 그대로 둔다. 사용자가 고친 음표·가사가 기본으로 들어가고(사용자 수정 포함),
자동 분석 원본이나 참조 악보를 고를 수도 있다. 내보낸 뒤 형식상 담지 못한 정보를 알려 준다.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from amp.export import formats as F
from amp.export import ustx as U
from amp.export.common import ExportOptions

FORMATS = {
    "ustx": ("OpenUtau 프로젝트 (.ustx)", ".ustx", "OpenUtau USTX (*.ustx)"),
    "midi": ("MIDI (.mid)", ".mid", "MIDI (*.mid)"),
    "musicxml": ("MusicXML (.musicxml)", ".musicxml", "MusicXML (*.musicxml)"),
    "json": ("분석 정보 JSON (.json)", ".json", "JSON (*.json)"),
    "csv": ("음표·음절 표 CSV (.csv)", ".csv", "CSV (*.csv)"),
    "lrc": ("시간 동기화 가사 LRC (.lrc)", ".lrc", "LRC (*.lrc)"),
}
DEFAULT_NAMES = {"ustx": "transcription", "midi": "transcription", "musicxml": "transcription", "json": "analysis",
                 "csv": "notes", "lrc": "lyrics"}
NOTES_ABOUT = {
    "ustx": "OpenUtau 0.6 형식(최신 OpenUtau가 열 때 올림). 가수·phonemizer는 OpenUtau에서 고르세요. 같은 음절의 다음 음표는 '+~'(이어 부르기).",
    "midi": "음표·시각·템포·박자·가사(UTF-8 메타). 피치를 넣으면 피치 벤드(±12반음).",
    "musicxml": "16분음표 격자로 맞춘 악보(마디·붙임줄·쉼표·가사·멜리스마 이음줄). 실제 시각·피치 곡선은 담지 못합니다.",
    "json": "음표(상태·신뢰도)·가사·발음·분석 정보 전부. 피치를 넣으면 프레임별 피치 곡선 포함.",
    "csv": "음표별 표와 음절별 표(-syllables.csv) 두 파일. Excel에서 한글이 보이도록 UTF-8 BOM.",
    "lrc": "가사 줄 시작 시각. '음절 시각'을 켜면 <mm:ss.xx> 확장 표기.",
}


class DataExportDialog(QDialog):
    def __init__(self, export_dir: str, has_reference: bool, region, selection, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("채보·가사 내보내기")
        self.setMinimumWidth(560)
        self.export_dir = export_dir
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.format = QComboBox()
        for k, (label, _e, _f) in FORMATS.items():
            self.format.addItem(label, k)
        self.source = QComboBox()
        self.source.addItem("사용자 수정 포함 (지금 악보)", "user")
        self.source.addItem("자동 분석 원본", "auto")
        if has_reference:
            self.source.addItem("참조 악보 (정렬 위치)", "reference")
        self.range = QComboBox()
        self.range.addItem("전체 곡", None)
        if region is not None:
            self.range.addItem(f"반복 구간 ({region[0]:.2f}~{region[1]:.2f}초)", tuple(region))
        if selection is not None:
            self.range.addItem(f"선택 음표 구간 ({selection[0]:.2f}~{selection[1]:.2f}초)", tuple(selection))
        self.lyrics = QCheckBox("가사 포함")
        self.lyrics.setChecked(True)
        self.pitch = QComboBox()
        for k, v in U.PITCH_MODES.items():
            self.pitch.addItem(v, k)
        self.lrc_words = QCheckBox("LRC 음절 시각")
        self.path = QLineEdit()
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        form.addRow("형식", self.format)
        form.addRow("음표", self.source)
        form.addRow("구간", self.range)
        form.addRow("", self.lyrics)
        form.addRow("피치 곡선", self.pitch)
        form.addRow("", self.lrc_words)
        form.addRow("저장 위치", row)
        lay.addLayout(form)
        self.about = QLabel("")
        self.about.setWordWrap(True)
        self.about.setProperty("role", "secondary")
        lay.addWidget(self.about)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("내보내기")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.format.currentIndexChanged.connect(self._on_format)
        self._on_format()

    def _on_format(self, *_a) -> None:
        k = self.format.currentData()
        self.pitch.setEnabled(k in ("ustx", "midi", "json"))
        self.lrc_words.setEnabled(k == "lrc")
        self.lyrics.setEnabled(k not in ("lrc",))
        self.about.setText(NOTES_ABOUT[k])
        cur = self.path.text()
        ext = FORMATS[k][1]
        if not cur or os.path.dirname(cur) == self.export_dir:
            self.path.setText(os.path.join(self.export_dir, DEFAULT_NAMES[k] + ext))

    def _browse(self) -> None:
        k = self.format.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "저장 위치", self.path.text(), FORMATS[k][2])
        if path:
            if not path.lower().endswith(FORMATS[k][1]):
                path += FORMATS[k][1]
            self.path.setText(path)

    def values(self) -> tuple[str, str, ExportOptions]:
        k = self.format.currentData()
        pitch = self.pitch.currentData() if self.pitch.isEnabled() else "none"
        return k, self.path.text().strip(), ExportOptions(source=self.source.currentData(), range=self.range.currentData(),
                                                           lyrics=self.lyrics.isChecked(), pitch=pitch,
                                                           lrc_words=self.lrc_words.isChecked())


class ExportController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.act = QAction("채보·가사 내보내기 (USTX·MIDI·MusicXML·JSON·CSV·LRC)…", window)
        self.act.triggered.connect(self.dialog)
        self.last_report = None

    def selection_span(self):
        ids = self.w.view.roll.selected_ids()
        notes = [self.w.doc.score.get(i) for i in ids if i in self.w.doc.score]
        return (min(n.start for n in notes), max(n.end for n in notes) + 1e-6) if notes else None

    def dialog(self) -> None:
        w = self.w
        folder = w.session.folder
        if folder is None:
            w.statusBar().showMessage("프로젝트를 먼저 여세요.", 4000)
            return
        folder.paths.exports.mkdir(parents=True, exist_ok=True)
        dlg = DataExportDialog(str(folder.paths.exports), w.doc.project.reference is not None,
                               w.player.loop_region, self.selection_span(), w)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kind, path, opts = dlg.values()
        if not path:
            return
        if os.path.exists(path):
            ans = QMessageBox.question(w, "덮어쓰기", f"{os.path.basename(path)}이(가) 이미 있습니다. 덮어쓸까요?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.export(kind, path, opts)

    def export(self, kind: str, path: str, opts: ExportOptions):
        w = self.w
        project = w.doc.project
        name = w.session.folder.name if w.session.folder else "SONATLAS"
        durations = dict(getattr(w.vocal, "durations", {}) or {})
        groups = dict(getattr(w.vocal, "groups", {}) or {})

        def job(progress, token):
            if kind == "ustx":
                return U.export(project, path, opts, name)
            if kind == "midi":
                return F.export_midi(project, path, opts)
            if kind == "musicxml":
                return F.export_musicxml(project, path, opts, title=name)
            if kind == "json":
                extra = {"melisma": [r for g in groups.values() if (r := g.melisma_record()) is not None]}
                return F.export_json(project, path, opts, extra)
            if kind == "csv":
                return F.export_csv(project, path, opts, durations, groups)
            return F.export_lrc(project, path, opts, title=name)

        handle = w.runner.submit("채보·가사 내보내기", job)

        def done(rep) -> None:
            self.last_report = rep
            w.statusBar().showMessage(f"{rep.format} 내보냄: {os.path.basename(path)}", 8000)
            QMessageBox.information(w, "내보내기 완료", rep.summary())

        handle.finished.connect(done)
        handle.failed.connect(lambda msg, _d: w._show_error("내보내기 실패", msg))
        return handle
