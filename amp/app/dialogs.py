"""대화상자: 새 프로젝트, 오디오 내보내기, 재분석 결과 반영."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from amp.app.widgets.track_panel import TRACK_HINTS, TRACK_LABELS
from amp.engine.merge import KEEP_EDITS, REPLACE_ALL, ScoreSummary
from amp.project.export import MP3_BITRATES, WAV_SUBTYPES, ExportOptions
from amp.project.layout import sanitize_name
from amp.project.workspace import COPY, LINK


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet("font-size: 18px; font-weight: 600;")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "secondary")
    lab.setWordWrap(True)
    return lab


@dataclass
class NewProjectChoice:
    name: str
    parent: str
    mode: str
    open_existing: str | None = None


class NewProjectDialog(QDialog):
    def __init__(self, audio_path: str, default_parent: str, existing: list[Path], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("새 프로젝트")
        self.setMinimumWidth(520)
        self._existing_choice: str | None = None
        self.name = QLineEdit(sanitize_name(Path(audio_path).stem))
        self.parent_dir = QLineEdit(default_parent)
        browse = QPushButton("변경…")
        browse.clicked.connect(self._browse)
        self.copy = QRadioButton("프로젝트 폴더에 원본 복사 (권장)")
        self.link = QRadioButton("원래 위치의 파일을 연결 (사본 없음)")
        self.copy.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.copy)
        group.addButton(self.link)

        lay = QVBoxLayout(self)
        lay.addWidget(_title("새 프로젝트 만들기"))
        lay.addWidget(_hint(f"원본: {audio_path}\n원본 파일은 수정하거나 지우지 않습니다."))
        form = QFormLayout()
        form.addRow("프로젝트 이름", self.name)
        row = QHBoxLayout()
        row.addWidget(self.parent_dir, 1)
        row.addWidget(browse)
        form.addRow("저장 위치", row)
        lay.addLayout(form)
        lay.addWidget(self.copy)
        lay.addWidget(_hint("    폴더를 옮기거나 다른 PC로 복사해도 그대로 열립니다."))
        lay.addWidget(self.link)
        lay.addWidget(_hint("    디스크를 아끼지만, 원본을 옮기면 '원본 다시 연결'이 필요합니다."))
        self.preview = _hint("")
        lay.addWidget(self.preview)
        if existing:
            box = QLabel(f"같은 음원으로 만든 프로젝트가 {len(existing)}개 있습니다: "
                         + ", ".join(p.name for p in existing[:3]))
            box.setProperty("state", "warn")
            box.setWordWrap(True)
            lay.addWidget(box)
            open_btn = QPushButton(f"기존 프로젝트 열기 ({existing[0].name})")
            open_btn.clicked.connect(lambda: self._open_existing(str(existing[0])))
            lay.addWidget(open_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("만들기")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.name.textChanged.connect(self._update_preview)
        self.parent_dir.textChanged.connect(self._update_preview)
        self._update_preview()

    def _update_preview(self) -> None:
        from amp.project.layout import unique_folder

        try:
            target = unique_folder(self.parent_dir.text(), self.name.text())
            self.preview.setText(f"만들 폴더: {target}")
        except OSError:
            self.preview.setText("")

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "프로젝트 저장 위치", self.parent_dir.text())
        if d:
            self.parent_dir.setText(d)

    def _open_existing(self, path: str) -> None:
        self._existing_choice = path
        self.accept()

    def choice(self) -> NewProjectChoice:
        return NewProjectChoice(self.name.text().strip() or "Untitled", self.parent_dir.text(),
                                COPY if self.copy.isChecked() else LINK, self._existing_choice)


@dataclass
class ExportJob:
    track: str
    path: str


class ExportDialog(QDialog):
    def __init__(self, project_name: str, available: dict[str, bool], out_dir: str, preselect: list[str],
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("오디오 내보내기")
        self.setMinimumWidth(520)
        self.project_name = project_name
        self.checks: dict[str, QCheckBox] = {}
        self.names: dict[str, QLineEdit] = {}
        lay = QVBoxLayout(self)
        lay.addWidget(_title("오디오 내보내기"))
        lay.addWidget(_hint("이미 만든 분리 결과를 형식만 바꿔 저장합니다. 보컬 분리를 다시 실행하지 않습니다."))
        form = QFormLayout()
        for track, label in TRACK_LABELS.items():
            cb = QCheckBox(f"{label} ({TRACK_HINTS[track]})")
            ok = available.get(track, False)
            cb.setEnabled(ok)
            cb.setChecked(ok and track in preselect)
            if not ok:
                cb.setToolTip("분리 결과가 없습니다. 먼저 보컬 분석을 실행하세요.")
            name = QLineEdit()
            name.setEnabled(ok)
            self.checks[track], self.names[track] = cb, name
            row = QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(name, 1)
            form.addRow(row)
        self.fmt = QComboBox()
        self.fmt.addItem("WAV", "wav")
        self.fmt.addItem("MP3", "mp3")
        self.wav = QComboBox()
        for k, v in WAV_SUBTYPES.items():
            self.wav.addItem(v, k)
        self.wav.setCurrentIndex(1)
        self.bitrate = QComboBox()
        for b in MP3_BITRATES:
            self.bitrate.addItem(f"{b} kbps", b)
        self.bitrate.setCurrentIndex(len(MP3_BITRATES) - 1)
        self.out = QLineEdit(out_dir)
        browse = QPushButton("변경…")
        browse.clicked.connect(self._browse)
        form.addRow("형식", self.fmt)
        form.addRow("WAV 형식", self.wav)
        form.addRow("MP3 비트레이트", self.bitrate)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out, 1)
        out_row.addWidget(browse)
        form.addRow("출력 폴더", out_row)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("내보내기")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.fmt.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self) -> None:
        mp3 = self.fmt.currentData() == "mp3"
        self.wav.setEnabled(not mp3)
        self.bitrate.setEnabled(mp3)
        ext = "." + self.fmt.currentData()
        for track, edit in self.names.items():
            stem = Path(edit.text()).stem if edit.text() else f"{self.project_name}_{track}"
            edit.setText(stem + ext)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "출력 폴더", self.out.text())
        if d:
            self.out.setText(d)

    def options(self) -> ExportOptions:
        return ExportOptions(self.fmt.currentData(), self.wav.currentData(), int(self.bitrate.currentData()))

    def jobs(self) -> list[ExportJob]:
        out = []
        for track, cb in self.checks.items():
            if cb.isChecked():
                name = sanitize_name(self.names[track].text()) or f"{track}{self.options().extension}"
                if not name.lower().endswith(self.options().extension):
                    name += self.options().extension
                out.append(ExportJob(track, os.path.join(self.out.text(), name)))
        return out


class ReanalysisDialog(QDialog):
    """새 분석 결과를 작업본에 어떻게 반영할지 고른다. 사용자 편집은 기본으로 보호한다."""

    def __init__(self, summary: ScoreSummary, new_notes: int, kept_preview: int, dropped_preview: int,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("분석 결과 반영")
        self.mode: str | None = None
        lay = QVBoxLayout(self)
        lay.addWidget(_title("작업본에 사용자 편집이 있습니다"))
        lines = [
            f"현재 작업본: 분석 원본 그대로 {summary.original}개 · 수정 {summary.modified}개 · 직접 추가 {summary.user}개"
            + (" · 템포 수정됨" if summary.tempo_edited else ""),
            f"새 분석 결과: 음표 {new_notes}개",
            "",
            f"[수정 유지하며 적용] 수정·추가한 음표 {kept_preview}개를 그대로 두고 손대지 않은 음표만 바꿉니다. "
            f"보존 음표와 겹치는 새 음표 {dropped_preview}개는 넣지 않습니다.",
            "[전부 새 분석으로 교체] 편집 내용을 버리고 새 결과로 바꿉니다.",
            "",
            "어느 쪽이든 편집 > 실행 취소(Ctrl+Z)로 되돌릴 수 있습니다.",
        ]
        lay.addWidget(_hint("\n".join(lines)))
        row = QHBoxLayout()
        keep = QPushButton("수정 유지하며 적용")
        keep.setProperty("primary", True)
        keep.setDefault(True)
        replace = QPushButton("전부 새 분석으로 교체")
        cancel = QPushButton("취소")
        keep.clicked.connect(lambda: self._done(KEEP_EDITS))
        replace.clicked.connect(lambda: self._done(REPLACE_ALL))
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        for b in (cancel, replace, keep):
            row.addWidget(b)
        lay.addLayout(row)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

    def _done(self, mode: str) -> None:
        self.mode = mode
        self.accept()
