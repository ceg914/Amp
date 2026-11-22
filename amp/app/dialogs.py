"""대화상자: 새 프로젝트, 오디오 내보내기, 재분석 결과 반영, 보컬 없는 구간."""

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
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from amp.app.widgets.track_panel import TRACK_HINTS, TRACK_LABELS
from amp.engine.merge import KEEP_EDITS, REPLACE_ALL, ScoreSummary
from amp.engine.settings import VOICE_LABELS, VOICE_PRESETS, SettingsError, normalize_pitch_regions, normalize_ranges
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
                 parent=None, no_vocal_ranges: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("오디오 내보내기")
        self.setMinimumWidth(520)
        self.project_name = project_name
        self.checks: dict[str, QCheckBox] = {}
        self.names: dict[str, QLineEdit] = {}
        lay = QVBoxLayout(self)
        lay.addWidget(_title("오디오 내보내기"))
        lay.addWidget(_hint("이미 만든 분리 결과를 형식만 바꿔 저장합니다. 보컬 분리를 다시 실행하지 않습니다."))
        if no_vocal_ranges:
            lay.addWidget(_hint(f"보컬 없는 구간 {no_vocal_ranges}곳은 보컬에서 비우고 그 소리를 반주에 더해 내보냅니다. "
                                "끄려면 분석 설정의 '보컬 트랙에서도 비우기'를 해제하세요."))
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


# --- 보컬 없는 구간 ----------------------------------------------------------------

def format_time(seconds: float) -> str:
    """75.25 → '1:15.25'"""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    return f"{m}:{seconds - 60 * m:05.2f}"


def parse_time(text: str) -> float:
    """'1:15.25', '75.25', '75' → 초. 잘못되면 ValueError."""
    t = text.strip().replace("：", ":")
    if not t:
        raise ValueError("시간이 비어 있습니다.")
    if ":" in t:
        m, sec = t.split(":", 1)
        value = int(m) * 60 + float(sec)
        if float(sec) >= 60:
            raise ValueError(f"초는 60 미만이어야 합니다: {text}")
    else:
        value = float(t)
    if value < 0 or value != value:
        raise ValueError(f"올바른 시간이 아닙니다: {text}")
    return value


class NoVocalRangesDialog(QDialog):
    """보컬이 없는 구간 목록 편집. 이 구간에서는 음표를 만들지 않는다."""

    def __init__(self, ranges, duration: float, playhead: float = 0.0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("보컬 없는 구간")
        self.duration = max(0.0, float(duration))
        self.playhead = max(0.0, float(playhead))
        self._ranges: tuple = tuple(ranges)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["시작", "끝"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for a, b in self._ranges:
            self._append(a, b)
        add = QPushButton("재생 위치에 추가")
        add.setToolTip("재생 위치부터 5초 구간을 추가합니다. 시간은 표에서 고칠 수 있습니다.")
        remove = QPushButton("선택 삭제")
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(_hint("전주·간주처럼 보컬이 없는데 분리 결과에 반주가 섞여 음표가 생기는 구간을 적습니다. "
                            "이 구간에서는 음표를 만들지 않습니다. 직접 고치거나 추가한 음표는 지우지 않습니다. "
                            "시간은 '분:초' 또는 초로 적습니다 (예: 0:28.0)."))
        lay.addWidget(self.table)
        lay.addLayout(row)
        lay.addWidget(buttons)
        self.resize(420, 360)

    def _append(self, a: float, b: float) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(format_time(a)))
        self.table.setItem(r, 1, QTableWidgetItem(format_time(b)))

    def _add(self) -> None:
        a = self.playhead
        b = a + 5.0 if not self.duration else min(self.duration, a + 5.0)
        if b - a < 0.1:
            a = max(0.0, b - 5.0)
        self._append(a, b)

    def _remove(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def parsed(self) -> tuple:
        out = []
        for r in range(self.table.rowCount()):
            cells = [self.table.item(r, c) for c in (0, 1)]
            a, b = (parse_time(c.text() if c else "") for c in cells)
            if b <= a:
                raise ValueError(f"{r + 1}번째 줄: 끝({format_time(b)})이 시작({format_time(a)})보다 뒤여야 합니다.")
            out.append((a, b))
        return normalize_ranges(out)

    def accept(self) -> None:
        try:
            self._ranges = self.parsed()
        except (ValueError, SettingsError) as e:
            QMessageBox.warning(self, "보컬 없는 구간", str(e))
            return
        super().accept()

    def ranges(self) -> tuple:
        return self._ranges


AUDIO_FILTER = "오디오 (*.wav *.flac *.mp3 *.ogg *.m4a *.aiff *.aif);;모든 파일 (*)"


class StemImportDialog(QDialog):
    """외부 분리 결과 가져오기: 보컬 파일(필수), 반주 파일(선택), 만든 도구 이름(기록용)."""

    def __init__(self, start_dir: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("외부 분리 결과 가져오기")
        self.setMinimumWidth(560)
        self._dir = start_dir
        lay = QVBoxLayout(self)
        lay.addWidget(_title("외부 분리 결과 가져오기"))
        lay.addWidget(_hint(
            "다른 프로그램(예: Ultimate Vocal Remover의 Mel-Band RoFormer 모델)으로 같은 곡을 분리한 파일을 가져옵니다. "
            "원본과 샘플레이트·시간 어긋남·길이를 자동으로 맞추고, 다른 곡이면 가져오지 않습니다. "
            "지금 분리 결과는 backup/ 폴더에 남깁니다. 가져온 뒤 '보컬 분석 실행'을 하면 Demucs를 다시 돌리지 않고 "
            "가져온 보컬로 피치·음표를 계산합니다."))
        form = QFormLayout()
        self.vocals = QLineEdit()
        self.instrumental = QLineEdit()
        self.instrumental.setPlaceholderText("비우면 원본 − 보컬로 만듭니다")
        self.tool = QLineEdit()
        self.tool.setPlaceholderText("예: UVR MelBand RoFormer (기록용, 선택)")
        for label, edit in (("보컬 파일", self.vocals), ("반주 파일 (선택)", self.instrumental)):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            b = QPushButton("찾아보기…")
            b.clicked.connect(lambda _c=False, e=edit: self._browse(e))
            row.addWidget(b)
            form.addRow(label, row)
        form.addRow("만든 도구", self.tool)
        lay.addLayout(form)
        self.error = QLabel("")
        self.error.setProperty("role", "warning")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("가져오기")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "파일 고르기", edit.text() or self._dir, AUDIO_FILTER)
        if path:
            edit.setText(path)
            self._dir = os.path.dirname(path)

    def values(self) -> tuple[str, str | None, str]:
        inst = self.instrumental.text().strip()
        return self.vocals.text().strip(), (inst or None), self.tool.text().strip()

    def _accept(self) -> None:
        vocals, inst, _tool = self.values()
        if not vocals or not os.path.isfile(vocals):
            self.error.setText("보컬 파일을 고르세요.")
            return
        if inst and not os.path.isfile(inst):
            self.error.setText("반주 파일을 찾을 수 없습니다.")
            return
        if inst and os.path.abspath(inst) == os.path.abspath(vocals):
            self.error.setText("보컬과 반주에 같은 파일을 골랐습니다.")
            return
        self.accept()


class RoformerModelDialog(QDialog):
    """Mel-Band RoFormer 모델 파일 지정: 체크포인트(필수), 설정 .yaml(선택)."""

    def __init__(self, files: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        files = files or {}
        self.setWindowTitle("RoFormer 모델 파일")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Mel-Band RoFormer 모델 파일"))
        lay.addWidget(_hint(
            "직접 받은 Mel-Band RoFormer 보컬 모델의 체크포인트(.ckpt·.pth)를 고르세요. SONATLAS는 모델을 포함하거나 "
            "내려받지 않습니다. 모델마다 가중치 라이선스가 다르니 받은 곳에서 확인하세요. "
            "층 수·대역 수는 파일에서 읽고, STFT 설정이 기본값(n_fft 2048, hop 441, 44.1kHz)과 다른 모델이면 "
            "함께 받은 .yaml 설정 파일도 고르세요. 파일은 텐서만 읽습니다(파일 안의 코드를 실행하지 않음)."))
        form = QFormLayout()
        self.checkpoint = QLineEdit(files.get("checkpoint", ""))
        self.config = QLineEdit(files.get("config", ""))
        self.config.setPlaceholderText("비우면 기본 설정 (Kimberley Jensen 보컬 모델과 같은 값)")
        for label, edit, filt in (("체크포인트", self.checkpoint, "PyTorch 체크포인트 (*.ckpt *.pth *.pt *.safetensors);;모든 파일 (*)"),
                                  ("설정 (선택)", self.config, "YAML 설정 (*.yaml *.yml);;모든 파일 (*)")):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            b = QPushButton("찾아보기…")
            b.clicked.connect(lambda _c=False, e=edit, f=filt: self._browse(e, f))
            row.addWidget(b)
            form.addRow(label, row)
        lay.addLayout(form)
        self.error = QLabel("")
        self.error.setProperty("role", "warning")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("확인")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self, edit: QLineEdit, filt: str) -> None:
        start = edit.text() or self.checkpoint.text()
        path, _ = QFileDialog.getOpenFileName(self, "파일 고르기", os.path.dirname(start) if start else "", filt)
        if path:
            edit.setText(path)

    def values(self) -> dict:
        return {"checkpoint": self.checkpoint.text().strip(), "config": self.config.text().strip()}

    def _accept(self) -> None:
        v = self.values()
        if not v["checkpoint"] or not os.path.isfile(v["checkpoint"]):
            self.error.setText("체크포인트 파일을 고르세요.")
            return
        if v["config"] and not os.path.isfile(v["config"]):
            self.error.setText("설정 파일을 찾을 수 없습니다.")
            return
        self.accept()


class ScoreImportDialog(QDialog):
    """참조 악보 가져오기: 비교할 보컬 성부를 고른다. 고르지 않은 성부(반주)는 섞지 않는다."""

    def __init__(self, file_name: str, parts, suggested: str | None, warnings=(), parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QListWidget, QListWidgetItem

        self.setWindowTitle("참조 악보 가져오기")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.addWidget(_title(f"참조 악보: {file_name}"))
        lay.addWidget(_hint(
            "자동 채보와 비교할 보컬 성부를 고르세요. 고른 성부만 참조 트랙으로 저장하고 다른 성부는 섞지 않습니다. "
            "가져온 악보는 자동 채보·직접 고친 음표를 바꾸지 않습니다. 파일 사본은 프로젝트 폴더 scores/에 둡니다."))
        self.list = QListWidget()
        for p in parts:
            tags = []
            if p.has_lyrics:
                tags.append("가사 있음")
            if p.polyphonic:
                tags.append("화음·여러 성부")
            if p.drums:
                tags.append("타악기")
            text = f"{p.name} — 음표 {p.notes}개, 음역 {p.low}~{p.high}" + (f" ({', '.join(tags)})" if tags else "")
            if p.id == suggested:
                text += "  ← 보컬 후보"
            it = QListWidgetItem(text)
            it.setData(256, p.id)
            self.list.addItem(it)
            if p.id == suggested:
                self.list.setCurrentItem(it)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        lay.addWidget(self.list)
        if warnings:
            w = QLabel("\n".join(f"• {x}" for x in warnings))
            w.setWordWrap(True)
            w.setProperty("role", "warning")
            lay.addWidget(w)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("가져오기")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def part_id(self) -> str | None:
        it = self.list.currentItem()
        return None if it is None else it.data(256)


class PitchRegionsDialog(QDialog):
    """v0.6.1: 피치 다시 찾기 구간 목록. 여러 목소리·악기가 겹쳐 음표가 빠진 구간을 목소리 음역을 정해 다시 추적한다."""

    def __init__(self, regions, duration: float, region: tuple | None = None, playhead: float = 0.0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("피치 다시 찾기 구간")
        self.duration = max(0.0, float(duration))
        self.playhead = max(0.0, float(playhead))
        self.region = region
        self._regions: tuple = tuple(regions)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["시작", "끝", "목소리", "음역(Hz)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for a, b, lo, hi in self._regions:
            self._append(a, b, lo, hi)
        add = QPushButton("반복 구간 추가" if region else "재생 위치에 추가")
        add.setToolTip("파형에서 고른 반복 구간(없으면 재생 위치부터 5초)을 남성 음역으로 추가합니다. 시간과 음역은 표에서 고칩니다.")
        remove = QPushButton("선택 삭제")
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(_hint("여러 목소리나 악기가 겹쳐 피치를 놓친 구간을 적습니다. 그 구간만 고른 목소리 음역으로 좁히고 "
                            "유성 판정을 늦춰 다시 찾습니다. 한 번에 한 목소리만 따라가므로, 남성을 고르면 그 구간의 높은 목소리 "
                            "음표는 남지 않습니다. 다시 찾은 음표는 신뢰도가 낮게 표시되니 들어 보고 확인하세요. "
                            "직접 고친 음표는 바꾸지 않습니다. 시간은 '분:초' 또는 초 (예: 0:34.0)."))
        lay.addWidget(self.table)
        lay.addLayout(row)
        lay.addWidget(buttons)
        self.resize(520, 360)

    def _append(self, a: float, b: float, lo: float, hi: float) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(format_time(a)))
        self.table.setItem(r, 1, QTableWidgetItem(format_time(b)))
        combo = QComboBox()
        for k, label in VOICE_LABELS.items():
            combo.addItem(label, k)
        combo.addItem("직접 입력", "custom")
        preset = next((k for k, v in VOICE_PRESETS.items() if v == (float(lo), float(hi))), "custom")
        combo.setCurrentIndex(combo.findData(preset))
        self.table.setCellWidget(r, 2, combo)
        self.table.setItem(r, 3, QTableWidgetItem(f"{lo:g}-{hi:g}"))
        combo.currentIndexChanged.connect(lambda _i, c=combo, row=r: self._on_preset(c, row))

    def _on_preset(self, combo, row: int) -> None:
        k = combo.currentData()
        if k in VOICE_PRESETS:
            lo, hi = VOICE_PRESETS[k]
            self.table.setItem(row, 3, QTableWidgetItem(f"{lo:g}-{hi:g}"))

    def _add(self) -> None:
        if self.region is not None:
            a, b = self.region
        else:
            a = self.playhead
            b = a + 5.0 if not self.duration else min(self.duration, a + 5.0)
        lo, hi = VOICE_PRESETS["male"]
        self._append(a, b, lo, hi)

    def _remove(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def parsed(self) -> tuple:
        out = []
        for r in range(self.table.rowCount()):
            a = parse_time(self.table.item(r, 0).text() if self.table.item(r, 0) else "")
            b = parse_time(self.table.item(r, 1).text() if self.table.item(r, 1) else "")
            if b <= a:
                raise ValueError(f"{r + 1}번째 줄: 끝({format_time(b)})이 시작({format_time(a)})보다 뒤여야 합니다.")
            text = (self.table.item(r, 3).text() if self.table.item(r, 3) else "").replace("~", "-")
            try:
                lo, hi = (float(x) for x in text.split("-")[:2])
            except ValueError:
                raise ValueError(f"{r + 1}번째 줄: 음역은 '최저-최고'(Hz)로 적습니다 (예: 70-420).") from None
            out.append((a, b, lo, hi))
        return normalize_pitch_regions(out)

    def accept(self) -> None:
        try:
            self._regions = self.parsed()
        except (ValueError, SettingsError) as e:
            QMessageBox.warning(self, "피치 다시 찾기 구간", str(e))
            return
        super().accept()

    def regions(self) -> tuple:
        return self._regions

