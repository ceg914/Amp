"""osu!mania 비트맵 내보내기 (v0.6.8): 내보내기 메뉴 → 제목·난이도·오프셋 대화상자 → .osz 저장.

멜로디(보컬 작업본 음표, 사용자가 고친 음표 포함)와 프로젝트 템포로 4K 채보를 만들고, 원곡을 OGG로 함께 넣는다.
정답 비트맵이 없는 자동 채보라 osu! 편집기에서 오프셋과 패턴을 확인하라고 안내한다.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from amp.export import osu_mania as O


class OsuExportDialog(QDialog):
    def __init__(self, title: str, parent=None, rhythm_label: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("osu!mania 비트맵 내보내기 (4K)")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        intro = QLabel("보컬 멜로디(지금 악보, 고친 음표 포함)와 프로젝트 BPM·첫 박으로 4키 채보를 만듭니다. "
                       "Normal부터는 드럼(없으면 반주) 타격으로 멜로디 사이를 채우고 마디 첫 박에 동시치기를 넣습니다. "
                       "원곡은 OGG로 함께 넣습니다. 자동 채보라 osu! 편집기에서 오프셋과 패턴을 확인하세요.")
        intro.setWordWrap(True)
        intro.setProperty("role", "secondary")
        lay.addWidget(intro)
        form = QFormLayout()
        self.title_unicode = QLineEdit(title)
        self.title = QLineEdit(O.ascii_text(title))
        self.title.setPlaceholderText("영문·로마자 (osu! 검색용)")
        self.artist_unicode = QLineEdit()
        self.artist = QLineEdit()
        self.artist.setPlaceholderText("영문·로마자 (비우면 Unknown)")
        self.artist_unicode.textChanged.connect(lambda t: self.artist.setText(O.ascii_text(t)))
        self.title_unicode.textChanged.connect(lambda t: self.title.setText(O.ascii_text(t)))
        form.addRow("제목", self.title_unicode)
        form.addRow("제목 (영문)", self.title)
        form.addRow("아티스트", self.artist_unicode)
        form.addRow("아티스트 (영문)", self.artist)
        diffs = QHBoxLayout()
        self.diffs = {}
        for name, d in O.DIFFICULTIES.items():
            cb = QCheckBox(name)
            cb.setChecked(name in O.DEFAULT_DIFFICULTIES)
            fill = f", 채우기 1/{int(round(1 / d.fill_snap))}박 상위 {d.fill_top:.0%}" if d.fill_snap else ""
            cb.setToolTip(f"멜로디 격자 1/{int(round(1 / d.snap))}박, 멜로디 사이 최소 {d.min_gap:g}박{fill}, "
                          f"롱노트 {d.hold_min:g}박 이상, OD {d.od:g}")
            self.diffs[name] = cb
            diffs.addWidget(cb)
        form.addRow("난이도", diffs)
        self.rhythm = QLabel(rhythm_label or "리듬: 원곡")
        self.rhythm.setProperty("role", "secondary")
        self.rhythm.setWordWrap(True)
        form.addRow("리듬 소리", self.rhythm)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-500.0, 500.0)
        self.offset.setDecimals(0)
        self.offset.setSuffix(" ms")
        self.offset.setToolTip("osu!에서 노트가 소리보다 늦게(빠르게) 오면 음수(양수)로. 모든 노트·타이밍선을 함께 옮깁니다")
        form.addRow("오프셋 보정", self.offset)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def meta(self) -> O.Meta:
        return O.Meta(title=self.title.text().strip(), title_unicode=self.title_unicode.text().strip(),
                      artist=self.artist.text().strip(), artist_unicode=self.artist_unicode.text().strip(),
                      offset_ms=float(self.offset.value()))

    def difficulties(self) -> tuple:
        return tuple(n for n, cb in self.diffs.items() if cb.isChecked())


def rhythm_source(w):
    """채우기·동시치기에 쓸 리듬 소리: 분리한 드럼 → 보컬 분석의 반주 → 원곡. (경로 또는 AudioData, 설명)."""
    s = w.session
    inst = getattr(w, "instruments", None)
    drums = inst.found.get("drums") if inst is not None else None
    if drums is not None:
        return str(drums.path), f"리듬: 드럼 ({drums.model_label})"
    accomp = s.track_audio("instrumental")
    if accomp is not None:
        return accomp, "리듬: 반주 (드럼을 분리하면 드럼 타격을 씁니다)"
    return s.audio, "리듬: 원곡 (보컬 분석·드럼 분리를 하면 더 정확합니다)"


def export_osz(w) -> None:
    """메인 창의 내보내기 메뉴에서 부른다."""
    s = w.session
    folder = s.folder
    notes = w.doc.score.notes()
    if folder is None or s.audio is None:
        return
    if not notes:
        QMessageBox.information(w, "osu!mania 비트맵", "보컬 음표가 없습니다. 보컬 분석을 먼저 하세요.")
        return
    if s.busy:
        w.statusBar().showMessage("분석이 끝난 뒤에 내보내세요.", 5000)
        return
    source, label = rhythm_source(w)
    dlg = OsuExportDialog(os.path.splitext(folder.meta.original.file_name)[0] or folder.name, w, label)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    diffs = dlg.difficulties()
    if not diffs:
        return
    meta = dlg.meta()
    folder.paths.exports.mkdir(parents=True, exist_ok=True)
    path, _f = QFileDialog.getSaveFileName(w, "osu! 비트맵 저장", os.path.join(str(folder.paths.exports),
                                                                         f"{O._safe(meta.title or folder.name)}.osz"),
                                           "osu! 비트맵 (*.osz)")
    if not path:
        return
    if not path.lower().endswith(".osz"):
        path += ".osz"
    tempo, audio = w.doc.score.tempo, s.audio

    def job(progress, token):
        progress(0.05, "osu! 비트맵: 리듬 타격 찾는 중")
        if isinstance(source, str):  # 드럼 분리 파일
            import soundfile as sf

            data, sr = sf.read(source, dtype="float32", always_2d=True)
            onsets = O.rhythm_onsets(data.mean(axis=1), sr)
        else:
            onsets = O.rhythm_onsets(source.mono(), source.sample_rate)
        progress(0.3, "osu! 비트맵: 음원을 OGG로 만드는 중")
        return O.write_osz(path, notes, tempo, audio, meta, diffs, onsets=onsets)

    handle = s._submit("osu! 비트맵 내보내기", job)

    def done(summaries) -> None:
        parts = [f"{x.name} {x.notes}개" for x in summaries]
        w.statusBar().showMessage(f"osu! 비트맵을 저장했습니다: {os.path.basename(path)} — " + ", ".join(parts)
                                  + ". osu!에서 열어 오프셋을 확인하세요.", 12000)

    handle.finished.connect(done)
