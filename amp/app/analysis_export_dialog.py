"""분석 결과 내보내기 창 (v0.7.0 명세 18). 내보내기 → 분석 결과 내보내기….

JSON·CSV·TXT·PNG와 선택 구간 오디오(WAV)를 한 폴더에 쓴다. 파일 쓰기와 그림 그리기는 작업 스레드에서 한다.
새 악보를 만들지 않는다 (MIDI·MusicXML·USTX·GP5는 기존 내보내기 그대로).
"""

from __future__ import annotations

import os
from datetime import datetime

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from amp.export import analysis_export as AE

PNG_ITEMS = {"pitch": "피치 그래프", "spectrogram": "스펙트로그램", "vibrato": "비브라토 그래프"}
WAV_TRACKS = {"original": "원본", "vocals": "보컬", "instrumental": "반주"}


class AnalysisExportDialog(QDialog):
    def __init__(self, parent, out_dir: str, region: tuple | None, available_tracks: set[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("분석 결과 내보내기")
        self.region = region
        lay = QVBoxLayout(self)
        hint = QLabel(AE.NOTE + " 새 악보는 만들지 않습니다.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        rg = QGroupBox("범위")
        rl = QHBoxLayout(rg)
        self.whole = QRadioButton("곡 전체")
        self.loop = QRadioButton(f"반복 구간 ({region[0]:.2f}~{region[1]:.2f}초)" if region else "반복 구간 (켜져 있지 않음)")
        self.loop.setEnabled(region is not None)
        (self.loop if region else self.whole).setChecked(True)
        rl.addWidget(self.whole)
        rl.addWidget(self.loop)
        lay.addWidget(rg)
        box = QGroupBox("형식")
        g = QGridLayout(box)
        self.json = QCheckBox("JSON (전체 구조)")
        self.txt = QCheckBox("TXT (가사·발음·음절 시간·요약)")
        self.json.setChecked(True)
        self.txt.setChecked(True)
        g.addWidget(self.json, 0, 0)
        g.addWidget(self.txt, 0, 1)
        g.addWidget(QLabel("CSV"), 1, 0)
        self.csv: dict[str, QCheckBox] = {}
        for i, k in enumerate(AE.CSV_ITEMS):
            cb = QCheckBox(AE.CSV_LABELS[k])
            cb.setChecked(True)
            self.csv[k] = cb
            g.addWidget(cb, 2 + i // 3, i % 3)
        g.addWidget(QLabel("PNG (구간 그림)"), 4, 0)
        self.png: dict[str, QCheckBox] = {}
        for i, (k, label) in enumerate(PNG_ITEMS.items()):
            cb = QCheckBox(label)
            self.png[k] = cb
            g.addWidget(cb, 5, i)
        g.addWidget(QLabel("WAV (선택 구간 오디오, 24bit)"), 6, 0)
        self.wav: dict[str, QCheckBox] = {}
        for i, (k, label) in enumerate(WAV_TRACKS.items()):
            cb = QCheckBox(label)
            cb.setEnabled(k in available_tracks)
            self.wav[k] = cb
            g.addWidget(cb, 7, i)
        lay.addWidget(box)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("폴더"))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.folder = QLineEdit(os.path.join(out_dir, f"analysis-{stamp}"))
        pick = QPushButton("고르기…")
        pick.clicked.connect(self._pick)
        frow.addWidget(self.folder, 1)
        frow.addWidget(pick)
        lay.addLayout(frow)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("내보내기")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "내보낼 폴더", self.folder.text())
        if d:
            self.folder.setText(d)

    def choice(self) -> dict:
        return {"region": self.region if self.loop.isChecked() else None,
                "kinds": tuple(k for k, cb in (("json", self.json), ("txt", self.txt)) if cb.isChecked())
                + (("csv",) if any(cb.isChecked() for cb in self.csv.values()) else ()),
                "csv_items": tuple(k for k, cb in self.csv.items() if cb.isChecked()),
                "png": tuple(k for k, cb in self.png.items() if cb.isChecked()),
                "wav": tuple(k for k, cb in self.wav.items() if cb.isChecked() and cb.isEnabled()),
                "folder": self.folder.text().strip()}


def run_export(window, choice: dict):
    """작업 스레드에서 쓴다. 결과: (쓴 파일 목록, 건너뛴 항목 → 이유)."""
    from amp.app import analysis_png as G

    w = window
    project = w.doc.project
    prec = getattr(w, "precision", None)
    src = AE.Sources(title=w.session.folder.name if w.session.folder else "", pitch_curve=project.analysis.pitch_curve
                     if project.analysis is not None else None, voices=prec.voices.vs if prec is not None else None,
                     acoustics=prec.acoustics.res if prec is not None else None, lyrics=w.doc.lyrics,
                     notes=list(w.doc.score.notes()), region=choice["region"])
    duration = w.session.audio.duration if w.session.audio is not None else 0.0
    t0, t1 = choice["region"] or (0.0, duration)
    audio = {k: w.session.track_audio(k) for k in set(choice["wav"]) | {"vocals", "original"}}
    folder, base = choice["folder"], (src.title or "sonatlas").replace(os.sep, "_")

    def job(progress, token):
        written, skipped = AE.export_all(src, folder, base, choice["kinds"], choice["csv_items"])
        progress(0.5, "그림 그리는 중")
        tag = f"{t0:.1f}-{t1:.1f}s"
        if "pitch" in choice["png"]:
            if src.pitch_curve is None and src.voices is None:
                skipped["png:pitch"] = "피치 결과가 없습니다."
            else:
                p = os.path.join(folder, f"{base}-pitch-{tag}.png")
                G.pitch_png(p, src.pitch_curve, t0, t1, src.notes, src.voices)
                written.append(p)
        if "vibrato" in choice["png"]:
            if src.acoustics is None or src.pitch_curve is None:
                skipped["png:vibrato"] = "음향 분석(비브라토) 결과가 없습니다."
            else:
                p = os.path.join(folder, f"{base}-vibrato-{tag}.png")
                G.vibrato_png(p, src.pitch_curve, src.acoustics.vibrato, t0, t1)
                written.append(p)
        if "spectrogram" in choice["png"]:
            a = audio.get("vocals") or audio.get("original")
            if a is None:
                skipped["png:spectrogram"] = "오디오가 없습니다."
            else:
                p = os.path.join(folder, f"{base}-spectrogram-{tag}.png")
                G.spectrogram_png(p, a.mono(), a.sample_rate, t0, t1)
                written.append(p)
        for k in choice["wav"]:
            token.raise_if_cancelled()
            a = audio.get(k)
            if a is None:
                skipped[f"wav:{k}"] = "트랙이 없습니다."
                continue
            p = os.path.join(folder, f"{base}-{k}-{tag}.wav")
            G.region_wav(p, a, t0, t1)
            written.append(p)
        progress(1.0, "분석 결과 내보내기 완료")
        return [str(x) for x in written], skipped

    return job
