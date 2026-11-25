"""v0.6.8 내보내기 > osu!mania 비트맵: 대화상자 → .osz 저장 (보컬 작업본 음표·프로젝트 템포·원곡 OGG)."""

from __future__ import annotations

import io
import zipfile

import soundfile as sf
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from amp.app import osu_controller as OC
from amp.export import osu_mania as O
from tests.ui.test_main_window import create


def test_export_osz_from_menu(qtbot, window, song, tmp_path, monkeypatch):
    w = window
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    OC.export_osz(w)  # 음표가 없으면 안내만
    assert shown and "보컬 분석" in shown[0]
    w.run_sample_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=20_000)
    assert w.act_osu.isEnabled()

    def fake_exec(dlg):
        assert [n for n, cb in dlg.diffs.items() if cb.isChecked()] == ["Normal", "Hard", "Insane", "Expert"]  # 기본
        assert dlg.rhythm.text().startswith("리듬: 원곡")  # 보컬 분석·드럼 분리 전
        dlg.title_unicode.setText("원본 노래")
        for name in ("Insane", "Expert"):
            dlg.diffs[name].setChecked(False)
        dlg.offset.setValue(-12)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(OC.OsuExportDialog, "exec", fake_exec)
    out = tmp_path / "맵.osz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    w.act_osu.trigger()
    qtbot.waitUntil(lambda: out.exists() and not w.session.busy, timeout=60_000)
    z = zipfile.ZipFile(out)
    names = sorted(z.namelist())
    assert names == ["Unknown - Weonbon Norae (SONATLAS) [Hard].osu", "Unknown - Weonbon Norae (SONATLAS) [Normal].osu",
                     "audio.ogg"]
    text = z.read(names[1]).decode("utf-8")
    assert "TitleUnicode:원본 노래" in text and "Mode: 3" in text
    objs = O.parse_hit_objects(text)
    first = min(n.start for n in w.doc.score.notes())
    half = w.doc.score.tempo.seconds_per_beat_at(first) * 500  # 반 박 (격자에 맞추며 옮겨질 수 있는 최대치)
    assert objs and min(abs(t - (first * 1000 - 12)) for _c, t, _e in objs) <= half + 1  # 첫 멜로디 음표 (보정 -12ms)
    y, sr = sf.read(io.BytesIO(z.read("audio.ogg")), always_2d=True)
    assert y.shape[0] == w.session.audio.frames and sr == w.session.audio.sample_rate
