"""v0.6.1 피치 다시 찾기 구간 화면: 대화상자 입력(목소리 음역 프리셋), 적용하면 다시 분석(구간만 계산),
프로젝트에 저장·다시 열기, 프로그램 기본 설정에는 남기지 않음."""

from __future__ import annotations

import json

import pytest
from PySide6.QtWidgets import QDialog

from amp.app import dialogs
from amp.engine.settings import VOICE_PRESETS
from tests.ui.test_main_window import IdentityPipeline, create


def test_dialog_presets_and_validation(qtbot):
    dlg = dialogs.PitchRegionsDialog([], 60.0, region=(34.0, 46.5))
    qtbot.addWidget(dlg)
    dlg._add()
    assert dlg.parsed() == ((34.0, 46.5) + VOICE_PRESETS["male"],)
    combo = dlg.table.cellWidget(0, 2)
    combo.setCurrentIndex(combo.findData("female"))
    assert dlg.parsed()[0][2:] == VOICE_PRESETS["female"]
    dlg.table.item(0, 3).setText("80~300")
    assert dlg.parsed()[0][2:] == (80.0, 300.0)
    dlg.table.item(0, 1).setText("0:30")  # 끝이 시작보다 앞
    with pytest.raises(ValueError):
        dlg.parsed()


def test_apply_reanalyses_and_saves_with_project(qtbot, window, song, monkeypatch):
    w = window
    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    folder = create(qtbot, w, song)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    regions = ((1.9, 2.7, 140.0, 1100.0),)

    def fake_exec(self):
        self._regions = regions
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs.PitchRegionsDialog, "exec", fake_exec)
    first = w.doc.project.analysis
    w.edit_pitch_regions()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not first and not w.session.busy, timeout=120_000)
    assert w.inspector.analysis.settings().pitch_regions == regions
    info = w.doc.project.analysis.models["pitch"].params["regions"]
    assert info[0]["start"] == 1.9
    assert "1개" in w.inspector.analysis.pitch_regions_info.text()
    assert w.save()
    meta = json.loads(folder.paths.project_json.read_text(encoding="utf-8"))
    assert meta["analysis_settings"]["pitch_regions"] == [list(regions[0])]
    from amp.app.main_window import SETTINGS_KEY

    assert w.settings.value(SETTINGS_KEY)["pitch_regions"] == []  # 곡마다 다르므로 프로그램 기본값에 남기지 않음
