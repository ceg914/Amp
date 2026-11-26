"""v0.7.0 보컬 정밀 → 다중 보컬: 분석(작업 프로세스) → 성부 표시·겹쳐 보기, 구간 바꾸기·되돌리기·다시 하기, 이름·확정,
다시 분석해도 확정한 성부 유지(이전 결과 보관). 음표는 바꾸지 않는다."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt

from amp.devtools import multipitch_eval as E
from amp.devtools.synth import SR
from amp.engine import voices as V
from tests.ui.test_main_window import create


@pytest.fixture
def duet(tmp_path):
    xa, _fa = E._voice([(0.3, 3.7, 69)], 4.0, 0, 0, 30.0, 5.5)
    xb, _fb = E._voice([(0.3, 3.7, 62)], 4.0, 1, 3, 30.0, 5.9)
    y = (xa + 0.8 * xb).astype(np.float32) * 0.6
    p = tmp_path / "두 목소리.wav"
    sf.write(p, np.stack([y, y], 1), SR)
    return str(p)


def test_multivoice_analysis_edit_undo_and_protect(qtbot, window, duet):
    w = window
    create(qtbot, w, duet)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.show_side_panel(w.precision.panel)
    w.precision.panel.sections.setCurrentWidget(w.precision.panel.voices)
    page = w.precision.panel.voices
    vc = w.precision.voices
    page.run.click()
    qtbot.waitUntil(lambda: vc.vs is not None and not w.session.busy, timeout=120_000)
    assert [v.voice_track_id for v in vc.vs.voices] == ["v1", "v2"] and page.voices.topLevelItemCount() == 2
    med = [v.median_midi() for v in vc.vs.voices]
    assert med[0] == pytest.approx(69, abs=0.5) and med[1] == pytest.approx(62, abs=0.5)
    keys = w.view.roll.overlay_curves.layer_keys()
    assert "voice:v1" in keys and "voice:v2" in keys
    analysis = w.session.folder.paths.analysis
    assert (analysis / V.JSON_NAME).is_file() and (analysis / V.NPZ_NAME).is_file()

    w.listening.set_region(1.0, 2.0)
    qtbot.waitUntil(lambda: page.swap.isEnabled(), timeout=5_000)
    page.swap.click()
    v1 = vc.vs.voice("v1")
    assert 69 + 12 * np.log2(v1.f0_hz[150] / 440) == pytest.approx(62, abs=0.6) and v1.edited
    assert V.load(analysis).voice("v1").edited  # 편집마다 저장
    page.undo.click()
    assert 69 + 12 * np.log2(vc.vs.voice("v1").f0_hz[150] / 440) == pytest.approx(69, abs=0.6)
    page.redo.click()
    assert 69 + 12 * np.log2(vc.vs.voice("v1").f0_hz[150] / 440) == pytest.approx(62, abs=0.6)
    page.undo.click()

    item = page.voices.topLevelItem(1)
    item.setText(0, "화음")
    assert vc.vs.voice("v2").display_name == "화음"
    page.voice_a.setCurrentIndex(page.voice_a.findData("v2"))
    page.confirm.setChecked(True)
    assert vc.vs.voice("v2").user_confirmed
    item = page.voices.topLevelItem(0)
    item.setCheckState(0, Qt.CheckState.Unchecked)  # 숨기기
    assert "voice:v1" not in w.view.roll.overlay_curves.layer_keys()

    page.run.click()  # 다시 분석: 확정한 '화음'은 그대로, 이전 결과는 보관
    qtbot.waitUntil(lambda: not w.session.busy and vc.vs is not None and vc.vs.voices[0].display_name == "화음",
                    timeout=120_000)
    assert vc.vs.voices[0].user_confirmed
    assert any((analysis / V.HISTORY_DIR).iterdir())
