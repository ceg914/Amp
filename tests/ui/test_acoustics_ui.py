"""v0.7.0 보컬 정밀 → 음향 분석: 분석(작업 프로세스) → 목록, 누르면 반복 구간, 호흡 제외 저장,
발성 경계는 교정 제안으로만 (음표 그대로)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from amp.devtools.synth import SR, breath_noise, sung
from tests.ui.test_main_window import create


@pytest.fixture
def phrase_song(tmp_path):
    mel = [(0.5, 1.4, 64), (1.5, 2.6, 67), (3.4, 4.4, 69), (4.5, 5.6, 67)]
    y = sung(mel, 6.5, vibrato_cents=45) + breath_noise(6.5, [(2.85, 3.3)], level_db=-32)
    y = y + np.random.default_rng(0).standard_normal(len(y)).astype(np.float32) * 1e-4
    p = tmp_path / "프레이즈.wav"
    sf.write(p, np.stack([y, y], 1), SR)
    return str(p)


def test_acoustics_page(qtbot, window, phrase_song):
    from amp.compare import suggestions as S
    from amp.engine.acoustics import run as AR

    w = window
    create(qtbot, w, phrase_song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.run_sample_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=20_000)
    notes_before = [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()]
    w.show_side_panel(w.precision.panel)
    w.precision.panel.sections.setCurrentWidget(w.precision.panel.acoustics)
    page, ac = w.precision.panel.acoustics, w.precision.acoustics
    page.run.click()
    qtbot.waitUntil(lambda: ac.res is not None and not w.session.busy, timeout=120_000)
    assert page.breaths.topLevelItemCount() == 1 and page.phrases.topLevelItemCount() == 2
    assert page.vibrato.topLevelItemCount() >= 2  # 45센트 비브라토를 넣은 음들
    assert "비브라토" in page.summary.toPlainText()
    assert (w.session.folder.paths.analysis / AR.FILES["boundaries"]).is_file()
    page.vibrato.setCurrentItem(page.vibrato.topLevelItem(0))
    ac._go(page.vibrato.topLevelItem(0))
    assert w.player.loop_region is not None

    page.breaths.topLevelItem(0).setSelected(True)
    page.b_reject.click()
    assert ac.res.breaths[0].status == "rejected"
    assert AR.load(w.session.folder.paths.analysis).breaths[0].status == "rejected"

    page.to_suggest.click()
    sug = w.doc.project.suggestions
    assert sug is not None and all(s.suggestion_type == "boundary" for s in sug.items)
    assert all(s.status == S.PENDING for s in sug.items)
    assert [(n.id, n.start, n.end, n.pitch) for n in w.doc.score.notes()] == notes_before  # 음표는 그대로
