"""v0.7.0 노래 분석 → 보컬 정밀 → 피치 모델: 모델 상태 표시, 기본·직접 선택 실행(작업 프로세스), 결과 저장과 겹쳐 보기,
두 결과 비교, 반복 구간 다시 분석. 음표는 바꾸지 않는다."""

from __future__ import annotations

import importlib.util

import pytest

from amp.engine import pitch_runs as PR
from tests.ui.test_main_window import create

HAS_CREPE = importlib.util.find_spec("torchcrepe") is not None and importlib.util.find_spec("torch") is not None


def open_panel(qtbot, w):
    w.show_side_panel(w.precision.panel)
    qtbot.waitUntil(lambda: w.precision.page.model_state["pyin"].text() != "확인 중…", timeout=20_000)
    return w.precision.page


def test_panel_lists_models_and_respects_registry(qtbot, window, song, monkeypatch):
    from amp.engine import model_registry as R

    real = R.scan

    def no_external(ctx=None, find_spec=None, **k):  # 외부 모델 패키지가 없는 컴퓨터
        fake = lambda name: None if name in ("torchcrepe", "torchfcpe") else importlib.util.find_spec(name)  # noqa: E731
        return real(ctx, fake, **k)

    monkeypatch.setattr(R, "scan", no_external)
    w = window
    create(qtbot, w, song)
    assert w._side_path() != ["노래 분석", "보컬 정밀"]
    p = open_panel(qtbot, w)
    assert w._side_path() == ["노래 분석", "보컬 정밀"]
    assert not p.models["crepe_full"].isEnabled() and "install_pitch" in p.model_state["fcpe"].text()
    assert p.models["pyin"].isChecked()  # 기본 방식 = pYIN
    p.mode.setCurrentIndex(p.mode.findData("precise"))
    assert w.precision.selected_models() == ("pyin",)  # 외부 모델이 없으면 pYIN만
    p.mode.setCurrentIndex(p.mode.findData("region"))
    assert "반복 구간을 먼저" in p.range_label.text() and not p.run.isEnabled()
    w.listening.set_region(1.0, 2.0)
    qtbot.waitUntil(lambda: "1.00–2.00" in p.range_label.text(), timeout=5_000)
    assert p.run.isEnabled()


@pytest.mark.skipif(not HAS_CREPE, reason="torchcrepe·torch 없음")
def test_run_compare_overlay_and_region(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.run_sample_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=20_000)
    notes_before = [(n.id, n.pitch, n.start) for n in w.doc.score.notes()]
    p = open_panel(qtbot, w)
    assert "원본" in p.source.text()  # 분리 결과가 없으면 원본을 분석
    p.mode.setCurrentIndex(p.mode.findData("select"))
    w.precision._set_checked({"pyin", "crepe_tiny"})
    p.run.click()
    qtbot.waitUntil(lambda: p.runs.topLevelItemCount() == 2 and not w.session.busy, timeout=180_000)
    assert "저장" in p.message.text() and "CREPE tiny" in p.message.text()
    store = PR.PitchRunStore(w.session.folder.paths.analysis)
    assert sorted(r.model_id for r in store.runs()) == ["crepe_tiny", "pyin"]
    assert len(p.checked_overlays()) == 2 and len(w.view.roll.overlay_curves.layer_keys()) == 2
    assert [(n.id, n.pitch, n.start) for n in w.doc.score.notes()] == notes_before  # 음표는 그대로

    p.compare.click()
    text = p.summary.toPlainText()
    assert "정확도가 아닙니다" in text and "유성·무성 판정" in text
    p.run.click()  # 같은 설정: 다시 계산하지 않는다
    qtbot.waitUntil(lambda: "이미 있는 결과 사용 2개" in p.message.text() and not w.session.busy, timeout=60_000)
    assert len(store.ids()) == 2

    p.mode.setCurrentIndex(p.mode.findData("region"))
    w.precision._set_checked({"crepe_tiny"})
    w.listening.set_region(1.5, 2.5)
    qtbot.waitUntil(lambda: p.run.isEnabled(), timeout=5_000)
    p.run.click()
    qtbot.waitUntil(lambda: p.runs.topLevelItemCount() == 3 and not w.session.busy, timeout=120_000)
    region = next(r for r in store.runs() if r.meta.analysis_range is not None)
    assert region.start_frame == 150 and region.meta.analysis_range == (1.5, 2.5)
    item = p.runs.topLevelItem(0)
    item.setCheckState(0, item.checkState(0).__class__.Unchecked)
    assert len(w.view.roll.overlay_curves.layer_keys()) == 2
