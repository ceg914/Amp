"""가벼운 모드 화면: 무거운 기능의 메뉴·단추를 숨기고 나머지는 그대로."""

from __future__ import annotations


def test_lite_mode_hides_heavy_features(make_window, monkeypatch):
    monkeypatch.setenv("SONATLAS_LITE", "1")
    w = make_window()
    assert w.lite and "가벼운 모드" in w.windowTitle()
    for a in (w.model_compare.act_mode, w.model_compare.act_run, w.hybrid.act_mode, w.act_probe):
        assert not a.isVisible()
    panel = w.inspector.analysis
    assert panel.separation.findData("mel_roformer") < 0 and panel.separation.findData("none") >= 0
    assert panel.device.currentData() == "cpu" and not panel.probe.isVisibleTo(panel)
    lp = w.lyrics_panel
    assert not lp.recognition_group.isVisibleTo(lp) and not lp.alignment_group.isVisibleTo(lp)
    assert lp.lite_hint.isVisibleTo(lp)
    # 남는 기능
    for a in (w.act_analyze, w.act_import_stems, w.act_pitch_regions, w.act_export):
        assert a.isVisible()


def test_full_mode_unchanged(make_window, monkeypatch):
    monkeypatch.delenv("SONATLAS_LITE", raising=False)
    w = make_window()
    assert not w.lite and "가벼운 모드" not in w.windowTitle()
    assert w.model_compare.act_mode.isVisible() and w.hybrid.act_mode.isVisible()
    assert w.inspector.analysis.separation.findData("mel_roformer") >= 0
