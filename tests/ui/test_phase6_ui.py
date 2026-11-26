"""v0.7.0 Phase 6 화면: 100ms 멈춤 기록, 분석 레인(비브라토·호흡·발성 경계), 모델 관리 창, 분석 선택 창과 차례 실행, 가벼운 모드."""

from __future__ import annotations

import json
import time

from PySide6.QtCore import QTimer

from amp.engine.acoustics.breath import Breath, Phrase
from amp.engine.acoustics.onsets import NoteBoundary
from amp.engine.acoustics.vibrato import VibratoSegment
from tests.ui.test_main_window import IdentityPipeline, create


def test_stall_monitor_records_100ms_with_last_action(qtbot, window, tmp_path):
    mon = window.ui_stalls
    assert mon.running and mon.threshold_ms == 100.0
    window.act_zoom_fit.trigger()  # 마지막 동작으로 남는다
    n0 = len(mon.records)
    QTimer.singleShot(0, lambda: time.sleep(0.25))  # 화면 스레드를 250ms 막는다
    qtbot.waitUntil(lambda: len(mon.records) > n0, timeout=3000)
    r = mon.records[-1]
    assert 180 <= r.duration_ms <= 400 and r.context == "전체 보기"
    qtbot.wait(150)
    assert len(mon.records) == n0 + 1  # 정상 간격은 남기지 않는다
    path = tmp_path / "stalls.json"
    mon.save(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["summary"]["count"] >= 1 and d["records"][-1]["context"] == "전체 보기" and "원인 판정이 아닙니다" in d["note"]


class _Res:
    def __init__(self):
        self.vibrato = [VibratoSegment(1.0, 2.0, 5.5, 40.0, 80.0, 50.0, 64.0, 0.9, 0.0, 0.0, 0.2, 5.5),
                        VibratoSegment(3.0, 3.5, 6.0, 15.0, 30.0, 20.0, 64.0, 0.5, 0.0, 0.0, None, 3.0, "uncertain")]
        self.breaths = [Breath("b1", 2.2, 2.4, -40.0, 0.3, 0.5, 0.8)]
        self.phrases = [Phrase("p1", 0.5, 2.2, 1.5, None, 0.2)]
        self.boundaries = [NoteBoundary("n1", 1.0, 2.0, 64, 0.98, 1.01, 1.05, 1.95, 2.1)]
        self.meta = {}


def test_analysis_lanes_are_off_by_default_and_show_values(qtbot, window):
    w = window
    lane = w.analysis_lane
    assert not lane.isVisible() and not any(a.isChecked() for a in w.lane_actions.values())
    w.lane_actions["vibrato"].setChecked(True)
    w.lane_actions["boundary"].setChecked(True)
    assert lane.isVisible() and lane.visible_rows == ["vibrato", "boundary"] and lane.height() == 2 * 20 + 2
    assert w.settings.value("view/analysis_lanes") == "vibrato,boundary"
    lane.set_result(_Res())
    tl = w.timeline
    assert "5.50Hz" in lane.describe(tl.time_to_x(1.5), 10) and "확인됨" in lane.describe(tl.time_to_x(1.5), 10)
    assert "불확실" in lane.describe(tl.time_to_x(3.2), 10)
    assert "목표 음높이 도달" in lane.describe(tl.time_to_x(1.05), 30) and "자동으로 적용하지 않음" in lane.describe(tl.time_to_x(1.05), 30)
    lane.grab()  # 그리기에서 오류가 나지 않는다
    w.lane_actions["breath"].setChecked(True)  # 줄을 켜면 음향 분석 결과(여기서는 없음)로 다시 맞춘다
    assert lane.breaths == []
    lane.set_result(_Res())
    assert "호흡 후보" in lane.describe(tl.time_to_x(2.3), 30)
    for a in w.lane_actions.values():
        a.setChecked(False)
    assert not lane.isVisible()


def test_model_manager_lists_registry_and_profiles(qtbot, window):
    from amp.app.model_manager_dialog import ModelManagerDialog
    from amp.engine import model_registry as R

    dlg = ModelManagerDialog(window, window.session.model_registry(save=False))
    qtbot.addWidget(dlg)
    assert dlg.tree.topLevelItemCount() == len(R.CATALOG)
    ids = [dlg.tree.topLevelItem(i).data(0, 256) for i in range(dlg.tree.topLevelItemCount())]
    dlg.tree.setCurrentItem(dlg.tree.topLevelItem(ids.index("pyin")))
    assert "pYIN" in dlg.detail.toPlainText() and "라이선스" in dlg.detail.toPlainText()
    assert "✓ 단성 피치 (pYIN)" in dlg.features.toPlainText()
    dlg.profile.setCurrentIndex(dlg.profile.findData("lite"))
    assert "✕ 보컬 분리" in dlg.features.toPlainText()
    dlg.rescan(False)
    qtbot.waitUntil(lambda: "모델" in dlg.message.text() and dlg.btn_refresh.isEnabled(), timeout=20_000)


def test_analysis_dialog_and_chain_run_steps_in_order(qtbot, window, song, monkeypatch):
    from amp.app.analysis_dialog import AnalysisChain, AnalysisDialog, context_for

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    w = window
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    ctx = context_for(w)
    dlg = AnalysisDialog(w, ctx)
    qtbot.addWidget(dlg)
    assert dlg.items["multipitch"].checkState(0).name == "Unchecked"  # 실험 기능은 권장에 없다
    dlg.set_selected(["vibrato"])
    plan = dlg.plan()
    assert plan.steps == ["vocal", "acoustics"] and "1. 보컬 분석" in dlg.plan_text.toPlainText()
    chain = AnalysisChain(w, plan)
    ends = []
    chain.finished.connect(ends.append)
    chain.start()
    qtbot.waitUntil(lambda: bool(ends), timeout=180_000)
    assert chain.done == ["vocal", "acoustics"], ends
    assert w.doc.project.analysis is not None and w.precision.acoustics.res is not None
    assert "끝난 단계" in w.statusBar().currentMessage()
    assert w.lane_actions["vibrato"].isChecked() is False  # 레인은 사용자가 켤 때만
    from amp.engine import analysis_plan as AP

    assert AP.resolve(["vibrato"], context_for(w)).steps == ["acoustics"]  # 보컬 분석이 있으니 다시 하지 않는다


def test_lite_window_keeps_new_features_safe(qtbot, make_window, monkeypatch):
    monkeypatch.setenv("SONATLAS_LITE", "1")
    w = make_window("lite")
    from amp.app.analysis_dialog import AnalysisDialog, context_for

    ctx = context_for(w)
    assert ctx.lite
    dlg = AnalysisDialog(w, ctx)
    qtbot.addWidget(dlg)
    assert not (dlg.items["multipitch"].flags() & dlg.items["multipitch"].flags().ItemIsEnabled)
    assert not (dlg.items["separation"].flags() & dlg.items["separation"].flags().ItemIsEnabled)
    assert dlg.items["vibrato"].flags() & dlg.items["vibrato"].flags().ItemIsEnabled
    assert w.ui_stalls.running and w.act_models.isEnabled()


def test_ui_font_has_every_ipa_symbol_we_produce(qtbot, window):
    """명세 16: IPA 표시. 발음 변환기(한국어·일본어·영어)가 만드는 IPA 글자가 화면 글꼴(Pretendard)에 모두 있는지
    (없으면 운영체제 대체 글꼴에 기대게 된다). 숫자만 쓰는 고정폭 글꼴(JetBrains Mono)은 IPA를 그리지 않는다."""
    import inspect

    from PySide6.QtGui import QRawFont

    from amp.app import theme
    from amp.lyrics.pronunciation import english as E
    from amp.lyrics.pronunciation import japanese as J
    from amp.lyrics.pronunciation import korean as K

    src = inspect.getsource(K) + inspect.getsource(J)
    hangul_or_kana = lambda c: 0xAC00 <= ord(c) <= 0xD7A3 or 0x1100 <= ord(c) <= 0x11FF or 0x3130 <= ord(c) <= 0x318F \
        or 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF  # noqa: E731
    chars = {c for c in src if ord(c) > 0x7F and not hangul_or_kana(c) and not c.isspace()
             and c not in "·…“”‘’→←−–—、。「」（）　"}
    chars |= set("".join(E.VOWELS.values()) + "".join(E.CONSONANTS.values()) + "ˈˌəɚ")
    raw = QRawFont.fromFont(theme.ui_font(14))
    missing = sorted(c for c in chars if raw.glyphIndexesForString(c) in ([0], []))
    assert raw.familyName() == "Pretendard" and not missing, missing
