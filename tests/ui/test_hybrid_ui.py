"""v0.6 Hybrid 보컬 화면: 편집 모드, 구간 레인(경계 끌기·실행 취소 한 단계), 모델 지정, 만들기, 전후 비교 듣기,
분석 소스로 쓰기(사용자 편집 보호·피치 청크 재사용), 검증 필요 표시, 저장·다시 열기, 끌기 응답 시간.

분리는 시험용 모델(_identity·_half)을 실제 작업 프로세스로 돌린다. 실제 htdemucs·RoFormer 가중치는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import time

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog

from amp.app import model_compare_controller as MC
from amp.engine import hybrid as H
from amp.engine.model_compare import ModelSpec
from tests.ui.test_main_window import create
from tests.ui.test_model_compare_ui import _compare


def _hashes(folder) -> dict:
    return {str(p.relative_to(folder.paths.stems)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.paths.stems.rglob("*.wav")}


@pytest.fixture
def hybrid(qtbot, window, song):
    folder = create(qtbot, window, song)
    _compare(qtbot, window, [ModelSpec("_identity"), ModelSpec("_half")])
    hc = window.hybrid
    hc.act_mode.trigger()
    assert hc.active and window.model_compare.active
    keys = {window.model_compare.entry(k).model: k for k in window.model_compare.outputs}
    return window, folder, hc, keys


def _mouse(widget, kind, x: float, button=Qt.MouseButton.LeftButton, mods=Qt.KeyboardModifier.NoModifier):
    pos = QPointF(x, widget.height() / 2)
    ev = QMouseEvent(kind, pos, widget.mapToGlobal(QPoint(int(x), widget.height() // 2)).toPointF(), button,
                     button if kind != QMouseEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton, mods)
    QApplication.sendEvent(widget, ev)


def test_mode_start_split_assign_and_undo(hybrid, qtbot):
    w, folder, hc, keys = hybrid
    tabs = w.model_compare.left_tabs
    assert tabs.currentWidget() is hc.panel and hc.lane.isVisible()
    assert hc.plan is None and hc.panel.buttons["start"].isVisible()
    hc.on_action("start")
    plan = hc.plan
    assert plan is not None and len(plan.segments) == 1
    assert plan.frames == w.session.audio.frames  # 원본과 같은 길이
    w.seek(2.5)
    hc.on_action("split")
    assert len(hc.plan.segments) == 2 and hc.plan.segments[1].start_frame == round(2.5 * plan.sample_rate)
    seg = hc.plan.segments[1].segment_id
    first = hc.plan.segments[0].source_key
    other = next(k for k in keys.values() if k != first)
    hc.set_model([seg], other)
    assert hc.plan.get(seg).source_key == other and hc.plan.segments[0].source_key == first
    assert all(st.ok for st in hc.status.values())
    label = w.model_compare.entry(other).label
    assert hc.panel.table.rowCount() == 2 and hc.panel.table.item(1, 2).text() == label
    n = w.doc.undo_stack.count()
    w.doc.undo_stack.undo()
    assert hc.plan.get(seg).source_key == first
    w.doc.undo_stack.redo()
    assert w.doc.undo_stack.count() == n and hc.plan.get(seg).source_key == other


def test_boundary_drag_is_one_undo_step_and_shows_delta(hybrid, qtbot):
    w, folder, hc, keys = hybrid
    hc.on_action("start")
    w.seek(2.0)
    hc.on_action("split")
    lane = hc.lane
    lane.resize(800, lane.height())
    w.timeline.update_view(100.0, 0.0)  # 1초 = 100px
    count = w.doc.undo_stack.count()
    _mouse(lane, QMouseEvent.Type.MouseButtonPress, 200.0)
    assert lane._drag_index == 1
    for x in range(201, 260, 3):
        _mouse(lane, QMouseEvent.Type.MouseMove, float(x), Qt.MouseButton.NoButton)
    assert w.doc.undo_stack.count() == count  # 끄는 동안에는 기록하지 않는다
    _mouse(lane, QMouseEvent.Type.MouseButtonRelease, 258.0)
    assert w.doc.undo_stack.count() == count + 1
    t = hc.plan.time(hc.plan.segments[1].start_frame)
    assert t == pytest.approx(2.58, abs=0.011)
    w.doc.undo_stack.undo()
    assert hc.plan.time(hc.plan.segments[1].start_frame) == pytest.approx(2.0, abs=1e-6)


def test_render_compare_and_use_as_source_keeps_user_edits(hybrid, qtbot, monkeypatch):
    w, folder, hc, keys = hybrid
    # 먼저 _identity로 분석 소스를 정하고, 음표 하나를 사용자가 고친다
    mc = w.model_compare
    mc.set_pair(keys["_identity"], keys["_half"])
    monkeypatch.setattr(MC.SourceSwitchDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    mc.use_as_source("a")
    qtbot.waitUntil(lambda: w.session.manifest is not None, timeout=20_000)
    note = w.doc.score.notes()[0]
    assert w.doc.move_notes([note.id], 0.0, 1)
    edited = next(n for n in w.doc.score.notes() if n.origin_ids == note.origin_ids)
    assert w.doc.project.note_status(edited) == "modified"
    before = _hashes(folder)

    hc.on_action("start")
    w.seek(2.6)
    hc.on_action("split")
    hc.set_model([hc.plan.segments[1].segment_id], keys["_half"])
    assert hc.lane.changed and hc.panel.changed.text().startswith("지금 분석 소스와 달라지는 범위")
    hc.on_action("compare")  # 만들고 A = 지금 소스, B = 새 Hybrid
    qtbot.waitUntil(lambda: hc.last_render is not None and not w.session.busy, timeout=60_000)
    hkey = hc.last_render.entry.key
    # 전후 비교는 만들기가 끝난 다음 이벤트 차례에 A/B를 바꾼다: 바뀐 뒤 트랙이 올라올 때까지 기다린다
    qtbot.waitUntil(lambda: mc.keys.get("b") == hkey and mc.tracks_loaded, timeout=30_000)
    assert mc.keys == {"a": keys["_identity"], "b": hkey}
    assert {MC.CMP_A, MC.CMP_B} <= set(w.player.tracks)
    after = _hashes(folder)
    assert {k: v for k, v in after.items() if k in before} == before  # 원본 분리 결과는 그대로
    assert any(k.startswith(f"models/{hkey}/") for k in after)

    seen = []
    monkeypatch.setattr(MC.SourceSwitchDialog, "exec",
                        lambda self: (seen.append(" ".join(lab.text() for lab in self.findChildren(MC.QLabel))),
                                      QDialog.DialogCode.Accepted)[1])
    hc.on_action("use")
    qtbot.waitUntil(lambda: w.session.manifest is not None and w.session.manifest.model == H.HYBRID, timeout=60_000)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=30_000)
    assert seen and "피치: 청크" in seen[0]
    a = w.doc.project.analysis
    assert a.models["separation"].name == H.HYBRID and a.models["separation"].params["store_key"] == hkey
    assert edited.id in w.doc.score and w.doc.score.get(edited.id) == edited  # 사용자 편집 유지
    assert hc.panel.stale.text() == ""
    # 저장·다시 열기
    assert w.save()
    assert folder.paths.hybrid_json.exists()
    saved = hc.plan
    from amp.project import workspace

    reopened, report = workspace.open_folder(folder.root)[0].load_project()
    assert reopened.hybrid == saved and not report.warnings


def test_missing_source_marks_segment_and_blocks_render(hybrid, qtbot):
    w, folder, hc, keys = hybrid
    hc.on_action("start")
    w.seek(2.0)
    hc.on_action("split")
    hc.set_model([hc.plan.segments[1].segment_id], keys["_half"])
    w.model_compare.keys = {"a": None, "b": None}
    w.model_compare._load_tracks()
    w.session.model_store().delete(keys["_half"])
    hc.refresh()
    st = hc.status[hc.plan.segments[1].segment_id]
    assert st.state == H.STATUS_CHECK and st.label == "검증 필요"
    assert not hc.panel.buttons["render"].isEnabled()
    assert "검증 필요" in hc.panel.table.item(1, 5).text()
    assert hc.plan.segments[1].source_key == keys["_half"]  # 계획은 자동으로 바꾸지 않는다


def test_lane_drag_stays_responsive_with_many_segments(hybrid, qtbot):
    w, folder, hc, keys = hybrid
    hc.on_action("start")
    plan = hc.plan
    for i in range(1, 60):  # 60구간
        plan = H.split(plan, plan.segments[-1].segment_id, i * 0.08)
    w.doc.set_hybrid(plan, "많은 구간")
    lane = hc.lane
    lane.resize(1200, lane.height())
    w.timeline.update_view(240.0, 0.0)
    x0 = w.timeline.time_to_x(plan.time(plan.segments[30].start_frame))
    _mouse(lane, QMouseEvent.Type.MouseButtonPress, x0)
    t0 = time.perf_counter()
    for k in range(40):
        _mouse(lane, QMouseEvent.Type.MouseMove, x0 + (k % 7) - 3, Qt.MouseButton.NoButton)
        lane.repaint()
    dt = time.perf_counter() - t0
    _mouse(lane, QMouseEvent.Type.MouseButtonRelease, x0 + 2)
    assert dt < 2.0, f"끌기 40번 + 다시 그리기 {dt:.2f}s"
    assert H.validate(hc.plan) == []
