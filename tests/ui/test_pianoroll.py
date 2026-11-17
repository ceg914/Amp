import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from amp.app.document import ProjectDocument
from amp.app.pianoroll.view import PianoRollView
from amp.app.timeline import TimelineState
from amp.core.model import AnalysisResult, LyricsTrack, Syllable
from amp.core.tempo import TempoMap
from tests.conftest import note


@pytest.fixture
def setup(qtbot):
    doc = ProjectDocument()
    result = AnalysisResult(
        10.0,
        TempoMap.constant(120),  # 1박 = 0.5초, 1/16 격자 = 0.125초
        (
            note("a", 1.0, 2.0, 60, syllable_id="s1"),
            note("b", 2.0, 2.5, 62, syllable_id="s1"),
            note("c", 3.0, 4.0, 64, syllable_id="s2"),
        ),
        lyrics=LyricsTrack("ko", (Syllable("s1", "가"), Syllable("s2", "나"))),
    )
    doc.set_analysis(result)
    timeline = TimelineState()
    view = PianoRollView(doc, timeline)
    qtbot.addWidget(view)
    view.resize(900, 500)
    view.show()
    qtbot.waitExposed(view)
    view.center_on_pitch(62)
    view.horizontalScrollBar().setValue(0)
    return doc, view


def vpos(view, t, pitch):
    m = view.mapper
    return view.mapFromScene(QPointF(m.time_to_x(t), m.pitch_center_y(pitch)))


def drag(view, start: QPoint, end: QPoint, modifier=Qt.KeyboardModifier.NoModifier):
    vp = view.viewport()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, modifier, start)
    steps = 5
    for i in range(1, steps + 1):
        p = start + (end - start) * i / steps
        QTest.mouseMove(vp, QPoint(int(p.x()), int(p.y())))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, modifier, end)


def test_items_and_labels(setup):
    doc, view = setup
    items = view.roll.items_by_id
    assert set(items) == {"a", "b", "c"}
    assert items["a"].label == "가" and items["b"].label == "–" and items["c"].label == "나"
    assert items["a"].status == "original"


def test_drag_moves_note_with_snap(setup):
    doc, view = setup
    start = vpos(view, 1.5, 60)
    end = vpos(view, 1.5 + 0.53, 63)
    drag(view, start, end)
    n = doc.score.get("a")
    assert n.pitch == 63
    assert n.start == pytest.approx(1.5)  # 1.0+0.53 → 1/16 격자(0.125)로 스냅
    assert view.roll.items_by_id["a"].status == "modified"
    doc.undo_stack.undo()
    assert doc.score.get("a").pitch == 60


def test_vertical_drag_keeps_time(setup):
    doc, view = setup
    start = vpos(view, 3.5, 64)
    drag(view, start, start + QPoint(2, -int(view.mapper.row_height * 2)))
    n = doc.score.get("c")
    assert n.pitch == 66 and n.start == pytest.approx(3.0)


def test_resize_end(setup):
    doc, view = setup
    r = view.roll.items_by_id["c"].rect()
    start = view.mapFromScene(QPointF(r.right() - 2, r.center().y()))
    end = start + QPoint(int(0.5 * view.mapper.px_per_sec), 0)
    drag(view, start, end)
    assert doc.score.get("c").end == pytest.approx(4.5)
    assert doc.score.get("c").start == pytest.approx(3.0)


def test_click_selects_and_keys_transpose(setup):
    doc, view = setup
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vpos(view, 3.5, 64))
    assert view.selected_ids() == ["c"]
    QTest.keyClick(view, Qt.Key.Key_Up)
    assert doc.score.get("c").pitch == 65
    QTest.keyClick(view, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert doc.score.get("c").pitch == 53
    QTest.keyClick(view, Qt.Key.Key_Right)
    assert doc.score.get("c").start == pytest.approx(3.125)
    assert view.selected_ids() == ["c"]


def test_ctrl_click_toggles(setup):
    doc, view = setup
    vp = view.viewport()
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vpos(view, 1.5, 60))
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, vpos(view, 3.5, 64))
    assert sorted(view.selected_ids()) == ["a", "c"]


def test_double_click_adds_note(setup):
    doc, view = setup
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vpos(view, 6.07, 67))
    added = [n for n in doc.score.notes() if n.id not in ("a", "b", "c")]
    assert len(added) == 1
    assert added[0].pitch == 67 and added[0].start == pytest.approx(6.0) and added[0].end == pytest.approx(6.125)
    assert view.roll.items_by_id[added[0].id].status == "user"


def test_split_and_merge_update_items(setup):
    doc, view = setup
    left, right = doc.split_note("a", 1.5)
    assert "a" not in view.roll.items_by_id
    assert sorted(view.selected_ids()) == sorted([left, right])
    assert view.roll.items_by_id[right].label == "–"
    merged = doc.merge_notes([left, right])
    assert view.selected_ids() == [merged]
    assert view.roll.items_by_id[merged].label == "가"


def test_zoom_keeps_anchor(setup):
    doc, view = setup
    view.horizontalScrollBar().setValue(100)
    t_before = view.timeline.x_to_time(300)
    view.zoom_time(2.0, t_before)
    assert view.mapper.px_per_sec == pytest.approx(200)
    assert view.timeline.x_to_time(300) == pytest.approx(t_before, abs=0.01)
    assert view.roll.items_by_id["a"].rect().left() == pytest.approx(200.0)
