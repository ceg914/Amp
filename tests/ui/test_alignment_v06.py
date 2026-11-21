"""v0.6 음절 정렬 다듬기: 끄는 중 정보(지금·처음 시각, 차이, 연결 음표, 겹침), 붙이기 끄기, 시간 수정 뒤 연결 방식 4가지
(확정 연결 보호), 새 검토 규칙(음절 몰림·겹침·정렬 확신 낮음·연결 음표가 음절 밖), 검토 항목 느리게 반복.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from amp.compare import review as R
from amp.core.model import Note
from amp.lyrics import linking, ops
from amp.lyrics.model import SyllableTiming
from tests.ui.test_lyrics_ui import _doc_with, lane  # noqa: F401 - 픽스처


def _drag(w, x0, x1, y=26, mods=Qt.KeyboardModifier.NoModifier):
    QTest.mousePress(w, Qt.MouseButton.LeftButton, mods, QPoint(x0, y))
    infos = []
    step = 1 if x1 >= x0 else -1
    for x in range(x0 + step, x1 + step, step * 3):
        QTest.mouseMove(w, QPoint(x, y))
        infos.append(w.drag_info)
    QTest.mouseMove(w, QPoint(x1, y))
    infos.append(w.drag_info)
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, mods, QPoint(x1, y))
    return infos


NOTES = (Note("n1", 1.0, 1.5, 60), Note("n2", 2.0, 2.5, 62), Note("n3", 3.0, 3.5, 64))


def test_drag_shows_times_delta_links_and_overlap(qtbot, lane):
    doc, (a, b, c) = _doc_with("가나다", [(1.0, 1.5), (2.0, 2.5), (3.0, 3.5)], NOTES)
    w = lane(doc)
    infos = _drag(w, 225, 162)  # '나'를 0.63초 앞으로 (1.37~1.87) → '가'(~1.5)와 130ms 겹침
    last = infos[-1]
    assert "처음 2.000s" in last and "ms" in last and "연결 음표: D4 2.00s" in last
    assert "앞 음절과" in last and "겹침" in last
    assert w.drag_info == ""  # 놓으면 지운다


def test_snap_can_be_turned_off(qtbot, lane):
    doc, (a, b, c) = _doc_with("가나다", [(1.0, 1.5), (2.1, 2.5), (3.0, 3.5)], NOTES)
    w = lane(doc)
    w.note_edges = [2.0, 2.5]
    _drag(w, 213, 208)  # 시작 2.1 → 2.05: 2.0 경계 5px 안이라 붙는다
    assert doc.lyrics.alignment.timing(b).start == pytest.approx(2.0)
    doc.undo_stack.undo()
    w.snap_enabled = False
    _drag(w, 213, 208)
    t = doc.lyrics.alignment.timing(b).start
    assert t == pytest.approx(2.05, abs=0.011)


@pytest.mark.parametrize("mode", ["time", "selected", "all", "keep"])
def test_relink_modes_after_time_edit(qtbot, lane, mode):
    doc, (a, b, c) = _doc_with("가나다", [(1.0, 1.5), (2.0, 2.5), (3.0, 3.5)], NOTES)
    # '가'는 사용자가 n1에 확정 연결
    doc.push_lyrics(linking.confirm_links(doc.lyrics, [a]), "확정")
    before = {s: [lk.note_id for lk in v] for s, v in doc.lyrics.alignment.links.items()}
    w = lane(doc)
    w.relink_mode = mode
    _drag(w, 225, 325)  # '나'를 1초 뒤(3.0~3.5)로 → n3와 겹침
    al = doc.lyrics.alignment
    links = {s: [lk.note_id for lk in v] for s, v in al.links.items()}
    assert links[a] == before[a] and al.links[a][0].confirmed  # 확정 연결은 어느 방식에서도 그대로
    if mode == "time":
        assert links[b] == before[b] == ["n2"]
    elif mode == "selected":
        assert links[b] == ["n3"] and links.get(c) == before[c]  # 바꾾 음절만
    elif mode == "all":
        assert links[b] == ["n3"] or links.get(c) == ["n3"]
    else:
        assert links[b] == ["n2"] and al.links[b][0].confirmed
    doc.undo_stack.undo()  # 시간과 연결이 한 번에 돌아간다
    assert {s: [lk.note_id for lk in v] for s, v in doc.lyrics.alignment.links.items()} == before
    assert doc.lyrics.alignment.timing(b).start == pytest.approx(2.0)


def test_new_alignment_review_rules():
    notes = (Note("n1", 1.0, 1.15, 60), Note("n2", 2.0, 3.0, 62), Note("n3", 5.0, 5.5, 64))
    doc, (a, b, c, d) = _doc_with("가나다라", [(1.0, 1.1), (1.05, 1.2), (2.0, 2.5), (3.4, 3.6)], notes)
    st = doc.lyrics
    st = linking.set_links(st, a, ["n1"])
    st = linking.set_links(st, b, ["n1"])  # 짧은 음표 하나에 음절 2개 (0.15초)
    st = linking.set_links(st, d, ["n3"])  # '라'(3.4~3.6)가 5.0초 음표에 연결 → 음절 밖
    al = st.alignment
    low = SyllableTiming(2.0, 2.5, 0.05, "model", "model")
    from dataclasses import replace

    st = replace(st, alignment=al.with_timings({c: low}))
    items = R.lyric_rules(st, {n.id: n for n in notes})
    kinds = {it.kind for it in items}
    assert {"syl_crowded", "syl_overlap", "syl_low_conf", "link_note_moved"} <= kinds
    crowded = next(it for it in items if it.kind == "syl_crowded")
    assert set(crowded.syllable_ids) == {a, b} and crowded.note_ids == ("n1",) or list(crowded.note_ids) == ["n1"]
    assert all(R.KIND_LABELS.get(k) for k in kinds)


def test_review_slow_loop_sets_half_speed(qtbot, window, song):
    from tests.ui.test_main_window import create

    w = window
    create(qtbot, w, song)
    st = ops.apply_text(w.doc.lyrics, "가")
    sid = st.text.syllables[0].id
    w.doc.push_lyrics(ops.set_times(st, {sid: (1.0, 1.4)}, source="model"), "가사")
    w.review.refresh()  # 작업 스레드에서 검사
    qtbot.waitUntil(lambda: any(it.kind == "syl_unlinked" for it, _s in w.review.panel._rows), timeout=10_000)
    rows = w.review.panel._rows
    r = next(i for i, (it, _s) in enumerate(rows) if it.kind == "syl_unlinked")
    w.review.panel.select_row(r)
    w.review.on_action("slow_loop")
    assert w.player.speed == pytest.approx(0.5)
    assert w.player.loop_region is not None and w.player.loop_region[0] <= 1.0 <= w.player.loop_region[1]
