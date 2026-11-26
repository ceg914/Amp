"""v0.7.0 화면 시험: 선택 구간만 다시 인식 — 인식 원본의 그 구간만 바꾸고 가사 텍스트·시간·연결은 그대로, 실행 취소 가능."""

from amp.lyrics import review
from amp.lyrics.model import AsrSegment, Recognition
from tests.conftest import note
from tests.ui.test_vocal_ui import _setup


def test_region_recognition_keeps_user_lyrics_and_is_undoable(qtbot, window):
    from dataclasses import replace

    w = window
    ids = _setup(w, "같이 아", ((1.0, 1.5), (1.5, 2.0), (2.0, 3.2)))
    old = Recognition("turbo", language="ko", segments=(AsrSegment("seg0", 1.0, 2.0, "가치"),
                                                         AsrSegment("seg1", 2.0, 3.2, "아")))
    w.doc.push_lyrics(replace(w.doc.lyrics, recognition=old), "시험 인식")
    before = (w.doc.lyrics.text, w.doc.lyrics.alignment)
    w.lyric_lane.select([ids[0], ids[1]])
    assert w.lyrics.region_for_recognition() == (1.0, 2.0)
    new = Recognition("turbo", language="ko", segments=(AsrSegment("seg0", 1.05, 1.95, "같이"),))
    w.lyrics.on_region_recognition(new, (1.0, 2.0))
    rec = w.doc.lyrics.recognition
    assert [s.text for s in rec.segments] == ["같이", "아"] and review.R_REGION in rec.segments[0].review
    assert (w.doc.lyrics.text, w.doc.lyrics.alignment) == before  # 사용자 가사는 그대로
    assert "인식 원본만 바꿈" in w.statusBar().currentMessage()
    assert w.doc.undo_stack.undoText() == "구간 다시 인식"
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.recognition == old
    w.lyric_lane.select([])
    assert w.lyrics.region_for_recognition() is None
    w.lyrics.recognize_region()
    assert "다시 인식할 구간이 없습니다" in w.statusBar().currentMessage()
    assert note  # 공용 도우미
