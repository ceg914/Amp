"""STEP 4 화면 시험: 선택 음절 패널(발음·IPA·길이·멜리스마), 수동 수정과 실행 취소, 표시 방식, 저장·다시 열기, 성능."""

import time

import numpy as np
from PySide6.QtGui import QImage, QPainter

from amp.core.model import AnalysisResult, PitchCurve
from amp.core.pitch import midi_to_hz
from amp.core.tempo import TempoMap
from amp.lyrics import linking, ops
from amp.lyrics.model import NoteLink
from tests.conftest import note
from tests.ui.test_main_window import create

FR = 100.0


def _curve(segs, total=10.0):
    n = int(total * FR)
    f0 = np.full(n, np.nan, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    for a, b, p in segs:
        i, j = int(a * FR), int(b * FR)
        f0[i:j] = midi_to_hz(float(p))
        conf[i:j] = 0.9
    return PitchCurve(0.0, FR, f0, conf)


def _zoom_to(w, px_per_sec: float) -> None:
    w.view.zoom_time(px_per_sec / w.view.mapper.px_per_sec)


def _setup(w, text="같이 아", spans=((1.0, 1.5), (1.5, 2.0), (2.0, 3.2)), notes=None, links=None, curve=True):
    notes = notes or [note("n1", 1.0, 1.5, 60), note("n2", 1.5, 2.0, 62), note("m1", 2.0, 2.4, 64),
                      note("m2", 2.4, 2.8, 65), note("m3", 2.8, 3.2, 67)]
    c = _curve([(n.start, n.end, n.pitch) for n in notes]) if curve else None
    w.doc.set_analysis(AnalysisResult(10.0, TempoMap.constant(120), tuple(notes), pitch_curve=c,
                                      onset_envelope=np.zeros(1000) if curve else None))
    w.lyrics_panel.applyTextRequested.emit(text)
    ids = [s.id for s in w.doc.lyrics.text.syllables]
    st = ops.set_times(w.doc.lyrics, {ids[i]: sp for i, sp in enumerate(spans)}, source="model")
    if links is None:
        st = linking.auto_link(st, w.doc.score.notes())
    else:
        st = st.__class__(st.text, st.alignment.with_links({ids[i]: [NoteLink(n, 0.9, "user", True) for n in ns]
                                                            for i, ns in links.items()}), st.recognition, st.history,
                          st.pronunciation)
    w.doc.push_lyrics(st, "시험")
    return ids


def test_selected_syllable_panel_shows_pronunciation_lengths_and_melisma(qtbot, window):
    w = window
    ids = _setup(w, links={0: ["n1"], 1: ["n2"], 2: ["m1", "m2", "m3"]})
    w.lyric_lane.select([ids[1]])
    f = w.lyrics_panel.fields
    assert f["pron"].text().startswith("[치]") and f["ipa"].text() == "/tɕʰi/"
    assert "구개음화" in f["pron_info"].text()
    assert f["span"].text().startswith("500 ms") and f["voiced"].text().startswith("500 ms")
    w.lyric_lane.select([ids[2]])
    assert f["mel"].text().startswith("멜리스마 (자동 판정")
    assert "E4 → F4 → G4" in f["mel_pitch"].text() and "범위 3반음" in f["mel_pitch"].text()
    assert w.vocal.melisma_records()[0]["syllable_id"] == ids[2]
    # 피아노 롤: 멜리스마 묶음은 판정 결과로 잇는다
    assert (["m1", "m2", "m3"], "auto") in w.view.roll.melisma_chains
    # 피치가 없으면 유성 길이는 0이 아니라 미측정
    w.doc.set_analysis(AnalysisResult(10.0, TempoMap.constant(120), tuple(w.doc.score.notes())))
    w.lyric_lane.select([ids[0]])
    assert f["voiced"].text().startswith("미측정")


def test_pronunciation_edit_undo_and_text_change_keeps_user_value(qtbot, window):
    w = window
    ids = _setup(w)
    p = w.lyrics_panel
    w.lyric_lane.select([ids[1]])
    p.pron_edit.setText("티")
    p.btn_pron_apply.click()
    e = w.doc.lyrics.pronunciation[ids[1]]
    assert e.user_edited and e.text == "티" and e.ipa == "tʰi"
    assert "사용자 확정" in p.fields["pron"].text()
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.pronunciation[ids[1]].text == "치" and not w.doc.lyrics.pronunciation[ids[1]].user_edited
    w.doc.undo_stack.redo()
    # 앞 음절을 고치면 자동 발음이 바뀌므로 사용자 값은 지키고 '재검토 필요'
    p.applyTextRequested.emit("갈이 아")
    e = w.doc.lyrics.pronunciation[ids[1]]
    assert e.text == "티" and e.stale and "재검토 필요" in p.fields["pron"].text()
    # 전체 다시 생성은 사용자 값을 덮지 않는다
    w.vocal.on_action("pron_regen_all", [])
    assert w.doc.lyrics.pronunciation[ids[1]].text == "티"
    # 기본 발음으로 (가사 레인 메뉴 경로)
    w.lyrics.on_syllable_action("pron_reset", [ids[1]])
    assert not w.doc.lyrics.pronunciation[ids[1]].user_edited
    # 확정하면 사용자 발음으로 보호된다
    w.lyrics.on_syllable_action("pron_confirm", [ids[0]])
    assert w.doc.lyrics.pronunciation[ids[0]].user_edited


def test_melisma_user_actions_with_piano_roll_selection_and_undo(qtbot, window):
    w = window
    # 같은 음 두 개: 자동은 '불확실' (재발음 근거 없음)
    notes = [note("a", 1.0, 1.5, 60), note("b", 1.5, 2.0, 60), note("c", 2.0, 2.5, 64)]
    ids = _setup(w, "아 나", ((1.0, 2.0), (2.0, 2.5)), notes, links={0: ["a", "b"], 1: ["c"]})
    g = w.vocal.groups[ids[0]]
    assert g.kind == "uncertain" and not g.is_melisma
    w.lyric_lane.select([ids[0]])
    w.view.roll.select_ids(["a", "b"])
    w.lyrics.on_syllable_action("mel_set", [ids[0]])
    d = w.doc.lyrics.alignment.melisma[ids[0]]
    assert d.kind == "melisma" and d.note_ids == ("a", "b")
    assert w.vocal.groups[ids[0]].is_melisma and "사용자 확정" in w.lyrics_panel.fields["mel"].text()
    assert (["a", "b"], "user") in w.view.roll.melisma_chains
    # 음표 'c'를 더하면 연결도 확정으로 생긴다
    w.view.roll.select_ids(["c"])
    w.lyrics.on_syllable_action("mel_add", [ids[0]])
    assert w.doc.lyrics.alignment.melisma[ids[0]].note_ids == ("a", "b", "c")
    assert "c" in w.doc.lyrics.alignment.linked_notes(ids[0])
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.alignment.melisma[ids[0]].note_ids == ("a", "b")
    w.lyrics.on_syllable_action("mel_not", [ids[0]])
    assert not w.vocal.groups[ids[0]].is_melisma
    w.lyrics.on_syllable_action("mel_clear", [ids[0]])
    assert ids[0] not in w.doc.lyrics.alignment.melisma
    # 음표를 나누면 사용자 판정은 남고 재확인 필요
    w.view.roll.select_ids(["a", "b"])
    w.lyrics.on_syllable_action("mel_set", [ids[0]])
    w.doc.split_note("b", 1.75)
    d = w.doc.lyrics.alignment.melisma[ids[0]]
    assert d.needs_review and len(d.note_ids) == 3 and w.vocal.groups[ids[0]].needs_review
    w.lyrics.on_syllable_action("mel_reviewed", [ids[0]])
    assert not w.doc.lyrics.alignment.melisma[ids[0]].needs_review


def test_display_modes_change_labels_lane_and_overlapping_labels_are_skipped(qtbot, window):
    w = window
    ids = _setup(w, links={0: ["n1"], 1: ["n2"], 2: ["m1", "m2", "m3"]})
    roll = w.view.roll
    item = roll.items_by_id["n2"]
    assert roll.label_lines(item) == [("lyric", "이")]
    w.vocal.mode_actions["all"].trigger()
    assert roll.label_lines(item) == [("lyric", "이"), ("pron", "[치]"), ("ipa", "/tɕʰi/")]
    assert w.lyric_lane.height() == 68 and w.lyric_lane.sub_mode == "pron"
    w.vocal.mode_actions["pron"].trigger()
    assert roll.label_lines(item) == [("pron", "[치]")]
    w.vocal.mode_actions["lyrics_ipa"].trigger()
    assert w.lyric_lane.sub_mode == "ipa"
    w.vocal.act_lengths.setChecked(True)
    assert w.lyric_lane.show_lengths
    w.vocal.mode_actions["lyrics"].trigger()
    w.vocal.act_lengths.setChecked(False)
    assert w.lyric_lane.height() == 52

    def render():
        img = QImage(1600, 900, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        roll.render(p, source=roll.sceneRect())
        p.end()
        return roll.labels_hidden

    # 같은 높이로 촘촘히 이어지는 음표: 많이 줄이면 글상자가 겹쳐 일부를 생략하고, 늘리면 모두 그린다
    qs = [note(f"q{i}", 4.0 + i * 0.2, 4.0 + i * 0.2 + 0.18, 60) for i in range(6)]
    _setup(w, "가나다라마바", tuple((q.start, q.end) for q in qs), qs, links={i: [q.id] for i, q in enumerate(qs)})
    w.vocal.mode_actions["all"].trigger()
    _zoom_to(w, 400.0)
    assert render() == 0
    _zoom_to(w, 8.0)
    assert render() > 0
    assert ids


def test_pronunciation_and_melisma_survive_save_and_reopen(qtbot, make_window, song):
    w = make_window("a")
    folder = create(qtbot, w, song)
    notes = [note("a", 1.0, 1.4, 64), note("b", 1.4, 1.9, 67), note("c", 1.9, 2.7, 69)]
    ids = _setup(w, "꽃잎", ((1.0, 1.9), (1.9, 2.7)), notes, links={0: ["a", "b"], 1: ["c"]})
    w.lyrics.on_syllable_action("pron_confirm", [ids[0]])
    w.lyric_lane.select([ids[1]])
    w.lyrics_panel.pron_edit.setText("닢")
    w.lyrics_panel.btn_pron_apply.click()
    w.view.roll.select_ids(["a", "b"])
    w.lyrics.on_syllable_action("mel_set", [ids[0]])
    before = (w.doc.lyrics.pronunciation, w.doc.lyrics.alignment.melisma)
    assert w.save()
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    w.close()
    w2 = make_window("b")
    assert w2.open_project(root + "/project.json")
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    assert (w2.doc.lyrics.pronunciation, w2.doc.lyrics.alignment.melisma) == before
    assert w2.vocal.groups[ids[0]].is_melisma and w2.vocal.groups[ids[0]].source == "user"
    assert not w2.doc.is_dirty


def test_many_syllables_refresh_and_render_quickly(qtbot, window):
    """수백 개 음절·음표: 판정 다시 계산과 화면 그리기 시간 (성능 측정값은 보고서 10절)."""
    w = window
    n = 600
    notes = [note(f"x{i}", 1.0 + i * 0.25, 1.0 + i * 0.25 + 0.22, 60 + (i % 7)) for i in range(n)]
    text = "\n".join("가나다라마바사아자차" for _ in range(n // 10))
    spans = tuple((1.0 + i * 0.25, 1.0 + i * 0.25 + 0.24) for i in range(n))
    links = {i: [f"x{i}"] for i in range(n)}
    _setup(w, text, spans, notes, links=links)
    t0 = time.perf_counter()
    w.vocal.refresh()
    refresh_ms = (time.perf_counter() - t0) * 1000
    w.vocal.mode_actions["all"].trigger()
    _zoom_to(w, 120.0)
    img = QImage(1600, 900, QImage.Format.Format_ARGB32)
    t0 = time.perf_counter()
    p = QPainter(img)
    w.view.roll.render(p, source=w.view.roll.sceneRect())
    p.end()
    render_ms = (time.perf_counter() - t0) * 1000
    print(f"\n음절·음표 {n}개: 판정·길이 다시 계산 {refresh_ms:.0f}ms, 장면 전체 그리기 {render_ms:.0f}ms")
    assert len(w.vocal.groups) == n and refresh_ms < 2000

