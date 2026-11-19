"""STEP 3 가사 화면 시험: 가사 탭·가사 레인·피아노 롤 가사·재생 연동·전체 흐름 (모델은 시험용 대역)."""

import json
import os

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from amp.app.document import ProjectDocument
from amp.app.lyrics_controller import read_text_file
from amp.app.timeline import TimelineState
from amp.app.widgets.lyric_lane import LyricLane
from amp.core.model import AnalysisResult
from amp.core.tempo import TempoMap
from amp.lyrics import engine, linking, ops
from amp.lyrics.model import AsrSegment, AsrWord, Recognition
from tests.conftest import note
from tests.test_lyrics import FakeRunner
from tests.ui.test_main_window import IdentityPipeline, create


def _doc_with(text, spans, notes=()):
    doc = ProjectDocument()
    if notes:
        doc.set_analysis(AnalysisResult(10.0, TempoMap.constant(120), tuple(notes)))
    st = ops.apply_text(doc.lyrics, text)
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {ids[i]: sp for i, sp in enumerate(spans)}, source="model")
    if notes:
        st = linking.auto_link(st, list(notes))
    doc.push_lyrics(st, "시험 가사")
    doc.undo_stack.clear()
    return doc, ids


@pytest.fixture
def lane(qtbot):
    def make(doc):
        tl = TimelineState()
        tl.update_view(100.0, 0.0)  # 1초 = 100px
        w = LyricLane(doc, tl)
        qtbot.addWidget(w)
        w.resize(1000, 52)
        w.show()
        qtbot.waitExposed(w)
        return w

    return make


def test_lane_drag_moves_syllable_and_undo(qtbot, lane):
    doc, (a, b, c) = _doc_with("가나다", [(1.0, 1.5), (2.0, 2.5), (3.0, 3.5)])
    w = lane(doc)
    y = 26
    QTest.mousePress(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(225, y))
    for x in (235, 250, 260):
        QTest.mouseMove(w, QPoint(x, y))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(260, y))
    t = doc.lyrics.alignment.timing(b)
    assert (t.start, t.end) == pytest.approx((2.35, 2.85)) and t.user_edited and t.start_source == "user"
    assert w.selected == [b]
    doc.undo_stack.undo()
    assert doc.lyrics.alignment.timing(b).start == pytest.approx(2.0)


def test_lane_boundary_drag_moves_touching_neighbour(qtbot, lane):
    doc, (a, b) = _doc_with("가나", [(1.0, 1.5), (1.5, 2.0)])
    w = lane(doc)
    QTest.mousePress(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(149, 26))
    for x in (155, 165, 170):
        QTest.mouseMove(w, QPoint(x, 26))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(170, 26))
    ta, tb = doc.lyrics.alignment.timing(a), doc.lyrics.alignment.timing(b)
    assert ta.end == pytest.approx(1.71, abs=0.02) and tb.start == pytest.approx(ta.end)  # 경계 조절
    assert tb.end == pytest.approx(2.0)
    assert doc.undo_stack.count() == 1  # 두 음절 변경이 한 번에 실행 취소된다


def test_lane_click_seeks_and_playhead_finds_current_syllable(qtbot, lane):
    doc, (a, b) = _doc_with("가나", [(1.0, 1.5), (2.0, 2.5)])
    w = lane(doc)
    with qtbot.waitSignal(w.seekRequested) as sig:
        QTest.mouseClick(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(210, 26))
    assert sig.args == [pytest.approx(2.0)]
    assert w.set_playhead(1.2) == a and w.set_playhead(1.7) is None and w.set_playhead(2.4) == b


def test_piano_roll_shows_lyrics_under_notes_for_all_relations(qtbot):
    from amp.app.pianoroll.view import PianoRollView

    notes = [note("n", 1.0, 1.6, 60), note("m1", 2.0, 2.3, 62), note("m2", 2.3, 2.6, 64)]
    doc, ids = _doc_with("사랑 아", [(1.0, 1.3), (1.3, 1.6), (2.0, 2.6)], notes)
    view = PianoRollView(doc, TimelineState())
    qtbot.addWidget(view)
    items = view.roll.items_by_id
    assert items["n"].label == "사랑"  # 여러 음절이 한 음표 (N:1)
    assert items["m1"].label == "아" and items["m2"].label == "–"  # 멜리스마
    assert ["m1", "m2"] in [c for c, _style in view.roll.melisma_chains]
    doc.split_note("n", 1.3)  # 음표를 나누면 각 조각에 자기 음절
    labels = sorted(i.label for i in items.values())
    assert labels == ["–", "랑", "사", "아"]
    doc.undo_stack.undo()
    assert view.roll.items_by_id["n"].label == "사랑"


def test_window_lyrics_tab_selection_sync_playback_and_toggle(qtbot, window):
    w = window
    w.doc.set_analysis(AnalysisResult(10.0, TempoMap.constant(120), (note("x", 1.0, 1.5), note("y", 2.0, 2.5))))
    w.lyrics_panel.editor.setPlainText("가나")
    w.lyrics_panel.applyTextRequested.emit(w.lyrics_panel.editor.toPlainText())
    a, b = [s.id for s in w.doc.lyrics.text.syllables]
    st = ops.set_times(w.doc.lyrics, {a: (1.0, 1.5), b: (2.0, 2.5)})
    w.doc.push_lyrics(linking.auto_link(st, w.doc.score.notes()), "시간")
    # 음표를 고르면 연결된 음절이 강조된다
    w.view.roll.select_ids(["y"])
    w.view.roll.selectionChanged.emit()
    assert w.lyric_lane.selected == [b]
    # 음절을 고르면 연결된 음표가 강조된다
    w.lyric_lane.select([a])
    assert w.view.roll.highlight_notes == {"x"}
    assert "1:1" in w.lyrics_panel.fields["rel"].text()
    # 재생 위치 → 현재 음절·음표 강조
    w.lyrics.on_position(2.1)
    assert w.lyric_lane.current == b and w.view.roll.current_notes == {"y"}
    # 가사 표시 끄기
    w.lyrics.act_show.setChecked(False)
    assert not w.lyric_lane.isVisible() and not w.view.roll.lyrics_visible
    w.lyrics.act_show.setChecked(True)
    assert w.lyric_lane.isVisible()
    # 패널에서 적용한 편집도 실행 취소된다
    w.lyrics_panel.applyTextRequested.emit("가나다")
    assert len(w.doc.lyrics.text.syllables) == 3
    w.doc.undo_stack.undo()
    assert [s.id for s in w.doc.lyrics.text.syllables] == [a, b]


def test_window_alignment_and_recognition_results_are_applied_once(qtbot, window, monkeypatch):
    w = window
    w.doc.set_analysis(AnalysisResult(10.0, TempoMap.constant(120), (note("x", 1.0, 1.5), note("y", 2.0, 2.5))))
    rec = Recognition("turbo", "", "ko", 0.9, "cpu", "t", {}, "s", (
        AsrSegment("seg0", 1.0, 2.5, "가나", -0.2, 0.9, 1.0, (), (AsrWord("가나", 1.0, 2.5, 0.9),)),))
    w.lyrics_panel.align_after.setChecked(False)  # 이 시험은 정렬 결과를 직접 넣는다
    w.session.recognitionReady.emit(rec, False)  # 가사가 비어 있으면 인식 결과로 가사를 만든다
    assert [s.text for s in w.doc.lyrics.text.syllables] == ["가", "나"] and w.doc.lyrics.recognition == rec
    a, b = [s.id for s in w.doc.lyrics.text.syllables]
    w.lyrics_panel.link_after.setChecked(True)
    out = engine.AlignOutput(w.doc.lyrics, results=[{"id": a, "start": 1.0, "end": 1.5, "confidence": 0.8},
                                                    {"id": b, "start": 2.0, "end": 2.5, "confidence": 0.7}],
                             targets={a, b}, info={"name": "MMS_FA", "device": "cpu"}, settings={"mode": "all"})
    n_before = w.doc.undo_stack.count()
    w.session.alignmentReady.emit(out)
    t = w.doc.lyrics.alignment.timing(a)
    assert (t.start, t.start_source) == (1.0, "model")
    assert w.doc.lyrics.alignment.linked_notes(a) == ("x",) and w.doc.lyrics.alignment.linked_notes(b) == ("y",)
    assert w.doc.undo_stack.count() == n_before + 1  # 정렬 + 자동 연결이 한 단계
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.alignment.timing(a).start_source == "asr_word"


def test_piano_roll_puts_lyric_on_main_note_after_approach(qtbot):
    from amp.app.pianoroll.view import PianoRollView

    notes = [note("a", 1.0, 1.12, 62), note("b", 1.12, 1.6, 64)]
    doc, ids = _doc_with("가", [(1.0, 1.6)], notes)
    view = PianoRollView(doc, TimelineState())
    qtbot.addWidget(view)
    items = view.roll.items_by_id
    assert (items["a"].label, items["b"].label) == ("·", "가")  # 짧은 접근음에는 점, 글자는 주 음표에
    assert ["a", "b"] in [c for c, _style in view.roll.melisma_chains]
    doc.push_lyrics(linking.set_approach(doc.lyrics, ids[0], "a", False), "해제")
    assert (items["a"].label, items["b"].label) == ("가", "–")


def test_txt_import_encodings(tmp_path):
    text = "사랑해\n너를"
    for enc in ("utf-8-sig", "cp949", "utf-16"):
        p = tmp_path / f"{enc}.txt"
        p.write_bytes(text.encode(enc))
        assert read_text_file(str(p)).replace("\r\n", "\n") == text


def test_full_flow_with_test_models_save_and_reopen(qtbot, make_window, song, monkeypatch):
    """MP3/WAV 불러오기 → 분리 → 음표 → 가사 입력 → 정렬 → 음절 수정 → 음표 연결 → 저장 → 다시 열기.

    분리는 시험용 모델(_identity), 정렬 모델 출력은 합성 값(FakeRunner)이고 CTC 정렬은 실제 코드다.
    """
    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    w = make_window("a")
    folder = create(qtbot, w, song)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    assert [n.pitch for n in w.doc.score.notes()] == [64, 67, 69]
    w.lyrics_panel.applyTextRequested.emit("미 솔 라")
    ids = [s.id for s in w.doc.lyrics.text.syllables]
    starts = [n.start for n in w.doc.score.notes()]
    fake = FakeRunner([s.reading for s in w.doc.lyrics.text.syllables], starts, 5.0)
    monkeypatch.setattr(engine, "run_worker", fake)
    w.lyrics.align("all")
    qtbot.waitUntil(lambda: not w.session.busy and w.doc.lyrics.alignment.timing(ids[0]).aligned, timeout=60_000)
    assert [w.doc.lyrics.alignment.timing(i).start for i in ids] == pytest.approx(starts, abs=0.05)
    assert [len(w.doc.lyrics.alignment.links.get(i, ())) for i in ids] == [1, 1, 1]
    # 음절 시간 수정 + 연결 확정
    st = ops.set_times(w.doc.lyrics, {ids[0]: (starts[0], starts[0] + 0.5)})
    w.doc.push_lyrics(linking.confirm_links(st, ids), "수정")
    assert w.save()
    lyr = json.loads(folder.paths.lyrics_json.read_text(encoding="utf-8"))
    assert lyr["text"] == "미 솔 라"
    root = str(folder.root)
    w.close()
    w2 = make_window("b")
    assert w2.open_project(os.path.join(root, "project.json"))
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    ly = w2.doc.lyrics
    assert [s.id for s in ly.text.syllables] == ids
    assert ly.alignment.timing(ids[0]).user_edited and ly.alignment.timing(ids[0]).end == pytest.approx(starts[0] + 0.5)
    assert all(lk.confirmed for i in ids for lk in ly.alignment.links[i])
    assert w2.lyrics_panel.editor.toPlainText() == "미 솔 라"
    assert len(w2.lyric_lane._rows) == 3


def test_recognition_runs_alignment_and_links_notes_by_default(qtbot, make_window, song, monkeypatch):
    """인식만으로 끝나면 음절 시간이 단어 시각을 나눈 추정값이다. 기본 설정은 정렬·음표 연결까지 잇는다."""
    from amp.lyrics.model import LyricsState

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    w = make_window("a")
    create(qtbot, w, song)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    notes = w.doc.score.notes()
    starts = [n.start for n in notes]
    # 인식 모델은 세 음절을 한 단어로, 실제보다 넓게 들었다고 하자 (단어 시각만 쓰면 음절 경계가 어긋난다)
    seg = {"start": 0.0, "end": notes[-1].end, "text": "미솔라", "avg_logprob": -0.2, "no_speech_prob": 0.1,
           "compression_ratio": 1.0, "activity": 0.9,
           "words": [{"text": "미솔라", "start": 0.0, "end": notes[-1].end, "probability": 0.9}]}
    readings = [s.reading for s in ops.apply_text(LyricsState(), "미솔라").text.syllables]
    fake = FakeRunner(readings, starts, 5.0, segments=[seg])
    monkeypatch.setattr(engine, "run_worker", fake)
    from amp.app.widgets.lyrics_panel import LyricsPanel

    fresh = LyricsPanel()
    qtbot.addWidget(fresh)
    assert fresh.align_after.isChecked() and fresh.settings()["align_after"] is True  # 기본값: 켜짐
    w.lyrics_panel.align_after.setChecked(True)  # 다른 시험이 저장한 설정과 무관하게
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: "ctc_align" in fake.calls and not w.session.busy
                    and all(w.doc.lyrics.alignment.links.get(s.id) for s in w.doc.lyrics.text.syllables), timeout=60_000)
    assert fake.calls[0] == "asr" and "emissions" in fake.calls
    ly = w.doc.lyrics
    ids = [s.id for s in ly.text.syllables]
    assert [ly.alignment.timing(i).start for i in ids] == pytest.approx(starts, abs=0.05)
    assert all(ly.alignment.timing(i).start_source == "model" for i in ids)
    assert [ly.alignment.linked_notes(i) for i in ids] == [(n.id,) for n in notes]
    # 실행 취소 한 번이면 정렬 전(인식 직후) 상태로 돌아간다
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.alignment.timing(ids[1]).start_source == "estimated" and not w.doc.lyrics.alignment.links


def test_recognition_without_align_after_keeps_estimates_and_says_so(qtbot, window):
    w = window
    w.lyrics_panel.align_after.setChecked(False)
    rec = Recognition("turbo", "", "ko", 0.9, "cpu", "t", {}, "s", (
        AsrSegment("seg0", 1.0, 2.5, "가나", -0.2, 0.9, 1.0, (), (AsrWord("가나", 1.0, 2.5, 0.9),)),))
    w.session.recognitionReady.emit(rec, False)
    assert not w.session.busy  # 정렬을 시작하지 않았다
    assert "F7" in w.statusBar().currentMessage()


# --- STEP 3.5: F6 한 번으로 인식 → 정렬 → 음표 연결 (가짜 모델) ------------------------------------------

def _seg(text, a, b):
    return {"start": a, "end": b, "text": text, "avg_logprob": -0.2, "no_speech_prob": 0.1, "compression_ratio": 1.0,
            "activity": 0.9, "words": [{"text": text, "start": a, "end": b, "probability": 0.9}]}


class _Blocking(FakeRunner):
    """정해진 명령에서 취소될 때까지 기다린다 (작업 프로세스가 오래 걸리는 상황)."""

    def __init__(self, *a, block: str, **kw):
        super().__init__(*a, **kw)
        self.block = block

    def __call__(self, command, args, progress=None, token=None, numba=None):
        if command == self.block:
            self.calls.append(command)
            import time

            from amp.engine.context import AnalysisCancelled

            while not token.cancelled:
                time.sleep(0.01)
            raise AnalysisCancelled("취소")
        return super().__call__(command, args, progress, token, numba)


@pytest.fixture
def analysed(qtbot, make_window, song, monkeypatch):
    """분석이 끝난 창과 '미솔라' 가짜 인식·정렬 대역."""
    from amp.lyrics.model import LyricsState

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    w = make_window("a")
    folder = create(qtbot, w, song)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    assert w.save()  # 연속 작업 전 저장된 상태
    notes = w.doc.score.notes()
    readings = [s.reading for s in ops.apply_text(LyricsState(), "미솔라").text.syllables]
    w.lyrics_panel.align_after.setChecked(True)
    w.lyrics_panel.link_after.setChecked(True)

    def runner(cls=FakeRunner, **kw):
        r = cls(readings, [n.start for n in notes], 5.0, segments=kw.pop("segments", [_seg("미솔라", 0.0, notes[-1].end)]), **kw)
        monkeypatch.setattr(engine, "run_worker", r)
        return r

    return w, folder, notes, runner


def test_chain_full_run_shows_stages_saves_and_reopens(qtbot, analysed, make_window):
    w, folder, notes, runner = analysed
    fake = runner()
    labels = []
    w.session.taskStarted.connect(lambda h, label: labels.append(label))
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=60_000)
    assert fake.calls == ["asr", "emissions", "ctc_align"]  # 앞 단계가 끝난 뒤 다음 단계
    assert labels == ["[1/3] 가사 인식", "[2/3] 가사 시간 정렬"]
    info = w.lyrics_panel.align_info.text()
    assert info.startswith("전체 작업 완료") and "인식" in info and "(CPU)" in info and "프로젝트 저장함" in info
    ly = w.doc.lyrics
    ids = [s.id for s in ly.text.syllables]
    assert all(ly.alignment.timing(i).start_source == "model" for i in ids)
    assert [ly.alignment.linked_notes(i) for i in ids] == [(n.id,) for n in notes]
    assert not w.doc.is_dirty and folder.paths.alignment_json.exists()
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    w.close()
    w2 = make_window("b")
    assert w2.open_project(os.path.join(root, "project.json"))
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    ly2 = w2.doc.lyrics
    assert [s.id for s in ly2.text.syllables] == ids and ly2.alignment.links == ly.alignment.links


def test_chain_cancel_during_recognition_changes_nothing(qtbot, analysed):
    w, _folder, notes, runner = analysed
    fake = runner(_Blocking, block="asr")
    before = (w.doc.lyrics, list(w.doc.score.notes()))
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: "asr" in fake.calls, timeout=10_000)
    w.session.cancel_analysis()
    qtbot.waitUntil(lambda: not w.session.busy and w.lyrics.chain is None, timeout=20_000)
    assert (w.doc.lyrics, list(w.doc.score.notes())) == before
    assert "취소" in w.lyrics_panel.align_info.text() and "F6" in w.lyrics_panel.align_info.text()
    assert "emissions" not in fake.calls


def test_chain_cancel_during_alignment_keeps_text_and_marks_estimates(qtbot, analysed):
    w, _folder, notes, runner = analysed
    fake = runner(_Blocking, block="emissions")
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: "emissions" in fake.calls, timeout=20_000)
    w.session.cancel_analysis()
    qtbot.waitUntil(lambda: not w.session.busy and w.lyrics.chain is None, timeout=20_000)
    ly = w.doc.lyrics
    assert [s.text for s in ly.text.syllables] == ["미", "솔", "라"]  # 인식 결과는 남는다
    assert all(ly.alignment.timing(s.id).start_source in ("asr_word", "estimated") for s in ly.text.syllables)
    assert not ly.alignment.links  # 추정 시간으로 연결하지 않는다
    info = w.lyrics_panel.align_info.text()
    assert "취소" in info and "추정값" in info and "F7" in info
    assert w.doc.is_dirty  # 끝나지 않은 작업은 저장하지 않는다


def test_chain_link_error_keeps_alignment(qtbot, analysed, monkeypatch):
    w, _folder, _notes, runner = analysed
    runner()

    def boom(*a, **k):
        raise RuntimeError("시험용 연결 오류")

    monkeypatch.setattr(linking, "auto_link", boom)
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=60_000)
    ly = w.doc.lyrics
    assert all(ly.alignment.timing(s.id).start_source == "model" for s in ly.text.syllables)
    assert not ly.alignment.links
    info = w.lyrics_panel.align_info.text()
    assert "음표 연결" in info and "시험용 연결 오류" in info and "음표 자동 연결" in info


def test_chain_gpu_out_of_memory_stops_with_cause(qtbot, analysed):
    from amp.engine.worker_client import WorkerError

    w, _folder, _notes, runner = analysed
    msg = "GPU 여유 메모리가 4.1GB라 large-v3 모델(약 5GB 필요)을 GPU로 돌릴 수 없습니다. 가사 탭에서 모델을 turbo로 바꿔 다시 실행하세요."
    runner(fail=("asr", WorkerError("oom_model", msg)))
    before = w.doc.lyrics
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=20_000)
    assert w.doc.lyrics == before
    assert any("turbo" in e for e in w.errors)  # 원인과 대체 모델 안내
    assert "실패" in w.lyrics_panel.align_info.text() and "turbo" in w.lyrics_panel.align_info.text()


def test_chain_empty_recognition_keeps_existing_lyrics(qtbot, analysed):
    w, _folder, _notes, runner = analysed
    w.lyrics_panel.applyTextRequested.emit("내 가사")
    before_text = w.doc.lyrics.text
    fake = runner(segments=[])
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=20_000)
    assert w.doc.lyrics.text == before_text and "emissions" not in fake.calls
    assert "비어" in w.lyrics_panel.align_info.text()


def _answer(monkeypatch, role):
    from PySide6.QtWidgets import QMessageBox

    def exec_(box):
        for b in box.buttons():
            if box.buttonRole(b) == role:
                b.click()
                return 0
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)


def test_chain_existing_and_user_edited_lyrics_are_protected(qtbot, analysed, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    w, _folder, notes, runner = analysed
    w.lyrics_panel.applyTextRequested.emit("도 레 미")
    ids = [s.id for s in w.doc.lyrics.text.syllables]
    st = ops.set_times(w.doc.lyrics, {ids[0]: (1.0, 1.5)})  # 사용자 수정 시간
    st = linking.set_links(st, ids[0], [notes[0].id])  # 사용자 연결 (확정)
    w.doc.push_lyrics(st, "사용자 편집")
    user_state, user_notes = w.doc.lyrics, list(w.doc.score.notes())
    fake = runner()
    _answer(monkeypatch, QMessageBox.ButtonRole.AcceptRole)  # '인식 원본만 보관'
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=20_000)
    assert w.doc.lyrics.text == user_state.text and w.doc.lyrics.alignment == user_state.alignment
    assert w.doc.lyrics.recognition is not None and "emissions" not in fake.calls
    # '인식 결과로 바꾸기'를 골라도 실행 취소로 사용자 가사·시간·연결이 그대로 돌아온다. 음표는 바뀌지 않는다
    _answer(monkeypatch, QMessageBox.ButtonRole.DestructiveRole)
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=60_000)
    assert [s.text for s in w.doc.lyrics.text.syllables] == ["미", "솔", "라"]
    assert "저장 안 함" in w.lyrics_panel.align_info.text()  # 작업 전에 저장하지 않은 편집이 있었다
    assert list(w.doc.score.notes()) == user_notes
    w.doc.undo_stack.undo()
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.text == user_state.text and w.doc.lyrics.alignment == user_state.alignment


def test_chain_leaves_out_subtitle_hallucinations(qtbot, analysed):
    w, _folder, notes, runner = analysed
    end = notes[-1].end
    runner(segments=[_seg("한글자막 by 한효주", 0.0, 0.3), _seg("미솔라", 0.3, end), _seg("감사합니다.", end, end + 0.5)])
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=60_000)
    ly = w.doc.lyrics
    assert [ln.text for ln in ly.text.lines] == ["미솔라"] and len(ly.recognition.segments) == 3
    assert "✕" in w.lyrics_panel.rec_view.toPlainText() and "가사에 넣지 않음" in w.lyrics_panel.rec_view.toPlainText()


def test_chain_with_only_hallucinations_keeps_existing_lyrics(qtbot, analysed):
    w, _folder, _notes, runner = analysed
    w.lyrics_panel.applyTextRequested.emit("내 가사")
    before = w.doc.lyrics.text
    fake = runner(segments=[_seg("한글자막 by 한효주", 0.0, 1.0), _seg("감사합니다.", 2.0, 3.0)])
    w.lyrics.recognize()
    qtbot.waitUntil(lambda: w.lyrics.chain is None and not w.session.busy, timeout=20_000)
    assert w.doc.lyrics.text == before and "emissions" not in fake.calls
    assert "비어" in w.lyrics_panel.align_info.text()


def test_play_from_syllable_and_syllable_loop_on_real_player(qtbot, make_window, song):
    """사용자 Windows 로그: 가사 레인에서 '여기서 재생'·'구간 반복'을 쓰면 TypeError('PlaybackState' object is not callable).
    player.state는 속성인데 함수처럼 불렀다. 실제 재생기로 두 경로를 지나 본다."""
    from amp.app.player import AudioPlayer, PlaybackState

    w = make_window("p")
    create(qtbot, w, song)
    st = ops.apply_text(w.doc.lyrics, "미솔라")
    ids = [s.id for s in st.text.syllables]
    w.doc.push_lyrics(ops.set_times(st, {ids[0]: (1.0, 1.8), ids[1]: (1.9, 2.7), ids[2]: (2.8, 4.0)}, source="model"), "시간")
    w.lyric_lane.playFromRequested.emit(1.9)  # 가사 레인 '여기서 재생'
    can_play = AudioPlayer.output_available()
    if can_play:
        qtbot.waitUntil(lambda: w.player.state is PlaybackState.PLAYING, timeout=3000)
    w.player.stop()
    w.lyrics.on_syllable_action("loop", [ids[1]])  # 구간 반복 1.9~2.7초
    assert w.lyrics.loop == (1.9, 2.7)
    if can_play:
        qtbot.waitUntil(lambda: w.player.state is PlaybackState.PLAYING, timeout=3000)
        w.lyrics.on_position(2.75)  # 끝을 지나면 처음으로
        assert abs(w.player.position() - 1.9) < 0.1
    w.lyrics.stop_loop()
    w.player.stop()
    assert w.errors == []
