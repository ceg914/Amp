"""v0.7.0 화면 시험: 프로젝트 발음 사전 (사전에 추가 단추, 사전 창, 실행 취소, 저장·다시 열기, 프로젝트 사이 격리)."""

from amp.lyrics.model import PronDictEntry
from tests.conftest import note
from tests.ui.test_main_window import create
from tests.ui.test_vocal_ui import _setup


def _prons(w):
    return [w.doc.lyrics.pronunciation[s.id].text for s in w.doc.lyrics.text.syllables]


def test_add_word_to_dictionary_applies_to_other_occurrences_and_undo(qtbot, window):
    w = window
    ids = _setup(w, "마음 마음", ((1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)),
                 [note("a", 1.0, 1.5, 60), note("b", 1.5, 2.0, 62), note("c", 2.0, 2.5, 64), note("d", 2.5, 3.0, 65)])
    assert _prons(w) == ["마", "음", "마", "음"]
    w.lyric_lane.select([ids[1]])
    w.lyrics_panel.pron_edit.setText("흠")
    w.lyrics_panel.btn_pron_apply.click()  # 첫 '마음'의 '음'을 직접 고친다
    assert _prons(w) == ["마", "흠", "마", "음"]
    w.lyric_lane.select([ids[1]])
    assert w.lyrics_panel.btn_pron_dict.isEnabled()
    w.lyrics_panel.btn_pron_dict.click()
    assert w.doc.lyrics.pron_dictionary == (PronDictEntry("마음", "마.흠", w.doc.lyrics.pron_dictionary[0].ipa, "ko", "word"),)
    assert _prons(w) == ["마", "흠", "마", "흠"]  # 같은 단어의 자동 발음에 쓰인다
    e = w.doc.lyrics.pronunciation[ids[3]]
    assert not e.user_edited and "사용자 사전" in e.rules and e.ipa
    assert not w.doc.lyrics.pronunciation[ids[1]].stale  # 사전 값이 사용자 값과 같아 재검토 표시 없음
    assert w.doc.undo_stack.undoText() == "발음 사전에 추가"
    w.doc.undo_stack.undo()
    assert w.doc.lyrics.pron_dictionary == () and _prons(w) == ["마", "흠", "마", "음"]
    w.doc.undo_stack.redo()
    assert len(w.doc.lyrics.pron_dictionary) == 1 and _prons(w) == ["마", "흠", "마", "흠"]
    # 사전 창: 항목을 빼면 자동 발음만 규칙으로 돌아가고 직접 고친 발음은 남는다
    assert w.vocal.apply_dictionary(())
    assert _prons(w) == ["마", "흠", "마", "음"] and w.doc.lyrics.pronunciation[ids[1]].user_edited
    assert not w.vocal.apply_dictionary(())  # 바뀐 것이 없으면 실행 취소 기록을 늘리지 않는다


def test_dictionary_dialog_add_remove_import_export(qtbot, tmp_path, window):
    from amp.app.pron_dictionary_dialog import PronDictionaryDialog

    dlg = PronDictionaryDialog((PronDictEntry("마음", "마흠"),), window, str(tmp_path))
    qtbot.addWidget(dlg)
    assert dlg.table.rowCount() == 1
    assert dlg.add_entry("라", "롸", "syllable") and not dlg.add_entry(" ", "x")
    assert dlg.add_entry("마음", "마.음")  # 같은 키는 새 값으로
    assert [(e.key, e.text, e.scope) for e in dlg.entries()] == [("라", "롸", "syllable"), ("마음", "마.음", "word")]
    from amp.lyrics.pronunciation import dictionary as D

    path = tmp_path / ("shared" + D.FILE_SUFFIX)
    D.save(path, D.PronDictionary.of(dlg.entries()))
    assert dlg.remove_rows([0, 1]) == 2 and dlg.entries() == ()
    assert dlg.import_file(str(path)) == 2 and len(dlg.entries()) == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "다른 것"}', encoding="utf-8")
    assert dlg.import_file(str(bad)) == 0 and len(dlg.entries()) == 2


def test_dictionary_saved_with_project_and_not_shared_with_other_project(qtbot, make_window, song, tmp_path):
    import shutil

    w = make_window("a")
    folder = create(qtbot, w, song)
    ids = _setup(w, "마음", ((1.0, 1.5), (1.5, 2.0)), [note("a", 1.0, 1.5, 60), note("b", 1.5, 2.0, 62)])
    assert w.vocal.apply_dictionary((PronDictEntry("마음", "마흠", language="ko"),))
    assert _prons(w) == ["마", "흠"]
    assert w.save()
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    w.close()
    w2 = make_window("b")
    assert w2.open_project(root + "/project.json")
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    assert w2.doc.lyrics.pron_dictionary == (PronDictEntry("마음", "마흠", language="ko"),)
    assert _prons(w2) == ["마", "흠"] and not w2.doc.is_dirty
    w2.doc.mark_saved(w2.doc.file_path or "")
    w2.close()
    # 다른 프로젝트: 같은 가사라도 사전이 없으니 규칙 발음
    song2 = str(tmp_path / "다른 곡.wav")
    shutil.copy(song, song2)
    w3 = make_window("c")
    create(qtbot, w3, song2)
    _setup(w3, "마음", ((1.0, 1.5), (1.5, 2.0)), [note("a", 1.0, 1.5, 60), note("b", 1.5, 2.0, 62)])
    assert w3.doc.lyrics.pron_dictionary == () and _prons(w3) == ["마", "음"]
    assert ids


def test_window_turns_on_installed_converters_and_skips_in_lite(qtbot, make_window, tmp_path, monkeypatch):
    from amp.lyrics import pronunciation as pron
    from amp.lyrics.pronunciation import english, kanji

    d = tmp_path / "cmudict.dict"
    d.write_text("love L AH1 V\n", encoding="utf-8")
    monkeypatch.setattr(english, "find_installed", lambda find_spec=None: d)
    monkeypatch.setattr(kanji, "available", lambda find_spec=None: True)
    w = make_window("on")
    assert pron.english_enabled() and pron.kanji_enabled()
    w.lyrics_panel.applyTextRequested.emit("love")
    e = next(iter(w.doc.lyrics.pronunciation.values()))
    assert e.ipa == "lʌv" and e.status == "valid"
    pron.disable_english()
    pron.disable_kanji()
    monkeypatch.setenv("SONATLAS_LITE", "1")
    w2 = make_window("lite")
    assert w2.lite and not pron.english_enabled() and not pron.kanji_enabled()
