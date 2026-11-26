"""v0.7.0 Feature H: 영어 발음(CMU 발음 사전 자료, 앱이 켤 때만)과 사용자 발음 사전. 시험은 작은 가짜 사전 파일을 쓴다."""

from __future__ import annotations

import pytest

from amp.lyrics import ops
from amp.lyrics import pronunciation as P
from amp.lyrics.model import LyricsState
from amp.lyrics.pronunciation import dictionary as D
from amp.lyrics.pronunciation import english as EN

MINI = """;;; 시험용
hello HH AH0 L OW1
hello(2) HH EH0 L OW1
beautiful B Y UW1 T AH0 F AH0 L
baby B EY1 B IY0
extra EH1 K S T R AH0
love L AH1 V
fire F AY1 ER0
everything EH1 V R IY0 TH IH2 NG
"""


@pytest.fixture
def english(tmp_path):
    p = tmp_path / "cmudict.dict"
    p.write_text(MINI, encoding="utf-8")
    assert P.enable_english(p)
    yield p
    P.disable_english()


@pytest.fixture(autouse=True)
def clean():
    yield
    P.disable_english()


def test_syllabify_maximal_onset():
    assert EN.syllabify("B Y UW1 T AH0 F AH0 L".split()) == [["B", "Y", "UW1"], ["T", "AH0"], ["F", "AH0", "L"]]
    assert EN.syllabify("EH1 K S T R AH0".split()) == [["EH1", "K"], ["S", "T", "R", "AH0"]]
    assert EN.syllabify("HH AH0 L OW1".split()) == [["HH", "AH0"], ["L", "OW1"]]
    assert EN.ipa_of(["L", "AH1", "V"]) == "ˈlʌv" and EN.ipa_of(["T", "AH0"]) == "tə"


def test_english_off_by_default_keeps_previous_behaviour():
    st = P.sync(ops.apply_text(LyricsState(), "love baby"))
    assert all(e.status == "unavailable" and e.ipa == "" for e in st.pronunciation.values())
    assert not P.english_enabled()


def test_english_adapter_per_syllable_mismatch_and_oov(english):
    st = P.sync(ops.apply_text(LyricsState(), "hello beautiful baby 사랑 love zork"))
    es = {s.text: st.pronunciation[s.id] for s in st.text.syllables}
    assert es["hel"].ipa == "hə" and es["lo"].ipa == "ˈloʊ" and es["hel"].status == "valid"
    assert "다른 발음" in es["hel"].notes[0]
    assert [es[k].ipa for k in ("beau", "ti", "ful")] == ["ˈbju", "tə", "fəl"]
    assert es["ba"].ipa == "ˈbeɪ" and es["by"].ipa == "bi" and es["love"].text == "lʌv"
    assert es["zork"].status == "unavailable" and "사전에 없는" in es["zork"].notes[0]
    assert es["사"].text == "사" and es["사"].language == "ko"  # 한국어는 그대로
    fire = P.sync(ops.apply_text(LyricsState(), "fire"))  # 가사 1음절, 사전 2음절 → 이어 적음
    e = next(iter(fire.pronunciation.values()))
    assert e.status == "valid" and e.ipa == "faɪɚ" and "한 음절" in e.notes[-1]
    assert es["love"].ipa == "lʌv"  # 한 음절 낱말은 강세 표시 없음
    many = P.sync(ops.apply_text(LyricsState(), "everything"))  # 가사 4음절(e·ve·ry·thing), 사전 3음절 → 불확실
    es2 = [many.pronunciation[s.id] for s in many.text.syllables]
    assert len(es2) == 4 and all(e.status == "uncertain" for e in es2) and "음절 수(3)" in es2[0].notes[-1]
    assert es2[0].ipa == "ˈɛv.ɹi.ˌθɪŋ" and es2[1].ipa == ""


def test_user_dictionary_word_and_syllable(tmp_path):
    d = D.PronDictionary().add(D.DictEntry("마음", "마흠", language="ko")).add(D.DictEntry("love", "러.브", "ɾʌ.bɯ", "ko"))
    d = d.add(D.DictEntry("라", "롸", scope="syllable"))
    base = P.sync(ops.apply_text(LyricsState(), "내 마음 라 love"))
    st, changed, kept = P.set_dictionary(base, d.as_tuple(), "발음 사전 수정")
    assert changed == 2 and kept == 0 and st.history[-1].action == "발음 사전 수정"
    es = {s.text: st.pronunciation[s.id] for s in st.text.syllables}
    assert es["마"].text == "마" and es["음"].text == "흠" and "사용자 사전" in es["음"].rules
    assert es["음"].ipa  # 한국어 IPA는 사전 발음으로 다시 만든다
    assert es["라"].text == "롸"
    assert "음절 수(2)" in es["love"].notes[-1] and es["love"].status == "unavailable"  # 맞지 않으면 적용하지 않음
    sid = next(s.id for s in st.text.syllables if s.text == "음")
    st2 = P.set_user(st, sid, "음")  # 직접 고친 발음은 사전보다 우선
    st3, _n, kept = P.regenerate(st2)
    assert st3.pronunciation[sid].text == "음" and kept >= 1
    st4, _n, kept = P.set_dictionary(st3, ())  # 사전을 비우면 자동 발음은 규칙으로, 사용자 발음은 그대로
    assert st4.pronunciation[sid].text == "음" and kept == 1
    ra = next(s.id for s in st4.text.syllables if s.text == "라")
    assert st4.pronunciation[ra].text == base.pronunciation[ra].text == "나"  # 사전 없이 규칙(ㅁ 뒤 ㄹ → ㄴ)으로
    path = tmp_path / "사전" / ("x" + D.FILE_SUFFIX)
    D.save(path, d)
    back = D.load(path)
    assert len(back) == 3 and back.word("마음").text == "마흠" and back.syllable("라").text == "롸"
    path.write_text("{깨짐", encoding="utf-8")
    assert len(D.load(path)) == 0
    assert D.split_units("きょう") == ["きょ", "う"] and D.split_units("러.브") == ["러", "브"]
    assert D.split_units("がっこう") == ["が", "っ", "こ", "う"]


def test_dictionary_is_per_project_and_saved_with_lyrics(tmp_path):
    """사전은 가사 상태 안에 있다: 다른 가사 상태(=다른 프로젝트)의 자동 발음에 쓰이지 않고 lyrics.json과 함께 저장·복원된다."""
    from amp.lyrics import io as LIO

    a = P.sync(ops.apply_text(LyricsState(), "마음"))
    a, _n, _k = P.set_dictionary(a, (D.DictEntry("마음", "마흠", language="ko"),))
    b = P.sync(ops.apply_text(LyricsState(), "마음"))
    assert [b.pronunciation[s.id].text for s in b.text.syllables] == ["마", "음"]
    assert [a.pronunciation[s.id].text for s in a.text.syllables] == ["마", "흠"]
    warnings: list = []
    body = LIO.lyrics_to_dict(a)
    assert body["pron_dictionary"]["entries"][0]["key"] == "마음"
    back = LIO.state_from_dicts(body, LIO.alignment_to_dict(a), warnings)
    assert back.pron_dictionary == a.pron_dictionary and not warnings
    assert P.sync(back).pronunciation == a.pronunciation
    assert "pron_dictionary" not in LIO.lyrics_to_dict(b)  # 사전이 없으면 이전 형식과 같다
    bad = dict(body, pron_dictionary={"entries": [{"key": "x"}, {"key": "y", "text": "와", "scope": "?"}]})
    w2: list = []
    assert LIO.state_from_dicts(bad, None, w2).pron_dictionary == () and len(w2) == 1


def test_dictionary_entry_from_word_and_user_value_not_marked_stale():
    st = P.sync(ops.apply_text(LyricsState(), "마음 12"))
    sylls = st.text.syllables
    st = P.set_user(st, sylls[1].id, "흠")
    word = st.text.lines[0].words[0]
    e, why = D.entry_for_word(word.text, [st.pronunciation[s.id] for s in word.syllables])
    assert why == "" and e.text == "마.흠" and e.language == "ko" and e.ipa.count(".") == 1
    st2, _n, kept = P.set_dictionary(st, (e,))
    assert kept == 1 and not st2.pronunciation[sylls[1].id].stale  # 사용자 값과 같은 사전 값으로 재검토 표시를 붙이지 않음
    num = st.text.lines[0].words[1]
    e2, _w = D.entry_for_word(num.text, [st.pronunciation[s.id] for s in num.syllables])
    assert e2 is None or "." not in e2.ipa or e2.ipa.count(".") == len(num.syllables) - 1
    assert D.entry_for_word("x", [None])[0] is None


class _Feat:
    def __init__(self, kana):
        self.kana = kana


class _Word:
    def __init__(self, surface, kana, ws=""):
        self.surface, self.feature, self.white_space = surface, _Feat(kana), ws


def _fake_tagger():
    """MeCab 대신 쓰는 작은 분석기 (시험이 fugashi 설치 여부에 기대지 않게)."""
    table = {"今日": "キョウ", "は": "ハ", "心": "ココロ", "が": "ガ", "痛い": "イタイ", "名前": "ナマエ", "を": "ヲ"}

    def tag(text):
        out, i, ws = [], 0, ""
        while i < len(text):
            if text[i] == " ":
                ws += " "
                i += 1
                continue
            for n in (2, 1):
                if text[i:i + n] in table:
                    out.append(_Word(text[i:i + n], table[text[i:i + n]], ws))
                    ws = ""
                    i += n
                    break
            else:
                out.append(_Word(text[i], "*", ws))
                ws = ""
                i += 1
        return out

    return tag


def test_kanji_reading_with_line_context_and_split_uncertainty():
    from amp.lyrics.pronunciation import kanji as K

    assert K.split_reading("痛い", "イタイ") == [("痛", "イタ")]
    assert K.split_reading("今日", "キョウ") == [("今日", "キョウ")]
    assert P.enable_kanji(_fake_tagger)
    try:
        st = P.sync(ops.apply_text(LyricsState(), "今日は心が痛い 名前"))
        es = {s.text: st.pronunciation[s.id] for s in st.text.syllables}
        assert es["心"].text == "ココロ" and es["心"].status == "valid" and es["心"].ipa and es["心"].language == "ja"
        assert es["痛"].text == "イタ" and es["痛"].status == "valid"  # 오쿠리가나 'い'는 가나 변환기가 맡는다
        assert es["い"].text == "イ"
        assert es["今"].text == "キョウ" and es["日"].text == "" and es["今"].status == es["日"].status == "uncertain"
        assert K.NOTE_SPLIT in es["今"].notes
        assert es["名"].text == "ナマエ" and es["前"].status == "uncertain"
    finally:
        P.disable_kanji()
    st = P.sync(ops.apply_text(LyricsState(), "心"))  # 끄면 이전처럼 변환 불가
    assert next(iter(st.pronunciation.values())).status == "unavailable"


def test_kanji_reading_failure_is_isolated():
    def broken():
        raise RuntimeError("사전 없음")

    assert P.enable_kanji(broken)
    try:
        st = P.sync(ops.apply_text(LyricsState(), "心 사랑"))
        es = {s.text: st.pronunciation[s.id] for s in st.text.syllables}
        assert es["心"].status == "unavailable" and "열지 못했습니다" in es["心"].notes[0]
        assert es["사"].text == "사"  # 다른 언어는 그대로
    finally:
        P.disable_kanji()


def test_real_mecab_unidic_if_installed(monkeypatch):
    from amp.lyrics.pronunciation import kanji as K

    monkeypatch.undo()  # 공용 픽스처의 '설치 안 됨' 가정을 풀고 실제 설치 여부를 본다
    if not K.available():
        pytest.skip("fugashi·UniDic이 설치되어 있지 않음 (선택 설치)")
    assert P.enable_kanji()
    try:
        st = P.sync(ops.apply_text(LyricsState(), "心が痛い"))
        es = {s.text: st.pronunciation[s.id] for s in st.text.syllables}
        assert es["心"].text == "ココロ" and es["痛"].text == "イタ"
    finally:
        P.disable_kanji()


def test_real_cmudict_if_installed(monkeypatch):
    monkeypatch.undo()
    p = EN.find_installed()
    if p is None:
        pytest.skip("cmudict 자료가 설치되어 있지 않음")
    assert P.enable_english(p)
    st = P.sync(ops.apply_text(LyricsState(), "tonight beautiful"))
    assert [st.pronunciation[s.id].ipa for s in st.text.syllables] == ["tə", "ˈnaɪt", "ˈbju", "tə", "fəl"]
