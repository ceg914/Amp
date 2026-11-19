"""STEP 4 발음 변환 시험.

정답은 표준 발음법(국립국어원, 1988 고시)의 조항 예시다. 이 예시들은 규칙을 만들 때 직접 보며 맞춘 자료라
독립 검증 자료가 아니다(보고서 5절). 일본어는 헵번식 박 표와 Help:IPA/Japanese의 대응을 기준으로 했다.
"""

import json

import pytest

from amp.lyrics import io, ops
from amp.lyrics import pronunciation as pron
from amp.lyrics.model import LyricsState
from amp.lyrics.pronunciation import japanese, korean
from amp.project import schema

# 표준 발음법 예시: 원문=발음 (장단 표시는 뺐다). 조항 순서대로.
STANDARD_EXAMPLES = """
닦다=닥따|키읔=키윽|옷=옫|웃다=욷따|있다=읻따|젖=젇|빚다=빋따|꽃=꼳|쫓다=쫃따|솥=솓|뱉다=밷따|앞=압|덮다=덥따
넋=넉|넋과=넉꽈|앉다=안따|여덟=여덜|넓다=널따|외곬=외골|핥다=할따|값=갑|없다=업따|밟다=밥따|밟는=밤는|넓죽하다=넙쭈카다
닭=닥|흙과=흑꽈|맑다=막따|늙지=늑찌|삶=삼|젊다=점따|읊고=읍꼬|읊다=읍따|맑게=말께|묽고=물꼬|얽거나=얼꺼나
놓고=노코|좋던=조턴|쌓지=싸치|많고=만코|않던=안턴|닳지=달치|각하=가카|먹히다=머키다|밝히다=발키다|맏형=마텽|좁히다=조피다|넓히다=널피다|꽂히다=꼬치다|앉히다=안치다
닿소=다쏘|많소=만쏘|싫소=실쏘|놓는=논는|쌓네=싼네|않네=안네|않는=안는|뚫네=뚤레|뚫는=뚤른|낳은=나은|놓아=노아|쌓이다=싸이다|많아=마나|않은=아는|닳아=다라|싫어도=시러도
깎아=까까|옷이=오시|있어=이써|낮이=나지|꽂아=꼬자|꽃을=꼬츨|쫓아=쪼차|밭에=바테|앞으로=아프로|덮이다=더피다
넋이=넉씨|앉아=안자|닭을=달글|젊어=절머|곬이=골씨|핥아=할타|읊어=을퍼|값을=갑쓸|없어=업써
밭 아래=바 다래|늪 앞=느 밥|맛없다=마덥따|넋 없다=너 겁따|닭 앞에=다 가페
곧이듣다=고지듣따|굳이=구지|미닫이=미다지|땀받이=땀바지|밭이=바치|벼훑이=벼훌치|굳히다=구치다|닫히다=다치다|묻히다=무치다
먹는=멍는|국물=궁물|깎는=깡는|키읔만=키응만|몫몫이=몽목씨|긁는=긍는|흙만=흥만|닫는=단는|짓는=진는|옷맵시=온맵씨|있는=인는|맞는=만는|젖멍울=전멍울|쫓는=쫀는|꽃망울=꼰망울|붙는=분는|놓는=논는|잡는=잠는|밥물=밤물|앞마당=암마당|밟는=밤는|읊는=음는|없는=엄는|값매다=감매다
책 넣는다=챙 넌는다|흙 말리다=흥 말리다|옷 맞추다=온 맏추다|밥 먹는다=밤 멍는다|값 매기다=감 매기다
담력=담녁|침략=침냑|강릉=강능|항로=항노|대통령=대통녕|막론=망논|백리=뱅니|협력=혐녁|십리=심니
난로=날로|신라=실라|천리=철리|광한루=광할루|대관령=대괄령|칼날=칼랄|물난리=물랄리|줄넘기=줄럼끼|할는지=할른지|닳는=달른|뚫는=뚤른|핥네=할레
국밥=국빱|깎다=깍따|넋받이=넉빠지|삯돈=삭똔|닭장=닥짱|칡범=칙뻠|뻗대다=뻗때다|옷고름=옫꼬름|있던=읻떤|꽂고=꼳꼬|꽃다발=꼳따발|낯설다=낟썰다|밭갈이=받까리|솥전=솓쩐|곱돌=곱똘|덮개=덥깨|옆집=엽찝|읊조리다=읍쪼리다|값지다=갑찌다
신고=신꼬|껴안다=껴안따|앉고=안꼬|얹다=언따|삼고=삼꼬|더듬지=더듬찌|닮고=담꼬|젊지=점찌|넓게=널께|훑소=훌쏘|떫지=떨찌
할 것을=할 꺼슬|갈 데가=갈 떼가|할 바를=할 빠를|할 수는=할 쑤는|할 적에=할 쩌게|할수록=할쑤록|할게=할께|할걸=할껄
솜이불=솜니불|막일=망닐|꽃잎=꼰닙|맨입=맨닙|할 일=할 릴
늴리리=닐리리|희망=히망|무늬=무니|가져=가저|다쳐=다처|쪄=쩌|같이=가치|사랑해=사랑해|좋아=조아
"""
# 형태소·단어 정보 없이는 맞출 수 없어 알려진 한계로 남긴 예시 (앞 둘은 '불확실' 표시가 붙는다)
KNOWN_MISSES = {"줄넘기": "줄럼기", "껴안다": "껴안다", "솜이불": "소미불"}


def _say(text: str) -> tuple[str, list]:
    st = pron.sync(ops.apply_text(LyricsState(), text))
    es = [st.pronunciation[s.id] for s in st.text.syllables]
    out, k = [], 0
    for word in text.split():
        n = len([c for c in word if korean.is_hangul(c)])
        out.append("".join(e.text for e in es[k:k + n]))
        k += n
    return " ".join(out), es


def test_standard_pronunciation_examples():
    pairs = [x.split("=") for x in STANDARD_EXAMPLES.replace("\n", "|").split("|") if "=" in x]
    assert len(pairs) == 201
    wrong = {src: got for src, want in pairs if (got := _say(src)[0]) != want}
    assert wrong == KNOWN_MISSES
    # 맞추지 못한 셋 중 둘은 사용자에게 '불확실'로 보인다
    assert _say("줄넘기")[1][2].status == "uncertain" and _say("껴안다")[1][2].status == "uncertain"


@pytest.mark.parametrize("text, want, rule", [
    ("사랑해", "사랑해", None),
    ("옷이", "오시", korean.R_LINK), ("넋이", "넉씨", korean.R_LINK), ("앉아", "안자", korean.R_LINK),
    ("국물", "궁물", korean.R_NASAL), ("밥 먹는다", "밤 멍는다", korean.R_NASAL),
    ("국밥", "국빱", korean.R_TENSE), ("할 수", "할 쑤", korean.R_TENSE_MOD),
    ("난로", "날로", korean.R_LIQUID), ("뚫는", "뚤른", korean.R_LIQUID),
    ("값", "갑", korean.R_REP), ("밟다", "밥따", korean.R_REP),
    ("같이", "가치", korean.R_PAL), ("놓고", "노코", korean.R_ASP), ("좋아", "조아", korean.R_H_DROP),
    ("담력", "담녁", korean.R_R_TO_N), ("희망", "히망", korean.R_UI), ("꽃잎", "꼰닙", korean.R_N_INSERT),
])
def test_rules_by_phenomenon(text, want, rule):
    got, es = _say(text)
    assert got == want
    if rule:
        assert any(rule in e.rules for e in es)


def test_spaces_and_punctuation_bound_the_context():
    assert _say("밥 먹어")[0] == "밤 머거"  # 띄어 써도 이어 부르면 비음화 (제18항 붙임)
    assert _say("밥, 먹어")[0] == "밥 머거"  # 쉼표에서는 끊는다
    st = pron.sync(ops.apply_text(LyricsState(), "사랑해!\n꽃 위에"))
    texts = [st.pronunciation[s.id].text for s in st.text.syllables]
    assert texts == ["사", "랑", "해", "꼬", "뒤", "에"]  # 문장부호는 음절이 아니다. 꽃 위[꼬뒤] (제15항)
    # 줄을 넘어서는 이어 바꾸지 않는다
    st = pron.sync(ops.apply_text(LyricsState(), "밥\n먹어"))
    assert [st.pronunciation[s.id].text for s in st.text.syllables] == ["밥", "머", "거"]


def test_english_mixed_is_not_forced_through_hangul_rules_and_never_gets_fake_ipa():
    st = pron.sync(ops.apply_text(LyricsState(), "love 하고 싶어 baby"))
    es = {s.text: st.pronunciation[s.id] for s in st.text.syllables}
    for w in ("love", "ba", "by"):
        assert es[w].status == "unavailable" and es[w].text == "" and es[w].ipa == "" and es[w].language == "en"
    assert es["싶"].text == "시" and es["어"].text == "퍼" and es["하"].status == "valid"


def test_japanese_kana_morae_and_ipa():
    st = pron.sync(ops.apply_text(LyricsState(), "きっと さんぽ ラーメン 東京"))
    es = [(s.text, st.pronunciation[s.id]) for s in st.text.syllables]
    got = [(t, e.text, e.ipa) for t, e in es]
    assert got[:3] == [("き", "キ", "ki"), ("っ", "ッ", "t"), ("と", "ト", "to")]
    assert ("ん", "ン", "m") in got  # さんぽ: ㅂ 앞 ん → [m]
    assert ("ー", "ー", "ː") in got
    kanji = [e for t, e in es if t in "東京"]
    assert all(e.status == "unavailable" and e.ipa == "" and e.language == "ja" for e in kanji)
    assert all(japanese.ipa_is_valid(e.ipa) for _t, e in es if e.status != "unavailable")
    st2 = pron.sync(ops.apply_text(LyricsState(), "はるは"))
    assert all(st2.pronunciation[s.id].status == "uncertain" for s in st2.text.syllables if s.text == "は")


def test_generated_ipa_is_valid_and_uses_ipa_symbols_only():
    lyrics = "같이 걷던 길 위에 꽃잎이 흩날리고\n넓은 하늘 아래 너를 안고 싶어\n밟는 순간 읽고 싶은 마음 없는 날"
    st = pron.sync(ops.apply_text(LyricsState(), lyrics))
    for s in st.text.syllables:
        e = st.pronunciation[s.id]
        assert korean.ipa_is_valid(e.ipa), (s.text, e.ipa)
        assert not set(e.ipa) & set("ycrqxzfv")  # 로마자 표기 글자가 섞이지 않는다
    ipa = {s.text + str(i): st.pronunciation[s.id].ipa for i, s in enumerate(st.text.syllables)}
    assert ipa["같0"] == "ka" and ipa["이1"] == "tɕʰi"  # 같이 [가치]
    assert ipa["던3"] == "t͈ʌn"  # 걷던 [걷떤] 경음
    assert ipa["길4"] == "ki" and ipa["위5"] == "ɾwi"  # 길 위에 [기뤼에]: ㄹ이 넘어가 탄설음


def test_user_edit_is_kept_marked_and_reset():
    st = pron.sync(ops.apply_text(LyricsState(), "같이 가자"))
    a, b = (s.id for s in st.text.syllables[:2])
    st = pron.set_user(st, b, "티")  # 사용자가 [가티]로 부른다고 정함
    e = st.pronunciation[b]
    assert e.user_edited and e.source == "user" and e.text == "티" and e.ipa == "tʰi" and e.auto_text == "치"
    # 다른 줄을 고쳐 다시 맞춰도 사용자 값은 그대로
    st2 = pron.sync(ops.apply_text(st, "같이 가자\n새 줄"))
    assert st2.pronunciation[b].text == "티" and not st2.pronunciation[b].stale
    # 자동 다시 생성도 사용자 값을 덮지 않는다
    st3, n, kept = pron.regenerate(st2)
    assert st3.pronunciation[b].text == "티" and kept == 1
    # 앞 음절 원문이 바뀌어 자동 발음이 달라지면 '재검토 필요'만 붙는다 (값은 지킴)
    st4 = pron.sync(ops.apply_text(st, "갈이 가자"))
    assert st4.pronunciation[b].text == "티" and st4.pronunciation[b].stale
    # 기본 발음으로 되돌리기
    st5 = pron.reset(st4, [b])
    assert not st5.pronunciation[b].user_edited and st5.pronunciation[b].text == st5.pronunciation[b].auto_text


def test_syllable_counts_can_differ_between_text_and_pronunciation():
    st = pron.sync(ops.apply_text(LyricsState(), "12 되어"))
    n12, doe, eo = st.text.syllables
    assert st.pronunciation[n12.id].status == "unavailable"
    st = pron.set_user(st, n12.id, "십이")  # 원문 1음절 → 발음 2음절
    st = pron.set_user(st, doe.id, "돼")
    st = pron.set_user(st, eo.id, "")  # 앞 음절과 합쳐 발음 → 발음 0음절
    units = [st.pronunciation[s.id].units for s in (n12, doe, eo)]
    assert units == [2, 1, 0]
    assert st.pronunciation[n12.id].ipa == "ɕi.bi" and st.pronunciation[n12.id].language == "ko"
    # 대응은 음절 id로 관리되어 저장·불러오기 뒤에도 그대로
    back = io.state_from_dicts(json.loads(json.dumps(io.lyrics_to_dict(st))), io.alignment_to_dict(st), [])
    assert back.pronunciation == st.pronunciation


def test_lyrics_json_v2_round_trip_and_v1_upgrade():
    st = pron.sync(ops.apply_text(LyricsState(), "꽃잎이 날려"))
    sid = st.text.syllables[0].id
    st = pron.set_user(st, sid, "꼳")
    d = schema.stamp(schema.LYRICS, io.lyrics_to_dict(st))
    assert d["schema_version"] == 2 and d["pronunciation"]["converter"] == pron.CONVERTER
    row = d["pronunciation"]["syllables"][0]
    for key in ("syllable_id", "original_text", "pronunciation_text", "ipa", "language", "pronunciation_source",
                "pronunciation_status", "is_user_edited", "conversion_model", "conversion_version"):
        assert key in row
    w: list = []
    back = io.state_from_dicts(schema.migrate(schema.LYRICS, json.loads(json.dumps(d)))[0], None, w)
    assert back.pronunciation == st.pronunciation and not w
    # STEP 3 파일(v1, 발음 없음)은 올린 뒤 발음이 비어 있고, sync가 자동 발음을 채운다
    v1 = {k: v for k, v in d.items() if k != "pronunciation"} | {"schema_version": 1}
    up, migrated = schema.migrate(schema.LYRICS, v1)
    assert migrated
    old = io.state_from_dicts(up, None, w)
    assert old.pronunciation == {} and len(pron.sync(old).pronunciation) == len(old.text.syllables)


def test_converter_version_change_regenerates_only_generated_entries(monkeypatch):
    st = pron.sync(ops.apply_text(LyricsState(), "같이 가자"))
    sid = st.text.syllables[1].id
    st = pron.set_user(st, sid, "티")
    monkeypatch.setattr(pron, "VERSION", "999")
    st2 = pron.sync(ops.apply_text(st, "같이 가자\n"))  # 텍스트는 그대로
    st2 = pron.regenerate(st2)[0]
    assert all(e.version == "999" for k, e in st2.pronunciation.items() if k != sid)
    assert st2.pronunciation[sid].text == "티"


def test_fixes_found_by_comparing_real_lyrics_with_g2pk2():
    """실제 가사 두 곡을 g2pk2(공개 규칙 변환기)와 비교하다 찾은 것. 이 비교 뒤에 고쳤으므로 독립 검증이 아니다."""
    # 연음으로 첫소리가 생긴 '의'는 [ㅢ]를 두고 허용 발음([에]·[이])을 알린다 (협의[혀븨/혀비]와 같은 경우)
    got, es = _say("추억의")
    assert got == "추어긔" and es[2].status == "uncertain"
    assert _say("희망")[0] == "히망"  # 원래 첫소리가 자음이면 [ㅣ]
    # 띄어 쓴 경계의 연음은 따로 표시한다 (제15항: 꽃 위[꼬뒤])
    _got, es = _say("꽃 위")
    assert korean.R_LINK_SPACE in es[1].rules and korean.R_LINK not in es[1].rules
    # 관형사형 -ㄹ 뒤 된소리는 품사를 몰라 바꾸지 않되 불확실로 알린다
    got, es = _say("갈 곳")
    assert got == "갈 곧" and es[1].status == "uncertain" and korean.U_MOD_SPACE in es[1].notes
    # ㄴ 첨가로 추정한 곳은 앞 음절도 불확실 (만일[마닐]처럼 틀릴 수 있다)
    _got, es = _say("만일")
    assert es[0].status == "uncertain" and es[1].status == "uncertain"
