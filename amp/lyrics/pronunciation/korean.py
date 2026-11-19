"""한국어 발음 변환: 가사 원문(한글) → 표준 발음 한글 표기 → IPA.

음절을 하나씩 바꾸지 않고 줄(한 호흡) 안에서 앞뒤 음절을 함께 보며 표준 발음법(국립국어원)의 조항을 차례로 적용한다.
한글 음절 수는 바뀌지 않으므로 원문 음절과 발음 음절은 1:1로 대응한다.

형태소 정보 없이는 정할 수 없는 경우가 있다. 이때는 가장 흔한 쪽을 고르고 '불확실'로 표시한다.
- 제24항(용언 어간 받침 ㄴ·ㅁ 뒤 된소리), 제26항(한자어 ㄹ 뒤 된소리), 제27항(관형사형 -ㄹ 뒤 된소리 일부),
  제28항(합성어 사잇소리), 제29항(ㄴ 첨가), 제20항 예외(의견란[의견난] 등), 조사 '의'
- 소리의 길이(장단)는 표기하지 않는다(단어마다 정해져 있어 사전 없이 알 수 없음).

IPA는 아래 규칙의 간략 표기다(국제음성학회 기호, 한국어 음운론에서 널리 쓰는 대응).
- 평음 ㄱㄷㅂㅈ: 무성 [k t p tɕ], 같은 어절 안 유성음 사이에서 유성 [ɡ d b dʑ]
- 경음 [k͈ t͈ p͈ s͈ t͈ɕ], 격음 [kʰ tʰ pʰ tɕʰ], ㅅ·ㅆ은 ㅣ·반모음 j 앞에서 [ɕ ɕ͈]
- ㄹ: 음절 첫소리 [ɾ], 받침과 ㄹㄹ [l]. 받침 ㄱㄷㅂ은 불파음 [k̚ t̚ p̚]
- 모음 ㅏ[a] ㅐ[ɛ] ㅓ[ʌ] ㅔ[e] ㅗ[o] ㅜ[u] ㅡ[ɯ] ㅣ[i] ㅚ[we] ㅟ[wi] ㅢ[ɰi], 반모음 [j w]
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
        "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

R_REP = "받침 대표음(제9~11항)"
R_LINK = "연음(제13~15항)"
R_ASP = "거센소리되기(제12항)"
R_H_DROP = "ㅎ 탈락(제12항)"
R_H_N = "ㅎ+ㄴ→ㄴ(제12항)"
R_H_S = "ㅎ+ㅅ→ㅆ(제12항)"
R_PAL = "구개음화(제17항)"
R_NASAL = "비음화(제18항)"
R_R_TO_N = "ㄹ의 비음화(제19항)"
R_LIQUID = "유음화(제20항)"
R_TENSE = "된소리되기(제23항)"
R_TENSE_STEM = "된소리되기(제24·25항)"
R_TENSE_MOD = "된소리되기(제27항)"
R_N_INSERT = "ㄴ 첨가(제29항)"
R_UI = "ㅢ→ㅣ(제5항)"
R_YEO = "져·쪄·쳐→저·쩌·처(제5항)"

U_STEM = "용언 어간이면 된소리, 아니면 예사소리(제24항). 형태소를 알 수 없어 된소리로 두었습니다."
U_STEM_PLAIN = "용언 어간 받침 뒤면 된소리(제24항, 예: 껴안다[껴안따], 줄넘기[줄럼끼])지만 어미·접미사로 보고 예사소리로 두었습니다."
U_SINO = "한자어나 합성어면 된소리로 날 수 있습니다(제26·28항). 원문 그대로 두었습니다."
U_SAI = "합성어면 사잇소리로 된소리가 날 수 있습니다(제28항). 원문 그대로 두었습니다."
U_MOD = "어미 '-ㄹ게·-ㄹ지·-ㄹ걸'이면 된소리, 어간+'-게·-지'면 예사소리입니다(제27항). 된소리로 두었습니다."
U_NI = "합성어면 ㄴ이 덧나 발음합니다(제29항, 예: 꽃잎[꼰닙]). 형태소를 알 수 없어 ㄴ 첨가로 두었습니다."
U_LIQ_EXC = "'-란·-량·-력·-론' 같은 한자어 접미사면 [ㄴㄴ]으로 발음합니다(제20항 예외, 예: 의견란[의견난])."
U_UI_PARTICLE = "조사 '의'는 [에]로, 첫음절 아닌 '의'는 [이]로도 발음할 수 있습니다(제5항 허용)."
U_SAM = "'삶'이 명사면 예사소리, 용언 '삶다'면 된소리입니다."

NIEUN_HEADS = {"잎", "일", "입", "엿", "약", "윷", "옆"}  # 합성어 뒷말로 흔한 ㄴ 첨가 형태소 (제29항 예)
BOUND_NOUNS = {"거", "것", "게", "걸", "수", "줄", "데", "바", "적", "지", "듯", "뻔"}  # 관형사형 -ㄹ 뒤 된소리 (제27항)
# 사잇소리·한자어 된소리 경고를 내지 않을 흔한 조사·어미 첫음절
ENDINGS = set("가과고게거걸겠기까께다도들대더든던데지자서세시사부보조구군긴길나니네는다던도록며면")
LIQ_EXC_HEADS = {"란", "량", "력", "령", "례", "론", "료", "류"}

ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
LENIS = set(TENSE)
DOUBLE = {"ㄳ": ("ㄱ", "ㅆ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
          "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅆ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"), "ㅄ": ("ㅂ", "ㅆ")}
# 받침 대표음 (제9~11항). 겹받침은 앞이나 뒤 하나만 남긴다
REPRESENT = {"ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ",
             "ㅎ": "ㄷ", "ㅍ": "ㅂ", "ㄿ": "ㅂ", "ㅄ": "ㅂ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄻ": "ㅁ", "ㄼ": "ㄹ", "ㄽ": "ㄹ",
             "ㄾ": "ㄹ", "ㅀ": "ㄹ"}
K_CLASS = {"ㄱ", "ㄲ", "ㅋ", "ㄳ", "ㄺ"}
T_CLASS = {"ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ"}
P_CLASS = {"ㅂ", "ㅍ", "ㄼ", "ㅄ"}
NASAL_OF = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}


def is_hangul(ch: str) -> bool:
    return len(ch) == 1 and 0xAC00 <= ord(ch) <= 0xD7A3


def decompose(ch: str) -> tuple[str, str, str]:
    code = ord(ch) - 0xAC00
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def compose(cho: str, jung: str, jong: str) -> str:
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


@dataclass
class _Syl:
    orig: str
    cho: str
    jung: str
    jong: str
    word: int  # 같은 어절인지 가르는 번호
    orig_jong: str = ""
    moved: bool = False  # 첫소리가 앞 음절 받침에서 넘어왔다 (연음)
    rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def rule(self, r: str) -> None:
        if r not in self.rules:
            self.rules.append(r)

    def note(self, n: str) -> None:
        if n not in self.notes:
            self.notes.append(n)

    @property
    def text(self) -> str:
        return compose(self.cho, self.jung, self.jong)


@dataclass(frozen=True)
class KoreanResult:
    text: str  # 발음 한글 한 음절
    rules: tuple[str, ...]
    notes: tuple[str, ...]  # 불확실한 이유


def _boundary(a: _Syl, b: _Syl, breaks: set[int], i: int) -> str:
    if i in breaks:
        return "break"
    return "word" if a.word == b.word else "space"


def pronounce(units: list[tuple[str, int]], breaks: set[int] | None = None) -> list[KoreanResult]:
    """한 줄의 한글 음절 → 음절마다 발음.

    units: (음절 글자, 어절 번호). breaks: i와 i+1 사이를 끊는 위치(문장부호·한글 아닌 글자). 같은 줄 안에서만 쓴다.
    """
    breaks = breaks or set()
    syls = []
    for ch, w in units:
        c, v, j = decompose(ch)
        syls.append(_Syl(ch, c, v, j, w, j))
    pairs = [(i, syls[i], syls[i + 1], _boundary(syls[i], syls[i + 1], breaks, i)) for i in range(len(syls) - 1)]
    pairs = [p for p in pairs if p[3] != "break"]
    for _i, a, b, bd in pairs:
        _h_rules(a, b, bd)
    for _i, a, b, bd in pairs:
        _palatal_and_insert(a, b, bd)
    for _i, a, b, bd in pairs:
        _liaison(a, b, bd)
    for idx, s in enumerate(syls):
        nxt = syls[idx + 1] if idx + 1 < len(syls) and idx not in breaks else None
        _neutralize(s, nxt)
    for k, (_i, a, b, bd) in enumerate(pairs):
        nxt2 = pairs[k + 1][2] if k + 1 < len(pairs) and pairs[k + 1][1] is b else None
        _tensify(a, b, bd, nxt2)
    for _i, a, b, bd in pairs:
        _nasal_liquid(a, b, bd)
    for idx, s in enumerate(syls):
        _vowels(s, syls[idx - 1] if idx > 0 and (idx - 1) not in breaks else None)
    for s in syls:
        if s.text != s.orig and not s.rules:
            s.rule(R_REP)
    return [KoreanResult(s.text, tuple(s.rules), tuple(s.notes)) for s in syls]


def _h_rules(a: _Syl, b: _Syl, bd: str) -> None:
    """제12항: ㅎ 받침·ㅎ 첫소리."""
    if a.jong in ("ㅎ", "ㄶ", "ㅀ"):
        rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[a.jong]
        if b.cho in ("ㄱ", "ㄷ", "ㅈ"):
            b.cho = ASPIRATE[b.cho]
            a.jong = rest
            a.rule(R_ASP)
            b.rule(R_ASP)
        elif b.cho == "ㅅ":
            b.cho = "ㅆ"
            a.jong = rest
            a.rule(R_H_S)
            b.rule(R_H_S)
        elif b.cho == "ㄴ":
            a.jong = "ㄴ" if a.jong == "ㅎ" else rest
            a.rule(R_H_N)
        elif b.cho == "ㅇ":
            a.jong = rest
            a.rule(R_H_DROP)
        return
    if b.cho != "ㅎ" or not a.jong:
        return
    j = a.jong
    keep = ""
    if j in DOUBLE and j not in ("ㄳ", "ㅄ"):
        keep, j = DOUBLE[j]
    if j in K_CLASS:
        new = "ㅋ"
    elif j in P_CLASS:
        new = "ㅍ"
    elif j in ("ㅈ", "ㅊ") and bd == "word":
        new = "ㅊ"
    elif j in T_CLASS:
        new = "ㅌ"
    else:
        return
    if new == "ㅌ" and b.jung == "ㅣ" and bd == "word" and j in ("ㄷ", "ㅌ"):
        new = "ㅊ"  # 굳히다[구치다] (제17항 붙임)
        a.rule(R_PAL)
        b.rule(R_PAL)
    b.cho = new
    a.jong = keep
    a.rule(R_ASP)
    b.rule(R_ASP)


def _palatal_and_insert(a: _Syl, b: _Syl, bd: str) -> None:
    if bd == "word" and b.cho == "ㅇ" and b.jung == "ㅣ" and a.jong in ("ㄷ", "ㅌ", "ㄾ"):
        if a.jong == "ㄾ":
            a.jong, b.cho = "ㄹ", "ㅊ"
        else:
            b.cho = "ㅈ" if a.jong == "ㄷ" else "ㅊ"
            a.jong = ""
        a.rule(R_PAL)
        b.rule(R_PAL)
        return
    if (b.cho == "ㅇ" and b.jung in "ㅣㅑㅕㅛㅠ" and b.orig in NIEUN_HEADS and a.jong not in ("", "ㅇ")
            and (bd == "word" or (bd == "space" and b.orig == "일" and a.jong == "ㄹ"))):
        b.cho = "ㄴ"
        b.rule(R_N_INSERT)
        b.note(U_NI)


def _liaison(a: _Syl, b: _Syl, bd: str) -> None:
    """제13~15항: 받침 + 모음으로 시작하는 음절."""
    if b.cho != "ㅇ" or a.jong in ("", "ㅇ"):
        return
    j = a.jong
    if bd == "space" or b.orig == "없":
        # 실질 형태소 앞(어절 경계, '맛없다'): 대표음으로 바꾼 뒤 넘긴다 (제15항: 닭 앞에[다가페], 넋 없다[너겁따])
        a.jong, b.cho = "", REPRESENT.get(j, j)
    elif j in DOUBLE:
        a.jong, b.cho = DOUBLE[j]
    else:
        a.jong, b.cho = "", j
    b.moved = True
    a.rule(R_LINK)
    b.rule(R_LINK)


def _neutralize(s: _Syl, nxt: _Syl | None) -> None:
    j = s.jong
    if not j or j in ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"):
        return
    if j == "ㄼ" and (s.orig == "밟" or (s.orig == "넓" and nxt is not None and nxt.orig in ("죽", "둥"))):
        new = "ㅂ"
    elif j == "ㄺ" and nxt is not None and nxt.cho == "ㄱ" and nxt.word == s.word and s.orig not in ("닭", "칡", "흙"):
        new = "ㄹ"  # 용언 어간 ㄺ + ㄱ (맑게[말께], 제11항 다만)
    else:
        new = REPRESENT.get(j, j)
    if new != j:
        s.jong = new
        s.rule(R_REP)


def _tensify(a: _Syl, b: _Syl, bd: str, after_b: _Syl | None) -> None:
    if b.cho not in LENIS or b.moved:
        return
    oj = a.orig_jong
    tense, rule, note = False, R_TENSE, ""
    if a.jong in ("ㄱ", "ㄷ", "ㅂ"):
        tense = True  # 제23항
    elif bd == "word" and oj in ("ㄼ", "ㄾ") and b.cho in ("ㄱ", "ㄷ", "ㅅ", "ㅈ"):
        tense, rule = True, R_TENSE_STEM  # 넓게[널께], 핥다[할따] (제25항)
    elif bd == "word" and oj == "ㄺ" and b.cho == "ㄱ" and a.jong == "ㄹ":
        tense, rule = True, R_TENSE_STEM  # 맑게[말께]
    elif bd == "word" and oj in ("ㄵ", "ㄻ") and b.cho in ("ㄱ", "ㄷ", "ㅅ", "ㅈ"):
        tense, rule = True, R_TENSE_STEM  # 앉고[안꼬], 닮고[담꼬] (제24항)
        if a.orig == "삶":
            note = U_SAM
    elif bd == "word" and oj == "ㄴ" and a.jong == "ㄴ" and b.orig == "고":
        tense, rule, note = True, R_TENSE_STEM, U_STEM  # 안고[안꼬] / 신고(申告)[신고]
    elif bd == "word" and oj == "ㅁ" and a.jong == "ㅁ" and b.orig in ("고", "다", "지", "게", "던", "더", "든", "소", "거"):
        tense, rule, note = True, R_TENSE_STEM, U_STEM  # 감고[감꼬], 참지[참찌]
    elif bd == "word" and oj in ("ㄴ", "ㅁ") and a.jong == oj and (b.orig == "기" or (b.orig == "다" and a.orig in ("안", "신"))):
        b.note(U_STEM_PLAIN)  # 줄넘기[줄럼끼]·감기[감기], 껴안다[껴안따]·한다[한다]
    elif a.jong == "ㄹ" and oj == "ㄹ":
        if bd == "word" and b.orig in ("게", "지", "걸"):
            tense, rule, note = True, R_TENSE_MOD, U_MOD  # 할게[할께] / 알게[알게]
        elif bd == "word" and b.orig == "수" and after_b is not None and after_b.orig == "록" and after_b.word == b.word:
            tense, rule = True, R_TENSE_MOD  # 할수록[할쑤록]
        elif bd == "space" and b.orig in BOUND_NOUNS:
            tense, rule = True, R_TENSE_MOD  # 할 수[할쑤], 갈 곳은 (BOUND_NOUNS 밖이면 그대로)
        elif bd == "word" and b.cho in ("ㄷ", "ㅅ", "ㅈ") and b.orig not in ENDINGS:
            b.note(U_SINO)  # 갈등[갈뜽] / 알지
    elif bd == "word" and a.jong in ("ㄴ", "ㅁ", "ㅇ", "ㄹ") and b.orig not in ENDINGS:
        b.note(U_SAI)  # 눈동자[눈똥자], 등불[등뿔]
    if tense:
        b.cho = TENSE[b.cho]
        b.rule(rule)
        if note:
            b.note(note)


def _nasal_liquid(a: _Syl, b: _Syl, bd: str) -> None:
    if b.cho == "ㄹ" and a.jong in ("ㅁ", "ㅇ", "ㄱ", "ㄷ", "ㅂ"):
        b.cho = "ㄴ"  # 담력[담녁], 막론[망논] (제19항)
        b.rule(R_R_TO_N)
    if a.jong in NASAL_OF and b.cho in ("ㄴ", "ㅁ"):
        a.jong = NASAL_OF[a.jong]  # 먹는[멍는], 밥물[밤물] (제18항, 두 단어를 이어 말할 때도)
        a.rule(R_NASAL)
    if a.jong == "ㄴ" and b.cho == "ㄹ":
        a.jong = "ㄹ"  # 난로[날로] (제20항)
        a.rule(R_LIQUID)
        if b.orig in LIQ_EXC_HEADS:
            a.note(U_LIQ_EXC)
    elif a.jong == "ㄹ" and b.cho == "ㄴ":
        b.cho = "ㄹ"  # 칼날[칼랄], 뚫는[뚤른]
        b.rule(R_LIQUID)


def _vowels(s: _Syl, prev: _Syl | None) -> None:
    if s.jung == "ㅢ":
        if s.cho != "ㅇ":
            s.jung = "ㅣ"  # 희망[히망] (제5항 다만 3)
            s.rule(R_UI)
        elif prev is not None and prev.word == s.word:
            s.note(U_UI_PARTICLE)  # 나의[나의/나에], 주의[주의/주이]
    if s.cho in ("ㅈ", "ㅉ", "ㅊ") and s.jung == "ㅕ":
        s.jung = "ㅓ"  # 가져[가저] (제5항 다만 1)
        s.rule(R_YEO)


# --- IPA ------------------------------------------------------------------------------

VOWEL_IPA = {"ㅏ": "a", "ㅐ": "ɛ", "ㅑ": "ja", "ㅒ": "jɛ", "ㅓ": "ʌ", "ㅔ": "e", "ㅕ": "jʌ", "ㅖ": "je", "ㅗ": "o",
             "ㅘ": "wa", "ㅙ": "wɛ", "ㅚ": "we", "ㅛ": "jo", "ㅜ": "u", "ㅝ": "wʌ", "ㅞ": "we", "ㅟ": "wi", "ㅠ": "ju",
             "ㅡ": "ɯ", "ㅢ": "ɰi", "ㅣ": "i"}
ONSET_IPA = {"ㄱ": "k", "ㄲ": "k͈", "ㄴ": "n", "ㄷ": "t", "ㄸ": "t͈", "ㄹ": "ɾ", "ㅁ": "m", "ㅂ": "p", "ㅃ": "p͈",
             "ㅅ": "s", "ㅆ": "s͈", "ㅇ": "", "ㅈ": "tɕ", "ㅉ": "t͈ɕ", "ㅊ": "tɕʰ", "ㅋ": "kʰ", "ㅌ": "tʰ", "ㅍ": "pʰ",
             "ㅎ": "h"}
VOICED_ONSET = {"ㄱ": "ɡ", "ㄷ": "d", "ㅂ": "b", "ㅈ": "dʑ"}
CODA_IPA = {"": "", "ㄱ": "k̚", "ㄴ": "n", "ㄷ": "t̚", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p̚", "ㅇ": "ŋ"}
SONORANT_CODA = {"", "ㄴ", "ㄹ", "ㅁ", "ㅇ"}
# 한국어 IPA 출력에 쓰는 기호 전부 (유효성 검사용)
IPA_SYMBOLS = set("aɛjʌeowuɯɰikɡtdpbsɕhmnŋɾlʑ") | {"ʰ", "͈", "̚", ".", " "}


def syllable_ipa(syllables: list[tuple[str, int]]) -> list[str]:
    """발음 한글 음절 [(글자, 어절 번호)] → 음절마다 IPA. 한글이 아니면 ""."""
    out: list[str] = []
    prev: tuple[str, str, str] | None = None
    prev_word = None
    for ch, w in syllables:
        if not is_hangul(ch):
            out.append("")
            prev = None
            continue
        c, v, j = decompose(ch)
        if j not in CODA_IPA:
            j = REPRESENT.get(j, j)
        same_word = prev is not None and prev_word == w
        onset = ONSET_IPA[c]
        if c == "ㄹ" and same_word and prev[2] == "ㄹ":
            onset = "l"  # ㄹㄹ → [ll]
        elif c in VOICED_ONSET and same_word and prev[2] in SONORANT_CODA:
            onset = VOICED_ONSET[c]  # 유성음 사이 평음
        if c in ("ㅅ", "ㅆ") and (v in ("ㅣ", "ㅟ") or VOWEL_IPA[v].startswith("j")):
            onset = "ɕ" if c == "ㅅ" else "ɕ͈"
        vowel = VOWEL_IPA[v]
        if c in ("ㅈ", "ㅉ", "ㅊ") and vowel.startswith("j"):
            vowel = vowel[1:]  # 자·쟈 구별 없음
        out.append(onset + vowel + CODA_IPA[j])
        prev, prev_word = (c, v, j), w
    return out


def ipa_is_valid(ipa: str) -> bool:
    return bool(ipa) and all(ch in IPA_SYMBOLS for ch in ipa)
