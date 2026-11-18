"""가사 텍스트를 줄·단어·음절 토큰으로 나눈다.

음절 분리는 순수한 텍스트 처리다. 실제로 언제 발성했는지(시간)는 정렬 단계가 따로 정한다.

- 한국어: 한글 음절 글자 하나가 음절 하나 (사랑해 → 사/랑/해). 낱자(ㅋ, ㅠ)도 하나씩.
- 영어 등 라틴 문자: 모음 묶음 기준 규칙으로 나눈다(little → lit/tle). 규칙 기반이라 틀릴 수 있어 표시를 남긴다.
- 일본어 가나: 박(모라) 단위. 작은 ゃゅょ 등은 앞 글자에 붙이고, っ·ー·ん은 각각 한 박.
- 한자: 읽기를 알 수 없어 글자 하나를 한 단위로 두고 검토 표시를 남긴다.
- 숫자: 읽는 법이 언어·문맥마다 달라 한 덩어리로 두고 검토 표시를 남긴다.
- 문장부호·기호: 음절이 아니다. 단어 원문에는 남는다.

``reading`` 은 정렬 모델(MMS)이 쓰는 로마자(a-z와 ')다. 한글 자모 표는 MMS 학습에 쓰인 uroman 표기와
11,172개 음절 전부에서 같은 결과를 낸다(개발 중 대조). 모르면 빈 문자열이며, 정렬기는 이 단위를 와일드카드로 다룬다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

HANGUL = "hangul"
JAMO = "jamo"
LATIN = "latin"
KANA = "kana"
KANJI = "kanji"
NUMBER = "number"

NOTE_LATIN = "영어 음절은 규칙으로 나눴습니다. 실제 발음과 다르면 나누기·합치기로 고치세요."
NOTE_KANJI = "한자는 읽기를 알 수 없어 한 글자를 한 단위로 두었습니다. 필요하면 나누거나 합치세요."
NOTE_NUMBER = "숫자는 읽는 법이 정해지지 않아 한 덩어리로 두었습니다. 읽는 대로 고쳐 쓰면 정확해집니다."

# --- 한글 → 로마자 (uroman과 같은 자모 표) ---------------------------------------------
_INITIAL = ("g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s", "ss", "", "j", "jj", "c", "k", "t", "p", "h")
_MEDIAL = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wai", "oe", "yo", "u", "weo", "we", "wi",
           "yu", "eu", "yi", "i")
_FINAL = ("", "g", "gg", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs",
          "s", "ss", "ng", "j", "c", "k", "t", "p", "h")
# 호환 자모(ㄱ~ㅣ) 낱자: 자음은 초성 표기(ㅇ은 ng), 모음은 중성 표기
_COMPAT_CONS = "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_COMPAT_CONS_READ = ("g", "gg", "gs", "n", "nj", "nh", "d", "dd", "r", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m",
                     "b", "bb", "bs", "s", "ss", "ng", "j", "jj", "c", "k", "t", "p", "h")
_COMPAT_VOWEL = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"


def hangul_reading(ch: str) -> str:
    code = ord(ch) - 0xAC00
    if 0 <= code < 11172:
        return _INITIAL[code // 588] + _MEDIAL[(code % 588) // 28] + _FINAL[code % 28]
    i = _COMPAT_CONS.find(ch)
    if i >= 0:
        return _COMPAT_CONS_READ[i]
    i = _COMPAT_VOWEL.find(ch)
    return _MEDIAL[i] if i >= 0 else ""


# --- 가나 → 로마자 ------------------------------------------------------------------
_GOJUON = (
    "あa いi うu えe おo かka きki くku けke こko がga ぎgi ぐgu げge ごgo さsa しshi すsu せse そso ざza じji ずzu "
    "ぜze ぞzo たta ちchi つtsu てte とto だda ぢji づzu でde どdo なna にni ぬnu ねne のno はha ひhi ふfu へhe ほho "
    "ばba びbi ぶbu べbe ぼbo ぱpa ぴpi ぷpu ぺpe ぽpo まma みmi むmu めme もmo やya ゆyu よyo らra りri るru れre ろro "
    "わwa ゐwi ゑwe をo んn ゔvu ぁa ぃi ぅu ぇe ぉo ゃya ゅyu ょyo ゎwa ゕka ゖke"
)
_KANA: dict[str, str] = {}
for _item in _GOJUON.split():
    _KANA[_item[0]] = _item[1:]
    _KANA[chr(ord(_item[0]) + 0x60)] = _item[1:]  # 같은 소리의 가타카나
_KANA.update({"ヲ": "wo", "ヷ": "va", "ヸ": "vi", "ヹ": "ve", "ヺ": "vo"})
_SMALL = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_SOKUON = set("っッ")
_CHOON = "ー"


def _is_kana(ch: str) -> bool:
    return 0x3041 <= ord(ch) <= 0x30FF and ch not in "・゠"


def _is_kanji(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or ch in "々〆"


def _is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def _is_jamo(ch: str) -> bool:
    return 0x3131 <= ord(ch) <= 0x318E


def _is_latin(ch: str) -> bool:
    if not ch.isalpha():
        return False
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def kana_morae(run: str) -> list[tuple[str, str]]:
    """가나 묶음 → [(박 글자, 로마자)]."""
    units: list[str] = []
    for ch in run:
        if ch in _SMALL and units and units[-1] not in _SOKUON and units[-1] != _CHOON:
            units[-1] += ch
        else:
            units.append(ch)
    readings: list[str] = []
    for u in units:
        if u[0] in _SOKUON or u == _CHOON:
            readings.append("")  # 아래에서 이웃 박을 보고 채운다
            continue
        base = _KANA.get(u[0], "")
        for small in u[1:]:
            r = _KANA.get(small, "")
            if not base or not r:
                base += r
            elif base in ("shi", "chi", "ji") and r[0] == "y":
                base = base[:-1] + r[1:]  # しゃ → sha
            elif base.endswith("i") and r[0] == "y":
                base = base[:-1] + r  # きゃ → kya
            else:
                base = base[:-1] + r if len(base) > 1 else base + r  # ふぁ → fa
        readings.append(base)
    for i, u in enumerate(units):
        if u[0] in _SOKUON:  # 다음 박의 첫 자음을 겹쳐 읽는다 (きって → kitte)
            nxt = readings[i + 1] if i + 1 < len(readings) else ""
            readings[i] = nxt[0] if nxt and nxt[0] not in "aeiou" else "t"
        elif u == _CHOON:  # 앞 박의 모음을 늘인다 (ラーメン → raamen)
            prev = readings[i - 1] if i > 0 else ""
            vowels = [c for c in prev if c in "aeiou"]
            readings[i] = vowels[-1] if vowels else ""
    return list(zip(units, readings))


# --- 영어 음절 (규칙 기반) -------------------------------------------------------------
_ONSET_DIGRAPHS = ("ch", "sh", "th", "ph", "wh", "gh", "qu", "wr", "kn")


def english_syllables(word: str) -> list[str]:
    """영어 단어를 모음 묶음 기준으로 나눈다. 사전이 아니라 규칙이라 근사값이다."""
    low = word.lower()
    n = len(low)
    if n <= 3:
        return [word]
    isv = [c in "aeiou" or (c == "y" and i > 0 and low[i - 1] not in "aeiou") for i, c in enumerate(low)]
    for i in range(1, n):
        if low[i] == "u" and low[i - 1] == "q":
            isv[i] = False
    groups: list[list[int]] = []
    for i in range(n):
        if isv[i]:
            if groups and groups[-1][1] == i:
                groups[-1][1] = i + 1
            else:
                groups.append([i, i + 1])
    if len(groups) > 1:  # 말음 e는 대개 소리가 나지 않는다 (love, make). 자음+le(table)는 예외
        s, e = groups[-1]
        if s == n - 1 and low[-1] == "e" and not (low[-2] == "l" and not isv[n - 3]):
            groups.pop()
        elif s == n - 2 and e == n - 1 and low[-2:] in ("ed", "es"):
            c = low[-3]
            keep = (low.endswith("ed") and c in "td") or (low.endswith("es") and (c in "sxzcg" or low[-4:-2] in ("ch", "sh")))
            if not keep:
                groups.pop()
    if len(groups) <= 1:
        return [word]
    cuts: list[int] = []
    for (s1, e1), (s2, e2) in zip(groups, groups[1:]):
        cons = low[e1:s2]
        k = len(cons)
        if k <= 1:
            cut = e1  # V-CV (ba-by)
        elif s2 == n - 1 and low[s2] == "e" and cons.endswith("l"):
            cut = s2 - 2  # 자음+le (ta-ble, lit-tle)
        elif cons[:2] == "ng" or cons[:2] == "ck":
            cut = e1 + 2  # sing-ing, pick-ing
        elif cons[-2:] in _ONSET_DIGRAPHS and k == 2:
            cut = e1  # fa-ther
        else:
            cut = e1 + 1  # VC-CV (hap-py)
        cuts.append(max(cut, cuts[-1] + 1 if cuts else 1))
    cuts = [c for c in cuts if 0 < c < n]
    bounds = [0] + cuts + [n]
    return [word[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


def latin_reading(text: str) -> str:
    plain = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in plain if "a" <= c <= "z" or c == "'")


# --- 토큰 ------------------------------------------------------------------------

@dataclass(frozen=True)
class SylToken:
    text: str
    kind: str
    reading: str = ""
    note: str = ""


@dataclass(frozen=True)
class WordTokens:
    text: str  # 원문 그대로 (문장부호 포함)
    syllables: tuple[SylToken, ...]


_FULLWIDTH = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}


def normalize_text(text: str) -> str:
    """NFC 정규화(맥에서 복사한 풀어쓴 한글 등), 전각 영숫자 → 반각, 줄바꿈 통일, 제어 문자 제거."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.translate(_FULLWIDTH).replace("　", " ").replace("﻿", "")
    # 반각 가타카나는 전각으로 (NFKC는 호환 자모까지 바꾸므로 이 범위에만 쓴다)
    text = re.sub("[ｦ-ﾟ]+", lambda m: unicodedata.normalize("NFKC", m.group()), text)
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")


def _apostrophe(ch: str) -> bool:
    return ch in "'’"


def tokenize_word(word: str) -> WordTokens:
    syl: list[SylToken] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if _is_hangul(ch):
            syl.append(SylToken(ch, HANGUL, hangul_reading(ch)))
            i += 1
        elif _is_jamo(ch):
            syl.append(SylToken(ch, JAMO, hangul_reading(ch)))
            i += 1
        elif _is_kana(ch):
            j = i
            while j < n and _is_kana(word[j]):
                j += 1
            syl.extend(SylToken(t, KANA, r) for t, r in kana_morae(word[i:j]))
            i = j
        elif _is_kanji(ch):
            syl.append(SylToken(ch, KANJI, "", NOTE_KANJI))
            i += 1
        elif _is_latin(ch):
            j = i
            while j < n and (_is_latin(word[j]) or (_apostrophe(word[j]) and j + 1 < n and _is_latin(word[j + 1]))):
                j += 1
            run = word[i:j]
            parts = english_syllables(run)
            note = NOTE_LATIN if len(parts) > 1 else ""
            syl.extend(SylToken(p, LATIN, latin_reading(p), note) for p in parts)
            i = j
        elif ch.isdigit():
            j = i
            while j < n and (word[j].isdigit() or (word[j] in ".,:" and j + 1 < n and word[j + 1].isdigit())):
                j += 1
            syl.append(SylToken(word[i:j], NUMBER, "", NOTE_NUMBER))
            i = j
        else:
            i += 1  # 문장부호·기호·이모지: 음절 아님
    return WordTokens(word, tuple(syl))


def tokenize_line(line: str) -> list[WordTokens]:
    return [tokenize_word(w) for w in line.split()]


def split_lines(text: str) -> list[str]:
    """정규화한 뒤 줄 목록. 빈 줄도 남긴다(문단 구분). 끝의 빈 줄은 뺀다."""
    lines = [ln.rstrip() for ln in normalize_text(text).split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def detect_language(text: str) -> str:
    """글자 수로 대략 판단: ko / ja / en / '' (판단 불가)."""
    counts = {"ko": 0, "ja": 0, "en": 0}
    for ch in text:
        if _is_hangul(ch) or _is_jamo(ch):
            counts["ko"] += 1
        elif _is_kana(ch):
            counts["ja"] += 2  # 가나가 조금만 있어도 일본어 (한자는 중국어와 겹친다)
        elif _is_kanji(ch):
            counts["ja"] += 1
        elif _is_latin(ch):
            counts["en"] += 0.3  # 영어는 글자 수가 많아 낮춰 센다
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ""
