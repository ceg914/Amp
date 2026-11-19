"""일본어 가나 → 발음 표기(가타카나) · IPA. 박(모라) 단위.

가사 텍스트 처리(``amp.lyrics.text.kana_morae``)가 이미 가나를 박으로 나누고 헵번식 로마자를 붙여 둔다.
여기서는 그 로마자를 IPA로 옮긴다. 한자는 읽기 사전이 없어 다루지 않는다(변환 불가).

IPA 간략 표기 규칙: 모음 a i ɯ e o, し[ɕi] ち[tɕi] つ[tsɯ] ふ[ɸɯ] ひ[çi] に[ɲi] じ·ぢ[dʑi] ず·づ[zɯ],
요음은 [kʲa]처럼 구개음화 표시, ら행 [ɾ], わ[ɰa], ん은 뒤 소리에 따라 [m n ŋ ɴ], っ은 뒤 자음을 겹침, ー는 [ː].
무성화(す·し의 모음이 약해지는 현상)는 표기하지 않는다.
"""

from __future__ import annotations

U_PARTICLE = "조사로 쓰이면 は는 [wa], へ는 [e]로 읽습니다. 문맥을 알 수 없어 글자 소리로 두었습니다."

_CONS = {
    "k": "k", "g": "ɡ", "s": "s", "z": "z", "t": "t", "d": "d", "n": "n", "h": "h", "b": "b", "p": "p", "m": "m",
    "r": "ɾ", "w": "ɰ", "f": "ɸ", "v": "v",
}
_SPECIAL = {
    "shi": "ɕi", "chi": "tɕi", "tsu": "tsɯ", "fu": "ɸɯ", "ji": "dʑi", "hi": "çi", "ni": "ɲi", "zu": "zɯ",
    "sha": "ɕa", "shu": "ɕɯ", "sho": "ɕo", "she": "ɕe", "cha": "tɕa", "chu": "tɕɯ", "cho": "tɕo", "che": "tɕe",
    "ja": "dʑa", "ju": "dʑɯ", "jo": "dʑo", "je": "dʑe", "nya": "ɲa", "nyu": "ɲɯ", "nyo": "ɲo", "hya": "ça",
    "hyu": "çɯ", "hyo": "ço", "wo": "o", "yi": "i",
}
_VOWEL = {"a": "a", "i": "i", "u": "ɯ", "e": "e", "o": "o"}
IPA_SYMBOLS = set("aiɯeokɡszdtnhbpmɾɰɸvjɕçɲʑʲɴŋː ʔ.")


def mora_ipa(romaji: str) -> str:
    """헵번식 박 하나 → IPA. 모르는 형태면 ""."""
    r = romaji.lower()
    if not r:
        return ""
    if r in _SPECIAL:
        return _SPECIAL[r]
    if r in _VOWEL:
        return _VOWEL[r]
    if r[-1] not in _VOWEL:
        return ""
    v = _VOWEL[r[-1]]
    head = r[:-1]
    if head == "y":
        return "j" + v
    if len(head) == 2 and head[1] == "y" and head[0] in _CONS:
        return _CONS[head[0]] + "ʲ" + v  # きゃ → kʲa
    if len(head) == 1 and head in _CONS:
        return _CONS[head] + v
    return ""


def _n_before(next_ipa: str) -> str:
    if not next_ipa:
        return "ɴ"
    c = next_ipa[0]
    if c in "pbm":
        return "m"
    if c in "tdnɾz" or next_ipa.startswith("ts") or next_ipa.startswith("dʑ"):
        return "n"
    if c in "kɡ":
        return "ŋ"
    return "ɴ"


def _first_consonant(ipa: str) -> str:
    for pre in ("tɕ", "ts", "dʑ"):
        if ipa.startswith(pre):
            return pre[0]  # っち → [t.tɕi]
    return ipa[0] if ipa and ipa[0] not in "aiɯeo" else "ʔ"


def to_katakana(text: str) -> str:
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in text)


def convert_morae(morae: list[tuple[str, str]]) -> list[tuple[str, str, tuple[str, ...]]]:
    """[(박 글자, 로마자)] → [(가타카나, IPA, 불확실 사유)]. 로마자가 없으면 IPA ""."""
    base = [mora_ipa(r) if t[0] not in "っッー" and t != "ん" and t != "ン" else "" for t, r in morae]
    out = []
    for i, (t, _r) in enumerate(morae):
        nxt = base[i + 1] if i + 1 < len(base) else ""
        if t in ("ん", "ン"):
            ipa = _n_before(nxt)
        elif t[0] in "っッ":
            ipa = _first_consonant(nxt) if nxt else "ʔ"
        elif t == "ー":
            ipa = "ː" if i > 0 else ""
        else:
            ipa = base[i]
        notes = (U_PARTICLE,) if t in ("は", "へ", "ハ", "ヘ") else ()
        out.append((to_katakana(t), ipa, notes))
    return out


def ipa_is_valid(ipa: str) -> bool:
    return bool(ipa) and all(ch in IPA_SYMBOLS for ch in ipa)
