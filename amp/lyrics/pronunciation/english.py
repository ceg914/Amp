"""영어 발음 (v0.7.0 Feature H): CMU 발음 사전(CMUdict, BSD 계열 라이선스) 자료만 읽어 IPA로 바꾼다.

- 사전 파일은 install_lyrics가 --no-deps로 설치한 cmudict 패키지 안의 data/cmudict.dict를 경로로 읽는다.
  그 패키지의 파이썬 코드(GPL)는 불러오지 않는다. 사전이 없으면 이전처럼 '변환 불가'다
- 단어의 ARPAbet 음소를 모음(강세 숫자가 붙은 음소)마다 음절로 나누고, 모음 사이 자음은 영어에서 낱말 첫머리에 올 수 있는
  가장 긴 묶음을 뒤 음절로 보낸다(최대 두음). 가사 음절 수와 사전 음절 수가 같으면 음절마다 IPA를 붙인다
- 가사가 한 음절인데 사전이 여러 음절이면(fire·our) 이어 적는다. 그 밖에 수가 다르면 단어 발음 전체를 첫 음절에 적고
  나머지는 비운 뒤 '불확실'로 둔다. 사전에 없는 단어는 '변환 불가'
- 발음은 가사 글자에서 사전으로 찾은 값이다. 가수가 실제로 부른 발음이 아니다
"""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

NAME = "english-cmudict"
VOWELS = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
          "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u"}
CONSONANTS = {"B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
              "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
              "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
ONSETS = {("S", "T", "R"), ("S", "P", "R"), ("S", "P", "L"), ("S", "K", "R"), ("S", "K", "W"), ("S", "T"), ("S", "P"),
          ("S", "K"), ("S", "M"), ("S", "N"), ("S", "L"), ("S", "W"), ("P", "L"), ("P", "R"), ("B", "L"), ("B", "R"),
          ("T", "R"), ("D", "R"), ("K", "L"), ("K", "R"), ("G", "L"), ("G", "R"), ("F", "L"), ("F", "R"),
          ("TH", "R"), ("SH", "R"), ("T", "W"), ("D", "W"), ("K", "W"), ("G", "W"), ("P", "Y"), ("B", "Y"),
          ("K", "Y"), ("F", "Y"), ("M", "Y"), ("HH", "Y"), ("V", "Y")}

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def load(path: str | Path) -> dict[str, tuple[str, ...]]:
    """cmudict.dict → 단어(소문자) → 발음 문자열들 (ARPAbet, 공백 구분). 같은 경로는 한 번만 읽는다 (약 1.5초)."""
    key = str(Path(path).resolve())
    with _lock:
        if key in _cache:
            return _cache[key]
        out: dict[str, tuple[str, ...]] = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith(";;;"):
                    continue
                word, _sp, phones = line.partition(" ")
                word = re.sub(r"\(\d+\)$", "", word).lower()
                out[word] = out.get(word, ()) + (sys.intern(phones.strip()),)
        _cache[key] = out
        return out


def find_installed(find_spec=None) -> Path | None:
    """설치된 cmudict 패키지의 사전 파일 경로 (패키지를 불러오지 않음)."""
    import importlib.util

    fs = find_spec or importlib.util.find_spec
    try:
        spec = fs("cmudict")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    p = Path(list(spec.submodule_search_locations)[0]) / "data" / "cmudict.dict"
    return p if p.is_file() else None


def syllabify(phones: list[str]) -> list[list[str]]:
    """ARPAbet 음소 → 음절 (최대 두음)."""
    nuclei = [i for i, p in enumerate(phones) if p[-1:].isdigit()]
    if not nuclei:
        return [phones] if phones else []
    sylls = []
    start = 0
    for k, v in enumerate(nuclei):
        if k + 1 < len(nuclei):
            cons = phones[v + 1:nuclei[k + 1]]
            take = 0
            for n in range(len(cons), 0, -1):
                tail = tuple(cons[len(cons) - n:])
                if n == 1 and tail[0] != "NG" or tail in ONSETS:
                    take = n
                    break
            cut = v + 1 + len(cons) - take
            sylls.append(phones[start:cut])
            start = cut
        else:
            sylls.append(phones[start:])
    return sylls


def ipa_of(phones: list[str], mark_stress: bool = True) -> str:
    s = []
    stress = ""
    for p in phones:
        base, digit = (p[:-1], p[-1]) if p[-1:].isdigit() else (p, "")
        if digit == "1":
            stress = "ˈ"
        elif digit == "2":
            stress = stress or "ˌ"
        if base in VOWELS:
            v = VOWELS[base]
            if base == "AH" and digit == "0":
                v = "ə"
            elif base == "ER" and digit == "0":
                v = "ɚ"
            s.append(v)
        else:
            s.append(CONSONANTS.get(base, base.lower()))
    return (stress if mark_stress else "") + "".join(s)


class EnglishAdapter:
    name = NAME
    language = "en"

    def __init__(self, path: str | Path) -> None:
        from amp.lyrics import text as tx

        self.kinds = (tx.LATIN,)
        self.path = Path(path)
        if not self.path.is_file():
            raise OSError(f"영어 발음 사전 파일이 없습니다: {path}")
        self._words: dict | None = None

    @property
    def words(self) -> dict:
        """처음 쓸 때 읽는다 (앱은 시작 뒤 작업 스레드에서 preload로 미리 읽어 화면이 멈추지 않게 한다)."""
        if self._words is None:
            self._words = load(self.path)
        return self._words

    def preload(self) -> None:
        _ = self.words

    def convert(self, units):
        from amp.lyrics.pronunciation import AdapterResult

        out: list = [None] * len(units)
        groups: dict[int, list[int]] = {}
        for i, u in enumerate(units):
            groups.setdefault(u[2], []).append(i)
        for _w, idx in groups.items():
            word = "".join(units[i][0] for i in idx)
            key = re.sub(r"[^a-z']", "", word.lower())
            prons = self.words.get(key) or self.words.get(key.strip("'"))
            if not prons:
                for i in idx:
                    out[i] = AdapterResult("", "", "en", "unavailable", (), (f"영어 발음 사전에 없는 단어입니다: {word}",))
                continue
            phones = prons[0].split()
            notes = (f"사전에 다른 발음 {len(prons) - 1}개가 더 있습니다.",) if len(prons) > 1 else ()
            sylls = syllabify(phones)
            mono = len(sylls) == 1  # 한 음절 낱말은 강세 표시를 붙이지 않는다
            if len(sylls) == len(idx):
                for i, sy in zip(idx, sylls):
                    ipa = ipa_of(sy, mark_stress=not mono)
                    out[i] = AdapterResult(ipa.lstrip("ˈˌ"), ipa, "en", "valid", ("CMUdict", "최대 두음 음절 나눔"), notes)
            elif len(idx) == 1:  # 가사 한 음절에 사전 여러 음절 (fire, our): 한 음절로 부른다고 보고 이어 적는다
                ipa = "".join(ipa_of(sy, mark_stress=False) for sy in sylls)
                note = (f"사전은 {len(sylls)}음절이지만 가사가 한 음절이라 이어 적었습니다.",)
                out[idx[0]] = AdapterResult(ipa, ipa, "en", "valid", ("CMUdict",), notes + note)
            else:
                whole = ".".join(ipa_of(sy) for sy in sylls)
                note = (f"사전 음절 수({len(sylls)})와 가사 음절 수({len(idx)})가 달라 첫 음절에 단어 발음을 적었습니다.",)
                for k, i in enumerate(idx):
                    out[i] = AdapterResult(whole.replace("ˈ", "").replace("ˌ", "") if k == 0 else "", whole if k == 0 else "",
                                           "en", "uncertain", ("CMUdict",), notes + note)
        return out
