"""일본어 한자 읽기 (v0.7.0 Feature H, 선택 설치): MeCab 형태소 분석기(fugashi, MIT·BSD-3) + UniDic 사전(unidic-lite, BSD-3).

- 가사 줄 전체를 문맥으로 분석해 한자 음절의 읽기(가나 표기 읽기)를 고른다. 가나 음절은 기존 가나 변환기가 그대로 맡는다
- 한 낱말에 한자가 하나면(오쿠리가나는 떼고) 그 한자 음절에 읽기를 넣는다 → '정상'
- 한자가 여럿 이어진 낱말(今日·名前 등)은 글자마다 읽기를 나눌 근거가 없어 첫 한자에 낱말 읽기 전체를 적고 '불확실'로 둔다
- 읽기는 사전이 문맥으로 고른 값이다. 가사에서 다른 읽기(当て字 등)를 쓰면 사용자가 고쳐야 한다. 실제 부른 발음이 아니다
- fugashi·사전이 없으면 이전처럼 '변환 불가'. 분석기는 처음 쓸 때 연다(사전 파일을 메모리에 올림)
"""

from __future__ import annotations

NAME = "japanese-mecab-unidic"
NOTE_SPLIT = "여러 한자의 읽기를 글자마다 나눌 수 없어 첫 글자에 낱말 읽기를 적었습니다."
NOTE_AUTO = "사전이 문맥으로 고른 읽기입니다. 가사에서 다른 읽기를 쓰면 직접 고치세요."


def available(find_spec=None) -> bool:
    """fugashi와 UniDic 사전(unidic-lite 또는 unidic)이 설치되어 있는지 (불러오지 않음)."""
    import importlib.util

    fs = find_spec or importlib.util.find_spec

    def has(m):
        try:
            return fs(m) is not None
        except (ImportError, ValueError):
            return False

    return has("fugashi") and (has("unidic_lite") or has("unidic"))


def _kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in s)


def _is_kana(c: str) -> bool:
    return 0x3041 <= ord(c) <= 0x30FF


def split_reading(surface: str, reading: str) -> list[tuple[str, str]]:
    """낱말 표기와 읽기(가타카나) → [(한자 덩어리, 그 읽기)]. 앞뒤 가나(오쿠리가나 등)는 읽기에서 뗀다.
    가운데에 가나가 끼어 덩어리가 여럿이면 앞 덩어리에 남은 읽기 전체를 준다 (나머지는 빈 읽기)."""
    s, r = list(surface), _kata(reading)
    while s and r and _is_kana(s[-1]) and r.endswith(_kata(s[-1])):
        r = r[:-1]
        s.pop()
    while s and r and _is_kana(s[0]) and r.startswith(_kata(s[0])):
        r = r[1:]
        s.pop(0)
    blocks: list[str] = []
    cur = ""
    for c in s:
        if _is_kana(c):
            if cur:
                blocks.append(cur)
                cur = ""
        else:
            cur += c
    if cur:
        blocks.append(cur)
    return [(b, r if k == 0 else "") for k, b in enumerate(blocks)]


class KanjiReadingAdapter:
    """한자 음절만 맡는다. ``convert_run`` 으로 줄 전체(문맥)를 받는다."""

    name = NAME
    language = "ja"

    def __init__(self, tagger_factory=None) -> None:
        from amp.lyrics import text as tx

        self.kinds = (tx.KANJI,)
        self._factory = tagger_factory
        self._tagger = None
        self.error = ""

    def tagger(self):
        if self._tagger is None and not self.error:
            try:
                if self._factory is not None:
                    self._tagger = self._factory()
                else:
                    import fugashi

                    self._tagger = fugashi.Tagger()
            except Exception as e:  # noqa: BLE001 - 사전이 깨져도 가사는 연다
                self.error = f"일본어 형태소 분석기를 열지 못했습니다: {e}"
        return self._tagger

    def _readings(self, text: str) -> list[tuple[int, int, str]]:
        """(시작, 끝, 읽기) 낱말 목록. 글자 위치는 text 기준."""
        out = []
        pos = 0
        for w in self.tagger()(text):
            pos += len(getattr(w, "white_space", "") or "")
            surf = w.surface
            start = text.find(surf, pos) if surf else pos
            if start < 0:
                start = pos
            kana = getattr(w.feature, "kana", None) or ""
            out.append((start, start + len(surf), kana if kana and kana != "*" else ""))
            pos = start + len(surf)
        return out

    def convert_run(self, flat, i: int, j: int):
        """flat: 줄의 (음절, 단어 번호, 끊김) 목록. [i, j) 한자 음절의 결과."""
        from amp.lyrics.pronunciation import AdapterResult, _kana_ipa

        if self.tagger() is None:
            return [AdapterResult("", "", "ja", "unavailable", (), (self.error,)) for _ in range(i, j)]
        text = ""
        offsets = []
        last_w = None
        for s, wi, _b in flat:
            if last_w is not None and wi != last_w:
                text += " "
            offsets.append(len(text))
            text += s.text
            last_w = wi
        words = self._readings(text)
        out = []
        for k in range(i, j):
            syl = flat[k][0]
            at = offsets[k]
            w = next(((a, b, rd) for a, b, rd in words if a <= at < b), None)
            if w is None or not w[2]:
                out.append(AdapterResult("", "", "ja", "unavailable", (), (f"사전에서 읽기를 찾지 못했습니다: {syl.text}",)))
                continue
            a, b, rd = w
            blocks = split_reading(text[a:b], rd)
            # 이 음절이 속한 한자 덩어리와 그 안의 위치
            pos, idx, inner = a, None, 0
            for bi, (blk, _r) in enumerate(blocks):
                start = text.find(blk, pos, b)
                if start <= at < start + len(blk):
                    idx, inner = bi, at - start
                    break
                pos = start + len(blk)
            if idx is None:
                out.append(AdapterResult("", "", "ja", "uncertain", (), (f"낱말 '{text[a:b]}'에서 이 한자의 읽기를 가리지 못했습니다.",)))
                continue
            blk, reading = blocks[idx]
            several = len(blk) > 1 or len(blocks) > 1
            value = reading if inner == 0 else ""
            ipa = _kana_ipa(value) if value else ""
            rules = ("MeCab+UniDic 문맥 읽기",)
            if several:
                out.append(AdapterResult(value, ipa, "ja", "uncertain", rules, (NOTE_SPLIT, NOTE_AUTO)))
            else:
                out.append(AdapterResult(value, ipa, "ja", "valid" if value else "unavailable", rules, (NOTE_AUTO,)))
        return out

    def convert(self, units):  # 줄 문맥 없이 불리면 (다른 경로) 변환하지 않는다
        from amp.lyrics.pronunciation import AdapterResult

        return [AdapterResult("", "", "ja", "unavailable", (), ("줄 문맥이 없어 한자를 읽지 않았습니다.",)) for _ in units]
