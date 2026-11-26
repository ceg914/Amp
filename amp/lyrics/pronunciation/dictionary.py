"""사용자 발음 사전 (v0.7.0 Feature H). 가사 원문 단어(어절)나 음절 → 발음 표기(+ IPA).

- 사전은 프로젝트마다 따로 둔다: ``LyricsState.pron_dictionary`` (lyrics.json에 저장). 다른 프로젝트에 섞이지 않고,
  항목을 넣거나 빼는 일도 가사 편집과 같은 실행 취소 기록에 들어간다
- 다른 프로젝트와 나누려면 사전 파일(JSON)로 내보내고 가져온다(``load``·``save``). 프로그램 전체에 자동으로 적용되는 사전은 없다
- 자동 발음을 만들 때만 적용한다. 사용자가 음절에서 직접 고친 발음(user_edited)은 사전보다 우선이며 바꾸지 않는다
- 단어 항목은 발음 표기의 음절 수(한글 글자 수·가나 박 수·'.' 구분)가 가사 음절 수와 같을 때만 음절마다 나눠 넣는다
  (다르면 적용하지 않고 그 이유를 남긴다). 음절 항목은 그 음절에 그대로 넣는다
- 사전 값도 가사 글자에서 정한 발음이다. 실제 부른 발음을 들은 것이 아니다
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from amp.lyrics.model import PronDictEntry

DictEntry = PronDictEntry
SCHEMA = "sonatlas.pron_dictionary"
SCHEMA_VERSION = 1
FILE_SUFFIX = ".sonatlas-dict.json"
SCOPES = ("word", "syllable")
SCOPE_LABELS = {"word": "단어", "syllable": "음절"}


@dataclass
class PronDictionary:
    entries: dict = field(default_factory=dict)  # (scope, key) → PronDictEntry (넣은 순서 유지)

    @classmethod
    def of(cls, entries) -> "PronDictionary":
        out = cls()
        for e in entries or ():
            out.entries[(e.scope, e.key)] = e
        return out

    def add(self, e: PronDictEntry) -> "PronDictionary":
        out = PronDictionary(dict(self.entries))
        out.entries.pop((e.scope, e.key), None)  # 같은 키는 새 값으로 (순서는 맨 뒤)
        out.entries[(e.scope, e.key)] = e
        return out

    def remove(self, scope: str, key: str) -> "PronDictionary":
        out = PronDictionary(dict(self.entries))
        out.entries.pop((scope, key), None)
        return out

    def merged(self, over: "PronDictionary | None") -> "PronDictionary":
        """over 항목이 같은 키를 덮는다."""
        if over is None:
            return self
        out = PronDictionary(dict(self.entries))
        for k, e in over.entries.items():
            out.entries.pop(k, None)
            out.entries[k] = e
        return out

    def word(self, key: str) -> PronDictEntry | None:
        return self.entries.get(("word", key))

    def syllable(self, key: str) -> PronDictEntry | None:
        return self.entries.get(("syllable", key))

    def as_tuple(self) -> tuple[PronDictEntry, ...]:
        return tuple(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION,
                "note": "SONATLAS 사용자 발음 사전. 가사 글자에서 정한 발음이며 실제 부른 발음이 아닙니다.",
                "entries": [{"key": e.key, "text": e.text, "ipa": e.ipa, "language": e.language, "scope": e.scope,
                             "note": e.note} for e in self.entries.values()]}


def split_units(text: str) -> list[str]:
    """발음 표기를 음절로: '.'이 있으면 그것으로, 없으면 한글 글자·가나(작은 글자·장음은 앞에 붙임)마다."""
    if "." in text:
        return [p for p in text.split(".")]
    out: list[str] = []
    small = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮー")
    for ch in text:
        if ch.isspace():
            continue
        if out and ch in small:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def entry_for_word(word_text: str, entries) -> tuple[PronDictEntry | None, str]:
    """단어 하나의 지금 발음(음절별 PronEntry 목록)으로 사전 항목을 만든다. 못 만들면 (None, 이유)."""
    es = list(entries)
    if not es or any(e is None or not e.text for e in es):
        return None, "이 단어에는 발음이 비어 있는 음절이 있어 사전에 넣지 않았습니다. 발음을 먼저 적어 주세요."
    if any("." in e.text for e in es):
        return None, "발음 표기에 '.'이 들어 있어 음절을 나눌 수 없습니다."
    ipa = ""
    if all(e.ipa for e in es) and not any("." in e.ipa for e in es):  # 한 음절 IPA에 '.'이 있으면 나눌 수 없어 비운다
        ipa = ".".join(e.ipa for e in es)
    langs = {e.language for e in es if e.language}
    lang = langs.pop() if len(langs) == 1 else ""
    return PronDictEntry(word_text, ".".join(e.text for e in es), ipa, lang, "word"), ""


def load(path: str | Path) -> PronDictionary:
    """사전 파일을 읽는다. 없거나 형식이 다르면 빈 사전 (예외를 내지 않음)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return PronDictionary()
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        return PronDictionary()
    out = PronDictionary()
    for x in d.get("entries") or []:
        try:
            e = PronDictEntry(str(x["key"]), str(x["text"]), str(x.get("ipa", "")), str(x.get("language", "")),
                              str(x.get("scope", "word")), str(x.get("note", "")))
        except (KeyError, TypeError):
            continue
        if e.key and e.scope in SCOPES:
            out.entries[(e.scope, e.key)] = e
    return out


def save(path: str | Path, d: PronDictionary) -> None:
    from amp.core.fileio import atomic_write_json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(path), d.to_json())
