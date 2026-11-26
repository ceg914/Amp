"""발음 엔진 (STEP 4): 가사 텍스트 → 음절별 발음 표기·IPA. 오디오 모델 없이 텍스트만으로 돈다.

발음은 가사 원문에서 추정한 값이다. 가수의 실제 발음을 인식한 것이 아니다.

- 언어별 변환기(``LanguagePronunciationAdapter``)를 음절 종류(한글·가나·라틴 문자 …)에 따라 고른다. 한 줄에 여러 언어가 섞여도 된다.
- 한국어: 표준 발음법 규칙(``korean``). 일본어: 가나 박 단위(``japanese``). 숫자: 읽는 법이 문맥마다 달라 '변환 불가'.
- 영어 (v0.7.0): CMU 발음 사전 자료가 설치되어 있고 앱이 ``enable_english``를 부른 경우에만 (``english``). 아니면 '변환 불가'.
- 일본어 한자 (v0.7.0): fugashi+UniDic이 설치되어 있고 앱이 ``enable_kanji``를 부른 경우에만 줄 문맥으로 읽기 (``kanji``).
- 사용자 발음 사전 (v0.7.0, ``dictionary``): 프로젝트마다 따로(``LyricsState.pron_dictionary``). 자동 발음을 만들 때 단어·음절 항목을
  먼저 쓴다. 음절에서 직접 고친 발음이 더 우선이다.
- 사용자가 고친 발음(``user_edited``)은 다시 만들 때 덮어쓰지 않는다. 원문이나 문맥이 바뀌면 ``stale`` (재검토 필요)만 붙인다.
- 자동 생성 결과는 ``CONVERTER``·``VERSION`` 을 남긴다. 규칙이 바뀌어 버전이 오르면 자동 생성 결과만 다시 만든다.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, replace
from typing import Iterable, Protocol

from amp.lyrics import text as tx
from amp.lyrics.model import LyricLine, LyricsState, LyricsText, PronEntry
from amp.lyrics.pronunciation import japanese, korean

CONVERTER = "sonatlas-g2p"
VERSION = "1"

N_EN = "영어 발음 변환기는 넣지 않았습니다(오프라인에서 쓸 수 있는 검증된 사전이 없음). 발음을 직접 입력할 수 있습니다."
N_KANJI = "한자는 읽기 사전이 없어 변환하지 않습니다. 읽는 소리를 직접 입력할 수 있습니다."
N_NUMBER = "숫자는 읽는 법이 문맥마다 달라(세 개·삼 년) 직접 입력해야 합니다."
N_JAMO = "낱자(ㅋ, ㅠ 등)는 읽는 소리가 정해져 있지 않습니다."
N_OTHER = "이 글자는 발음 변환을 지원하지 않습니다."
N_IPA_USER = "직접 입력한 발음이 한글·가나가 아니어서 IPA를 만들지 않았습니다."

STATUS_LABELS = {"valid": "규칙 적용", "uncertain": "불확실", "unavailable": "변환 불가"}
SOURCE_LABELS = {"generated": "자동 생성", "user": "사용자 확정", "imported": "가져옴"}
# 화면에서 색 외에 붙이는 상태 기호 (UI 명세 2.5: 상태를 색만으로 구분하지 않는다)
MARK_STALE, MARK_USER, MARK_UNCERTAIN = "↻", "✓", "?"


def status_mark(e: PronEntry | None) -> str:
    """발음 상태 기호: 재검토 필요 ↻ > 사용자 확정 ✓ > 불확실·변환 불가 ? > 없음."""
    if e is None:
        return MARK_UNCERTAIN
    if e.stale:
        return MARK_STALE
    if e.user_edited:
        return MARK_USER
    return MARK_UNCERTAIN if e.status != "valid" else ""


def marked_text(e: PronEntry | None, ipa: bool = False) -> str:
    """기호를 붙인 발음(또는 IPA). 값이 없으면 발음은 '?', IPA는 '—' (기호를 겹쳐 붙이지 않음)."""
    if e is None:
        return "—" if ipa else MARK_UNCERTAIN
    value = e.ipa if ipa else e.text
    if not value:
        if ipa:
            return "—"
        return "∅" + (MARK_STALE if e.stale else MARK_USER) if e.user_edited else MARK_UNCERTAIN
    return value + status_mark(e)

_PUNCT_BREAK = set(string.punctuation) | set("、。，．！？…～〜「」『』（）·")


@dataclass(frozen=True)
class AdapterResult:
    text: str
    ipa: str
    language: str
    status: str
    rules: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class LanguagePronunciationAdapter(Protocol):
    """언어 하나의 발음 변환기. 한 줄 안에서 자기 언어 음절만 받아 문맥을 보고 변환한다."""

    name: str
    language: str
    kinds: tuple[str, ...]

    def convert(self, units: list[tuple[str, str, int, bool]]) -> list[AdapterResult]:
        """units: (음절 글자, 정렬용 로마자, 어절 번호, 이 음절 뒤에서 끊김). 같은 길이 결과를 돌려준다."""
        ...


class KoreanAdapter:
    name = "korean-standard-pronunciation"
    language = "ko"
    kinds = (tx.HANGUL,)

    def convert(self, units):
        breaks = {i for i, u in enumerate(units) if u[3]}
        res = korean.pronounce([(u[0], u[2]) for u in units], breaks)
        out = []
        for r in res:
            out.append(AdapterResult(r.text, "", "ko", "uncertain" if r.notes else "valid", r.rules, r.notes))
        return out


class JapaneseKanaAdapter:
    name = "japanese-kana"
    language = "ja"
    kinds = (tx.KANA,)

    def convert(self, units):
        out: list[AdapterResult] = []
        run: list[tuple[str, str]] = []

        def flush():
            for kata, ipa, notes in japanese.convert_morae(run):
                status = "uncertain" if notes else ("valid" if japanese.ipa_is_valid(ipa) else "uncertain")
                out.append(AdapterResult(kata, ipa, "ja", status, (), notes))
            run.clear()

        for i, (t, reading, _w, brk) in enumerate(units):
            run.append((t, reading))
            if brk or i == len(units) - 1:
                flush()
        return out


ADAPTERS: tuple[LanguagePronunciationAdapter, ...] = (KoreanAdapter(), JapaneseKanaAdapter())
_EXTRA: list = []  # v0.7.0 선택 변환기 (영어 사전이 설치되어 있을 때 앱이 켬)


def enable_english(path) -> bool:
    """CMU 발음 사전 파일로 영어 변환기를 켠다. 파일을 읽지 못하면 False (영어는 계속 '변환 불가')."""
    from amp.lyrics.pronunciation import english

    disable_english()
    try:
        _EXTRA.append(english.EnglishAdapter(path))
    except OSError:
        return False
    return True


def disable_english() -> None:
    _EXTRA[:] = [a for a in _EXTRA if getattr(a, "language", "") != "en"]


def english_enabled() -> bool:
    return any(getattr(a, "language", "") == "en" for a in _EXTRA)


def preload() -> None:
    """켜 둔 선택 변환기의 사전을 미리 읽는다 (작업 스레드에서 부름). 실패해도 조용히 넘어간다 (쓸 때 다시 시도)."""
    for a in list(_EXTRA):
        try:
            getattr(a, "preload", lambda: None)()
        except Exception:  # noqa: BLE001
            pass


def enable_kanji(tagger_factory=None) -> bool:
    """일본어 한자 읽기(MeCab+UniDic)를 켠다. fugashi·사전이 없으면 False. 분석기는 처음 쓸 때 연다."""
    from amp.lyrics.pronunciation import kanji

    disable_kanji()
    if tagger_factory is None and not kanji.available():
        return False
    _EXTRA.append(kanji.KanjiReadingAdapter(tagger_factory))
    return True


def disable_kanji() -> None:
    _EXTRA[:] = [a for a in _EXTRA if getattr(a, "name", "") != "japanese-mecab-unidic"]


def kanji_enabled() -> bool:
    return any(getattr(a, "name", "") == "japanese-mecab-unidic" for a in _EXTRA)


_UNAVAILABLE = {tx.LATIN: ("en", N_EN), tx.KANJI: ("ja", N_KANJI), tx.NUMBER: ("", N_NUMBER), tx.JAMO: ("ko", N_JAMO)}


def _adapter_for(kind: str) -> LanguagePronunciationAdapter | None:
    for a in (*_EXTRA, *ADAPTERS):
        if kind in a.kinds:
            return a
    return None


def _line_units(line: LyricLine):
    """줄 안 음절을 순서대로 (음절, 어절 번호, 뒤에서 끊기는가)로. 문장부호로 끝나는 어절 뒤와 줄 끝에서 끊는다."""
    flat = []
    for wi, w in enumerate(line.words):
        punct_end = bool(w.text) and w.text[-1] in _PUNCT_BREAK
        for si, s in enumerate(w.syllables):
            last = si == len(w.syllables) - 1
            flat.append((s, wi, last and punct_end))
    return flat


def _dict_of(entries):
    if not entries:
        return None
    from amp.lyrics.pronunciation.dictionary import PronDictionary

    return entries if isinstance(entries, PronDictionary) else PronDictionary.of(entries)


def generate(text: LyricsText, dictionary=None) -> dict[str, PronEntry]:
    """모든 음절의 자동 발음. 같은 변환기를 쓰는 음절이 이어지는 구간마다 문맥을 보고 변환한다.
    dictionary(사용자 발음 사전 항목들)가 있으면 변환 결과보다 먼저 쓴다."""
    out: dict[str, PronEntry] = {}
    d = _dict_of(dictionary)
    for line in text.lines:
        flat = _line_units(line)
        i = 0
        while i < len(flat):
            s = flat[i][0]
            adapter = _adapter_for(s.kind)
            if adapter is None:
                lang, note = _UNAVAILABLE.get(s.kind, ("", N_OTHER))
                out[s.id] = _entry(s.id, s.text, AdapterResult("", "", lang, "unavailable", (), (note,)), adapter_name="")
                i += 1
                continue
            j = i
            while j < len(flat) and _adapter_for(flat[j][0].kind) is adapter:
                j += 1
            run = flat[i:j]
            units = [(u[0].text, u[0].reading, u[1], u[2] or k == len(run) - 1) for k, u in enumerate(run)]
            results = adapter.convert_run(flat, i, j) if hasattr(adapter, "convert_run") else adapter.convert(units)
            for (syl, _w, _b), r in zip(run, results):
                out[syl.id] = _entry(syl.id, syl.text, r, adapter.name)
            i = j
        if d is not None:
            _apply_dictionary(line, out, d)
    _fill_ipa(text, out, auto=True)
    return out


def _apply_dictionary(line: LyricLine, out: dict[str, PronEntry], d) -> None:
    """사용자 사전: 단어 항목(음절 수가 맞을 때만 나눠 넣음) → 음절 항목 순서로 자동 발음을 바꾼다."""
    from amp.lyrics.pronunciation.dictionary import split_units

    model = f"{CONVERTER}/user-dictionary"
    for w in line.words:
        sylls = list(w.syllables)
        e = d.word(w.text) or d.word(w.text.strip(string.punctuation + "…·“”‘’"))
        if e is not None:
            parts = split_units(e.text)
            ipas = e.ipa.split(".") if e.ipa else []
            if len(parts) == len(sylls):
                for k, sy in enumerate(sylls):
                    old = out[sy.id]
                    ipa = ipas[k] if len(ipas) == len(sylls) else ""
                    lang = e.language or old.language
                    out[sy.id] = PronEntry(sy.id, sy.text, parts[k], ipa, lang, "generated", "valid", False, model,
                                           VERSION, ("사용자 사전",), (), False, parts[k], ipa, False)
                continue
            for sy in sylls:
                old = out[sy.id]
                out[sy.id] = replace(old, notes=old.notes + (f"사용자 사전 '{w.text}'의 음절 수({len(parts)})가 가사 음절 수"
                                                             f"({len(sylls)})와 달라 적용하지 않았습니다.",))
        for sy in sylls:
            se = d.syllable(sy.text)
            if se is not None:
                old = out[sy.id]
                lang = se.language or old.language
                out[sy.id] = PronEntry(sy.id, sy.text, se.text, se.ipa, lang, "generated", "valid", False, model, VERSION,
                                       ("사용자 사전",), (), False, se.text, se.ipa, False)


def _entry(sid: str, original: str, r: AdapterResult, adapter_name: str) -> PronEntry:
    model = f"{CONVERTER}/{adapter_name}" if adapter_name else CONVERTER
    return PronEntry(sid, original, r.text, r.ipa, r.language, "generated", r.status, False, model, VERSION, r.rules,
                     r.notes, False, r.text, r.ipa, False)


def _fill_ipa(text: LyricsText, entries: dict[str, PronEntry], auto: bool) -> None:
    """한국어 IPA는 줄 안 발음 순서 전체를 보고 만든다(유성음 사이 평음 등). 사용자가 IPA를 고친 항목은 그대로 둔다.

    auto=True면 자동 발음(auto_text)으로 auto_ipa를, 아니면 최종 발음(text)으로 ipa를 채운다.
    """
    for line in text.lines:
        flat = _line_units(line)
        chars: list[tuple[str, int]] = []
        owners: list[str | None] = []
        for s, wi, brk in flat:
            e = entries.get(s.id)
            value = None if e is None else (e.auto_text if auto else e.text)
            korean_here = e is not None and e.language == "ko" and (
                s.kind == tx.HANGUL or (not auto and e.user_edited and all(korean.is_hangul(c) for c in value)))
            if korean_here and not auto and e.user_edited and len(value) > 1:
                # 사용자는 '십이'처럼 철자로 적기도 한다. 발음 규칙을 거쳐 실제 소리([시비])로 IPA를 만든다 (발음대로 적었으면 그대로)
                value = "".join(r.text for r in korean.pronounce([(c, 0) for c in value]))
            if not korean_here:
                chars.append(("|", wi))  # 한국어가 아닌 음절: 문맥을 끊는다
                owners.append(None)
                continue
            for ch in value:
                chars.append((ch, wi))
                owners.append(s.id)
            if brk:
                chars.append(("|", wi))
                owners.append(None)
        ipas = korean.syllable_ipa(chars)
        per: dict[str, list[str]] = {}
        for sid, ipa in zip(owners, ipas):
            if sid is not None:
                per.setdefault(sid, []).append(ipa)
        for sid, parts in per.items():
            e = entries[sid]
            ipa = ".".join(p for p in parts if p)
            if auto:
                entries[sid] = replace(e, auto_ipa=ipa, ipa=ipa if not e.user_edited else e.ipa)
            elif not e.ipa_edited:
                entries[sid] = replace(e, ipa=ipa)


def sync(state: LyricsState) -> LyricsState:
    """텍스트에 맞춰 발음을 맞춘다. 자동 생성 항목은 새로 만들고, 사용자 항목은 값을 지키며 원문·문맥이 바뀌었으면 재검토 표시."""
    text = state.text
    gen = generate(text, state.pron_dictionary)
    old = state.pronunciation
    out: dict[str, PronEntry] = {}
    for s in text.syllables:
        g = gen[s.id]
        e = old.get(s.id)
        if e is None or not e.user_edited:
            out[s.id] = g
            continue
        stale = e.stale or e.original_text != s.text or e.auto_text != g.text
        out[s.id] = replace(e, stale=stale, auto_text=g.text, auto_ipa=g.auto_ipa, language=e.language or g.language)
    _user_ipa(text, out)
    if out == dict(old):
        return state
    return replace(state, pronunciation=out)


def _user_ipa(text: LyricsText, entries: dict[str, PronEntry]) -> None:
    """사용자 발음이 있는 줄은 최종 발음 순서로 IPA를 다시 만든다(이웃 음절의 유성음화가 바뀔 수 있다)."""
    users = {sid for sid, e in entries.items() if e.user_edited}
    if not users:
        return
    lines = [ln for ln in text.lines if any(s.id in users for s in ln.syllables)]
    sub = LyricsText(text.language, tuple(lines))
    _fill_ipa(sub, entries, auto=False)
    for sid in users:
        e = entries[sid]
        if e.language == "ja" and not e.ipa_edited:
            entries[sid] = replace(e, ipa=_kana_ipa(e.text))
        elif e.language not in ("ko", "ja") and not e.ipa_edited:
            entries[sid] = replace(e, ipa=_any_ipa(e.text))


def _kana_ipa(value: str) -> str:
    morae = tx.kana_morae("".join(c for c in value if 0x3041 <= ord(c) <= 0x30FF))
    return "".join(ipa for _k, ipa, _n in japanese.convert_morae(morae))


def _any_ipa(value: str) -> str:
    """언어를 정하지 못한 음절(영어·숫자)에 사용자가 한글이나 가나로 발음을 적었으면 그 소리로 IPA를 만든다."""
    if value and all(korean.is_hangul(c) for c in value):
        said = [(r.text, 0) for r in korean.pronounce([(c, 0) for c in value])]
        return ".".join(korean.syllable_ipa(said))
    if value and all(0x3041 <= ord(c) <= 0x30FF for c in value):
        return _kana_ipa(value)
    return ""


def set_user(state: LyricsState, sid: str, value: str, ipa: str | None = None) -> LyricsState:
    """사용자가 발음을 정한다(확정). 빈 값은 '앞 음절과 합쳐 발음해 따로 소리가 없음'. ipa를 주면 IPA도 사용자 값으로 둔다."""
    if sid not in state.text.index:
        raise KeyError(sid)
    syl = state.text.syllable(sid)
    base = state.pronunciation.get(sid) or generate(state.text, state.pron_dictionary)[sid]
    value = value.strip()
    notes: tuple[str, ...] = ()
    lang = base.language
    if lang not in ("ko", "ja") and value:
        lang = "ko" if all(korean.is_hangul(c) for c in value) else ("ja" if all(0x3041 <= ord(c) <= 0x30FF for c in value)
                                                                 else lang)
    e = replace(base, original_text=syl.text, text=value, source="user", status="valid", user_edited=True, stale=False,
                language=lang, rules=(), notes=notes, ipa_edited=ipa is not None, ipa=(ipa.strip() if ipa is not None else ""))
    entries = dict(state.pronunciation)
    entries[sid] = e
    _user_ipa(state.text, entries)
    if value and not entries[sid].ipa and ipa is None:
        entries[sid] = replace(entries[sid], notes=(N_IPA_USER,))
    return replace(state, pronunciation=entries).with_history("발음 수정", f"{syl.text} → [{value}]")


def confirm(state: LyricsState, ids: Iterable[str]) -> LyricsState:
    """지금 발음을 사용자 확정으로 둔다(자동 발음이면 그 값을 확정, 재검토 표시는 지운다)."""
    entries = dict(state.pronunciation)
    n = 0
    for sid in ids:
        e = entries.get(sid)
        if e is None:
            continue
        entries[sid] = replace(e, source="user" if e.source == "generated" else e.source, user_edited=True, stale=False,
                               original_text=state.text.syllable(sid).text)
        n += 1
    if not n:
        return state
    return replace(state, pronunciation=entries).with_history("발음 확정", f"{n}개")


def reset(state: LyricsState, ids: Iterable[str]) -> LyricsState:
    """사용자 발음을 지우고 기본(자동) 발음으로 되돌린다."""
    ids = [i for i in ids if i in state.pronunciation and state.pronunciation[i].user_edited]
    if not ids:
        return state
    entries = {k: v for k, v in state.pronunciation.items() if k not in ids}
    out = sync(replace(state, pronunciation=entries))
    return out.with_history("기본 발음으로 되돌림", f"{len(ids)}개")


def regenerate(state: LyricsState, ids: Iterable[str] | None = None) -> tuple[LyricsState, int, int]:
    """자동 발음을 지금 규칙으로 다시 만든다. 사용자 발음은 건드리지 않는다. (새 상태, 다시 만든 수, 지킨 사용자 발음 수)."""
    chosen = set(state.text.order) if ids is None else {i for i in ids if i in state.text.index}
    entries = {k: v for k, v in state.pronunciation.items() if not (k in chosen and not v.user_edited)}
    kept = sum(1 for k in chosen if k in entries)
    out = sync(replace(state, pronunciation=entries))
    if out.pronunciation == state.pronunciation:
        return state, len(chosen) - kept, kept
    return out.with_history("발음 다시 생성", f"{len(chosen) - kept}개"), len(chosen) - kept, kept


def set_dictionary(state: LyricsState, entries, action: str = "발음 사전 수정",
                   detail: str = "") -> tuple[LyricsState, int, int]:
    """프로젝트 사용자 발음 사전을 바꾸고 자동 발음을 다시 만든다. 직접 고친 발음은 값을 그대로 둔다.
    (새 상태, 값이 바뀐 자동 발음 수, 지킨 사용자 발음 수). 바뀐 것이 없으면 state를 그대로 돌려준다."""
    entries = tuple(entries)
    if entries == state.pron_dictionary:
        return state, 0, 0
    new = sync(replace(state, pron_dictionary=entries))
    old_p, new_p = state.pronunciation, dict(new.pronunciation)
    changed = kept = 0
    for sid, e in new_p.items():
        o = old_p.get(sid)
        if e.user_edited:
            kept += 1
            # 사전에 넣은 값이 사용자 발음과 같으면 사전 때문에 재검토 표시를 새로 붙이지 않는다
            if o is not None and not o.stale and e.stale and e.original_text == o.original_text and e.auto_text == e.text:
                new_p[sid] = replace(e, stale=False)
        elif o is None or (o.text, o.ipa, o.status) != (e.text, e.ipa, e.status):
            changed += 1
    new = replace(new, pronunciation=new_p)
    return new.with_history(action, detail), changed, kept


def summary(state: LyricsState) -> dict[str, int]:
    out = {"valid": 0, "uncertain": 0, "unavailable": 0, "user": 0, "stale": 0}
    for s in state.text.syllables:
        e = state.pronunciation.get(s.id)
        if e is None:
            continue
        out[e.status] = out.get(e.status, 0) + 1
        out["user"] += e.user_edited
        out["stale"] += e.stale
    return out
