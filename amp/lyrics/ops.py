"""가사 편집 연산 (Qt 비의존 순수 함수). 모두 새 ``LyricsState`` 를 돌려주고 입력을 바꾸지 않는다.

텍스트 반영(``apply_text``)은 바뀐 줄만 다시 나눈다.
- 그대로인 줄은 줄·단어·음절 객체를 그대로 쓴다 (사용자가 나누거나 합친 음절도 유지).
- 바뀐 줄 안에서도 글자가 같은 음절은 id를 이어받아 시간·음표 연결이 남는다.
- 같은 자리에서 글자만 바뀐 음절은 시간을 남기되 ``stale`` (다시 정렬 필요)로 표시한다.
- 새로 생긴 음절은 미정렬이다. 바뀐 덩어리의 이전 시간 범위가 있으면 그 안에 글자 수로 나눈 추정값을 넣는다.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import replace
from typing import Iterable, Mapping

from amp.core.errors import EditError
from amp.core.ids import new_id
from amp.lyrics import text as tx
from amp.lyrics.model import (
    LyricLine,
    LyricsState,
    LyricsText,
    LyricSyllable,
    LyricWord,
    NoteLink,
    SyllableTiming,
)

MIN_SYLLABLE_S = 0.02  # 음절 최소 길이


# --- 텍스트 → 구조 ----------------------------------------------------------------------

def _make_word(word: tx.WordTokens, ids: list[str] | None = None) -> LyricWord:
    syls = []
    for i, t in enumerate(word.syllables):
        sid = ids[i] if ids and i < len(ids) and ids[i] else new_id()
        syls.append(LyricSyllable(sid, t.text, t.kind, t.reading, t.note))
    return LyricWord(new_id(), word.text, tuple(syls))


def make_line(text: str, origin: str = "manual", review: str = "", asr_segment: str | None = None) -> LyricLine:
    return LyricLine(new_id(), text, tuple(_make_word(w) for w in tx.tokenize_line(text)), origin, review, asr_segment)


def make_text(text: str, language: str = "", origin: str = "manual") -> LyricsText:
    lines = tx.split_lines(text)
    return LyricsText(language or tx.detect_language(text), tuple(make_line(ln, origin) for ln in lines))


def _rebuild_line(new_text: str, old: LyricLine | None, origin: str,
                  carry: dict[str, str]) -> LyricLine:
    """바뀐 줄을 다시 나눈다. ``carry`` 는 새 토큰 위치 → 이어받을 음절 id (호출자가 채움)."""
    words = tx.tokenize_line(new_text)
    out_words = []
    k = 0
    for w in words:
        ids = []
        for _ in w.syllables:
            ids.append(carry.get(k, ""))
            k += 1
        out_words.append(_make_word(w, ids))
    if old is not None:
        # 줄 정체성은 유지하되 사용자가 고쳤으니 검토 표시는 지운다
        return LyricLine(old.id, new_text, tuple(out_words), old.origin, "", old.asr_segment)
    return LyricLine(new_id(), new_text, tuple(out_words), origin)


def _estimate_block(times: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    if not times or n <= 0:
        return []
    a = min(t[0] for t in times)
    b = max(t[1] for t in times)
    if not (b - a > MIN_SYLLABLE_S * n):
        return []
    step = (b - a) / n
    return [(a + i * step, a + (i + 1) * step) for i in range(n)]


def apply_text(state: LyricsState, text: str, *, origin: str = "manual", language: str | None = None,
               action: str = "가사 수정") -> LyricsState:
    old_text = state.text
    new_lines = tx.split_lines(text)
    old_lines = list(old_text.lines)
    al = state.alignment
    timing_updates: dict[str, SyllableTiming | None] = {}
    out_lines: list[LyricLine] = []
    kept_ids: set[str] = set()

    sm = difflib.SequenceMatcher(None, [ln.text for ln in old_lines], new_lines, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for ln in old_lines[i1:i2]:
                out_lines.append(ln)
                kept_ids.update(s.id for s in ln.syllables)
            continue
        old_block = old_lines[i1:i2]
        old_syls = [s for ln in old_block for s in ln.syllables]
        new_tokens = [(li, t) for li, ln in enumerate(new_lines[j1:j2]) for w in tx.tokenize_line(ln) for t in w.syllables]
        carry_flat: dict[int, str] = {}
        stale: set[str] = set()
        estimate_slots: list[list[int]] = []
        smx = difflib.SequenceMatcher(None, [s.text for s in old_syls], [t.text for _, t in new_tokens], autojunk=False)
        for op2, a1, a2, b1, b2 in smx.get_opcodes():
            if op2 == "equal":
                for a, b in zip(range(a1, a2), range(b1, b2)):
                    carry_flat[b] = old_syls[a].id
            elif op2 == "replace" and a2 - a1 == b2 - b1:
                for a, b in zip(range(a1, a2), range(b1, b2)):  # 같은 자리 글자 수정: 시간은 남기고 다시 정렬 표시
                    carry_flat[b] = old_syls[a].id
                    stale.add(old_syls[a].id)
            elif op2 in ("replace", "insert"):
                times = [(al.timing(s.id).start, al.timing(s.id).end) for s in old_syls[a1:a2] if al.timing(s.id).aligned]
                slots = list(range(b1, b2))
                est = _estimate_block(times, len(slots))
                if est:
                    estimate_slots.append(slots)
                    for b, (t0, t1) in zip(slots, est):
                        carry_flat[b] = new_id()
                        timing_updates[carry_flat[b]] = SyllableTiming(t0, t1, None, "estimated", "estimated", stale=True)
        # 새 줄마다 carry를 줄 안 위치로 바꿔 만든다
        per_line: dict[int, dict[int, str]] = {}
        counters: dict[int, int] = {}
        for b, (li, _t) in enumerate(new_tokens):
            k = counters.get(li, 0)
            counters[li] = k + 1
            if b in carry_flat:
                per_line.setdefault(li, {})[k] = carry_flat[b]
        n_new = j2 - j1
        for li in range(n_new):
            # 줄 수가 같으면 자리마다 이전 줄의 정체성을 잇는다 (한 줄 고치기)
            old_line = old_block[li] if len(old_block) == n_new else None
            out_lines.append(_rebuild_line(new_lines[j1 + li], old_line, origin, per_line.get(li, {})))
        for sid in stale:
            t = al.timing(sid)
            if t.aligned and not t.user_edited:
                timing_updates[sid] = replace(t, stale=True)
        kept_ids.update(carry_flat.values())

    new_text = LyricsText(language if language is not None else (old_text.language or tx.detect_language(text)),
                          tuple(out_lines))
    live = {s.id for s in new_text.syllables}
    alignment = al.restricted_to(live).with_timings({k: v for k, v in timing_updates.items() if k in live})
    removed = len([s for s in old_text.syllables if s.id not in live])
    added = len([s for s in new_text.syllables if s.id not in old_text.order])
    out = replace(state, text=new_text, alignment=alignment)
    return out.with_history(action, f"음절 {len(new_text.syllables)}개 (추가 {added}, 삭제 {removed})")


def replace_lines(state: LyricsState, first: int, last: int, new_text: str, origin: str = "manual") -> LyricsState:
    """first~last 줄(포함)을 new_text로 바꾼다. 나머지 줄은 그대로다."""
    lines = [ln.text for ln in state.text.lines]
    if not (0 <= first <= last < len(lines)):
        raise EditError("바꿀 줄 범위가 올바르지 않습니다.")
    merged = lines[:first] + tx.split_lines(new_text) + lines[last + 1:]
    return apply_text(state, "\n".join(merged), origin=origin, action=f"{first + 1}~{last + 1}줄 교체")


def repeat_lines(state: LyricsState, line_ids: Iterable[str], times: int = 1) -> LyricsState:
    """선택한 줄들을 마지막 선택 줄 뒤에 times번 더 넣는다 (후렴 반복). 새 줄은 새 음절 id를 갖는다."""
    ids = list(dict.fromkeys(line_ids))
    if not ids or times < 1:
        raise EditError("반복할 줄을 고르세요.")
    pos = {ln.id: i for i, ln in enumerate(state.text.lines)}
    if any(i not in pos for i in ids):
        raise EditError("반복할 줄을 찾을 수 없습니다.")
    chosen = sorted(pos[i] for i in ids)
    after = chosen[-1]
    copies = []
    for _ in range(times):
        for i in chosen:
            src = state.text.lines[i]
            copies.append(make_line(src.text, "repeat"))
    lines = state.text.lines[: after + 1] + tuple(copies) + state.text.lines[after + 1:]
    out = replace(state, text=replace(state.text, lines=lines))
    return out.with_history("반복 가사 지정", f"{len(chosen)}줄 × {times}")


def set_line_review(state: LyricsState, line_ids: Iterable[str], review: str = "") -> LyricsState:
    ids = set(line_ids)
    lines = tuple(replace(ln, review=review) if ln.id in ids else ln for ln in state.text.lines)
    return replace(state, text=replace(state.text, lines=lines)).with_history(
        "검토 완료" if not review else "검토 필요 표시", f"{len(ids)}줄")


def set_language(state: LyricsState, language: str) -> LyricsState:
    return replace(state, text=replace(state.text, language=language))


# --- 음절 나누기·합치기 ---------------------------------------------------------------

def _replace_word(text: LyricsText, li: int, wi: int, word: LyricWord) -> LyricsText:
    ln = text.lines[li]
    words = ln.words[:wi] + (word,) + ln.words[wi + 1:]
    lines = text.lines[:li] + (replace(ln, words=words),) + text.lines[li + 1:]
    return replace(text, lines=lines)


def split_syllable(state: LyricsState, sid: str, pos: int, at_time: float | None = None) -> tuple[LyricsState, str]:
    """음절 텍스트를 pos 글자에서 둘로 나눈다 (예: 東京 → 東/京, beau → beau 는 불가). 새 음절 id를 돌려준다."""
    text = state.text
    if sid not in text.index:
        raise EditError("나눌 음절을 찾을 수 없습니다.")
    li, wi, si = text.index[sid]
    word = text.lines[li].words[wi]
    syl = word.syllables[si]
    if len(syl.text) < 2:
        raise EditError("한 글자 음절은 나눌 수 없습니다. 한 음절을 여러 음표로 부르면 음표 연결(멜리스마)로 표현하세요.")
    if not 0 < pos < len(syl.text):
        raise EditError("나눌 위치가 음절 안에 있어야 합니다.")
    a_text, b_text = syl.text[:pos], syl.text[pos:]
    a = replace(syl, text=a_text, reading=_reading(a_text, syl.kind), note=syl.note)
    b = LyricSyllable(new_id(), b_text, syl.kind, _reading(b_text, syl.kind), syl.note)
    word2 = replace(word, syllables=word.syllables[:si] + (a, b) + word.syllables[si + 1:])
    al = state.alignment
    t = al.timing(sid)
    updates: dict[str, SyllableTiming | None] = {}
    link_updates: dict[str, tuple[NoteLink, ...] | None] = {}
    if t.aligned:
        cut = at_time if at_time is not None else t.start + (t.end - t.start) * pos / len(syl.text)
        cut = min(max(cut, t.start + MIN_SYLLABLE_S), t.end - MIN_SYLLABLE_S)
        if t.end - t.start < 2 * MIN_SYLLABLE_S:
            raise EditError("음절이 너무 짧아 나눌 수 없습니다.")
        updates[sid] = replace(t, end=cut, end_source="user", user_edited=True)
        updates[b.id] = SyllableTiming(cut, t.end, t.confidence, "user", t.end_source, True, t.stale)
    out = replace(state, text=_replace_word(text, li, wi, word2), alignment=al.with_timings(updates).with_links(link_updates))
    return out.with_history("음절 나누기", f"{syl.text} → {a_text}/{b_text}"), b.id


def _reading(text: str, kind: str) -> str:
    toks = tx.tokenize_word(text).syllables
    return "".join(t.reading for t in toks)


def merge_syllables(state: LyricsState, ids: Iterable[str]) -> tuple[LyricsState, str]:
    """같은 줄에서 이어진 음절들을 하나로 합친다. 시간은 처음 시작~마지막 끝, 음표 연결은 합친다."""
    text = state.text
    ids = [i for i in dict.fromkeys(ids) if i in text.index]
    if len(ids) < 2:
        raise EditError("합칠 음절을 두 개 이상 고르세요.")
    order = sorted(ids, key=lambda i: text.order[i])
    lines = {text.index[i][0] for i in order}
    if len(lines) != 1:
        raise EditError("같은 줄의 음절만 합칠 수 있습니다.")
    seq = [text.order[i] for i in order]
    if seq != list(range(seq[0], seq[0] + len(seq))):
        raise EditError("이어진 음절만 합칠 수 있습니다.")
    syls = [text.syllable(i) for i in order]
    keep = syls[0]
    merged_text = "".join(s.text for s in syls)
    kind = keep.kind if all(s.kind == keep.kind for s in syls) else "mixed"
    merged = replace(keep, text=merged_text, kind=kind, reading="".join(s.reading for s in syls),
                     note=next((s.note for s in syls if s.note), ""))
    # 첫 음절 자리에 합친 음절, 나머지는 각자 단어에서 뺀다
    li = text.index[keep.id][0]
    ln = text.lines[li]
    drop = {s.id for s in syls[1:]}
    words = []
    for w in ln.words:
        new_syls = tuple(merged if s.id == keep.id else s for s in w.syllables if s.id not in drop)
        words.append(replace(w, syllables=new_syls))
    new_text = replace(text, lines=text.lines[:li] + (replace(ln, words=tuple(words)),) + text.lines[li + 1:])
    al = state.alignment
    ts = [al.timing(s.id) for s in syls if al.timing(s.id).aligned]
    updates: dict[str, SyllableTiming | None] = {s.id: None for s in syls[1:]}
    if ts:
        first, last = ts[0], ts[-1]
        confs = [t.confidence for t in ts if t.confidence is not None]
        updates[keep.id] = SyllableTiming(min(t.start for t in ts), max(t.end for t in ts),
                                          min(confs) if confs else None, first.start_source, last.end_source,
                                          any(t.user_edited for t in ts), any(t.stale for t in ts))
    links: list[NoteLink] = []
    for s in syls:
        for lk in al.links.get(s.id, ()):
            if all(x.note_id != lk.note_id for x in links):
                links.append(lk)
    link_updates = {s.id: None for s in syls[1:]}
    link_updates[keep.id] = tuple(links)
    out = replace(state, text=new_text, alignment=al.with_timings(updates).with_links(link_updates))
    return out.with_history("음절 합치기", merged_text), keep.id


# --- 시간 -----------------------------------------------------------------------

def check_time(start, end, duration: float | None = None) -> tuple[float, float]:
    """사용자·모델이 준 시간을 검사한다. 고칠 수 없는 값이면 EditError."""
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        raise EditError("음절 시간은 숫자여야 합니다.") from None
    if not (math.isfinite(start) and math.isfinite(end)):
        raise EditError("음절 시간에 NaN이나 무한대가 들어갈 수 없습니다.")
    if start < 0:
        raise EditError("음절은 0초 이전에 시작할 수 없습니다.")
    if end - start < MIN_SYLLABLE_S - 1e-9:
        raise EditError(f"음절 끝이 시작보다 {MIN_SYLLABLE_S * 1000:.0f}ms 이상 뒤여야 합니다.")
    if duration is not None and duration > 0 and start >= duration:
        raise EditError("음절 시작이 곡 길이를 넘습니다.")
    if duration is not None and duration > 0:
        end = min(end, duration)
    return start, end


def set_times(state: LyricsState, updates: Mapping[str, tuple[float, float]], *, duration: float | None = None,
              source: str = "user", action: str = "음절 시간 수정") -> LyricsState:
    al = state.alignment
    new: dict[str, SyllableTiming | None] = {}
    for sid, (s, e) in updates.items():
        if sid not in state.text.index:
            raise EditError("시간을 바꿀 음절을 찾을 수 없습니다.")
        s, e = check_time(s, e, duration)
        old = al.timing(sid)
        user = source == "user"
        new[sid] = SyllableTiming(s, e, None if user else old.confidence, source, source, user or old.user_edited, False)
    out = replace(state, alignment=al.with_timings(new))
    return out.with_history(action, f"{len(new)}개")


def shift_times(state: LyricsState, ids: Iterable[str] | None, dt: float, *, duration: float | None = None) -> LyricsState:
    """음절 시간을 dt초 옮긴다 (ids가 None이면 전체 — 시간 오프셋 보정)."""
    if not math.isfinite(dt):
        raise EditError("이동량이 올바르지 않습니다.")
    al = state.alignment
    targets = [s.id for s in state.text.syllables] if ids is None else list(ids)
    timed = [(sid, al.timing(sid)) for sid in targets if al.timing(sid).aligned]
    if not timed:
        raise EditError("옮길 정렬된 음절이 없습니다.")
    lo = min(t.start for _, t in timed)
    dt = max(dt, -lo)
    if duration:
        hi = max(t.end for _, t in timed)
        dt = min(dt, max(0.0, duration - hi))
    new = {sid: replace(t, start=t.start + dt, end=t.end + dt, user_edited=True) for sid, t in timed}
    out = replace(state, alignment=al.with_timings(new))
    return out.with_history("음절 시간 이동" if ids is not None else "가사 전체 시간 보정", f"{dt * 1000:+.0f}ms")


def clear_times(state: LyricsState, ids: Iterable[str]) -> LyricsState:
    ids = list(ids)
    out = replace(state, alignment=state.alignment.with_timings({i: None for i in ids}))
    return out.with_history("음절 시간 지우기", f"{len(ids)}개")


def timing_issues(state: LyricsState, duration: float | None = None) -> list[tuple[str, str]]:
    """잘못된 시간: (음절 id, 사유). 순서 역전·곡 길이 초과·지나친 겹침을 찾는다."""
    issues = []
    prev = None
    for s in state.text.syllables:
        t = state.alignment.timing(s.id)
        if not t.aligned:
            continue
        if duration and t.end > duration + 0.05:
            issues.append((s.id, "곡 길이를 넘습니다"))
        if prev is not None:
            pt = state.alignment.timing(prev)
            if t.start < pt.start - 1e-6:
                issues.append((s.id, "앞 음절보다 먼저 시작합니다 (순서 역전)"))
            elif pt.end - t.start > 0.5 * (t.end - t.start) + 0.05:
                issues.append((s.id, "앞 음절과 많이 겹칩니다"))
        prev = s.id
    return issues


def sanitize_alignment(state: LyricsState, duration: float | None = None) -> tuple[LyricsState, list[str]]:
    """불러온 시간에서 쓸 수 없는 값(NaN·음수·끝<시작·없는 음절)을 미정렬로 돌린다. 경고 목록을 돌려준다."""
    al = state.alignment
    live = set(state.text.index)
    bad: dict[str, None] = {}
    for sid, t in al.timings.items():
        if sid not in live:
            bad[sid] = None
            continue
        if t.start is None or t.end is None:
            continue
        try:
            check_time(t.start, t.end, duration)
        except EditError:
            bad[sid] = None
    warnings = []
    n_bad = len([k for k in bad if k in live])
    if n_bad:
        warnings.append(f"시간 값이 올바르지 않은 음절 {n_bad}개를 미정렬로 돌렸습니다.")
    orphan_links = [k for k in al.links if k not in live]
    out = replace(state, alignment=al.with_timings(bad).with_links({k: None for k in orphan_links}))
    return out, warnings
