"""음절 ↔ 음표 연결.

관계
- 1:1  음절 하나 ↔ 음표 하나
- 1:N  음절 하나 ↔ 음표 여러 개 (멜리스마: '아' → C4, D4, E4)
- N:1  음절 여러 개 ↔ 음표 하나 (음표 검출이 같은 높이로 다시 부른 음절을 하나로 합친 경우)

자동 연결은 시간이 겹친다는 사실만으로 확정하지 않는다. 연결마다 신뢰도(``confidence``)를 매기고,
사용자가 확정한 연결(``confirmed``)이나 직접 만든 연결(source user/sequential)은 자동 재계산이 바꾸지 않는다.

음표를 나누거나 합치거나 지우면(``reconcile``) 음절 텍스트·시간은 그대로 두고 연결만 다시 잇는다.

접근음 (STEP 3.5)
- 음절이 짧고 낮은 음표로 시작해 바로 위 음표로 올라가면(끌어올림이 따로 음표로 잡힌 경우), 그 짧은 음표는
  같은 음절의 '접근음'(role="approach")으로 잇고 가사 글자는 뒤의 주 음표에 붙인다. 음표는 지우지 않는다.
- 조건: 짧은 음표가 음절 시작(±ONSET_TOL)에서 시작, 길이 APPROACH_MAX_S 이하, 바로 뒤(틈 APPROACH_GAP_S 이하)에
  1~7반음 위이고 더 길거나 같은 음표가 같은 음절에 이어진다, 장식음으로 표시된 음표가 아니다.
- 연결 시간·음높이만 본다(피치 곡선은 보지 않는다). 음표 검출이 끌어올림을 이미 합쳤으면 이 경우가 생기지 않는다.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import replace
from typing import Iterable, Sequence

from amp.core.errors import EditError
from amp.core.model import Note
from amp.lyrics.model import LyricsState, NoteLink, SyllableTiming

ONSET_TOL = 0.12  # 음표·음절 시작이 이만큼 가까우면 같은 발성으로 본다
PROTECTED_SOURCES = ("user", "sequential")
# 접근음 판정 (amp.engine.notes의 접근음 흡수와 같은 길이·간격·음정 기준)
APPROACH_MAX_S = 0.25
APPROACH_GAP_S = 0.08
APPROACH_MAX_INTERVAL = 7


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _timing_quality(t: SyllableTiming) -> float:
    """음절 시간 자체를 얼마나 믿을지. 추정값으로 맺은 연결은 신뢰도를 낮춘다."""
    if t.user_edited or t.start_source in ("user", "notes"):
        q = 0.95
    elif t.start_source == "model":
        q = 0.4 + 0.6 * (t.confidence if t.confidence is not None else 0.5)
    elif t.start_source == "asr_word":
        q = 0.5
    else:
        q = 0.35
    return q * (0.6 if t.stale else 1.0)


def _protected(links: Iterable[NoteLink]) -> bool:
    return any(lk.confirmed or lk.source in PROTECTED_SOURCES for lk in links)


def auto_link(state: LyricsState, notes: Sequence[Note], *, only: Iterable[str] | None = None) -> LyricsState:
    """정렬된 음절과 음표를 시간으로 잇는다. 사용자가 만든·확정한 연결은 그대로 둔다."""
    al = state.alignment
    scope = set(only) if only is not None else None
    syls = [s.id for s in state.text.syllables if scope is None or s.id in scope]
    kept = {sid: al.links[sid] for sid in syls if sid in al.links and _protected(al.links[sid])}
    claimed = {lk.note_id for links in kept.values() for lk in links}
    timed = [(sid, al.timing(sid)) for sid in syls if sid not in kept and al.timing(sid).aligned]
    timed.sort(key=lambda x: x[1].start)
    starts = [t.start for _, t in timed]
    notes = sorted((n for n in notes if n.id not in claimed), key=lambda n: n.start)

    owners: dict[str, list[tuple[Note, float]]] = {}
    for n in notes:
        lo = bisect.bisect_left(starts, n.start - 2.0)
        hi = bisect.bisect_right(starts, n.end + ONSET_TOL)
        best = None
        for sid, t in timed[lo:hi]:
            ov = _overlap(n.start, n.end, t.start, t.end)
            d_on = abs(t.start - n.start)
            if ov <= 0 and d_on > ONSET_TOL:
                continue
            cover = ov / max(n.duration, 1e-6)
            score = cover - 0.5 * min(1.0, d_on / 0.3)
            if best is None or score > best[0]:
                best = (score, sid, t, cover, d_on)
        if best is None:
            continue
        _score, sid, t, cover, d_on = best
        if cover < 0.3 and d_on > ONSET_TOL:
            continue
        # 시작이 맞는 음표(첫 음)는 시작 차이로, 음절 안에 들어 있는 뒤쪽 음표(멜리스마)는 덮인 비율로 매긴다
        onset_term = math.exp(-max(0.0, d_on - 0.03) / 0.2) * min(1.0, 0.5 + cover)
        inside_term = 0.75 * cover
        conf = max(onset_term, inside_term) * _timing_quality(t)
        owners.setdefault(sid, []).append((n, round(conf, 3)))

    # N:1 — 음표를 하나도 얻지 못한 음절이 어떤 음표 안에서 시작하면 그 음표에 함께 잇는다
    note_starts = [n.start for n in notes]
    for sid, t in timed:
        if sid in owners:
            continue
        i = bisect.bisect_right(note_starts, t.start + ONSET_TOL) - 1
        cands = [n for n in notes[max(0, i - 3): i + 1] if n.start - ONSET_TOL <= t.start < n.end - 0.03]
        if not cands:
            continue
        n = max(cands, key=lambda n: _overlap(n.start, n.end, t.start, t.end))
        ov = _overlap(n.start, n.end, t.start, t.end) / max(t.end - t.start, 1e-6)
        owners[sid] = [(n, round(0.6 * min(1.0, ov) * _timing_quality(t), 3))]

    timing_of = dict(timed)
    updates: dict[str, tuple[NoteLink, ...] | None] = {}
    for sid in syls:
        if sid in kept:
            continue
        pairs = sorted(owners.get(sid, []), key=lambda p: p[0].start)
        approach = _approach_note(sid, [n for n, _c in pairs], timing_of.get(sid))
        updates[sid] = tuple(NoteLink(n.id, conf, "auto", False, "approach" if n.id == approach else "")
                             for n, conf in pairs)
    out = replace(state, alignment=al.with_links(updates))
    n_links = sum(len(v or ()) for v in updates.values())
    return out.with_history("음표 자동 연결", f"연결 {n_links}개, 사용자 연결 유지 {len(kept)}음절")


def _approach_note(sid: str, notes: Sequence[Note], t: SyllableTiming | None) -> str | None:
    """음절의 첫 음표가 접근음이면 그 id. 규칙은 모듈 설명 참고."""
    if len(notes) < 2 or t is None or t.start is None:
        return None
    a, b = notes[0], notes[1]
    if getattr(a, "ornament", False):
        return None
    if abs(a.start - t.start) > ONSET_TOL or a.duration > APPROACH_MAX_S or b.duration < a.duration:
        return None
    if not (-1e-6 <= b.start - a.end <= APPROACH_GAP_S):
        return None
    if not (1 <= b.pitch - a.pitch <= APPROACH_MAX_INTERVAL):
        return None
    return a.id


def main_note_id(links: Sequence[NoteLink], notes_by_id: dict[str, Note]) -> str | None:
    """가사 글자를 붙일 주 음표: 시간순으로 접근음이 아닌 첫 음표 (모두 접근음이면 첫 음표)."""
    ns = sorted(((notes_by_id[lk.note_id], lk) for lk in links if lk.note_id in notes_by_id),
                key=lambda x: (x[0].start, x[0].id))
    for n, lk in ns:
        if lk.role != "approach":
            return n.id
    return ns[0][0].id if ns else None


def set_approach(state: LyricsState, sid: str, note_id: str, approach: bool) -> LyricsState:
    """사용자가 연결 음표 하나를 접근음으로 표시하거나 해제한다. 그 음절의 연결은 확정된다."""
    links = state.alignment.links.get(sid, ())
    if all(lk.note_id != note_id for lk in links):
        raise EditError("이 음절에 연결된 음표가 아닙니다.")
    if approach and len(links) < 2:
        raise EditError("음표가 하나뿐인 연결에서는 접근음을 지정할 수 없습니다.")
    new = tuple(replace(lk, role=("approach" if approach else "") if lk.note_id == note_id else lk.role, confirmed=True)
                for lk in links)
    if all(lk.role == "approach" for lk in new):
        raise EditError("주 음표가 하나는 있어야 합니다.")
    out = replace(state, alignment=state.alignment.with_links({sid: new}))
    return out.with_history("접근음 표시" if approach else "접근음 표시 해제", "음표 1개")


def link_sequential(state: LyricsState, syllable_ids: Sequence[str], notes: Sequence[Note], *,
                    set_times: bool = True) -> LyricsState:
    """고른 음절을 고른 음표에 순서대로 하나씩 잇는다 (모델 없이 손으로 맞출 때). 남는 쪽은 그대로 둔다."""
    order = state.text.order
    sids = sorted((s for s in dict.fromkeys(syllable_ids) if s in order), key=order.get)
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    if not sids or not notes:
        raise EditError("순서대로 이으려면 음절과 음표를 모두 골라야 합니다.")
    link_updates = {}
    time_updates = {}
    for sid, n in zip(sids, notes):
        link_updates[sid] = (NoteLink(n.id, None, "sequential", False),)
        if set_times:
            time_updates[sid] = SyllableTiming(n.start, n.end, None, "notes", "notes", False)
    al = state.alignment.with_links(link_updates).with_timings(time_updates)
    out = replace(state, alignment=al)
    return out.with_history("순서대로 연결", f"음절 {len(sids)}개 · 음표 {len(notes)}개 중 {len(link_updates)}쌍")


def set_links(state: LyricsState, sid: str, note_ids: Sequence[str], *, mode: str = "replace") -> LyricsState:
    """사용자 연결 편집. mode: replace(바꾸기) / add(더하기) / remove(빼기). 결과는 확정 연결이다."""
    if sid not in state.text.index:
        raise EditError("연결할 음절을 찾을 수 없습니다.")
    current = list(state.alignment.links.get(sid, ()))
    ids = list(dict.fromkeys(note_ids))
    if mode == "replace":
        new = [NoteLink(i, None, "user", True) for i in ids]
    elif mode == "add":
        have = {lk.note_id for lk in current}
        new = current + [NoteLink(i, None, "user", True) for i in ids if i not in have]
    elif mode == "remove":
        new = [lk for lk in current if lk.note_id not in set(ids)]
    else:
        raise ValueError(mode)
    out = replace(state, alignment=state.alignment.with_links({sid: new}))
    return out.with_history({"replace": "음표 연결 바꾸기", "add": "음표 연결 더하기", "remove": "음표 연결 빼기"}[mode],
                            f"음표 {len(ids)}개")


def confirm_links(state: LyricsState, sids: Iterable[str], confirmed: bool = True) -> LyricsState:
    al = state.alignment
    updates = {sid: tuple(replace(lk, confirmed=confirmed) for lk in al.links.get(sid, ())) for sid in sids}
    out = replace(state, alignment=al.with_links(updates))
    return out.with_history("연결 확정" if confirmed else "연결 확정 해제", f"{len(updates)}음절")


def clear_links(state: LyricsState, sids: Iterable[str]) -> LyricsState:
    sids = list(sids)
    out = replace(state, alignment=state.alignment.with_links({s: None for s in sids}))
    return out.with_history("음표 연결 지우기", f"{len(sids)}음절")


def times_from_links(state: LyricsState, sids: Iterable[str], notes_by_id: dict[str, Note]) -> LyricsState:
    """연결한 음표의 첫 시작~마지막 끝을 음절 시간으로 쓴다."""
    updates = {}
    for sid in sids:
        ns = [notes_by_id[lk.note_id] for lk in state.alignment.links.get(sid, ()) if lk.note_id in notes_by_id]
        if ns:
            updates[sid] = SyllableTiming(min(n.start for n in ns), max(n.end for n in ns), None, "notes", "notes")
    if not updates:
        raise EditError("시간을 가져올 연결 음표가 없습니다.")
    out = replace(state, alignment=state.alignment.with_timings(updates))
    return out.with_history("음표 시간 가져오기", f"{len(updates)}음절")


def reconcile(state: LyricsState, removed: Sequence[Note], added: Sequence[Note]) -> LyricsState:
    """음표 교체(나누기·합치기·지우기·재분석) 뒤 연결을 다시 잇는다. 음절 텍스트·시간은 바꾸지 않는다.

    사라진 음표에 걸린 연결은 그 음표와 시간이 겹치는 새 음표로 옮긴다. 음절 시간이 있으면 음절과도 겹치는
    음표만 고른다(나눈 음표의 앞 조각 '사', 뒤 조각 '랑'). 옮길 곳이 없으면 연결을 지우고 음절은 '연결 없음'이 된다.
    """
    al = state.alignment
    if not al.links:
        return state
    removed_map = {n.id: n for n in removed}
    added_ids = {n.id for n in added}
    gone = {nid for nid in removed_map if nid not in added_ids}
    if not gone:
        return state
    affected = {sid: links for sid, links in al.links.items() if any(lk.note_id in gone for lk in links)}
    if not affected:
        return state
    added_sorted = sorted(added, key=lambda n: n.start)
    updates: dict[str, tuple[NoteLink, ...]] = {}
    for sid, links in affected.items():
        t = al.timing(sid)
        out: list[NoteLink] = []
        for lk in links:
            if lk.note_id not in gone:
                if all(x.note_id != lk.note_id for x in out):
                    out.append(lk)
                continue
            old = removed_map[lk.note_id]
            cands = [a for a in added_sorted if _overlap(a.start, a.end, old.start, old.end) > 1e-6]
            if t.aligned and not t.stale and len(cands) > 1:
                inside = [a for a in cands if _overlap(a.start, a.end, t.start, t.end) > 1e-6]
                if inside:
                    cands = inside
            src = lk.source if lk.source in PROTECTED_SOURCES else "remapped"
            for a in cands:  # 바뀐 음표의 역할(접근음)은 새 음표에 넘기지 않는다
                if all(x.note_id != a.id for x in out):
                    out.append(NoteLink(a.id, lk.confidence, src, lk.confirmed))
        if out and all(x.role == "approach" for x in out):
            out = [replace(x, role="") for x in out]
        updates[sid] = tuple(out)
    return replace(state, alignment=al.with_links(updates))


def prune(state: LyricsState, note_ids: Iterable[str]) -> tuple[LyricsState, int]:
    """없는 음표를 가리키는 연결을 지운다 (파일을 따로 고친 경우 등). 지운 연결 수를 돌려준다."""
    present = set(note_ids)
    al = state.alignment
    updates = {}
    dropped = 0
    for sid, links in al.links.items():
        keep = tuple(lk for lk in links if lk.note_id in present)
        if len(keep) != len(links):
            dropped += len(links) - len(keep)
            updates[sid] = keep
    if not updates:
        return state, 0
    return replace(state, alignment=al.with_links(updates)), dropped


def relation_kind(state: LyricsState, sid: str) -> str:
    """'none' / '1:1' / '1:N' (멜리스마) / 'N:1' (여러 음절이 한 음표) / 'N:M'."""
    links = state.alignment.links.get(sid, ())
    if not links:
        return "none"
    shared = any(len(state.alignment.note_to_syllables.get(lk.note_id, ())) > 1 for lk in links)
    if len(links) == 1:
        return "N:1" if shared else "1:1"
    if shared:
        return "N:M"
    return "1:1" if sum(1 for lk in links if lk.role != "approach") == 1 else "1:N"
