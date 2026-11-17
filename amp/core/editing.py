"""음표 편집 연산 (Qt 비의존 순수 함수).

모든 함수는 새 Note 객체를 돌려주고 입력을 바꾸지 않는다.
실행 취소 명령은 이 결과를 ``Score.replace_notes`` 로 반영한다.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from amp.core.errors import EditError
from amp.core.ids import new_id
from amp.core.model import Note
from amp.core.pitch import MIDI_MAX, MIDI_MIN

MIN_NOTE_DURATION = 0.03  # 초. 이보다 짧은 음표는 만들지 않는다.


def create_note(start: float, end: float, pitch: int, *, note_id: str | None = None) -> Note:
    if start < 0:
        raise EditError("음표는 0초 이전에 시작할 수 없습니다.")
    if end - start < MIN_NOTE_DURATION - 1e-9:
        raise EditError(f"음표 길이는 최소 {MIN_NOTE_DURATION * 1000:.0f}ms 이상이어야 합니다.")
    return Note(id=note_id or new_id(), start=start, end=end, pitch=int(pitch))


def clamp_move(notes: Sequence[Note], dt: float, dpitch: int) -> tuple[float, int]:
    """모든 음표가 0초 이후, MIDI 범위 안에 머물도록 이동량을 줄인다."""
    if not notes:
        return 0.0, 0
    min_start = min(n.start for n in notes)
    lo = min(n.pitch for n in notes)
    hi = max(n.pitch for n in notes)
    dt = max(dt, -min_start)
    dpitch = max(MIDI_MIN - lo, min(MIDI_MAX - hi, int(dpitch)))
    return dt, dpitch


def move_notes(notes: Sequence[Note], dt: float, dpitch: int) -> list[Note]:
    dt, dpitch = clamp_move(notes, dt, dpitch)
    return [n.with_(start=n.start + dt, end=n.end + dt, pitch=n.pitch + dpitch) for n in notes]


def clamp_resize(notes: Sequence[Note], edge: str, delta: float) -> float:
    """길이 조절량을 최소 길이·0초 제한에 맞게 줄인다."""
    if not notes:
        return 0.0
    if edge == "end":
        # 끝을 당길 때 가장 짧은 음표가 최소 길이를 지키도록
        limit = max(MIN_NOTE_DURATION - n.duration for n in notes)
        return max(delta, limit)
    if edge == "start":
        limit_hi = min(n.duration - MIN_NOTE_DURATION for n in notes)
        limit_lo = -min(n.start for n in notes)
        return min(max(delta, limit_lo), limit_hi)
    raise ValueError(f"edge는 'start' 또는 'end'여야 합니다: {edge}")


def resize_notes(notes: Sequence[Note], edge: str, delta: float) -> list[Note]:
    delta = clamp_resize(notes, edge, delta)
    if edge == "end":
        return [n.with_(end=n.end + delta) for n in notes]
    return [n.with_(start=n.start + delta) for n in notes]


def split_note(note: Note, at: float, *, ids: tuple[str, str] | None = None) -> tuple[Note, Note]:
    """``at`` 시점에서 음표를 둘로 나눈다.

    두 조각은 같은 음절(syllable_id)과 출처(origin_ids)를 물려받는다.
    같은 음절에 음표가 둘 이상이 되므로 멜리스마로 해석된다.
    """
    if not (note.start + MIN_NOTE_DURATION - 1e-9 <= at <= note.end - MIN_NOTE_DURATION + 1e-9):
        raise EditError("분할 위치가 음표 경계에 너무 가깝거나 음표 밖에 있습니다.")
    left_id, right_id = ids or (new_id(), new_id())
    left = note.with_(id=left_id, end=at)
    right = note.with_(id=right_id, start=at)
    return left, right


def merge_notes(notes: Sequence[Note], *, note_id: str | None = None) -> Note:
    """여러 음표를 하나로 합친다.

    - 범위: 가장 이른 시작 ~ 가장 늦은 끝
    - 음높이: 합친 길이가 가장 긴 음높이 (같으면 먼저 나온 음)
    - 음절: 시간 순으로 처음 연결된 음절. 다른 음절 연결은 끊긴다.
    - 출처: 모든 출처를 시간 순으로 합침
    """
    if len(notes) < 2:
        raise EditError("병합하려면 음표를 두 개 이상 선택해야 합니다.")
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    weight: dict[int, float] = defaultdict(float)
    first_seen: dict[int, int] = {}
    for i, n in enumerate(ordered):
        weight[n.pitch] += n.duration
        first_seen.setdefault(n.pitch, i)
    pitch = max(weight, key=lambda p: (weight[p], -first_seen[p]))

    origins: list[str] = []
    for n in ordered:
        for o in n.origin_ids:
            if o not in origins:
                origins.append(o)
    confs = [n.confidence for n in ordered]
    return Note(
        id=note_id or new_id(),
        start=ordered[0].start,
        end=max(n.end for n in ordered),
        pitch=pitch,
        velocity=max(n.velocity for n in ordered),
        confidence=min(confs) if all(c is not None for c in confs) else None,
        vibrato=None,
        syllable_id=next((n.syllable_id for n in ordered if n.syllable_id), None),
        origin_ids=tuple(origins),
    )
