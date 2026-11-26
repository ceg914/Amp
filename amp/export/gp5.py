"""베이스 TAB → Guitar Pro 5 (.gp5) (v0.6.3). pyguitarpro(LGPL-3.0)로 쓴다.

Guitar Pro·TuxGuitar·MuseScore·alphaTab(SID-Sound 등)에서 열린다. 마디를 넘는 음은 나눠서 붙임줄로 잇고,
음과 음 사이는 쉼표로 채운다. 칠 수 없는 음(조율·최대 프렛 밖)은 쉼표가 된다.
"""

from __future__ import annotations

import os

from amp.engine.bass import Tab

from amp.engine.bass import TPB

# (눈금 수(박당 24), Duration 값, 점음표) — 곧은 음표
_STRAIGHT = ((96, 1, False), (72, 2, True), (48, 2, False), (36, 4, True), (24, 4, False), (18, 8, True), (12, 8, False),
             (9, 16, True), (6, 16, False), (3, 32, False))
# 셋잇단 (3:2): 4분 셋잇단 16, 8분 셋잇단 8, 16분 셋잇단 4, 32분 셋잇단 2
_TRIPLET = ((16, 4), (8, 8), (4, 16), (2, 32))
GM_ELECTRIC_BASS_FINGER = 33  # GM 프로그램 34 (0부터 세면 33)
GP_TICKS_PER_TICK = 960 // TPB
# 한국어 Guitar Pro 파일은 cp949로 글자를 저장한다 (SID-Sound가 만든 조각 나비 GP5도 cp949). cp949에 없는 글자는 ?로 바꾼다.
ENCODING = "cp949"


class Gp5Error(ValueError):
    pass


def _text(s: str) -> str:
    return str(s or "").encode(ENCODING, "replace").decode(ENCODING)[:120]


def _straight(n: int) -> list[tuple[int, bool, bool]]:
    out = []
    while n > 0:
        for units, value, dotted in _STRAIGHT:
            if units <= n:
                out.append((value, dotted, False))
                n -= units
                break
        else:
            raise Gp5Error(f"곧은 음표로 나타낼 수 없는 길이입니다 ({n}/{TPB}박)")
    return out


def _triplet(n: int) -> list[tuple[int, bool, bool]]:
    out = []
    while n > 0:
        for units, value in _TRIPLET:
            if units <= n:
                out.append((value, False, True))
                n -= units
                break
        else:
            raise Gp5Error(f"셋잇단으로 나타낼 수 없는 길이입니다 ({n}/{TPB}박)")
    return out


def _durations(start: int, length: int, triplet_beats: set[int]) -> list[tuple[int, bool, bool]]:
    """TAB 눈금 [start, start+length) → Guitar Pro 음표 길이들 (값, 점, 셋잇단). 셋잇단 박이 있으면 박 경계에서 나눈다."""
    if not triplet_beats:
        return _straight(length)
    out = []
    t, end = start, start + length
    while t < end:
        beat = t // TPB
        cut = min(end, (beat + 1) * TPB)
        out += _triplet(cut - t) if beat in triplet_beats else _straight(cut - t)
        t = cut
    return out


def _duration(gp, value: int, dotted: bool, triplet: bool):
    return gp.Duration(value=value, isDotted=dotted, tuplet=gp.Tuplet(3, 2) if triplet else gp.Tuplet(1, 1))


def build_song(tab: Tab, title: str = "", artist: str = ""):
    """v0.7.0: 박당 24눈금으로 쓴다 — 8·16·32분음표와 셋잇단(3:2), 마디마다 다른 박자(박자표 변화·못갖춘마디),
    템포 변화(믹스 테이블). 마디를 넘는 음은 나눠서 붙임줄로 잇고 음과 음 사이는 쉼표로 채운다. 칠 수 없는 음은 쉼표."""
    import guitarpro as gp

    opts = tab.options
    song = gp.Song()
    song.title, song.artist = _text(title), _text(artist)
    song.tempo = int(round(tab.bpm))
    track = song.tracks[0]
    track.name = "Bass"
    opens = list(opts.open_midis)
    n_str = len(opens)
    track.strings = [gp.GuitarString(number=i + 1, value=m) for i, m in enumerate(reversed(opens))]  # 1 = 가장 높은 줄
    track.channel.instrument = GM_ELECTRIC_BASS_FINGER
    lengths = tab._lengths() or [tab.beats_per_bar]  # noqa: SLF001
    bars = len(lengths)
    starts = [sum(lengths[:k]) * TPB for k in range(bars)]
    while len(song.measureHeaders) < bars:
        song.newMeasure()
    for h, n in zip(song.measureHeaders, lengths):
        h.timeSignature.numerator = int(n)
    trip = {int(b) for b in tab.triplet_beats}
    events: list[list] = [[] for _ in range(bars)]  # (마디 안 시작 눈금, 길이 눈금, 줄, 프렛, 붙임줄)
    for n in tab.notes:
        if n.string is None:
            continue
        s = int(round(n.start_beat * TPB))
        length = max(1, int(round(n.beats * TPB)))
        first = True
        while length > 0:
            k = max(0, next((i for i, s0 in enumerate(starts) if s0 > s), bars) - 1)
            if k >= bars or s >= starts[-1] + lengths[-1] * TPB:
                break
            bar_end = starts[k] + lengths[k] * TPB
            take = min(length, bar_end - s)
            events[k].append((s - starts[k], take, n.string, n.fret, not first))
            s += take
            length -= take
            first = False
    beat_at: list[list[tuple[int, object]]] = [[] for _ in range(bars)]  # 마디마다 (시작 눈금, GP 박)
    for b, measure in enumerate(track.measures[:bars]):
        voice = measure.voices[0]
        voice.beats = []
        pos = 0
        base = starts[b]

        def add(kind, at, ln, string=None, fret=None, tie=False):
            t = at
            for j, (value, dotted, tri) in enumerate(_durations(base + at, ln, trip)):
                status = gp.BeatStatus.rest if kind == "rest" else gp.BeatStatus.normal
                beat = gp.Beat(voice, duration=_duration(gp, value, dotted, tri), status=status)
                if kind == "note":
                    ntype = gp.NoteType.tie if (tie or j > 0) else gp.NoteType.normal
                    beat.notes.append(gp.Note(beat, value=fret, string=n_str - string, type=ntype))
                voice.beats.append(beat)
                beat_at[b].append((t, beat))
                t += beat.duration.time // GP_TICKS_PER_TICK

        for k, take, string, fret, tie in sorted(events[b]):
            if k < pos:  # 겹침 (quantize가 막지만 방어)
                continue
            if k > pos:
                add("rest", pos, k - pos)
            add("note", k, take, string, fret, tie)
            pos = k + take
        if pos < lengths[b] * TPB:
            add("rest", pos, lengths[b] * TPB - pos)
    for tb, bpm in tab.tempo_points[1:]:  # 템포 변화: 그 자리(또는 바로 뒤)에서 시작하는 박에 믹스 테이블
        t = int(round(tb * TPB))
        k = max(0, next((i for i, s0 in enumerate(starts) if s0 > t), bars) - 1)
        target = next((bt for at, bt in beat_at[k] if starts[k] + at >= t), None)
        if target is None and k + 1 < bars and beat_at[k + 1]:
            target = beat_at[k + 1][0][1]
        if target is not None:
            target.effect.mixTableChange = gp.MixTableChange(tempo=gp.MixTableItem(value=int(round(bpm)), duration=0),
                                                             hideTempo=False)
    return song


def write_gp5(tab: Tab, path: str | os.PathLike, title: str = "", artist: str = "") -> None:
    import guitarpro as gp

    from amp.core.fileio import atomic_write_bytes

    song = build_song(tab, title, artist)
    tmp = f"{path}.tmp-gp5"
    try:
        gp.write(song, tmp, version=(5, 1, 0), encoding=ENCODING)
        with open(tmp, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    atomic_write_bytes(path, data)


def read_gp5(path: str | os.PathLike) -> list[dict]:
    """시험·검증용: GP5의 첫 트랙 음표 (박 위치, 길이, 줄(0=가장 낮은 줄), 프렛, MIDI, 붙임줄)."""
    import guitarpro as gp

    song = gp.parse(str(path), encoding=ENCODING)
    tr = song.tracks[0]
    n_str = len(tr.strings)
    tuning = {s.number: s.value for s in tr.strings}
    out, bar_start = [], 0.0
    for mh, meas in zip(song.measureHeaders, tr.measures):
        pos = 0.0
        for beat in meas.voices[0].beats:
            dur = beat.duration.time / 960.0
            for n in beat.notes:
                out.append({"beat": round(bar_start + pos, 4), "beats": round(dur, 4), "string": n_str - n.string,
                            "fret": n.value, "midi": tuning[n.string] + n.value, "tie": n.type == gp.NoteType.tie})
            pos += dur
        bar_start += mh.timeSignature.numerator * 4.0 / mh.timeSignature.denominator.value
    return out
