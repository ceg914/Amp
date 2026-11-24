"""베이스 TAB → Guitar Pro 5 (.gp5) (v0.6.3). pyguitarpro(LGPL-3.0)로 쓴다.

Guitar Pro·TuxGuitar·MuseScore·alphaTab(SID-Sound 등)에서 열린다. 마디를 넘는 음은 나눠서 붙임줄로 잇고,
음과 음 사이는 쉼표로 채운다. 칠 수 없는 음(조율·최대 프렛 밖)은 쉼표가 된다.
"""

from __future__ import annotations

import os

from amp.engine.bass import Tab

# (16분음표 수, Duration 값, 점음표)
_DURS = ((16, 1, False), (12, 2, True), (8, 2, False), (6, 4, True), (4, 4, False), (3, 8, True), (2, 8, False),
         (1, 16, False))
GM_ELECTRIC_BASS_FINGER = 33  # GM 프로그램 34 (0부터 세면 33)
# 한국어 Guitar Pro 파일은 cp949로 글자를 저장한다 (SID-Sound가 만든 조각 나비 GP5도 cp949). cp949에 없는 글자는 ?로 바꾼다.
ENCODING = "cp949"


def _text(s: str) -> str:
    return str(s or "").encode(ENCODING, "replace").decode(ENCODING)[:120]


def _split_units(n: int) -> list[tuple[int, bool]]:
    out = []
    while n > 0:
        for units, value, dotted in _DURS:
            if units <= n:
                out.append((value, dotted))
                n -= units
                break
    return out


def build_song(tab: Tab, title: str = "", artist: str = ""):
    import guitarpro as gp

    opts = tab.options
    unit = 0.25  # 16분음표 = 0.25박. 격자(0.25·0.5)는 모두 이 배수다
    per_bar = int(round(tab.beats_per_bar / unit))
    song = gp.Song()
    song.title, song.artist = _text(title), _text(artist)
    song.tempo = int(round(tab.bpm))
    track = song.tracks[0]
    track.name = "Bass"
    opens = list(opts.open_midis)
    n_str = len(opens)
    track.strings = [gp.GuitarString(number=i + 1, value=m) for i, m in enumerate(reversed(opens))]  # 1 = 가장 높은 줄
    track.channel.instrument = GM_ELECTRIC_BASS_FINGER
    bars = max(1, tab.bars)
    while len(song.measureHeaders) < bars:
        song.newMeasure()
    for h in song.measureHeaders:
        h.timeSignature.numerator = tab.beats_per_bar
    # 16분음표 칸마다 (음 시작 여부, 줄, 프렛, 남은 길이)
    events: list[list] = [[] for _ in range(bars)]
    for n in tab.notes:
        if n.string is None:
            continue
        s = int(round(n.start_beat / unit))
        length = max(1, int(round(n.beats / unit)))
        first = True
        while length > 0 and s // per_bar < bars:
            b, k = divmod(s, per_bar)
            take = min(length, per_bar - k)
            events[b].append((k, take, n.string, n.fret, not first))
            s += take
            length -= take
            first = False
    for b, measure in enumerate(track.measures[:bars]):
        voice = measure.voices[0]
        voice.beats = []
        pos = 0
        for k, take, string, fret, tie in sorted(events[b]):
            if k < pos:  # 겹침 (quantize가 막지만 방어)
                continue
            for value, dotted in _split_units(k - pos):
                voice.beats.append(gp.Beat(voice, duration=gp.Duration(value=value, isDotted=dotted),
                                           status=gp.BeatStatus.rest))
            for j, (value, dotted) in enumerate(_split_units(take)):
                beat = gp.Beat(voice, duration=gp.Duration(value=value, isDotted=dotted), status=gp.BeatStatus.normal)
                ntype = gp.NoteType.tie if (tie or j > 0) else gp.NoteType.normal
                beat.notes.append(gp.Note(beat, value=fret, string=n_str - string, type=ntype))
                voice.beats.append(beat)
            pos = k + take
        for value, dotted in _split_units(per_bar - pos):
            voice.beats.append(gp.Beat(voice, duration=gp.Duration(value=value, isDotted=dotted),
                                       status=gp.BeatStatus.rest))
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
