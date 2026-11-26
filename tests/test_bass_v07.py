"""v0.7.0 Feature I: 베이스 박자 처리(템포 맵·박자표 변화·못갖춘마디·32분·셋잇단·붙임줄·실제 시간 보관)와 운지(사용자 지정 줄)."""

from __future__ import annotations

import pytest

from amp.core.model import Note
from amp.core.tempo import TempoMap, TempoPoint
from amp.engine.bass import (BassError, BassOptions, BassState, assign_fingering, format_meter, parse_meter, quantize,
                             text_tab)


def _beats(tab):
    return [(n.note_id, n.start_beat, n.beats) for n in tab.notes]


def test_tempo_map_changes_are_counted_and_written(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from amp.export.gp5 import read_gp5, write_gp5

    t = TempoMap((TempoPoint(0.0, 120.0), TempoPoint(2.0, 60.0)))  # 4박 뒤(2초)부터 60BPM
    notes = [Note("a", 0.0, 0.5, 33), Note("b", 2.0, 3.0, 35), Note("c", 3.0, 4.0, 36)]
    tab = quantize(notes, t, BassOptions())
    assert _beats(tab) == [("a", 0.0, 1.0), ("b", 4.0, 1.0), ("c", 5.0, 1.0)]
    assert tab.tempo_points == [(0.0, 120.0), (4.0, 60.0)]
    assert "템포 바뀜 1곳" in text_tab(tab) and "2마디 1.00박부터 ♩=60.0" in text_tab(tab)
    path = tmp_path / "t.gp5"
    write_gp5(tab, path)
    song = gp.parse(str(path), encoding="cp949")
    changes = [(mi, b.effect.mixTableChange.tempo.value) for mi, m in enumerate(song.tracks[0].measures)
               for b in m.voices[0].beats if b.effect.mixTableChange and b.effect.mixTableChange.tempo]
    assert song.tempo == 120 and changes == [(1, 60)]
    assert [g["beat"] for g in read_gp5(path) if not g["tie"]] == [0.0, 4.0, 5.0]


def test_meter_changes_and_partial_pickup(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from amp.export.gp5 import write_gp5

    t = TempoMap.constant(60.0, 1.0)  # 1박 = 1초, 첫 박 1초
    notes = [Note("p", 0.0, 0.5, 33), Note("a", 1.0, 2.0, 33), Note("b", 6.0, 7.0, 35), Note("c", 9.0, 10.0, 36)]
    o = BassOptions(meter=((2, 3),), pickup="partial")
    tab = quantize(notes, t, o)
    assert tab.pickup_beats == 1 and tab.pickup_bars == 0
    assert tab.bar_lengths[:4] == [1, 4, 3, 3]  # 못갖춘마디 1박, 1마디 4박, 2마디부터 3박
    assert _beats(tab)[:2] == [("p", 0.0, 0.5), ("a", 1.0, 1.0)]
    txt = text_tab(tab)
    assert "1박 못갖춘마디" in txt and "박자 1/4·3/4·4/4" in txt
    e_row = next(line for line in txt.splitlines() if line.startswith("E|"))
    assert [len(x) for x in e_row.split("|")[1:-1]] == [1 * 4 * 3, 4 * 4 * 3, 3 * 4 * 3, 3 * 4 * 3]
    path = tmp_path / "m.gp5"
    write_gp5(tab, path)
    song = gp.parse(str(path), encoding="cp949")
    nums = [mh.timeSignature.numerator for mh in song.measureHeaders]
    assert nums[:4] == [1, 4, 3, 3]
    for mh, meas in zip(song.measureHeaders, song.tracks[0].measures):
        assert sum(b.duration.time for b in meas.voices[0].beats) == 960 * mh.timeSignature.numerator
    full = quantize(notes, t, BassOptions())  # 이전 방식: 온마디를 붙임
    assert full.pickup_bars == 1 and full.notes[0].start_beat == 3.0


def test_meter_text_parse():
    assert parse_meter("9:3, 17/4") == ((9, 3), (17, 4)) and parse_meter("") == ()
    assert format_meter(((9, 3), (17, 4))) == "9:3, 17:4"
    for bad in ("x", "0:4", "3:0", "3:40"):
        with pytest.raises(BassError):
            parse_meter(bad)
    o = BassOptions(meter=[[9, 3]], tuplets="auto", pickup="partial", riff=True)
    assert BassOptions.from_dict(o.to_dict()) == o and o.meter == ((9, 3),)
    assert not BassOptions().riff and BassOptions().tuplets == "off"  # 실험·새 기능은 기본으로 끔
    with pytest.raises(BassError):
        BassOptions(grid=1 / 3)  # 셋잇단은 격자가 아니라 tuplets로


def test_same_slot_note_is_moved_not_dropped_and_raw_times_kept():
    t = TempoMap.constant(120.0, 0.0)  # 1박 0.5초, 16분 = 0.125초
    notes = [Note("a", 0.0, 0.1, 33), Note("b", 0.05, 0.25, 35), Note("c", 0.5, 0.6, 36), Note("d", 0.51, 0.7, 36)]
    tab = quantize(notes, t, BassOptions())
    got = {n.note_id: n for n in tab.notes}
    assert set(got) == {"a", "b", "c"} and tab.dropped[0][0] == "d"  # d는 c와 10ms(0.02박): 같은 소리로 보고 기록
    assert got["b"].start_beat == 0.125 and "moved" in got["b"].flags and tab.moved == 1  # 16분 → 32분 자리로
    assert got["a"].beats == 0.125  # 앞 음은 뒤 음 시작에서 자름
    assert got["b"].start_s == 0.05 and got["b"].end_s == 0.25 and got["b"].raw_start_beat == 0.1  # 실제 시간 보관
    txt = text_tab(tab)
    assert "32분음표" in txt and "잘게 옮겼습니다" in txt and "뺀 음 1개" in txt


def test_32nd_grid_and_auto_triplets(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from amp.export.gp5 import read_gp5, write_gp5

    t = TempoMap.constant(60.0, 0.0)  # 1박 = 1초
    n32 = quantize([Note("a", 0.0, 0.125, 33), Note("b", 0.125, 0.25, 35)], t, BassOptions(grid=0.125))
    assert _beats(n32) == [("a", 0.0, 0.125), ("b", 0.125, 0.125)]
    trip = [Note("x", 0.0, 0.33, 33), Note("y", 0.333, 0.66, 35), Note("z", 0.667, 1.0, 36), Note("w", 1.0, 2.0, 38)]
    off = quantize(trip, t, BassOptions())
    assert [n.start_beat for n in off.notes] == [0.0, 0.25, 0.75, 1.0]  # 셋잇단을 끄면 16분에 맞춤
    on = quantize(trip, t, BassOptions(tuplets="auto"))
    assert [round(n.start_beat, 4) for n in on.notes] == [0.0, 0.3333, 0.6667, 1.0] and on.triplet_beats == [0]
    assert "triplet" in on.notes[0].flags and "triplet" not in on.notes[3].flags
    even = quantize([Note("e", 0.0, 0.25, 33), Note("f", 0.25, 0.5, 35), Note("g", 0.5, 1.0, 36)], t,
                    BassOptions(tuplets="auto"))
    assert even.triplet_beats == []  # 곧은 리듬은 셋잇단으로 바꾸지 않는다
    path = tmp_path / "t.gp5"
    write_gp5(on, path)
    song = gp.parse(str(path), encoding="cp949")
    beats = song.tracks[0].measures[0].voices[0].beats
    assert [b.duration.tuplet.enters for b in beats[:3]] == [3, 3, 3] and beats[3].duration.tuplet.enters == 1
    assert sum(b.duration.time for b in beats) == 3840
    heads = [g for g in read_gp5(path) if not g["tie"]]
    assert [round(g["beat"], 3) for g in heads] == [0.0, 0.333, 0.667, 1.0]
    assert "1마디 한 칸 = 1/12박" in text_tab(on)  # 셋잇단 박이 든 마디는 16분과 셋잇단을 함께 놓는 칸


def test_long_note_is_split_with_ties(tmp_path):
    pytest.importorskip("guitarpro")
    from amp.export.gp5 import read_gp5, write_gp5

    t = TempoMap.constant(60.0, 0.0)
    tab = quantize([Note("a", 0.0, 1.25, 33), Note("b", 3.5, 5.75, 35)], t, BassOptions())
    path = tmp_path / "tie.gp5"
    write_gp5(tab, path)
    got = read_gp5(path)
    a = [g for g in got if g["midi"] == 33]
    assert len(a) == 2 and a[1]["tie"] and sum(g["beats"] for g in a) == 1.25  # 4분 + 16분 붙임줄
    b = [g for g in got if g["midi"] == 35]
    assert b[0]["beat"] == 3.5 and sum(g["beats"] for g in b) == 2.25 and all(g["tie"] for g in b[1:])


def test_user_fingering_constrains_dp_and_is_saved():
    from amp.project import bass_io

    o = BassOptions()
    assert assign_fingering([33], o) == [(0, 5)]  # 기본(개방현 적게): A1을 E줄 5프렛
    assert assign_fingering([33], o, fixed={0: 1}) == [(1, 0)]  # A 개방현으로 지정
    assert assign_fingering([28], o, fixed={0: 3}) == [(0, 0)]  # G줄로는 E1을 칠 수 없어 지정 무시
    t = TempoMap.constant(60.0, 0.0)
    tab = quantize([Note("a", 0.0, 1.0, 33), Note("b", 1.0, 2.0, 28)], t, o, {"a": 1, "b": 3})
    assert [(n.string, n.fret, n.flags) for n in tab.notes] == [(1, 0, ("fixed",)), (0, 0, ("fixed_ignored",))]
    st = BassState(o, None, (Note("a", 0.0, 1.0, 33),), {}, {"a": 0})
    back = bass_io.from_dict(bass_io.to_dict(st))
    assert back.fingering == {"a": 0}
    assert "fingering" not in bass_io.to_dict(BassState())  # 없으면 이전 파일과 같다
