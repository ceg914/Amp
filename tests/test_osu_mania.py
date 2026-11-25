"""v0.6.8 osu!mania 4K 내보내기: 채보 규칙(격자·간격·잭 피하기·롱노트·열 범위), .osu 형식, .osz 내용, 음원 시간축.

osu! 프로그램으로 열어 보는 시험은 아니다(이 환경에 osu!가 없음). 파일 형식은 osu! 파일 형식 v14의 mania 규칙에 맞춰
다시 읽어 확인한다.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
import soundfile as sf

from amp.audio.data import AudioData
from amp.core.model import Note
from amp.core.tempo import TempoMap
from amp.export import osu_mania as O

T = TempoMap.constant(120.0, 0.5)  # 박 = 0.5초, 첫 박 0.5초
MEL = [(0.5, 0.75, 60), (0.75, 1.0, 62), (1.0, 1.25, 64), (1.25, 1.5, 65), (1.5, 2.5, 67), (2.5, 2.75, 67),
       (2.75, 3.0, 67), (3.0, 3.25, 64), (3.25, 3.5, 60), (3.5, 5.5, 72), (5.5, 5.75, 55), (5.6, 5.7, 57)]


def _notes(mel=MEL):
    return [Note(f"n{i}", a, b, p) for i, (a, b, p) in enumerate(mel)]


def _onsets(seconds: float = 8.0):
    """120BPM·첫 박 0.5초 격자에 맞춘 합성 타격 세기 (마디 첫 박 1.0, 박 0.7, 8분 0.5, 16분 0.3, 그 밖 0.05)."""
    t = np.arange(0, seconds, 0.01)
    s = np.full(len(t), 0.05)
    for k in range(int((seconds - 0.5) / 0.125) + 1):
        tt = 0.5 + k * 0.125
        i = int(round(tt / 0.01))
        if i < len(s):
            s[i] = 1.0 if k % 16 == 0 else 0.7 if k % 4 == 0 else 0.5 if k % 2 == 0 else 0.3
    return np.stack([t, s], axis=1)


def _check_rules(notes, d, keys=4):
    grid = min(d.snap, d.fill_snap or d.snap)
    assert all(abs(n.beat / grid - round(n.beat / grid)) < 1e-9 for n in notes)  # 격자 위
    assert all(0 <= n.column < keys for n in notes)
    rows = {}
    for n in notes:
        rows.setdefault(n.beat, []).append(n.column)
    assert all(len(c) == len(set(c)) and len(c) <= 1 + d.chord_size for c in rows.values())  # 같은 줄 같은 열 없음
    free_at = {}
    for n in sorted(notes, key=lambda n: (n.beat, n.column)):
        if n.column in free_at:  # 롱노트가 끝난 뒤 한 칸, 단타 뒤 jack_gap 이상 지나야 같은 열을 다시 친다
            assert n.beat - free_at[n.column][0] >= free_at[n.column][1] - 1e-9
        free_at[n.column] = (n.end_beat, d.snap) if n.end_beat is not None else (n.beat, d.jack_gap)
        if n.end_beat is not None:
            assert n.end_beat - n.beat >= d.hold_min - 1e-9
    mel = [n.beat for n in notes if n.kind == "melody"]
    assert all(b2 - b1 >= d.min_gap - 1e-9 for b1, b2 in zip(mel, mel[1:]))  # 멜로디 최소 간격


def test_chart_rules_per_difficulty():
    counts = {}
    for name, d in O.DIFFICULTIES.items():
        notes, s = O.chart(_notes(), T, d, onsets=_onsets())
        _check_rules(notes, d)
        counts[name] = len(notes)
        assert s.notes == len(notes) and sum(s.columns) == len(notes)
        assert s.fills == sum(n.kind == "fill" for n in notes) and s.chords == sum(n.kind == "chord" for n in notes)
    assert counts["Easy"] < counts["Normal"] < counts["Hard"] < counts["Insane"] <= counts["Expert"]
    easy, s = O.chart(_notes(), T, O.DIFFICULTIES["Easy"], onsets=_onsets())
    assert s.fills == 0 and s.chords == 0  # Easy는 멜로디만
    expert, s = O.chart(_notes(), T, O.DIFFICULTIES["Expert"], onsets=_onsets())
    assert s.fills > 0 and s.chords > 0 and max(map(len, _rows(expert).values())) == 3  # 마디 첫 박 세 열


def _rows(notes):
    rows = {}
    for n in notes:
        rows.setdefault(n.beat, []).append(n)
    return rows


def test_melody_contour_and_holds():
    for name in ("Easy", "Hard"):
        notes, _s = O.chart(_notes(), T, O.DIFFICULTIES[name])
        mel = {n.beat: n for n in notes if n.kind == "melody"}
        assert mel[1.0].column > mel[0.0].column  # 올라가면 오른쪽
    hard, s = O.chart(_notes(), T, O.DIFFICULTIES["Hard"])  # 타격 정보가 없으면 채우기 없음, 동시치기는 마디 첫 박만
    assert s.fills == 0 and all(n.beat % 4 == 0 for n in hard if n.kind == "chord")
    mel = {n.beat: n for n in hard if n.kind == "melody"}
    assert mel[5.5].column < mel[5.0].column  # 내려가면 왼쪽
    assert mel[6.0].end_beat is not None and mel[2.0].end_beat is not None  # 길게 끄는 음 = 롱노트
    with_fill, _s = O.chart(_notes(), T, O.DIFFICULTIES["Insane"], onsets=_onsets())
    held = [n for n in with_fill if n.end_beat is not None]
    assert any(f.kind == "fill" and h.beat < f.beat < h.end_beat for h in held for f in with_fill)  # 롱노트 잡는 동안 다른 열에 채우기


def test_osu_text_format():
    d = O.DIFFICULTIES["Normal"]
    notes, _s = O.chart(_notes(), T, d, onsets=_onsets())
    text = O.osu_text(notes, T, d, O.Meta(title="Jogag Nabi", title_unicode="조각 나비", offset_ms=-10), "audio.ogg")
    assert text.startswith("osu file format v14\r\n")
    lines = text.split("\r\n")
    assert "Mode: 3" in lines and "CircleSize:4" in lines and "AudioFilename: audio.ogg" in lines
    assert "TitleUnicode:조각 나비" in lines and "Version:Normal" in lines
    tp = lines[lines.index("[TimingPoints]") + 1].split(",")
    assert tp[0] == "490" and float(tp[1]) == 500.0 and tp[2] == "4" and tp[6] == "1"  # 첫 박 500ms - 보정 10ms
    objs = O.parse_hit_objects(text)
    assert len(objs) == len(notes)
    xs = {int(line.split(",")[0]) for line in lines[lines.index("[HitObjects]") + 1:] if line}
    assert xs <= {64, 192, 320, 448}
    for (col, t, end), n in zip(objs, sorted(notes, key=lambda n: (n.beat, n.column))):
        assert col == n.column and t == round(T.time_at(n.beat) * 1000 - 10)
        assert (end is None) == (n.end_beat is None)
        if end is not None:
            assert end == round(T.time_at(n.end_beat) * 1000 - 10) and end > t
    late = TempoMap.constant(100.0, 3.1)  # 첫 박이 마디(2.4초)보다 늦으면 마디 단위로 앞당긴 0ms 이후 첫 마디
    assert O.timing_points(late)[0].split(",")[0] == "700"


def test_ascii_title():
    assert O.ascii_text("조각 나비") == "Jogag Nabi"
    assert O.ascii_text("Red Peter [T7Q]") == "Red Peter [T7Q]"
    assert O.ascii_text("こんにちは") == ""


def test_osz_contents_and_audio_alignment():
    sr = 44100
    x = np.zeros((sr * 4, 2), np.float32)
    for k in range(7):
        i = int((0.5 + 0.5 * k) * sr)
        x[i:i + 300] = (np.hanning(600)[:300] * 0.8)[:, None]
    audio = AudioData(x, sr)
    data, summaries = O.build_osz(_notes(), T, audio, O.Meta(title="Test"), ("Easy", "Hard"))
    z = zipfile.ZipFile(io.BytesIO(data))
    names = sorted(z.namelist())
    assert names == ["Unknown - Test (SONATLAS) [Easy].osu", "Unknown - Test (SONATLAS) [Hard].osu", "audio.ogg"]
    assert [s.name for s in summaries] == ["Easy", "Hard"]
    y, ysr = sf.read(io.BytesIO(z.read("audio.ogg")), dtype="float32", always_2d=True)
    assert ysr == sr and y.shape[0] == x.shape[0]
    seg = slice(int(0.4 * sr), int(3.9 * sr))
    lag = max(range(-100, 101), key=lambda L: float(np.dot(y[seg.start + L:seg.stop + L, 0], x[seg, 0])))
    assert lag == 0  # 분석에 쓴 시간축 그대로 (노트와 소리가 어긋나지 않음)


def test_long_audio_encodes_without_crash():
    """libsndfile Vorbis 인코더는 60초 넘게 한 번에 쓰면 죽었다. 나눠 쓰는지 확인 (실제 곡 길이)."""
    sr = 44100
    x = (np.random.default_rng(0).standard_normal((sr * 75, 2)) * 0.05).astype(np.float32)
    data = O.encode_ogg(AudioData(x, sr))
    y, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    assert y.shape == x.shape


@pytest.mark.parametrize("mel", [[], [(1.0, 1.1, 60)]])
def test_empty_and_single(mel):
    notes, s = O.chart(_notes(mel), T, O.DIFFICULTIES["Normal"])
    assert len(notes) == len(mel) and s.notes == len(mel)
