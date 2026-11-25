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


def test_chart_rules_per_difficulty():
    counts = {}
    for name, d in O.DIFFICULTIES.items():
        notes, s = O.chart(_notes(), T, d)
        counts[name] = len(notes)
        beats = [n.beat for n in notes]
        assert all(abs(b / d.snap - round(b / d.snap)) < 1e-9 for b in beats)  # 격자 위
        assert all(b2 - b1 >= d.min_gap - 1e-9 for b1, b2 in zip(beats, beats[1:]))  # 최소 간격
        assert all(0 <= n.column < 4 for n in notes)
        for a, b in zip(notes, notes[1:]):
            if a.column == b.column:
                assert b.beat - a.beat >= d.jack_gap - 1e-9  # 짧은 같은 열 연타 없음
            if a.end_beat is not None:
                assert a.end_beat <= b.beat - d.snap + 1e-9  # 롱노트는 다음 음 한 칸 전에 놓는다
        for n in notes:
            if n.end_beat is not None:
                assert n.end_beat - n.beat >= d.hold_min - 1e-9
        assert s.notes == len(notes) and sum(s.columns) == len(notes)
    assert counts["Easy"] < counts["Normal"] <= counts["Hard"]
    hard, _s = O.chart(_notes(), T, O.DIFFICULTIES["Hard"])
    by_beat = {n.beat: n for n in hard}
    assert by_beat[0.5].column > by_beat[0.0].column  # 올라가면 오른쪽
    assert by_beat[5.5].column < by_beat[5.0].column  # 내려가면 왼쪽
    assert by_beat[6.0].end_beat is not None  # 2초 끄는 음 = 롱노트
    assert by_beat[2.0].end_beat is not None


def test_osu_text_format():
    d = O.DIFFICULTIES["Normal"]
    notes, _s = O.chart(_notes(), T, d)
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
    for (col, t, end), n in zip(objs, notes):
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
