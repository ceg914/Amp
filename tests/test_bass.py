"""v0.6.3 베이스 TAB: 퉁김 분할, 옥타브 도약 보존, 줄·프렛 배치, 박 맞춤, 텍스트 TAB, GP5 왕복, 파이프라인(분리·캐시).

합성한 뜯은 줄 소리(amp.devtools.synth.plucked)로 시험한다. 실제 베이스 녹음 정확도는 조각 나비 GP5 기준으로
따로 쟀다(저장소 밖, 상용 음원이라 넣지 않음): ±100ms F1 0.915, 앱 템포로 16분음표 칸까지 맞은 음 90.6%.
"""

from __future__ import annotations

import io
import threading
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from amp.core.model import Note
from amp.core.tempo import TempoMap
from amp.engine.bass import (BassError, BassOptions, Tab, TabNote, assign_fingering, pluck_onsets, positions,
                             quantize, split_at_onsets, string_names, text_tab)

# 120BPM, 첫 박 0.5초. (MIDI, 박): 같은 음 붙여 반복, 아래로 큰 도약(A2→E1), 옥타브 오가기(E1↔E2)
LINE = [(33, 0.5), (33, 0.5), (33, 0.5), (33, 0.5), (38, 1.0), (36, 0.5), (28, 1.5), (40, 0.25), (43, 0.25),
        (45, 0.5), (28, 0.5), (40, 0.5), (28, 0.5), (40, 0.5)]
T0, SPB = 0.5, 0.5


def _melody(line=LINE):
    mel, b = [], 0.0
    for p, d in line:
        mel.append((T0 + b * SPB, T0 + (b + d) * SPB - 0.005, p))
        b += d
    return mel, b


def _expected(line=LINE):
    out, b = [], 0.0
    for p, d in line:
        out.append((b, d, p))
        b += d
    return out


def _bass_audio():
    from amp.devtools.synth import plucked

    mel, beats = _melody()
    return plucked(mel, T0 + beats * SPB + 0.5)


# --- 설정 ----------------------------------------------------------------------------

def test_options_validate_and_roundtrip():
    o = BassOptions(tuning="5_standard", max_fret=22, open_strings="many", grid=0.5)
    assert BassOptions.from_dict(o.to_dict()) == o
    assert o.midi_range() == (23, 43 + 22)
    lo, hi = BassOptions().hz_range()
    assert 38 < lo < 39.5 and hi <= 400.0
    for bad in ({"tuning": "7_weird"}, {"max_fret": 30}, {"open_strings": "x"}, {"grid": 1 / 3}):
        with pytest.raises(BassError):
            BassOptions(**bad)
    assert BassOptions.from_dict({"max_fret": 99, "tuning": "4_drop_d"}) == BassOptions()  # 잘못된 값이면 기본값
    assert BassOptions.from_dict({"tuning": "4_drop_d", "unknown": 1}).tuning == "4_drop_d"
    assert string_names(BassOptions()) == ["E", "A", "D", "G"]
    assert string_names(BassOptions(tuning="5_standard")) == ["B", "E", "A", "D", "G"]


# --- 음표 검출 -------------------------------------------------------------------------

def test_plucked_line_is_detected_and_quantized_exactly():
    pytest.importorskip("librosa")
    from amp.devtools.synth import SR
    from amp.engine.bass import detect_bass_notes
    from amp.engine.dsp.pitch import track_pitch

    o = BassOptions()
    f = track_pitch(_bass_audio(), SR, fmin=o.hz_range()[0], fmax=o.hz_range()[1])
    notes, curve, _art = detect_bass_notes(f, o)
    starts = [m[0] for m in _melody()[0]]
    assert len(notes) == len(LINE)
    assert max(abs(n.start - s) for n, s in zip(notes, starts)) < 0.04  # 큰 도약 뒤 E1도 늦지 않게
    tab = quantize(notes, TempoMap.constant(120.0, T0), o)
    assert [(n.start_beat, n.beats, n.pitch) for n in tab.notes] == _expected()
    assert tab.pickup_bars == 0 and tab.unplayable == 0
    assert np.isfinite(curve.f0_hz).any()


def test_octave_jumps_survive_curve_cleanup():
    """보컬용 옥타브 보정은 주변과 한 옥타브 어긋난 짧은 구간을 옮긴다. 베이스에서는 끈다(E1↔E2가 흔하다)."""
    from amp.engine.pitch_post import build_curve

    fr = 100.0
    midi = np.r_[np.full(30, 40.0), np.full(5, np.nan), np.full(20, 28.0), np.full(5, np.nan), np.full(30, 40.0)]
    hz = 440 * 2 ** ((midi - 69) / 12)  # E2 · E1 · E2 (음 사이 50ms 쉼)
    prob = np.full(len(hz), 0.9)
    rms = np.full(len(hz), 0.1)
    vocal = build_curve(hz, prob, rms, fr).midi()
    bass = build_curve(hz, prob, rms, fr, fix_octaves=False).midi()
    assert np.nanmedian(vocal[37:53]) == pytest.approx(40, abs=0.1)  # 보컬 기본값은 그대로 (옥타브 오류로 보고 옮김)
    assert np.nanmedian(bass[37:53]) == pytest.approx(28, abs=0.1)


def test_pluck_onsets_and_split():
    fr = 100.0
    db = np.full(200, -30.0)
    for i in (20, 60, 61, 120):  # 61은 60의 80ms 안이라 하나로
        db[i:i + 3] = [-20, -15, -12]
        db[i + 3:i + 30] = np.linspace(-14, -28, 27)
    db[150:] = -70  # 바닥 아래는 무시
    db[160:163] = [-60, -55, -52]
    rms = 10 ** (db / 20)
    on = pluck_onsets(rms, fr)
    assert on == pytest.approx([0.2, 0.6, 1.2])
    n = Note("a", 0.1, 1.4, 33)
    parts = split_at_onsets([n], on)
    assert [t for p in parts for t in (p.start, p.end)] == pytest.approx([0.1, 0.2, 0.2, 0.6, 0.6, 1.2, 1.2, 1.4])
    assert len(split_at_onsets([Note("b", 0.1, 0.63, 33)], on)) == 2  # 0.2에서만. 끝 0.03초 앞의 0.6은 조각이 짧아 자르지 않음
    assert parts[0].id == "a" and len({p.id for p in parts}) == len(parts)
    assert all(p.pitch == 33 for p in parts)


# --- 줄·프렛 ----------------------------------------------------------------------------

def test_fingering_is_playable_and_follows_options():
    o = BassOptions()
    line = [33, 36, 38, 40, 43, 45, 47, 48, 28, 40, 52]
    path = assign_fingering(line, o)
    for p, (s, f) in zip(line, path):
        assert o.open_midis[s] + f == p and 0 <= f <= o.max_fret
    # 칠 수 없는 음: 4현 표준의 B0(23), 최대 프렛 위
    path = assign_fingering([23, 33, 70], o)
    assert path[0] is None and path[2] is None and path[1] is not None
    five = BassOptions(tuning="5_standard")
    assert assign_fingering([23], five) == [(0, 0)]
    assert positions(33, BassOptions(max_fret=5)) == [(0, 5), (1, 0)]
    # 개방현 선호: A1·D2를 오갈 때 '많이'는 개방현, '적게'는 같은 손 위치(5프렛)
    riff = [33, 38, 33, 38, 35, 36]
    many = assign_fingering(riff, BassOptions(open_strings="many"))
    few = assign_fingering(riff, BassOptions(open_strings="few"))
    assert sum(f == 0 for _s, f in many) > sum(f == 0 for _s, f in few)


# --- 박 맞춤·텍스트 ---------------------------------------------------------------------

def test_quantize_overlap_duplicate_and_pickup():
    t = TempoMap.constant(120.0, 1.0)  # 첫 박 1초
    notes = [Note("p", 0.5, 0.75, 33),  # 첫 박 1박 앞 → 앞에 1마디
             Note("a", 1.0, 1.6, 33), Note("b", 1.02, 1.3, 40),  # 같은 칸 → 앞 것만
             Note("c", 1.5, 2.0, 38)]  # a와 겹침 → a를 c 시작에서 자름
    tab = quantize(notes, t, BassOptions())
    assert tab.pickup_bars == 1
    got = [(n.note_id, n.start_beat, n.beats) for n in tab.notes]
    assert got == [("p", 3.0, 0.5), ("a", 4.0, 1.0), ("c", 5.0, 1.0)]
    short = quantize([Note("x", 1.0, 1.01, 33)], t, BassOptions(grid=0.5))
    assert short.notes[0].beats == 0.5  # 격자보다 짧아도 한 칸


def test_text_tab_layout():
    o = BassOptions()
    tab = Tab(o, 100.0, 4, [TabNote(0.0, 1.0, 33, 0, 5), TabNote(1.0, 1.0, 45, 2, 7), TabNote(4.5, 0.5, 52, 3, 12)])
    text = text_tab(tab, title="시험")
    lines = text.splitlines()
    assert lines[0] == "시험" and "4현 표준" in lines[1] and "16분음표" in lines[1]
    g, d, a, e = (next(line for line in lines if line.startswith(f"{s}|")) for s in "GDAE")
    assert e.startswith("E|5--") and d[2 + 4 * 3] == "7"  # 1박 = 16분 4칸 = 12글자 뒤
    bar2 = g.split("|")[2]
    assert bar2[2 * 3:2 * 3 + 2] == "12"
    assert set(a[2:]) == {"-", "|"}  # A줄은 비었다
    assert len({len(x) for x in (g, d, a, e)}) == 1


def test_text_tab_file(tmp_path):
    from amp.export.tab_text import write_text_tab

    tab = Tab(BassOptions(), 100.0, 4, [TabNote(0.0, 1.0, 33, 0, 5)])
    path = tmp_path / "베이스 TAB.txt"
    write_text_tab(tab, path, title="조각 나비")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw  # 메모장용 BOM·줄바꿈
    assert raw.decode("utf-8-sig").replace("\r\n", "\n") == text_tab(tab, title="조각 나비")


# --- GP5 ----------------------------------------------------------------------------

def test_gp5_roundtrip(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from amp.export.gp5 import read_gp5, write_gp5

    o = BassOptions(tuning="5_standard")
    t = TempoMap.constant(96.0, 0.0)
    spb = 60 / 96
    notes = [Note("a", 0.0, 2 * spb, 23), Note("b", 2 * spb, 2.75 * spb, 35), Note("c", 3.5 * spb, 6 * spb, 40),
             Note("d", 7 * spb, 7.25 * spb, 60)]
    tab = quantize(notes, t, o)
    path = tmp_path / "베이스.gp5"
    write_gp5(tab, path, "조각 나비 (연습)", "SONATLAS")
    song = gp.parse(str(path), encoding="cp949")
    assert song.title == "조각 나비 (연습)" and song.tempo == 96
    assert [(s.number, s.value) for s in song.tracks[0].strings] == [(1, 43), (2, 38), (3, 33), (4, 28), (5, 23)]
    for mh, meas in zip(song.measureHeaders, song.tracks[0].measures):
        total = sum(b.duration.time for b in meas.voices[0].beats)
        assert total == 960 * mh.timeSignature.numerator  # 쉼표로 마디를 채운다
    got = read_gp5(path)
    heads = [g for g in got if not g["tie"]]
    assert [(g["beat"], g["midi"]) for g in heads] == [(n.start_beat, n.pitch) for n in tab.notes]
    assert [(g["string"], g["fret"]) for g in heads] == [(n.string, n.fret) for n in tab.notes]
    # c는 3.5박~6박: 마디를 넘어 붙임줄로 이어진다
    c = [g for g in got if g["midi"] == 40]
    assert c[0]["beat"] == 3.5 and any(g["tie"] for g in c) and sum(g["beats"] for g in c) == 2.5


# --- 파이프라인 -------------------------------------------------------------------------

def _runner(calls: Counter):
    from amp.engine import worker

    lock = threading.Lock()

    def run(command, args, progress=None, token=None, numba_cache_dir=None):
        with lock:
            calls[command] += 1
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    return run


@pytest.fixture()
def song(tmp_path):
    from amp.devtools.synth import SR

    y = _bass_audio()
    path = tmp_path / "베이스 곡.wav"
    sf.write(path, np.stack([y, y], 1), SR)
    return path


def _request(song, tmp_path, **kw):
    from amp.audio.decoder import load_audio
    from amp.engine.bass_pipeline import BassRequest
    from amp.engine.cache import file_content_key

    a = load_audio(str(song))
    return BassRequest(str(song), file_content_key(str(song)), a.duration, stems_dir=str(tmp_path / "stems"),
                       cache_root=str(tmp_path / "cache"), device="cpu", **kw)


def test_pipeline_separates_once_then_reuses(song, tmp_path):
    pytest.importorskip("librosa")
    from amp.engine.bass_pipeline import BassPipeline, find_bass_stem

    calls: Counter = Counter()
    req = _request(song, tmp_path)
    out = BassPipeline(_runner(calls), separation_model="_withbass").run(req)
    assert calls == Counter({"separate": 1, "pitch": 1}) and out.source["separated"]
    assert [n.pitch for n in out.notes] == [p for p, _d in LINE]
    assert find_bass_stem(req.stems_dir, req.source_sha256) is not None
    again = BassPipeline(_runner(calls), separation_model="_withbass").run(req)
    assert calls == Counter({"separate": 1, "pitch": 1}) and again.reused == {"separation": True, "pitch": True}
    assert [(n.start, n.pitch) for n in again.notes] == [(n.start, n.pitch) for n in out.notes]
    wide = BassPipeline(_runner(calls), separation_model="_withbass").run(
        _request(song, tmp_path, options=BassOptions(tuning="5_standard")))
    assert calls["pitch"] == 2 and wide.reused["separation"]  # 음역이 바뀌면 피치만 다시


def test_pipeline_adds_bass_without_touching_existing_stems(song, tmp_path):
    """베이스가 없는 예전 분리 결과: 보컬·반주 파일과 stems_id는 그대로 두고 bass.wav만 더한다."""
    pytest.importorskip("librosa")
    from amp.engine import stems as stem_store
    from amp.engine.bass_pipeline import BassPipeline
    from amp.engine.model_store import ModelStore, identity_for

    req = _request(song, tmp_path)
    store = ModelStore(req.stems_dir, req.source_sha256)
    tmp = store.new_tmp()
    calls: Counter = Counter()
    _runner(calls)("separate", {"input": str(song), "out_dir": str(tmp), "model": "_withbass"})
    (tmp / "bass.wav").unlink()  # 베이스 저장 이전(v0.6.2)에 분리한 결과처럼
    entry = store.commit(tmp, identity_for("_withbass", req.source_sha256), {"device": "cpu"})
    store.activate(entry.key)
    vocals = (entry.path / "vocals.wav").read_bytes()
    sid = store.pointer().stems_id
    assert stem_store.optional_path(entry.path, entry.manifest, "bass") is None

    out = BassPipeline(_runner(calls), separation_model="_withbass").run(req)
    assert out.source["store_key"] == entry.key and out.source["separated"]
    assert (entry.path / "vocals.wav").read_bytes() == vocals
    pointer = store.pointer()
    assert pointer.stems_id == sid and "bass" in pointer.files  # 보컬 피치 캐시가 그대로 유효하다
    assert stem_store.validate(req.stems_dir, pointer, req.source_sha256) == (True, "")
    assert len(store.entries()) == 1


def test_pipeline_original_and_no_separation(song, tmp_path):
    pytest.importorskip("librosa")
    from amp.core.errors import AnalysisError
    from amp.engine.bass_pipeline import BassPipeline

    calls: Counter = Counter()
    with pytest.raises(AnalysisError):
        BassPipeline(_runner(calls)).run(_request(song, tmp_path, allow_separation=False))
    assert calls["separate"] == 0
    out = BassPipeline(_runner(calls)).run(_request(song, tmp_path, use_original=True))
    assert calls == Counter({"pitch": 1}) and out.source["kind"] == "original"
    assert [n.pitch for n in out.notes] == [p for p, _d in LINE]


def test_five_string_low_b_uses_longer_frame(recwarn):
    """5현 B0(30.9Hz)는 1024 창(64ms)에 두 주기가 들어가지 않는다 → 2048 창. 4현은 기존 창 그대로."""
    pytest.importorskip("librosa")
    from amp.devtools.synth import SR, plucked
    from amp.engine.bass import detect_bass_notes
    from amp.engine.dsp.pitch import FRAME_LENGTH, frame_length_for, track_pitch

    assert frame_length_for(BassOptions().hz_range()[0]) == FRAME_LENGTH
    o = BassOptions(tuning="5_standard")
    lo, hi = o.hz_range()
    assert frame_length_for(lo) == 2048
    mel = [(0.5, 0.99, 23), (1.0, 1.49, 23), (1.5, 1.99, 28), (2.0, 2.49, 35), (2.5, 3.0, 23)]
    f = track_pitch(plucked(mel, 3.5), SR, fmin=lo, fmax=hi, frame_length=frame_length_for(lo))
    notes, _c, _a = detect_bass_notes(f, o)
    assert [n.pitch for n in notes] == [23, 23, 28, 35, 23]
    assert max(abs(n.start - m[0]) for n, m in zip(notes, mel)) < 0.03
    assert not [w for w in recwarn if "two periods" in str(w.message)]
