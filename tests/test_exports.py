"""STEP 5 Phase C: USTX·MIDI·MusicXML·JSON·CSV·LRC 내보내기를 쓰고 다시 읽어 확인한다."""

import json

import numpy as np
import pytest
import yaml

from amp.compare.score_io import parse_midi, parse_musicxml
from amp.core.model import AnalysisResult, PitchCurve, Project, Score
from amp.core.tempo import TempoMap, TempoPoint
from amp.export import formats as F
from amp.export import ustx as U
from amp.export.common import EXTEND, ExportOptions, TickMap
from amp.lyrics import ops
from amp.lyrics.model import LyricsState, NoteLink
from tests.conftest import note

FR = 100.0


def _project(tempo=None):
    """'사 랑 해': 사 = 음표 a, 랑 = 음표 b·c·d (멜리스마, c~d 사이 0.1초 빈틈), 해 = e. 템포 120 → 2초부터 60."""
    auto = [note("a", 0.5, 1.0, 60), note("b", 1.0, 1.4, 62), note("c", 1.4, 1.8, 64), note("d", 1.9, 2.3, 65),
            note("e", 3.0, 4.0, 67)]
    f0 = np.full(600, np.nan)
    for n in auto:
        i0, i1 = int(round(n.start * FR)), int(round(n.end * FR))
        f0[i0:i1] = 440 * 2 ** ((n.pitch - 69 + 0.2 * np.sin(np.arange(i1 - i0) / 3)) / 12)
    curve = PitchCurve(0.0, FR, f0, np.where(np.isfinite(f0), 0.8, 0.0))
    tempo = tempo or TempoMap((TempoPoint(0.0, 120.0), TempoPoint(2.0, 60.0)), 4)
    an = AnalysisResult(6.0, tempo, tuple(auto), pitch_curve=curve)
    score = Score.from_analysis(an)
    p = Project(analysis=an, score=score)
    st = ops.apply_text(LyricsState(), "사 랑 해")
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {ids[0]: (0.5, 1.0), ids[1]: (1.0, 2.3), ids[2]: (3.0, 4.0)}, source="model")
    links = {ids[0]: ["a"], ids[1]: ["b", "c", "d"], ids[2]: ["e"]}
    st = LyricsState(st.text, st.alignment.with_links({k: [NoteLink(n, 0.9, "user", True) for n in v]
                                                       for k, v in links.items()}), st.recognition, st.history,
                     st.pronunciation)
    p.lyrics = st
    return p, ids


def test_tick_map_handles_tempo_changes_without_drift():
    tm = TickMap([(0.0, 120.0), (2.0, 60.0)])
    assert tm.tick(1.0) == 960 and tm.tick(2.0) == 1920 and tm.tick(3.0) == 2400  # 60BPM: 1초 = 480틱
    assert tm.seconds(2400) == pytest.approx(3.0)
    # 절대 시각에서 바로 바꾸므로 많은 음표를 거쳐도 오차가 쌓이지 않는다
    times = np.cumsum(np.full(1000, 0.1234))
    # 오차는 언제나 반 틱 이하 (가장 느린 60BPM에서 반 틱 = 1.04ms)
    assert max(abs(tm.seconds(tm.tick(t)) - t) for t in times) <= 0.5 / 480 * 60 / 60 + 1e-9


def test_ustx_notes_tempo_lyrics_melisma_and_utf8(tmp_path):
    p, _ids = _project()
    path = tmp_path / "한글 파일명.ustx"
    rep = U.export(p, str(path), ExportOptions())
    text = path.read_bytes().decode("utf-8")
    doc = yaml.safe_load(text)
    assert doc["ustx_version"] == U.USTX_VERSION
    assert doc["tempos"] == [{"position": 0, "bpm": 120.0}, {"position": 1920, "bpm": 60.0}]
    assert doc["time_signatures"] == [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}]
    assert "singer" not in doc["tracks"][0] and "phonemizer" not in doc["tracks"][0]  # 보이스뱅크를 지어 넣지 않는다
    notes = doc["voice_parts"][0]["notes"]
    assert [(n["position"], n["tone"], n["lyric"]) for n in notes] == [
        (480, 60, "사"), (960, 62, "랑"), (1344, 64, EXTEND), (1824, 65, EXTEND), (2400, 67, "해")]
    # c(1.4~1.8)와 d(1.9~) 사이 빈틈은 이어 부르기 조건(앞 끝 = 이 시작)에 맞게 c를 늘림
    c, d = notes[2], notes[3]
    assert c["position"] + c["duration"] == d["position"] and rep.gaps_closed == 1
    assert notes[1]["position"] + notes[1]["duration"] == c["position"]
    assert notes[4]["duration"] == 480  # 60BPM에서 1초
    assert rep.lyrics == 3 and rep.extended == 2
    assert notes[0]["pitch"]["data"] == [{"x": -40.0, "y": 0.0, "shape": "io"}, {"x": 40.0, "y": 0.0, "shape": "io"}]


def test_ustx_pitch_modes_and_sources(tmp_path):
    p, _ids = _project()
    simple, _ = U.build(p, ExportOptions(pitch="simple"))
    detailed, _ = U.build(p, ExportOptions(pitch="detailed"))
    ns = simple["voice_parts"][0]["notes"]
    nd = detailed["voice_parts"][0]["notes"]
    pts_s = sum(len(n["pitch"]["data"]) for n in ns)
    pts_d = sum(len(n["pitch"]["data"]) for n in nd)
    assert pts_s < pts_d and all(len(n["pitch"]["data"]) <= 8 for n in ns)
    tm = TickMap([(0.0, 120.0), (2.0, 60.0)])
    for n in nd:  # 피치점은 음표 안쪽(ms), 0.1반음 단위, 음표 높이 기준
        dur_ms = (tm.seconds(n["position"] + n["duration"]) - tm.seconds(n["position"])) * 1000
        assert all(-1 <= pt["x"] <= dur_ms + 1 for pt in n["pitch"]["data"])
        assert all(abs(pt["y"]) <= 2.5 for pt in n["pitch"]["data"])  # 곡선이 ±0.2반음만 흔들림
    assert not ns[0]["pitch"]["snap_first"]
    # 가사 빼기
    no_lyr, _ = U.build(p, ExportOptions(lyrics=False))
    assert {n["lyric"] for n in no_lyr["voice_parts"][0]["notes"]} == {U.DEFAULT_LYRIC}
    # 구간
    part, rep = U.build(p, ExportOptions(range=(0.9, 2.5)))
    assert [n["tone"] for n in part["voice_parts"][0]["notes"]] == [62, 64, 65] and rep.range == (0.9, 2.5)


def test_midi_round_trip_keeps_times_tempo_and_korean_lyrics(tmp_path):
    p, _ids = _project()
    path = tmp_path / "곡.mid"
    rep = F.export_midi(p, str(path), ExportOptions(pitch="simple"))
    ps = parse_midi(path.read_bytes())
    assert ps.tempo == [(0.0, 120.0), (2.0, 60.0)]
    got = ps.notes[ps.parts[-1].id]
    assert [(round(n.start, 3), round(n.end, 3), n.pitch) for n in got] == [
        (0.5, 1.0, 60), (1.0, 1.4, 62), (1.4, 1.8, 64), (1.9, 2.3, 65), (3.0, 4.0, 67)]
    assert [n.lyric for n in got] == ["사", "랑", None, None, "해"]
    assert rep.lyrics == 3 and any("피치 벤드" in w for w in rep.warnings)


def test_musicxml_round_trip_quantized_with_ties_lyrics_and_extend(tmp_path):
    tempo = TempoMap((TempoPoint(0.0, 120.0),), 4)
    p, _ids = _project(tempo)
    path = tmp_path / "곡.musicxml"
    rep = F.export_musicxml(p, str(path), ExportOptions(), title="사랑해")
    ps = parse_musicxml(path.read_bytes())
    got = ps.notes["P1"]
    # 120BPM, 16분음표 = 0.125초 격자: 1.9 → 1.875, 2.3 → 2.25
    assert [(round(n.start, 3), round(n.end, 3), n.pitch) for n in got] == [
        (0.5, 1.0, 60), (1.0, 1.375, 62), (1.375, 1.75, 64), (1.875, 2.25, 65), (3.0, 4.0, 67)]
    assert [n.lyric for n in got] == ["사", "랑", None, None, "해"]
    assert got[2].lyric_extend and got[3].lyric_extend
    assert ps.tempo[0] == (0.0, 120.0) and ps.time_signatures[0] == (0.0, 4, 4)
    assert any("16분음표" in x for x in rep.omitted)
    # 마디를 넘는 음은 붙임줄로 나뉘었다가 다시 하나로 읽힌다
    long = Project(analysis=p.analysis, score=Score(tempo, [note("x", 1.5, 2.75, 60)]))
    F.export_musicxml(long, str(tmp_path / "tie.musicxml"), ExportOptions(lyrics=False))
    t = parse_musicxml((tmp_path / "tie.musicxml").read_bytes()).notes["P1"]
    assert len(t) == 1 and t[0].tied and (t[0].start, t[0].end) == (1.5, 2.75)


def test_json_csv_lrc(tmp_path):
    p, ids = _project()
    j = F.export_json(p, str(tmp_path / "a.json"), ExportOptions(pitch="detailed"))
    doc = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert len(doc["notes"]) == 5 and doc["notes"][2]["extend"] and doc["lyrics"]["syllables"][1]["notes"] == ["b", "c", "d"]
    assert len(doc["pitch_curve"]["f0_hz"]) == 600 and doc["pitch_curve"]["f0_hz"][0] is None
    assert "추정한 발음" in doc["lyrics"]["pronunciation_note"] and j.notes == 5
    F.export_csv(p, str(tmp_path / "notes.csv"), ExportOptions())
    raw = (tmp_path / "notes.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # Excel용 UTF-8 BOM
    assert "랑" in raw.decode("utf-8-sig") and (tmp_path / "notes-syllables.csv").is_file()
    rows = (tmp_path / "notes-syllables.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 4 and rows[2].startswith(f"{ids[1]},랑,1.0000,2.3000")
    F.export_lrc(p, str(tmp_path / "a.lrc"), ExportOptions(), title="사랑해")
    lrc = (tmp_path / "a.lrc").read_text(encoding="utf-8").splitlines()
    assert lrc[0] == "[ti:사랑해]" and lrc[-1] == "[00:00.50]사 랑 해"
    F.export_lrc(p, str(tmp_path / "b.lrc"), ExportOptions(lrc_words=True))
    assert "<00:01.00>랑" in (tmp_path / "b.lrc").read_text(encoding="utf-8")


def test_user_edits_win_over_auto_results(tmp_path):
    from dataclasses import replace

    p, _ids = _project()
    p.score.replace_notes(["e"], [replace(p.score.get("e"), pitch=68)])  # 사용자 수정 (같은 id, 높이만)
    user, _ = U.build(p, ExportOptions(source="user"))
    auto, _ = U.build(p, ExportOptions(source="auto"))
    assert user["voice_parts"][0]["notes"][-1]["tone"] == 68
    assert auto["voice_parts"][0]["notes"][-1]["tone"] == 67
    j = F.export_json(p, str(tmp_path / "u.json"), ExportOptions())
    doc = json.loads((tmp_path / "u.json").read_text(encoding="utf-8"))
    assert doc["notes"][-1]["status"] == "modified" and doc["notes"][0]["status"] == "original" and j.source == "user"


# OpenUtau(github.com/stakira/OpenUtau master, 2026-11-20 확인)의 YAML 필드 (C# 필드 이름에 underscored 규칙 적용).
# OpenUtau.Core/Ustx/UProject.cs·UNote.cs·UPart.cs·UTrack.cs에서 [YamlIgnore]가 아닌 공개 필드만.
OPENUTAU_FIELDS = {
    "project": {"name", "comment", "output_dir", "cache_dir", "ustx_version", "bpm", "beat_per_bar", "beat_unit",
                "expressions", "exp_selectors", "exp_primary", "exp_secondary", "expression_graphs",
                "default_expression_graphs", "key", "time_signatures", "tempos", "tracks", "voice_parts", "wave_parts"},
    "tempo": {"position", "bpm"},
    "time_signature": {"bar_position", "beat_per_bar", "beat_unit"},
    "track": {"singer", "phonemizer", "renderer_settings", "track_name", "track_color", "mute", "solo", "volume", "pan",
              "expression_graph", "voice_color_names", "track_expressions", "mix_fx"},
    "voice_part": {"name", "comment", "track_no", "position", "duration", "notes", "curves"},
    "note": {"position", "duration", "tone", "lyric", "pitch", "vibrato", "tuning", "phonemizer", "phoneme_expressions",
             "phoneme_overrides"},
    "pitch": {"data", "snap_first"},
    "point": {"x", "y", "shape"},
    "vibrato": {"length", "period", "depth", "in", "out", "shift", "drift", "vol_link"},
}


def test_ustx_keys_exist_in_openutau_model():
    p, _ids = _project()
    doc, _ = U.build(p, ExportOptions(pitch="detailed"))
    F_ = OPENUTAU_FIELDS
    assert set(doc) <= F_["project"]
    assert all(set(t) <= F_["tempo"] for t in doc["tempos"])
    assert all(set(t) <= F_["time_signature"] for t in doc["time_signatures"])
    assert all(set(t) <= F_["track"] for t in doc["tracks"])
    for part in doc["voice_parts"]:
        assert set(part) <= F_["voice_part"]
        for n in part["notes"]:
            assert set(n) <= F_["note"] and set(n["pitch"]) <= F_["pitch"] and set(n["vibrato"]) <= F_["vibrato"]
            assert all(set(pt) <= F_["point"] and pt["shape"] in ("io", "l", "i", "o", "sp") for pt in n["pitch"]["data"])
    # 버전은 YAML 스칼라 "0.6" (OpenUtau가 System.Version으로 읽음)
    assert "ustx_version: '0.6'" in U.dumps(doc) or "ustx_version: \"0.6\"" in U.dumps(doc) or "ustx_version: 0.6" in U.dumps(doc)
