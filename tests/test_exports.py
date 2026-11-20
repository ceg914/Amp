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


# --- UFData · 조옮김 · MIDI 가사 인코딩 (VOX Factory 등 보컬 합성 프로그램) -------------------------

# utaformatix-data 소스(Kotlin Document/Project/Track/Note/Pitch/Tempo/TimeSignature.kt)의 필드 이름
UFDATA_FIELDS = {
    "document": {"formatVersion", "project"},
    "project": {"name", "tracks", "timeSignatures", "tempos", "measurePrefix"},
    "track": {"name", "notes", "pitch"},
    "note": {"key", "tickOn", "tickOff", "lyric", "phoneme"},
    "tempo": {"tickPosition", "bpm"},
    "time_signature": {"measurePosition", "numerator", "denominator"},
    "pitch": {"ticks", "values", "isAbsolute"},
}


def test_ufdata_matches_utaformatix_model_and_keeps_korean_lyrics(tmp_path):
    from amp.export import ufdata as UF

    p, _ids = _project()
    path = tmp_path / "곡.ufdata"
    rep = UF.export(p, str(path), ExportOptions())
    doc = json.loads(path.read_bytes().decode("utf-8"))
    assert set(doc) == UFDATA_FIELDS["document"] and doc["formatVersion"] == 1
    proj = doc["project"]
    assert set(proj) == UFDATA_FIELDS["project"]
    assert proj["tempos"] == [{"tickPosition": 0, "bpm": 120.0}, {"tickPosition": 1920, "bpm": 60.0}]
    assert proj["timeSignatures"] == [{"measurePosition": 0, "numerator": 4, "denominator": 4}]
    (track,) = proj["tracks"]
    assert set(track) == UFDATA_FIELDS["track"] and track["pitch"] is None
    assert all(set(n) == UFDATA_FIELDS["note"] for n in track["notes"])
    assert [(n["tickOn"], n["key"], n["lyric"]) for n in track["notes"]] == [
        (480, 60, "사"), (960, 62, "랑"), (1344, 64, "-"), (1824, 65, "-"), (2400, 67, "해")]
    assert all(n["tickOff"] > n["tickOn"] for n in track["notes"])
    assert rep.lyrics == 3 and rep.extended == 2
    assert any("첫 음: 0.50초 = 마디 1, 2.00박" in x for x in rep.info)
    assert any("음역: C4(60) ~ G4(67)" in x for x in rep.info)


def test_ufdata_bpm_is_not_rounded_and_pitch_curve_is_absolute_semitones():
    from amp.export import ufdata as UF

    p, _ids = _project(tempo=TempoMap((TempoPoint(0.0, 95.93),), 4))
    doc, _rep = UF.build(p, ExportOptions(pitch="detailed", extend_lyric=""))
    proj = doc["project"]
    assert proj["tempos"] == [{"tickPosition": 0, "bpm": 95.93}]  # 정수로 반올림하면 곡 끝에서 시간이 밀린다
    notes = proj["tracks"][0]["notes"]
    assert [n["lyric"] for n in notes] == ["사", "랑", "", "", "해"]
    pitch = proj["tracks"][0]["pitch"]
    assert set(pitch) == UFDATA_FIELDS["pitch"] and pitch["isAbsolute"] is True
    assert pitch["ticks"] == sorted(pitch["ticks"]) and len(pitch["ticks"]) == len(pitch["values"])
    vals = [v for v in pitch["values"] if v is not None]
    assert min(vals) > 59 and max(vals) < 68  # 음표 높이(60~67) 근처의 반음 값 (센트 아님)
    assert None in pitch["values"]  # 무성 구간(음표 사이)은 null
    assert all(a != b for a, b in zip(pitch["values"], pitch["values"][1:]))  # 바뀌는 점만


def test_transpose_moves_notes_and_pitch_together_in_every_synth_format():
    from amp.export import ufdata as UF

    p, _ids = _project()
    base, _ = UF.build(p, ExportOptions(pitch="detailed"))
    up, rep = UF.build(p, ExportOptions(pitch="detailed", transpose=-5))
    k0 = [n["key"] for n in base["project"]["tracks"][0]["notes"]]
    k1 = [n["key"] for n in up["project"]["tracks"][0]["notes"]]
    assert k1 == [k - 5 for k in k0]
    v0 = [v for v in base["project"]["tracks"][0]["pitch"]["values"] if v is not None]
    v1 = [v for v in up["project"]["tracks"][0]["pitch"]["values"] if v is not None]
    assert v1 == pytest.approx([v - 5 for v in v0], abs=0.011)
    assert any("-5반음 조옮김" in w for w in rep.warnings)
    assert [n.pitch for n in p.score.notes()] == [60, 62, 64, 65, 67]  # 프로젝트 음표는 그대로
    # USTX: 음표는 옮기고, 음표 기준 피치점은 그대로(같이 옮겼으므로)
    u0, _ = U.build(p, ExportOptions(pitch="detailed"))
    u1, _ = U.build(p, ExportOptions(pitch="detailed", transpose=-5))
    n0, n1 = u0["voice_parts"][0]["notes"], u1["voice_parts"][0]["notes"]
    assert [n["tone"] for n in n1] == [n["tone"] - 5 for n in n0]
    assert [n["pitch"]["data"] for n in n1] == [n["pitch"]["data"] for n in n0]


def test_midi_lyric_encoding_cp949_and_bends_follow_transpose(tmp_path):
    import mido

    p, _ids = _project()
    path = tmp_path / "cp949.mid"
    rep = F.export_midi(p, str(path), ExportOptions(midi_lyric_encoding="cp949", transpose=2, pitch="simple"))
    m = mido.MidiFile(str(path))
    lyr = [msg.text.encode("latin-1") for msg in m.tracks[1] if msg.type == "lyrics"]
    assert [b.decode("cp949") for b in lyr] == ["사", "랑", "해"]
    assert "가사 인코딩: CP949 (lyrics 메타 이벤트)" in rep.info
    notes = [msg.note for msg in m.tracks[1] if msg.type == "note_on" and msg.velocity]
    assert notes == [62, 64, 66, 67, 69]
    bends = [msg.pitch for msg in m.tracks[1] if msg.type == "pitchwheel"]
    assert max(abs(b) for b in bends) < 8191 * 0.5 / 12 * 2  # 조옮김과 함께 옮겼으므로 벤드는 ±0.5반음 안
    # UTF-8은 그대로
    F.export_midi(p, str(tmp_path / "u.mid"), ExportOptions())
    lyr8 = [msg.text.encode("latin-1").decode("utf-8") for msg in mido.MidiFile(str(tmp_path / "u.mid")).tracks[1]
            if msg.type == "lyrics"]
    assert lyr8 == ["사", "랑", "해"]


def test_note_range_uses_duration_weighted_core():
    from amp.export.common import note_range, range_text

    r = note_range([(48, 0.1), (60, 2.0), (62, 2.0), (64, 2.0), (79, 0.1)])
    assert (r["low"], r["high"]) == (48, 79)
    assert (r["core_low"], r["core_high"]) == (60, 64)  # 아주 짧은 끝 음은 주 음역에서 빠진다
    assert range_text(r).startswith("C3(48) ~ G5(79)") and "주 음역 C4(60) ~ E4(64)" in range_text(r)
    assert "A#3(58)" in range_text(r, -2)
