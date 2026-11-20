"""STEP 5 Phase B (Qt 없음): 악보 가져오기, 악보 정렬, 음정 비교, 분석 오류 검토."""

import io
import zipfile

import mido
import numpy as np
import pytest

from amp.compare import reference as R
from amp.compare.align import auto_align
from amp.compare.intonation import analyse
from amp.compare.review import (ReviewState, lyric_rules, mark, note_rules, state_from_dict, state_to_dict, vocal_rules)
from amp.compare.score_io import build_reference, guess_vocal_part, parse_midi, parse_musicxml
from amp.core.errors import AnalysisError
from amp.core.model import AnalysisResult, PitchCurve
from amp.core.tempo import TempoMap
from tests.conftest import note

# --- MIDI ----------------------------------------------------------------------------


def _midi(tracks, tpb=480, tempos=((0, 500000),), ts=((0, 4, 4),), lyrics=()):
    """tracks: [(이름, 채널, [(시작 틱, 길이 틱, 음높이)])]."""
    mid = mido.MidiFile(ticks_per_beat=tpb, type=1)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    ev = [(t, mido.MetaMessage("set_tempo", tempo=us)) for t, us in tempos]
    ev += [(t, mido.MetaMessage("time_signature", numerator=n, denominator=d)) for t, n, d in ts]
    _write(meta, ev)
    for name, ch, notes in tracks:
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        ev = [(0, mido.MetaMessage("track_name", name=name))]
        for on, dur, p in notes:
            ev.append((on, mido.Message("note_on", channel=ch, note=p, velocity=90)))
            ev.append((on + dur, mido.Message("note_off", channel=ch, note=p, velocity=0)))
        # 실제 한글 가사 MIDI처럼 UTF-8/CP949 바이트를 그대로 넣는다 (mido는 latin-1로 쓰고 읽는다)
        ev += [(t, mido.MetaMessage("lyrics", text=txt.encode(enc).decode("latin-1"))) for t, txt, enc in lyrics
               if name == "Vocal"]
        _write(tr, ev)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _write(track, events):
    events = sorted(events, key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))
    last = 0
    for t, msg in events:
        track.append(msg.copy(time=t - last))
        last = t


def test_single_track_midi_times_and_rests():
    data = _midi([("Vocal", 0, [(0, 480, 60), (960, 240, 62), (1200, 720, 64)])])  # 120BPM: 4분음표 0.5초
    ps = parse_midi(data)
    assert len(ps.parts) == 1 and ps.tempo == [(0.0, 120.0)] and ps.time_signatures == [(0.0, 4, 4)]
    notes = ps.notes[ps.parts[0].id]
    assert [(round(n.start, 3), round(n.end, 3), n.pitch) for n in notes] == [(0.0, 0.5, 60), (1.0, 1.25, 62), (1.25, 2.0, 64)]
    assert notes[1].beat == 2.0 and notes[2].beats == 1.5  # 쉼표(1박)는 음표 사이 간격으로 남는다


def test_tempo_and_time_signature_changes_in_midi():
    data = _midi([("Vocal", 0, [(0, 480, 60), (960, 480, 62), (1920, 480, 64)])],
                 tempos=((0, 500000), (960, 1000000)), ts=((0, 4, 4), (1920, 3, 4)))  # 2박 뒤 60BPM
    ps = parse_midi(data)
    n = ps.notes[ps.parts[0].id]
    assert n[1].start == pytest.approx(1.0) and n[1].end == pytest.approx(2.0)  # 60BPM에서 1박 = 1초
    assert n[2].start == pytest.approx(3.0)
    assert ps.tempo == [(0.0, 120.0), (1.0, 60.0)] and ps.time_signatures[1] == (3.0, 3, 4)


def test_multi_track_midi_part_choice_keeps_accompaniment_out():
    data = _midi([("Piano", 0, [(0, 960, 48), (0, 960, 52), (0, 960, 55)]),
                  ("Vocal", 1, [(0, 480, 67), (480, 480, 69)]),
                  ("Drums", 9, [(0, 120, 36)])],
                 lyrics=[(0, "사", "utf-8"), (480, "랑", "cp949")])
    ps = parse_midi(data)
    names = {p.id: p for p in ps.parts}
    assert len(ps.parts) == 3
    piano = next(p for p in ps.parts if p.name.startswith("Piano"))
    assert piano.polyphonic and next(p for p in ps.parts if p.drums)
    vid = guess_vocal_part(ps)
    assert names[vid].name.startswith("Vocal") and names[vid].has_lyrics
    ref = build_reference(ps, vid, "song.mid", "abc")
    assert [n.pitch for n in ref.notes] == [67, 69] and [n.lyric for n in ref.notes] == ["사", "랑"]
    assert len(ref.parts) == 3  # 다른 성부는 목록에만 남고 음표는 섞지 않는다


def test_midi_without_lyrics_is_not_invented():
    ps = parse_midi(_midi([("Lead", 0, [(0, 480, 60)])]))
    assert ps.notes[ps.parts[0].id][0].lyric is None


def test_broken_midi_is_rejected():
    with pytest.raises(AnalysisError):
        parse_midi(b"MThd garbage")


# --- MusicXML ------------------------------------------------------------------------

XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Voice</part-name></score-part>
    <score-part id="P2"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <direction><sound tempo="120"/></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <lyric number="1"><syllabic>begin</syllabic><text>노</text></lyric></note>
      <note><rest/><duration>2</duration><voice>1</voice></note>
      <note><pitch><step>E</step><alter>-1</alter><octave>4</octave></pitch><duration>4</duration><voice>1</voice>
        <tie type="start"/><lyric number="1"><text>래</text></lyric></note>
    </measure>
    <measure number="2">
      <attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
      <direction><sound tempo="60"/></direction>
      <note><pitch><step>E</step><alter>-1</alter><octave>4</octave></pitch><duration>2</duration><voice>1</voice><tie type="stop"/></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <lyric number="1"><extend/></lyric></note>
      <note><grace/><pitch><step>A</step><octave>4</octave></pitch><voice>1</voice></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice></note>
      <note><chord/><pitch><step>G</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice></note>
      <backup><duration>4</duration></backup>
      <note><pitch><step>C</step><octave>2</octave></pitch><duration>4</duration><voice>2</voice></note>
    </measure>
  </part>
</score-partwise>"""


def test_musicxml_lyrics_ties_rests_tempo_and_time_changes():
    ps = parse_musicxml(XML.encode())
    assert [p.id for p in ps.parts] == ["P1", "P2"]
    v = ps.notes["P1"]
    # 마디 1 (120BPM, 4/4): C4 0~0.5, 쉼표 0.5~1.0, E♭4 1.0~2.0 + 붙임줄로 마디 2 첫 박(60BPM, 1초)까지 → 2.0+1.0
    assert [(round(n.start, 3), round(n.end, 3), n.pitch) for n in v] == [
        (0.0, 0.5, 60), (1.0, 3.0, 63), (3.0, 4.0, 67), (4.0, 5.0, 65)]
    assert v[1].tied and v[1].lyric == "래" and v[0].lyric == "노"
    assert v[2].lyric is None and v[2].lyric_extend  # 앞 음절을 늘여 부름
    assert ps.time_signatures == [(0.0, 4, 4), (2.0, 3, 4)] and ps.tempo == [(0.0, 120.0), (2.0, 60.0)]
    piano = {p.id: p for p in ps.parts}["P2"]
    assert piano.polyphonic and piano.notes == 3  # 화음·두 번째 성부(backup)
    assert guess_vocal_part(ps) == "P1"


def test_compressed_mxl_is_read():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="score.xml"/></rootfiles></container>')
        z.writestr("score.xml", XML)
    ps = parse_musicxml(buf.getvalue(), compressed=True)
    assert len(ps.notes["P1"]) == 4


def test_reference_round_trip_and_offset():
    ps = parse_musicxml(XML.encode())
    ref = build_reference(ps, "P1", "s.musicxml", "x", stored_path="scores/s.musicxml")
    ref = ref.with_alignment(R.ScoreAlignment(offset=12.5))
    assert ref.audio_span(ref.notes[0]) == (12.5, 13.0)  # 악보 시작 오프셋
    back = R.from_dict(R.to_dict(ref))
    assert back == ref and back.notes[1].tied and back.stored_path == "scores/s.musicxml"
    nudged = back.alignment.nudged(0.5)
    assert nudged.offset == 13.0 and not nudged.confirmed


# --- 정렬·음정 비교 ---------------------------------------------------------------------

FR = 100.0


def _curve_from(segments, dur):
    """segments: [(시작, 끝, midi 함수(t))]."""
    f0 = np.full(int(dur * FR), np.nan)
    for a, b, fn in segments:
        i0, i1 = int(round(a * FR)), min(int(round(b * FR)), f0.size)
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / FR
        f0[i0:i1] = 440 * 2 ** ((fn(t) - 69) / 12)
    return PitchCurve(0.0, FR, f0, np.where(np.isfinite(f0), 0.9, 0.05))


def _melody(n=40, seed=3):
    rng = np.random.default_rng(seed)
    notes, t = [], 0.0
    for k in range(n):
        d = float(rng.choice([0.3, 0.5, 0.8]))
        notes.append(R.ReferenceNote(f"r{k}", t, t + d * 0.9, int(rng.choice([60, 62, 64, 65, 67, 69, 71]))))
        t += d
    return notes, t


def test_auto_align_finds_long_intro_and_tempo_change():
    notes, total = _melody()
    half = total / 2

    def true_map(s):
        return 9.0 + (s if s < half else half + (s - half) * 1.12)  # 9초 전주, 뒤 절반 12% 느려짐

    curve = _curve_from([(true_map(n.start), true_map(n.end), lambda t, p=n.pitch: p + 0 * t) for n in notes],
                        true_map(total) + 2)
    al = auto_align(notes, curve, true_map(total) + 2)
    errs = [abs(al.audio_time(n.start) - true_map(n.start)) for n in notes]
    assert np.median(errs) < 0.06 and max(errs) < 0.2
    assert al.quality["match"] > 0.9 and not al.quality["uncertain"] and not al.confirmed


def test_auto_align_marks_wrong_part_uncertain():
    notes, total = _melody(seed=4)
    other, _ = _melody(seed=9)
    curve = _curve_from([(n.start + 2, n.end + 2, lambda t, p=n.pitch: p + 0 * t) for n in other], total + 4)
    al = auto_align(notes, curve, total + 4)
    assert al.quality["uncertain"] and "직접" in al.quality["reason"]


def _ref(notes, confirmed=True, **al):
    return R.ReferenceScore("s", "", "midi", notes=tuple(notes), alignment=R.ScoreAlignment(confirmed=confirmed, **al))


def test_intonation_exact_sharp_flat_vibrato_and_unvoiced():
    notes = [R.ReferenceNote("a", 1, 2, 60), R.ReferenceNote("b", 2, 3, 62), R.ReferenceNote("c", 3, 4, 64),
             R.ReferenceNote("d", 4, 5, 65), R.ReferenceNote("e", 5, 6, 67)]
    curve = _curve_from([(1, 2, lambda t: 60 + 0 * t), (2, 3, lambda t: 62.5 + 0 * t), (3, 4, lambda t: 63.5 + 0 * t),
                         (4, 5, lambda t: 65 + 0.6 * np.sin(2 * np.pi * 6 * t))], 7)
    rep = analyse(_ref(notes), curve)
    by = {n.ref_id: n for n in rep.notes}
    assert by["a"].median_cents == pytest.approx(0, abs=1) and by["a"].within50 == 1.0
    assert by["b"].median_cents == pytest.approx(50, abs=1) and by["b"].within50 == 1.0  # 경계는 안으로
    assert by["c"].median_cents == pytest.approx(-50, abs=1)
    assert by["d"].vibrato_cents == pytest.approx(60, abs=12) and abs(by["d"].median_cents) < 10  # 비브라토는 오차가 아니다
    assert by["e"].status == "missing"
    assert rep.scored == 4 and rep.missing == 1 and sum(rep.histogram) > 0
    assert "실수로 판정하지 않습니다" in rep.caveat


def test_intonation_portamento_reach_time_and_short_notes():
    notes = [R.ReferenceNote("a", 1.0, 2.0, 64), R.ReferenceNote("s", 2.0, 2.08, 66)]
    curve = _curve_from([(1.0, 2.0, lambda t: np.minimum(62 + t / 0.2 * 2, 64)), (2.0, 2.08, lambda t: 66 + 0 * t)], 3)
    rep = analyse(_ref(notes), curve)
    a = rep.note("a")
    assert a.reach_s == pytest.approx(0.15, abs=0.04)  # 아래에서 끌어올려 목표에 닿기까지
    assert abs(a.median_cents) < 5  # 시작 끌어올림은 오차 통계에서 뺀다
    assert rep.note("s").short and "접근음" in rep.note("s").notes[0]


def test_intonation_alignment_error_and_tempo_change_via_warp():
    notes = [R.ReferenceNote("a", 0.0, 1.0, 60), R.ReferenceNote("b", 1.0, 2.0, 62)]
    # 음원: a는 1~2초, b는 2.5~4초 (템포가 바뀜) → warp로 맞추면 정확
    curve = _curve_from([(1.0, 2.0, lambda t: 60 + 0 * t), (2.5, 4.0, lambda t: 62 + 0 * t)], 5)
    good = analyse(_ref(notes, warp=((0.0, 1.0), (1.0, 2.5), (2.0, 4.0))), curve)
    assert all(n.status == "scored" and abs(n.median_cents) < 2 for n in good.notes)
    bad = analyse(_ref(notes, confirmed=False, offset=1.0), curve)  # 정렬 틀림 + 미확정
    assert bad.note("b").status in ("uncertain", "missing")
    assert any("정렬을 확정하지 않아" in x for n in bad.notes for x in n.notes)
    outside = analyse(_ref(notes, offset=10.0), curve)  # 정렬 범위 밖
    assert all(n.status == "not_scored" for n in outside.notes)


def test_intonation_sustained_high_section_and_extra_voiced():
    notes = [R.ReferenceNote(f"n{i}", 1 + i, 2 + i, 60) for i in range(4)]
    curve = _curve_from([(1 + i, 2 + i, lambda t: 60.4 + 0 * t) for i in range(4)] + [(5.2, 5.6, lambda t: 70 + 0 * t)], 7)
    notes.append(R.ReferenceNote("z", 6.0, 6.5, 60))
    rep = analyse(_ref(notes), curve)
    assert rep.sustained and rep.sustained[0][2] == "high" and rep.sustained[0][3] == pytest.approx(40, abs=2)
    assert rep.extra and rep.extra[0][0] == pytest.approx(5.2, abs=0.02)


# --- 오류 검토 ------------------------------------------------------------------------


def _analysis(curve, notes, artic=None):
    return AnalysisResult(curve.f0_hz.size / FR, TempoMap.constant(120), tuple(notes), pitch_curve=curve,
                          onset_envelope=artic)


def test_review_note_rules_low_conf_missing_and_merged_repeat():
    curve = _curve_from([(1.0, 3.0, lambda t: 60 + 0 * t), (4.0, 5.0, lambda t: 64 + 0 * t)], 6)
    artic = np.zeros(curve.f0_hz.size)
    artic[200] = 2.5  # 2.0초에 재발음 지표 2.5 (분할 기준 4의 절반 이상·미만)
    notes = [note("n1", 1.0, 3.0, 60, confidence=0.15)]
    items = note_rules(notes, _analysis(curve, notes, artic))
    kinds = {i.kind: i for i in items}
    assert kinds["note_low_conf"].severity == "high" and "0.15" in kinds["note_low_conf"].reason
    assert "voiced_uncovered" in kinds and kinds["voiced_uncovered"].start == pytest.approx(4.0, abs=0.02)
    assert "note_repeat_merged" in kinds and "2.00초" in kinds["note_repeat_merged"].reason


def test_review_octave_leap_and_no_vocal_rules():
    notes = [note("a", 0.0, 0.5, 60), note("b", 0.55, 0.9, 72), note("c", 0.95, 1.4, 61), note("d", 3.0, 3.5, 62)]
    items = note_rules(notes, None, no_vocal_ranges=[(2.9, 4.0)], original_ids=frozenset({"a", "b", "c"}))
    kinds = {i.kind for i in items}
    assert "note_octave" in kinds and "sep_novocal_notes" in kinds
    nv = next(i for i in items if i.kind == "sep_novocal_notes")
    assert nv.note_ids == ("d",) and "직접 추가" in nv.reason


def test_review_lyric_and_melisma_rules():
    from amp.lyrics import linking, ops, pronunciation
    from amp.lyrics.model import LyricsState

    notes = [note("n1", 1.0, 1.5, 60), note("n2", 2.0, 2.4, 62)]
    st = ops.apply_text(LyricsState(), "가 나 다")
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {ids[0]: (1.0, 1.5), ids[1]: (1.8, 2.4)}, source="model")
    st = linking.auto_link(st, notes)
    st = pronunciation.sync(st)
    items = lyric_rules(st, {n.id: n for n in notes})
    kinds = {(i.kind, i.syllable_ids) for i in items}
    assert ("syl_unaligned", (ids[2],)) in kinds
    gap = next(i for i in items if i.kind == "onset_gap")
    assert gap.syllable_ids == (ids[1],) and "+200ms" in gap.reason
    assert vocal_rules(st, {}) == []  # 판정 결과가 없으면 멜리스마 항목 없음


def test_review_state_save_restore_and_recheck_after_change():
    curve = _curve_from([(1.0, 2.0, lambda t: 60 + 0 * t)], 3)
    notes = [note("n1", 1.0, 2.0, 60, confidence=0.3)]
    items = note_rules(notes, _analysis(curve, notes))
    low = next(i for i in items if i.kind == "note_low_conf")
    st = mark(ReviewState(), [low], "checked")
    assert st.status_of(low) == "checked"
    back = state_from_dict(state_to_dict(st))
    assert back == st and back.status_of(low) == "checked"
    # 재분석 뒤 같은 음표의 근거가 바뀌면 '재검토 필요'
    notes2 = [note("n1", 1.0, 2.0, 60, confidence=0.25)]
    low2 = next(i for i in note_rules(notes2, _analysis(curve, notes2)) if i.kind == "note_low_conf")
    assert low2.key == low.key and back.status_of(low2) == "recheck"
    assert mark(back, [low2], "open").status_of(low2) == "open"
