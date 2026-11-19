"""STEP 4 음절 길이·멜리스마 판정 시험. 피치 곡선·재발음 지표는 정답을 아는 합성 데이터로 만든다."""

import json

import numpy as np
import pytest

from amp.core.model import PitchCurve, Vibrato
from amp.core.pitch import midi_to_hz
from amp.lyrics import io, linking, ops, vocal
from amp.lyrics.model import LyricsState, NoteLink, SyllableTiming
from tests.conftest import note

FR = 100.0


def curve(segments, total=6.0):
    """[(시작, 끝, MIDI 또는 (시작음, 끝음))] → 100fps 피치 곡선. 구간 밖은 무성(NaN)."""
    n = int(total * FR)
    f0 = np.full(n, np.nan, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    for a, b, p in segments:
        i, j = int(round(a * FR)), int(round(b * FR))
        if isinstance(p, tuple):
            m = np.linspace(p[0], p[1], j - i)
        else:
            m = np.full(j - i, float(p))
        f0[i:j] = midi_to_hz(m)
        conf[i:j] = 0.9
    return PitchCurve(0.0, FR, f0, conf)


def onset_at(times, total=6.0, value=6.0):
    o = np.zeros(int(total * FR))
    for t in times:
        o[int(round(t * FR))] = value
    return o


def timed(text, spans, notes, links=None):
    st = ops.apply_text(LyricsState(), text)
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {ids[i]: sp for i, sp in enumerate(spans)}, source="model")
    if links is None:
        st = linking.auto_link(st, notes)
    else:
        st = st.__class__(st.text, st.alignment.with_links({ids[i]: [NoteLink(n, 0.9, "user", True) for n in ns]
                                                            for i, ns in links.items()}), st.recognition, st.history)
    return st, ids


# --- 음절 길이 ----------------------------------------------------------------------

def test_single_and_long_vowel_syllable_lengths():
    n = [note("a", 1.0, 3.0, 64)]
    st, ids = timed("아", [(1.0, 3.0)], n)
    d = vocal.syllable_duration(st, ids[0], {x.id: x for x in n}, curve([(1.0, 3.0, 64)]))
    assert d.span == pytest.approx(2.0) and d.voiced == pytest.approx(2.0, abs=0.011)
    assert d.span_status == "measured"  # 시작·끝 모두 검출값
    assert d.note_union == pytest.approx(2.0)
    # 실제 MMS 정렬은 끝을 다음 음절·보컬 끊김으로 추정한다(inferred) → '추정'
    t = st.alignment.timing(ids[0])
    est = st.__class__(st.text, st.alignment.with_timings({ids[0]: SyllableTiming(t.start, t.end, 0.8, "model", "inferred")}))
    assert vocal.syllable_duration(est, ids[0], {}, None).span_status == "estimated"


def test_unvoiced_consonant_and_gaps_are_not_voiced_time():
    # '사' 앞 0.1초는 무성 자음(ㅅ), 두 음절 사이 0.2초 공백
    c = curve([(1.1, 1.6, 64), (1.8, 2.4, 65)])
    st, ids = timed("사랑", [(1.0, 1.6), (1.8, 2.4)], [note("a", 1.1, 1.6, 64), note("b", 1.8, 2.4, 65)])
    ds = vocal.all_durations(st, {n.id: n for n in [note("a", 1.1, 1.6, 64), note("b", 1.8, 2.4, 65)]}, c)
    assert ds[ids[0]].span == pytest.approx(0.6) and ds[ids[0]].voiced == pytest.approx(0.5, abs=0.011)
    assert ds[ids[1]].voiced == pytest.approx(0.6, abs=0.011)  # 공백은 어느 음절에도 들어가지 않는다


def test_overlapping_notes_are_counted_once():
    ns = [note("a", 1.0, 1.6, 64), note("b", 1.4, 2.0, 66)]
    st, ids = timed("아", [(1.0, 2.0)], ns, links={0: ["a", "b"]})
    d = vocal.syllable_duration(st, ids[0], {n.id: n for n in ns}, curve([(1.0, 2.0, 64)]))
    assert d.note_union == pytest.approx(1.0)  # 0.6 + 0.6이 아니라 겹친 0.2초를 한 번만


def test_unaligned_invalid_user_and_unmeasured():
    st, ids = timed("가나다", [(1.0, 1.5)], [])
    c = curve([(1.0, 1.5, 60)])
    assert vocal.syllable_duration(st, ids[1], {}, c).span_status == "unaligned"
    assert vocal.syllable_duration(st, ids[1], {}, c).voiced is None
    # 잘못된 시간 (파일을 손으로 고친 경우 등): 정렬 편집 함수를 거치지 않고 넣어 본다
    bad = st.__class__(st.text, st.alignment.with_timings({ids[2]: SyllableTiming(2.0, 1.9, None, "model", "model")}))
    d = vocal.syllable_duration(bad, ids[2], {}, c)
    assert d.span_status == "invalid" and d.span is None and d.voiced is None
    # 사용자가 시간을 고치면 '검출' 상태 (추정 아님)
    st2 = ops.set_times(st, {ids[0]: (1.0, 1.4)}, source="user")
    d = vocal.syllable_duration(st2, ids[0], {}, c)
    assert d.span_status == "measured" and d.span == pytest.approx(0.4)
    # 피치가 없거나 유성음이 없으면 0이 아니라 미측정
    assert vocal.syllable_duration(st, ids[0], {}, None).voiced is None
    silent = curve([], total=3.0)
    d = vocal.syllable_duration(st, ids[0], {}, silent)
    assert d.voiced is None and "0으로 채우지 않음" in d.voiced_reason
    # 보컬 없는 구간으로 지정한 곳은 빼고 잰다
    ds = vocal.all_durations(st, {}, c, no_vocal_ranges=[(1.0, 1.5)])
    assert ds[ids[0]].voiced is None


def test_stale_timing_is_shown_as_estimate():
    st, ids = timed("가나", [(1.0, 1.5), (1.5, 2.0)], [])
    st2 = ops.apply_text(st, "가너")
    d = vocal.syllable_duration(st2, st2.text.syllables[1].id, {}, None)
    assert d.span_status == "stale"


# --- 멜리스마 -------------------------------------------------------------------------

def _analyze(st, notes, c=None, onset=None):
    return vocal.analyze(st, notes, c, onset)


def test_one_note_is_single():
    ns = [note("a", 1.0, 2.0, 64)]
    st, ids = timed("아", [(1.0, 2.0)], ns)
    g = _analyze(st, ns, curve([(1.0, 2.0, 64)]))[ids[0]]
    assert g.kind == "single_note" and not g.is_melisma and g.note_count == 1


def test_three_different_notes_are_melisma_with_sequence_range_direction():
    ns = [note("a", 1.0, 1.4, 60), note("b", 1.4, 1.8, 62), note("c", 1.8, 2.4, 64), note("d", 2.4, 2.9, 62)]
    st, ids = timed("아", [(1.0, 2.9)], ns, links={0: ["a", "b", "c", "d"]})
    c = curve([(1.0, 1.4, 60), (1.4, 1.8, 62), (1.8, 2.4, 64), (2.4, 2.9, 62)])
    g = _analyze(st, ns, c, onset_at([]))[ids[0]]
    assert g.kind == "melisma" and g.is_melisma and g.source == "auto"
    assert g.pitch_sequence == ("C4", "D4", "E4", "D4") and g.pitch_range == 4 and g.direction == "mixed"
    rec = g.melisma_record()
    assert rec["note_count"] == 4 and rec["start_time"] == 1.0 and rec["end_time"] == 2.9
    assert rec["duration"] == pytest.approx(1.9) and rec["detection_source"].startswith(vocal.DETECTOR)
    assert set(rec) >= {"melisma_id", "syllable_id", "linked_note_ids", "pitch_sequence", "pitch_range",
                        "pitch_direction", "confidence", "is_user_confirmed"}
    up = _analyze(*timed("아", [(1.0, 2.4)], ns[:3], links={0: ["a", "b", "c"]})[0:1], ns[:3], c, onset_at([]))
    assert next(iter(up.values())).direction == "ascending"


def test_same_pitch_repeat_vs_oversplit():
    ns = [note("a", 1.0, 1.5, 64), note("b", 1.5, 2.0, 64)]
    st, ids = timed("아", [(1.0, 2.0)], ns, links={0: ["a", "b"]})
    c = curve([(1.0, 2.0, 64)])
    # 경계에 재발음 지표가 있으면 같은 음 반복
    assert _analyze(st, ns, c, onset_at([1.5]))[ids[0]].kind == "repeated_note"
    # 무성 틈으로 다시 부른 것도 반복
    c_gap = curve([(1.0, 1.46, 64), (1.52, 2.0, 64)])
    assert _analyze(st, ns, c_gap, onset_at([]))[ids[0]].kind == "repeated_note"
    # 근거가 없으면 잘못 나뉜 음표일 수 있어 불확실
    g = _analyze(st, ns, c, onset_at([]))[ids[0]]
    assert g.kind == "uncertain" and "possible_oversplit" in g.flags and not g.is_melisma
    # 재발음 지표가 아예 없으면 판단 못 함
    assert "no_reartic_info" in _analyze(st, ns, c, None)[ids[0]].flags


def test_portamento_is_a_slide_not_a_melisma():
    ns = [note("a", 1.0, 1.08, 60), note("b", 1.08, 1.16, 62), note("c", 1.16, 1.24, 64), note("d", 1.24, 2.0, 66)]
    st, ids = timed("아", [(1.0, 2.0)], ns, links={0: ["a", "b", "c", "d"]})
    c = curve([(1.0, 1.24, (60, 65.5)), (1.24, 2.0, 66)])
    g = _analyze(st, ns, c, onset_at([]))[ids[0]]
    assert g.kind == "pitch_slide" and not g.is_melisma and g.direction == "ascending"


def test_short_approach_note_is_not_counted():
    ns = [note("a", 1.0, 1.12, 62), note("b", 1.12, 1.8, 64)]
    st, ids = timed("가", [(1.0, 1.8)], ns)  # auto_link가 접근음 역할을 붙인다 (STEP 3.5)
    assert any(lk.role == "approach" for lk in st.alignment.links[ids[0]])
    g = _analyze(st, ns, curve([(1.0, 1.12, (62, 63.5)), (1.12, 1.8, 64)]), onset_at([]))[ids[0]]
    assert g.kind == "single_note" and g.approach_ids == ("a",) and g.note_ids == ("b",) and "approach" in g.flags


def test_clear_ornament_is_decoration_of_single_note():
    ns = [note("o", 1.0, 1.04, 66, ornament=True, confidence=0.9), note("b", 1.04, 1.8, 64)]
    st, ids = timed("가", [(1.0, 1.8)], ns, links={0: ["o", "b"]})
    g = _analyze(st, ns, curve([(1.0, 1.04, 66), (1.04, 1.8, 64)]), onset_at([]))[ids[0]]
    assert g.kind == "single_note" and "ornament" in g.flags and g.ornament_ids == ("o",)


def test_vibrato_long_note_and_split_vibrato():
    ns = [note("a", 1.0, 3.0, 64, vibrato=Vibrato(6.0, 60.0))]
    st, ids = timed("아", [(1.0, 3.0)], ns)
    assert _analyze(st, ns, curve([(1.0, 3.0, 64)]))[ids[0]].kind == "single_note"
    # 비브라토가 위아래 1반음씩 짧은 음표로 쪼개진 경우: 멜리스마로 세지 않고 불확실
    ps = [64, 65, 64, 65, 64]
    ns2 = [note(f"v{i}", 1.0 + i * 0.16, 1.16 + i * 0.16, p) for i, p in enumerate(ps)]
    st2, ids2 = timed("아", [(1.0, 1.8)], ns2, links={0: [n.id for n in ns2]})
    g = _analyze(st2, ns2, curve([(1.0, 1.8, 64.5)]), onset_at([]))[ids2[0]]
    assert g.kind == "uncertain" and "vibrato_split" in g.flags


def test_missing_note_leaves_voiced_part_uncovered():
    ns = [note("a", 1.0, 1.5, 60)]  # 1.5~2.0의 다른 음은 검출되지 않음
    st, ids = timed("아", [(1.0, 2.0)], ns)
    g = _analyze(st, ns, curve([(1.0, 1.5, 60), (1.5, 2.0, 63)]))[ids[0]]
    assert "uncovered_voicing" in g.flags and g.confidence < 0.9


def test_new_syllable_inside_or_shared_note_makes_it_uncertain():
    ns = [note("a", 1.0, 1.5, 60), note("b", 1.5, 2.0, 64)]
    st, ids = timed("가나", [(1.0, 2.0), (1.48, 2.0)], ns, links={0: ["a", "b"], 1: ["b"]})
    c = curve([(1.0, 2.0, 60)])
    g = _analyze(st, ns, c, onset_at([]))
    assert g[ids[0]].kind == "uncertain" and {"new_syllable_inside", "shared_note"} <= set(g[ids[0]].flags)
    # 여러 음절이 음표 하나: 각 음절은 단일 음, '나눠 씀' 표시
    one = [note("x", 1.0, 2.0, 62)]
    st2, ids2 = timed("사랑", [(1.0, 1.5), (1.5, 2.0)], one, links={0: ["x"], 1: ["x"]})
    g2 = _analyze(st2, one, curve([(1.0, 2.0, 62)]))
    assert all(g2[i].kind == "single_note" and "shared_note" in g2[i].flags for i in ids2)


def test_user_confirmed_melisma_survives_note_changes_save_and_reload():
    ns = [note("a", 1.0, 1.5, 60), note("b", 1.5, 2.0, 60)]
    st, ids = timed("아", [(1.0, 2.0)], ns, links={0: ["a", "b"]})
    by = {n.id: n for n in ns}
    # 같은 음이라 자동은 '불확실'이지만 사용자가 멜리스마로 확정
    st = vocal.set_melisma(st, ids[0], ["a", "b"], by)
    g = _analyze(st, ns, curve([(1.0, 2.0, 60)]), onset_at([]))[ids[0]]
    assert g.is_melisma and g.source == "user" and g.confirmed and g.kind == "uncertain"
    # 음표만 다시 검출(같은 자리, 새 id) → 판정이 새 음표로 옮겨 가고 재확인은 필요 없다
    a2, b2 = note("a2", 1.0, 1.5, 60), note("b2", 1.5, 2.0, 60)
    st2 = linking.reconcile(st, ns, [a2, b2])
    d = st2.alignment.melisma[ids[0]]
    assert d.note_ids == ("a2", "b2") and not d.needs_review
    # 음표 하나가 둘로 나뉘면 판정은 남기되 재확인 필요
    b3, b4 = note("b3", 1.5, 1.75, 60), note("b4", 1.75, 2.0, 61)
    st3 = linking.reconcile(st2, [b2], [b3, b4])
    d3 = st3.alignment.melisma[ids[0]]
    assert d3.kind == "melisma" and d3.note_ids == ("a2", "b3", "b4") and d3.needs_review
    g3 = _analyze(st3, [a2, b3, b4], curve([(1.0, 2.0, 60)]), onset_at([]))[ids[0]]
    assert g3.needs_review and "user_review" in g3.flags
    # 저장·다시 읽기 (alignment.json v3)
    back = io.alignment_from_dict(json.loads(json.dumps(io.alignment_to_dict(st3))), [])
    assert back.melisma == st3.alignment.melisma
    # 파일에 없는 음표를 가리키면 불러올 때 빼고 재확인 표시
    pruned, _ = linking.prune(st3, ["a2", "b3"])
    assert pruned.alignment.melisma[ids[0]].note_ids == ("a2", "b3") and pruned.alignment.melisma[ids[0]].needs_review


def test_user_release_add_remove_and_clear():
    ns = [note("a", 1.0, 1.4, 60), note("b", 1.4, 1.8, 62), note("c", 1.8, 2.4, 64), note("x", 2.4, 2.8, 65)]
    by = {n.id: n for n in ns}
    st, ids = timed("아", [(1.0, 2.4)], ns[:3], links={0: ["a", "b", "c"]})
    c = curve([(1.0, 2.8, 62)])
    auto = _analyze(st, ns, c, onset_at([]))[ids[0]]
    assert auto.is_melisma
    off = vocal.set_not_melisma(st, ids[0])
    g = _analyze(off, ns, c, onset_at([]))[ids[0]]
    assert not g.is_melisma and g.user_kind == "not_melisma" and g.kind == "melisma"  # 자동 판정은 참고로 남는다
    # 음표 추가: 연결 안 된 음표는 확정 연결로 함께 잇는다
    plus = vocal.add_notes(st, ids[0], ["x"], by, auto)
    assert plus.alignment.melisma[ids[0]].note_ids == ("a", "b", "c", "x")
    assert any(lk.note_id == "x" and lk.confirmed for lk in plus.alignment.links[ids[0]])
    # 음표 빼기 (연결은 그대로), 2개 미만이면 멜리스마 아님
    minus = vocal.remove_notes(plus, ids[0], ["x", "c"], by, None)
    assert minus.alignment.melisma[ids[0]].note_ids == ("a", "b")
    one = vocal.remove_notes(minus, ids[0], ["b"], by, None)
    assert one.alignment.melisma[ids[0]].kind == "not_melisma"
    back = vocal.clear_decision(one, [ids[0]])
    assert ids[0] not in back.alignment.melisma
    with pytest.raises(ValueError):
        vocal.set_melisma(st, ids[0], ["a"], by)


def test_analysis_never_changes_notes_or_lyrics():
    ns = [note("a", 1.0, 1.4, 60), note("b", 1.4, 1.8, 62)]
    st, ids = timed("아", [(1.0, 1.8)], ns, links={0: ["a", "b"]})
    before = (st, list(ns))
    vocal.analyze(st, ns, curve([(1.0, 1.8, 61)]), onset_at([]))
    vocal.all_durations(st, {n.id: n for n in ns}, curve([(1.0, 1.8, 61)]))
    assert (st, list(ns)) == before
