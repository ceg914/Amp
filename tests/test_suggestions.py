"""v0.6 교정 제안: 근거 기반 6종 제안, 사용자 음표 제외, 적용 계획, 지난 제안, 상태 이어받기, 저장 형식.

피치 곡선·재발음 지표를 직접 만들어 넣는다. 제안은 음표를 바꾸지 않고, plan_apply는 바꿀 음표만 계산한다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from amp.compare import suggestions as S
from amp.core.errors import EditError
from amp.core.model import AnalysisResult, Note, PitchCurve, Project, Score
from amp.core.pitch import midi_to_hz
from amp.core.tempo import TempoMap

FR = 100.0
DUR = 8.0


def _project(segments, notes, art_peaks=(), conf=0.9, onset_len=None):
    """segments: [(시작, 끝, midi)] 유성 구간. notes: [(id, 시작, 끝, pitch)]."""
    n = int(DUR * FR)
    midi = np.full(n, np.nan)
    for a, b, m in segments:
        midi[int(a * FR):int(b * FR)] = m
    f0 = np.where(np.isfinite(midi), midi_to_hz(np.nan_to_num(midi, nan=60.0)), np.nan).astype(np.float32)
    curve = PitchCurve(0.0, FR, f0, np.where(np.isfinite(midi), conf, 0.0).astype(np.float32))
    art = np.zeros(onset_len or n, np.float32)
    for t, v in art_peaks:
        art[int(round(t * FR))] = v
    ns = tuple(Note(i, a, b, p) for i, a, b, p in notes)
    res = AnalysisResult(duration=DUR, tempo=TempoMap.constant(120), notes=ns, pitch_curve=curve, onset_envelope=art)
    return Project(analysis=res, score=Score.from_analysis(res))


def _by_type(sugs, kind):
    return [s for s in sugs if s.suggestion_type == kind]


def test_pitch_suggestion_from_curve_with_evidence():
    p = _project([(1.0, 2.0, 65.0)], [("a", 1.0, 2.0, 64)])
    s = _by_type(S.detect(p), "pitch")
    assert len(s) == 1 and s[0].proposed_changes == {"from": 64, "pitch": 65}
    assert any("피치 곡선 중앙값 65.00" in e for e in s[0].evidence)
    assert 0 <= s[0].score <= 1
    removed, added = S.plan_apply(p, s[0])
    assert [n.id for n in removed] == ["a"] and added[0].pitch == 65 and added[0].start == 1.0
    # 곡선이 음표와 같으면 제안 없음
    assert not _by_type(S.detect(_project([(1.0, 2.0, 64.1)], [("a", 1.0, 2.0, 64)])), "pitch")


def test_split_suggestion_when_pitch_changes_inside_note():
    p = _project([(1.0, 1.5, 60.0), (1.5, 2.0, 62.0)], [("a", 1.0, 2.0, 60)])
    s = _by_type(S.detect(p), "split")
    assert len(s) == 1
    at = s[0].proposed_changes["at"]
    assert abs(at - 1.5) <= 0.05 and s[0].proposed_changes["pitches"] == [60, 62]
    removed, added = S.plan_apply(p, s[0])
    assert [(round(n.start, 2), round(n.end, 2), n.pitch) for n in added] == [(1.0, round(at, 2), 60), (round(at, 2), 2.0, 62)]


def test_split_suggestion_from_rearticulation_below_auto_threshold():
    p = _project([(1.0, 2.0, 64.0)], [("a", 1.0, 2.0, 64)], art_peaks=[(1.5, 3.0)])
    s = _by_type(S.detect(p, split_strength=4.0), "split")
    assert len(s) == 1 and s[0].proposed_changes["at"] == pytest.approx(1.5)
    assert any("재발음 지표 3.0" in e and "4.0" in e for e in s[0].evidence)
    # 기준의 60% 미만이면 제안하지 않는다
    assert not _by_type(S.detect(_project([(1.0, 2.0, 64.0)], [("a", 1.0, 2.0, 64)], art_peaks=[(1.5, 2.0)])), "split")


def test_merge_and_approach_suggestions():
    p = _project([(1.0, 2.0, 64.0)], [("a", 1.0, 1.5, 64), ("b", 1.5, 2.0, 64)])
    m = _by_type(S.detect(p), "merge")
    assert len(m) == 1 and set(m[0].target_note_ids) == {"a", "b"}
    _, added = S.plan_apply(p, m[0])
    assert len(added) == 1 and (added[0].start, added[0].end) == (1.0, 2.0)
    # 뒤 음 시작에 재발음 근거가 있으면 합치자고 하지 않는다
    p2 = _project([(1.0, 2.0, 64.0)], [("a", 1.0, 1.5, 64), ("b", 1.5, 2.0, 64)], art_peaks=[(1.5, 3.0)])
    assert not _by_type(S.detect(p2), "merge")
    # 접근음: 0.15초 동안 62 → 63.5로 올라가다 64
    n = int(DUR * FR)
    segs = [(3.0, 3.15, None), (3.15, 4.0, 64.0)]
    proj = _project([(3.15, 4.0, 64.0)], [("s", 3.0, 3.15, 62), ("t", 3.15, 4.0, 64)])
    midi = proj.analysis.pitch_curve.midi().astype(float)
    midi[300:315] = np.linspace(62.0, 63.6, 15)
    f0 = np.where(np.isfinite(midi), midi_to_hz(np.nan_to_num(midi, nan=60)), np.nan).astype(np.float32)
    curve = PitchCurve(0.0, FR, f0, np.where(np.isfinite(midi), 0.9, 0.0).astype(np.float32))
    import dataclasses

    proj.analysis = dataclasses.replace(proj.analysis, pitch_curve=curve)
    a = _by_type(S.detect(proj), "approach")
    assert len(a) == 1 and a[0].proposed_changes["pitch"] == 64 and n
    _, added = S.plan_apply(proj, a[0])
    assert len(added) == 1 and added[0].pitch == 64 and added[0].start == 3.0
    del segs


def test_missing_and_boundary_suggestions():
    p = _project([(1.0, 2.0, 64.0), (3.0, 3.4, 67.0)], [("a", 1.0, 2.3, 64)])
    sugs = S.detect(p)
    miss = _by_type(sugs, "missing")
    assert len(miss) == 1 and miss[0].proposed_changes["pitch"] == 67
    assert miss[0].proposed_changes["start"] == pytest.approx(3.0) and miss[0].target_note_ids == ()
    removed, added = S.plan_apply(p, miss[0])
    assert removed == [] and added[0].pitch == 67
    b = _by_type(sugs, "boundary")
    assert len(b) == 1 and b[0].proposed_changes["end"] == pytest.approx(2.0)  # 끝 0.3초 무성
    _, added = S.plan_apply(p, b[0])
    assert added[0].end == pytest.approx(2.0) and added[0].start == 1.0


def test_user_edited_notes_are_never_targeted():
    p = _project([(1.0, 2.0, 65.0), (3.0, 4.0, 60.0)], [("a", 1.0, 2.0, 64), ("b", 3.0, 4.0, 61)])
    p.score.replace_notes(["a"], [p.score.get("a").with_(end=1.9)])  # 사용자가 고침 (수정됨)
    sugs = S.detect(p)
    assert all("a" not in s.target_note_ids for s in sugs)
    assert any("b" in s.target_note_ids for s in sugs)


def test_outdated_after_edit_and_cannot_apply():
    p = _project([(1.0, 2.0, 65.0)], [("a", 1.0, 2.0, 64)])
    s = _by_type(S.detect(p), "pitch")[0]
    state = S.SuggestionState((s,))
    p.score.replace_notes(["a"], [p.score.get("a").with_(start=1.05)])
    assert S.is_outdated(s, p)
    assert S.refresh(state, p).items[0].status == S.OUTDATED
    with pytest.raises(EditError):
        S.plan_apply(p, s)


def test_redetect_keeps_rejected_and_dismissed():
    p = _project([(1.0, 2.0, 65.0), (3.0, 4.0, 62.0)], [("a", 1.0, 2.0, 64), ("b", 3.0, 4.0, 61)])
    first = S.SuggestionState(tuple(S.detect(p)))
    ids = [s.suggestion_id for s in first.items]
    state = first.with_status([ids[0]], S.REJECTED).with_status([ids[1]], S.DISMISSED)
    again = S.merge_detected(state, S.detect(p))
    st = {s.suggestion_id: s.status for s in again.items}
    assert st[ids[0]] == S.REJECTED and st[ids[1]] == S.DISMISSED
    assert S.detect(p)[0].suggestion_id == first.items[0].suggestion_id  # 같은 상황이면 같은 id


def test_roundtrip_and_score_is_not_called_probability():
    p = _project([(1.0, 2.0, 65.0)], [("a", 1.0, 2.0, 64)])
    state = S.SuggestionState(tuple(S.detect(p)), "2026-11-21T00:00:00+00:00")
    d = json.loads(json.dumps(S.to_dict(state)))
    item = d["items"][0]
    for k in ("suggestion_id", "suggestion_type", "target_note_ids", "start_time", "end_time", "proposed_changes",
              "evidence", "confidence_or_score", "status", "detector_version"):
        assert k in item
    assert "확률이 아닙니다" in d["score_note"]
    assert S.from_dict(d) == state


def test_no_analysis_no_suggestions():
    assert S.detect(Project()) == []


def test_detection_is_fast_on_song_sized_input():
    import time

    rng = np.random.default_rng(0)
    segs, notes = [], []
    t = 0.2
    for i in range(400):  # 400음표 ≈ 7분 곡 수준
        d = float(rng.uniform(0.15, 0.6))
        m = 60 + int(rng.integers(0, 12))
        segs.append((t, t + d, m + float(rng.normal(0, 0.2))))
        notes.append((f"n{i}", t, t + d, m))
        t += d + 0.02
    global DUR
    old = DUR
    DUR = t + 1
    try:
        p = _project(segs, notes)
        t0 = time.perf_counter()
        S.detect(p)
        dt = time.perf_counter() - t0
    finally:
        DUR = old
    assert dt < 3.0, f"{dt:.2f}s"
