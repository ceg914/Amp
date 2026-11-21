"""v0.6 품질 평가: 정답 없는 지표, 대시보드 숫자, 참조 악보 비교(단서 포함), Hybrid 출처, 다른 오디오 실행 비교 거부."""

from __future__ import annotations

import numpy as np
import pytest

from amp.compare import quality as Q
from amp.compare import review as R
from amp.core.model import AnalysisResult, ModelInfo, Note, PitchCurve, Project, Score
from amp.core.pitch import midi_to_hz
from amp.core.tempo import TempoMap
from amp.project import runs


def _project():
    n = 600
    midi = np.full(n, np.nan)
    midi[100:200] = 64.0
    midi[300:400] = 67.0
    f0 = np.where(np.isfinite(midi), midi_to_hz(np.nan_to_num(midi, nan=60)), np.nan).astype(np.float32)
    curve = PitchCurve(0.0, 100.0, f0, np.where(np.isfinite(midi), 0.8, 0.0).astype(np.float32))
    notes = (Note("a", 1.0, 2.0, 64), Note("b", 3.0, 3.5, 67))
    res = AnalysisResult(duration=6.0, tempo=TempoMap.constant(120), notes=notes, pitch_curve=curve,
                         models={"separation": ModelInfo("hybrid"), "notes": ModelInfo("segment", "4")})
    return Project(analysis=res, score=Score.from_analysis(res))


def test_indicators_without_reference_are_not_called_accuracy():
    p = _project()
    rep = Q.assess(p)
    ind = rep.indicators
    assert ind["음표 수"] == 2 and ind["유성 프레임 중 음표가 덮은 비율"] == pytest.approx(0.75)
    assert ind["분리 소스"] == "hybrid" and ind["음표 검출 버전"] == "4"
    assert rep.reference is None and rep.notes == [Q.NO_REFERENCE_NOTE]
    assert not any("정확도" in k for k in ind)


def test_dashboard_counts_follow_review_states():
    p = _project()
    items = [R._item("note_short", "note", 1.0, 1.1, "x", "low", ["a"]),
             R._item("syl_unlinked", "lyrics", 2.0, 2.2, "y", "medium", syllable_ids=["s1"]),
             R._item("mel_uncertain", "melisma", 3.0, 3.2, "z", "low", ["b"], ["s2"])]
    state = R.mark(p.review, [items[2]], "checked")
    rows = R.merge_states(items, state)
    rep = Q.assess(p, rows)
    assert rep.counts["review_notes"][1] == 1 and "note_short" in rep.counts["review_notes"][2]
    assert rep.counts["unlinked"][1] == 1
    assert rep.counts["melisma"][1] == 0 and rep.counts["reviewed"][1] == 1 and rep.counts["total"][1] == 3


def test_hybrid_sources_are_listed():
    from amp.engine import hybrid as H

    p = _project()
    plan = H.single(100, 600, "k1", "htdemucs")
    plan = H.split(plan, plan.segments[0].segment_id, 2.5)
    plan = H.set_source(plan, [plan.segments[1].segment_id], "k2", "mel_roformer")
    p.hybrid = plan
    rep = Q.assess(p, (), {"k1": "Demucs", "k2": "RoFormer"})
    assert rep.sources == [(0.0, 2.5, "Demucs"), (2.5, 6.0, "RoFormer")]


def test_reference_scores_include_caveat():
    from amp.compare.reference import ReferenceNote, ReferenceScore, ScoreAlignment

    p = _project()
    ref = ReferenceScore("x.mid", "sha", "midi", notes=(ReferenceNote("r1", 1.0, 2.0, 64), ReferenceNote("r2", 3.0, 3.5, 66)))
    p.reference = ref.with_alignment(ScoreAlignment())  # 악보 시간 = 음원 시간
    out = Q.reference_scores(p)
    assert out is not None and out["ref_notes"] == 2
    assert out["onset±100ms"]["matched"] == 1 and out["onset±100ms"]["f1"] == pytest.approx(0.5)
    assert "악보는 가수가 실제로 부른" in out["caveat"]
    assert out["pitch_curve_vs_reference"]["notes_with_pitch"] == 2


def test_runs_from_different_audio_are_not_compared():
    p = _project()
    a = runs.record_from_result(p.analysis, "full", quality={"suggestions_pending": 3})
    b = runs.record_from_result(p.analysis, "redetect")
    assert a.quality["notes_version"] == "4" and a.quality["suggestions_pending"] == 3
    runs.compare_runs(a, b)  # 같은 오디오(빈 해시 포함)는 비교한다
    a.audio_sha256, b.audio_sha256 = "aaa", "bbb"
    with pytest.raises(runs.RunCompareError):
        runs.compare_runs(a, b)
    old = runs.RunRecord.from_dict({k: v for k, v in a.to_dict().items() if k != "quality"})
    assert old.quality == {}  # 이전 기록도 읽는다
