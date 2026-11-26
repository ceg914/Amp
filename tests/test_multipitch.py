"""v0.7.0 Feature B: 다중 피치 후보와 성부 추적·편집·저장 (합성 자료, 명세 20.1).

기대값은 개발용 합성 묶음('dev')에서 잰 값보다 조금 낮게 둔 회귀 기준이다. 실제 노래의 정확도가 아니다.
"""

from __future__ import annotations

import numpy as np
import pytest

from amp.core import result_meta as RM
from amp.devtools import multipitch_eval as E
from amp.engine import voices as V
from amp.engine.dsp.multipitch import Candidates, candidates


@pytest.fixture(scope="module")
def dev():
    return {r["case"]: r for r in E.evaluate("dev")}


@pytest.mark.parametrize("case,min_f1", [("단일 목소리", 0.95), ("사인 2개", 0.95), ("두 목소리 3도", 0.95),
                                         ("두 목소리 5도", 0.95), ("세 목소리", 0.93), ("반주 유입", 0.9),
                                         ("성부 교차", 0.9), ("성부 중단", 0.85)])
def test_multi_f0_frame_scores(dev, case, min_f1):
    r = dev[case]
    assert r["f1"] >= min_f1, r
    assert r["cents_median"] is not None and r["cents_median"] < 10


@pytest.mark.parametrize("case,min_acc,voices", [("단일 목소리", 0.97, 1), ("두 목소리 3도", 0.95, 2),
                                                 ("두 목소리 5도", 0.95, 2), ("세 목소리", 0.93, 3),
                                                 ("반주 유입", 0.95, 2), ("성부 교차", 0.85, 2)])
def test_voice_tracking(dev, case, min_acc, voices):
    r = dev[case]
    assert r["voice_accuracy"] >= min_acc and r["voices_found"] == voices, r
    assert r["voice_switches"] <= 2


def test_known_limits_are_reported_not_hidden(dev):
    """옥타브·유니즌은 후보 하나로 합쳐지는 한계가 있다 (보고서에 적는 값). 유니즌은 성부 하나로만 남는다."""
    assert dev["유니즌"]["recall"] <= 0.6 and dev["유니즌"]["voices_found"] == 1
    assert dev["옥타브"]["recall"] < 0.8


def test_not_a_repeated_mono_tracker():
    """두 목소리를 섞은 신호에서 한 프레임에 서로 다른 후보 둘을 함께 찾는다 (단성 모델을 두 번 돌린 결과가 아님)."""
    case = next(c for c in E.cases("dev") if c.name == "두 목소리 5도")
    cand = candidates(case.y16, 2)
    both = np.isfinite(cand.f0_hz).all(axis=1)
    assert both[50:350].mean() > 0.9
    gap = np.abs(1200 * np.log2(cand.f0_hz[both, 0] / cand.f0_hz[both, 1]))
    assert np.median(gap) == pytest.approx(700, abs=30)


def _set(n=300):
    t = np.arange(n)
    f_hi = np.where((t >= 20) & (t < 280), 440.0, np.nan)
    f_lo = np.where((t >= 20) & (t < 280), 330.0, np.nan)
    f0 = np.stack([f_hi, f_lo], 1)
    sal = np.where(np.isfinite(f0), [1.0, 0.7], 0.0)
    c = Candidates(f0, sal, np.where(np.isfinite(f0), 0.9, 0.0), np.ones_like(f0), np.zeros(n))
    return V.analyze(c, RM.make_meta(V.SCHEMA, 1, "a" * 64, input_kind="vocals", input_id="stems:x",
                                     input_hash="b" * 64), 2)


def test_voice_set_editing_ops():
    vs = _set()
    assert [v.voice_track_id for v in vs.voices] == ["v1", "v2"] and vs.voices[0].median_midi() == pytest.approx(69)
    assert vs.voices[0].voiced_intervals() == [(0.2, 2.8)]
    s2 = V.swap_range(vs, "v1", "v2", 1.0, 1.5)
    assert s2.voice("v1").f0_hz[120] == pytest.approx(330) and s2.voice("v2").f0_hz[120] == pytest.approx(440)
    assert vs.voice("v1").f0_hz[120] == pytest.approx(440)  # 원래 묶음은 그대로 (되돌리기용)
    assert s2.voice("v1").edited and s2.history[-1].startswith("v1↔v2")
    s3, conflicts = V.move_range(vs, "v2", "v1", 1.0, 1.5)
    assert conflicts == 51 and s3.voice("v1").f0_hz[120] == pytest.approx(440)  # 겹친 프레임은 옮기지 않음
    s4 = V.clear_range(vs, "v1", 1.0, 1.5)
    assert np.isnan(s4.voice("v1").f0_hz[120]) and np.isfinite(s4.voice("v1").f0_hz[160])
    s5, moved = V.move_range(s4, "v2", "v1", 1.0, 1.5)
    assert moved == 0 and s5.voice("v1").f0_hz[120] == pytest.approx(330) and np.isnan(s5.voice("v2").f0_hz[120])
    s6, n = V.pick_candidate(s4, "v1", 1.0, 1.5, 64.0)  # E4 근처 후보(330Hz)로 다시 고름
    assert n == 51 and s6.voice("v1").f0_hz[120] == pytest.approx(330)
    s7, new = V.split_voice(vs, "v2", 2.0)
    assert np.isnan(s7.voice("v2").f0_hz[250]) and s7.voice(new).f0_hz[250] == pytest.approx(330)
    s8, conf = V.merge_voices(s7, "v2", new)
    assert conf == 0 and s8.voice(new) is None and s8.voice("v2").f0_hz[250] == pytest.approx(330)
    s9, vid = V.add_voice(vs, "코러스")
    s9 = V.set_attr(s9, vid, display_name="코러스 2", role="harmony_candidate", user_confirmed=True)
    v = s9.voice(vid)
    assert v.display_name == "코러스 2" and v.role == "harmony_candidate" and v.user_confirmed and v.edited
    hidden = V.set_attr(vs, "v1", hidden=True)
    assert hidden.voice("v1").hidden and not hidden.voice("v1").edited  # 숨기기는 편집으로 보지 않음
    with pytest.raises(ValueError):
        V.set_attr(vs, "v1", role="lead")


def test_save_load_archive_and_protect_user_edits(tmp_path):
    vs = V.set_attr(_set(), "v2", display_name="화음", user_confirmed=True)
    V.save(tmp_path, vs)
    back = V.load(tmp_path)
    assert back is not None and [v.display_name for v in back.voices] == ["성부 1", "화음"]
    assert back.voice("v2").user_confirmed and back.meta.input_id == "stems:x"
    assert np.allclose(back.voice("v1").f0_hz, vs.voice("v1").f0_hz, equal_nan=True)
    import json

    assert json.loads((tmp_path / V.JSON_NAME).read_text(encoding="utf-8"))["pitch_only"] is True
    new = _set()
    merged = V.merge_reanalysis(back, new)
    names = [v.display_name for v in merged.voices]
    assert names[0] == "화음" and merged.voices[0].user_confirmed  # 사용자가 확정한 성부는 그대로 남는다
    assert len({v.voice_track_id for v in merged.voices}) == len(merged.voices)
    assert V.merge_reanalysis(None, new) is new
    dst = V.archive(tmp_path, "다시 분석")
    assert dst is not None and (dst / V.JSON_NAME).is_file() and (dst / "reason.txt").read_text(encoding="utf-8") == "다시 분석"
    assert V.load(tmp_path) is None and V.load(dst).voice("v2").display_name == "화음"  # 지우지 않고 옮김
    (tmp_path / V.JSON_NAME).write_text("{깨짐", encoding="utf-8")
    assert V.load(tmp_path) is None


def test_band_limited_region_reanalysis_replaces_one_voice():
    """특정 음역만 다시 분석한 후보로 한 성부의 구간을 바꾼다."""
    case = next(c for c in E.cases("dev") if c.name == "두 목소리 3도")
    from amp.engine.dsp.multipitch import candidates as cands

    full = cands(case.y16, 2)
    vs = V.analyze(full, RM.make_meta(V.SCHEMA, 1, "a" * 64), 2)
    low = vs.voices[-1].voice_track_id
    vs = V.clear_range(vs, low, 1.0, 1.5)
    seg = case.y16[100 * 160: 150 * 160 + 1]
    region = cands(seg, 1, fmin=300.0, fmax=360.0)  # 아래 목소리(E4 근처)만
    region.start_frame = 100
    out, n = V.replace_range_from(vs, low, region, 1.0, 1.5)
    assert n > 40
    got = 69 + 12 * np.log2(out.voice(low).f0_hz[110:140] / 440)
    assert np.nanmedian(got) == pytest.approx(64, abs=0.5)
