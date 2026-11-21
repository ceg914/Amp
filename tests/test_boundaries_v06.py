"""v0.6 Phase 3: 음표 경계 (검출 v4).

- 재발음 지표에 유성 확률 하락 근거를 더한다: 음량 하락이 얕은(-7dB) 같은 음 재발음을 나눈다 (v3는 못 나눔).
- 프레이즈 시작(무성 → 유성)은 하락으로 보지 않는다: 끌어올림은 계속 한 음으로 합친다.
- 비브라토·트레몰로 긴 음은 여전히 한 음이다. 모든 합성 사례에서 v3보다 나빠지지 않는다.
- 피치 곡선은 바꾸지 않는다.
- 음 끝 잔향 자르기는 실험 기능(기본 끔): 끝 시각을 아는 잔향 합성 사례에서 오차가 준다. 줄이기만 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")

from amp.core.model import Note  # noqa: E402
from amp.devtools import benchmark as B  # noqa: E402
from amp.devtools.evaluation import score_notes  # noqa: E402
from amp.engine import notes as N  # noqa: E402
from amp.engine import pitch_post as P  # noqa: E402
from amp.engine.boundaries import trim_release  # noqa: E402
from amp.engine.dsp.pitch import track_pitch  # noqa: E402
from amp.engine.settings import AnalysisSettings  # noqa: E402

CASES = {c.name: c for c in B.CASES + B.HELDOUT_CASES + B.APPROACH_CASES}


@pytest.fixture(scope="module")
def frames():
    out = {}
    for name in ("h_dip7", "h_dip10", "h_tremolo", "h_tremolo_reverb", "a_scoop_glide", "a_scoop_vibrato",
                 "vibrato_wide", "h_repeat_reverb", "scale", "repeat_gap"):
        c = CASES[name]
        out[name] = (c, track_pitch(B.render(c), B.SR))
    return out


def _detect(f, v4: bool, sens: str = "standard"):
    curve = P.build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate, rms_fine=f.rms_fine)
    art = P.articulation_envelope(f.onset, f.rms, f.frame_rate, rms_fine=f.rms_fine,
                                  voiced_prob=f.voiced_prob if v4 else None, f0_hz=f.f0_hz)
    return N.detect_notes(curve, art, AnalysisSettings.for_sensitivity(sens).note_params()), curve


def test_shallow_dip_rearticulation_is_split_by_v4_only(frames):
    c, f = frames["h_dip7"]
    v3, _ = _detect(f, False)
    v4, _ = _detect(f, True)
    assert score_notes(c.reference(), v3).f1 < 0.5  # v3: 음량 -7dB 하락은 근거가 약해 한 음
    s = score_notes(c.reference(), v4)
    assert s.f1 == pytest.approx(1.0) and len(v4) == 4


@pytest.mark.parametrize("name", ["h_tremolo", "h_tremolo_reverb", "vibrato_wide", "a_scoop_glide", "a_scoop_vibrato"])
def test_v4_keeps_long_notes_and_scoops_whole(frames, name):
    c, f = frames[name]
    v4, _ = _detect(f, True)
    assert len(v4) == len(c.reference()), [(round(n.start, 2), n.pitch) for n in v4]
    assert score_notes(c.reference(), v4).f1 == pytest.approx(1.0)


def test_phrase_start_is_not_a_voicing_dip():
    """무성 → 유성 전환(프레이즈 시작)에서는 유성 근거가 0, 유성 사이 짧은 하락에서는 1."""
    fr = 100.0
    vp = np.r_[np.zeros(50), np.full(100, 0.95), np.full(3, 0.2), np.full(100, 0.95)]
    onset = np.zeros_like(vp)
    onset[50] = onset[151] = 5.0
    rms = np.full_like(vp, 0.1)  # 음량 하락 없음
    art3 = P.articulation_envelope(onset, rms, fr, rms_fine=rms)
    art4 = P.articulation_envelope(onset, rms, fr, rms_fine=rms, voiced_prob=vp, f0_hz=np.full_like(vp, 220.0))
    other = np.r_[np.full(152, 220.0), np.full(101, 247.0)]  # 하락 뒤 높이가 바뀜 (2반음): 유성 근거를 쓰지 않는다
    assert P.articulation_envelope(onset, rms, fr, rms_fine=rms, voiced_prob=vp, f0_hz=other)[151] == 0
    assert art3[151] == 0 and art4[151] == pytest.approx(5.0)
    assert art4[50] == 0  # 앞쪽이 무성: 하락이 아니다


def test_v4_not_worse_than_v3_on_all_synthetic_cases():
    worse = []
    for c in CASES.values():
        f = track_pitch(B.render(c), B.SR)
        if c.extra == "octave":
            f.f0_hz = B._inject_octave_errors(f.f0_hz)
        a = score_notes(c.reference(), _detect(f, False)[0]).f1
        b = score_notes(c.reference(), _detect(f, True)[0]).f1
        if b < a - 1e-9:
            worse.append((c.name, a, b))
    assert not worse


def test_detection_does_not_modify_pitch_curve(frames):
    c, f = frames["h_dip7"]
    curve = P.build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate, rms_fine=f.rms_fine)
    before = curve.f0_hz.copy()
    art = P.articulation_envelope(f.onset, f.rms, f.frame_rate, rms_fine=f.rms_fine, voiced_prob=f.voiced_prob,
                                  f0_hz=f.f0_hz)
    N.detect_notes(curve, art, AnalysisSettings().note_params())
    assert np.array_equal(np.nan_to_num(before), np.nan_to_num(curve.f0_hz))


def test_approach_cap_default_is_v3_fixed_value():
    assert N.APPROACH_TARGET_RATIO is None
    assert N.approach_cap(200, 100.0) == N.APPROACH_MAX_S
    old = N.APPROACH_TARGET_RATIO
    try:
        N.APPROACH_TARGET_RATIO = 0.5
        assert N.approach_cap(20, 100.0) == N.APPROACH_CAP_MIN_S  # 0.2s 뒤 음 → 0.1 → 최소 0.12
        assert N.approach_cap(300, 100.0) == N.APPROACH_CAP_MAX_S
    finally:
        N.APPROACH_TARGET_RATIO = old


def test_trim_release_shortens_reverb_tails_only(frames):
    c, f = frames["h_repeat_reverb"]
    notes, _ = _detect(f, True)
    trimmed = trim_release(notes, f.rms_fine, f.frame_rate, drop_db=10)
    before = score_notes(c.reference(), notes).offset_mae_ms
    after = score_notes(c.reference(), trimmed).offset_mae_ms
    assert after < before
    assert all(t.end <= n.end and t.start == n.start and t.pitch == n.pitch for n, t in zip(notes, trimmed))
    assert all(t.end - t.start >= min(0.08, (n.end - n.start) / 2) - 1e-9 for n, t in zip(notes, trimmed))
    # 잔향 없는 사례는 거의 그대로 (20ms 이내)
    c2, f2 = frames["scale"]
    n2, _ = _detect(f2, True)
    t2 = trim_release(n2, f2.rms_fine, f2.frame_rate, drop_db=10)
    assert max(n.end - t.end for n, t in zip(n2, t2)) <= 0.02 + 1e-9
    assert trim_release([Note("a", 0.0, 0.03, 60)], f2.rms_fine, 100.0)[0].end == 0.03  # 너무 짧은 음은 그대로


def test_pipeline_records_notes_version_4():
    from amp.engine.transcription import NOTES_VERSION

    assert NOTES_VERSION == "4"
