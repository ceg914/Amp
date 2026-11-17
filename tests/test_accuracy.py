"""STEP 2.5 정확도 시험: 합성 음원 전 과정(pYIN → 후처리 → 음표 검출)과 변경 전후(v1↔v2) 비교.

합성 음원은 성대 펄스열 + 포먼트 필터로 만든 모음이다(amp.devtools.synth). 실제 노래·보컬 분리 품질을
대신하지 않는다. 허용 오차는 아래 TOL에 모았고 보고서(STEP_2_5_VALIDATION_REPORT.md)에도 같은 값을 적었다.

    음표 짝짓기: 시작 시각 ±50ms 안, 정수 음높이 일치 (mir_eval 기본값과 같은 기준)
    F1 ≥ 0.9, 시작 오차 평균 ≤ 20ms, 끝 오차 평균 ≤ 30ms
    피치: 정답과 같은 프레임의 중앙 오차 ≤ 10cent, 50cent 이내 비율 ≥ 0.95, 옥타브 오류율 0
    무성 오경보(정답이 무음인 프레임에 피치가 남은 비율) ≤ 0.06
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")

from amp.devtools import benchmark as B
from amp.devtools import synth
from amp.devtools.evaluation import score_notes
from amp.engine.dsp.pitch import FRAME_RATE, onset_envelope, track_pitch
from amp.engine.notes import detect_notes
from amp.engine.pitch_post import articulation_envelope, build_curve, gap_gate
from amp.engine.settings import AnalysisSettings

TOL = {
    "f1_min": 0.9,
    "onset_mae_ms": 20.0,
    "offset_mae_ms": 30.0,
    "median_cents": 10.0,
    "rpa_50c_min": 0.95,
    "octave_error_rate": 0.0,
    "voicing_false_alarm": 0.06,
}
# 넓은 비브라토(±80cent, 5.5Hz)는 프레임(10ms)마다 최대 28cent 움직인다. 반 프레임 시각 차이만으로
# 약 14cent가 생기므로 이 사례의 피치 중앙 오차 한도는 20cent로 둔다.
TOL_CASE = {"vibrato_wide": {"median_cents": 20.0}}


@pytest.fixture(scope="module")
def bench():
    """사례 8개 × 민감도 3개 × 버전 2개. pYIN은 사례마다 한 번만 돌린다."""
    rows = B.evaluate(B.CASES, ("simple", "standard", "precise"))
    return {(r["case"], r["sensitivity"], r["version"]): r for r in rows}


# --- 요구 시험 1~8: v2가 허용 오차 안에 드는가 ---------------------------------------

@pytest.mark.parametrize("sens", ["standard", "precise"])
@pytest.mark.parametrize("case", [c.name for c in B.CASES])
def test_v2_within_tolerance(bench, case, sens):
    r = bench[(case, sens, "v2")]
    tol = {**TOL, **TOL_CASE.get(case, {})}
    assert r["f1"] >= tol["f1_min"], r["notes"]
    assert r["onset_mae_ms"] <= tol["onset_mae_ms"]
    assert r["offset_mae_ms"] <= tol["offset_mae_ms"]
    assert r["median_abs_cents"] <= tol["median_cents"]
    assert r["rpa_50c"] >= tol["rpa_50c_min"]
    assert r["octave_error_rate"] <= tol["octave_error_rate"]
    assert r["voicing_false_alarm"] <= tol["voicing_false_alarm"]


def test_1_single_frequency_note():
    """일정 주파수(A3 220Hz 배음 톤 1.5초) → 음표 1개, 피치 오차 10cent 이내, 경계 오차 20/30ms 이내."""
    t = np.arange(int(synth.SR * 3.0)) / synth.SR
    y = sum(0.3 / k * np.sin(2 * np.pi * 220 * k * t) for k in range(1, 8))
    y = np.where((t >= 1.0) & (t < 2.5), y, 0.0).astype(np.float32)
    f = track_pitch(y, synth.SR)
    curve = build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate, rms_fine=f.rms_fine)
    notes = detect_notes(curve, articulation_envelope(f.onset, f.rms, f.frame_rate, f.rms_fine),
                         AnalysisSettings().note_params())
    assert [n.pitch for n in notes] == [57]
    assert abs(notes[0].start - 1.0) <= 0.02 and abs(notes[0].end - 2.5) <= 0.03
    tone = curve.midi()[int(1.1 * FRAME_RATE):int(2.4 * FRAME_RATE)]
    assert abs(float(np.nanmedian(tone)) - 57.0) * 100 <= TOL["median_cents"]


def test_3_repeated_syllables_are_separate_notes(bench):
    for case in ("repeat_gap", "repeat_dip"):
        r = bench[(case, "standard", "v2")]
        assert r["est"] == r["ref"] == r["matched"], r["notes"]


def test_4_vibrato_one_note_with_rate_and_extent():
    y = synth.sung([(0.5, 2.5, 67)], 3.0, vibrato_cents=60, vibrato_hz=5.5)
    f = track_pitch(y, synth.SR)
    curve = build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate, rms_fine=f.rms_fine)
    art = articulation_envelope(f.onset, f.rms, f.frame_rate, f.rms_fine)
    for sens in ("simple", "standard", "precise"):
        notes = detect_notes(curve, art, AnalysisSettings.for_sensitivity(sens).note_params())
        assert [n.pitch for n in notes] == [67], sens
        v = notes[0].vibrato
        assert v is not None and abs(v.rate_hz - 5.5) <= 0.6 and 40 <= v.extent_cents <= 80


def test_5_portamento_not_over_split(bench):
    for sens in ("simple", "standard", "precise"):
        r = bench[("portamento", sens, "v2")]
        assert r["est"] == 3 and r["matched"] == 3, (sens, r["notes"])


def test_6_short_notes_and_articulated_ornament():
    case = next(c for c in B.CASES if c.name == "short_notes")
    f = track_pitch(B.render(case), synth.SR)
    curve = build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate, rms_fine=f.rms_fine)
    art = articulation_envelope(f.onset, f.rms, f.frame_rate, f.rms_fine)
    notes = detect_notes(curve, art, AnalysisSettings().note_params())
    s = score_notes(B._reference(case), notes)
    assert s.f1 >= TOL["f1_min"]
    grace = [n for n in notes if n.pitch == 74]
    assert len(grace) == 1 and grace[0].ornament and abs(grace[0].start - 1.6) <= 0.02


def test_7_silence_and_breath_produce_no_notes():
    y = np.zeros(int(synth.SR * 2.0), np.float32) + synth.breath_noise(2.0, [(0.3, 1.7)], -30)
    f = track_pitch(y, synth.SR)
    curve = build_curve(f.f0_hz, f.voiced_prob, f.rms, f.frame_rate, rms_fine=f.rms_fine)
    for sens in ("simple", "standard", "precise"):
        notes = detect_notes(curve, articulation_envelope(f.onset, f.rms, f.frame_rate, f.rms_fine),
                             AnalysisSettings.for_sensitivity(sens).note_params())
        assert notes == [], sens
    zero = track_pitch(np.zeros(synth.SR, np.float32), synth.SR)
    assert not np.isfinite(build_curve(zero.f0_hz, zero.voiced_prob, zero.rms, 100.0, rms_fine=zero.rms_fine).midi()).any()


def test_8_octave_errors_mitigated(bench):
    for sens in ("standard", "precise"):
        r = bench[("octave_inject", sens, "v2")]
        assert r["octave_error_rate"] == 0.0 and r["f1"] == 1.0


def test_simple_never_splits_more_than_standard(bench):
    for c in B.CASES:
        assert bench[(c.name, "simple", "v2")]["est"] <= bench[(c.name, "standard", "v2")]["est"], c.name


# --- 변경 전후 비교 (기준선 v1 = STEP 2 커밋 3d49cd4) ----------------------------------

@pytest.mark.parametrize("sens", ["standard", "precise"])
def test_v2_not_worse_than_v1_on_any_case(bench, sens):
    for c in B.CASES:
        v1, v2 = bench[(c.name, sens, "v1")], bench[(c.name, sens, "v2")]
        assert v2["f1"] >= v1["f1"], (c.name, v1["notes"], v2["notes"])
        assert v2["voicing_false_alarm"] <= v1["voicing_false_alarm"], c.name
        assert v2["octave_error_rate"] <= v1["octave_error_rate"], c.name


@pytest.mark.parametrize("case", ["repeat_gap", "repeat_dip", "short_notes"])
def test_v2_fixes_v1_failures(bench, case):
    """v1이 실패하던 사례: 같은 음 반복 병합, 짧은 음 누락."""
    v1, v2 = bench[(case, "standard", "v1")], bench[(case, "standard", "v2")]
    assert v1["f1"] < 0.7 <= 0.9 <= v2["f1"]


def test_v2_fixes_v1_portamento_over_split_in_precise(bench):
    v1, v2 = bench[("portamento", "precise", "v1")], bench[("portamento", "precise", "v2")]
    assert v1["est"] > v1["ref"] and v2["est"] == v2["ref"]


def test_v2_lowers_voicing_false_alarm_overall(bench):
    fa = {v: np.mean([bench[(c.name, "standard", v)]["voicing_false_alarm"] for c in B.CASES]) for v in ("v1", "v2")}
    assert fa["v2"] < fa["v1"] * 0.5


# --- 구성 요소 ------------------------------------------------------------------------

def test_onset_envelope_peaks_at_note_starts():
    """v1(librosa 기본값)은 피크가 40~80ms 늦었다. v2는 20ms 이내."""
    from scipy.signal import find_peaks
    import librosa

    melody = [(0.5 + 0.4 * i, 0.9 + 0.4 * i, p) for i, p in enumerate([60, 64, 67, 72])]
    y = librosa.resample(synth.sung(melody, 2.4), orig_sr=synth.SR, target_sr=16000)
    o = onset_envelope(y)
    peaks, _ = find_peaks(o, height=0.3 * o.max(), distance=5)
    for a, _b, _m in melody:
        assert np.min(np.abs(peaks / FRAME_RATE - a)) <= 0.02


def test_gap_gate_cuts_short_silence_between_syllables():
    db = np.full(200, -15.0)
    db[100:104] = -80.0  # 40ms 무음
    db[150:153] = -35.0  # 20dB 하락은 틈이 아니다
    gate = gap_gate(10 ** (db / 20), 100.0)
    assert not gate[100:104].any() and gate[150:153].all() and gate[:100].all()


def test_articulation_uses_fine_rms_when_given():
    n = 300
    onset = np.ones(n)
    onset[150] = 10.0
    rms = np.full(n, 0.1)  # 64ms RMS에는 하락이 안 보인다
    fine = np.full(n, 0.1)
    fine[145:150] = 0.01  # 20ms RMS에는 20dB 하락
    assert articulation_envelope(onset, rms, 100.0)[150] < 1.0
    assert articulation_envelope(onset, rms, 100.0, rms_fine=fine)[150] > 8.0
