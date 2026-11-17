"""합성 신호 벤치마크: 기준선 v1(STEP 2)과 현재 알고리즘을 같은 pYIN 결과 위에서 비교한다.

합성 신호는 보컬 분리 모델 평가용이 아니다. 피치 후처리와 음표 검출 알고리즘의 검증에만 쓴다.
pYIN은 두 버전이 같으므로 사례마다 한 번만 돌린다. 버전별로 다른 것은
    v1: STEP 2 온셋(128ms 창) + 64ms RMS + v1 후처리·음표 검출·프리셋 (amp/engine/baselines)
    v2: 현재 온셋(32ms 창) + 20ms RMS + 현재 후처리·음표 검출·프리셋

실행: python -m amp.devtools.benchmark  (결과 표를 출력)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from amp.devtools import synth
from amp.devtools.evaluation import RefNote, pitch_metrics, score_notes
from amp.engine.settings import AnalysisSettings

SR = synth.SR


@dataclass
class Case:
    name: str
    label: str
    melody: list  # [(시작, 끝, MIDI)]
    duration: float
    synth_kw: dict = field(default_factory=dict)
    extra: str = ""  # 추가 가공: "breath" | "octave" | "dips"

    def reference(self) -> list[RefNote]:
        return [RefNote(a, b, m) for a, b, m in self.melody]


def _repeat(pitch, n, dur, gap, start=0.5):
    out, t = [], start
    for _ in range(n):
        out.append((round(t, 3), round(t + dur, 3), pitch))
        t += dur + gap
    return out


CASES = [
    Case("scale", "연속 음계 (레가토, 비브라토 15c)", [(0.5 + 0.4 * i, 0.9 + 0.4 * i, p) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])],
         4.0, {"vibrato_cents": 15}),
    Case("repeat_gap", "같은 음 반복 (40ms 무음 틈)", _repeat(64, 5, 0.3, 0.04), 2.6, {"same_vowel": True}),
    Case("repeat_dip", "같은 음 반복 (끊김 없이 음량만 -18dB 하락)", [(0.5, 2.1, 64)], 2.6, {"same_vowel": True}, extra="dips"),
    Case("vibrato_wide", "넓은 비브라토 (±80c, 5.5Hz)", [(0.5, 2.5, 67), (2.7, 4.2, 64)], 4.6, {"vibrato_cents": 80}),
    Case("portamento", "포르타멘토 (200ms 미끄러짐)", [(0.5, 1.3, 60), (1.3, 2.1, 65), (2.1, 2.9, 62)], 3.3, {"glide_s": 0.2}),
    Case("short_notes", "짧은 음 (110ms 연속 + 70ms 장식음)",
         [(0.5 + 0.11 * i, 0.61 + 0.11 * i, p) for i, p in enumerate([60, 62, 64, 65, 67, 65, 64, 62])]
         + [(1.6, 1.67, 74), (1.67, 2.4, 72)], 2.8),
    Case("breath", "쉼표 사이 숨소리 (-30dB 잡음)", [(0.5, 1.2, 62), (2.0, 2.7, 65)], 3.2, extra="breath"),
    Case("octave_inject", "옥타브 오류 주입 (pYIN 출력에 ±12 구간·튐 삽입)", [(0.5, 2.0, 57), (2.2, 3.6, 64)], 4.0, extra="octave"),
]

# "repeat_dip"의 정답: 음량 하락 지점마다 새 음표 (다시 부름)
_DIP_TIMES = (0.9, 1.3, 1.7)


def _reference(case: Case) -> list[RefNote]:
    if case.extra == "dips":
        edges = [0.5, *_DIP_TIMES, 2.1]
        return [RefNote(edges[i], edges[i + 1], 64) for i in range(len(edges) - 1)]
    return case.reference()


def render(case: Case) -> np.ndarray:
    y = synth.sung(case.melody, case.duration, **case.synth_kw)
    t = np.arange(len(y)) / SR
    if case.extra == "dips":
        for d in _DIP_TIMES:  # 60ms 동안 -18dB로 떨어졌다가 오르는 재발음
            sel = (t > d - 0.06) & (t < d + 0.01)
            y[sel] *= 10 ** (-18 / 20)
    if case.extra == "breath":
        y = y + synth.breath_noise(case.duration, [(1.3, 1.9)], -30)
    return y.astype(np.float32)


def _inject_octave_errors(f0: np.ndarray) -> np.ndarray:
    f0 = f0.copy()
    for a, b, factor in ((80, 95, 2.0), (140, 141, 0.5), (250, 262, 0.5), (300, 301, 2.0), (330, 342, 2.0)):
        f0[a:b] *= factor  # 150ms·120ms 구간 옥타브 오류, 1프레임 튐
    return f0


@dataclass
class VersionResult:
    notes: list
    curve_midi: np.ndarray
    seconds: float


def v1_onset(y: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    """STEP 2와 같은 방법으로 계산한 온셋 강도 (v1 비교용)."""
    import librosa

    from amp.engine.baselines.features_v1 import onset_v1
    from amp.engine.dsp.pitch import PITCH_SR, _fit

    y16 = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=PITCH_SR, res_type="soxr_hq")
    return _fit(onset_v1(y16), n_frames)


def run_versions(frames, settings: AnalysisSettings, onset_v1: np.ndarray) -> dict[str, VersionResult]:
    from amp.engine import notes as notes_v2
    from amp.engine import pitch_post as post_v2
    from amp.engine.baselines import notes_v1, pitch_post_v1
    from amp.engine.baselines.settings_v1 import note_params_v1

    out = {}
    t0 = time.perf_counter()
    curve = pitch_post_v1.build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate, frames.start_time)
    artic = pitch_post_v1.articulation_envelope(onset_v1, frames.rms, frames.frame_rate)
    notes = notes_v1.detect_notes(curve, artic, note_params_v1(settings))
    out["v1"] = VersionResult(notes, curve.midi(), time.perf_counter() - t0)

    t0 = time.perf_counter()
    curve = post_v2.build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate, frames.start_time,
                                rms_fine=frames.rms_fine)
    artic = post_v2.articulation_envelope(frames.onset, frames.rms, frames.frame_rate, rms_fine=frames.rms_fine)
    notes = notes_v2.detect_notes(curve, artic, settings.note_params())
    out["v2"] = VersionResult(notes, curve.midi(), time.perf_counter() - t0)
    return out


def evaluate(cases=CASES, sensitivities=("standard",)) -> list[dict]:
    from amp.engine.dsp.pitch import track_pitch

    rows = []
    for case in cases:
        y = render(case)
        frames = track_pitch(y, SR)
        onset1 = v1_onset(y, SR, len(frames.f0_hz))
        if case.extra == "octave":
            frames.f0_hz = _inject_octave_errors(frames.f0_hz)
        ref_midi = synth.ref_midi_frames(case.melody, case.duration, **case.synth_kw)
        for sens in sensitivities:
            settings = AnalysisSettings.for_sensitivity(sens)
            for version, res in run_versions(frames, settings, onset1).items():
                s = score_notes(_reference(case), res.notes)
                rows.append({"case": case.name, "label": case.label, "sensitivity": sens, "version": version,
                             **s.as_dict(), **pitch_metrics(res.curve_midi, ref_midi),
                             "ms": round(res.seconds * 1000, 1), "notes": [(round(n.start, 2), round(n.end, 2), n.pitch) for n in res.notes]})
    return rows


def markdown_table(rows: list[dict]) -> str:
    head = "| 사례 | 민감도 | 버전 | 정답/추정/일치 | F1 | 시작 오차 ms | 끝 오차 ms | 피치 50c 이내 | 옥타브 오류율 | 무성 오경보 |"
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['label']} | {r['sensitivity']} | {r['version']} | {r['ref']}/{r['est']}/{r['matched']} | "
                     f"{r['f1']:.2f} | {r['onset_mae_ms']} | {r['offset_mae_ms']} | {r['rpa_50c']} | {r['octave_error_rate']} | "
                     f"{r['voicing_false_alarm']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sens = tuple(sys.argv[1:]) or ("standard",)
    rows = evaluate(sensitivities=sens)
    print(markdown_table(rows))
