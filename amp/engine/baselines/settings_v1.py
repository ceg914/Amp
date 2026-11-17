"""[기준선 v1 — 고정됨] STEP 2 커밋 3d49cd4의 민감도 프리셋 내부 값. 변경 전후 비교에만 쓰며 수정하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass

PRESET_INTERNAL_V1 = {
    "simple": dict(hold_s=0.12, center_window_s=0.22, voiced_threshold=0.05, bridge_gap_s=0.06,
                   onset_split_z=None, onset_split_rel=None, ornament_min_s=None, ornament_conf=1.0, merge_gap_s=0.08),
    "standard": dict(hold_s=0.08, center_window_s=0.22, voiced_threshold=0.02, bridge_gap_s=0.04,
                     onset_split_z=4.0, onset_split_rel=0.6, ornament_min_s=0.05, ornament_conf=0.8, merge_gap_s=0.05),
    "precise": dict(hold_s=0.05, center_window_s=0.22, voiced_threshold=0.01, bridge_gap_s=0.03,
                    onset_split_z=3.0, onset_split_rel=0.45, ornament_min_s=0.035, ornament_conf=0.65, merge_gap_s=0.03),
}


@dataclass(frozen=True)
class NoteDetectionParamsV1:
    min_note_s: float
    change_semitones: float
    hold_s: float
    center_window_s: float
    voiced_threshold: float
    bridge_gap_s: float
    onset_split_z: float | None
    onset_split_rel: float | None
    ornament_min_s: float | None
    ornament_conf: float
    merge_gap_s: float
    pitch_low: int
    pitch_high: int


def note_params_v1(settings) -> NoteDetectionParamsV1:
    """AnalysisSettings → v1 내부 값 (STEP 2의 AnalysisSettings.note_params와 같은 계산)."""
    internal = dict(PRESET_INTERNAL_V1[settings.sensitivity])
    min_s = settings.min_note_ms / 1000.0
    internal["hold_s"] = min(internal["hold_s"], min_s)
    if internal["ornament_min_s"] is not None and internal["ornament_min_s"] >= min_s:
        internal["ornament_min_s"] = None
    return NoteDetectionParamsV1(min_note_s=min_s, change_semitones=settings.pitch_change_semitones,
                                 pitch_low=settings.pitch_low, pitch_high=settings.pitch_high, **internal)
