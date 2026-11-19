"""분석 설정.

사용자가 패널에서 바꾸는 값(``AnalysisSettings``)과, 거기서 계산되는
음표 검출 내부 값(``NoteDetectionParams``)을 나눈다. 민감도 프리셋은 내부 값의 묶음이다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace

from amp.core.errors import AmpError
from amp.core.pitch import MIDI_MAX, MIDI_MIN

SENSITIVITIES = ("simple", "standard", "precise")
SENSITIVITY_LABELS = {"simple": "단순", "standard": "표준", "precise": "정밀"}
DEVICES = ("auto", "cpu", "cuda")
DEVICE_LABELS = {"auto": "자동 (GPU가 있으면 GPU)", "cpu": "CPU", "cuda": "GPU (CUDA)"}
SEPARATIONS = ("htdemucs", "none")
SEPARATION_LABELS = {"htdemucs": "Demucs htdemucs", "none": "분리 안 함 (무반주 음원)"}

# 민감도별 사용자 노출 기본값: (최소 음표 길이 ms, 음높이 변화 기준 반음)
_PRESET_USER = {
    "simple": (150.0, 1.0),
    "standard": (100.0, 1.0),
    "precise": (70.0, 0.8),
}

# 민감도별 내부 값 (v2, STEP 2.5). v1 값은 amp/engine/baselines/settings_v1.py에 고정해 두었다.
# split_strength: 재발음 지표(온셋 = 멜 대역 평균 dB 증가량 × 음량 하락 비율) 피크의 절대 임계.
#   합성 음원에서 실제 재발음은 5.9 이상, 지속음 안의 흔들림은 4.3 이하였다 (amp.devtools.benchmark).
_PRESET_INTERNAL = {
    "simple": dict(hold_s=0.12, center_window_s=0.22, voiced_threshold=0.05, bridge_gap_s=0.06,
                   split_strength=None, ornament_min_s=None, ornament_conf=1.0, merge_gap_s=0.08),
    "standard": dict(hold_s=0.08, center_window_s=0.22, voiced_threshold=0.02, bridge_gap_s=0.04,
                     split_strength=4.0, ornament_min_s=0.05, ornament_conf=0.8, merge_gap_s=0.05),
    "precise": dict(hold_s=0.05, center_window_s=0.22, voiced_threshold=0.01, bridge_gap_s=0.03,
                    split_strength=3.0, ornament_min_s=0.035, ornament_conf=0.65, merge_gap_s=0.03),
}


MAX_NO_VOCAL_RANGES = 500


class SettingsError(AmpError):
    pass


def normalize_ranges(ranges, strict: bool = True) -> tuple[tuple[float, float], ...]:
    """[(시작초, 끝초)] → 정렬하고 겹치거나 맞닿은 구간은 합친 튜플.

    strict면 잘못된 구간(숫자가 아님, 음수, 끝 ≤ 시작)에서 SettingsError, 아니면 그 구간만 버린다.
    """
    clean = []
    for r in ranges or ():
        try:
            a, b = float(r[0]), float(r[1])
            ok = math.isfinite(a) and math.isfinite(b) and a >= 0.0 and b - a >= 0.01
        except (TypeError, ValueError, IndexError):
            ok = False
        if not ok:
            if strict:
                raise SettingsError(f"보컬 없는 구간이 올바르지 않습니다: {r!r} (시작 < 끝, 0초 이상)")
            continue
        clean.append((a, b))
    clean.sort()
    merged: list[list[float]] = []
    for a, b in clean:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if len(merged) > MAX_NO_VOCAL_RANGES:
        if strict:
            raise SettingsError(f"보컬 없는 구간은 {MAX_NO_VOCAL_RANGES}개까지 지정할 수 있습니다.")
        merged = merged[:MAX_NO_VOCAL_RANGES]
    return tuple((round(a, 3), round(b, 3)) for a, b in merged)


@dataclass(frozen=True)
class NoteDetectionParams:
    min_note_s: float
    change_semitones: float
    hold_s: float  # 새 음높이가 이만큼 유지돼야 음표를 나눈다
    center_window_s: float  # 비브라토를 상쇄하는 중심선 중앙값 창 (4.5Hz 이상 비브라토 한 주기 이상)
    voiced_threshold: float  # pYIN 유성 확률 하한. 유성 판정의 주체는 pYIN의 HMM이고, 이 값은 아주 낮은 잡음만 거른다
    bridge_gap_s: float  # 이보다 짧은 무성 틈은 같은 프레이즈로 잇는다
    split_strength: float | None  # 같은 음 재발음 분할 임계 (재발음 지표 피크). None이면 나누지 않는다
    ornament_min_s: float | None  # 최소 길이보다 짧아도 이 길이 이상·고신뢰면 장식음으로 남김
    ornament_conf: float
    merge_gap_s: float  # 같은 음 사이 틈이 이보다 짧으면 합친다
    pitch_low: int
    pitch_high: int
    no_vocal_ranges: tuple = ()  # 이 구간(초)의 프레임은 무성으로 보고 음표를 만들지 않는다
    merge_approach: bool = True  # 음 시작의 짧은 끌어올림을 목표 음에 합친다 (STEP 3.5)
    syllable_starts: tuple = ()  # 정렬한 가사 음절 시작(초). 있으면 다음 음절의 음을 접근음 대상으로 삼지 않는다


@dataclass(frozen=True)
class AnalysisSettings:
    sensitivity: str = "standard"
    min_note_ms: float = 100.0
    pitch_change_semitones: float = 1.0
    pitch_low: int = 36  # C2
    pitch_high: int = 84  # C6
    device: str = "auto"
    use_cache: bool = True
    separation: str = "htdemucs"
    # 보컬이 없는 구간 [(시작초, 끝초)]. 분리 결과에 반주가 섞여 생기는 잘못된 음표를 막는다.
    # 곡마다 다르므로 프로젝트에만 저장하고 프로그램 기본 설정에는 저장하지 않는다.
    no_vocal_ranges: tuple = ()
    # 음 시작에서 낮은 데서 올라가는 짧은 구간(끌어올림)을 목표 음에 합친다. 끄면 STEP 3까지와 같은 결과.
    merge_approach: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "no_vocal_ranges", normalize_ranges(self.no_vocal_ranges))
        if self.sensitivity not in SENSITIVITIES:
            raise SettingsError(f"알 수 없는 민감도: {self.sensitivity}")
        if self.device not in DEVICES:
            raise SettingsError(f"알 수 없는 처리 장치: {self.device}")
        if self.separation not in SEPARATIONS:
            raise SettingsError(f"알 수 없는 분리 방식: {self.separation}")
        if not (20.0 <= self.min_note_ms <= 2000.0):
            raise SettingsError("최소 음표 길이는 20~2000ms 사이여야 합니다.")
        if not (0.3 <= self.pitch_change_semitones <= 12.0):
            raise SettingsError("음높이 변화 기준은 0.3~12반음 사이여야 합니다.")
        if not (MIDI_MIN <= self.pitch_low < self.pitch_high <= MIDI_MAX):
            raise SettingsError("음역의 최저음은 최고음보다 낮아야 합니다.")

    @classmethod
    def for_sensitivity(cls, sensitivity: str, base: "AnalysisSettings | None" = None) -> "AnalysisSettings":
        """민감도 프리셋의 최소 길이·변화 기준을 적용한다. 나머지 값은 base에서 가져온다."""
        if sensitivity not in _PRESET_USER:
            raise SettingsError(f"알 수 없는 민감도: {sensitivity}")
        min_ms, change = _PRESET_USER[sensitivity]
        return replace(base or cls(), sensitivity=sensitivity, min_note_ms=min_ms, pitch_change_semitones=change)

    def note_params(self) -> NoteDetectionParams:
        internal = dict(_PRESET_INTERNAL[self.sensitivity])
        min_s = self.min_note_ms / 1000.0
        internal["hold_s"] = min(internal["hold_s"], min_s)
        if internal["ornament_min_s"] is not None and internal["ornament_min_s"] >= min_s:
            internal["ornament_min_s"] = None
        return NoteDetectionParams(
            min_note_s=min_s,
            change_semitones=self.pitch_change_semitones,
            pitch_low=self.pitch_low,
            pitch_high=self.pitch_high,
            no_vocal_ranges=self.no_vocal_ranges,
            merge_approach=bool(self.merge_approach),
            **internal,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "AnalysisSettings":
        """모르는 키는 무시하고 빠진 키는 기본값을 쓴다. 값이 잘못되면 기본 설정을 돌려준다."""
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in d.items() if k in names}
        if "no_vocal_ranges" in values:  # 잘못된 구간 하나 때문에 다른 설정까지 기본값이 되지 않게
            values["no_vocal_ranges"] = normalize_ranges(values["no_vocal_ranges"], strict=False)
        try:
            return cls(**values)
        except (SettingsError, TypeError, ValueError):
            return cls()
