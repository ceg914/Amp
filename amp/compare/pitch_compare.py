"""피치 모델 결과 비교 (v0.7.0 Feature A, 명세 4.5). Qt 비의존, 결과를 바꾸지 않는다.

두 결과가 겹치는 프레임에서 유성·무성 판정 차이, 둘 다 유성인 프레임의 센트 차이, 옥타브 차이 의심 구간,
한 모델에서만 유성인 구간, 처리 시간·메모리를 센다. 정답이 없으면 이것은 '차이'이지 정확도가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amp.engine.pitch_runs import CONFIDENCE_LABELS, PitchRun

OCTAVE_TOL_CENTS = 150.0  # |차이| ≈ 1200 ± 150이면 옥타브 차이로 본다
FIFTH_TOL_CENTS = 60.0  # 700 ± 60 (5도, 배음을 기본음으로 잡은 경우 등)
LARGE_CENTS = 50.0
MIN_REGION_S = 0.10


@dataclass
class PitchDiff:
    a: str  # run_id
    b: str
    a_label: str
    b_label: str
    start: float  # 겹치는 범위 (초)
    end: float
    frames: int = 0
    both_voiced: int = 0
    only_a: int = 0
    only_b: int = 0
    neither: int = 0
    median_abs_cents: float | None = None
    p90_abs_cents: float | None = None
    mean_cents: float | None = None  # a − b 평균 (부호 있음)
    over_50c: int = 0
    octave_frames: int = 0
    fifth_frames: int = 0
    octave_regions: list = field(default_factory=list)  # (시작, 끝, 중앙 차이 센트)
    only_a_regions: list = field(default_factory=list)  # (시작, 끝)
    only_b_regions: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)  # run_id → {elapsed_s, device, gpu_peak_mb, peak_rss_mb}
    notes: list = field(default_factory=list)

    @property
    def voicing_agreement(self) -> float | None:
        return None if self.frames == 0 else (self.both_voiced + self.neither) / self.frames

    def summary_lines(self) -> list[str]:
        if self.frames == 0:
            return ["두 결과가 겹치는 구간이 없습니다."]
        lines = [f"비교 구간 {self.start:.2f}–{self.end:.2f}s, 프레임 {self.frames}개 (10ms)",
                 f"유성·무성 판정이 같은 프레임 {100 * self.voicing_agreement:.1f}%: 둘 다 유성 {self.both_voiced}, "
                 f"{self.a_label}만 유성 {self.only_a}, {self.b_label}만 유성 {self.only_b}"]
        if self.both_voiced:
            lines.append(f"둘 다 유성인 프레임의 센트 차이: 중앙 {self.median_abs_cents:.1f}, 90% {self.p90_abs_cents:.1f}, "
                         f"50센트 넘음 {self.over_50c}프레임, 옥타브 차이 {self.octave_frames}프레임, 5도 차이 {self.fifth_frames}프레임")
        lines.append(f"옥타브 차이 의심 구간 {len(self.octave_regions)}곳, {self.a_label}에서만 유성 {len(self.only_a_regions)}곳, "
                     f"{self.b_label}에서만 유성 {len(self.only_b_regions)}곳 ({int(MIN_REGION_S * 1000)}ms 이상)")
        lines.extend(self.notes)
        lines.append("정답 자료가 없으므로 위 값은 모델 사이의 차이이며 정확도가 아닙니다.")
        return lines


def runs_regions(mask: np.ndarray, t0: float, fr: float, min_s: float = MIN_REGION_S, values: np.ndarray | None = None):
    """참인 프레임이 이어진 구간 (min_s 이상). values를 주면 (시작, 끝, 구간 중앙값)."""
    out = []
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return out
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for s, e in zip(starts, ends):
        if (e - s) / fr + 1e-9 < min_s:
            continue
        a, b = round(t0 + s / fr, 3), round(t0 + e / fr, 3)
        if values is None:
            out.append((a, b))
        else:
            out.append((a, b, round(float(np.nanmedian(values[s:e])), 1)))
    return out


def cents_between(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(fa / fb)


def compare(a: PitchRun, b: PitchRun, min_region_s: float = MIN_REGION_S) -> PitchDiff:
    fr = a.frame_rate
    diff = PitchDiff(a.run_id, b.run_id, a.label, b.label, 0.0, 0.0)
    for r in (a, b):
        diff.resources[r.run_id] = {k: r.info.get(k) for k in ("elapsed_s", "device", "gpu_peak_mb", "peak_rss_mb",
                                                                "from_cache") if r.info.get(k) is not None}
    if abs(a.frame_rate - b.frame_rate) > 1e-6:
        diff.notes.append("프레임 간격이 달라 비교하지 않았습니다.")
        return diff
    g0, g1 = max(a.start_frame, b.start_frame), min(a.end_frame, b.end_frame)
    if g1 <= g0:
        return diff
    fa, va, _ = a.window(g0, g1)
    fb, vb, _ = b.window(g0, g1)
    t0 = g0 / fr
    diff.start, diff.end, diff.frames = round(t0, 3), round(g1 / fr, 3), int(g1 - g0)
    both = va & vb
    diff.both_voiced = int(both.sum())
    diff.only_a = int((va & ~vb).sum())
    diff.only_b = int((vb & ~va).sum())
    diff.neither = int((~va & ~vb).sum())
    c = np.where(both, cents_between(fa, fb), np.nan)
    if diff.both_voiced:
        ac = np.abs(c[both])
        diff.median_abs_cents = round(float(np.median(ac)), 1)
        diff.p90_abs_cents = round(float(np.percentile(ac, 90)), 1)
        diff.mean_cents = round(float(np.mean(c[both])), 1)
        diff.over_50c = int((ac > LARGE_CENTS).sum())
    octave = both & (np.abs(np.abs(c) - 1200.0) <= OCTAVE_TOL_CENTS)
    fifth = both & (np.abs(np.abs(c) - 700.0) <= FIFTH_TOL_CENTS)
    diff.octave_frames, diff.fifth_frames = int(octave.sum()), int(fifth.sum())
    diff.octave_regions = runs_regions(octave, t0, fr, min_region_s, c)
    diff.only_a_regions = runs_regions(va & ~vb, t0, fr, min_region_s)
    diff.only_b_regions = runs_regions(vb & ~va, t0, fr, min_region_s)
    if a.confidence_kind != b.confidence_kind:
        diff.notes.append(f"신뢰도는 비교하지 않습니다: {CONFIDENCE_LABELS.get(a.confidence_kind, a.confidence_kind)}와 "
                          f"{CONFIDENCE_LABELS.get(b.confidence_kind, b.confidence_kind)}는 뜻이 다릅니다.")
    if a.meta.analysis_source_id != b.meta.analysis_source_id:
        diff.notes.append("두 결과의 입력 소리가 다릅니다 (보컬 분리 결과 등). 모델 차이와 입력 차이가 섞여 있습니다.")
    return diff


def cents_series(a: PitchRun, b: PitchRun) -> tuple[np.ndarray, np.ndarray]:
    """겹치는 구간의 (시각, a − b 센트). 한쪽이라도 무성이면 NaN. 그래프용."""
    g0, g1 = max(a.start_frame, b.start_frame), min(a.end_frame, b.end_frame)
    if g1 <= g0:
        return np.zeros(0), np.zeros(0)
    fa, va, _ = a.window(g0, g1)
    fb, vb, _ = b.window(g0, g1)
    return (g0 + np.arange(g1 - g0)) / a.frame_rate, np.where(va & vb, cents_between(fa, fb), np.nan)
