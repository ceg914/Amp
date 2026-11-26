"""발성·음높이 변화 분석 (v0.7.0 Feature F, 명세 9장). Qt 비의존.

피치 곡선(100fps)과 (있으면) 음표·비브라토·멜리스마 구간에서 음높이 변화 사건을 찾는다. 한 시점에 여러 종류가 겹칠 수 있다
(예: 멜리스마 안의 포르타멘토와 비브라토). 한 종류로만 나누지 않는다.

종류: stable(안정) · rise(상승) · fall(하강) · approach(접근음: 유성 시작에서 목표보다 낮게/높게 시작해 도달) ·
portamento(두 안정 구간 사이 이어진 미끄러짐) · jump(급격한 전환: 40ms 안에 1반음 넘게) · vibrato · repeated(같은 높이 반복 발성) ·
melisma · voicing(유성·무성 전환)

느린 흐름은 2Hz 저역 통과(영위상)로 비브라토를 지운 곡선에서 본다. 근거(evidence)와 불확실성(0~1)을 함께 적는다.
음향 지표(RMS 변화·스펙트럼 중심·배음 분포·HNR·피치 안정도·유성 비율)는 음표·프레이즈마다 요약한다.
분리 과정에서 음색·배음이 바뀔 수 있다. 이 지표만으로 흉성·두성·믹스보이스·성대 상태를 정하지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ANALYSIS_VERSION = "1"
KINDS = {"stable": "안정", "rise": "상승", "fall": "하강", "approach": "접근음", "portamento": "포르타멘토",
         "jump": "급격한 전환", "vibrato": "비브라토", "repeated": "반복 발성", "melisma": "멜리스마", "voicing": "유성·무성 전환"}
STABLE_CENTS = 25.0
STABLE_MIN_S = 0.15
MOVE_MIN_CENTS = 50.0
JUMP_CENTS = 100.0
JUMP_MAX_S = 0.04
SETTLE_CENTS = 30.0


@dataclass
class PitchEvent:
    kind: str
    start: float
    end: float
    start_midi: float | None = None
    end_midi: float | None = None
    delta_cents: float | None = None
    rate_cents_per_s: float | None = None
    settle_s: float | None = None  # 안정화까지 걸린 시간 (접근음·포르타멘토)
    evidence: str = ""
    uncertainty: float = 0.0  # 0 = 확실, 1 = 매우 불확실
    note_id: str | None = None
    syllable_id: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def smooth_midi(f0_hz: np.ndarray, frame_rate: float = 100.0) -> np.ndarray:
    """유성 덩어리마다 2Hz 저역 통과한 MIDI (무성 NaN)."""
    from scipy.signal import butter, sosfiltfilt

    f0 = np.asarray(f0_hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 69 + 12 * np.log2(f0 / 440.0)
    out = np.full_like(m, np.nan)
    sos = butter(2, 2.0, btype="lowpass", fs=frame_rate, output="sos")
    ok = np.isfinite(m)
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        x = m[a:b]
        out[a:b] = sosfiltfilt(sos, x, padlen=min(len(x) - 1, 9)) if len(x) > 12 else x
    return out


def _runs(mask):
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def detect_events(f0_hz: np.ndarray, frame_rate: float = 100.0, notes=(), vibrato=(), melisma=()) -> list[PitchEvent]:
    """notes: amp.core.model.Note 목록(있으면), vibrato: VibratoSegment 목록, melisma: (시작, 끝, 음절 id) 목록."""
    fr = float(frame_rate)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 69 + 12 * np.log2(f0 / 440.0)
    sm = smooth_midi(f0, fr)
    ev: list[PitchEvent] = []
    t = lambda i: i / fr  # noqa: E731
    voiced_runs = _runs(np.isfinite(raw))
    for a, b in voiced_runs:
        ev.append(PitchEvent("voicing", t(a), t(a), evidence="무성 → 유성"))
        ev.append(PitchEvent("voicing", t(b), t(b), evidence="유성 → 무성"))
        x = sm[a:b]
        # 안정 구간: 이동 중앙값에서 ±25센트 안에 150ms 이상
        w = max(3, int(0.1 * fr))
        med = np.array([np.median(x[max(0, i - w):i + w + 1]) for i in range(len(x))])
        stable = np.abs(x - med) * 100 <= STABLE_CENTS
        slope = np.gradient(x) * fr * 100  # 센트/초
        stable &= np.abs(slope) < 150
        st_runs = [(s, e) for s, e in _runs(stable) if (e - s) / fr >= STABLE_MIN_S]
        for s, e in st_runs:
            seg = raw[a + s:a + e]
            spread = float(np.nanstd(seg) * 100)
            ev.append(PitchEvent("stable", t(a + s), t(a + e), float(np.median(x[s:e])), float(np.median(x[s:e])), 0.0, 0.0,
                                 evidence=f"2Hz 저역 곡선이 ±{STABLE_CENTS:.0f}센트 안 (원곡선 표준편차 {spread:.0f}센트)",
                                 uncertainty=float(np.clip(spread / 100, 0, 1))))
        # 안정 구간 사이: 이동 (포르타멘토·상승·하강·급격한 전환)
        bounds = [(0, 0)] + st_runs + [(len(x), len(x))]
        for (s0, e0), (s1, e1) in zip(bounds, bounds[1:]):
            if s1 - e0 < 2:
                continue
            i0, i1 = e0, s1
            seg = x[i0:i1]
            if len(seg) < 2 or not np.isfinite(seg).all():
                continue
            start_m = float(x[e0 - 1]) if e0 > 0 else float(seg[0])
            end_m = float(x[s1]) if s1 < len(x) else float(seg[-1])
            delta = (end_m - start_m) * 100
            dur = (i1 - i0) / fr
            if abs(delta) < MOVE_MIN_CENTS:
                continue
            between_stables = e0 > 0 and s1 < len(x)
            r0 = a + max(0, i0 - 2)
            raw_seg = raw[r0:a + min(len(x), i1 + 2)]
            steps = np.abs(np.diff(raw_seg)) * 100
            steps = np.where(np.isfinite(steps), steps, 0.0)
            k = int(np.argmax(steps)) if len(steps) else 0
            if len(steps) and steps[k] >= JUMP_CENTS and steps[k] >= 0.7 * abs(delta):
                # 원곡선에서 한 프레임 사이에 대부분 바뀜: 급격한 전환 (저역 곡선은 이것을 늘려 보이게 한다)
                ev.append(PitchEvent("jump", t(r0 + k), t(r0 + k + 1), start_m, end_m, float(delta),
                                     float(delta * fr), 0.0, evidence=f"원곡선에서 10ms 사이 {steps[k]:.0f}센트",
                                     uncertainty=0.1))
                continue
            if dur <= JUMP_MAX_S * 2:
                kind = "jump"
            elif between_stables:
                kind = "portamento"
            else:
                kind = "rise" if delta > 0 else "fall"
            if e0 == 0 and s1 < len(x) and kind != "jump":
                kind = "approach"  # 유성 시작에서 목표 높이로 다가감
            settle = None
            if s1 < len(x):
                target = float(np.median(x[s1:e1]))
                within = np.flatnonzero(np.abs(x[i0:] - target) * 100 <= SETTLE_CENTS)
                settle = float(within[0] / fr) if len(within) else None
            ev.append(PitchEvent(kind, t(a + i0), t(a + i1), start_m, end_m, float(delta), float(delta / max(dur, 1e-3)),
                                 settle, evidence=f"{dur * 1000:.0f}ms 동안 {delta:+.0f}센트",
                                 uncertainty=0.3 if dur < 0.06 else 0.1))
    for s in vibrato:
        ev.append(PitchEvent("vibrato", s.start, s.end, s.center_midi, s.center_midi, 0.0, 0.0,
                             evidence=f"{s.rate_hz:.1f}Hz, 한쪽 {s.extent_cents:.0f}센트",
                             uncertainty=0.0 if s.status == "completed" else 0.6))
    notes = sorted(notes, key=lambda n: n.start)
    for n0, n1 in zip(notes, notes[1:]):
        if n1.pitch == n0.pitch and 0 <= n1.start - n0.end < 0.08:
            ev.append(PitchEvent("repeated", n0.start, n1.end, float(n0.pitch), float(n1.pitch), 0.0, 0.0,
                                 evidence=f"같은 높이 두 음 (쉼 {max(0.0, n1.start - n0.end) * 1000:.0f}ms)",
                                 note_id=n1.id))
    for a, b, sid in melisma:
        ev.append(PitchEvent("melisma", float(a), float(b), evidence="한 음절에 여러 음", syllable_id=sid))
    if notes:
        starts = np.array([n.start for n in notes])
        for e in ev:
            if e.note_id is None:
                k = int(np.searchsorted(starts, e.start + 1e-6)) - 1
                if 0 <= k < len(notes) and notes[k].start - 1e-6 <= e.start < notes[k].end:
                    e.note_id = notes[k].id
    ev.sort(key=lambda e: (e.start, e.kind))
    return ev


@dataclass
class SpanMetrics:
    start: float
    end: float
    note_id: str | None = None
    rms_db_mean: float | None = None
    rms_db_change: float | None = None  # 뒤쪽 1/3 − 앞쪽 1/3 (dB)
    centroid_hz: float | None = None
    h1_h2_db: float | None = None
    hnr_db: float | None = None
    pitch_stability_cents: float | None = None  # 2Hz 저역 곡선 대비 원곡선 표준편차 (비브라토 포함)
    voiced_ratio: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


def span_metrics(feat: dict, f0_hz: np.ndarray, spans, frame_rate: float = 100.0) -> list[SpanMetrics]:
    """spans: [(시작, 끝, note_id|None)]. 지표가 없는 칸은 None."""
    fr = float(frame_rate)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    sm = smooth_midi(f0, fr)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 69 + 12 * np.log2(f0 / 440.0)
    out = []
    for a, b, nid in spans:
        i0, i1 = max(0, int(a * fr)), int(b * fr)
        m = SpanMetrics(float(a), float(b), nid)
        if i1 - i0 >= 3:
            db = np.asarray(feat["rms_db"][i0:i1], dtype=np.float64)
            third = max(1, (i1 - i0) // 3)
            m.rms_db_mean = float(np.nanmean(db))
            m.rms_db_change = float(np.nanmean(db[-third:]) - np.nanmean(db[:third]))
            v = np.isfinite(raw[i0:i1])
            m.voiced_ratio = float(v.mean())
            if v.any():
                m.centroid_hz = float(np.nanmedian(np.asarray(feat["centroid_hz"][i0:i1])[v]))
                h = np.asarray(feat.get("h1_h2_db", np.full(i1, np.nan))[i0:i1])[v]
                q = np.asarray(feat.get("hnr_db", np.full(i1, np.nan))[i0:i1])[v]
                m.h1_h2_db = float(np.nanmedian(h)) if np.isfinite(h).any() else None
                m.hnr_db = float(np.nanmedian(q)) if np.isfinite(q).any() else None
                d = (raw[i0:i1] - sm[i0:i1])[v] * 100
                m.pitch_stability_cents = float(np.nanstd(d)) if np.isfinite(d).any() else None
        out.append(m)
    return out
