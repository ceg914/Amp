"""호흡·프레이즈 분석 (v0.7.0 Feature E, 명세 8장). Qt 비의존.

프레임 분류 (100fps): voiced · unvoiced · breath_candidate · silence · unknown
- silence: 음량이 곡의 바닥(10번째 백분위, -80dB 아래로는 안 봄)보다 6dB 안
- voiced: 피치가 있는 프레임 (pYIN 유성)
- 무성 덩어리(무음도 유성도 아닌 프레임이 이어진 것)를 나눈다
  · breath_candidate: 120ms~1.2초, 잡음 같고(평탄도 기준 이상), 유성 중앙값보다 10dB 넘게 작고, 앞뒤가 유성 발성에
    바로 붙어 있지 않음(양쪽 모두 40ms 안에 유성이면 자음으로 봄)
  · unvoiced: 짧거나 유성 발성에 붙은 것 (무성 자음 등)
  · unknown: 길거나(1.2초 초과) 잡음 같지 않은 것 (악기 소리 유입·마이크 잡음 등일 수 있음)
모든 무성 구간을 호흡으로 보지 않는다. 폐활량·호흡 건강을 추정하지 않는다.

프레이즈: 유성 덩어리를 phrase_gap(기본 0.3초)보다 짧은 쉼으로 이어 붙인 것. 쉼 안에 호흡 후보가 있으면 거기서 나눈다.
가사 음절 시작 시각을 주면 프레이즈마다 음절 수·초당 음절 수를 센다.

사용자 수정(호흡 확정·제외, 프레이즈 경계 옮기기·합치기·나누기)은 overrides로 따로 두고, 다시 분석해도 남긴다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ANALYSIS_VERSION = "1"
CLASSES = ("silence", "voiced", "unvoiced", "breath_candidate", "unknown")
CLASS_LABELS = {"silence": "무음", "voiced": "유성", "unvoiced": "무성(자음 등)", "breath_candidate": "호흡 후보",
                "unknown": "모름"}


@dataclass(frozen=True)
class BreathSettings:
    silence_margin_db: float = 6.0
    min_breath_s: float = 0.12
    max_breath_s: float = 1.2
    min_flatness: float = 0.12
    below_voiced_db: float = 10.0
    attach_s: float = 0.04
    phrase_gap_s: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Breath:
    breath_id: str
    start: float
    end: float
    level_db: float
    flatness: float
    hf_ratio: float
    score: float  # 0~1 (규칙을 얼마나 넉넉히 만족하는지)
    status: str = "candidate"  # candidate · confirmed · rejected (사용자)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "level_db", "flatness", "hf_ratio", "score"):
            d[k] = round(float(d[k]), 3)
        return d


@dataclass
class Phrase:
    phrase_id: str
    start: float
    end: float
    voiced_s: float
    gap_before_s: float | None
    gap_after_s: float | None
    breath_ids: list = field(default_factory=list)  # 앞 쉼에 있는 호흡 후보
    syllables: int | None = None
    syllables_per_s: float | None = None
    user_edited: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        for k in ("start", "end", "voiced_s", "gap_before_s", "gap_after_s", "syllables_per_s"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


def _runs(mask):
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def classify(feat: dict, voiced: np.ndarray, frame_rate: float = 100.0, settings: BreathSettings | None = None):
    """(프레임 분류 배열(문자열), 호흡 후보 목록)."""
    st = settings or BreathSettings()
    db = np.asarray(feat["rms_db"], dtype=np.float64)
    n = min(len(db), len(voiced))
    db, voiced = db[:n], np.asarray(voiced[:n], dtype=bool)
    flat = np.asarray(feat["flatness"], dtype=np.float64)[:n]
    hf = np.asarray(feat["hf_ratio"], dtype=np.float64)[:n]
    floor = max(float(np.percentile(db, 10)), -80.0)
    silence = db < floor + st.silence_margin_db
    cls = np.full(n, "unknown", dtype=object)
    cls[silence] = "silence"
    cls[voiced & ~silence] = "voiced"
    vmed = float(np.median(db[voiced & ~silence])) if (voiced & ~silence).any() else float(np.max(db))
    rest = ~voiced & ~silence
    attach = int(round(st.attach_s * frame_rate))
    breaths: list[Breath] = []
    for a, b in _runs(rest):
        dur = (b - a) / frame_rate
        before = voiced[max(0, a - attach):a].any()
        after = voiced[b:b + attach].any()
        level = float(np.percentile(db[a:b], 90))
        fl = float(np.median(flat[a:b]))
        if dur < st.min_breath_s or (before and after):
            cls[a:b] = "unvoiced"
            continue
        if dur > st.max_breath_s or fl < st.min_flatness or level > vmed - st.below_voiced_db:
            cls[a:b] = "unknown" if dur > st.max_breath_s or fl < st.min_flatness else "unvoiced"
            continue
        cls[a:b] = "breath_candidate"
        score = float(np.clip(0.4 * min(1.0, (fl - st.min_flatness) / 0.2 + 0.5)
                              + 0.3 * min(1.0, (vmed - level - st.below_voiced_db) / 10 + 0.5)
                              + 0.3 * (0.0 if (before or after) else 1.0), 0, 1))
        breaths.append(Breath(f"b{len(breaths) + 1}", a / frame_rate, b / frame_rate, level, fl,
                              float(np.median(hf[a:b])), score))
    return cls, breaths


def phrases(cls: np.ndarray, breaths: list[Breath], frame_rate: float = 100.0,
            settings: BreathSettings | None = None, syllable_starts: list[float] | None = None) -> list[Phrase]:
    st = settings or BreathSettings()
    voiced = np.asarray(cls == "voiced")
    vr = _runs(voiced)
    if not vr:
        return []
    groups = [[vr[0]]]
    for a, b in vr[1:]:
        pa, pb = groups[-1][-1]
        gap = (a - pb) / frame_rate
        has_breath = any(pb / frame_rate - 0.02 <= br.start and br.end <= a / frame_rate + 0.02
                         for br in breaths if br.status != "rejected")
        if gap < st.phrase_gap_s and not has_breath:
            groups[-1].append((a, b))
        else:
            groups.append([(a, b)])
    out = []
    syl = sorted(syllable_starts or [])
    for i, g in enumerate(groups):
        s, e = g[0][0] / frame_rate, g[-1][1] / frame_rate
        prev_end = groups[i - 1][-1][1] / frame_rate if i else None
        next_start = groups[i + 1][0][0] / frame_rate if i + 1 < len(groups) else None
        ids = [br.breath_id for br in breaths if br.status != "rejected" and prev_end is not None
               and prev_end - 0.02 <= br.start and br.end <= s + 0.02]
        ph = Phrase(f"p{i + 1}", s, e, sum(b - a for a, b in g) / frame_rate,
                    None if prev_end is None else s - prev_end, None if next_start is None else next_start - e, ids)
        if syllable_starts is not None:
            k = sum(1 for t in syl if s - 0.05 <= t < e)
            ph.syllables = k
            ph.syllables_per_s = k / ph.duration if ph.duration > 0 else None
        out.append(ph)
    return out


def apply_overrides(breaths: list[Breath], plist: list[Phrase], overrides: dict) -> tuple[list[Breath], list[Phrase]]:
    """사용자 수정 반영. overrides = {"breath_status": {id: status}, "breath_added": [[s,e]], "phrases": [[s,e], ...]|None}."""
    st = overrides.get("breath_status") or {}
    out = []
    for b in breaths:
        b2 = Breath(**{**asdict(b), "status": st.get(b.breath_id, b.status)})
        out.append(b2)
    for k, (s, e) in enumerate(overrides.get("breath_added") or [], 1):
        out.append(Breath(f"u{k}", float(s), float(e), float("nan"), float("nan"), float("nan"), 1.0, "confirmed"))
    if overrides.get("phrases"):
        plist = [Phrase(f"p{i + 1}", float(s), float(e), float(e) - float(s), None, None, [], None, None, True)
                 for i, (s, e) in enumerate(overrides["phrases"])]
    return out, plist


def summary(cls: np.ndarray, breaths: list[Breath], plist: list[Phrase], frame_rate: float = 100.0) -> dict:
    counts = {c: int(np.sum(cls == c)) for c in CLASSES}
    durs = [p.duration for p in plist]
    sps = [p.syllables_per_s for p in plist if p.syllables_per_s is not None]
    return {"frames": {k: round(v / frame_rate, 2) for k, v in counts.items()},
            "breath_candidates": sum(1 for b in breaths if b.status == "candidate"),
            "breath_confirmed": sum(1 for b in breaths if b.status == "confirmed"),
            "phrases": len(plist), "phrase_mean_s": round(float(np.mean(durs)), 2) if durs else None,
            "syllables_per_s_median": round(float(np.median(sps)), 2) if sps else None}
