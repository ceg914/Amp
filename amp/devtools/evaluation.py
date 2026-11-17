"""채보 결과 평가 지표.

음표: 정답 음표와 추정 음표를 시작 시각 허용 오차 안에서 1:1로 짝짓는다(음높이가 같아야 짝).
      정밀도 = 짝지은 수 / 추정 수, 재현율 = 짝지은 수 / 정답 수, F1. 짝지은 음표의 시작·끝 오차(초).
피치: 프레임별 cents 오차, 50cent 이내 비율(RPA), 옥타브 오류율, 유성 판정 재현율·오경보율.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ONSET_TOL = 0.05  # 초. mir_eval transcription 기본값과 같은 50ms
PITCH_TOL = 0.5  # 반음. 정수 음높이가 같아야 한다는 뜻


@dataclass(frozen=True)
class RefNote:
    start: float
    end: float
    pitch: int


@dataclass
class NoteScore:
    n_ref: int
    n_est: int
    matched: int
    onset_errors: list[float] = field(default_factory=list)  # 추정 − 정답 (초)
    offset_errors: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.matched / self.n_est if self.n_est else (1.0 if self.n_ref == 0 else 0.0)

    @property
    def recall(self) -> float:
        return self.matched / self.n_ref if self.n_ref else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def onset_mae_ms(self) -> float:
        return float(np.mean(np.abs(self.onset_errors)) * 1000) if self.onset_errors else float("nan")

    @property
    def offset_mae_ms(self) -> float:
        return float(np.mean(np.abs(self.offset_errors)) * 1000) if self.offset_errors else float("nan")

    def as_dict(self) -> dict:
        return {"ref": self.n_ref, "est": self.n_est, "matched": self.matched, "precision": round(self.precision, 3),
                "recall": round(self.recall, 3), "f1": round(self.f1, 3),
                "onset_mae_ms": round(self.onset_mae_ms, 1), "offset_mae_ms": round(self.offset_mae_ms, 1)}


def score_notes(ref: list[RefNote], est, onset_tol: float = ONSET_TOL, pitch_tol: float = PITCH_TOL) -> NoteScore:
    """est는 start·end·pitch 속성을 가진 객체 목록 (Note)."""
    pairs = []
    for i, r in enumerate(ref):
        for j, e in enumerate(est):
            d = abs(e.start - r.start)
            if d <= onset_tol and abs(e.pitch - r.pitch) <= pitch_tol:
                pairs.append((d, i, j))
    pairs.sort()
    used_r, used_e = set(), set()
    score = NoteScore(len(ref), len(est), 0)
    for _d, i, j in pairs:
        if i in used_r or j in used_e:
            continue
        used_r.add(i)
        used_e.add(j)
        score.matched += 1
        score.onset_errors.append(est[j].start - ref[i].start)
        score.offset_errors.append(est[j].end - ref[i].end)
    return score


def pitch_metrics(est_midi: np.ndarray, ref_midi: np.ndarray) -> dict:
    """같은 프레임 격자의 MIDI 배열 두 개 (무성 = NaN)."""
    n = min(len(est_midi), len(ref_midi))
    est, ref = np.asarray(est_midi[:n], float), np.asarray(ref_midi[:n], float)
    rv, ev = np.isfinite(ref), np.isfinite(est)
    both = rv & ev
    cents = (est[both] - ref[both]) * 100
    octave = np.abs(np.abs(est[both] - ref[both]) - 12) < 1.0
    return {
        "voiced_ref": int(rv.sum()),
        "voicing_recall": round(float(both.sum() / rv.sum()), 3) if rv.any() else float("nan"),
        "voicing_false_alarm": round(float((ev & ~rv).sum() / max(1, (~rv).sum())), 3),
        "rpa_50c": round(float(np.mean(np.abs(cents) <= 50)), 3) if cents.size else float("nan"),
        "median_abs_cents": round(float(np.median(np.abs(cents))), 1) if cents.size else float("nan"),
        "octave_error_rate": round(float(np.mean(octave)), 4) if cents.size else float("nan"),
    }
