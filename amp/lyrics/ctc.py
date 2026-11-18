"""CTC 강제 정렬 (numpy). 가사 글자열을 프레임별 로그 확률 위에 가장 그럴듯한 경로로 놓는다.

- 모델 출력이 있는 글자(로마자)는 직접 검출한 위치다(``start_source='model'``).
- 읽기를 모르는 단위(한자·숫자)는 와일드카드 '*'로 두며, 그 위치는 앞뒤 글자 사이에서 정해진 추정값이다.
- CTC는 글자가 '튀는' 짧은 프레임만 주므로 음절 끝은 직접 검출되지 않는다. 다음 음절 시작과
  보컬이 끊기는 지점으로 정하고 ``end_source='inferred'`` 로 표시한다.
- 사용자가 정한 음절(고정점)은 움직이지 않는다. 고정점 사이 구간만 따로 정렬한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NEG = -1e30
INACTIVE_PENALTY = 6.0  # 보컬이 없는 프레임에 글자를 놓을 때 깎는 로그 확률
MAX_CELLS = 120_000_000  # 역추적 표 크기 상한 (프레임 × 상태). 넘으면 거칠게 먼저 맞춘 뒤 줄 단위로 나눈다
MAX_TAIL_S = 3.0  # 음절 끝을 보컬 끊김까지 늘일 때의 상한


@dataclass(frozen=True)
class Unit:
    """정렬할 음절 하나."""

    syllable_id: str
    tokens: tuple[int, ...]  # 레이블 번호. 비어 있으면 와일드카드 하나로 정렬
    line: int = 0


@dataclass(frozen=True)
class UnitResult:
    syllable_id: str
    start: float
    end: float
    confidence: float
    start_source: str  # model / estimated
    end_source: str  # inferred
    detected_end: float  # 마지막 글자가 검출된 프레임 끝 (참고값)


class AlignmentFailed(Exception):
    pass


def viterbi(logp: np.ndarray, tokens: np.ndarray, blank: int = 0) -> np.ndarray:
    """프레임별 상태 경로. 상태 2k+1 = k번째 토큰, 짝수 = blank."""
    T = logp.shape[0]
    L = len(tokens)
    if L == 0:
        raise AlignmentFailed("정렬할 글자가 없습니다.")
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tokens
    need = L + int(np.sum(tokens[1:] == tokens[:-1]))
    if T < need:
        raise AlignmentFailed(f"구간이 너무 짧습니다 (프레임 {T}개 < 필요 {need}개).")
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    alpha = np.full(S, NEG)
    alpha[0] = logp[0, ext[0]]
    alpha[1] = logp[0, ext[1]]
    bp = np.zeros((T, S), dtype=np.int8)
    idx = np.arange(S)
    emit = logp[:, ext]  # (T, S)
    for t in range(1, T):
        a1 = np.empty(S)
        a1[0] = NEG
        a1[1:] = alpha[:-1]
        a2 = np.full(S, NEG)
        a2[2:] = alpha[:-2]
        a2[~skip] = NEG
        stack = np.stack((alpha, a1, a2))
        choice = stack.argmax(axis=0)
        alpha = stack[choice, idx] + emit[t]
        bp[t] = choice
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[s] <= NEG / 2:
        raise AlignmentFailed("가능한 정렬 경로가 없습니다.")
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


def token_spans(path: np.ndarray, n_tokens: int) -> list[tuple[int, int]]:
    """토큰 k가 차지한 프레임 [시작, 끝) 목록."""
    spans = [(-1, -1)] * n_tokens
    T = len(path)
    t = 0
    while t < T:
        s = int(path[t])
        u = t
        while u < T and path[u] == s:
            u += 1
        if s % 2 == 1:
            k = s // 2
            a, b = spans[k]
            spans[k] = (t, u) if a < 0 else (a, u)
        t = u
    return spans


def apply_activity(logp: np.ndarray, active: np.ndarray | None, blank: int, star: int) -> np.ndarray:
    """보컬이 없는 프레임에서는 글자 확률을 깎는다 (blank와 '*'는 그대로)."""
    if active is None:
        return logp
    out = logp.copy()
    mask = ~np.asarray(active, dtype=bool)[: len(out)]
    cols = [c for c in range(out.shape[1]) if c not in (blank, star)]
    out[np.ix_(np.flatnonzero(mask), cols)] -= INACTIVE_PENALTY
    return out


def _align_window(logp: np.ndarray, units: list[Unit], f0: int, f1: int, star: int, blank: int,
                  dt: float, active: np.ndarray | None, next_start: float | None) -> list[UnitResult]:
    """프레임 [f0, f1) 안에서 units를 정렬한다. 양 끝에 '*'를 둬 구간의 남는 소리를 흡수한다."""
    seq: list[int] = [star]
    owner: list[int] = [-1]
    for ui, u in enumerate(units):
        toks = u.tokens or (star,)
        for tok in toks:
            seq.append(tok)
            owner.append(ui)
    seq.append(star)
    owner.append(-1)
    window = logp[f0:f1]
    tokens = np.asarray(seq, dtype=np.int64)
    if window.shape[0] * (2 * len(tokens) + 1) > MAX_CELLS and len({u.line for u in units}) > 1:
        return _align_coarse_then_lines(logp, units, f0, f1, star, blank, dt, active, next_start)
    path = viterbi(window, tokens, blank)
    spans = token_spans(path, len(tokens))
    starts: list[int] = [-1] * len(units)
    ends: list[int] = [-1] * len(units)
    scores: list[list[float]] = [[] for _ in units]
    for k, (a, b) in enumerate(spans):
        ui = owner[k]
        if ui < 0 or a < 0:
            continue
        if starts[ui] < 0:
            starts[ui] = a
        ends[ui] = b
        if seq[k] != star:
            scores[ui].extend(window[a:b, seq[k]].tolist())
    results: list[UnitResult] = []
    for ui, u in enumerate(units):
        a, b = starts[ui], ends[ui]
        if a < 0:
            continue
        conf = float(math.exp(float(np.mean(scores[ui])))) if scores[ui] else 0.0
        results.append(UnitResult(u.syllable_id, (f0 + a) * dt, (f0 + b) * dt, conf,
                                  "model" if u.tokens else "estimated", "inferred", (f0 + b) * dt))
    return _infer_ends(results, active, dt, next_start, f1 * dt)


def _align_coarse_then_lines(logp, units, f0, f1, star, blank, dt, active, next_start) -> list[UnitResult]:
    """긴 구간: 프레임을 4배 줄여 줄 위치를 먼저 잡고, 줄마다 원래 해상도로 다시 정렬한다."""
    k = 4
    n = (f1 - f0) // k
    coarse = logp[f0:f0 + n * k].reshape(n, k, -1).max(axis=1)
    coarse = coarse - np.logaddexp.reduce(coarse, axis=1, keepdims=True)
    rough = _align_window(coarse, units, 0, n, star, blank, dt * k, None, None)
    by_id = {r.syllable_id: r for r in rough}
    lines: dict[int, list[Unit]] = {}
    for u in units:
        lines.setdefault(u.line, []).append(u)
    out: list[UnitResult] = []
    keys = sorted(lines)
    for i, ln in enumerate(keys):
        us = lines[ln]
        rs = [by_id[u.syllable_id] for u in us if u.syllable_id in by_id]
        if not rs:
            continue
        a = max(f0, f0 + int(min(r.start for r in rs) / dt) - int(1.0 / dt))
        b = min(f1, f0 + int(max(r.detected_end for r in rs) / dt) + int(1.0 / dt))
        if out:
            a = max(a, int(out[-1].detected_end / dt))
        nxt = None
        if i + 1 < len(keys):
            nr = [by_id[u.syllable_id] for u in lines[keys[i + 1]] if u.syllable_id in by_id]
            if nr:
                nxt = f0 * dt + min(r.start for r in nr)
        try:
            out.extend(_align_window(logp, us, a, max(b, a + 1), star, blank, dt, active, nxt or next_start))
        except AlignmentFailed:
            continue
    return out


def _infer_ends(results: list[UnitResult], active: np.ndarray | None, dt: float, next_start: float | None,
                window_end: float) -> list[UnitResult]:
    """음절 끝 = min(다음 음절 시작, 보컬이 끊기는 곳). 단, 검출된 마지막 글자 끝보다 앞당기지 않는다."""
    out = []
    for i, r in enumerate(results):
        limit = results[i + 1].start if i + 1 < len(results) else (next_start if next_start is not None else window_end)
        end = limit
        if active is not None:
            f = int(r.detected_end / dt)
            g = f
            fmax = min(len(active), int((r.detected_end + MAX_TAIL_S) / dt))
            while g < fmax and active[g]:
                g += 1
            end = min(limit, max(r.detected_end, g * dt))
        else:
            end = min(limit, r.detected_end + MAX_TAIL_S)
        end = max(end, r.start + 0.02)
        out.append(UnitResult(r.syllable_id, r.start, end, r.confidence, r.start_source, r.end_source, r.detected_end))
    return out


RELEASE_STEP = 8  # 정렬에 실패하면 양옆의 '풀 수 있는' 고정점을 이만큼씩 풀고 다시 시도
MAX_RELEASE_ROUNDS = 4


def align(logp: np.ndarray, units: list[Unit], *, dt: float, star: int, blank: int = 0,
          anchors: dict[int, tuple[float, float]] | None = None, active: np.ndarray | None = None,
          soft: set[int] | None = None) -> tuple[list[UnitResult], list[str], list[int]]:
    """units 전체를 정렬한다. anchors: units 순번 → 고정할 (시작, 끝) — 이 음절은 정렬하지 않고 경계로만 쓴다.

    soft: 풀어도 되는 고정점(모델이 정한 시간). 그 사이에 새 음절이 들어갈 자리가 없어 실패하면
    양옆의 soft 고정점을 풀어 함께 다시 정렬한다(가사를 고친 뒤 '바뀐 부분만' 정렬할 때).
    사용자가 고친 음절은 soft에 넣지 않으므로 풀리지 않는다.

    (결과, 경고, 풀어서 함께 정렬한 순번). 결과에 없는 음절은 정렬하지 못한 것이다.
    """
    anchors = dict(anchors or {})
    soft = set(soft or ()) & set(anchors)
    T = logp.shape[0]
    lp = apply_activity(logp, active, blank, star)
    results: list[UnitResult] = []
    done: dict[int, UnitResult] = {}
    released: list[int] = []
    warnings: list[str] = []
    n = len(units)
    i = 0
    while i < n:
        if i in anchors:
            i += 1
            continue
        j = i
        while j < n and j not in anchors:
            j += 1
        rounds = 0
        while True:
            if i - 1 in anchors:
                prev_end = anchors[i - 1][1]
            elif i - 1 in done:
                prev_end = done[i - 1].detected_end
            else:
                prev_end = 0.0
            next_start = anchors[j][0] if j in anchors else None
            f0 = max(0, int(round(prev_end / dt)))
            f1 = min(T, int(round(next_start / dt)) if next_start is not None else T)
            seg = units[i:j]
            try:
                if f1 - f0 < 2:
                    raise AlignmentFailed("고정한 음절 사이에 공간이 없습니다.")
                out = _align_window(lp, seg, f0, f1, star, blank, dt, active, next_start)
                index = {u.syllable_id: i + k for k, u in enumerate(seg)}
                for r in out:
                    done[index[r.syllable_id]] = r
                results.extend(out)
                break
            except AlignmentFailed as e:
                grown = False
                if rounds < MAX_RELEASE_ROUNDS:
                    for _ in range(RELEASE_STEP):  # 왼쪽
                        k = i - 1
                        if k in anchors and k in soft:
                            del anchors[k]
                            released.append(k)
                            i = k
                            grown = True
                        else:
                            break
                    for _ in range(RELEASE_STEP):  # 오른쪽
                        if j in anchors and j in soft:
                            del anchors[j]
                            released.append(j)
                            j += 1
                            while j < n and j not in anchors:
                                j += 1
                            grown = True
                        else:
                            break
                rounds += 1
                if not grown:
                    warnings.append(f"음절 {len(seg)}개 구간을 정렬하지 못했습니다: {e}")
                    break
        i = j
    return results, warnings, sorted(released)


def activity_mask(rms_db: np.ndarray, floor_db: float = 35.0, dilate_frames: int = 8) -> np.ndarray:
    """보컬 활동 프레임: 상위 5% 크기보다 floor_db 이상 작지 않은 곳 (앞뒤로 dilate_frames 넓힘)."""
    x = np.asarray(rms_db, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    ref = np.percentile(x, 95)
    act = x > ref - floor_db
    if dilate_frames > 0 and act.any():
        k = np.ones(2 * dilate_frames + 1)
        act = np.convolve(act.astype(float), k, mode="same") > 0
    return act
