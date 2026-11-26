"""다중 피치·성부 추적 합성 평가 (v0.7.0, 명세 20.1). 앱 실행에는 쓰지 않는다.

사례: 단일 목소리, 2개 기본주파수(사인), 배음 있는 두 목소리(3도·5도), 옥타브, 유니즌, 성부 교차, 성부 중단, 반주 유입, 세 목소리.
- 'dev' 묶음은 알고리즘 문턱값을 고를 때 쓴 사례다.
- 'holdout' 묶음은 씨앗·선율·모음·음량·비브라토를 바꿔 따로 만들었다. 처음에는 보지 않았지만, 순음 두 개 사례의 실패를 보고
  봉우리 다듬기(REFINE_MAX_CENTS)·같은 높이 제외 폭(SAME_CENTS)을 고쳤으므로 완전히 독립된 검증이 아니다.
- 'fresh' 묶음은 마지막 보고에 한 번만 쓴다 (개발 중 결과를 보지 않음).
모두 합성 자료이며 실제 노래의 정확도를 뜻하지 않는다.

지표:
- 프레임 다중 피치 P/R/F1: 정답 f0와 50센트 안이면 맞음 (한 추정은 한 정답에만)
- 센트 오차(맞은 것의 중앙값), 허위 피치(정답이 없는 프레임·초과 추정)
- 성부 정확도: 정답 성부마다 가장 많이 겹치는 추정 성부를 짝지어, 정답 유성 프레임 중 그 성부가 50센트 안인 비율
- 성부 전환 오류: 정답 성부를 따라가는 추정 성부가 바뀐 횟수 (구간 사이 쉼은 셈하지 않음)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amp.devtools.synth import SR, VOWELS, envelope, formant_voice, melody_f0

TOL = 50.0


@dataclass
class Case:
    name: str
    y16: np.ndarray
    refs: list  # 프레임 격자 f0 (NaN = 무성), 목소리마다
    max_voices: int
    split: str  # dev / holdout


def _voice(mel, dur, seed, vowel, vib, vib_hz, glide_s=0.06, bends=None):
    kw = {"vibrato_cents": vib, "vibrato_hz": vib_hz}
    if bends is not None:
        kw["bends"] = bends
    f0 = melody_f0(mel, dur, SR, glide_s=glide_s, **kw)
    x = formant_voice(f0, SR, seed=seed, vowel=VOWELS[vowel % len(VOWELS)])
    return x / (np.abs(x).max() + 1e-12) * envelope(mel, dur, SR), f0


def _frames(f0: np.ndarray, n: int) -> np.ndarray:
    idx = np.clip((np.arange(n) * SR / 100).astype(int), 0, len(f0) - 1)
    f = f0[idx]
    return np.where(f > 0, f, np.nan)


def _mix(parts, dur, extra=None):
    from amp.engine.dsp.pitch import to_pitch_sr

    y = np.zeros(int(dur * SR))
    refs = []
    for x, f0, gain in parts:
        y += gain * x
        refs.append(f0)
    if extra is not None:
        y += extra
    y16 = to_pitch_sr(y.astype(np.float32), SR)
    n = 1 + len(y16) // 160
    return y16, [_frames(f, n) for f in refs]


def _sines(freqs_spans, dur):
    t = np.arange(int(dur * SR)) / SR
    parts = []
    for f, a, b in freqs_spans:
        x = np.where((t >= a) & (t < b), np.sin(2 * np.pi * f * t), 0.0)
        f0 = np.where((t >= a) & (t < b), f, 0.0)
        parts.append((x, f0, 0.4))
    return parts


def _accompaniment(dur, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for f in (130.8, 196.0, 261.6):  # 피아노 비슷한 감쇠 화음 (2초마다)
        for start in np.arange(0, dur, 2.0):
            m = t >= start
            out[m] += 0.08 * np.sin(2 * np.pi * f * (t[m] - start)) * np.exp(-(t[m] - start) * 1.5)
    out += 0.01 * rng.standard_normal(len(t))
    return out


def cases(split: str = "dev") -> list[Case]:
    D = 4.0
    if split == "dev":
        s, v, vib, vh, g2 = 0, 0, 30.0, 5.5, 0.8
        mel = {"single": [(0.3, 3.7, 64)], "a": [(0.3, 1.9, 67), (2.0, 3.7, 69)], "b": [(0.3, 1.9, 64), (2.0, 3.7, 65)],
               "fa": [(0.3, 3.7, 69)], "fb": [(0.3, 3.7, 62)], "oa": [(0.3, 3.7, 69)], "ob": [(0.3, 3.7, 57)],
               "u": [(0.3, 3.7, 64)], "t3": ([(0.3, 3.7, 72)], [(0.3, 3.7, 67)], [(0.3, 3.7, 64)]),
               "da": [(0.3, 3.7, 69)], "db": [(0.3, 1.5, 64), (2.5, 3.7, 62)], "sine": (440.0, 330.0)}
    elif split == "fresh":
        s, v, vib, vh, g2 = 13, 2, 20.0, 4.8, 0.9
        mel = {"single": [(0.35, 1.2, 69), (1.3, 2.4, 67), (2.5, 3.65, 65)], "a": [(0.35, 2.0, 74), (2.1, 3.65, 72)],
               "b": [(0.35, 2.0, 70), (2.1, 3.65, 69)], "fa": [(0.35, 3.65, 71)], "fb": [(0.35, 3.65, 64)],
               "oa": [(0.35, 3.65, 66)], "ob": [(0.35, 3.65, 54)], "u": [(0.35, 3.65, 62)],
               "t3": ([(0.35, 3.65, 76)], [(0.35, 3.65, 72)], [(0.35, 3.65, 69)]),
               "da": [(0.35, 3.65, 67)], "db": [(0.35, 1.6, 63), (2.6, 3.65, 60)], "sine": (523.3, 349.2)}
    else:
        s, v, vib, vh, g2 = 7, 1, 45.0, 6.2, 0.65
        mel = {"single": [(0.4, 1.8, 62), (1.9, 3.6, 66)], "a": [(0.4, 1.7, 71), (1.8, 3.6, 72)],
               "b": [(0.4, 1.7, 67), (1.8, 3.6, 69)], "fa": [(0.4, 3.6, 67)], "fb": [(0.4, 3.6, 60)],
               "oa": [(0.4, 3.6, 64)], "ob": [(0.4, 3.6, 52)], "u": [(0.4, 3.6, 67)],
               "t3": ([(0.4, 3.6, 74)], [(0.4, 3.6, 71)], [(0.4, 3.6, 67)]),
               "da": [(0.4, 3.6, 72)], "db": [(0.4, 1.3, 67), (2.4, 3.6, 64)], "sine": (392.0, 293.7)}
    out = []

    def add(name, parts, mv, extra=None):
        y16, refs = _mix(parts, D, extra)
        out.append(Case(name, y16, refs, mv, split))

    x, f = _voice(mel["single"], D, s, v, vib, vh)
    add("단일 목소리", [(x, f, 1.0)], 2)
    f1, f2 = mel["sine"]
    add("사인 2개", _sines([(f1, 0.3, 3.7), (f2, 0.3, 3.7)], D), 2)
    xa, fa = _voice(mel["a"], D, s, v, vib, vh)
    xb, fb = _voice(mel["b"], D, s + 1, v + 2, vib, vh * 0.93)
    add("두 목소리 3도", [(xa, fa, 1.0), (xb, fb, g2)], 2)
    xa, fa = _voice(mel["fa"], D, s, v, vib, vh)
    xb, fb = _voice(mel["fb"], D, s + 1, v + 3, vib, vh * 1.07)
    add("두 목소리 5도", [(xa, fa, 1.0), (xb, fb, g2)], 2)
    xa, fa = _voice(mel["oa"], D, s, v + 1, vib, vh)
    xb, fb = _voice(mel["ob"], D, s + 1, v + 2, vib, vh * 0.9)
    add("옥타브", [(xa, fa, 1.0), (xb, fb, g2)], 2)
    xa, fa = _voice(mel["u"], D, s, v, vib, vh)
    xb, fb = _voice(mel["u"], D, s + 1, v + 2, vib, vh * 1.1)
    add("유니즌", [(xa, fa, 1.0), (xb, fb, g2)], 2)
    # 교차: 한 목소리는 올라가고 다른 목소리는 내려간다 (가운데에서 만남)
    lo, hi = {"dev": (60, 72), "holdout": (62, 71), "fresh": (57, 67)}[split]
    xa, fa = _voice([(0.3, 3.7, lo)], D, s, v, 0.0, vh, bends=[(0.3, 3.7, lo, hi, 0.0)])
    xb, fb = _voice([(0.3, 3.7, hi)], D, s + 1, v + 2, 0.0, vh, bends=[(0.3, 3.7, hi, lo, 0.0)])
    add("성부 교차", [(xa, fa, 1.0), (xb, fb, g2)], 2)
    xa, fa = _voice(mel["da"], D, s, v, vib, vh)
    xb, fb = _voice(mel["db"], D, s + 1, v + 2, vib, vh * 0.95)
    add("성부 중단", [(xa, fa, 1.0), (xb, fb, g2)], 2)
    xa, fa = _voice(mel["a"], D, s, v, vib, vh)
    xb, fb = _voice(mel["b"], D, s + 1, v + 2, vib, vh * 0.93)
    add("반주 유입", [(xa, fa, 1.0), (xb, fb, g2)], 2, extra=_accompaniment(D, s))
    parts = []
    for i, m in enumerate(mel["t3"]):
        x, f = _voice(m, D, s + i, v + i, vib, vh * (1 + 0.05 * i))
        parts.append((x, f, 1.0 if i == 0 else g2))
    add("세 목소리", parts, 3)
    return out


def frame_scores(est: np.ndarray, refs: list) -> dict:
    """est (T, K) 추정 f0, refs 목소리별 (T,)."""
    T = est.shape[0]
    tp = fp = fn = 0
    errs = []
    false_frames = 0
    for t in range(T):
        e = [x for x in est[t] if np.isfinite(x)]
        g = [r[t] for r in refs if t < len(r) and np.isfinite(r[t])]
        used = set()
        for gt in g:
            best, bi = None, None
            for i, x in enumerate(e):
                if i in used:
                    continue
                c = abs(1200 * np.log2(x / gt))
                if c <= TOL and (best is None or c < best):
                    best, bi = c, i
            if bi is None:
                fn += 1
            else:
                tp += 1
                used.add(bi)
                errs.append(best)
        extra = len(e) - len(used)
        fp += extra
        if not g and e:
            false_frames += 1
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
            "cents_median": round(float(np.median(errs)), 1) if errs else None, "false_pitches": int(fp),
            "false_frames_no_voice": int(false_frames)}


def voice_scores(voices: list, refs: list) -> dict:
    """voices: 추정 성부 f0 배열들. 정답 성부마다 짝 성부의 정확도와 전환 횟수."""
    acc, switches = [], 0
    for r in refs:
        on = np.isfinite(r)
        if not on.any():
            continue
        match = np.zeros((len(voices), len(r)), dtype=bool)
        for i, v in enumerate(voices):
            n = min(len(v), len(r))
            with np.errstate(divide="ignore", invalid="ignore"):
                match[i, :n] = np.isfinite(v[:n]) & (np.abs(1200 * np.log2(v[:n] / r[:n])) <= TOL)
        if len(voices) == 0:
            acc.append(0.0)
            continue
        best = int(np.argmax(match[:, on].sum(axis=1)))
        acc.append(float(match[best, on].mean()))
        who = np.where(match.any(axis=0), np.argmax(match, axis=0), -1)[on]
        who = who[who >= 0]
        switches += int(np.count_nonzero(np.diff(who) != 0)) if len(who) > 1 else 0
    return {"voice_accuracy": round(float(np.mean(acc)), 3) if acc else None, "voice_switches": int(switches)}


def evaluate(split: str = "dev") -> list[dict]:
    from amp.core import result_meta as RM
    from amp.engine import voices as V
    from amp.engine.dsp.multipitch import candidates

    rows = []
    for c in cases(split):
        cand = candidates(c.y16, c.max_voices)
        vs = V.analyze(cand, RM.make_meta(V.SCHEMA, 1, "0" * 64), c.max_voices)
        rows.append({"case": c.name, "split": split, **frame_scores(cand.f0_hz, c.refs),
                     **voice_scores([v.f0_hz for v in vs.voices], c.refs), "voices_found": len(vs.voices),
                     "uncertain_frames": int(vs.uncertain.sum())})
    return rows
