"""v0.7.0 최종 합성 평가 (명세 20.1~20.3). 앱 실행에는 쓰지 않는다.

이 파일의 사례(씨앗·음높이·속도·폭·숨소리 크기·잡음)는 최종 보고를 위해 한 번 만들었고, 규칙·문턱값 조정에 쓰지 않았다.
개발 중 쓴 합성 사례(tests/, scratchpad의 점검 스크립트)와 값이 겹치지 않게 골랐다.
그래도 같은 합성기(amp.devtools.synth)로 만든 소리라 실제 노래에서의 정확도를 뜻하지 않는다.

    python -m amp.devtools.final_eval_v07 [vibrato|breath|multipitch|all]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from amp.devtools.synth import SR, backing, breath_noise, sung

# --- 비브라토 (20.2): 소리 → pYIN → 비브라토 분석 -----------------------------------------------------

VIB_RATES = (4.7, 5.9, 6.6, 7.3)
VIB_EXTENTS = (28.0, 47.0, 63.0)  # 한쪽 진폭 (센트)
VIB_BASES = (59, 66, 73)
VIB_NOTE = (0.6, 2.9)  # 음 구간 (초)


def vibrato_cases():
    for rate in VIB_RATES:
        for ext in VIB_EXTENTS:
            for base in VIB_BASES:
                yield {"name": f"{rate}Hz ±{ext:.0f}c MIDI{base}", "rate": rate, "extent": ext, "base": base, "kind": "vibrato"}
    for base in VIB_BASES:
        yield {"name": f"평탄 MIDI{base}", "rate": 0.0, "extent": 0.0, "base": base, "kind": "flat"}
        yield {"name": f"포르타멘토 MIDI{base}→{base + 3}", "rate": 0.0, "extent": 0.0, "base": base, "kind": "portamento"}


def _vibrato_audio(c, dur=3.5, seed=0):
    a, b = VIB_NOTE
    if c["kind"] == "portamento":
        mel = [(a, b, c["base"])]
        y = sung(mel, dur, vibrato_cents=0.0, bends=[(a, b, c["base"], c["base"] + 3, 0.35)])
    else:
        y = sung([(a, b, c["base"])], dur, vibrato_cents=c["extent"], vibrato_hz=c["rate"] or 5.5)
    rng = np.random.default_rng(seed + 31)
    return y + rng.standard_normal(len(y)).astype(np.float32) * 10 ** (-50 / 20)


def evaluate_vibrato() -> dict:
    from amp.engine.acoustics.vibrato import analyze_curve
    from amp.engine.dsp.pitch import track_pitch

    rows = []
    for i, c in enumerate(vibrato_cases()):
        y = _vibrato_audio(c, seed=i)
        fr = track_pitch(y, SR)
        segs = [s for s in analyze_curve(np.asarray(fr.f0_hz, float), fr.frame_rate) if s.status == "completed"]
        row = {"case": c["name"], "kind": c["kind"], "found": len(segs)}
        if c["kind"] == "vibrato" and segs:
            s = max(segs, key=lambda s: s.end - s.start)
            row.update(rate_err=round(abs(s.rate_hz - c["rate"]), 3), extent_err=round(abs(s.extent_cents - c["extent"]), 1),
                       pp_err=round(abs(s.extent_pp_cents - 2 * c["extent"]), 1),
                       start_err=round(abs(s.start - VIB_NOTE[0]), 3), end_err=round(abs(s.end - VIB_NOTE[1]), 3))
        rows.append(row)
    vib = [r for r in rows if r["kind"] == "vibrato"]
    hit = [r for r in vib if "rate_err" in r]
    rate = np.array([r["rate_err"] for r in hit])
    ext = np.array([r["extent_err"] for r in hit])
    neg = [r for r in rows if r["kind"] != "vibrato"]
    return {"cases": len(rows), "vibrato_cases": len(vib), "detected": len(hit),
            "rate_err_median_hz": round(float(np.median(rate)), 3) if len(rate) else None,
            "rate_err_max_hz": round(float(rate.max()), 3) if len(rate) else None,
            "rate_within_0.3hz": int((rate <= 0.3).sum()), "extent_err_median_c": round(float(np.median(ext)), 1) if len(ext) else None,
            "extent_err_max_c": round(float(ext.max()), 1) if len(ext) else None, "extent_within_10c": int((ext <= 10).sum()),
            "start_err_median_s": round(float(np.median([r["start_err"] for r in hit])), 3) if hit else None,
            "end_err_median_s": round(float(np.median([r["end_err"] for r in hit])), 3) if hit else None,
            "false_detections_flat": sum(r["found"] for r in neg if r["kind"] == "flat"),
            "false_detections_portamento": sum(r["found"] for r in neg if r["kind"] == "portamento"), "rows": rows}


# --- 호흡 (20.3) ---------------------------------------------------------------------------------

BREATH_MEL = [(0.4, 1.6, 62), (1.7, 3.1, 65), (3.9, 5.0, 67), (5.1, 6.6, 65), (7.6, 9.0, 62), (9.9, 11.3, 60)]
BREATH_SPANS = [(3.25, 3.75), (6.85, 7.4), (9.2, 9.7)]  # 숨 (음 사이 쉼 안)
BREATH_LEVELS = (-26.0, -34.0, -40.0)


def _breath_mix(level, with_backing, with_tail, seed):
    dur = 12.0
    y = sung(BREATH_MEL, dur)
    rng = np.random.default_rng(seed + 101)
    x = y + breath_noise(dur, BREATH_SPANS, level_db=level, seed=seed + 7)
    cons = np.zeros(len(y), np.float32)  # 무성 자음: 음 시작 바로 앞 50ms
    for a in (1.65, 5.05):
        i = int(a * SR)
        cons[i - int(0.05 * SR):i] = rng.standard_normal(int(0.05 * SR)).astype(np.float32) * 0.025
    x = x + cons
    if with_tail:  # 잔향 꼬리: 음 끝 뒤 0.25초 지수 감쇠
        tail = np.zeros(len(y), np.float32)
        for _a, b, _m in BREATH_MEL:
            i = int(b * SR)
            seg = y[max(0, i - int(0.25 * SR)):i][::-1] * np.exp(-np.arange(min(i, int(0.25 * SR))) / (0.08 * SR))
            tail[i:i + len(seg)] += seg[: len(tail) - i] * 0.4
        x = x + tail
    if with_backing:
        x = x + backing(dur, seed=seed + 3) * 10 ** (-30 / 20)
    return x + rng.standard_normal(len(y)).astype(np.float32) * 1e-4


def evaluate_breath() -> dict:
    from amp.engine.acoustics.breath import classify
    from amp.engine.acoustics.features import frame_features
    from amp.engine.dsp.pitch import to_pitch_sr, track_pitch

    rows = []
    k = 0
    for level in BREATH_LEVELS:
        for backing_on in (False, True):
            for tail in (False, True):
                x = _breath_mix(level, backing_on, tail, k)
                k += 1
                fr = track_pitch(x, SR)
                feat = frame_features(to_pitch_sr(x, SR), fr.f0_hz)
                _cls, breaths = classify(feat, np.isfinite(fr.f0_hz))
                found = [(b.start, b.end) for b in breaths]
                hit = sum(1 for a, b in BREATH_SPANS if any(s < b and e > a for s, e in found))
                false = sum(1 for s, e in found if not any(s < b and e > a for a, b in BREATH_SPANS))
                rows.append({"level_db": level, "backing": backing_on, "reverb_tail": tail, "true": len(BREATH_SPANS),
                             "detected": len(found), "hit": hit, "false": false})
    tp = sum(r["hit"] for r in rows)
    fp = sum(r["false"] for r in rows)
    total = sum(r["true"] for r in rows)
    return {"mixes": len(rows), "recall": round(tp / total, 3), "precision": round(tp / max(1, tp + fp), 3),
            "false_candidates": fp, "rows": rows}


# --- 다중 피치 (20.1): 이미 있는 평가기의 'fresh' 묶음 -----------------------------------------------

def evaluate_multipitch() -> list[dict]:
    from amp.devtools import multipitch_eval as E

    return E.evaluate("fresh")


def main(argv=None) -> int:
    what = (argv or sys.argv[1:] or ["all"])[0]
    out = {}
    if what in ("vibrato", "all"):
        out["vibrato"] = evaluate_vibrato()
    if what in ("breath", "all"):
        out["breath"] = evaluate_breath()
    if what in ("multipitch", "all"):
        out["multipitch"] = evaluate_multipitch()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
