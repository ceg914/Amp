"""품질 평가 (v0.6). Qt 비의존.

- 정답이 없으면 '지표'만 낸다(음표 수, 유성 구간 덮음 비율, 검토 필요 수 …). 정확도라고 부르지 않는다.
- 참조 악보를 정렬해 두었으면 그 악보와 비교한 P/R/F1·시작·끝 오차·센트 오차·일치율을 낸다. 악보는 가수가 실제로 부른
  것과 다를 수 있다는 단서를 함께 붙인다.
- Hybrid면 구간마다 어느 모델 결과를 썼는지 함께 보여 준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REFERENCE_CAVEAT = ("참조 악보와의 비교입니다. 악보는 가수가 실제로 부른 리듬·꾸밈·음높이와 다를 수 있어, 이 값을 "
                    "채보 정확도로 보면 안 됩니다. 악보 정렬(시작 위치·템포)의 오차도 섞여 있습니다.")
NO_REFERENCE_NOTE = "정답(참조 악보)이 없어 정확도를 계산하지 않았습니다. 아래 값은 검토를 돕는 지표입니다."


@dataclass
class QualityReport:
    indicators: dict = field(default_factory=dict)  # 이름 → 값
    counts: dict = field(default_factory=dict)  # 대시보드 숫자: 키 → (이름, 개수, 검토 목록 필터 종류 목록)
    reference: dict | None = None  # 참조 악보 비교 (없으면 None)
    sources: list = field(default_factory=list)  # Hybrid 구간별 출처 [(시작, 끝, 모델 이름)]
    notes: list = field(default_factory=list)  # 안내 문장


# 대시보드 숫자 → 검토 목록에서 보여 줄 항목 종류
COUNT_KINDS = {
    "review_notes": ("검토 필요 음표", ("note_low_conf", "note_short", "note_leap", "note_octave", "note_repeat_merged",
                                    "note_oversplit", "note_portamento", "voiced_uncovered", "sep_novocal_notes", "sep_leak")),
    "unlinked": ("음표 없는 음절", ("syl_unlinked", "syl_unaligned")),
    "pron": ("발음 검토", ("pron_unavailable", "pron_stale")),
    "melisma": ("멜리스마 불확실", ("mel_uncertain", "mel_review")),
    "model_diff": ("모델 차이 큰 구간", ("model_only_a", "model_only_b", "model_pitch", "model_octave", "model_curve",
                                    "model_voiced_a", "model_voiced_b", "model_repeat")),
    "alignment": ("음절 정렬 검토", ("syl_crowded", "syl_overlap", "syl_low_conf", "link_note_moved", "onset_gap",
                                  "link_uncertain", "link_review")),
}


def assess(project, review_rows=(), hybrid_labels: dict | None = None) -> QualityReport:
    """review_rows: (ReviewItem, 상태) 목록 (검토 탭과 같은 것). hybrid_labels: 소스 키 → 이름."""
    rep = QualityReport()
    notes = sorted(project.score.notes(), key=lambda n: n.start)
    a = project.analysis
    ind = rep.indicators
    ind["음표 수"] = len(notes)
    if notes:
        span = max(n.end for n in notes) - min(n.start for n in notes)
        ind["분당 음표"] = round(60 * len(notes) / max(span, 1e-6), 1)
        ind["100ms 미만 음표 비율"] = round(sum(1 for n in notes if n.duration < 0.1) / len(notes), 3)
        status = [project.note_status(n) for n in notes]
        ind["사용자 수정·추가 음표"] = sum(1 for s in status if s != "original")
    if a is not None and a.pitch_curve is not None and len(a.pitch_curve):
        c = a.pitch_curve
        midi = np.asarray(c.midi(), dtype=float)
        voiced = np.isfinite(midi)
        ind["유성 프레임 비율"] = round(float(voiced.mean()), 3)
        if voiced.any():
            covered = np.zeros(len(midi), dtype=bool)
            for n in notes:
                i, j = int((n.start - c.start_time) * c.frame_rate), int((n.end - c.start_time) * c.frame_rate)
                covered[max(0, i):max(0, j)] = True
            ind["유성 프레임 중 음표가 덮은 비율"] = round(float((covered & voiced).sum() / voiced.sum()), 3)
            conf = np.asarray(c.confidence, dtype=float)[voiced]
            ind["유성 프레임 신뢰도 중앙값"] = round(float(np.median(conf)), 3)
        sep = a.models.get("separation")
        if sep is not None:
            ind["분리 소스"] = sep.name
        nm = a.models.get("notes")
        if nm is not None:
            ind["음표 검출 버전"] = nm.version
    ly = project.lyrics
    if not ly.is_empty:
        n_syl = len(ly.text.syllables)
        ind["음절 수"] = n_syl
        ind["음표와 연결된 음절 비율"] = round((n_syl - len(ly.unlinked())) / max(1, n_syl), 3)
    sug = project.suggestions
    if sug is not None:
        from amp.compare import suggestions as S

        counts = {}
        for s in sug.items:
            st = S.OUTDATED if s.status == S.PENDING and S.is_outdated(s, project) else s.status
            counts[st] = counts.get(st, 0) + 1
        ind["교정 제안 대기"] = counts.get(S.PENDING, 0)
        ind["교정 제안 적용"] = counts.get(S.ACCEPTED, 0)
    rows = list(review_rows)
    for key, (label, kinds) in COUNT_KINDS.items():
        n = sum(1 for it, st in rows if it.kind in kinds and st in ("open", "recheck"))
        rep.counts[key] = (label, n, kinds)
    reviewed = sum(1 for _it, st in rows if st in ("checked", "fixed", "ignored"))
    rep.counts["reviewed"] = ("검토 완료", reviewed, ())
    rep.counts["total"] = ("검토 항목 전체", len(rows), ())
    if project.hybrid is not None:
        labels = hybrid_labels or {}
        for s in project.hybrid.segments:
            a0, b0 = project.hybrid.span(s)
            rep.sources.append((round(a0, 3), round(b0, 3), labels.get(s.source_key, s.source_model)))
    rep.reference = reference_scores(project)
    rep.notes.append(REFERENCE_CAVEAT if rep.reference else NO_REFERENCE_NOTE)
    return rep


def reference_scores(project, tolerances=(0.05, 0.1)) -> dict | None:
    """정렬한 참조 악보가 있으면 비교 값. 음높이는 같은 MIDI일 때만 맞은 것으로 본다."""
    ref = project.reference
    if ref is None or getattr(ref, "alignment", None) is None or not ref.notes:
        return None
    from amp.devtools.evaluation import RefNote, score_notes

    try:
        spans = ref.aligned_notes()
    except Exception:  # noqa: BLE001 - 정렬이 깨졌으면 비교하지 않는다
        return None
    refs = [RefNote(a, b, n.pitch) for n, a, b in spans if b > a]
    if not refs:
        return None
    lo, hi = min(r.start for r in refs) - 0.5, max(r.end for r in refs) + 0.5
    est = [n for n in project.score.notes() if lo <= n.start <= hi]
    out = {"ref_notes": len(refs), "est_notes": len(est), "caveat": REFERENCE_CAVEAT}
    for tol in tolerances:
        sc = score_notes(refs, est, onset_tol=tol)
        out[f"onset±{int(tol * 1000)}ms"] = {
            "precision": round(sc.precision, 3), "recall": round(sc.recall, 3), "f1": round(sc.f1, 3),
            "matched": sc.matched, "onset_mae_ms": _r(sc.onset_mae_ms), "offset_mae_ms": _r(sc.offset_mae_ms)}
    a = project.analysis
    if a is not None and a.pitch_curve is not None and len(a.pitch_curve):
        c = a.pitch_curve
        midi = np.asarray(c.midi(), dtype=float)
        cents = []
        for r in refs:
            i, j = int((r.start - c.start_time) * c.frame_rate), int((r.end - c.start_time) * c.frame_rate)
            tr = (j - i) // 5
            seg = midi[max(0, i + tr):max(0, j - tr)]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                cents.append((float(np.median(seg)) - r.pitch) * 100)
        if cents:
            ab = np.abs(np.array(cents))
            out["pitch_curve_vs_reference"] = {"notes_with_pitch": len(cents), "median_abs_cents": round(float(np.median(ab)), 1),
                                               "within_50c": round(float(np.mean(ab <= 50)), 3)}
    return out


def _r(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 1)


def model_table(outputs: dict, labels: dict) -> list[dict]:
    """모델별 같은 지표 (정답 없이). outputs: 키 → TranscriptionOutput."""
    rows = []
    for key, out in outputs.items():
        r = getattr(out, "result", None)
        if r is None:
            continue
        row = {"model": labels.get(key, key), "notes": len(r.notes)}
        c = r.pitch_curve
        if c is not None and len(c):
            midi = np.asarray(c.midi(), dtype=float)
            v = np.isfinite(midi)
            row["voiced"] = round(float(v.mean()), 3)
            row["conf_median"] = round(float(np.median(np.asarray(c.confidence)[v])), 3) if v.any() else None
        tm = getattr(out, "timings", {}) or {}
        row["seconds"] = round(float(tm.get("total", 0.0)), 1)
        res = (getattr(out, "resources", {}) or {}).get("separation", {})
        if res.get("gpu_peak_mb") is not None:
            row["gpu_peak_mb"] = res.get("gpu_peak_mb")
        rows.append(row)
    return rows
