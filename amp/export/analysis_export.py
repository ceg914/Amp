"""보컬·음향 분석 결과 내보내기 (v0.7.0 명세 18). Qt 비의존 (PNG는 app/analysis_png.py).

새 악보를 만들지 않는다. 지금 프로젝트에 있는 결과(피치 곡선·성부·비브라토·호흡·프레이즈·발성 지표·발성 경계·가사·발음·
음절 시간)를 그대로 옮겨 적는다. 값이 없는 항목은 파일을 만들지 않고 건너뛴 이유를 돌려준다.

- JSON: 전체 구조 (항목마다 공통 머리말 meta·상태를 그대로)
- CSV: 피치 프레임 · 성부별 피치 · 비브라토 · 호흡 후보 · 발성 구간(프레이즈) · 음절별 시간 (UTF-8 BOM, 엑셀에서 열림)
- TXT: 가사 · 발음 · 음절 시간 · 요약
분석 결과는 자동 추정값이다. 사용자가 고친 값과 자동 값을 열로 구분한다.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CSV_ITEMS = ("pitch", "voices", "vibrato", "breaths", "phrases", "syllables")
CSV_LABELS = {"pitch": "피치 프레임", "voices": "성부별 피치", "vibrato": "비브라토", "breaths": "호흡 후보",
              "phrases": "발성 구간(프레이즈)", "syllables": "음절별 시간"}
NOTE = "SONATLAS 자동 분석 결과입니다 (추정값). 사용자가 고친 값은 별도 열/키로 구분합니다."


@dataclass
class Sources:
    """내보낼 결과 모음 (없으면 None). 화면 쪽이 지금 프로젝트에서 채운다."""

    title: str = ""
    pitch_curve: object = None  # core.model.PitchCurve
    voices: object = None  # engine.voices.VoiceSet
    acoustics: object = None  # engine.acoustics.run.AcousticResult
    lyrics: object = None  # lyrics.model.LyricsState
    notes: list = field(default_factory=list)
    region: tuple | None = None  # (시작, 끝) 초. None = 전체


def _f(x, nd=4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _in(region, a, b=None) -> bool:
    if region is None:
        return True
    b = a if b is None else b
    return b >= region[0] and a <= region[1]


def pitch_rows(src: Sources) -> list[dict]:
    c = src.pitch_curve
    if c is None:
        return []
    f0 = np.asarray(c.f0_hz, dtype=np.float64)
    conf = np.asarray(c.confidence, dtype=np.float64)
    out = []
    for i in range(len(f0)):
        t = c.start_time + i / c.frame_rate
        if not _in(src.region, t):
            continue
        hz = f0[i]
        out.append({"time_s": round(t, 4), "f0_hz": _f(hz, 3), "midi": _f(69 + 12 * math.log2(hz / 440.0), 3)
                    if math.isfinite(hz) and hz > 0 else None, "confidence": _f(conf[i], 3)})
    return out


def voice_rows(src: Sources) -> list[dict]:
    vs = src.voices
    if vs is None:
        return []
    out = []
    for v in vs.voices:
        f0 = np.asarray(v.f0_hz, dtype=np.float64)
        conf = np.asarray(v.confidence, dtype=np.float64)
        for i in range(len(f0)):
            if not math.isfinite(f0[i]) or f0[i] <= 0:
                continue
            t = (vs.start_frame + i) / vs.frame_rate
            if not _in(src.region, t):
                continue
            out.append({"voice": v.display_name, "voice_id": v.voice_track_id, "role": v.role, "time_s": round(t, 4),
                        "f0_hz": _f(f0[i], 3), "midi": _f(69 + 12 * math.log2(f0[i] / 440.0), 3),
                        "confidence": _f(conf[i], 3), "user_confirmed": v.user_confirmed, "edited": v.edited,
                        "uncertain_link": bool(vs.uncertain[i]) if i < len(vs.uncertain) else None})
    return out


def vibrato_rows(src: Sources) -> list[dict]:
    r = src.acoustics
    if r is None:
        return []
    return [{"start_s": _f(s.start), "end_s": _f(s.end), "rate_hz": _f(s.rate_hz, 3),
             "extent_one_side_cents": _f(s.extent_cents, 2), "extent_peak_to_peak_cents": _f(s.extent_pp_cents, 2),
             "max_extent_cents": _f(s.max_extent_cents, 2), "regularity": _f(s.regularity, 3),
             "onset_delay_s": _f(s.onset_delay_s), "cycles": _f(s.cycles, 2), "status": s.status, "reason": s.reason}
            for s in r.vibrato if _in(src.region, s.start, s.end)]


def breath_rows(src: Sources) -> list[dict]:
    r = src.acoustics
    if r is None:
        return []
    return [{"breath_id": b.breath_id, "start_s": _f(b.start), "end_s": _f(b.end), "level_db": _f(b.level_db, 2),
             "score": _f(b.score, 3), "status": b.status, "user_decided": b.status in ("confirmed", "rejected")}
            for b in r.breaths if _in(src.region, b.start, b.end)]


def phrase_rows(src: Sources) -> list[dict]:
    r = src.acoustics
    if r is None:
        return []
    return [{"phrase_id": p.phrase_id, "start_s": _f(p.start), "end_s": _f(p.end), "voiced_s": _f(p.voiced_s),
             "gap_before_s": _f(p.gap_before_s), "gap_after_s": _f(p.gap_after_s), "syllables": p.syllables,
             "syllables_per_s": _f(p.syllables_per_s, 3), "user_edited": p.user_edited}
            for p in r.phrases if _in(src.region, p.start, p.end)]


def syllable_rows(src: Sources) -> list[dict]:
    ly = src.lyrics
    if ly is None or ly.text.is_empty:
        return []
    out = []
    for n_line, line in enumerate(ly.text.lines, 1):
        for w in line.words:
            for s in w.syllables:
                t = ly.alignment.timing(s.id)
                if t.aligned and not _in(src.region, t.start, t.end):
                    continue
                if not t.aligned and src.region is not None:
                    continue
                e = ly.pronunciation.get(s.id)
                out.append({"line": n_line, "word": w.text, "syllable": s.text, "syllable_id": s.id,
                            "start_s": _f(t.start) if t.aligned else None, "end_s": _f(t.end) if t.aligned else None,
                            "start_source": t.start_source if t.aligned else "", "end_source": t.end_source if t.aligned else "",
                            "time_user_edited": t.user_edited, "pronunciation": e.text if e else "",
                            "ipa": e.ipa if e else "", "pron_status": e.status if e else "",
                            "pron_user_edited": e.user_edited if e else False,
                            "linked_notes": ";".join(ly.alignment.linked_notes(s.id))})
    return out


ROWS = {"pitch": pitch_rows, "voices": voice_rows, "vibrato": vibrato_rows, "breaths": breath_rows,
        "phrases": phrase_rows, "syllables": syllable_rows}


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    if not rows:
        return ""
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return buf.getvalue()


def to_json(src: Sources) -> dict:
    from amp import __version__

    out: dict = {"schema": "sonatlas.analysis_export", "schema_version": 1, "app_version": __version__,
                 "title": src.title, "note": NOTE, "region": list(src.region) if src.region else None}
    r = src.acoustics
    if r is not None:
        out["acoustics"] = {"meta": {k: m.to_dict() for k, m in (r.meta or {}).items()},
                            "vibrato_summary": r.vibrato_summary, "breath_summary": r.breath_summary,
                            "vibrato": vibrato_rows(src), "breaths": breath_rows(src), "phrases": phrase_rows(src),
                            "phonation_events": [e.to_dict() for e in r.events if _in(src.region, e.start, e.end)],
                            "note_metrics": [m.to_dict() for m in r.note_metrics],
                            "boundaries": [b.to_dict() for b in r.boundaries if _in(src.region, b.note_start, b.note_end)],
                            "boundaries_applied": False}
    if src.voices is not None:
        vs = src.voices
        out["voices"] = {"meta": vs.meta.to_dict() if hasattr(vs.meta, "to_dict") else None,
                         "note": "피치만 있는 성부입니다 (소리 파일 아님).",
                         "voices": [{"voice_id": v.voice_track_id, "name": v.display_name, "role": v.role,
                                     "user_confirmed": v.user_confirmed, "edited": v.edited} for v in vs.voices],
                         "frames": voice_rows(src)}
    if src.pitch_curve is not None:
        out["pitch"] = {"frame_rate": src.pitch_curve.frame_rate, "frames": pitch_rows(src)}
    if src.lyrics is not None and not src.lyrics.text.is_empty:
        out["lyrics"] = {"syllables": syllable_rows(src),
                         "pron_dictionary": [asdict(e) for e in src.lyrics.pron_dictionary]}
    if src.notes:
        out["notes"] = [{"id": n.id, "start_s": _f(n.start), "end_s": _f(n.end), "pitch": n.pitch,
                         "vibrato_rate_hz": _f(n.vibrato.rate_hz) if n.vibrato else None,
                         "vibrato_extent_cents": _f(n.vibrato.extent_cents) if n.vibrato else None}
                        for n in src.notes if _in(src.region, n.start, n.end)]
    return out


def summary_text(src: Sources) -> str:
    lines = [f"SONATLAS 분석 요약 — {src.title}" if src.title else "SONATLAS 분석 요약", NOTE]
    if src.region:
        lines.append(f"구간: {src.region[0]:.2f}~{src.region[1]:.2f}초")
    c = src.pitch_curve
    if c is not None:
        f0 = np.asarray(c.f0_hz, dtype=np.float64)
        v = f0[np.isfinite(f0) & (f0 > 0)]
        if len(v):
            lo, hi = 69 + 12 * np.log2(np.percentile(v, [5, 95]) / 440.0)
            lines.append(f"피치: 유성 프레임 {len(v)}/{len(f0)} · 음역(5~95%) MIDI {lo:.1f}~{hi:.1f}")
    r = src.acoustics
    if r is not None:
        vs, bs = r.vibrato_summary or {}, r.breath_summary or {}
        lines.append(f"비브라토: {vs.get('completed', 0)}구간" + (f" · 평균 {vs['rate_hz']}Hz · 한쪽 {vs['extent_cents']}센트"
                                                            if vs.get("rate_hz") else ""))
        lines.append(f"호흡 후보 {bs.get('breath_candidates', 0)} · 확정 {bs.get('breath_confirmed', 0)} · 프레이즈 {bs.get('phrases', 0)}")
    if src.voices is not None:
        lines.append(f"성부(피치만): {len(src.voices.voices)}개 · 연결 불확실 프레임 {int(np.sum(src.voices.uncertain))}")
    if src.notes:
        lines.append(f"음표: {len(src.notes)}개")
    return "\n".join(lines)


def lyrics_text(src: Sources) -> str:
    """가사 · 발음 · 음절 시간 (사람이 읽는 표)."""
    rows = syllable_rows(src)
    if not rows:
        return ""
    out, line = [], None
    for r in rows:
        if r["line"] != line:
            if line is not None:
                out.append("")
            line = r["line"]
            out.append(f"[{line}줄]")
        tm = (f"{r['start_s']:.3f}~{r['end_s']:.3f}" if r["start_s"] is not None else "시간 없음")
        mark = (" (발음 직접 고침)" if r["pron_user_edited"] else "") + (" (시간 직접 고침)" if r["time_user_edited"] else "")
        out.append(f"{r['syllable']}\t[{r['pronunciation']}]\t/{r['ipa']}/\t{tm}{mark}")
    return "\n".join(out)


def export_all(src: Sources, folder: str | Path, base: str, kinds: tuple = ("json", "csv", "txt"),
               csv_items: tuple = CSV_ITEMS) -> tuple[list[Path], dict[str, str]]:
    """폴더에 파일들을 쓴다. (쓴 파일, 건너뛴 항목 → 이유)."""
    from amp.core.fileio import atomic_write_bytes

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: dict[str, str] = {}

    def put(name: str, text: str, bom: bool = True) -> None:
        p = folder / name
        data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
        atomic_write_bytes(p, (b"\xef\xbb\xbf" if bom else b"") + data)
        written.append(p)

    if "json" in kinds:
        p = folder / f"{base}-analysis.json"
        atomic_write_bytes(p, json.dumps(to_json(src), ensure_ascii=False, indent=1).encode("utf-8"))
        written.append(p)
    if "csv" in kinds:
        for item in csv_items:
            rows = ROWS[item](src)
            if not rows:
                skipped[f"csv:{item}"] = f"{CSV_LABELS[item]} 결과가 없습니다."
                continue
            put(f"{base}-{item}.csv", csv_text(rows))
    if "txt" in kinds:
        put(f"{base}-summary.txt", summary_text(src))
        lt = lyrics_text(src)
        if lt:
            put(f"{base}-lyrics.txt", lt)
        else:
            skipped["txt:lyrics"] = "가사가 없습니다."
    return written, skipped
