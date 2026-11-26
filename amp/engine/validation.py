"""검증 자료 (v0.7.0 명세 19). Qt 비의존.

사용자가 10~30초 구간을 골라, 직접 확인하고 고친 결과(피치·성부·발성 경계·비브라토·호흡·가사 음절 시간)를 '정답'으로
묶어 저장한다. 나중에 다시 분석한 결과와 비교해 숫자를 낸다.

- 자료 종류를 반드시 적는다: 합성 / 개발용 실제 곡 / 독립 검증 / 주관적 청취 평가.
  개발(규칙 조정)에 쓴 자료는 독립 검증으로 쓰지 않는다: 독립 검증 자료로 잰 값에 개발 표시를 남기면 비교 결과에 경고를 단다
- 프로젝트 폴더 validation/ 안에만 둔다. 모델 학습 자료로 쓰거나 밖으로 보내지 않는다 (그런 기능이 없다)
- 저장한 자료는 사용자가 지우라고 할 때만 지운다
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "sonatlas.validation"
SCHEMA_VERSION = 1
DIR_NAME = "validation"
CATEGORIES = {"synthetic": "합성 데이터", "dev_real": "개발용 실제 곡", "independent": "독립 검증", "listening": "주관적 청취 평가"}
ITEMS = {"pitch": "피치 후보 확정", "voices": "성부 연결 확정", "boundaries": "발성 경계 확정", "vibrato": "비브라토 구간 확정",
         "breaths": "호흡 구간 확정", "syllables": "가사 및 음절 시간 확정"}
MIN_S, MAX_S = 10.0, 30.0
POLICY = "이 프로젝트 안에만 보관합니다. 모델 학습 자료로 쓰거나 외부로 보내지 않습니다."


class ValidationError(ValueError):
    pass


@dataclass
class ValidationSet:
    set_id: str
    category: str
    start: float
    end: float
    items: dict  # 종류 → 확정한 자료
    created: str = ""
    note: str = ""
    source: dict = field(default_factory=dict)  # 원본 해시·분석 입력
    used_for_development: bool = False  # 이 자료를 보고 규칙을 고친 적이 있음 (그러면 독립 검증이 아니다)
    policy: str = POLICY

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, **asdict(self)}

    @property
    def label(self) -> str:
        return f"{CATEGORIES.get(self.category, self.category)} {self.start:.1f}~{self.end:.1f}초 ({', '.join(ITEMS[k] for k in self.items)})"


def _f(x, nd=4):
    if x is None:
        return None
    v = float(x)
    return round(v, nd) if math.isfinite(v) else None


def _within(a, b, s, e) -> bool:
    return b >= s and a <= e


def snapshot(start: float, end: float, category: str, chosen, *, pitch_curve=None, voices=None, acoustics=None,
             lyrics=None, note: str = "", source: dict | None = None) -> ValidationSet:
    """지금 결과(사용자가 확인·수정한 상태)를 구간만 잘라 검증 자료로."""
    if category not in CATEGORIES:
        raise ValidationError(f"알 수 없는 자료 종류: {category}")
    dur = end - start
    if not (MIN_S - 1e-6 <= dur <= MAX_S + 1e-6):
        raise ValidationError(f"구간은 {MIN_S:.0f}~{MAX_S:.0f}초여야 합니다 (지금 {dur:.1f}초).")
    chosen = [k for k in chosen if k in ITEMS]
    if not chosen:
        raise ValidationError("확정할 항목을 하나 이상 고르세요.")
    items: dict = {}
    missing = []
    for k in chosen:
        if k == "pitch" and pitch_curve is not None:
            n = len(pitch_curve.f0_hz)
            ts = pitch_curve.start_time + np.arange(n) / pitch_curve.frame_rate
            m = (ts >= start) & (ts <= end)
            items[k] = {"frame_rate": pitch_curve.frame_rate, "times": [round(float(t), 4) for t in ts[m]],
                        "f0_hz": [_f(v, 3) for v in np.asarray(pitch_curve.f0_hz)[m]]}
        elif k == "voices" and voices is not None:
            out = []
            for v in voices.voices:
                ts = (voices.start_frame + np.arange(len(v.f0_hz))) / voices.frame_rate
                m = (ts >= start) & (ts <= end)
                out.append({"voice_id": v.voice_track_id, "name": v.display_name, "role": v.role,
                            "times": [round(float(t), 4) for t in ts[m]], "f0_hz": [_f(x, 3) for x in np.asarray(v.f0_hz)[m]]})
            items[k] = out
        elif k == "boundaries" and acoustics is not None:
            items[k] = [b.to_dict() for b in acoustics.boundaries if _within(b.note_start, b.note_end, start, end)]
        elif k == "vibrato" and acoustics is not None:
            items[k] = [{"start": _f(s.start), "end": _f(s.end), "rate_hz": _f(s.rate_hz, 3), "extent_cents": _f(s.extent_cents, 2)}
                        for s in acoustics.vibrato if s.status == "completed" and _within(s.start, s.end, start, end)]
        elif k == "breaths" and acoustics is not None:
            items[k] = [{"start": _f(b.start), "end": _f(b.end)} for b in acoustics.breaths
                        if b.status != "rejected" and _within(b.start, b.end, start, end)]
        elif k == "syllables" and lyrics is not None and not lyrics.text.is_empty:
            rows = []
            for s in lyrics.text.syllables:
                t = lyrics.alignment.timing(s.id)
                if t.aligned and _within(t.start, t.end, start, end):
                    rows.append({"syllable_id": s.id, "text": s.text, "start": _f(t.start), "end": _f(t.end)})
            items[k] = rows
        else:
            missing.append(ITEMS[k])
    if missing:
        raise ValidationError("결과가 없어 확정할 수 없는 항목: " + ", ".join(missing))
    return ValidationSet(uuid.uuid4().hex[:12], category, round(start, 3), round(end, 3), items,
                         datetime.now(timezone.utc).isoformat(timespec="seconds"), note, dict(source or {}))


def folder(project_root: str | Path) -> Path:
    return Path(project_root) / DIR_NAME


def save(project_root: str | Path, vs: ValidationSet) -> Path:
    from amp.core.fileio import atomic_write_json

    d = folder(project_root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{vs.set_id}.json"
    atomic_write_json(p, vs.to_json())
    return p


def load_all(project_root: str | Path) -> list[ValidationSet]:
    out = []
    d = folder(project_root)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            if j.get("schema") != SCHEMA:
                continue
            out.append(ValidationSet(j["set_id"], j["category"], float(j["start"]), float(j["end"]), dict(j["items"]),
                                     j.get("created", ""), j.get("note", ""), dict(j.get("source") or {}),
                                     bool(j.get("used_for_development", False)), j.get("policy", POLICY)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def delete(project_root: str | Path, set_id: str) -> bool:
    """사용자가 확인한 뒤에만 부른다."""
    p = folder(project_root) / f"{set_id}.json"
    if p.is_file():
        p.unlink()
        return True
    return False


# --- 비교 ----------------------------------------------------------------------------

def _cents(a, b):
    return 1200.0 * np.log2(np.asarray(a, float) / np.asarray(b, float))


def compare_pitch(ref: dict, curve) -> dict:
    t = np.asarray(ref["times"], float)
    f_ref = np.array([np.nan if v is None else v for v in ref["f0_hz"]], float)
    if curve is None or not len(t):
        return {"available": False}
    ts = curve.start_time + np.arange(len(curve.f0_hz)) / curve.frame_rate
    f_est = np.interp(t, ts, np.nan_to_num(np.asarray(curve.f0_hz, float), nan=0.0), left=0.0, right=0.0)
    rv, ev = f_ref > 0, f_est > 0
    both = rv & ev
    err = np.abs(_cents(f_est[both], f_ref[both])) if both.any() else np.array([])
    return {"available": True, "frames": int(len(t)), "ref_voiced": int(rv.sum()),
            "voicing_recall": _f(both.sum() / max(1, rv.sum()), 3), "voicing_false_alarm": _f((ev & ~rv).sum() / max(1, (~rv).sum()), 3),
            "raw_pitch_accuracy_50c": _f((err <= 50).sum() / max(1, rv.sum()), 3), "median_cents_error": _f(np.median(err), 1)
            if len(err) else None}


def _match(ref: list, est: list, key: str = "start", tol: float = 0.15) -> list[tuple[dict, object]]:
    pairs, used = [], set()
    for r in ref:
        best = None
        for i, e in enumerate(est):
            if i in used:
                continue
            d = abs(getattr(e, key) - r[key]) if not isinstance(e, dict) else abs(e[key] - r[key])
            if d <= tol and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            used.add(best[1])
            pairs.append((r, est[best[1]]))
    return pairs


def compare(vs: ValidationSet, *, pitch_curve=None, acoustics=None, lyrics=None) -> dict:
    """검증 자료와 지금 결과 비교. 개발용 자료면 결과에 '독립 검증 아님'을 붙인다."""
    out: dict = {"set_id": vs.set_id, "category": vs.category, "region": [vs.start, vs.end],
                 "independent": vs.category == "independent" and not vs.used_for_development}
    if vs.category != "independent" or vs.used_for_development:
        out["warning"] = "독립 검증 자료가 아닙니다 (개발·합성·주관 평가 자료). 정확도 주장에 쓰지 마세요."
    it = vs.items
    if "pitch" in it:
        out["pitch"] = compare_pitch(it["pitch"], pitch_curve)
    if "vibrato" in it and acoustics is not None:
        est = [s for s in acoustics.vibrato if s.status == "completed"]
        pairs = _match(it["vibrato"], est, tol=0.3)
        out["vibrato"] = {"ref": len(it["vibrato"]), "matched": len(pairs),
                          "rate_error_hz": _f(np.median([abs(e.rate_hz - r["rate_hz"]) for r, e in pairs]), 3) if pairs else None,
                          "extent_error_cents": _f(np.median([abs(e.extent_cents - r["extent_cents"]) for r, e in pairs]), 1)
                          if pairs else None}
    if "breaths" in it and acoustics is not None:
        est = [b for b in acoustics.breaths if b.status != "rejected"]
        pairs = _match(it["breaths"], est, tol=0.15)
        n_est = sum(1 for b in est if _within(b.start, b.end, vs.start, vs.end))
        out["breaths"] = {"ref": len(it["breaths"]), "detected": n_est, "matched": len(pairs),
                          "recall": _f(len(pairs) / max(1, len(it["breaths"])), 3), "precision": _f(len(pairs) / max(1, n_est), 3)}
    if "boundaries" in it and acoustics is not None:
        by_id = {b.note_id: b for b in acoustics.boundaries}
        errs = {}
        for fld in ("acoustic_onset", "voiced_onset", "stable_pitch_onset", "voiced_end"):
            d = [abs(getattr(by_id[r["note_id"]], fld) - r[fld]) for r in it["boundaries"]
                 if r.get(fld) is not None and r["note_id"] in by_id and getattr(by_id[r["note_id"]], fld) is not None]
            errs[fld] = _f(np.median(d) * 1000, 1) if d else None
        out["boundaries_median_error_ms"] = errs
    if "syllables" in it and lyrics is not None:
        d = []
        for r in it["syllables"]:
            t = lyrics.alignment.timing(r["syllable_id"])
            if t.aligned:
                d.append(abs(t.start - r["start"]))
        out["syllables"] = {"ref": len(it["syllables"]), "aligned": len(d),
                            "median_start_error_ms": _f(np.median(d) * 1000, 1) if d else None}
    return out
