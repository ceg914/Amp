"""analysis/bass.json 읽기·쓰기 (v0.6.3 베이스 TAB).

    {"options": {...}, "source": {...},
     "analysis": {"created", "duration", "audio_sha256", "models", "warnings", "notes": [...],
                  "pitch": {"frame_rate", "start_time", "f0_hz": [...], "confidence": [...]}},
     "edits": {"notes": [... "status": original|modified|user]}}

notes.json처럼 자동 분석 원본(analysis)과 사용자 작업본(edits)을 나눈다. 박 위치는 보컬 프로젝트의 템포(tempo.json)를
함께 쓰므로 여기 두지 않는다. 자동 저장(복구 파일)도 같은 내용을 담는다.
"""

from __future__ import annotations

import numpy as np

from amp.core.model import AnalysisResult, PitchCurve, Project, Score
from amp.core.tempo import TempoMap
from amp.engine.bass import BassOptions, BassState
from amp.project import schema
from amp.storage import serialization as ser


def _num(x, nd):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _notes_project(state: BassState) -> Project:
    return Project(analysis=state.analysis, score=Score(None, list(state.notes)))


def to_dict(state: BassState) -> dict:
    from amp.project.analysis_io import _model_dict, _note_out

    a = state.analysis
    analysis = None
    if a is not None:
        c = a.pitch_curve
        pitch = None
        if c is not None:
            pitch = {"frame_rate": c.frame_rate, "start_time": c.start_time,
                     "f0_hz": [_num(v, 2) for v in np.asarray(c.f0_hz, dtype=np.float64)],
                     "confidence": [_num(v, 3) for v in np.asarray(c.confidence, dtype=np.float64)]}
        analysis = {"created": a.created_at, "duration": a.duration, "audio_sha256": a.audio_fingerprint,
                    "models": {k: _model_dict(m) for k, m in a.models.items()}, "warnings": list(a.warnings),
                    "settings": a.settings, "notes": [_note_out(n) for n in a.notes], "pitch": pitch}
    proj = _notes_project(state)
    edits = {"notes": [_note_out(n, proj.note_status(n)) for n in sorted(state.notes, key=lambda n: n.start)]}
    body = {"options": state.options.to_dict(), "source": dict(state.source), "analysis": analysis, "edits": edits}
    if state.fingering:  # v0.7.0 사용자가 정한 줄 (없으면 적지 않아 이전 파일과 같다)
        body["fingering"] = {str(k): int(v) for k, v in state.fingering.items()}
    return schema.stamp(schema.BASS, body)


def from_dict(d: dict) -> BassState:
    from amp.project.analysis_io import _model_from

    an = d.get("analysis")
    analysis = None
    if an:
        curve = None
        p = an.get("pitch")
        if p:
            f0 = np.array([np.nan if v is None else v for v in p["f0_hz"]], dtype=np.float32)
            conf = np.array([0.0 if v is None else v for v in p["confidence"]], dtype=np.float32)
            if len(f0) != len(conf):
                raise ValueError("베이스 피치 배열 길이가 서로 다릅니다.")
            curve = PitchCurve(float(p.get("start_time", 0.0)), float(p["frame_rate"]), f0, conf)
        analysis = AnalysisResult(
            duration=float(an.get("duration", 0.0)), tempo=TempoMap.constant(120.0),
            notes=tuple(ser.note_from_dict(n) for n in an.get("notes", [])), pitch_curve=curve,
            models={k: _model_from(m) for k, m in (an.get("models") or {}).items()},
            created_at=str(an.get("created", "")), audio_fingerprint=str(an.get("audio_sha256", "")),
            warnings=tuple(an.get("warnings", ())), settings=dict(an.get("settings") or {}))
    notes = tuple(ser.note_from_dict(n) for n in (d.get("edits") or {}).get("notes", []))
    fing = {}
    for k, v in (d.get("fingering") or {}).items():
        try:
            fing[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return BassState(BassOptions.from_dict(d.get("options")), analysis, notes, dict(d.get("source") or {}), fing)
