"""analysis/ 폴더의 pitch.json · notes.json · tempo.json 읽기/쓰기.

- pitch.json: 분석 결과만 담는다(사용자가 고치지 않는다). 프레임별 시간·주파수·MIDI·신뢰도·유성 여부·온셋.
- notes.json: ``analysis`` (자동 분석 원본, 분석할 때만 바뀜)와 ``edits`` (사용자 작업본)를 나눠 담는다.
- tempo.json: ``analysis`` (추정값·후보·박 위치)와 ``user`` (사용자가 정한 템포)를 나눠 담는다.

모든 쓰기는 원자적 교체이고, notes·tempo는 직전 파일을 .bak으로 남긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from amp.core.fileio import atomic_write_json, read_json_with_backup
from amp.core.model import AnalysisResult, ModelInfo, Note, PitchCurve, Project, Score
from amp.core.tempo import TempoMap
from amp.project import schema
from amp.project.layout import ProjectPaths
from amp.storage import serialization as ser


def _num(x, nd):
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _arr(values, nd):
    return [_num(v, nd) for v in np.asarray(values, dtype=np.float64).tolist()]


def _model_dict(m: ModelInfo) -> dict:
    return {"name": m.name, "version": m.version, "placeholder": m.placeholder, "params": m.params}


def _model_from(d: dict) -> ModelInfo:
    return ModelInfo(str(d.get("name", "")), str(d.get("version", "")), bool(d.get("placeholder", False)),
                     dict(d.get("params") or {}))


# --- pitch.json --------------------------------------------------------------

def pitch_to_dict(a: AnalysisResult) -> dict:
    c = a.pitch_curve
    n = len(c)
    midi = c.midi()
    onset = a.onset_envelope if a.onset_envelope is not None and len(a.onset_envelope) == n else None
    return schema.stamp(schema.PITCH, {
        "created": a.created_at,
        "model": _model_dict(a.models["pitch"]) if "pitch" in a.models else None,
        "settings": a.settings,
        "frame_rate": c.frame_rate,
        "start_time": c.start_time,
        "frames": n,
        "time": _arr(c.times(), 3),
        "f0_hz": _arr(c.f0_hz, 2),
        "midi": _arr(midi, 3),
        "confidence": _arr(c.confidence, 3),
        "voiced": [int(v) for v in np.isfinite(c.f0_hz)],
        "onset": _arr(onset, 3) if onset is not None else None,
    })


def pitch_from_dict(d: dict) -> tuple[PitchCurve, np.ndarray | None]:
    f0 = np.array([np.nan if v is None else v for v in d["f0_hz"]], dtype=np.float32)
    conf = np.array([0.0 if v is None else v for v in d["confidence"]], dtype=np.float32)
    curve = PitchCurve(float(d.get("start_time", 0.0)), float(d["frame_rate"]), f0, conf)
    onset = d.get("onset")
    return curve, (np.array([0.0 if v is None else v for v in onset], dtype=np.float32) if onset else None)


# --- notes.json --------------------------------------------------------------

def _note_out(n: Note, status: str | None = None) -> dict:
    d = ser.note_to_dict(n)
    d["name"] = n.name  # 사람이 읽기 쉽게 (불러올 때는 pitch를 쓴다)
    if status is not None:
        d["status"] = status
    return d


def notes_to_dict(project: Project) -> dict:
    a = project.analysis
    analysis = None
    if a is not None:
        analysis = {
            "created": a.created_at,
            "duration": a.duration,
            "audio_sha256": a.audio_fingerprint,
            "settings": a.settings,
            "models": {k: _model_dict(m) for k, m in a.models.items()},
            "warnings": list(a.warnings),
            "notes": [_note_out(n) for n in a.notes],
            "lyrics": ser.lyrics_to_dict(a.lyrics) if a.lyrics else None,
        }
    edits = {
        "notes": [_note_out(n, project.note_status(n)) for n in project.score.notes()],
        "lyrics": ser.lyrics_to_dict(project.score.lyrics),
    }
    return schema.stamp(schema.NOTES, {"analysis": analysis, "edits": edits})


# --- tempo.json --------------------------------------------------------------

def tempo_to_dict(project: Project) -> dict:
    a = project.analysis
    analysis = None
    if a is not None:
        info = dict(a.tempo_info)
        analysis = {
            "bpm": a.tempo.initial_bpm,
            "first_beat": a.tempo.first_beat_time,
            "time_signature": [a.tempo.beats_per_bar, a.tempo.beat_unit],
            "candidates": info.pop("candidates", []),
            "beats": info.pop("beats", []),
            "reliable": info.pop("reliable", None),
            "algorithm": info.pop("algorithm", a.models["tempo"].name if "tempo" in a.models else ""),
            "model": _model_dict(a.models["tempo"]) if "tempo" in a.models else None,
            "extra": info,
            "tempo_map": ser.tempo_to_dict(a.tempo),
        }
    t = project.score.tempo
    user = {
        "bpm": t.initial_bpm,
        "first_beat": t.first_beat_time,
        "time_signature": [t.beats_per_bar, t.beat_unit],
        "tempo_map": ser.tempo_to_dict(t),
        "edited": a is not None and t != a.tempo,
    }
    return schema.stamp(schema.TEMPO, {"analysis": analysis, "user": user})


# --- 전체 저장·불러오기 ----------------------------------------------------------------

@dataclass
class LoadReport:
    migrated: list[str] = field(default_factory=list)
    from_backup: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def save_analysis_files(paths: ProjectPaths, project: Project, *, write_pitch: bool = True) -> None:
    paths.analysis.mkdir(parents=True, exist_ok=True)
    if write_pitch and project.analysis is not None and project.analysis.pitch_curve is not None:
        atomic_write_json(paths.pitch_json, pitch_to_dict(project.analysis), indent=None)
    atomic_write_json(paths.notes_json, notes_to_dict(project), backup=True)
    atomic_write_json(paths.tempo_json, tempo_to_dict(project), backup=True)


def _load(path, kind, report: LoadReport):
    data, from_bak = read_json_with_backup(path, what=schema.LABELS[kind])
    data, migrated = schema.migrate(kind, data)
    if from_bak:
        report.from_backup.append(path.name)
        report.warnings.append(f"{path.name}이(가) 손상되어 직전 백업(.bak)에서 불러왔습니다.")
    if migrated:
        report.migrated.append(path.name)
    return data


def load_analysis_files(paths: ProjectPaths, report: LoadReport) -> tuple[AnalysisResult | None, Score]:
    if not paths.notes_json.exists():
        return None, Score()
    notes = _load(paths.notes_json, schema.NOTES, report)
    tempo = _load(paths.tempo_json, schema.TEMPO, report) if paths.tempo_json.exists() else {"analysis": None, "user": None}

    edits = notes.get("edits") or {}
    user_tempo = (tempo.get("user") or {}).get("tempo_map")
    score = Score(
        ser.tempo_from_dict(user_tempo) if user_tempo else None,
        [ser.note_from_dict(n) for n in edits.get("notes", [])],
        ser.lyrics_from_dict(edits.get("lyrics") or {}),
    )

    an = notes.get("analysis")
    if not an:
        return None, score
    curve, onset = None, None
    if paths.pitch_json.exists():
        try:
            curve, onset = pitch_from_dict(_load(paths.pitch_json, schema.PITCH, report))
        except Exception as e:  # noqa: BLE001 - 피치가 깨져도 음표 편집본은 살린다
            report.warnings.append(f"pitch.json을 읽지 못해 피치 곡선 없이 엽니다: {e}")
    ta = tempo.get("analysis") or {}
    tempo_map = ser.tempo_from_dict(ta["tempo_map"]) if ta.get("tempo_map") else (score.tempo or TempoMap.constant(120))
    tempo_info = {k: ta[k] for k in ("candidates", "beats", "reliable", "algorithm") if k in ta}
    tempo_info.update(ta.get("extra") or {})
    lyr = an.get("lyrics")
    result = AnalysisResult(
        duration=float(an.get("duration", 0.0)),
        tempo=tempo_map,
        notes=tuple(ser.note_from_dict(n) for n in an.get("notes", [])),
        pitch_curve=curve,
        lyrics=ser.lyrics_from_dict(lyr) if lyr else None,
        models={k: _model_from(m) for k, m in (an.get("models") or {}).items()},
        created_at=str(an.get("created", "")),
        audio_fingerprint=str(an.get("audio_sha256", "")),
        onset_envelope=onset,
        tempo_info=tempo_info,
        warnings=tuple(an.get("warnings", ())),
        settings=dict(an.get("settings") or {}),
    )
    return result, score
