"""analysis/ 폴더의 pitch.json · notes.json · tempo.json 읽기/쓰기.

- pitch.json: 분석 결과만 담는다(사용자가 고치지 않는다). 프레임별 시간·주파수·MIDI·신뢰도·유성 여부·온셋.
- notes.json: ``analysis`` (자동 분석 원본, 분석할 때만 바뀜)와 ``edits`` (사용자 작업본)를 나눠 담는다.
- tempo.json: ``analysis`` (추정값·후보·박 위치)와 ``user`` (사용자가 정한 템포)를 나눠 담는다.
- lyrics.json · alignment.json: 가사 텍스트와 음절 시간·음표 연결 (``amp.lyrics.io``). notes.json v2부터 음표는 가사를 모른다.

모든 쓰기는 원자적 교체이고, notes·tempo는 직전 파일을 .bak으로 남긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from amp.core.fileio import atomic_write_json, read_json_with_backup
from amp.core.model import AnalysisResult, ModelInfo, Note, PitchCurve, Project, Score
from amp.core.tempo import TempoMap
from amp.lyrics import io as lyrics_io
from amp.lyrics import linking, pronunciation
from amp.lyrics import ops as lyrics_ops
from amp.lyrics.model import LyricsState
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
        }
    edits = {
        "notes": [_note_out(n, project.note_status(n)) for n in project.score.notes()],
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
    warnings: list[str] = field(default_factory=list)  # 사용자가 꼭 알아야 할 문제 (손상 복구 등)
    notes: list[str] = field(default_factory=list)  # 참고 안내 (폴더 이동·복사 감지 등)
    legacy_lyrics: dict | None = None  # notes.json v1에서 꺼낸 STEP 1 가사 (새 형식으로 옮길 것)


def save_analysis_files(paths: ProjectPaths, project: Project, *, write_pitch: bool = True) -> None:
    paths.analysis.mkdir(parents=True, exist_ok=True)
    if write_pitch and project.analysis is not None and project.analysis.pitch_curve is not None:
        atomic_write_json(paths.pitch_json, pitch_to_dict(project.analysis), indent=None)
    atomic_write_json(paths.notes_json, notes_to_dict(project), backup=True)
    atomic_write_json(paths.tempo_json, tempo_to_dict(project), backup=True)
    save_lyrics_files(paths, project.lyrics)
    save_step5_files(paths, project)


def save_step5_files(paths: ProjectPaths, project: Project) -> None:
    """검토 상태·참조 악보. 내용이 없고 파일도 없으면 만들지 않는다(이전 프로젝트 구조 그대로)."""
    from amp.compare import reference as ref_io
    from amp.compare import review as review_io

    if project.review.entries or paths.review_json.exists():
        atomic_write_json(paths.review_json, schema.stamp(schema.REVIEW, review_io.state_to_dict(project.review)),
                          backup=True)
    if project.reference is not None:
        atomic_write_json(paths.reference_json, schema.stamp(schema.REFERENCE, ref_io.to_dict(project.reference)),
                          backup=True)
    elif paths.reference_json.exists():  # 참조 악보를 지웠다: 파일은 백업으로 남기고 지운다
        backup = paths.reference_json.with_name(paths.reference_json.name + ".removed.bak")
        paths.reference_json.replace(backup)


def load_step5_files(paths: ProjectPaths, report: LoadReport):
    from amp.compare import reference as ref_io
    from amp.compare import review as review_io

    review = review_io.ReviewState()
    reference = None
    if paths.review_json.exists():
        try:
            review = review_io.state_from_dict(_load(paths.review_json, schema.REVIEW, report))
        except Exception as e:  # noqa: BLE001 - 검토 상태가 깨져도 프로젝트는 연다
            report.warnings.append(f"검토 상태 파일을 읽지 못해 빈 상태로 시작합니다: {e}")
    if paths.reference_json.exists():
        try:
            reference = ref_io.from_dict(_load(paths.reference_json, schema.REFERENCE, report))
        except Exception as e:  # noqa: BLE001
            report.warnings.append(f"참조 악보 파일을 읽지 못했습니다: {e}")
    return review, reference


def save_lyrics_files(paths: ProjectPaths, state: LyricsState) -> None:
    """가사가 없고 파일도 없으면 만들지 않는다. 가사를 모두 지웠으면 빈 파일로 남긴다."""
    if state.is_empty and not paths.lyrics_json.exists() and not paths.alignment_json.exists():
        return
    paths.analysis.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.lyrics_json, schema.stamp(schema.LYRICS, lyrics_io.lyrics_to_dict(state)), backup=True)
    atomic_write_json(paths.alignment_json, schema.stamp(schema.ALIGNMENT, lyrics_io.alignment_to_dict(state)),
                      backup=True)


def load_lyrics_files(paths: ProjectPaths, report: LoadReport, note_ids, duration: float | None) -> LyricsState:
    """lyrics.json·alignment.json을 읽는다. 없으면 notes.json v1의 STEP 1 가사를 옮긴다.

    가사 파일이 깨져도 음표 편집본은 열려야 하므로 여기서 난 오류는 경고로 바꾼다.
    """
    warnings: list[str] = []
    state = LyricsState()
    try:
        if paths.lyrics_json.exists():
            lyr = _load(paths.lyrics_json, schema.LYRICS, report)
            ali = _load(paths.alignment_json, schema.ALIGNMENT, report) if paths.alignment_json.exists() else None
            state = lyrics_io.state_from_dicts(lyr, ali, warnings)
        elif report.legacy_lyrics:
            legacy = report.legacy_lyrics
            state = lyrics_io.from_legacy(legacy.get("lyrics"), dict(legacy.get("note_links") or {}))
            if not state.is_empty:
                report.notes.append(f"이전 형식의 가사 음절 {len(state.text.syllables)}개를 lyrics.json·alignment.json으로 옮겼습니다.")
    except Exception as e:  # noqa: BLE001
        report.warnings.append(f"가사 파일을 읽지 못해 가사 없이 엽니다 (파일은 그대로 둡니다): {e}")
        return LyricsState()
    state, w2 = lyrics_ops.sanitize_alignment(state, duration)
    state, dropped = linking.prune(state, note_ids)
    if dropped:
        w2.append(f"없는 음표를 가리키던 가사 연결 {dropped}개를 지웠습니다.")
    try:  # 발음이 없거나(이전 형식) 변환 규칙 버전이 바뀐 자동 발음은 메모리에서 다시 만든다. 사용자 발음은 그대로
        state = pronunciation.sync(state)
    except Exception as e:  # noqa: BLE001 - 발음 변환이 실패해도 가사는 연다
        w2.append(f"발음을 만들지 못했습니다(가사는 그대로 엽니다): {e}")
    report.warnings.extend(warnings + w2)
    return state


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
    report.legacy_lyrics = notes.get("legacy_lyrics")
    user_tempo = (tempo.get("user") or {}).get("tempo_map")
    score = Score(
        ser.tempo_from_dict(user_tempo) if user_tempo else None,
        [ser.note_from_dict(n) for n in edits.get("notes", [])],
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
    result = AnalysisResult(
        duration=float(an.get("duration", 0.0)),
        tempo=tempo_map,
        notes=tuple(ser.note_from_dict(n) for n in an.get("notes", [])),
        pitch_curve=curve,
        models={k: _model_from(m) for k, m in (an.get("models") or {}).items()},
        created_at=str(an.get("created", "")),
        audio_fingerprint=str(an.get("audio_sha256", "")),
        onset_envelope=onset,
        tempo_info=tempo_info,
        warnings=tuple(an.get("warnings", ())),
        settings=dict(an.get("settings") or {}),
    )
    return result, score
