"""데이터 모델 ↔ JSON 호환 dict 변환.

모델 클래스에 저장 형식을 섞지 않으려고 변환 함수를 여기에 모았다.
필드를 추가할 때는 읽기 쪽에서 기본값을 주어 이전 파일도 열리게 한다.
"""

from __future__ import annotations

import math
from typing import Any

from amp.core.errors import ProjectFormatError
from amp.core.model import (
    AnalysisResult,
    AudioRef,
    LyricsTrack,
    ModelInfo,
    Note,
    PitchCurve,
    Score,
    Syllable,
    Vibrato,
)
from amp.core.tempo import TempoMap, TempoPoint


def _req(d: dict, key: str, where: str) -> Any:
    try:
        return d[key]
    except (KeyError, TypeError):
        raise ProjectFormatError(f"{where}에 '{key}' 항목이 없습니다.") from None


# --- 템포 -----------------------------------------------------------------

def tempo_to_dict(t: TempoMap) -> dict:
    return {
        "points": [{"time": p.time, "bpm": p.bpm} for p in t.points],
        "beats_per_bar": t.beats_per_bar,
        "beat_unit": t.beat_unit,
    }


def tempo_from_dict(d: dict) -> TempoMap:
    pts = tuple(TempoPoint(float(_req(p, "time", "템포")), float(_req(p, "bpm", "템포"))) for p in _req(d, "points", "템포"))
    return TempoMap(pts, int(d.get("beats_per_bar", 4)), int(d.get("beat_unit", 4)))


# --- 음표 -----------------------------------------------------------------

def note_to_dict(n: Note) -> dict:
    d: dict[str, Any] = {"id": n.id, "start": n.start, "end": n.end, "pitch": n.pitch, "velocity": n.velocity}
    if n.confidence is not None and math.isfinite(n.confidence):  # NaN이면 JSON 저장 자체가 실패한다
        d["confidence"] = n.confidence
    if n.vibrato is not None and math.isfinite(n.vibrato.rate_hz) and math.isfinite(n.vibrato.extent_cents):
        d["vibrato"] = {"rate_hz": n.vibrato.rate_hz, "extent_cents": n.vibrato.extent_cents}
    if n.syllable_id is not None:
        d["syllable_id"] = n.syllable_id
    if n.origin_ids:
        d["origin_ids"] = list(n.origin_ids)
    if n.ornament:
        d["ornament"] = True
    return d


def note_from_dict(d: dict) -> Note:
    vib = d.get("vibrato")
    return Note(
        id=str(_req(d, "id", "음표")),
        start=float(_req(d, "start", "음표")),
        end=float(_req(d, "end", "음표")),
        pitch=int(_req(d, "pitch", "음표")),
        velocity=int(d.get("velocity", 100)),
        confidence=None if d.get("confidence") is None else float(d["confidence"]),
        vibrato=Vibrato(float(vib["rate_hz"]), float(vib["extent_cents"])) if vib else None,
        syllable_id=d.get("syllable_id"),
        origin_ids=tuple(d.get("origin_ids", ())),
        ornament=bool(d.get("ornament", False)),
    )


# --- 가사 -----------------------------------------------------------------

def lyrics_to_dict(t: LyricsTrack) -> dict:
    return {
        "language": t.language,
        "syllables": [
            {"id": s.id, "text": s.text, "phonetic": s.phonetic, "line": s.line, "start": s.start, "end": s.end}
            for s in t.syllables
        ],
    }


def lyrics_from_dict(d: dict) -> LyricsTrack:
    syls = tuple(
        Syllable(
            id=str(_req(s, "id", "음절")),
            text=str(_req(s, "text", "음절")),
            phonetic=s.get("phonetic"),
            line=int(s.get("line", 0)),
            start=s.get("start"),
            end=s.get("end"),
        )
        for s in d.get("syllables", ())
    )
    return LyricsTrack(str(d.get("language", "ko")), syls)


# --- 작업본 / 원본 ---------------------------------------------------------

def score_to_dict(s: Score) -> dict:
    return {
        "tempo": tempo_to_dict(s.tempo),
        "notes": [note_to_dict(n) for n in s.notes()],
        "lyrics": lyrics_to_dict(s.lyrics),
    }


def score_from_dict(d: dict) -> Score:
    return Score(
        tempo_from_dict(_req(d, "tempo", "작업본")),
        [note_from_dict(n) for n in d.get("notes", ())],
        lyrics_from_dict(d.get("lyrics", {})),
    )


def analysis_to_dict(a: AnalysisResult) -> dict:
    """피치 곡선 배열은 별도 파일(npz)로 저장하므로 여기에 넣지 않는다."""
    return {
        "duration": a.duration,
        "tempo": tempo_to_dict(a.tempo),
        "notes": [note_to_dict(n) for n in a.notes],
        "lyrics": lyrics_to_dict(a.lyrics) if a.lyrics else None,
        "models": {
            k: {"name": m.name, "version": m.version, "placeholder": m.placeholder, "params": m.params}
            for k, m in a.models.items()
        },
        "created_at": a.created_at,
        "audio_fingerprint": a.audio_fingerprint,
        "has_pitch_curve": a.pitch_curve is not None,
        "tempo_info": a.tempo_info,
        "warnings": list(a.warnings),
        "settings": a.settings,
    }


def analysis_from_dict(d: dict, pitch_curve: PitchCurve | None, onset=None) -> AnalysisResult:
    models = {
        k: ModelInfo(str(m.get("name", "")), str(m.get("version", "")), bool(m.get("placeholder", False)), dict(m.get("params", {})))
        for k, m in d.get("models", {}).items()
    }
    lyr = d.get("lyrics")
    return AnalysisResult(
        duration=float(_req(d, "duration", "분석 결과")),
        tempo=tempo_from_dict(_req(d, "tempo", "분석 결과")),
        notes=tuple(note_from_dict(n) for n in d.get("notes", ())),
        pitch_curve=pitch_curve,
        lyrics=lyrics_from_dict(lyr) if lyr else None,
        models=models,
        created_at=str(d.get("created_at", "")),
        audio_fingerprint=str(d.get("audio_fingerprint", "")),
        onset_envelope=onset,
        tempo_info=dict(d.get("tempo_info") or {}),
        warnings=tuple(str(w) for w in d.get("warnings", ())),
        settings=dict(d.get("settings") or {}),
    )


def audio_ref_to_dict(a: AudioRef) -> dict:
    return {
        "path": a.path,
        "fingerprint": a.fingerprint,
        "duration": a.duration,
        "sample_rate": a.sample_rate,
        "channels": a.channels,
    }


def audio_ref_from_dict(d: dict) -> AudioRef:
    return AudioRef(
        path=str(_req(d, "path", "오디오 정보")),
        fingerprint=str(d.get("fingerprint", "")),
        duration=float(d.get("duration", 0.0)),
        sample_rate=int(d.get("sample_rate", 0)),
        channels=int(d.get("channels", 0)),
    )
