"""프로젝트 파일(.ampproj) 읽기/쓰기.

.ampproj 는 ZIP 파일이며 다음 항목을 담는다.

    manifest.json          형식 이름과 버전
    project.json           오디오 파일 참조
    analysis/result.json   원본 분석 결과 (분석을 실행한 경우에만)
    analysis/pitch.npz     원본 피치 곡선 배열
    score.json             사용자 작업본

원본과 작업본을 다른 파일에 두어, 작업본을 아무리 고쳐도 원본은 그대로 남는다.
쓰기는 임시 파일에 먼저 기록한 뒤 교체하므로 저장 도중 중단돼도 기존 파일이 깨지지 않는다.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone

import numpy as np

from amp import __version__
from amp.core.errors import EditError, ProjectFormatError
from amp.core.model import PitchCurve, Project
from amp.lyrics import io as lyrics_io
from amp.storage import serialization as ser

PROJECT_EXTENSION = ".ampproj"
FORMAT_NAME = "amp-project"
FORMAT_VERSION = 1


def _dump(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")


def _pitch_to_bytes(curve: PitchCurve, onset=None) -> bytes:
    buf = io.BytesIO()
    arrays = dict(
        start_time=np.float64(curve.start_time),
        frame_rate=np.float64(curve.frame_rate),
        f0_hz=curve.f0_hz,
        confidence=curve.confidence,
    )
    if onset is not None:
        arrays["onset"] = np.asarray(onset, dtype=np.float32)
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _pitch_from_bytes(data: bytes):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        curve = PitchCurve(float(z["start_time"]), float(z["frame_rate"]), z["f0_hz"], z["confidence"])
        onset = z["onset"].copy() if "onset" in z.files else None
    return curve, onset


def project_to_members(project: Project, project_path: str | None = None) -> dict[str, bytes]:
    """ZIP에 넣을 (경로 → 바이트) 목록. 자동 저장도 이 함수를 쓴다."""
    audio = None
    if project.audio is not None:
        audio = ser.audio_ref_to_dict(project.audio)
        if project_path:
            try:
                audio["relative_path"] = os.path.relpath(project.audio.path, os.path.dirname(os.path.abspath(project_path)))
            except ValueError:  # Windows에서 드라이브가 다르면 상대 경로를 만들 수 없다
                pass
    members = {
        "manifest.json": _dump(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "app_version": __version__,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ),
        "project.json": _dump({"audio": audio}),
        "score.json": _dump(ser.score_to_dict(project.score)),
    }
    if not project.lyrics.is_empty:  # 자동 저장·복구에도 가사 편집을 담는다
        members["lyrics.json"] = _dump(lyrics_io.lyrics_to_dict(project.lyrics))
        members["alignment.json"] = _dump(lyrics_io.alignment_to_dict(project.lyrics))
    if project.analysis is not None:
        members["analysis/result.json"] = _dump(ser.analysis_to_dict(project.analysis))
        if project.analysis.pitch_curve is not None:
            members["analysis/pitch.npz"] = _pitch_to_bytes(project.analysis.pitch_curve, project.analysis.onset_envelope)
    if project.bass is not None:  # v0.6.3: 자동 저장·복구에도 베이스 편집을 담는다
        from amp.project import bass_io

        members["analysis/bass.json"] = _dump(bass_io.to_dict(project.bass))
    return members


def write_members(path: str, members: dict[str, bytes]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".ampsave-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in members.items():
                z.writestr(name, data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def save_project(project: Project, path: str) -> None:
    write_members(path, project_to_members(project, path))


def load_project(path: str) -> Project:
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = set(z.namelist())

            def read_json(name: str):
                try:
                    return json.loads(z.read(name).decode("utf-8"))
                except KeyError:
                    raise ProjectFormatError(f"프로젝트 파일에 {name} 이(가) 없습니다.") from None
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise ProjectFormatError(f"{name} 을(를) 읽을 수 없습니다: {e}") from e

            manifest = read_json("manifest.json")
            if manifest.get("format") != FORMAT_NAME:
                raise ProjectFormatError("Amp 프로젝트 파일이 아닙니다.")
            version = int(manifest.get("format_version", 0))
            if version > FORMAT_VERSION:
                raise ProjectFormatError(
                    f"더 새로운 버전(형식 {version})에서 저장한 파일입니다. 프로그램을 업데이트해 주세요."
                )
            meta = read_json("project.json")
            score_d = read_json("score.json")
            score = ser.score_from_dict(score_d)
            warnings: list[str] = []
            if "lyrics.json" in names:
                lyrics = lyrics_io.state_from_dicts(read_json("lyrics.json"),
                                                    read_json("alignment.json") if "alignment.json" in names else None,
                                                    warnings)
            else:  # STEP 1 형식: score.json 안의 lyrics + 음표 syllable_id
                lyrics = lyrics_io.from_legacy(score_d.get("lyrics"), ser.legacy_note_links(score_d.get("notes")))
            analysis = None
            if "analysis/result.json" in names:
                curve, onset = _pitch_from_bytes(z.read("analysis/pitch.npz")) if "analysis/pitch.npz" in names else (None, None)
                analysis = ser.analysis_from_dict(read_json("analysis/result.json"), curve, onset)
            bass = None
            if "analysis/bass.json" in names:
                from amp.project import bass_io, schema

                try:  # 베이스가 깨져도 보컬 음표·가사는 복구한다
                    bass = bass_io.from_dict(schema.migrate(schema.BASS, read_json("analysis/bass.json"))[0])
                except Exception:  # noqa: BLE001
                    bass = None
    except FileNotFoundError:
        raise ProjectFormatError(f"파일을 찾을 수 없습니다: {path}") from None
    except zipfile.BadZipFile:
        raise ProjectFormatError("프로젝트 파일이 손상되었거나 형식이 올바르지 않습니다.") from None
    except ProjectFormatError:
        raise
    except (ValueError, TypeError, KeyError, EditError) as e:
        raise ProjectFormatError(f"프로젝트 내용이 올바르지 않습니다: {e}") from e

    audio = None
    audio_d = meta.get("audio")
    if audio_d:
        audio = ser.audio_ref_from_dict(audio_d)
        rel = audio_d.get("relative_path")
        if not os.path.exists(audio.path) and rel:
            candidate = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), rel))
            if os.path.exists(candidate):
                audio = dataclasses.replace(audio, path=candidate)
    return Project(audio=audio, analysis=analysis, score=score, lyrics=lyrics, bass=bass)
