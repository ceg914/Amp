"""분리 결과(stems/) 관리.

    stems/
        vocals.wav         보컬 (원본과 같은 샘플레이트·프레임 수, PCM 24bit)
        instrumental.wav   반주
        stems.json         목록(manifest). 마지막에 쓰는 '완료 표시'다.

stems.json이 없거나, 원본 해시·파일 크기·프레임 수가 맞지 않으면 분리 결과를 쓰지 않는다.
다시 분리할 때는 기존 stems.json을 먼저 지워(무효화) 파일 교체 중 죽어도 반쯤 바뀐 결과를
정상으로 보지 않게 하고, 새 파일을 옮긴 뒤 새 stems.json을 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from amp.audio.data import AudioData
from amp.core.errors import ProjectFormatError
from amp.core.fileio import atomic_write_json, read_json

STEM_NAMES = ("vocals", "instrumental")
OPTIONAL_STEMS = ("bass",)  # v0.6.3: htdemucs 베이스 (베이스 TAB용). 없어도 분리 결과는 유효하다
MANIFEST = "stems.json"
SCHEMA = "sonatlas.stems"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StemManifest:
    model: str
    source_sha256: str
    sample_rate: int
    frames: int
    channels: int
    created: str
    files: dict  # 이름 → {"path", "size", "frames"}
    model_info: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    offset_frames: int = 0  # 원본 대비 시간 오프셋(프레임). 길이·위치를 맞춰 저장하므로 항상 0
    alignment: str = "원본과 같은 디코더로 읽고, 지연 없는 리샘플(resample_poly) 후 원본 프레임 수에 맞춤"

    @property
    def stems_id(self) -> str:
        text = f"{self.model}|{self.created}|{self.source_sha256}"
        if self.model_info.get("vocals_sha256"):  # 가져온 결과: 같은 초에 다시 가져와도 파일이 다르면 다른 id
            text += f"|{self.model_info['vocals_sha256']}|{self.model_info.get('instrumental_sha256', '')}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(schema=SCHEMA, schema_version=SCHEMA_VERSION)
        return d


def read_manifest(stems_dir: str | os.PathLike) -> StemManifest | None:
    path = Path(stems_dir) / MANIFEST
    try:
        d = read_json(path, what="분리 결과 목록")
    except (FileNotFoundError, ProjectFormatError):
        return None
    if not isinstance(d, dict) or d.get("schema") != SCHEMA or int(d.get("schema_version", 0)) > SCHEMA_VERSION:
        return None
    try:
        return StemManifest(
            model=str(d["model"]), source_sha256=str(d["source_sha256"]), sample_rate=int(d["sample_rate"]),
            frames=int(d["frames"]), channels=int(d["channels"]), created=str(d["created"]), files=dict(d["files"]),
            model_info=dict(d.get("model_info") or {}), settings=dict(d.get("settings") or {}),
            offset_frames=int(d.get("offset_frames", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def validate(stems_dir: str | os.PathLike, manifest: StemManifest | None, source_sha256: str) -> tuple[bool, str]:
    """(사용 가능 여부, 이유)."""
    if manifest is None:
        return False, "분리 결과가 없습니다."
    if source_sha256 and manifest.source_sha256 != source_sha256:
        return False, "원본 파일이 분리 이후 바뀌었습니다."
    import soundfile as sf

    for name in STEM_NAMES:
        entry = manifest.files.get(name)
        if not entry:
            return False, f"{name} 항목이 목록에 없습니다."
        path = Path(stems_dir) / entry["path"]
        if not path.is_file():
            return False, f"{entry['path']} 파일이 없습니다."
        if path.stat().st_size != int(entry.get("size", -1)):
            return False, f"{entry['path']} 파일 크기가 목록과 다릅니다."
        try:
            info = sf.info(str(path))
        except Exception:  # noqa: BLE001
            return False, f"{entry['path']} 파일을 읽을 수 없습니다."
        if info.frames != manifest.frames or info.samplerate != manifest.sample_rate:
            return False, f"{entry['path']} 길이·샘플레이트가 원본과 다릅니다."
    return True, ""


def commit(tmp_dir: str | os.PathLike, stems_dir: str | os.PathLike, source_sha256: str, settings: dict) -> StemManifest:
    """작업 프로세스가 tmp_dir에 쓴 결과를 stems/로 옮기고 목록을 쓴다."""
    import soundfile as sf

    tmp_dir, stems_dir = Path(tmp_dir), Path(stems_dir)
    info = json.loads((tmp_dir / "info.json").read_text(encoding="utf-8"))
    stems_dir.mkdir(parents=True, exist_ok=True)
    old = stems_dir / MANIFEST
    if old.exists():
        old.unlink()  # 교체 중 중단되면 '결과 없음'으로 보이게 먼저 무효화한다
    files = {}
    channels = 0
    for name in STEM_NAMES:
        src = tmp_dir / f"{name}.wav"
        dst = stems_dir / f"{name}.wav"
        os.replace(src, dst)
        sfi = sf.info(str(dst))
        channels = sfi.channels
        files[name] = {"path": dst.name, "size": dst.stat().st_size, "frames": sfi.frames}
    for name in OPTIONAL_STEMS:
        src = tmp_dir / f"{name}.wav"
        dst = stems_dir / f"{name}.wav"
        if src.exists():
            os.replace(src, dst)
            sfi = sf.info(str(dst))
            files[name] = {"path": dst.name, "size": dst.stat().st_size, "frames": sfi.frames}
        elif dst.exists():  # 다시 분리했는데 이번에는 없음: 이전 파일을 남기지 않는다
            dst.unlink()
    manifest = StemManifest(
        model=str(info.get("model", "")), source_sha256=source_sha256, sample_rate=int(info["sample_rate"]),
        frames=int(info["frames"]), channels=channels,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"), files=files,
        model_info={k: v for k, v in info.items() if k not in ("frames", "sample_rate")}, settings=settings,
    )
    atomic_write_json(stems_dir / MANIFEST, manifest.to_dict())
    return manifest


def optional_path(stems_dir: str | os.PathLike, manifest: StemManifest | None, name: str) -> Path | None:
    """선택 stem(베이스 등)의 파일 경로. 목록에 없거나 파일이 없거나 크기·길이가 맞지 않으면 None."""
    if manifest is None or name not in manifest.files:
        return None
    entry = manifest.files[name]
    path = Path(stems_dir) / entry["path"]
    if not path.is_file() or path.stat().st_size != int(entry.get("size", -1)):
        return None
    if int(entry.get("frames", manifest.frames)) != manifest.frames:
        return None
    return path


def load(stems_dir: str | os.PathLike, manifest: StemManifest) -> dict[str, AudioData]:
    import soundfile as sf

    out = {}
    for name in STEM_NAMES:
        data, sr = sf.read(str(Path(stems_dir) / manifest.files[name]["path"]), dtype="float32", always_2d=True)
        out[name] = AudioData(np.ascontiguousarray(data), sr)
    return out


def remove(stems_dir: str | os.PathLike) -> None:
    """분리 결과를 지운다 (사용자가 요청했을 때만 호출)."""
    d = Path(stems_dir)
    for name in (MANIFEST, *(f"{n}.wav" for n in STEM_NAMES + OPTIONAL_STEMS)):
        p = d / name
        if p.exists():
            p.unlink()
    for tmp in d.glob(".tmp-*"):
        shutil.rmtree(tmp, ignore_errors=True)
