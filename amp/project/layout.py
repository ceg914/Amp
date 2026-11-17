"""프로젝트 폴더 구조와 이름 규칙.

    SONATLAS Projects/
    └── <프로젝트 이름>/
        ├── original.<ext>          원본 사본 (복사 방식일 때만)
        ├── stems/                  분리 결과 (정식 결과, 원본과 같은 시간축)
        │   ├── vocals.wav
        │   ├── instrumental.wav
        │   └── stems.json          분리 결과 목록·검증 정보
        ├── analysis/               분석 결과 (정식 결과 + 사용자 편집)
        │   ├── pitch.json
        │   ├── notes.json
        │   └── tempo.json
        ├── cache/                  다시 만들 수 있는 중간 결과 (지워도 안전)
        ├── exports/                사용자가 내보낸 파일
        ├── backup/                 마이그레이션 전 백업
        └── project.json            프로젝트 상태
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

PROJECT_FILE = "project.json"
STEMS_DIR = "stems"
ANALYSIS_DIR = "analysis"
CACHE_DIR = "cache"
EXPORTS_DIR = "exports"
BACKUP_DIR = "backup"
LOCK_FILE = ".sonatlas.lock"
DEFAULT_ROOT_NAME = "SONATLAS Projects"
MAX_NAME_LEN = 80  # Windows 경로 길이 제한(260자)을 고려한 폴더 이름 상한

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class ProjectPaths:
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    @property
    def project_json(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def stems(self) -> Path:
        return self.root / STEMS_DIR

    @property
    def analysis(self) -> Path:
        return self.root / ANALYSIS_DIR

    @property
    def pitch_json(self) -> Path:
        return self.analysis / "pitch.json"

    @property
    def notes_json(self) -> Path:
        return self.analysis / "notes.json"

    @property
    def tempo_json(self) -> Path:
        return self.analysis / "tempo.json"

    @property
    def cache(self) -> Path:
        return self.root / CACHE_DIR

    @property
    def exports(self) -> Path:
        return self.root / EXPORTS_DIR

    @property
    def backup(self) -> Path:
        return self.root / BACKUP_DIR

    @property
    def lock(self) -> Path:
        return self.root / LOCK_FILE

    def ensure_dirs(self) -> None:
        for d in (self.stems, self.analysis, self.cache, self.exports):
            d.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    """Windows에서 쓸 수 있는 폴더 이름으로 바꾼다. 한글·공백·대부분의 특수문자는 그대로 둔다."""
    name = unicodedata.normalize("NFC", name).strip()
    name = _INVALID.sub("_", name)
    name = name.rstrip(" .")
    if not name:
        name = "Untitled"
    if name.split(".")[0].upper() in _RESERVED:
        name = f"_{name}"
    return name[:MAX_NAME_LEN].rstrip(" .") or "Untitled"


def unique_folder(parent: str | os.PathLike, name: str) -> Path:
    """parent 안에서 겹치지 않는 폴더 경로. 기존 폴더는 절대 재사용하지 않는다."""
    parent = Path(parent)
    base = sanitize_name(name)
    candidate = parent / base
    n = 2
    while candidate.exists():
        candidate = parent / f"{base} ({n})"
        n += 1
    return candidate


def is_project_folder(path: str | os.PathLike) -> bool:
    return (Path(path) / PROJECT_FILE).is_file()
