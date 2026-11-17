"""파일 스키마 이름·버전과 마이그레이션.

각 JSON 파일은 ``schema`` (종류)와 ``schema_version`` 을 가진다.
구조를 바꿀 때는 버전을 올리고 ``MIGRATIONS[종류][이전버전]`` 에 변환 함수를 등록한다.
불러올 때 ``migrate`` 가 버전을 한 단계씩 올린다. 더 새로운 버전 파일은 열지 않는다.
"""

from __future__ import annotations

from typing import Callable

from amp.core.errors import ProjectFormatError

PROJECT = "sonatlas.project"
PITCH = "sonatlas.pitch"
NOTES = "sonatlas.notes"
TEMPO = "sonatlas.tempo"

CURRENT = {PROJECT: 1, PITCH: 1, NOTES: 1, TEMPO: 1}

# 종류 → {이전 버전: (dict → dict)}. 버전 1이 첫 버전이라 아직 등록된 변환은 없다.
MIGRATIONS: dict[str, dict[int, Callable[[dict], dict]]] = {k: {} for k in CURRENT}

LABELS = {PROJECT: "프로젝트 파일", PITCH: "피치 분석 파일", NOTES: "음표 파일", TEMPO: "템포 파일"}


def stamp(kind: str, body: dict) -> dict:
    return {"schema": kind, "schema_version": CURRENT[kind], **body}


def migrate(kind: str, data: object) -> tuple[dict, bool]:
    """(최신 버전 dict, 마이그레이션 여부)."""
    label = LABELS.get(kind, kind)
    if not isinstance(data, dict) or data.get("schema") != kind:
        raise ProjectFormatError(f"{label} 형식이 올바르지 않습니다.")
    try:
        version = int(data.get("schema_version", 0))
    except (TypeError, ValueError):
        raise ProjectFormatError(f"{label}의 버전 정보가 올바르지 않습니다.") from None
    current = CURRENT[kind]
    if version > current:
        raise ProjectFormatError(f"{label}이(가) 더 새로운 버전(v{version})에서 저장되었습니다. 프로그램을 업데이트해 주세요.")
    migrated = False
    while version < current:
        step = MIGRATIONS[kind].get(version)
        if step is None:
            raise ProjectFormatError(f"{label} v{version}을(를) 변환할 방법이 없습니다.")
        data = step(dict(data))
        version += 1
        data["schema_version"] = version
        migrated = True
    return data, migrated
