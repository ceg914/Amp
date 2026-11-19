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
LYRICS = "sonatlas.lyrics"
ALIGNMENT = "sonatlas.alignment"

CURRENT = {PROJECT: 1, PITCH: 1, NOTES: 2, TEMPO: 1, LYRICS: 1, ALIGNMENT: 2}

LABELS = {PROJECT: "프로젝트 파일", PITCH: "피치 분석 파일", NOTES: "음표 파일", TEMPO: "템포 파일",
          LYRICS: "가사 파일", ALIGNMENT: "가사 정렬 파일"}


def _notes_v1_to_v2(d: dict) -> dict:
    """STEP 3: 가사를 notes.json 밖(lyrics.json·alignment.json)으로 옮긴다.

    v1의 edits.lyrics와 음표 syllable_id는 ``legacy_lyrics`` 로 모아 두고, 불러오는 쪽이 새 형식으로 바꾼다.
    """
    edits = dict(d.get("edits") or {})
    legacy: dict = {}
    lyr = edits.pop("lyrics", None)
    if isinstance(lyr, dict) and lyr.get("syllables"):
        legacy["lyrics"] = lyr
    links = {}
    notes = []
    for n in edits.get("notes") or ():
        n = dict(n)
        sid = n.pop("syllable_id", None)
        if sid:
            links[str(n.get("id"))] = str(sid)
        notes.append(n)
    edits["notes"] = notes
    if links:
        legacy["note_links"] = links
    d["edits"] = edits
    an = d.get("analysis")
    if isinstance(an, dict):
        an = dict(an)
        an.pop("lyrics", None)
        an["notes"] = [{k: v for k, v in n.items() if k != "syllable_id"} for n in an.get("notes") or ()]
        d["analysis"] = an
    if legacy:
        d["legacy_lyrics"] = legacy
    return d


def _alignment_v1_to_v2(d: dict) -> dict:
    """STEP 3.5: 연결에 역할(role: 접근음)이 생겼다. v1 연결은 모두 보통 연결이라 값은 바꾸지 않는다."""
    return d


# 종류 → {이전 버전: (dict → dict)}
MIGRATIONS: dict[str, dict[int, Callable[[dict], dict]]] = {k: {} for k in CURRENT}
MIGRATIONS[NOTES][1] = _notes_v1_to_v2
MIGRATIONS[ALIGNMENT][1] = _alignment_v1_to_v2


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
