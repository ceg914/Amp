"""안전한 파일 쓰기·읽기 (Qt 비의존).

- 원자적 교체: 같은 폴더의 임시 파일에 쓰고 fsync한 뒤 ``os.replace`` 로 바꾼다.
  쓰는 도중 프로그램이 죽어도 기존 파일은 그대로 남는다.
- 백업: ``backup=True`` 면 교체 직전 기존 파일을 ``<이름>.bak`` 으로 남긴다(한 세대).
- Windows에서 백신·검색 색인이 파일을 잠깐 잡고 있으면 교체가 실패할 수 있어 몇 번 다시 시도한다.
- 디스크 부족·권한 오류는 한국어 메시지를 담은 StorageError로 바꾼다.
- Windows 긴 경로(260자 이상)는 ``\\\\?\\`` 접두사로 연다.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from amp.core.errors import ProjectFormatError, StorageError

_REPLACE_RETRIES = 5


def long_path(path: str | os.PathLike) -> str:
    """Windows에서 긴 절대 경로를 \\\\?\\ 형식으로 바꾼다. 다른 OS에서는 그대로."""
    p = os.fspath(path)
    if os.name != "nt" or p.startswith("\\\\?\\") or len(p) < 240:
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def storage_error(e: OSError, path: str | os.PathLike) -> StorageError:
    name = os.fspath(path)
    if e.errno == errno.ENOSPC or getattr(e, "winerror", None) in (39, 112):
        return StorageError(f"디스크 공간이 부족해 저장하지 못했습니다: {name}")
    if isinstance(e, PermissionError) or e.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return StorageError(f"쓰기 권한이 없거나 다른 프로그램이 파일을 사용 중입니다: {name}")
    return StorageError(f"파일을 저장하지 못했습니다: {name} ({e.strerror or e})")


def _replace(src: str, dst: str) -> None:
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_bytes(path: str | os.PathLike, data: bytes, *, backup: bool = False) -> None:
    path = long_path(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".sonatlas-", suffix=".tmp", dir=directory)
    except OSError as e:
        raise storage_error(e, path) from e
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        _replace(tmp, path)
    except OSError as e:
        _silent_remove(tmp)
        raise storage_error(e, path) from e
    except BaseException:
        _silent_remove(tmp)
        raise


def atomic_write_json(path: str | os.PathLike, obj, *, backup: bool = False, indent: int | None = 1) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False)
    atomic_write_bytes(path, text.encode("utf-8"), backup=backup)


def read_json(path: str | os.PathLike, *, what: str = "파일"):
    """JSON을 읽는다. 없으면 FileNotFoundError, 깨졌으면 ProjectFormatError."""
    p = long_path(path)
    try:
        with open(p, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        raise
    except OSError as e:
        raise StorageError(f"{what}을(를) 읽지 못했습니다: {path} ({e.strerror or e})") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProjectFormatError(f"{what}이(가) 손상되었습니다: {Path(path).name} ({e})") from e


def read_json_with_backup(path: str | os.PathLike, *, what: str = "파일") -> tuple[object, bool]:
    """본 파일이 깨졌으면 .bak을 읽는다. (데이터, 백업에서 읽었는지)."""
    try:
        return read_json(path, what=what), False
    except ProjectFormatError:
        bak = os.fspath(path) + ".bak"
        if os.path.exists(bak):
            return read_json(bak, what=what), True
        raise


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
