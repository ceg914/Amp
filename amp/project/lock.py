"""프로젝트 폴더 잠금 (같은 프로젝트의 동시 편집 방지).

운영체제 파일 잠금을 쓴다 (Windows: msvcrt.locking, 그 밖: fcntl.flock).
- 잠금은 열린 파일 핸들에 붙는다. 프로세스가 죽으면 OS가 자동으로 푼다 → 비정상 종료 뒤에도 남지 않는다.
- 폴더를 복사하면 파일 내용만 복사되고 잠금은 따라가지 않는다 → 복사본은 바로 열 수 있다.
- 다른 컴퓨터에서 복사해 온 잠금 파일도 의미가 없다. 활성 여부는 PID나 파일 존재가 아니라 OS 잠금으로만 판단한다.
- 같은 프로세스 안의 두 창도 막는다 (flock은 파일을 다시 열면 별개로 취급하고, 프로세스 내부 목록도 함께 확인).

``.sonatlas.lock``       OS 잠금을 거는 빈 파일. 지우지 않는다 (지우면 두 프로세스가 각자 새 파일을 잠글 수 있다).
``.sonatlas.owner.json`` 누가 열었는지 안내용 정보 (인스턴스 ID·PID·컴퓨터 이름·시각·경로). 판단에는 쓰지 않는다.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from amp import __version__
from amp.core.errors import AmpError
from amp.core.fileio import atomic_write_json, long_path

log = logging.getLogger(__name__)

LOCK_FILE = ".sonatlas.lock"
OWNER_FILE = ".sonatlas.owner.json"
INSTANCE_ID = uuid.uuid4().hex  # 이 프로세스(실행 인스턴스) 식별자
_PROCESS_STARTED = datetime.now(timezone.utc).isoformat(timespec="seconds")
_UNSUPPORTED = {errno.ENOLCK, getattr(errno, "EOPNOTSUPP", -1), getattr(errno, "ENOTSUP", -1), errno.ENOSYS}

_held: dict[str, "ProjectLock"] = {}
_held_guard = threading.Lock()


@dataclass(frozen=True)
class LockOwner:
    instance_id: str = ""
    pid: int = 0
    host: str = ""
    user: str = ""
    process_started: str = ""
    acquired: str = ""
    project_path: str = ""
    project_id: str = ""
    app_version: str = ""

    def describe(self) -> str:
        if not self.instance_id:
            return "다른 프로그램"
        same = "이 프로그램의 다른 창" if self.instance_id == INSTANCE_ID else "다른 SONATLAS 실행"
        return f"{same} (컴퓨터 {self.host or '?'}, PID {self.pid}, {self.acquired} 부터)"


class ProjectLockedError(AmpError):
    def __init__(self, root: Path, owner: LockOwner) -> None:
        super().__init__(
            f"이 프로젝트는 {owner.describe()}에서 이미 열려 있습니다.\n{root}\n"
            "같은 프로젝트를 두 곳에서 동시에 고치면 내용이 섞일 수 있어 막았습니다. "
            "사본이 필요하면 '다른 이름으로 저장'을 쓰세요."
        )
        self.root = root
        self.owner = owner


def _key(root: Path) -> str:
    return os.path.normcase(os.path.realpath(root))


def _os_lock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # 0번 바이트 1개 (파일 끝을 넘어도 잠글 수 있다)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def read_owner(root: str | os.PathLike) -> LockOwner:
    try:
        with open(long_path(Path(root) / OWNER_FILE), encoding="utf-8") as f:
            d = json.load(f)
        return LockOwner(**{k: d[k] for k in LockOwner.__dataclass_fields__ if k in d})
    except (OSError, ValueError, TypeError):
        return LockOwner()


class ProjectLock:
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self._fd: int | None = None
        self.supported = True  # OS 잠금을 쓸 수 없는 파일 시스템(일부 네트워크 드라이브)이면 False

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, project_id: str = "") -> None:
        """잠금을 얻는다. 다른 곳에서 쓰는 중이면 ProjectLockedError."""
        key = _key(self.root)
        with _held_guard:
            if key in _held:
                raise ProjectLockedError(self.root, read_owner(self.root))
            fd = os.open(long_path(self.root / LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                _os_lock(fd)
            except OSError as e:
                if e.errno in _UNSUPPORTED:
                    self.supported = False
                    log.warning("이 위치는 파일 잠금을 지원하지 않습니다. 같은 프로그램 안의 중복만 막습니다: %s", self.root)
                else:
                    os.close(fd)
                    raise ProjectLockedError(self.root, read_owner(self.root)) from None
            self._fd = fd
            _held[key] = self
        owner = LockOwner(
            instance_id=INSTANCE_ID, pid=os.getpid(), host=socket.gethostname(),
            user=os.environ.get("USERNAME") or os.environ.get("USER", ""), process_started=_PROCESS_STARTED,
            acquired=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            project_path=str(self.root.resolve()), project_id=project_id, app_version=__version__,
        )
        try:  # 이전 실행(비정상 종료)이 남긴 정보는 여기서 덮어쓴다
            atomic_write_json(self.root / OWNER_FILE, asdict(owner))
        except AmpError as e:
            log.warning("잠금 소유자 정보를 쓰지 못했습니다 (잠금은 유효): %s", e)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with _held_guard:
            _held.pop(_key(self.root), None)
        try:
            if read_owner(self.root).instance_id == INSTANCE_ID:
                os.remove(long_path(self.root / OWNER_FILE))
        except OSError:
            pass
        try:
            if self.supported:
                _os_unlock(fd)
        except OSError:
            pass
        os.close(fd)

    def __enter__(self) -> "ProjectLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def is_locked(root: str | os.PathLike) -> LockOwner | None:
    """지금 누가 쓰고 있으면 그 정보를, 아니면 None. 잠금을 잠깐 잡았다가 바로 푼다."""
    probe = ProjectLock(root)
    try:
        probe.acquire()
    except ProjectLockedError as e:
        return e.owner
    except OSError:
        return None
    probe.release()
    return None
