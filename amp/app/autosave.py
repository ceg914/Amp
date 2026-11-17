"""자동 저장과 비정상 종료 복구.

실행 중인 세션마다 복구 폴더에 ``<세션id>.lock`` 을 잡고,
변경 사항이 있으면 주기적으로 ``<세션id>.ampproj`` 와 ``<세션id>.json`` 을 쓴다.
정상 종료하면 세 파일을 모두 지운다.

다음 실행 때 잠금이 풀린(주인 프로세스가 사라진) 복구 파일이 남아 있으면
비정상 종료로 보고 복구를 제안한다. 실행 중인 다른 창의 파일은 잠금이 살아 있어 건드리지 않는다.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from PySide6.QtCore import QLockFile, QObject, QStandardPaths, QTimer

from amp.app.document import ProjectDocument
from amp.core.ids import new_id
from amp.storage.project_file import project_to_members, write_members

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_MS = 60_000


def default_recovery_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return os.path.join(base or os.path.expanduser("~/.amp"), "recovery")


@dataclass(frozen=True)
class RecoveryEntry:
    session_id: str
    project_path: str
    original_path: str | None
    saved_at: str

    @property
    def directory(self) -> str:
        return os.path.dirname(self.project_path)


def _paths(directory: str, sid: str) -> tuple[str, str, str]:
    return (
        os.path.join(directory, f"{sid}.ampproj"),
        os.path.join(directory, f"{sid}.json"),
        os.path.join(directory, f"{sid}.lock"),
    )


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("복구 파일 삭제 실패 %s: %s", path, e)


class AutosaveManager(QObject):
    def __init__(
        self,
        doc: ProjectDocument,
        directory: str | None = None,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.doc = doc
        self.directory = directory or default_recovery_dir()
        os.makedirs(self.directory, exist_ok=True)
        self.session_id = new_id()
        self.project_path, self.meta_path, lock_path = _paths(self.directory, self.session_id)
        self._lock = QLockFile(lock_path)
        self._lock.setStaleLockTime(0)  # 시간 기준 만료를 끄고, 프로세스 생존 여부로만 판단
        if not self._lock.tryLock(0):
            log.warning("자동 저장 잠금을 얻지 못했습니다: %s", lock_path)
        self._saved_revision: int | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    def save_now(self) -> bool:
        """변경이 있으면 복구 파일을 쓴다. 쓴 경우 True."""
        if not self.doc.is_dirty:
            self.discard()
            return False
        if self._saved_revision == self.doc.revision and os.path.exists(self.project_path):
            return False
        write_members(self.project_path, project_to_members(self.doc.project, self.doc.file_path))
        meta = {
            "session_id": self.session_id,
            "original_path": self.doc.file_path,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        self._saved_revision = self.doc.revision
        return True

    def _on_timer(self) -> None:
        try:
            self.save_now()
        except Exception:  # noqa: BLE001 - 자동 저장 실패로 편집을 막지 않는다
            log.exception("자동 저장 실패")

    def discard(self) -> None:
        _remove(self.project_path)
        _remove(self.meta_path)
        self._saved_revision = None

    def shutdown(self) -> None:
        self._timer.stop()
        self.discard()
        self._lock.unlock()

    # --- 복구 -------------------------------------------------------------

    @staticmethod
    def find_orphans(directory: str | None = None) -> list[RecoveryEntry]:
        """주인이 사라진 복구 파일 목록 (최근 것부터)."""
        directory = directory or default_recovery_dir()
        entries = []
        for project_path in glob.glob(os.path.join(directory, "*.ampproj")):
            sid = os.path.splitext(os.path.basename(project_path))[0]
            _, meta_path, lock_path = _paths(directory, sid)
            lock = QLockFile(lock_path)
            lock.setStaleLockTime(0)
            if not lock.tryLock(0):
                continue  # 다른 창이 사용 중
            lock.unlock()
            meta = {}
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                pass
            saved_at = meta.get("saved_at") or datetime.fromtimestamp(
                os.path.getmtime(project_path), timezone.utc
            ).isoformat(timespec="seconds")
            entries.append(RecoveryEntry(sid, project_path, meta.get("original_path"), saved_at))
        entries.sort(key=lambda e: e.saved_at, reverse=True)
        return entries

    @staticmethod
    def remove_entry(entry: RecoveryEntry) -> None:
        for path in _paths(entry.directory, entry.session_id):
            _remove(path)
