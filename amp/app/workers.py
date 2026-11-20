"""백그라운드 작업 실행기.

오디오 디코딩·분석처럼 오래 걸리는 일을 QThreadPool에서 실행하고,
진행률·완료·실패를 Qt 시그널로 UI 스레드에 전달한다.
작업 함수는 ``fn(progress, token)`` 형태이며 Qt를 몰라도 된다.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot

from amp.core.errors import AmpError, AnalysisCancelled
from amp.engine.context import CancelToken, ProgressFn

log = logging.getLogger(__name__)

TaskFn = Callable[[ProgressFn, CancelToken], Any]
_PROGRESS_INTERVAL = 0.05  # 초. 진행률 시그널이 UI를 뒤덮지 않도록 간격을 둔다.


class TaskHandle(QObject):
    """작업 하나의 상태 창구. UI 스레드에 살고, 시그널은 작업 스레드에서 발생한다."""

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)  # 사용자용 메시지, 상세(traceback)
    cancelled = Signal()
    ended = Signal()  # 성공·실패·취소 어느 경우든 마지막에 한 번

    def __init__(self, name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.token = CancelToken()
        self.done = False

    def cancel(self) -> None:
        self.token.cancel()


class _Runnable(QRunnable):
    def __init__(self, fn: TaskFn, handle: TaskHandle) -> None:
        super().__init__()
        self._fn = fn
        self._handle = handle
        self._last_emit = 0.0

    def _progress(self, fraction: float, message: str) -> None:
        now = time.monotonic()
        if now - self._last_emit >= _PROGRESS_INTERVAL or fraction >= 1.0:
            self._last_emit = now
            self._handle.progress.emit(float(fraction), message)

    def run(self) -> None:
        h = self._handle
        try:
            result = self._fn(self._progress, h.token)
        except AnalysisCancelled:
            h.cancelled.emit()
        except AmpError as e:
            h.failed.emit(str(e), traceback.format_exc())
        except MemoryError:
            h.failed.emit("메모리가 부족해 작업을 마치지 못했습니다.", traceback.format_exc())
        except ModuleNotFoundError as e:  # 설치 목록에서 빠진 패키지: '예상하지 못한 오류' 대신 설치 방법을 알린다
            log.exception("작업 실패 (패키지 없음): %s", h.name)
            h.failed.emit(f"필요한 패키지 '{e.name}'가 설치되어 있지 않습니다. 저장소 폴더의 run.bat을 다시 실행하면 "
                          "설치합니다(또는 pip install -r requirements.txt).", traceback.format_exc())
        except Exception as e:  # noqa: BLE001 - 어떤 오류든 UI에 알려야 한다
            log.exception("작업 실패: %s", h.name)
            h.failed.emit(f"예상하지 못한 오류가 발생했습니다: {e}", traceback.format_exc())
        else:
            if h.token.cancelled:
                h.cancelled.emit()
            else:
                h.finished.emit(result)
        finally:
            h.ended.emit()


class TaskRunner(QObject):
    """작업을 스레드 풀에서 실행한다.

    ``submit`` 이 돌려준 핸들에 시그널을 연결하면 된다. 실제 실행은 다음 이벤트 루프에서 시작한다.
    """

    activeCountChanged = Signal(int)
    _startRequested = Signal(object)

    def __init__(self, parent: QObject | None = None, max_threads: int = 2) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max_threads)
        self._active: set[TaskHandle] = set()
        self._startRequested.connect(self._start, Qt.ConnectionType.QueuedConnection)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def submit(self, name: str, fn: TaskFn) -> TaskHandle:
        handle = TaskHandle(name, self)
        self._active.add(handle)
        # 람다가 아닌 메서드로 연결해야 실행기가 먼저 삭제될 때 연결도 함께 끊긴다
        handle.ended.connect(self._on_ended)
        # 다음 이벤트 루프에서 시작한다(큐 연결). 호출한 쪽이 submit() 직후 시그널을 연결해도
        # 아주 빨리 끝나는 작업의 결과를 놓치지 않게 하기 위해서다.
        self._startRequested.emit(_Runnable(fn, handle))
        self.activeCountChanged.emit(len(self._active))
        return handle

    @Slot(object)
    def _start(self, runnable: "_Runnable") -> None:
        self._pool.start(runnable)

    @Slot()
    def _on_ended(self) -> None:
        handle = self.sender()
        if not isinstance(handle, TaskHandle):
            return
        handle.done = True
        self._active.discard(handle)
        handle.deleteLater()  # 이미 대기 중인 다른 연결의 ended 처리가 끝난 뒤 삭제된다
        self.activeCountChanged.emit(len(self._active))

    def cancel_all(self) -> None:
        for h in list(self._active):
            h.cancel()

    def wait(self, msecs: int = -1) -> bool:
        return self._pool.waitForDone(msecs)

    def shutdown(self, msecs: int = 5000) -> bool:
        """모든 작업에 취소를 요청하고 끝날 때까지 기다린다. 창을 닫기 전에 호출한다."""
        self.cancel_all()
        return self.wait(msecs)
