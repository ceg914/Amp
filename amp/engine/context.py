"""분석 작업의 진행률 보고와 취소."""

from __future__ import annotations

import threading
from typing import Callable

from amp.core.errors import AnalysisCancelled

ProgressFn = Callable[[float, str], None]


class CancelToken:
    """다른 스레드에서 취소를 요청할 수 있는 플래그."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise AnalysisCancelled("분석을 취소했습니다.")


class StageContext:
    """한 분석 단계에 넘기는 진행률/취소 창구.

    단계는 0~1 범위로만 보고하고, 전체 진행률 환산은 파이프라인이 맡는다.
    """

    def __init__(self, report: ProgressFn, token: CancelToken, base: float = 0.0, span: float = 1.0) -> None:
        self._report = report
        self.token = token
        self._base = base
        self._span = span

    def progress(self, fraction: float, message: str = "") -> None:
        self.token.raise_if_cancelled()
        fraction = min(max(fraction, 0.0), 1.0)
        self._report(self._base + self._span * fraction, message)

    def check_cancelled(self) -> None:
        self.token.raise_if_cancelled()
