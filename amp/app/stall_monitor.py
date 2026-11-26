"""화면 멈춤 기록 (v0.7.0 명세 14.5). 메인 스레드가 100ms 이상 이벤트를 처리하지 못한 때를 적는다.

- 메인 스레드에서 짧은 간격(기본 25ms) 타이머를 돌려, 다음 신호가 예정보다 늦게 온 만큼을 '멈춘 시간'으로 본다
  (타이머 간격만큼의 오차가 있다: 125ms 멈춤은 100~150ms로 보일 수 있다)
- 그때 사용자가 마지막으로 누른 메뉴·단추 이름과 실행 중인 작업 이름을 같이 남긴다 (원인을 좁히는 단서일 뿐, 원인 판정이 아님)
- 기록은 메모리(최근 200개)와 앱 로그에 남기고, 사용자가 '저장'하면 JSON 파일로 쓴다. 자동으로 파일을 만들지 않는다
- v0.6.0 보고서의 466ms 멈춤은 원인을 찾지 못했다. 이 기록이 해결을 뜻하지 않는다
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass

from PySide6.QtCore import QObject, QTimer, Signal

log = logging.getLogger(__name__)

THRESHOLD_MS = 100.0
INTERVAL_MS = 25
KEEP = 200


@dataclass(frozen=True)
class Stall:
    at: str  # 끝난 시각 (ISO)
    duration_ms: float  # 예정보다 늦어진 시간 (≈ 멈춘 시간)
    context: str = ""  # 마지막으로 누른 메뉴·단추
    task: str = ""  # 그때 돌던 작업
    uptime_s: float = 0.0


class StallMonitor(QObject):
    stalled = Signal(object)  # Stall

    def __init__(self, parent=None, threshold_ms: float = THRESHOLD_MS, interval_ms: int = INTERVAL_MS,
                 clock=time.perf_counter) -> None:
        super().__init__(parent)
        self.threshold_ms = float(threshold_ms)
        self.interval_ms = int(interval_ms)
        self._clock = clock
        self._t0 = clock()
        self._last = None
        self.records: deque[Stall] = deque(maxlen=KEEP)
        self.context = ""
        self.task = ""
        self.worst_ms = 0.0
        self.ticks = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self.interval_ms)
        self._timer.timeout.connect(self._tick)

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        self._last = self._clock()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._last = None

    def note_action(self, text: str) -> None:
        self.context = str(text or "")

    def _tick(self) -> None:
        now = self._clock()
        self.ticks += 1
        if self._last is not None:
            late = (now - self._last) * 1000.0 - self.interval_ms
            if late >= self.threshold_ms:
                from datetime import datetime, timezone

                s = Stall(datetime.now(timezone.utc).isoformat(timespec="milliseconds"), round(late, 1), self.context,
                          self.task, round(now - self._t0, 2))
                self.records.append(s)
                self.worst_ms = max(self.worst_ms, s.duration_ms)
                log.info("화면 멈춤 %.0fms (마지막 동작: %s, 작업: %s)", s.duration_ms, s.context or "-", s.task or "-")
                self.stalled.emit(s)
        self._last = now

    def summary(self) -> dict:
        ds = [r.duration_ms for r in self.records]
        return {"count": len(ds), "worst_ms": max(ds) if ds else 0.0,
                "over_200ms": sum(1 for d in ds if d >= 200), "over_500ms": sum(1 for d in ds if d >= 500),
                "threshold_ms": self.threshold_ms, "interval_ms": self.interval_ms}

    def to_json(self) -> dict:
        return {"schema": "sonatlas.ui_stalls", "schema_version": 1, "summary": self.summary(),
                "note": "타이머 간격만큼 오차가 있습니다. '마지막 동작'은 원인을 좁히는 단서일 뿐 원인 판정이 아닙니다.",
                "records": [asdict(r) for r in self.records]}

    def save(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
