"""분석 작업 프로세스를 실행하고 진행률·결과·오류를 받는다.

이 함수는 GUI의 작업 스레드(TaskRunner)에서 호출된다. 취소 토큰을 0.1초마다 확인하고,
취소되면 프로세스를 종료한 뒤 AnalysisCancelled를 던진다. 임시 결과 폴더 정리는 호출한 쪽
(``AnalysisCache.writer``)이 맡는다.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from amp.core.errors import AnalysisCancelled, AnalysisError
from amp.engine.context import CancelToken, ProgressFn

log = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

ERROR_HINTS = {
    "missing_dependency": "보컬 분리에는 PyTorch와 Demucs가 필요합니다. 저장소 폴더의 install_separation.bat 을 실행하세요. "
                          "반주가 없는 음원이라면 분석 설정에서 보컬 분리를 '분리 안 함'으로 바꿔도 됩니다.",
    "model_download": "처음 분석할 때 Demucs 모델(약 80MB)을 인터넷에서 내려받습니다. 인터넷 연결과 디스크 여유 공간을 확인하세요.",
    "oom": "메모리가 부족합니다. 다른 프로그램을 닫거나, 분석 설정에서 처리 장치를 CPU로 바꿔 보세요.",
    "decode": "오디오 파일을 다시 확인하세요.",
}


class WorkerError(AnalysisError):
    def __init__(self, code: str, message: str, detail: str = "") -> None:
        hint = ERROR_HINTS.get(code, "")
        super().__init__(f"{message}\n{hint}".strip())
        self.code = code
        self.detail = detail


def worker_env(numba_cache_dir: str | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if numba_cache_dir and "NUMBA_CACHE_DIR" not in env:
        env["NUMBA_CACHE_DIR"] = numba_cache_dir  # 다음 실행부터 JIT 컴파일 결과를 재사용
    return env


def run_worker(
    command: str,
    args: dict,
    progress: ProgressFn | None = None,
    token: CancelToken | None = None,
    numba_cache_dir: str | None = None,
    python: str | None = None,
) -> dict:
    report = progress or (lambda f, m: None)
    cmd = [python or sys.executable, "-m", "amp.engine.worker", command, "--args-json",
           json.dumps(args, ensure_ascii=True)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, cwd=PROJECT_ROOT,
            env=worker_env(numba_cache_dir), creationflags=flags, encoding="utf-8", errors="replace",
        )
    except OSError as e:
        raise AnalysisError(f"분석 프로세스를 시작하지 못했습니다: {e}") from e

    events: queue.Queue = queue.Queue()
    stderr_tail: deque[str] = deque(maxlen=60)

    def read_stdout() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if not line:
                continue
            try:
                events.put(json.loads(line))
            except json.JSONDecodeError:
                stderr_tail.append(line)
        events.put(None)

    def read_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_tail.append(line.rstrip())

    readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for t in readers:
        t.start()

    result: dict | None = None
    error: dict | None = None
    try:
        while True:
            if token is not None and token.cancelled:
                _terminate(proc)
                raise AnalysisCancelled("분석을 취소했습니다.")
            try:
                ev = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if ev is None:
                break
            kind = ev.get("event")
            if kind == "progress":
                report(float(ev.get("fraction", 0.0)), str(ev.get("message", "")))
            elif kind == "result":
                result = ev
            elif kind == "error":
                error = ev
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            _terminate(proc)
        for t in readers:
            t.join(timeout=2)

    if error is not None:
        raise WorkerError(str(error.get("code", "internal")), str(error.get("message", "")), str(error.get("detail", "")))
    if result is None or proc.returncode != 0:
        tail = "\n".join(stderr_tail)[-2000:]
        log.error("분석 프로세스 비정상 종료 (code=%s)\n%s", proc.returncode, tail)
        raise WorkerError("internal", f"분석 프로세스가 비정상 종료했습니다 (종료 코드 {proc.returncode}).", tail)
    result.pop("event", None)
    return result


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except OSError:
        pass
