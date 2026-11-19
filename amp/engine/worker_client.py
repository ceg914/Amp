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
import tempfile
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
    "oom_model": "",  # 메시지에 이미 해결 방법(더 작은 모델)이 들어 있다
    "decode": "오디오 파일을 다시 확인하세요.",
    "model_file": "분석 설정의 보컬 분리에서 RoFormer 모델 파일(.ckpt)을 다시 고르세요.",
    "model_config": "모델과 함께 받은 .yaml 설정 파일을 분석 설정에서 지정하세요. 다른 구조의 RoFormer(BS-RoFormer 등)는 아직 지원하지 않습니다.",
    "disk_full": "디스크 공간을 확보한 뒤 다시 실행하세요. 분리 결과는 곡 1분당 약 30MB가 필요합니다.",
    "killed": "메모리 부족이나 외부 요인으로 분석 프로세스가 강제 종료되었을 수 있습니다. 다른 프로그램을 닫거나 "
              "처리 장치를 CPU로 바꿔 다시 시도하세요. 완성되지 않은 결과는 저장하지 않았습니다.",
}
# Windows 예외 종료 코드 (접근 위반, 스택 오버런, 메모리 부족 등)
_WINDOWS_CRASH_CODES = {0xC0000005, 0xC0000409, 0xC0000017, 0xC00000FD, 0xC0000374}


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
    # 인자는 임시 파일로 넘긴다. 명령줄에 JSON을 그대로 붙이면 가사 정렬(음절 수백 개의 읽기·고정점)이
    # Windows 명령줄 한도(32,767자)를 넘어 "[WinError 206] 파일 이름이나 확장명이 너무 깁니다"로 시작조차 못 했다.
    try:
        fd, args_path = tempfile.mkstemp(prefix="sonatlas-args-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=True)
    except OSError as e:
        raise AnalysisError(f"분석 프로세스에 넘길 임시 파일을 만들지 못했습니다 (디스크 공간을 확인하세요): {e}") from e
    cmd = [python or sys.executable, "-m", "amp.engine.worker", command, "--args-file", args_path]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, cwd=PROJECT_ROOT,
            env=worker_env(numba_cache_dir), creationflags=flags, encoding="utf-8", errors="replace",
        )
    except OSError as e:
        _remove(args_path)
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
        _remove(args_path)

    if error is not None:
        log.error("분석 프로세스 오류 (명령 %s, 코드 %s): %s\n%s", command, error.get("code"), error.get("message"),
                  error.get("detail", ""))
        raise WorkerError(str(error.get("code", "internal")), str(error.get("message", "")), str(error.get("detail", "")))
    if result is None or proc.returncode != 0:
        tail = "\n".join(stderr_tail)[-2000:]
        code = proc.returncode
        log.error("분석 프로세스 비정상 종료 (명령 %s, 종료 코드 %s)\n%s", command, code, tail)
        killed = code is not None and (code < 0 or (code & 0xFFFFFFFF) in _WINDOWS_CRASH_CODES)
        raise WorkerError("killed" if killed else "internal",
                          f"분석 프로세스가 비정상 종료했습니다 (종료 코드 {code}).", tail)
    result.pop("event", None)
    return result


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass  # 남아도 임시 폴더라 다음 정리 때 지워진다


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except OSError:
        pass
