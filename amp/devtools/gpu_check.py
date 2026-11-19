"""GPU 가사 인식 진단 도구 (STEP 3.5). 사용자 PC(Windows + NVIDIA)에서 실행해 결과 파일을 보내 받는다.

    python scripts/gpu_check.py                      # 합성 음성 60초로 large-v3·turbo·MMS 확인
    python scripts/gpu_check.py --audio 보컬.wav      # 실제 분리 보컬로 (프로젝트의 stems/vocals.wav)
    python scripts/gpu_check.py --models turbo --seconds 30 --no-cancel

앱과 같은 작업 프로세스(amp.engine.worker의 asr·emissions 명령)를 돌린다. 이 도구 자신은 CUDA를 쓰지 않는다
(쓰면 그 메모리가 측정에 섞인다). GPU 메모리는 두 가지로 잰다.
- 작업 프로세스 안 PyTorch 할당기 최대치(모델 로딩 순간, 작업 전체): torch.cuda.max_memory_allocated
- GPU 전체 사용량(nvidia-smi, 0.5초 간격): 다른 프로그램·CUDA 컨텍스트까지 포함. 작업 전 대비 증가분을 적는다.
GPU가 없으면 GPU 항목은 '확인 못 함'으로 적고 성공으로 표시하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

MB = 2**20
RELEASE_TOL_MB = 200  # 작업이 끝난 뒤 GPU 사용량이 작업 전보다 이만큼 넘게 많으면 '반환 안 됨'

_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device_count"] = torch.cuda.device_count()
        out["gpu_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        out["gpu_free_mb"], out["gpu_total_mb"] = round(free / 2**20), round(total / 2**20)
        out["cudnn"] = torch.backends.cudnn.version()
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    import whisper
    out["whisper"] = getattr(whisper, "__version__", "?")
except Exception as e:
    out["whisper_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""


def _run_probe() -> dict:
    """torch·CUDA 정보는 별도 프로세스에서 읽는다 (이 도구에 CUDA 컨텍스트가 생기지 않게)."""
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="replace")
        line = (r.stdout.strip().splitlines() or ["{}"])[-1]
        out = json.loads(line)
        if r.returncode != 0:
            out["probe_error"] = r.stderr[-1500:]
        return out
    except Exception as e:  # noqa: BLE001
        return {"probe_error": f"{type(e).__name__}: {e}"}


def nvidia_smi(query: str = "memory.used,memory.total,name,driver_version") -> list[str] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=10, **_no_window())
        if r.returncode != 0:
            return None
        return [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    except Exception:  # noqa: BLE001
        return None


def _no_window() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def gpu_used_mb() -> float | None:
    row = nvidia_smi("memory.used")
    try:
        return float(row[0]) if row else None
    except (TypeError, ValueError):
        return None


class GpuMonitor:
    """작업 동안 GPU 전체 사용량(MB)을 0.5초마다 재서 최대치를 남긴다."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "GpuMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = gpu_used_mb()
            if v is not None:
                self.samples.append((round(time.monotonic() - self._t0, 1), v))
            self._stop.wait(self.interval)

    @property
    def peak(self) -> float | None:
        return max((v for _t, v in self.samples), default=None)


def wait_for_release(baseline: float | None, timeout: float = 15.0) -> tuple[float | None, float]:
    """작업 프로세스가 끝난 뒤 GPU 사용량이 작업 전 수준으로 돌아오는지 본다. (마지막 값, 걸린 초)."""
    t0 = time.monotonic()
    last = gpu_used_mb()
    while baseline is not None and last is not None and last > baseline + RELEASE_TOL_MB and time.monotonic() - t0 < timeout:
        time.sleep(0.5)
        last = gpu_used_mb()
    return last, round(time.monotonic() - t0, 1)


def _app_model_dirs() -> tuple[str, str]:
    """앱이 모델을 받아 두는 폴더 (다시 내려받지 않게 같은 곳을 쓴다).

    Qt 앱 객체는 명령줄 실행(main)에서만 만든다. 시험 등 다른 프로세스 안에서 만들면 그 뒤 화면 위젯을 만들 수 없다.
    """
    try:
        from PySide6.QtCore import QCoreApplication

        from amp.app.session import app_data_dir

        if QCoreApplication.instance() is not None:
            return app_data_dir("models", "whisper"), app_data_dir("models", "mms")
    except Exception:  # noqa: BLE001
        pass
    base = os.path.expanduser("~/.sonatlas")
    return os.path.join(base, "models", "whisper"), os.path.join(base, "models", "mms")


def _cli_app():
    """명령줄에서 앱과 같은 이름의 Qt 앱 객체를 만든다 (앱과 같은 모델 폴더를 찾기 위해)."""
    try:
        from PySide6.QtCore import QCoreApplication

        from amp import APP_NAME, ORGANIZATION

        app = QCoreApplication.instance() or QCoreApplication([])
        app.setOrganizationName(ORGANIZATION)
        app.setApplicationName(APP_NAME)
        return app
    except Exception:  # noqa: BLE001
        return None


def prepare_audio(path: str | None, seconds: float, out_dir: Path) -> tuple[str, str]:
    """(검사에 쓸 WAV 경로, 설명). 지정 파일이 없으면 합성 음성(가사 없음)을 만든다."""
    import numpy as np
    import soundfile as sf

    dst = out_dir / "input.wav"
    if path:
        from amp.audio.decoder import load_audio

        a = load_audio(path)
        y = a.mono()[: int(seconds * a.sample_rate)] if seconds > 0 else a.mono()
        sf.write(dst, y, a.sample_rate)
        return str(dst), f"{os.path.basename(path)} 앞 {len(y) / a.sample_rate:.0f}초"
    from amp.devtools.synth import SR, sung

    dur = max(10.0, seconds or 60.0)
    melody, t, k = [], 0.5, 0
    while t + 0.6 < dur:
        melody.append((t, t + 0.45, [60, 62, 64, 67, 69, 67, 64, 62][k % 8]))
        t += 0.5 + (0.4 if k % 8 == 7 else 0.0)
        k += 1
    y = sung(melody, dur, vibrato_cents=20)
    sf.write(dst, np.asarray(y, np.float32), SR)
    return str(dst), f"합성 음성 {dur:.0f}초 (가사 없음: 인식 글자는 의미 없고 속도·메모리만 본다)"


def run_job(command: str, args: dict, *, cancel_after: float | None = None, cancel_when: str = "") -> dict:
    """작업 프로세스 하나를 돌리며 시간·GPU 사용량을 잰다."""
    from amp.engine.context import AnalysisCancelled, CancelToken
    from amp.engine.worker_client import WorkerError, run_worker

    token = CancelToken()
    msgs: list[str] = []
    started = {"t": None}

    def progress(f: float, m: str) -> None:
        if m and (not msgs or msgs[-1] != m):
            msgs.append(m)
        if cancel_after is not None and started["t"] is None and cancel_when in (m or ""):
            started["t"] = time.monotonic()
        if cancel_after is not None and started["t"] is not None and time.monotonic() - started["t"] >= cancel_after:
            token.cancel()

    baseline = gpu_used_mb()
    out: dict = {"command": command, "gpu_used_before_mb": baseline}
    t0 = time.monotonic()
    with GpuMonitor() as mon:
        try:
            res = run_worker(command, args, progress, token)
            out.update(ok=True, result={k: v for k, v in res.items() if k != "results"})
        except AnalysisCancelled:
            out.update(ok=False, cancelled=True)
        except WorkerError as e:
            out.update(ok=False, error_code=e.code, error=str(e))
        except Exception as e:  # noqa: BLE001
            out.update(ok=False, error_code="internal", error=f"{type(e).__name__}: {e}")
    out["wall_s"] = round(time.monotonic() - t0, 1)
    if out.get("cancelled") and started["t"] is not None:
        out["cancel_to_exit_s"] = round(time.monotonic() - started["t"] - (cancel_after or 0), 1)
    out["gpu_used_peak_mb"] = mon.peak
    if baseline is not None and mon.peak is not None:
        out["gpu_used_increase_peak_mb"] = round(mon.peak - baseline)
    after, waited = wait_for_release(baseline)
    out["gpu_used_after_mb"], out["release_wait_s"] = after, waited
    out["gpu_released"] = None if baseline is None or after is None else after <= baseline + RELEASE_TOL_MB
    out["messages"] = msgs[-12:]
    return out


def check(models: list[str], audio: str | None, seconds: float, out_root: Path, device: str = "cuda",
          cancel: bool = True, mms: bool = True) -> tuple[dict, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (out_root / f"gpu_check-{stamp}").resolve()  # 작업 프로세스는 다른 폴더에서 돈다
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"created": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
                 "machine": platform.machine(), "requested_device": device}
    from amp import __version__

    rep["app_version"] = __version__
    print("환경 확인 중…", flush=True)
    rep["env"] = _run_probe()
    smi = nvidia_smi()
    rep["nvidia_smi"] = dict(zip(("memory_used_mb", "memory_total_mb", "name", "driver"), smi)) if smi else None
    whisper_root, mms_root = _app_model_dirs()
    rep["model_dirs"] = {"whisper": whisper_root, "mms": mms_root}
    has_cuda = bool(rep["env"].get("cuda_available"))
    if device == "cuda" and not has_cuda:
        rep["summary"] = {"gpu": "확인 못 함", "reason": "PyTorch가 CUDA를 쓰지 못합니다 (CUDA판 PyTorch·드라이버 확인). "
                                                        "GPU 시험을 하지 않았습니다."}
        return _write(rep, out_dir)
    wav, desc = prepare_audio(audio, seconds, out_dir)
    rep["audio"] = desc
    from amp.lyrics.asr import resolve_model

    free = rep["env"].get("gpu_free_mb")
    rep["auto_model_would_pick"] = resolve_model("auto", device if has_cuda else "cpu",
                                                 int(free * MB) if free is not None else None)
    rep["runs"] = []
    for m in models:
        print(f"가사 인식 {m} ({device}) 실행 중… 처음이면 모델을 내려받습니다.", flush=True)
        with tempfile.TemporaryDirectory(prefix="sonatlas-gpu-") as tmp:
            r = run_job("asr", {"input": wav, "out_dir": tmp, "model": m, "language": "auto", "device": device,
                                "download_root": whisper_root})
            p = Path(tmp) / "asr.json"
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                r["asr"] = {k: raw.get(k) for k in ("model", "device", "language", "note")}
                r["asr"]["stats"] = raw.get("stats")
                r["asr"]["segments"] = len(raw.get("segments") or ())
        r["model_requested"] = m
        rep["runs"].append(r)
        _print_run(r)
    if cancel:
        m = models[0]
        print(f"취소 시험: {m} 인식 시작 3초 뒤 취소…", flush=True)
        with tempfile.TemporaryDirectory(prefix="sonatlas-gpu-") as tmp:
            r = run_job("asr", {"input": wav, "out_dir": tmp, "model": m, "language": "ko", "device": device,
                                "download_root": whisper_root}, cancel_after=3.0, cancel_when="가사 인식 중")
        r["model_requested"] = m
        rep["cancel"] = r
        _print_run(r)
    if mms:
        print(f"시간 정렬 모델(MMS) 계산 ({device}) 중… 처음이면 약 1.2GB를 내려받습니다.", flush=True)
        with tempfile.TemporaryDirectory(prefix="sonatlas-gpu-") as tmp:
            r = run_job("emissions", {"input": wav, "out_dir": tmp, "device": device, "model_dir": mms_root})
            p = Path(tmp) / "info.json"
            if p.exists():
                r["mms"] = json.loads(p.read_text(encoding="utf-8"))
        rep["mms"] = r
        _print_run(r)
    rep["summary"] = summarize(rep, device)
    return _write(rep, out_dir)


def summarize(rep: dict, device: str) -> dict:
    s: dict = {}
    on_gpu = [r for r in rep.get("runs", []) if r.get("ok") and (r.get("asr") or {}).get("device") == "cuda"]
    s["gpu"] = ("GPU 인식 성공" if on_gpu else "GPU 인식 안 됨") if device == "cuda" else "CPU로만 실행 (GPU 확인 아님)"
    for r in rep.get("runs", []):
        m = r["model_requested"]
        if r.get("ok"):
            st = (r.get("asr") or {}).get("stats") or {}
            s[m] = f"성공 · {(r.get('asr') or {}).get('device')} · 로딩 {st.get('load_s')}초 · 인식 {st.get('transcribe_s')}초"
        elif r.get("error_code") == "oom_model":
            s[m] = "GPU 메모리 부족 → 안내 메시지로 멈춤 (조용히 CPU로 넘어가지 않음)"
        else:
            s[m] = f"실패 ({r.get('error_code')}): {str(r.get('error'))[:200]}"
        if r.get("gpu_released") is False:
            s[m] += " · 작업 뒤 GPU 메모리가 돌아오지 않음"
    c = rep.get("cancel")
    if c:
        s["cancel"] = ("취소됨 · GPU 메모리 " + ("반환됨" if c.get("gpu_released") else "확인 못 함" if c.get("gpu_released") is None
                                             else "반환 안 됨")) if c.get("cancelled") else "취소 시험이 끝나기 전에 작업이 끝났거나 실패함"
    mm = rep.get("mms")
    if mm:
        info = mm.get("mms") or {}
        s["mms"] = f"{'성공' if mm.get('ok') else '실패'} · {info.get('device', '?')} · {mm.get('wall_s')}초"
    return s


def _print_run(r: dict) -> None:
    st = ((r.get("asr") or {}).get("stats")) or ((r.get("mms") or {}).get("stats")) or {}
    print(f"  → ok={r.get('ok')} cancelled={r.get('cancelled', False)} code={r.get('error_code', '')} "
          f"{r.get('wall_s')}초, GPU 증가 최대 {r.get('gpu_used_increase_peak_mb')}MB, 반환 {r.get('gpu_released')}, "
          f"장치 {st.get('device') or (r.get('mms') or {}).get('device')}", flush=True)


def _mb(b) -> str:
    return "—" if b is None else f"{b / MB:,.0f}MB"


def _v(x, unit: str = "") -> str:
    return "—" if x is None else f"{x}{unit}"


def markdown(rep: dict) -> str:
    env = rep.get("env", {})
    lines = [f"# SONATLAS GPU 진단 ({rep['created']})", "",
             f"- 앱 버전 {_v(rep.get('app_version'))} · {_v(rep.get('platform'))} · Python {_v(env.get('python'))}",
             f"- PyTorch {_v(env.get('torch'))} (CUDA {_v(env.get('torch_cuda'))}) · cuDNN {_v(env.get('cudnn'))} · "
             f"Whisper {_v(env.get('whisper'))}",
             f"- CUDA 사용 가능: {_v(env.get('cuda_available'))} · GPU: {_v(env.get('gpu_name'))} · "
             f"전체 {_v(env.get('gpu_total_mb'), 'MB')} · 여유 {_v(env.get('gpu_free_mb'), 'MB')}",
             f"- nvidia-smi: {rep.get('nvidia_smi') or '없음'}",
             f"- 입력: {rep.get('audio', '—')}",
             f"- '자동' 모델 선택이 고를 모델(지금 여유 메모리 기준): {rep.get('auto_model_would_pick', '—')}", "",
             "## 요약", ""]
    lines += [f"- {k}: {v}" for k, v in (rep.get("summary") or {}).items()]
    if rep.get("runs"):
        lines += ["", "## 가사 인식", "",
                  "| 요청 모델 | 결과 | 실제 장치·모델 | 로딩 | 인식 | 소리 길이 | VRAM 최대(로딩) | VRAM 최대(전체) | GPU 사용 증가 최대 | 최대 RAM | 반환 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rep["runs"]:
            a = r.get("asr") or {}
            st = a.get("stats") or {}
            res = "성공" if r.get("ok") else f"실패 ({r.get('error_code')})"
            lines.append(f"| {r['model_requested']} | {res} | {_v(a.get('device'))} {a.get('model') or ''} | {_v(st.get('load_s'), '초')} | "
                         f"{_v(st.get('transcribe_s'), '초')} | {_v(st.get('audio_s'), '초')} | {_mb(st.get('gpu_peak_load_bytes'))} | "
                         f"{_mb(st.get('gpu_peak_bytes'))} | {_v(r.get('gpu_used_increase_peak_mb'), 'MB')} | "
                         f"{_v(st.get('peak_rss_mb'), 'MB')} | {_v(r.get('gpu_released'))} |")
        for r in rep["runs"]:
            if not r.get("ok"):
                lines += ["", f"**{r['model_requested']} 오류**: {r.get('error')}"]
            note = (r.get("asr") or {}).get("note")
            if note:
                lines += ["", f"{r['model_requested']} 안내: {note}"]
    c = rep.get("cancel")
    if c:
        lines += ["", "## 취소 시험", "",
                  f"- 요청 모델 {c.get('model_requested')}: 취소됨 {c.get('cancelled', False)}, 취소 뒤 종료까지 {_v(c.get('cancel_to_exit_s'), '초')}, "
                  f"GPU 사용량 작업 전 {_v(c.get('gpu_used_before_mb'), 'MB')} → 최대 {_v(c.get('gpu_used_peak_mb'), 'MB')} → 끝난 뒤 "
                  f"{_v(c.get('gpu_used_after_mb'), 'MB')} (반환 {_v(c.get('gpu_released'))}, {c.get('release_wait_s')}초 기다림)"]
        if not c.get("cancelled"):
            lines.append(f"- 결과: {'취소 전에 끝남' if c.get('ok') else '오류 ' + str(c.get('error'))}")
    mm = rep.get("mms")
    if mm:
        info = mm.get("mms") or {}
        st = info.get("stats") or {}
        lines += ["", "## 시간 정렬 모델 (MMS)", "",
                  f"- 결과 {'성공' if mm.get('ok') else '실패 ' + str(mm.get('error'))} · 장치 {_v(info.get('device'))} "
                  f"{info.get('note', '')} · 로딩 {_v(st.get('load_s'), '초')} · 계산 {_v(st.get('emissions_s'), '초')} · "
                  f"VRAM 최대 {_v(st.get('gpu_peak_mb'), 'MB')} · 최대 RAM {_v(st.get('peak_rss_mb'), 'MB')} · "
                  f"GPU 반환 {_v(mm.get('gpu_released'))}"]
    lines += ["", "이 파일과 같은 폴더의 gpu_check.json을 함께 보내 주세요. 오디오는 들어 있지 않습니다(input.wav 제외)."]
    return "\n".join(lines) + "\n"


def _write(rep: dict, out_dir: Path) -> tuple[dict, Path]:
    (out_dir / "gpu_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    md = out_dir / "gpu_check.md"
    md.write_text(markdown(rep), encoding="utf-8")
    return rep, md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SONATLAS GPU 가사 인식 진단")
    ap.add_argument("--models", default="large-v3,turbo", help="쉼표로 구분 (기본 large-v3,turbo)")
    ap.add_argument("--audio", default=None, help="검사에 쓸 오디오 (예: 프로젝트의 stems/vocals.wav). 없으면 합성 음성")
    ap.add_argument("--seconds", type=float, default=60.0, help="앞에서부터 이만큼만 쓴다 (0이면 전체)")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-cancel", action="store_true", help="취소 시험을 건너뛴다")
    ap.add_argument("--no-mms", action="store_true", help="시간 정렬 모델 시험을 건너뛴다")
    ap.add_argument("--out", default="gpu_check_reports", help="결과 폴더")
    a = ap.parse_args(argv)
    app = _cli_app()  # 모델 폴더 경로에 필요 (끝날 때까지 살아 있어야 한다)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    rep, md = check(models, a.audio, a.seconds, Path(a.out), a.device, not a.no_cancel, not a.no_mms)
    del app
    print("\n" + md.read_text(encoding="utf-8"))
    print(f"결과 파일: {md.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
