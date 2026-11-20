"""Demucs 보컬 분리.

시간 정렬을 지키는 규칙
- 입력은 GUI와 같은 디코더(``amp.audio.decoder.load_audio``)로 읽어 프레임 수가 같다.
- 모델 샘플레이트(44.1kHz)로 바꿀 때와 되돌릴 때 모두 ``scipy.signal.resample_poly`` 를 쓴다.
  선형 위상 필터를 가운데 맞춰 적용하므로 지연이 생기지 않는다.
- 마지막에 원본 프레임 수에 맞춰 자르거나 0으로 채운다.

``_identity``·``_sleep`` 모델은 torch 없이 위 경로(리샘플·길이 맞춤·파일 쓰기·취소)를 시험하기 위한 것이다.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np

DEMUCS_MODELS = ("htdemucs",)
ROFORMER_MODELS = ("mel_roformer",)  # 사용자가 고른 체크포인트 파일로 실행 (amp/engine/dsp/roformer.py)
TEST_MODELS = ("_identity", "_sleep", "_crash", "_half", "_oom")
OOM_GPU_MESSAGE = "GPU 메모리가 부족해 보컬 분리를 마치지 못했습니다. CPU로 다시 실행할 수 있지만 훨씬 오래 걸립니다."
MODEL_RATE = 44100
# demucs는 shifts마다 0~0.5초 임의 이동을 쓴다. 같은 곡을 다시 분리해도 같은 결과가 나오도록 시드를 고정한다.
SHIFT_SEED = 0


class SeparationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """(channels, frames) 배열을 리샘플한다. 같은 sr이면 그대로."""
    if src == dst:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(src, dst)
    return resample_poly(x, dst // g, src // g, axis=-1).astype(np.float32)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[..., :n]
    pad = np.zeros(x.shape[:-1] + (n - x.shape[-1],), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def find_nvidia_smi() -> str | None:
    """nvidia-smi 경로. PATH에 없어도 Windows 표준 위치(System32, NVSMI)를 본다."""
    import os
    import shutil

    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if os.name == "nt":
        for p in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
                  os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI",
                               "nvidia-smi.exe")):
            if os.path.isfile(p):
                return p
    return None


def nvidia_gpu() -> dict | None:
    """NVIDIA GPU가 보이면 {"name", "driver"}. PyTorch와 상관없이 드라이버로 확인한다."""
    import os
    import subprocess

    exe = find_nvidia_smi()
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()[:1]
    if out.returncode != 0 or not line:
        return None
    name, _, driver = line[0].partition(",")
    return {"name": name.strip(), "driver": driver.strip()}


def cpu_fallback_hint(torch_version: str, gpu: dict | None) -> str:
    """GPU가 있는데 PyTorch가 GPU를 못 쓸 때 안내. 없으면 빈 문자열."""
    if not gpu:
        return ""
    return (f"NVIDIA GPU({gpu.get('name') or '이름 모름'})가 있지만 설치된 PyTorch({torch_version})가 GPU를 쓸 수 없어 "
            "CPU로 분리했습니다. 저장소 폴더의 install_separation.bat을 다시 실행하면 GPU용 PyTorch를 설치합니다 "
            "(GPU 드라이버가 오래되었으면 드라이버를 먼저 업데이트하세요).")


def choose_device(requested: str) -> tuple[str, str]:
    """(실제 장치, 안내 메시지)."""
    import torch

    has_cuda = torch.cuda.is_available()
    if requested in ("auto", "cuda") and has_cuda:
        return "cuda", ""
    if requested in ("auto", "cuda"):
        hint = cpu_fallback_hint(torch.__version__, nvidia_gpu())
        if hint:
            return "cpu", hint
        if requested == "cuda":
            return "cpu", "GPU(CUDA)를 쓸 수 없어 CPU로 실행합니다. CUDA판 PyTorch가 설치되어 있는지 확인하세요."
    return "cpu", ""


def probe() -> dict:
    """설치 상태 확인 (분석 설정 패널의 [GPU 확인])."""
    info: dict = {"torch": None, "demucs": None, "cuda": False, "gpu": None, "cuda_version": None}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        info["error"] = f"PyTorch를 불러오지 못했습니다: {e}"
        return info
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    info["cuda_version"] = torch.version.cuda
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        gpu = nvidia_gpu()
        info["nvidia_gpu"] = gpu
        info["hint"] = cpu_fallback_hint(torch.__version__, gpu)
    try:
        import demucs

        info["demucs"] = getattr(demucs, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        info["error"] = f"Demucs를 불러오지 못했습니다: {e}"
    return info


def separate(
    samples: np.ndarray,
    sr: int,
    model_name: str = "htdemucs",
    device: str = "auto",
    progress: Callable[[float, str], None] | None = None,
    model_files: dict | None = None,
    cpu_fallback: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """(frames, channels) → (vocals, accompaniment, info). 출력은 입력과 같은 sr·프레임 수다.

    model_files: RoFormer용 {"checkpoint": 경로, "config": .yaml 경로 또는 ""}.
    cpu_fallback: False면 GPU 메모리가 부족할 때 CPU로 다시 돌리지 않고 SeparationError("oom_gpu")를 낸다.
    """
    report = progress or (lambda f, m: None)
    if model_name not in DEMUCS_MODELS + ROFORMER_MODELS + TEST_MODELS:
        raise SeparationError("internal", f"알 수 없는 분리 모델: {model_name}")
    files = model_files or {}
    if model_name in ROFORMER_MODELS and not files.get("checkpoint"):
        raise SeparationError("model_file", "RoFormer 모델 파일(.ckpt)을 고르지 않았습니다. 분석 설정에서 모델 파일을 지정하세요.")
    frames, in_ch = samples.shape
    wav = samples.T.astype(np.float32)  # (ch, frames)
    if in_ch == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif in_ch > 2:
        wav = wav[:2]
    wav = _resample(wav, sr, MODEL_RATE)
    info: dict = {"model": model_name}

    ref = wav.mean(axis=0)
    mean, std = float(ref.mean()), float(ref.std())
    if std < 1e-8:  # 무음: 정규화하면 0으로 나누게 되므로 분리를 건너뛴다
        vocals = np.zeros_like(wav)
        info["silent_input"] = True
        report(1.0, "무음 입력 — 분리를 건너뜀")
    elif model_name in ROFORMER_MODELS:  # 정규화 없이 원래 크기로 (참조 구현과 같게). 반주 = 원본 − 보컬
        vocals, dev_info = _run_roformer(wav, files, device, report, cpu_fallback)
        info.update(dev_info)
        accompaniment = (wav - vocals).astype(np.float32)
    else:
        norm = (wav - mean) / std
        if model_name in TEST_MODELS:
            sources = _run_test_model(model_name, norm, report, cpu_fallback, device)
            info["device"] = "cpu"
        else:
            sources, dev_info = _run_demucs(model_name, norm, device, report, cpu_fallback)
            info.update(dev_info)
        vocals = sources["vocals"] * std + mean
        others = [v for k, v in sources.items() if k != "vocals"]
        accompaniment = (np.sum(others, axis=0) * std + mean) if others else wav - vocals
        accompaniment = accompaniment.astype(np.float32)
    if std < 1e-8:
        accompaniment = wav.copy()

    out = []
    for x in (vocals, accompaniment):
        x = _fit_length(_resample(x.astype(np.float32), MODEL_RATE, sr), frames)
        x = x.mean(axis=0, keepdims=True) if in_ch == 1 else x
        out.append(np.ascontiguousarray(x.T, dtype=np.float32))
    return out[0], out[1], info


def _run_test_model(name: str, norm: np.ndarray, report, cpu_fallback: bool = True, device: str = "cpu") -> dict:
    if name == "_crash":  # 강제 종료 시험용: 정리 코드 없이 프로세스가 죽는다 (메모리 부족 강제 종료 흉내)
        report(0.5, "시험용 모델이 강제 종료됩니다")
        import os
        import signal

        if hasattr(signal, "SIGKILL"):
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(0xC0000017 & 0xFF)
    if name == "_sleep":  # 취소 시험용: 오래 걸리는 척한다
        import time

        for i in range(600):
            time.sleep(0.05)
            report(i / 600, "시험용 모델 실행 중")
    if name == "_oom" and device != "cpu" and not cpu_fallback:  # GPU 메모리 부족 시험용 (CPU를 고르면 성공)
        raise SeparationError("oom_gpu", OOM_GPU_MESSAGE)
    report(1.0, "시험용 모델 완료")
    if name == "_half":  # 비교 시험용: 보컬 = 입력 절반, 반주 = 나머지 절반
        return {"vocals": norm * 0.5, "other": norm * 0.5}
    return {"vocals": norm.copy(), "other": np.zeros_like(norm)}  # 반주는 0, 보컬은 입력 그대로


def _run_roformer(wav: np.ndarray, files: dict, device: str, report, cpu_fallback: bool = True) -> tuple[np.ndarray, dict]:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise SeparationError("missing_dependency", "PyTorch가 설치되어 있지 않습니다. install_separation.bat을 실행하세요.")
    from amp.engine.dsp import roformer

    try:
        return roformer.run(wav, files["checkpoint"], files.get("config") or None, device, report, cpu_fallback)
    except roformer.RoformerError as e:
        raise SeparationError(e.code, str(e)) from e


def _run_demucs(model_name: str, norm: np.ndarray, device: str, report, cpu_fallback: bool = True) -> tuple[dict, dict]:
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as e:
        raise SeparationError("missing_dependency", f"PyTorch 또는 Demucs가 설치되어 있지 않습니다 ({e.name}).") from e

    actual, note = choose_device(device)
    report(0.03, "Demucs 모델 불러오는 중 (처음에는 약 80MB를 내려받습니다)")
    try:
        model = get_model(model_name)
    except Exception as e:  # noqa: BLE001 - 네트워크·디스크·해시 오류 등 다양하다
        raise SeparationError("model_download", f"Demucs 모델({model_name})을 불러오지 못했습니다: {e}") from e
    model.eval()
    length = norm.shape[-1]
    # 모델 묶음(BagOfModels)은 segment가 없고 안쪽 모델이 가진다
    inner = getattr(model, "models", None)
    seg_s = getattr(model, "segment", None) or (getattr(inner[0], "segment", None) if inner else None)

    def callback(d: dict) -> None:
        if d.get("state") != "end":
            return
        models = max(1, int(d.get("models", 1)))
        seg = float(seg_s) * MODEL_RATE if seg_s else length
        within = min(1.0, (float(d.get("segment_offset", 0)) + seg) / length)
        frac = (int(d.get("model_idx_in_bag", 0)) + within) / models
        report(0.05 + 0.95 * frac, f"보컬 분리 중 ({actual.upper()})")

    x = torch.from_numpy(np.ascontiguousarray(norm))[None]
    info = {"device": actual, "torch": torch.__version__, "note": note, "shift_seed": SHIFT_SEED}
    try:
        import demucs

        info["demucs"] = getattr(demucs, "__version__", "")
    except Exception:  # noqa: BLE001
        pass

    def run(dev: str):
        random.seed(SHIFT_SEED)  # GPU 메모리 부족으로 CPU에서 다시 돌릴 때도 같은 이동을 쓴다
        with torch.no_grad():
            return apply_model(model, x, device=dev, shifts=1, split=True, overlap=0.25, progress=False,
                               callback=callback)[0]

    try:
        out = run(actual)
    except Exception as e:  # noqa: BLE001
        oom = isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()
        if not oom:
            raise
        if actual != "cuda":
            raise SeparationError("oom", "메모리가 부족해 보컬 분리를 마치지 못했습니다. 다른 프로그램을 닫고 다시 시도하세요.") from e
        if not cpu_fallback:
            raise SeparationError("oom_gpu", OOM_GPU_MESSAGE) from e
        torch.cuda.empty_cache()
        info["device"] = "cpu"
        info["note"] = "GPU 메모리가 부족해 CPU로 다시 실행했습니다."
        report(0.05, "GPU 메모리 부족 — CPU로 다시 실행")
        out = run("cpu")
    sources = {name: out[i].cpu().numpy() for i, name in enumerate(model.sources)}
    return sources, info
