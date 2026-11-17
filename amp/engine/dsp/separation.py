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
from typing import Callable

import numpy as np

DEMUCS_MODELS = ("htdemucs",)
TEST_MODELS = ("_identity", "_sleep")
MODEL_RATE = 44100


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


def choose_device(requested: str) -> tuple[str, str]:
    """(실제 장치, 안내 메시지)."""
    import torch

    has_cuda = torch.cuda.is_available()
    if requested == "cuda" and not has_cuda:
        return "cpu", "GPU(CUDA)를 쓸 수 없어 CPU로 실행합니다. CUDA판 PyTorch가 설치되어 있는지 확인하세요."
    if requested in ("auto", "cuda") and has_cuda:
        return "cuda", ""
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
) -> tuple[np.ndarray, np.ndarray, dict]:
    """(frames, channels) → (vocals, accompaniment, info). 출력은 입력과 같은 sr·프레임 수다."""
    report = progress or (lambda f, m: None)
    if model_name not in DEMUCS_MODELS and model_name not in TEST_MODELS:
        raise SeparationError("internal", f"알 수 없는 분리 모델: {model_name}")
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
    else:
        norm = (wav - mean) / std
        if model_name in TEST_MODELS:
            sources = _run_test_model(model_name, norm, report)
            info["device"] = "cpu"
        else:
            sources, dev_info = _run_demucs(model_name, norm, device, report)
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


def _run_test_model(name: str, norm: np.ndarray, report) -> dict:
    if name == "_sleep":  # 취소 시험용: 오래 걸리는 척한다
        import time

        for i in range(600):
            time.sleep(0.05)
            report(i / 600, "시험용 모델 실행 중")
    report(1.0, "시험용 모델 완료")
    return {"vocals": norm.copy(), "other": np.zeros_like(norm)}  # 반주는 0, 보컬은 입력 그대로


def _run_demucs(model_name: str, norm: np.ndarray, device: str, report) -> tuple[dict, dict]:
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
    info = {"device": actual, "torch": torch.__version__, "note": note}
    try:
        import demucs

        info["demucs"] = getattr(demucs, "__version__", "")
    except Exception:  # noqa: BLE001
        pass

    def run(dev: str):
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
        torch.cuda.empty_cache()
        info["device"] = "cpu"
        info["note"] = "GPU 메모리가 부족해 CPU로 다시 실행했습니다."
        report(0.05, "GPU 메모리 부족 — CPU로 다시 실행")
        out = run("cpu")
    sources = {name: out[i].cpu().numpy() for i, name in enumerate(model.sources)}
    return sources, info
