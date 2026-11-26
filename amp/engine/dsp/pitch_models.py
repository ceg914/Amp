"""외부 단성 피치 모델 (v0.7.0 Feature A): CREPE(full·tiny), FCPE. 작업 프로세스 안에서만 불러온다 (torch).

신경망 계산은 torchcrepe 0.0.24와 torchfcpe 0.0.4(둘 다 MIT, amp/engine/dsp/LICENSE-pitch-models.txt)의 추론 부분을
torch만으로 옮겼다. torchaudio·resampy·einops·local-attention이 필요 없다. 가중치는 사용자가 install_pitch로 설치한
그 패키지 안의 파일을 경로로 읽고(패키지를 import하지 않음), torch.load(weights_only=True)로만 연다.

공통 출력 격자: 16kHz, 홉 160 = 100fps, 프레임 i의 중심 = i × 10ms (pYIN과 같다).
- CREPE: 프레임 i = 중심 i×160의 1024샘플 창 (앞뒤 512 0 채움). 360칸(20cent) 활성값 → 좁은 띠 비터비 경로 →
  경로 칸 주변 ±4칸 가중 평균. 신뢰도 = 경로 칸의 활성값(시그모이드, 0~1)
- FCPE: 128칸 멜(창 1024, 홉 160, 앞 432 반사 채움, center=False) → 합성곱 6층 → 360칸 → 최댓값 주변 ±4칸 가중 평균.
  FCPE 프레임 i의 중심은 i×160 + 80샘플이라 공통 격자로 반 프레임 옮긴다(센트 선형 보간). 신뢰도 = 최댓값(0~1)
- 신뢰도는 모델마다 뜻이 달라 서로 같은 확률로 보지 않는다 (confidence_kind로 구분)
- 무성: 신뢰도가 기준보다 낮거나, 창 음량이 -60dBFS보다 작거나, 주파수가 fmin~fmax 밖
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

PITCH_SR = 16000
HOP = 160
FRAME_RATE = PITCH_SR / HOP
SILENCE_DB = -60.0
MODELS = {
    "crepe_full": {"package": "torchcrepe", "file": "full.pth", "label": "CREPE full", "confidence": "crepe_activation",
                   "threshold": 0.5},
    "crepe_tiny": {"package": "torchcrepe", "file": "tiny.pth", "label": "CREPE tiny", "confidence": "crepe_activation",
                   "threshold": 0.5},
    "fcpe": {"package": "torchfcpe", "file": "fcpe_c_v001.pt", "label": "FCPE", "confidence": "fcpe_latent_max",
             "threshold": 0.006},  # torchfcpe 기본값
}
CODE_VERSION = 1
CREPE_CENTS0 = 1997.3794084376191  # 0번 칸 (torchcrepe.convert)


class PitchModelError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def weights_path(model_id: str, find_spec=importlib.util.find_spec) -> Path:
    spec_ = MODELS.get(model_id)
    if spec_ is None:
        raise PitchModelError("internal", f"알 수 없는 피치 모델: {model_id}")
    try:
        spec = find_spec(spec_["package"])
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        from amp.core.install_hints import script

        raise PitchModelError("missing_pitch_model",
                              f"{spec_['label']} 가중치 패키지({spec_['package']})가 설치되어 있지 않습니다. "
                              f"{script('pitch')}을 실행하세요.")
    p = Path(list(spec.submodule_search_locations)[0]) / "assets" / spec_["file"]
    if not p.is_file():
        raise PitchModelError("missing_pitch_model", f"{spec_['package']} 패키지에 가중치 파일({spec_['file']})이 없습니다.")
    return p


# --- CREPE -------------------------------------------------------------------------------------

def _crepe_module(size: str):
    import functools

    import torch
    import torch.nn.functional as F

    class Crepe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if size == "full":
                inc, outc, self.in_features = [1, 1024, 128, 128, 128, 256], [1024, 128, 128, 128, 256, 512], 2048
            else:
                inc, outc, self.in_features = [1, 128, 16, 16, 16, 32], [128, 16, 16, 16, 32, 64], 256
            ks = [(512, 1)] + 5 * [(64, 1)]
            st = [(4, 1)] + 5 * [(1, 1)]
            bn = functools.partial(torch.nn.BatchNorm2d, eps=0.0010000000474974513, momentum=0.0)
            for i in range(6):
                setattr(self, f"conv{i + 1}", torch.nn.Conv2d(inc[i], outc[i], ks[i], st[i]))
                setattr(self, f"conv{i + 1}_BN", bn(outc[i]))
            self.classifier = torch.nn.Linear(self.in_features, 360)

        @staticmethod
        def _layer(x, conv, norm, pad=(0, 0, 31, 32)):
            return F.max_pool2d(norm(F.relu(conv(F.pad(x, pad)))), (2, 1), (2, 1))

        def forward(self, x):
            x = x[:, None, :, None]
            x = self._layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
            for i in range(2, 7):
                x = self._layer(x, getattr(self, f"conv{i}"), getattr(self, f"conv{i}_BN"))
            x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)
            return torch.sigmoid(self.classifier(x))

    return Crepe()


def crepe_frames(y16: np.ndarray) -> np.ndarray:
    """프레임 i = 중심 i×160의 1024샘플 (앞뒤 512 0 채움), 프레임마다 평균 0·표준편차 1로 (torchcrepe와 같음)."""
    n = 1 + len(y16) // HOP
    yp = np.pad(np.asarray(y16, dtype=np.float32), (512, 512))
    fr = np.lib.stride_tricks.sliding_window_view(yp, 1024)[::HOP][:n].astype(np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr /= np.maximum(1e-10, fr.std(axis=1, keepdims=True))
    return fr


def crepe_bins_to_hz(bins: np.ndarray) -> np.ndarray:
    return 10.0 * 2.0 ** ((20.0 * np.asarray(bins, dtype=np.float64) + CREPE_CENTS0) / 1200.0)


def banded_viterbi(act: np.ndarray, width: int = 12) -> np.ndarray:
    """활성값 (T, 360) → 칸 경로. 전이는 |i-j| < width에서 삼각형 (torchcrepe와 같은 모양), 그 밖은 막는다."""
    T, S = act.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    emit = np.log(act / np.maximum(act.sum(axis=1, keepdims=True), 1e-12) + 1e-12)
    offs = np.arange(-(width - 1), width)
    tri = (width - np.abs(offs)).astype(np.float64)
    # 행마다 정규화 (가장자리는 범위 안 이웃만)
    rows = np.arange(S)[:, None] + offs[None, :]
    valid = (rows >= 0) & (rows < S)
    w = np.where(valid, tri[None, :], 0.0)
    logt = np.log(np.where(valid, w / w.sum(axis=1, keepdims=True), 1.0))
    logt[~valid] = -np.inf
    src = np.clip(rows, 0, S - 1)  # 상태 j로 오는 이전 상태 i = j + off (대칭)
    score = emit[0].copy()
    back = np.zeros((T, S), dtype=np.int16)
    for t in range(1, T):
        cand = score[src] + logt  # (S, 2w-1)
        k = np.argmax(cand, axis=1)
        back[t] = k
        score = cand[np.arange(S), k] + emit[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = src[path[t], back[t, path[t]]]
    return path


def local_average_bins(act: np.ndarray, center: np.ndarray, radius: int = 4) -> np.ndarray:
    T, S = act.shape
    idx = center[:, None] + np.arange(-radius, radius + 1)[None, :]
    ok = (idx >= 0) & (idx < S)
    idx = np.clip(idx, 0, S - 1)
    w = np.take_along_axis(act, idx, axis=1) * ok
    return (w * idx).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)


# --- FCPE --------------------------------------------------------------------------------------

def _fcpe_module(cfg: dict):
    import torch
    from torch import nn

    hidden = int(cfg.get("hidden_dims", 512))
    n_layers = int(cfg.get("n_layers", 6))
    n_mels = 128
    out_dims = int(cfg.get("out_dims", 360))
    if not cfg.get("conv_only", False):
        raise PitchModelError("pitch_model_file", "이 FCPE 체크포인트는 attention 층을 써서 지원하지 않습니다 (conv_only 모델만).")

    class ConvModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            inner = hidden * 2
            self.norm = nn.LayerNorm(hidden)
            self.pw1 = nn.Conv1d(hidden, inner * 2, 1)
            self.dw = nn.Conv1d(inner, inner, 31, padding=15, groups=inner)
            self.pw2 = nn.Conv1d(inner, hidden, 1)

        def forward(self, x):  # (B, T, C)
            y = self.norm(x).transpose(1, 2)
            y = nn.functional.glu(self.pw1(y), dim=1)
            y = self.pw2(nn.functional.silu(self.dw(y)))
            return y.transpose(1, 2)

    class Fcpe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in0 = nn.Conv1d(n_mels, hidden, 3, 1, 1)
            self.gn = nn.GroupNorm(4, hidden)
            self.in1 = nn.Conv1d(hidden, hidden, 3, 1, 1)
            self.layers = nn.ModuleList([ConvModule() for _ in range(n_layers)])
            self.norm = nn.LayerNorm(hidden)
            self.proj = nn.Linear(hidden, out_dims)

        def forward(self, mel):  # (B, T, 128)
            x = self.in1(nn.functional.leaky_relu(self.gn(self.in0(mel.transpose(1, 2))))).transpose(1, 2)
            for layer in self.layers:
                x = x + layer(x)
            return torch.sigmoid(self.proj(self.norm(x)))

    return Fcpe()


def _fcpe_state(sd: dict) -> tuple[dict, object]:
    """torchfcpe의 이름 → 옮긴 구조의 이름. weight_norm(g·v)은 실제 가중치로 합친다."""
    import torch

    out = {"in0.weight": sd["input_stack.0.weight"], "in0.bias": sd["input_stack.0.bias"],
           "gn.weight": sd["input_stack.1.weight"], "gn.bias": sd["input_stack.1.bias"],
           "in1.weight": sd["input_stack.3.weight"], "in1.bias": sd["input_stack.3.bias"],
           "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"], "proj.bias": sd["output_proj.bias"]}
    if "output_proj.weight_v" in sd:
        v, g = sd["output_proj.weight_v"], sd["output_proj.weight_g"]
        out["proj.weight"] = v * (g / torch.linalg.vector_norm(v, dim=1, keepdim=True))
    elif "output_proj.parametrizations.weight.original1" in sd:
        g = sd["output_proj.parametrizations.weight.original0"]
        v = sd["output_proj.parametrizations.weight.original1"]
        out["proj.weight"] = v * (g / torch.linalg.vector_norm(v, dim=1, keepdim=True))
    else:
        out["proj.weight"] = sd["output_proj.weight"]
    i = 0
    while f"net.encoder_layers.{i}.conformer.net.0.weight" in sd:
        p = f"net.encoder_layers.{i}.conformer.net."
        out.update({f"layers.{i}.norm.weight": sd[p + "0.weight"], f"layers.{i}.norm.bias": sd[p + "0.bias"],
                    f"layers.{i}.pw1.weight": sd[p + "2.weight"], f"layers.{i}.pw1.bias": sd[p + "2.bias"],
                    f"layers.{i}.dw.weight": sd[p + "4.conv.weight"], f"layers.{i}.dw.bias": sd[p + "4.conv.bias"],
                    f"layers.{i}.pw2.weight": sd[p + "6.weight"], f"layers.{i}.pw2.bias": sd[p + "6.bias"]})
        i += 1
    return out, sd.get("cent_table")


def fcpe_mel(y16: np.ndarray):
    """torchfcpe MelModule(key_shift 0)과 같은 계산: (T, 128) log 멜, T = len//160 + 1."""
    import librosa
    import torch

    y = torch.from_numpy(np.asarray(y16, dtype=np.float32))
    n = 1 + len(y16) // HOP
    pad_l = (1024 - HOP) // 2
    pad_r = max((1024 - HOP + 1) // 2, 1024 - y.shape[0] - pad_l)
    mode = "reflect" if pad_r < y.shape[0] else "constant"
    yp = torch.nn.functional.pad(y[None, None, :], (pad_l, pad_r), mode=mode)[0]
    spec = torch.stft(yp, 1024, hop_length=HOP, win_length=1024, window=torch.hann_window(1024), center=False,
                      normalized=False, onesided=True, return_complex=True)
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)[0]
    basis = torch.from_numpy(librosa.filters.mel(sr=PITCH_SR, n_fft=1024, n_mels=128, fmin=0, fmax=8000)).float()
    mel = torch.log(torch.clamp(basis @ mag, min=1e-5)).T  # (T', 128)
    if mel.shape[0] < n:
        mel = torch.cat([mel, mel[-1:].expand(n - mel.shape[0], -1)], 0)
    return mel[:n]


def shift_half_frame(f0: np.ndarray, conf: np.ndarray, voiced: np.ndarray):
    """중심이 (i + 0.5) 프레임인 값을 i 프레임 격자로: 이웃 두 프레임이 다 유성이면 센트 평균, 아니면 앞 프레임."""
    n = len(f0)
    if n < 2:
        return f0, conf, voiced
    prev_f, cur_f = np.r_[f0[:1], f0[:-1]], f0
    prev_v, cur_v = np.r_[voiced[:1], voiced[:-1]], voiced
    both = prev_v & cur_v
    out = np.where(both, np.exp2((np.log2(np.where(both, prev_f, 1.0)) + np.log2(np.where(both, cur_f, 1.0))) / 2),
                   np.where(prev_v, prev_f, cur_f))
    v = prev_v | cur_v
    v[0] = voiced[0]
    out[0] = f0[0]
    c = np.maximum(np.r_[conf[:1], conf[:-1]], conf) if conf.size else conf
    return np.where(v, out, np.nan), c, v


# --- 실행 --------------------------------------------------------------------------------------

def frame_db(y16: np.ndarray, n: int) -> np.ndarray:
    """프레임 i 중심 1024샘플 창의 RMS (dBFS)."""
    yp = np.pad(np.asarray(y16, dtype=np.float32), (512, 512))
    win = np.lib.stride_tricks.sliding_window_view(yp, 1024)[::HOP][:n]
    rms = np.sqrt(np.mean(win.astype(np.float64) ** 2, axis=1))
    out = np.full(n, -200.0)
    out[: len(rms)] = 20 * np.log10(np.maximum(rms, 1e-10))
    return out


def _pick_device(device: str) -> str:
    import torch

    if device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if device == "cuda":
        raise PitchModelError("no_cuda", "CUDA GPU를 쓸 수 없습니다. 처리 장치를 '자동'이나 'CPU'로 바꾸세요.")
    return "cpu"


def run(model_id: str, y16: np.ndarray, device: str = "auto", fmin: float = 55.0, fmax: float = 1400.0,
        threshold: float | None = None, progress: Callable[[float, str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None, batch: int | None = None,
        find_spec=importlib.util.find_spec) -> dict:
    """16kHz 모노 → {"f0_hz", "confidence", "voiced", "info"}. 프레임 수 = len//160 + 1 (pYIN과 같음).
    GPU 메모리가 모자라면 한 번 작은 묶음으로 다시 하고, 그래도 안 되면 CPU로 바꿔 계속한다(info에 적음)."""
    import torch

    report = progress or (lambda f, m: None)
    spec_ = MODELS[model_id]
    path = weights_path(model_id, find_spec)
    dev = _pick_device(device)
    t0 = time.perf_counter()
    n = 1 + len(y16) // HOP
    info = {"model_id": model_id, "model_name": spec_["label"], "code_version": CODE_VERSION,
            "weights_file": path.name, "weights_size": path.stat().st_size, "device_requested": device,
            "confidence_kind": spec_["confidence"], "frame_rate": FRAME_RATE, "warnings": []}
    try:
        from amp.engine.model_store import file_sha256

        info["weights_sha256"] = file_sha256(str(path))
    except OSError:
        info["weights_sha256"] = ""
    thr = float(spec_["threshold"] if threshold is None else threshold)
    attempts = [(dev, batch)]
    if dev == "cuda":
        attempts += [("cuda", max(16, (batch or 256) // 4)), ("cpu", None)]
    last_err = None
    for k, (d, b) in enumerate(attempts):
        try:
            if d == "cuda":
                torch.cuda.reset_peak_memory_stats()
            report(0.02, f"{spec_['label']} 불러오는 중 ({d.upper()})")
            if model_id.startswith("crepe"):
                f0, conf, voiced = _run_crepe(model_id, path, y16, d, thr, b, report, is_cancelled)
            else:
                f0, conf, voiced = _run_fcpe(path, y16, d, thr, report, is_cancelled)
            info["device"] = d
            if d == "cuda":
                info["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            break
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            last_err = e
            torch.cuda.empty_cache()
            nxt = attempts[k + 1] if k + 1 < len(attempts) else None
            if nxt is None:
                raise PitchModelError("oom", "GPU 메모리가 부족합니다.") from e
            info["warnings"].append("GPU 메모리가 부족해 " + ("작은 묶음으로 다시 계산했습니다." if nxt[0] == "cuda"
                                                         else "CPU로 바꿔 계산했습니다."))
    else:  # pragma: no cover - 위에서 끝난다
        raise PitchModelError("internal", str(last_err))
    db = frame_db(y16, n)
    voiced = voiced & (db > SILENCE_DB) & np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax)
    f0 = np.where(voiced, f0, np.nan).astype(np.float32)
    info.update(elapsed_s=round(time.perf_counter() - t0, 3), threshold=thr, fmin=fmin, fmax=fmax,
                silence_db=SILENCE_DB, frames=int(n), voiced_frames=int(voiced.sum()))
    if dev == "cuda":
        torch.cuda.empty_cache()
    report(1.0, f"{spec_['label']} 완료")
    return {"f0_hz": f0, "confidence": conf.astype(np.float32), "voiced": voiced, "info": info}


def _cancel_check(is_cancelled) -> None:
    if is_cancelled and is_cancelled():
        from amp.core.errors import AnalysisCancelled

        raise AnalysisCancelled("피치 분석을 취소했습니다.")


def _run_crepe(model_id, path, y16, dev, thr, batch, report, is_cancelled):
    import torch

    model = _crepe_module("full" if model_id == "crepe_full" else "tiny")
    model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    model.eval().to(dev)
    frames = crepe_frames(y16)
    n = len(frames)
    bs = batch or (512 if dev == "cuda" else 128)
    acts = np.zeros((n, 360), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, n, bs):
            _cancel_check(is_cancelled)
            x = torch.from_numpy(frames[i:i + bs]).to(dev)
            acts[i:i + bs] = model(x).float().cpu().numpy()
            report(0.05 + 0.85 * min(1.0, (i + bs) / max(n, 1)), f"{MODELS[model_id]['label']} 계산 중")
    del model
    report(0.92, "경로 찾는 중")
    path_bins = banded_viterbi(acts)
    conf = acts[np.arange(n), path_bins]
    bins = local_average_bins(acts, path_bins)
    f0 = crepe_bins_to_hz(bins)
    return f0, conf, conf >= thr


def _run_fcpe(path, y16, dev, thr, report, is_cancelled):
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    cfg = dict((ck.get("config_dict") or {}).get("model") or {})
    mel_cfg = dict((ck.get("config_dict") or {}).get("mel") or {})
    if mel_cfg and (int(mel_cfg.get("sr", PITCH_SR)) != PITCH_SR or int(mel_cfg.get("hop_size", HOP)) != HOP
                    or int(mel_cfg.get("num_mels", 128)) != 128):
        raise PitchModelError("pitch_model_file", "FCPE 체크포인트의 멜 설정이 지원하는 값(16kHz·홉 160·128칸)과 다릅니다.")
    model = _fcpe_module(cfg)
    state, cent_table = _fcpe_state(ck["model"])
    model.load_state_dict(state)
    model.eval().to(dev)
    if cent_table is None:
        lo = 1200 * math.log2(float(cfg.get("f0_min", 32.7)) / 10)
        hi = 1200 * math.log2(float(cfg.get("f0_max", 1975.5)) / 10)
        cent_table = torch.linspace(lo, hi, int(cfg.get("out_dims", 360)))
    cents_tab = cent_table.numpy().astype(np.float64)
    mel = fcpe_mel(y16)
    n = mel.shape[0]
    chunk, ctx = 3000, 200  # 30초씩, 앞뒤 2초 문맥 (받아들이는 범위 약 ±0.9초)
    lat = np.zeros((n, len(cents_tab)), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, chunk):
            _cancel_check(is_cancelled)
            a, b = max(0, s - ctx), min(n, s + chunk + ctx)
            out = model(mel[a:b][None].to(dev))[0].float().cpu().numpy()
            lat[s:min(n, s + chunk)] = out[s - a: s - a + min(chunk, n - s)]
            report(0.05 + 0.85 * min(1.0, (s + chunk) / max(n, 1)), "FCPE 계산 중")
    del model
    top = np.argmax(lat, axis=1)
    conf = lat[np.arange(n), top]
    idx = local_average_bins(lat, top)
    cents = np.interp(idx, np.arange(len(cents_tab)), cents_tab)
    f0 = 10.0 * 2.0 ** (cents / 1200.0)
    voiced = conf > thr
    return shift_half_frame(f0, conf, voiced)
