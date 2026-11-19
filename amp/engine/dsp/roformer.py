"""Mel-Band RoFormer 보컬 분리 (추론 전용, PyTorch만 사용). 작업 프로세스에서만 import한다.

모델 구조는 ZFTurbo/Music-Source-Separation-Training의 ``models/bs_roformer/mel_band_roformer.py``
(MIT, Copyright (c) 2024 Roman Solovyev; lucidrains/BS-RoFormer, Copyright (c) 2023 Phil Wang 기반)와
rotary-embedding-torch(MIT, Copyright (c) 2021 Phil Wang)에서 추론에 필요한 계산만 옮겼다. 고지 전문은
``LICENSE-roformer.txt``. 모듈·파라미터 이름을 원본과 같게 두어 그 코드로 학습한 체크포인트를 그대로 읽는다.
einops·beartype·rotary-embedding-torch 대신 같은 계산을 torch 연산으로 직접 쓴다(추가 설치 없음).

가중치는 저장소에 넣지 않는다. 사용자가 고른 파일을 ``torch.load(weights_only=True)``로 읽어 텐서만 꺼낸다
(체크포인트는 pickle 형식이라 그냥 읽으면 파일 안의 코드가 실행될 수 있다).

설정(층 수·대역 수 등)은 체크포인트마다 다르다. 순서: 기본값(Kimberley Jensen 보컬 모델 설정) ← 사용자가 고른 .yaml
← 체크포인트 텐서 모양에서 읽은 구조. STFT·샘플레이트는 텐서에서 알 수 없으므로 대역별 입력 크기를 다시 계산해
체크포인트와 맞는지 확인하고, 맞지 않으면 .yaml을 지정하라고 알린다.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, fields
from typing import Callable

import numpy as np

MODEL_NAME = "mel_roformer"
MODEL_RATE = 44100


class RoformerError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class RoformerConfig:
    dim: int = 384
    depth: int = 6
    stereo: bool = True
    num_stems: int = 1
    time_transformer_depth: int = 1
    freq_transformer_depth: int = 1
    num_bands: int = 60
    dim_head: int = 64
    heads: int = 8
    sample_rate: int = 44100
    stft_n_fft: int = 2048
    stft_hop_length: int = 441
    stft_win_length: int = 2048
    stft_normalized: bool = False
    mask_estimator_depth: int = 2
    mlp_expansion_factor: int = 4
    ff_mult: int = 4
    zero_dc: bool = True
    skip_connection: bool = False
    gates: bool = True  # 체크포인트에 to_gates가 있으면 True (옛 버전 호환)
    norm_output: bool = True  # Transformer 마지막 RMSNorm 유무
    # 추론
    chunk_size: int = 352800  # 8초 (학습 때 쓴 길이)
    num_overlap: int = 2
    target: str = "vocals"  # 출력 스템 이름(들). num_stems>1이면 instruments 순서
    instruments: tuple = ("vocals", "other")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["instruments"] = list(self.instruments)
        return d


# 구조를 결정하는 값 (텐서 모양에서 읽을 수 있는 것)
STRUCTURE_KEYS = ("dim", "depth", "num_stems", "time_transformer_depth", "freq_transformer_depth", "num_bands",
                  "dim_head", "heads", "mask_estimator_depth", "mlp_expansion_factor", "ff_mult", "gates", "norm_output")
UNSUPPORTED = {"linear_transformer_depth": 0, "use_pope": False}


# --- 설정 --------------------------------------------------------------------


def read_yaml_config(path: str) -> dict:
    """ZFTurbo 형식 .yaml → RoformerConfig 필드 dict. !!python/tuple 태그도 읽는다(코드는 실행하지 않음)."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=_Loader) or {}
    except (OSError, yaml.YAMLError) as e:
        raise RoformerError("model_config", f"모델 설정 파일을 읽지 못했습니다: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise RoformerError("model_config", "모델 설정 파일에 'model:' 항목이 없습니다. 모델과 함께 받은 .yaml인지 확인하세요.")
    model = data["model"]
    for k, ok in UNSUPPORTED.items():
        if model.get(k, ok) not in (ok, None):
            raise RoformerError("model_config", f"이 설정({k}={model.get(k)})의 모델은 아직 지원하지 않습니다.")
    names = {f.name for f in fields(RoformerConfig)}
    out = {k: v for k, v in model.items() if k in names}
    inference = data.get("inference") or {}
    audio = data.get("audio") or {}
    chunk = inference.get("chunk_size") or audio.get("chunk_size")
    if chunk:
        out["chunk_size"] = int(chunk)
    if inference.get("num_overlap"):
        out["num_overlap"] = int(inference["num_overlap"])
    training = data.get("training") or {}
    if training.get("target_instrument"):
        out["target"] = str(training["target_instrument"])
    if training.get("instruments"):
        out["instruments"] = tuple(str(x) for x in training["instruments"])
    return out


def _indices(sd: dict, pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return sorted({int(m.group(1)) for k in sd for m in [rx.match(k)] if m})


def infer_structure(sd: dict) -> dict:
    """체크포인트 텐서 모양에서 구조 값을 읽는다. Mel-Band RoFormer가 아니면 RoformerError."""
    bands = _indices(sd, r"band_split\.to_features\.(\d+)\.1\.weight$")
    layers = _indices(sd, r"layers\.(\d+)\.")
    if not bands or not layers:
        raise RoformerError("model_file", "Mel-Band RoFormer 모델 파일이 아닙니다 (band_split·layers 가중치가 없습니다).")
    if _indices(sd, r"layers\.0\.(\d+)\.layers") not in ([0, 1],):
        raise RoformerError("model_file", "선형 어텐션 층이 있는 RoFormer 변형은 아직 지원하지 않습니다.")
    w0 = sd["band_split.to_features.0.1.weight"]
    dim = int(w0.shape[0])
    out: dict = {"dim": dim, "num_bands": len(bands), "depth": len(layers)}
    out["time_transformer_depth"] = len(_indices(sd, r"layers\.0\.0\.layers\.(\d+)\.0\.to_qkv\.weight$"))
    out["freq_transformer_depth"] = len(_indices(sd, r"layers\.0\.1\.layers\.(\d+)\.0\.to_qkv\.weight$"))
    qkv = sd["layers.0.0.layers.0.0.to_qkv.weight"]
    gates = sd.get("layers.0.0.layers.0.0.to_gates.weight")
    out["gates"] = gates is not None
    freqs = sd.get("layers.0.0.layers.0.0.rotary_embed.freqs")
    dim_inner = int(qkv.shape[0]) // 3
    if gates is not None:
        out["heads"] = int(gates.shape[0])
        out["dim_head"] = dim_inner // out["heads"]
    elif freqs is not None and freqs.numel() > 0:  # 게이트가 없는 옛 체크포인트: 회전 주파수 개수 = dim_head / 2
        out["dim_head"] = 2 * int(freqs.numel())
        out["heads"] = dim_inner // out["dim_head"]
    out["norm_output"] = "layers.0.0.norm.gamma" in sd
    ff = sd.get("layers.0.0.layers.0.1.net.1.weight")
    if ff is not None:
        out["ff_mult"] = int(ff.shape[0]) // dim
    out["num_stems"] = len(_indices(sd, r"mask_estimators\.(\d+)\.to_freqs\.0\."))
    lin = _indices(sd, r"mask_estimators\.0\.to_freqs\.0\.0\.(\d+)\.weight$")
    out["mask_estimator_depth"] = max(1, len(lin) - 1)
    hid = sd.get("mask_estimators.0.to_freqs.0.0.0.weight")
    if hid is not None:
        out["mlp_expansion_factor"] = int(hid.shape[0]) // dim
    out["_band_dims"] = [int(sd[f"band_split.to_features.{b}.1.weight"].shape[1]) for b in bands]
    return out


def band_layout(cfg: RoformerConfig):
    """(freqs_per_band bool (bands, freqs), freq_indices, num_bands_per_freq, band_dims). 참조 구현과 같은 규칙."""
    import torch
    from librosa import filters

    n_freqs = cfg.stft_n_fft // 2 + 1
    mel = torch.from_numpy(filters.mel(sr=cfg.sample_rate, n_fft=cfg.stft_n_fft, n_mels=cfg.num_bands))
    mel[0][0] = 1.0  # 참조 구현: 첫 주파수가 빠지는 것을 막는다
    mel[-1, -1] = 1.0  # 참조 구현: 환경에 따라 0이 되는 마지막 칸
    fpb = mel > 0
    if not bool(fpb.any(dim=0).all()):
        raise RoformerError("model_config", "멜 대역이 모든 주파수를 덮지 못합니다 (설정 확인 필요).")
    channels = 2 if cfg.stereo else 1
    freq_indices = torch.arange(n_freqs).expand(cfg.num_bands, n_freqs)[fpb]
    if cfg.stereo:
        freq_indices = (freq_indices[:, None] * 2 + torch.arange(2)).reshape(-1)
    nfpb = fpb.sum(dim=1)
    band_dims = tuple(int(2 * f * channels) for f in nfpb.tolist())
    return fpb, freq_indices, fpb.sum(dim=0), band_dims


def resolve_config(sd: dict, yaml_path: str | None = None) -> tuple[RoformerConfig, list[str]]:
    """기본값 ← .yaml ← 체크포인트 구조. (설정, 안내 메시지)."""
    notes: list[str] = []
    base = RoformerConfig().to_dict()
    from_yaml = read_yaml_config(yaml_path) if yaml_path else {}
    base.update(from_yaml)
    inferred = infer_structure(sd)
    band_dims = inferred.pop("_band_dims")
    for k, v in inferred.items():
        if k in from_yaml and from_yaml[k] != v:
            raise RoformerError("model_config", f"설정 파일의 {k}={from_yaml[k]}이(가) 모델 파일({k}={v})과 다릅니다. "
                                                "이 모델과 함께 받은 설정 파일을 고르세요.")
        base[k] = v
    base["instruments"] = tuple(base["instruments"])
    cfg = RoformerConfig(**base)
    # STFT·샘플레이트·채널 확인: 대역별 입력 크기가 체크포인트와 같아야 한다
    stereo_options = [cfg.stereo] if "stereo" in from_yaml else [cfg.stereo, not cfg.stereo]
    for stereo in stereo_options:
        cand = RoformerConfig(**{**cfg.to_dict(), "stereo": stereo, "instruments": cfg.instruments})
        if list(band_layout(cand)[3]) == band_dims:
            if stereo != cfg.stereo:
                notes.append("모노 모델로 확인되어 두 채널 평균으로 분리합니다.")
            return cand, notes
    hint = "" if yaml_path else " 모델과 함께 받은 .yaml 설정 파일을 지정하세요."
    raise RoformerError("model_config", f"모델 파일의 대역 구성이 설정(n_fft {cfg.stft_n_fft}, {cfg.sample_rate}Hz, "
                                        f"대역 {cfg.num_bands}개)과 맞지 않습니다.{hint}")


# --- 체크포인트 ------------------------------------------------------------------


def file_identity(path: str) -> dict:
    """체크포인트를 바꿨는지 빠르게 보는 값 (해시 없이). 분리 결과 재사용 판단에 쓴다."""
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(path: str) -> dict:
    """텐서만 읽는다. Lightning·DataParallel 접두어(model., module.)와 {'state_dict': ...} 감싸기를 푼다."""
    import torch

    if not os.path.isfile(path):
        raise RoformerError("model_file", f"모델 파일이 없습니다: {path}")
    try:
        if path.lower().endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise RoformerError("missing_dependency", ".safetensors 모델을 읽으려면 safetensors 패키지가 필요합니다.") from e
            obj = load_file(path, device="cpu")
        else:
            obj = torch.load(path, map_location="cpu", weights_only=True)
    except RoformerError:
        raise
    except Exception as e:  # noqa: BLE001 - 손상 파일·텐서 외 객체가 든 파일 등
        raise RoformerError("model_file", "모델 파일을 읽지 못했습니다. 텐서만 담긴 PyTorch 체크포인트(.ckpt/.pth)여야 합니다. "
                                          f"({type(e).__name__}: {str(e)[:300]})") from e
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise RoformerError("model_file", "모델 파일에 가중치 목록(state_dict)이 없습니다.")
    sd = {str(k): v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if not sd:
        raise RoformerError("model_file", "모델 파일에 가중치 텐서가 없습니다.")
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


# --- 모델 -----------------------------------------------------------------------


def _build_classes():
    """torch를 쓰는 클래스는 함수 안에서 만든다 (이 모듈을 torch 없이 import해도 되게)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.scale = dim ** 0.5
            self.gamma = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return F.normalize(x, dim=-1) * self.scale * self.gamma

    class RotaryEmbedding(nn.Module):
        """rotary-embedding-torch의 RotaryEmbedding(freqs_for='lang', theta=10000)과 같은 계산. 짝지은 차원(2i, 2i+1)을 돌린다."""

        def __init__(self, dim, theta=10000):
            super().__init__()
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
            self.freqs = nn.Parameter(freqs, requires_grad=False)

        def rotate_queries_or_keys(self, t):
            n = t.shape[-2]
            pos = torch.arange(n, device=t.device, dtype=t.dtype).type(self.freqs.dtype)
            f = (pos[:, None] * self.freqs[None, :]).repeat_interleave(2, dim=-1)  # (n, d): '... n -> ... (n r)'
            pairs = t.unflatten(-1, (-1, 2))
            rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)  # rotate_half
            return (t * f.cos() + rotated * f.sin()).type(t.dtype)

    class Attention(nn.Module):
        def __init__(self, dim, heads, dim_head, rotary_embed, gates):
            super().__init__()
            self.heads = heads
            dim_inner = heads * dim_head
            self.rotary_embed = rotary_embed
            self.norm = RMSNorm(dim)
            self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
            self.to_gates = nn.Linear(dim, heads) if gates else None
            self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False))  # 원본의 Dropout(인덱스 1)은 파라미터 없음

        def forward(self, x):
            x = self.norm(x)
            b, n, _ = x.shape
            q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)  # 'b n (qkv h d) -> qkv b h n d'
            if self.rotary_embed is not None:
                q = self.rotary_embed.rotate_queries_or_keys(q)
                k = self.rotary_embed.rotate_queries_or_keys(k)
            out = F.scaled_dot_product_attention(q, k, v)
            if self.to_gates is not None:
                out = out * self.to_gates(x).permute(0, 2, 1).unsqueeze(-1).sigmoid()  # 'b n h -> b h n 1'
            return self.to_out(out.transpose(1, 2).reshape(b, n, -1))

    class FeedForward(nn.Module):
        def __init__(self, dim, mult):
            super().__init__()
            inner = int(dim * mult)
            # 인덱스를 원본(RMSNorm, Linear, GELU, Dropout, Linear, Dropout)과 맞춘다
            self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Identity(), nn.Linear(inner, dim))

        def forward(self, x):
            return self.net(x)

    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, dim_head, ff_mult, rotary_embed, gates, norm_output):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleList([Attention(dim, heads, dim_head, rotary_embed, gates), FeedForward(dim, ff_mult)])
                for _ in range(depth)
            ])
            self.norm = RMSNorm(dim) if norm_output else nn.Identity()

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return self.norm(x)

    class BandSplit(nn.Module):
        def __init__(self, dim, dim_inputs):
            super().__init__()
            self.dim_inputs = tuple(dim_inputs)
            self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

        def forward(self, x):
            parts = x.split(self.dim_inputs, dim=-1)
            return torch.stack([f(p) for p, f in zip(parts, self.to_features)], dim=-2)

    def mlp(dim_in, dim_out, dim_hidden, depth):
        dims = (dim_in, *((dim_hidden,) * depth), dim_out)
        net = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            net.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                net.append(nn.Tanh())
        return nn.Sequential(*net)

    class MaskEstimator(nn.Module):
        def __init__(self, dim, dim_inputs, depth, expansion):
            super().__init__()
            self.to_freqs = nn.ModuleList([
                nn.Sequential(mlp(dim, d * 2, dim * expansion, depth), nn.GLU(dim=-1)) for d in dim_inputs
            ])

        def forward(self, x):
            return torch.cat([m(band) for band, m in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)

    class MelBandRoformer(nn.Module):
        def __init__(self, cfg: RoformerConfig):
            super().__init__()
            self.cfg = cfg
            self.audio_channels = 2 if cfg.stereo else 1
            time_rot, freq_rot = RotaryEmbedding(cfg.dim_head), RotaryEmbedding(cfg.dim_head)
            kw = dict(dim=cfg.dim, heads=cfg.heads, dim_head=cfg.dim_head, ff_mult=cfg.ff_mult, gates=cfg.gates,
                      norm_output=cfg.norm_output)
            self.layers = nn.ModuleList([
                nn.ModuleList([Transformer(depth=cfg.time_transformer_depth, rotary_embed=time_rot, **kw),
                               Transformer(depth=cfg.freq_transformer_depth, rotary_embed=freq_rot, **kw)])
                for _ in range(cfg.depth)
            ])
            _fpb, freq_indices, bands_per_freq, band_dims = band_layout(cfg)
            self.register_buffer("freq_indices", freq_indices, persistent=False)
            self.register_buffer("num_bands_per_freq", bands_per_freq, persistent=False)
            self.register_buffer("window", torch.hann_window(cfg.stft_win_length), persistent=False)
            self.band_split = BandSplit(cfg.dim, band_dims)
            self.mask_estimators = nn.ModuleList([
                MaskEstimator(cfg.dim, band_dims, cfg.mask_estimator_depth, cfg.mlp_expansion_factor)
                for _ in range(cfg.num_stems)
            ])

        def _stft_kwargs(self):
            c = self.cfg
            return dict(n_fft=c.stft_n_fft, hop_length=c.stft_hop_length, win_length=c.stft_win_length,
                        normalized=c.stft_normalized, window=self.window)

        def forward(self, raw):
            """(b, s, t) → (b, n, s, t)."""
            b, s, t = raw.shape
            spec = torch.stft(raw.reshape(b * s, t), return_complex=True, **self._stft_kwargs())
            n_f, n_t = spec.shape[-2], spec.shape[-1]
            spec = torch.view_as_real(spec).reshape(b, s, n_f, n_t, 2).permute(0, 2, 1, 3, 4).reshape(b, n_f * s, n_t, 2)
            x = spec[:, self.freq_indices]  # (b, f', t, 2)
            x = x.permute(0, 2, 1, 3).reshape(b, n_t, -1)  # 'b f t c -> b t (f c)'
            x = self.band_split(x)  # (b, t, bands, d)
            store = []
            for time_tr, freq_tr in self.layers:
                if self.cfg.skip_connection:
                    for prev in store:
                        x = x + prev
                nb, d = x.shape[2], x.shape[3]
                x = time_tr(x.permute(0, 2, 1, 3).reshape(b * nb, n_t, d)).reshape(b, nb, n_t, d).permute(0, 2, 1, 3)
                x = freq_tr(x.reshape(b * n_t, nb, d)).reshape(b, n_t, nb, d)
                if self.cfg.skip_connection:
                    store.append(x)
            masks = torch.stack([m(x) for m in self.mask_estimators], dim=1)  # (b, n, t, f'*2)
            n = masks.shape[1]
            masks = masks.reshape(b, n, n_t, -1, 2).permute(0, 1, 3, 2, 4)  # 'b n t (f c) -> b n f t c'
            # 겹치는 대역의 마스크 평균 (실수·허수를 따로 더한다)
            idx = self.freq_indices.view(1, 1, -1, 1, 1).expand(b, n, -1, n_t, 2)
            summed = torch.zeros(b, n, n_f * s, n_t, 2, dtype=masks.dtype, device=masks.device).scatter_add_(2, idx, masks)
            denom = self.num_bands_per_freq.repeat_interleave(s).view(1, 1, -1, 1, 1).clamp(min=1e-8)
            avg = torch.view_as_complex((summed / denom).contiguous())
            out = torch.view_as_complex(spec.contiguous()).unsqueeze(1) * avg  # (b, n, f*s, t)
            out = out.reshape(b, n, n_f, s, n_t).permute(0, 1, 3, 2, 4).reshape(b * n * s, n_f, n_t)
            if self.cfg.zero_dc:
                out[:, 0] = 0.0
            audio = torch.istft(out, length=t, **self._stft_kwargs())
            return audio.reshape(b, n, s, t)

    return MelBandRoformer


_CLASSES = None


def build_model(cfg: RoformerConfig):
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = _build_classes()
    return _CLASSES(cfg)


def load_model(checkpoint: str, config_path: str | None = None):
    """(model, cfg, notes). 체크포인트와 구조가 맞지 않으면 RoformerError."""
    sd = load_state_dict(checkpoint)
    cfg, notes = resolve_config(sd, config_path or None)
    model = build_model(cfg)
    expected = model.state_dict()
    missing = [k for k in expected if k not in sd and not k.endswith("rotary_embed.freqs")]
    unexpected = [k for k in sd if k not in expected]
    wrong = [k for k in expected if k in sd and tuple(sd[k].shape) != tuple(expected[k].shape)]
    if missing or unexpected or wrong:
        parts = []
        for label, keys in (("빠진 가중치", missing), ("모르는 가중치", unexpected), ("모양이 다른 가중치", wrong)):
            if keys:
                parts.append(f"{label} {len(keys)}개 (예: {', '.join(keys[:3])})")
        raise RoformerError("model_config", "모델 파일이 Mel-Band RoFormer 구조와 맞지 않습니다: " + "; ".join(parts))
    model.load_state_dict({k: v for k, v in sd.items() if k in expected}, strict=False)
    model.eval()
    return model, cfg, notes


# --- 조각 나눠 추론 ----------------------------------------------------------------


def chunk_window(size: int, fade: int):
    import torch

    w = torch.ones(size)
    if fade > 0:
        w[:fade] = torch.linspace(0, 1, fade)
        w[-fade:] = torch.linspace(1, 0, fade)
    return w


def demix(model_fn: Callable, mix: np.ndarray, chunk_size: int, num_overlap: int,
          report: Callable[[float, str], None] | None = None, label: str = "") -> np.ndarray:
    """(channels, frames) → (stems, channels, frames). 조각을 겹쳐 돌리고 창 가중 평균으로 잇는다.

    양 끝은 반사 패딩으로 문맥을 붙이고 끝에서 잘라낸다. 창 합으로 나누므로 모델이 입력을 그대로 내면
    출력도 입력과 같다(시간 어긋남 0).
    """
    import torch
    import torch.nn.functional as F

    report = report or (lambda f, m: None)
    x = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))
    n = x.shape[-1]
    c = int(chunk_size)
    step = max(1, c // max(1, int(num_overlap)))
    fade = c // 10
    border = c - step
    padded = n > 2 * border and border > 0
    if padded:
        x = F.pad(x[None], (border, border), mode="reflect")[0]
    total = x.shape[-1]
    base = chunk_window(c, fade)
    result = None
    counter = torch.zeros(total)
    starts = list(range(0, total, step))
    for k, i in enumerate(starts):
        part = x[:, i:i + c]
        length = part.shape[-1]
        if length < c:
            mode = "reflect" if length > c // 2 + 1 else "constant"
            part = F.pad(part[None], (0, c - length), mode=mode)[0]
        out = model_fn(part)  # (stems, channels, c)
        if result is None:
            result = torch.zeros(out.shape[0], out.shape[1], total)
        w = base.clone()
        if i == 0:
            w[:fade] = 1
        if i + step >= total:
            w[-fade:] = 1
        result[..., i:i + length] += out[..., :length] * w[:length]
        counter[i:i + length] += w[:length]
        report((k + 1) / len(starts), label)
    est = result / counter.clamp(min=1e-8)
    if padded:
        est = est[..., border:total - border]
    return est.numpy()


def run(wav: np.ndarray, checkpoint: str, config_path: str | None, device: str,
        report: Callable[[float, str], None]) -> tuple[np.ndarray, dict]:
    """44.1kHz (2, frames) → (보컬 (2, frames), 정보). 반주는 호출한 쪽에서 원본 − 보컬로 만든다."""
    import torch

    from amp.engine.dsp.separation import choose_device

    actual, note = choose_device(device)
    report(0.02, "RoFormer 모델 불러오는 중")
    model, cfg, notes = load_model(checkpoint, config_path)
    if cfg.sample_rate != MODEL_RATE:
        raise RoformerError("model_config", f"{cfg.sample_rate}Hz 모델은 아직 지원하지 않습니다 (44100Hz만).")
    names = list(cfg.instruments) if cfg.num_stems > 1 else [cfg.target]
    info = {"device": actual, "torch": torch.__version__, "note": " ".join([note, *notes]).strip(),
            "checkpoint": os.path.basename(checkpoint), "checkpoint_identity": file_identity(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint), "config": os.path.basename(config_path) if config_path else "",
            "roformer": {k: v for k, v in cfg.to_dict().items() if k not in ("instruments",)},
            "outputs": names, "params": int(sum(p.numel() for p in model.parameters()))}
    channels = 2 if cfg.stereo else 1
    mix = wav if channels == 2 else wav.mean(axis=0, keepdims=True)

    def go(dev: str) -> np.ndarray:
        m = model.to(dev)

        def fn(part):
            with torch.inference_mode():
                return m(part[None].to(dev))[0].float().cpu()

        return demix(fn, mix, cfg.chunk_size, cfg.num_overlap,
                     lambda f, msg: report(0.05 + 0.95 * f, msg), f"보컬 분리 중 (RoFormer · {dev.upper()})")

    try:
        est = go(actual)
    except Exception as e:  # noqa: BLE001
        oom = isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()
        if not oom:
            raise
        if actual != "cuda":
            raise RoformerError("oom", "메모리가 부족해 보컬 분리를 마치지 못했습니다. 다른 프로그램을 닫고 다시 시도하세요.") from e
        torch.cuda.empty_cache()
        info["device"] = "cpu"
        info["note"] = "GPU 메모리가 부족해 CPU로 다시 실행했습니다."
        report(0.05, "GPU 메모리 부족 — CPU로 다시 실행")
        est = go("cpu")
    if "vocals" in names:
        vocals = est[names.index("vocals")]
    elif len(names) == 1:  # 반주(또는 다른 악기)를 내는 모델: 보컬 = 원본 − 출력
        vocals = mix - est[0]
        info["vocals_from"] = f"원본 − {names[0]}"
    else:
        raise RoformerError("model_config", f"이 모델의 출력({', '.join(names)})에 보컬이 없습니다.")
    if channels == 1:
        vocals = np.repeat(vocals, 2, axis=0)
    return vocals.astype(np.float32), info
