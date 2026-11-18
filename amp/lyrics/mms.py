"""MMS 다국어 강제 정렬 모델 (wav2vec2 large, 약 3억 매개변수, 1,130개 언어 31K시간 학습).

torchaudio의 ``MMS_FA`` 와 같은 구조를 PyTorch만으로 만들고 같은 공개 가중치를 읽는다.
torchaudio 2.9 이후로 C++ 확장이 torch 버전에 묶여 있어(사용자 PC torch 2.15 + torchaudio 2.11 조합은 보장되지 않음)
의존하지 않는다. 구조·정규화·출력 순서는 torchaudio 2.11.1 소스와 같고, 개발 중 출력이 일치함을 대조했다.

가중치 라이선스: CC-BY-NC 4.0 (비상업적 이용만 허용). 앱 안내와 보고서에 적는다.
입력: 16kHz 모노. 출력: 프레임(20ms)마다 [blank, a, i, e, …, x, *] 로그 확률. '*'는 아무 소리나 받는 와일드카드다.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
MODEL_FILE = "mms_fa_ctc_alignment_mling_uroman.pt"
MODEL_SIZE_BYTES = 1_262_047_414
MODEL_NAME = "MMS_FA (wav2vec2 CTC, uroman)"
LICENSE = "CC-BY-NC 4.0 (비상업적 이용)"
SAMPLE_RATE = 16000
HOP = 320  # 합성곱 보폭 곱 = 20ms
LABELS = ("-", "a", "i", "e", "n", "o", "u", "t", "s", "r", "m", "k", "l", "d", "g", "h", "y", "b", "p", "w", "c",
          "v", "j", "z", "f", "'", "q", "x", "*")
BLANK = 0
STAR = len(LABELS) - 1
LABEL_INDEX = {c: i for i, c in enumerate(LABELS)}
_REMOVE_AXES = (1, 2, 3)  # fairseq 사전의 쓰지 않는 자리 (torchaudio와 같게 뺀다)
_CONV = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
CHUNK_S = 20.0  # 한 번에 넣는 길이 (자기 주의 메모리 제한)
CONTEXT_S = 1.5  # 조각 양 끝에 덧붙여 계산하고 버리는 여유


def build_model():
    import torch
    from torch import nn

    class _ChanLayerNorm(nn.LayerNorm):
        def forward(self, x):
            return nn.functional.layer_norm(x.transpose(-2, -1), self.normalized_shape, self.weight, self.bias,
                                            self.eps).transpose(-2, -1)

    class _ConvBlock(nn.Module):
        def __init__(self, cin, cout, k, s):
            super().__init__()
            self.conv = nn.Conv1d(cin, cout, k, s, bias=True)
            self.layer_norm = _ChanLayerNorm(cout, elementwise_affine=True)

        def forward(self, x):
            return nn.functional.gelu(self.layer_norm(self.conv(x)))

    class _FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            blocks, cin = [], 1
            for cout, k, s in _CONV:
                blocks.append(_ConvBlock(cin, cout, k, s))
                cin = cout
            self.conv_layers = nn.ModuleList(blocks)

        def forward(self, x):
            x = x.unsqueeze(1)
            for b in self.conv_layers:
                x = b(x)
            return x.transpose(1, 2)

    class _Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = nn.LayerNorm(512)
            self.projection = nn.Linear(512, 1024)

        def forward(self, x):
            return self.projection(self.layer_norm(x))

    class _PosConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1024, 1024, 128, padding=64, groups=16)

        def forward(self, x):
            x = self.conv(x.transpose(-2, -1))[..., :-1]  # 짝수 커널이라 마지막 프레임 하나를 버린다
            return nn.functional.gelu(x).transpose(-2, -1)

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(1024, 1024)
            self.v_proj = nn.Linear(1024, 1024)
            self.q_proj = nn.Linear(1024, 1024)
            self.out_proj = nn.Linear(1024, 1024)

        def forward(self, x):
            b, n, _ = x.shape
            shape = (b, n, 16, 64)
            q = self.q_proj(x).view(*shape).transpose(2, 1)
            k = self.k_proj(x).view(*shape).transpose(2, 1)
            v = self.v_proj(x).view(*shape).transpose(2, 1)
            out = nn.functional.scaled_dot_product_attention(q, k, v)
            return self.out_proj(out.transpose(1, 2).reshape(b, n, 1024))

    class _FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.intermediate_dense = nn.Linear(1024, 4096)
            self.output_dense = nn.Linear(4096, 1024)

        def forward(self, x):
            return self.output_dense(nn.functional.gelu(self.intermediate_dense(x)))

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _Attention()
            self.layer_norm = nn.LayerNorm(1024)
            self.feed_forward = _FeedForward()
            self.final_layer_norm = nn.LayerNorm(1024)

        def forward(self, x):  # 정규화 먼저 (layer_norm_first)
            x = x + self.attention(self.layer_norm(x))
            return x + self.feed_forward(self.final_layer_norm(x))

    class _Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_conv_embed = _PosConv()
            self.layer_norm = nn.LayerNorm(1024)
            self.layers = nn.ModuleList(_Layer() for _ in range(24))

        def forward(self, x):
            x = x + self.pos_conv_embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.layer_norm(x)

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = _Projection()
            self.transformer = _Transformer()

        def forward(self, x):
            return self.transformer(self.feature_projection(x))

    class MmsAligner(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = _FeatureExtractor()
            self.encoder = _Encoder()
            self.aux = nn.Linear(1024, len(LABELS) - 1)

        def forward(self, wave):  # (batch, samples) → (batch, frames, labels) 로그 확률, 마지막 열은 '*'
            wave = nn.functional.layer_norm(wave, wave.shape)
            x = self.aux(self.encoder(self.feature_extractor(wave)))
            x = torch.log_softmax(x, dim=-1)
            star = torch.zeros(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)
            return torch.cat((x, star), dim=-1)

    return MmsAligner()


def convert_state_dict(sd: dict) -> dict:
    """공개 체크포인트(torchaudio 이름) → 이 모듈 이름. weight_norm(g, v)는 합쳐 일반 가중치로 만든다."""
    import torch

    sd = dict(sd)
    for key in ("aux.weight", "aux.bias"):
        mat = sd[key]
        sd[key] = torch.stack([mat[i] for i in range(mat.size(0)) if i not in _REMOVE_AXES])
    pre = "encoder.transformer.pos_conv_embed.conv."
    if pre + "weight_g" in sd:
        g, v = sd.pop(pre + "weight_g"), sd.pop(pre + "weight_v")
        norm = v.norm(dim=(0, 1), keepdim=True)  # dim=2(커널 축)만 남기고 정규화
        sd[pre + "weight"] = v * (g / norm)
    for key in (pre + "parametrizations.weight.original0", pre + "parametrizations.weight.original1"):
        if key in sd:  # 새 torch 이름으로 저장된 경우
            g = sd.pop(pre + "parametrizations.weight.original0")
            v = sd.pop(pre + "parametrizations.weight.original1")
            sd[pre + "weight"] = v * (g / v.norm(dim=(0, 1), keepdim=True))
            break
    return sd


def model_path(model_dir: str) -> str:
    return os.path.join(model_dir, MODEL_FILE)


def is_downloaded(model_dir: str) -> bool:
    p = model_path(model_dir)
    return os.path.isfile(p) and os.path.getsize(p) == MODEL_SIZE_BYTES


def download(model_dir: str, report=None) -> str:
    """가중치를 받는다(약 1.2GB). 임시 파일에 받은 뒤 크기를 확인하고 바꿔 넣어, 중간에 끊겨도 깨진 파일이 남지 않는다."""
    import urllib.request

    os.makedirs(model_dir, exist_ok=True)
    dst = model_path(model_dir)
    if is_downloaded(model_dir):
        return dst
    tmp = dst + ".part"
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or MODEL_SIZE_BYTES)
        got = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if report:
                report(got / total, f"정렬 모델 내려받는 중 {got / 2**20:.0f}/{total / 2**20:.0f}MB")
    if os.path.getsize(tmp) != total:
        os.remove(tmp)
        raise OSError("정렬 모델을 끝까지 받지 못했습니다. 네트워크를 확인하고 다시 시도하세요.")
    os.replace(tmp, dst)
    return dst


def load(model_dir: str, device: str = "cpu"):
    import torch

    path = model_path(model_dir)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(convert_state_dict(sd), strict=True)
    model.eval()
    return model.to(device)


def emissions(model, wave16k: np.ndarray, device: str = "cpu", report=None, cancelled=None) -> np.ndarray:
    """16kHz 모노 → (프레임 수, 레이블 수) 로그 확률. 프레임 i는 i×20ms 부근이다.

    긴 곡은 CHUNK_S 조각으로 나누고 양 끝에 CONTEXT_S를 덧붙여 계산한 뒤 가운데만 쓴다.
    """
    import torch

    y = np.asarray(wave16k, dtype=np.float32)
    n_frames = max(1, len(y) // HOP)
    out = np.full((n_frames, len(LABELS)), np.nan, dtype=np.float32)
    chunk = int(CHUNK_S * SAMPLE_RATE) // HOP * HOP
    ctx = int(CONTEXT_S * SAMPLE_RATE) // HOP * HOP
    starts = list(range(0, len(y), chunk)) or [0]
    with torch.inference_mode():
        for k, s in enumerate(starts):
            if cancelled is not None and cancelled():
                raise KeyboardInterrupt
            a = max(0, s - ctx)
            b = min(len(y), s + chunk + ctx)
            seg = y[a:b]
            if len(seg) < 400:  # 첫 합성곱 커널보다 짧으면 0으로 채운다
                seg = np.pad(seg, (0, 400 - len(seg)))
            x = torch.from_numpy(np.ascontiguousarray(seg))[None].to(device)
            lp = model(x)[0].float().cpu().numpy()
            f0 = (s - a) // HOP  # 이 조각에서 버릴 앞쪽 여유 프레임
            want0 = s // HOP
            want1 = min(n_frames, (s + chunk) // HOP)
            take = lp[f0: f0 + (want1 - want0)]
            out[want0: want0 + len(take)] = take
            if report:
                report((k + 1) / len(starts), f"정렬 모델 계산 중 ({device.upper()})")
    # 끝의 남는 프레임(조각 경계 반올림)은 바로 앞 값으로 채운다
    bad = np.isnan(out[:, 0])
    if bad.any():
        idx = np.where(~bad, np.arange(n_frames), 0)
        np.maximum.accumulate(idx, out=idx)
        out = out[idx]
        out[np.isnan(out)] = 0.0
    return out
