"""Mel-Band RoFormer 이식: 계산 경로, 파라미터 이름, 체크포인트 읽기(안전·호환), 설정 해석, 조각 추론, 분리·파이프라인 연결.

실제 학습된 가중치는 쓰지 않는다(저장소에 없고 이 시험 환경에서 내려받지 않는다). 무작위 가중치의 작은 모델로
구조와 시간축을 확인한다. 분리 품질은 이 시험의 범위가 아니다.
"""

import io
import os
import re
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from amp.engine.dsp import roformer as R  # noqa: E402

SR = 44100
TINY = dict(dim=16, depth=2, num_bands=16, heads=2, dim_head=8, mlp_expansion_factor=1, mask_estimator_depth=1,
            ff_mult=2, chunk_size=44100, num_overlap=2)


def tiny_model(**kw):
    torch.manual_seed(0)
    return R.build_model(R.RoformerConfig(**{**TINY, **kw})).eval()


def save_ckpt(path, model, wrap=None, prefix=""):
    sd = {prefix + k: v for k, v in model.state_dict().items()}
    torch.save({wrap: sd} if wrap else sd, str(path))
    return str(path)


def write_yaml(path, **model_kw):
    lines = ["model:"] + [f"  {k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in model_kw.items()
                          if k not in ("chunk_size", "num_overlap")]
    lines += ["  multi_stft_resolutions_window_sizes: !!python/tuple", "  - 4096", "  - 2048",
              "training:", "  instruments:", "  - vocals", "  - other", "  target_instrument: vocals",
              "inference:", f"  num_overlap: {model_kw.get('num_overlap', 2)}",
              f"  chunk_size: {model_kw.get('chunk_size', 352800)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --- 계산 경로 -----------------------------------------------------------------


def test_forward_keeps_length_and_identity_mask_reconstructs_input():
    m = tiny_model()
    x = torch.randn(1, 2, SR) * 0.1
    with torch.inference_mode():
        assert tuple(m(x).shape) == (1, 1, 2, SR)

    class One(torch.nn.Module):  # 모든 대역에서 마스크 1+0j → 겹친 대역 평균도 1 → 입력 그대로
        def forward(self, h):
            out = torch.zeros(h.shape[0], h.shape[1], sum(m.band_split.dim_inputs))
            out[..., 0::2] = 1.0
            return out

    m.mask_estimators = torch.nn.ModuleList([One()])
    m.cfg.zero_dc = False
    with torch.inference_mode():
        y = m(x)[0, 0]
    assert float((y - x[0]).abs().max()) < 1e-6


def test_rotary_matches_complex_rotation_of_interleaved_pairs():
    rot = tiny_model().layers[0][0].layers[0][0].rotary_embed
    q = torch.randn(1, 1, 12, 8)
    out = rot.rotate_queries_or_keys(q)
    for p in (0, 5, 11):
        z = torch.complex(q[0, 0, p, 0::2].double(), q[0, 0, p, 1::2].double()) * torch.exp(1j * p * rot.freqs.double())
        ref = torch.stack([z.real, z.imag], -1).flatten().float()
        assert float((out[0, 0, p] - ref).abs().max()) < 1e-5
    m = tiny_model()  # 회전 모듈은 시간 축 층끼리 공유 (원본과 같음)
    assert m.layers[0][0].layers[0][0].rotary_embed is m.layers[1][0].layers[0][0].rotary_embed


def test_band_layout_covers_all_frequencies_and_interleaves_stereo():
    cfg = R.RoformerConfig()
    fpb, idx, per_freq, dims = R.band_layout(cfg)
    assert fpb.shape == (60, 1025) and bool(fpb.any(dim=0).all())
    assert int(per_freq.min()) >= 1
    assert len(dims) == 60 and sum(dims) == idx.numel() * 2
    assert idx[:4].tolist() == [0, 1, 2, 3]  # 첫 대역 주파수 0의 왼쪽(0)·오른쪽(1), 주파수 1의 0·1 …


def test_parameter_names_follow_reference_layout():
    """참조 구현(ZFTurbo mel_band_roformer.py)을 읽고 정리한 이름 형태와 같아야 그 체크포인트를 읽을 수 있다."""
    m = R.build_model(R.RoformerConfig())
    keys = list(m.state_dict())
    shapes = {re.sub(r"\d+", "#", k) for k in keys}
    assert shapes == {
        "band_split.to_features.#.#.gamma", "band_split.to_features.#.#.weight", "band_split.to_features.#.#.bias",
        "layers.#.#.layers.#.#.norm.gamma", "layers.#.#.layers.#.#.to_qkv.weight",
        "layers.#.#.layers.#.#.to_gates.weight", "layers.#.#.layers.#.#.to_gates.bias",
        "layers.#.#.layers.#.#.to_out.#.weight", "layers.#.#.layers.#.#.rotary_embed.freqs",
        "layers.#.#.layers.#.#.net.#.gamma", "layers.#.#.layers.#.#.net.#.weight", "layers.#.#.layers.#.#.net.#.bias",
        "layers.#.#.norm.gamma",
        "mask_estimators.#.to_freqs.#.#.#.weight", "mask_estimators.#.to_freqs.#.#.#.bias",
    }
    assert "layers.5.1.layers.0.1.net.4.weight" in keys and "layers.0.0.layers.0.0.to_out.0.weight" in keys
    assert "mask_estimators.0.to_freqs.59.0.4.weight" in keys  # mask_estimator_depth 2 → Linear 0, 2, 4
    assert sum(p.numel() for p in m.parameters()) == 228_202_852


# --- 체크포인트 읽기 ---------------------------------------------------------------


@pytest.mark.parametrize("wrap,prefix", [(None, ""), ("state_dict", "model."), ("state_dict", "module.")])
def test_load_model_reads_structure_from_tensors(tmp_path, wrap, prefix):
    src = tiny_model()
    path = save_ckpt(tmp_path / "m.ckpt", src, wrap, prefix)
    model, cfg, notes = R.load_model(path)
    for k in ("dim", "depth", "num_bands", "heads", "dim_head", "mlp_expansion_factor", "mask_estimator_depth", "ff_mult"):
        assert getattr(cfg, k) == TINY[k], k
    assert cfg.stereo and notes == []
    x = torch.randn(1, 2, SR) * 0.1
    with torch.inference_mode():
        assert torch.allclose(model(x), src(x), atol=1e-6)


def test_old_checkpoints_without_gates_or_output_norm_load(tmp_path):
    src = R.build_model(R.RoformerConfig(**TINY, gates=False, norm_output=False))
    model, cfg, _ = R.load_model(save_ckpt(tmp_path / "old.ckpt", src))
    assert not cfg.gates and not cfg.norm_output
    assert (cfg.heads, cfg.dim_head) == (TINY["heads"], TINY["dim_head"])  # 회전 주파수 개수에서 계산
    x = torch.randn(1, 2, SR) * 0.1
    with torch.inference_mode():
        assert torch.allclose(model(x), src.eval()(x), atol=1e-6)


def test_mono_model_is_detected(tmp_path):
    src = R.build_model(R.RoformerConfig(**{**TINY, "stereo": False}))
    model, cfg, notes = R.load_model(save_ckpt(tmp_path / "mono.ckpt", src))
    assert cfg.stereo is False and notes


def test_checkpoint_with_code_objects_is_refused(tmp_path):
    """weights_only: 텐서가 아닌 객체(임의 클래스)를 담은 pickle은 열지 않는다."""
    path = tmp_path / "evil.ckpt"
    torch.save({"weights": Counter(a=1)}, str(path))  # collections.Counter는 weights_only 허용 목록에 없다
    with pytest.raises(R.RoformerError) as e:
        R.load_state_dict(str(path))
    assert e.value.code == "model_file"


def test_bad_files_give_clear_errors(tmp_path):
    with pytest.raises(R.RoformerError, match="없습니다"):
        R.load_model(str(tmp_path / "없음.ckpt"))
    (tmp_path / "broken.ckpt").write_bytes(b"not a checkpoint")
    with pytest.raises(R.RoformerError, match="읽지 못했습니다"):
        R.load_model(str(tmp_path / "broken.ckpt"))
    torch.save({"conv.weight": torch.zeros(3)}, str(tmp_path / "other.ckpt"))
    with pytest.raises(R.RoformerError, match="Mel-Band RoFormer 모델 파일이 아닙니다"):
        R.load_model(str(tmp_path / "other.ckpt"))


def test_stft_mismatch_asks_for_yaml_and_yaml_fixes_it(tmp_path):
    src = R.build_model(R.RoformerConfig(**TINY, stft_n_fft=1024, stft_win_length=1024))
    path = save_ckpt(tmp_path / "m.ckpt", src)
    with pytest.raises(R.RoformerError, match=r"\.yaml"):
        R.load_model(path)
    y = write_yaml(tmp_path / "c.yaml", **{**TINY, "stft_n_fft": 1024, "stft_win_length": 1024, "chunk_size": 88200,
                                         "num_overlap": 4})
    model, cfg, _ = R.load_model(path, y)
    assert cfg.stft_n_fft == 1024 and cfg.chunk_size == 88200 and cfg.num_overlap == 4 and cfg.target == "vocals"


def test_yaml_that_contradicts_checkpoint_is_rejected(tmp_path):
    path = save_ckpt(tmp_path / "m.ckpt", tiny_model())
    y = write_yaml(tmp_path / "c.yaml", **{**TINY, "dim": 32})
    with pytest.raises(R.RoformerError, match="dim=32"):
        R.load_model(path, y)
    (tmp_path / "bad.yaml").write_text("model:\n  dim: 16\n  linear_transformer_depth: 1\n", encoding="utf-8")
    with pytest.raises(R.RoformerError, match="지원하지 않습니다"):
        R.read_yaml_config(str(tmp_path / "bad.yaml"))


# --- 조각 추론 -------------------------------------------------------------------


@pytest.mark.parametrize("n,overlap", [(30_000, 2), (60_000, 2), (44100 * 5 + 7, 2), (44100 * 5 + 7, 4)])
def test_demix_is_exact_for_identity_model(n, overlap):
    mix = np.random.default_rng(n).standard_normal((2, n)).astype(np.float32)
    est = R.demix(lambda part: part[None], mix, 44100, overlap)
    assert est.shape == (1, 2, n)
    assert float(np.abs(est[0] - mix).max()) < 1e-5


def test_separate_with_roformer_keeps_frames_and_sums_to_original(tmp_path):
    from amp.engine.dsp.separation import separate

    ckpt = save_ckpt(tmp_path / "m.ckpt", tiny_model())
    y = write_yaml(tmp_path / "c.yaml", **TINY)
    files = {"checkpoint": ckpt, "config": y}
    x = (np.random.default_rng(1).standard_normal((SR * 3, 2)) * 0.1).astype(np.float32)
    msgs = []
    vocals, inst, info = separate(x, SR, "mel_roformer", "cpu", lambda f, m: msgs.append((f, m)), files)
    assert vocals.shape == inst.shape == x.shape
    assert np.abs(vocals + inst - x).max() < 1e-6  # 반주 = 원본 − 보컬
    assert np.abs(vocals).max() > 0
    # 48kHz 입력: 44.1kHz로 분리하고 원래 샘플레이트·프레임 수로 되돌린다
    x48 = (np.random.default_rng(2).standard_normal((48000 * 2 + 5, 1)) * 0.1).astype(np.float32)
    v48, i48, _ = separate(x48, 48000, "mel_roformer", "cpu", None, files)
    assert v48.shape == i48.shape == x48.shape
    assert info["model"] == "mel_roformer" and info["device"] == "cpu"
    assert info["checkpoint"] == "m.ckpt" and len(info["checkpoint_sha256"]) == 64
    assert info["checkpoint_identity"]["size"] == os.path.getsize(ckpt)
    assert any("RoFormer" in m for _f, m in msgs)


def test_separate_requires_checkpoint():
    from amp.engine.dsp.separation import SeparationError, separate

    with pytest.raises(SeparationError) as e:
        separate(np.zeros((1000, 2), np.float32), SR, "mel_roformer", "cpu", None, {})
    assert e.value.code == "model_file"


# --- 작업 프로세스·파이프라인 ---------------------------------------------------------


def _in_process(calls):
    from amp.engine import worker

    def runner(command, args, progress=None, token=None, numba_cache_dir=None):
        calls.append((command, dict(args)))
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    return runner


def test_worker_check_model_command(tmp_path):
    from amp.engine import worker

    ckpt = save_ckpt(tmp_path / "m.ckpt", tiny_model())
    old = worker._PROTO
    worker._PROTO = io.StringIO()
    try:
        out = worker.COMMANDS["check_model"]({"checkpoint": ckpt, "config": ""})
        with pytest.raises(Exception) as e:
            worker.COMMANDS["check_model"]({"checkpoint": str(tmp_path / "x.ckpt"), "config": ""})
    finally:
        worker._PROTO = old
    assert out["config"]["num_bands"] == 16 and out["params"] > 0 and out["identity"]["size"] > 0
    assert worker._classify(e.value)[0] == "model_file"


def test_pipeline_uses_roformer_and_reuses_until_checkpoint_changes(tmp_path):
    from dataclasses import replace

    from amp.audio.decoder import load_audio
    from amp.core.errors import AnalysisError
    from amp.devtools.synth import sung
    from amp.engine.cache import file_content_key
    from amp.engine.settings import AnalysisSettings
    from amp.engine.transcription import TranscriptionPipeline, TranscriptionRequest

    y = sung([(1.0, 1.8, 64), (1.9, 2.7, 67)], 3.0)
    path = tmp_path / "곡.wav"
    sf.write(path, np.stack([y, y], 1), SR)
    ckpt = save_ckpt(tmp_path / "m.ckpt", tiny_model())
    files = {"checkpoint": ckpt, "config": write_yaml(tmp_path / "c.yaml", **TINY)}
    calls = []
    pipe = TranscriptionPipeline(_in_process(calls))
    st = AnalysisSettings(separation="mel_roformer", device="cpu")
    req = TranscriptionRequest(load_audio(str(path)), str(path), file_content_key(str(path)), st,
                               str(tmp_path / "p" / "stems"), str(tmp_path / "p" / "cache"), model_files=files)
    with pytest.raises(AnalysisError, match="모델 파일"):
        pipe.run(replace(req, model_files={}))
    out = pipe.run(req)
    seps = [a for c, a in calls if c == "separate"]
    assert len(seps) == 1 and seps[0]["model"] == "mel_roformer" and seps[0]["model_files"] == files
    assert out.manifest.model == "mel_roformer" and out.result.models["separation"].name == "mel_roformer"
    assert out.stems["vocals"].frames == req.audio.frames
    pipe.run(req)  # 같은 모델 파일 → 재사용
    assert len([c for c, _ in calls if c == "separate"]) == 1
    pipe.run(replace(req, model_files={}))  # 파일 설정이 없어도 있는 결과는 쓴다
    assert len([c for c, _ in calls if c == "separate"]) == 1
    st_ = os.stat(ckpt)
    os.utime(ckpt, ns=(st_.st_atime_ns, st_.st_mtime_ns + 10_000_000_000))  # 모델 파일이 바뀜
    pipe.run(req)
    assert len([c for c, _ in calls if c == "separate"]) == 2
