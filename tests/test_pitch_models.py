"""v0.7.0 Feature A: 외부 단성 피치 모델(CREPE·FCPE)을 torch만으로 옮긴 구현.

- 옮긴 계산이 원래 패키지(torchcrepe 0.0.24·torchfcpe 0.0.4)와 같은 값을 내는지 (패키지가 설치된 경우만)
- 합성 사례: 일정한 음(±10cent), 시작 시각(±20ms), 무음, 음역 밖, 비터비 경로, 반 프레임 옮기기
가중치 패키지가 없으면 그 시험은 건너뛴다(설치는 install_pitch). 시험이 가중치를 내려받지 않는다.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from amp.engine.dsp import pitch_models as PM

torch = pytest.importorskip("torch")
HAS_CREPE = importlib.util.find_spec("torchcrepe") is not None
HAS_FCPE = importlib.util.find_spec("torchfcpe") is not None
SR = 16000


def tone(f0: float, seconds: float, start: float = 0.0, total: float | None = None, harmonics: int = 5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = sum(0.4 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, harmonics + 1))
    y = np.zeros(int(SR * (total or start + seconds)), dtype=np.float32)
    y[int(SR * start): int(SR * start) + len(t)] = x.astype(np.float32)
    return y


def test_viterbi_follows_smooth_path_and_ignores_far_spike():
    T, S = 40, 360
    act = np.full((T, S), 0.01, dtype=np.float32)
    true = (100 + np.arange(T) // 4).astype(int)
    act[np.arange(T), true] = 0.9
    act[20, 300] = 0.95  # 한 프레임만 멀리 튐: 전이 폭(12칸) 밖이라 따라가지 않는다
    path = PM.banded_viterbi(act)
    assert np.array_equal(path, true)
    avg = PM.local_average_bins(act, path)
    assert np.max(np.abs(avg - true)) < 0.2


def test_shift_half_frame():
    f0 = np.array([100.0, 100.0, 200.0, np.nan, np.nan])
    v = np.array([True, True, True, False, False])
    out, _c, ov = PM.shift_half_frame(f0, np.ones(5), v)
    assert ov.tolist() == [True, True, True, True, False]  # (i-1, i) 중 하나라도 유성이면
    assert out[1] == pytest.approx(100.0) and out[2] == pytest.approx(np.sqrt(100 * 200))  # 센트 평균
    assert out[3] == pytest.approx(200.0) and np.isnan(out[4])


def test_missing_weights_package_reports_install_script():
    with pytest.raises(PM.PitchModelError) as e:
        PM.weights_path("crepe_full", find_spec=lambda name: None)
    assert e.value.code == "missing_pitch_model" and "install_pitch" in str(e.value)


@pytest.mark.parametrize("model_id", [pytest.param("crepe_tiny", marks=pytest.mark.skipif(not HAS_CREPE, reason="torchcrepe 없음")),
                                      pytest.param("fcpe", marks=pytest.mark.skipif(not HAS_FCPE, reason="torchfcpe 없음"))])
def test_constant_tone_onset_and_silence(model_id):
    y = tone(220.0, 2.0, start=1.0, total=3.5)  # 1.0~3.0초 220Hz, 앞뒤 무음
    r = PM.run(model_id, y, "cpu")
    f0, v = r["f0_hz"], r["voiced"]
    assert len(f0) == 1 + len(y) // PM.HOP and r["info"]["device"] == "cpu"
    cents = 1200 * np.log2(f0[v] / 220.0)
    assert abs(float(np.median(cents))) < 10
    inside = v[105:295]
    assert inside.mean() > 0.95
    assert not v[:95].any() and not v[310:].any()  # 무음에서는 무성
    assert abs(np.argmax(v) / 100.0 - 1.0) <= 0.02  # 시작 시각 ±20ms (창 64ms의 절반 안)
    assert r["info"]["confidence_kind"] == PM.MODELS[model_id]["confidence"]
    assert len(r["info"]["weights_sha256"]) == 64


@pytest.mark.skipif(not HAS_CREPE, reason="torchcrepe 없음")
def test_range_limits_mark_unvoiced():
    y = tone(110.0, 1.0, total=1.0)
    r = PM.run("crepe_tiny", y, "cpu", fmin=200.0, fmax=1000.0)
    assert not r["voiced"].any() and np.isnan(r["f0_hz"]).all()


def _stub_heavy_imports(monkeypatch):
    """원래 패키지를 불러오려면 필요한 모듈을 가짜로 둔다. conv_only FCPE·16kHz 입력·CREPE 신경망 계산에서는 쓰지 않는다."""
    for name in ("torchaudio", "torchaudio.transforms", "einops", "local_attention", "resampy", "tqdm"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torchaudio.transforms"].Resample = object
    sys.modules["torchaudio"].transforms = sys.modules["torchaudio.transforms"]
    sys.modules["einops"].rearrange = sys.modules["einops"].repeat = None
    sys.modules["local_attention"].LocalAttention = object
    sys.modules["tqdm"].tqdm = lambda x, **k: x
    for name in [m for m in sys.modules if m.split(".")[0] in ("torchcrepe", "torchfcpe")]:
        monkeypatch.delitem(sys.modules, name)


def _wobble(seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * (200 + 30 * np.sin(2 * np.pi * 5 * t)) * t)
            + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.mark.skipif(not HAS_CREPE, reason="torchcrepe 없음")
def test_crepe_port_matches_torchcrepe(monkeypatch):
    _stub_heavy_imports(monkeypatch)
    import torchcrepe

    y = _wobble()
    for size in ("tiny", "full"):
        w = torch.load(str(PM.weights_path(f"crepe_{size}")), map_location="cpu", weights_only=True)
        ref = torchcrepe.Crepe(size)
        ref.load_state_dict(w)
        mine = PM._crepe_module(size)
        mine.load_state_dict(w)
        ref.eval(), mine.eval()
        frames = torch.from_numpy(PM.crepe_frames(y)[:120])
        ref_frames = next(torchcrepe.preprocess(torch.from_numpy(y)[None], SR, PM.HOP, None, "cpu", True))[:120]
        assert float((frames - ref_frames).abs().max()) < 5e-3  # float32 표준편차 계산 순서 차이
        with torch.no_grad():
            assert float((ref(ref_frames) - mine(ref_frames)).abs().max()) < 1e-4


@pytest.mark.skipif(not HAS_FCPE, reason="torchfcpe 없음")
def test_fcpe_port_matches_torchfcpe(monkeypatch):
    _stub_heavy_imports(monkeypatch)
    import torchfcpe

    y = _wobble()
    ref = torchfcpe.spawn_bundled_infer_model("cpu")
    wav = torch.from_numpy(y)[None, :, None]
    with torch.no_grad():
        mel_ref = ref.wav2mel(wav, SR)[0]
        lat_ref = ref.model(mel_ref[None])[0].numpy()
        f0_ref = ref.infer(wav, SR, decoder_mode="local_argmax", threshold=0.006)[0, :, 0].numpy()
    mel = PM.fcpe_mel(y)
    assert tuple(mel.shape) == tuple(mel_ref.shape) and float((mel - mel_ref).abs().max()) < 1e-4
    ck = torch.load(str(PM.weights_path("fcpe")), map_location="cpu", weights_only=True)
    m = PM._fcpe_module(ck["config_dict"]["model"])
    state, table = PM._fcpe_state(ck["model"])
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        lat = m(mel[None])[0].numpy()
    assert float(np.abs(lat - lat_ref).max()) < 1e-4
    idx = PM.local_average_bins(lat, lat.argmax(1))
    f0 = 10 * 2 ** (np.interp(idx, np.arange(len(table)), table.numpy()) / 1200)
    ok = f0_ref > 0
    assert ((lat.max(1) > 0.006) == ok).all()
    assert float(np.max(np.abs(1200 * np.log2(f0[ok] / f0_ref[ok])))) < 0.1
