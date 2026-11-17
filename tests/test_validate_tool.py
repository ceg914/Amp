"""STEP 2.5 요구 2: 실제 음원 검증 도구 자체의 동작 확인 (합성 음원, 실제 음악 검증 아님)."""

import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from amp.devtools import synth, validate


def test_parse_pitch_and_reference(tmp_path):
    assert validate.parse_pitch("64") == 64 and validate.parse_pitch("E4") == 64
    assert validate.parse_pitch("C#5") == 73 and validate.parse_pitch("Bb3") == 58
    with pytest.raises(ValueError):
        validate.parse_pitch("H2")
    p = tmp_path / "정답.csv"
    p.write_text("start,end,pitch\n1.0,1.5,E4\n0.5,0.9,62\n", encoding="utf-8")
    ref = validate.read_reference(str(p))
    assert [(r.start, r.pitch) for r in ref] == [(0.5, 62), (1.0, 64)]


def test_separation_metrics_measure_lag_and_residual():
    sr = 16000
    rng = np.random.default_rng(0)
    v = rng.standard_normal((sr * 2, 2)).astype(np.float32) * 0.1
    i = rng.standard_normal((sr * 2, 2)).astype(np.float32) * 0.1
    mix = v + i
    m = validate.separation_metrics(mix, v, i, sr)
    assert m["same_length"] and m["lag_vocals_plus_instr_samples"] == 0
    assert m["reconstruction_residual_db"] is None or m["reconstruction_residual_db"] < -60
    late = np.roll(v + i, 40, axis=0)  # 40샘플(2.5ms) 늦은 분리 결과
    m2 = validate.separation_metrics(mix, late, np.zeros_like(i), sr)
    assert m2["lag_vocals_plus_instr_samples"] == 40 and m2["reconstruction_residual_db"] > -10


def test_validate_end_to_end_with_reference(tmp_path):
    melody = [(0.5, 1.0, 62), (1.1, 1.6, 65), (1.7, 2.4, 69)]
    y = synth.sung(melody, 3.0)  # 시험용 분리 모델(_identity)은 입력을 그대로 보컬로 내므로 반주를 섞지 않는다
    wav = tmp_path / "검증 곡 #1.wav"
    sf.write(wav, np.stack([y, y], 1), synth.SR)
    ref = tmp_path / "ref.csv"
    ref.write_text("\n".join(f"{a},{b},{m}" for a, b, m in melody), encoding="utf-8")
    res = validate.validate(str(wav), "B", str(ref), str(tmp_path / "out"), "cpu", "_identity")
    d = json.loads((res.report_dir / "report.json").read_text(encoding="utf-8"))
    assert d["separation_metrics"]["lag_vocals_plus_instr_samples"] == 0
    assert set(d["notes"]) == {"simple", "standard", "precise", "standard_v1"}
    assert {"separation", "pitch", "tempo"} <= set(d["timings_s"])
    assert d["resources"]["pitch"].get("peak_rss_mb", 0) > 0
    assert d["reference"]["standard"]["f1"] >= 0.9
    md = res.report_md.read_text(encoding="utf-8")
    assert "직접 확인할 항목" in md and "비브라토" in md and "분리 품질을 보장하지 않는다" in md
    assert (res.report_dir / "stems" / "vocals.wav").exists()
    assert not (res.report_dir / "work-pitch").exists()
