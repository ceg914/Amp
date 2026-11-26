"""v0.7.0 명세 15.2: 보컬 유사 성분(근사 지표). 합성 신호로 값의 뜻을 확인한다."""

import numpy as np

from amp.engine.stem_compare import vocal_share, vocal_share_files


def test_vocal_share_matches_mixed_in_amount(tmp_path):
    rng = np.random.default_rng(0)
    sr = 8000
    v = rng.standard_normal(sr * 10).astype(np.float32)
    inst = rng.standard_normal(sr * 10).astype(np.float32)
    leaked = inst + 0.5 * v  # 보컬 에너지 0.25 : 악기 1
    r = vocal_share(leaked, v, sr)
    assert r.seconds == 10 and abs(r.share_db - 10 * np.log10(0.25 / 1.25)) < 0.3
    clean = vocal_share(inst, v, sr)
    assert clean.share_db < -25  # 섞이지 않았으면 아주 낮다 (우연한 상관만)
    quiet = v.copy()
    quiet[: sr * 5] = 0  # 보컬이 없는 구간은 세지 않는다
    assert vocal_share(leaked, quiet, sr).seconds == 5
    assert vocal_share(leaked, np.zeros_like(v), sr).share_db is None
    import soundfile as sf

    sf.write(tmp_path / "i.wav", leaked, sr, subtype="FLOAT")  # 1을 넘는 값이 잘리지 않게
    sf.write(tmp_path / "v.wav", v, sr, subtype="FLOAT")
    f = vocal_share_files(tmp_path / "i.wav", tmp_path / "v.wav")
    assert abs(f.share_db - r.share_db) < 0.2 and "근사" in f.to_dict()["note"]
