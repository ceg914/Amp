"""[기준선 v1 — 고정됨] STEP 2 커밋 3d49cd4의 온셋 강도 계산. 변경 전후 비교에만 쓰며 수정하지 않는다.

librosa 기본값(128ms 창, center=True)이라 피크가 실제 발성 시작보다 40~80ms 늦다.
"""

from __future__ import annotations

import numpy as np

from amp.engine.dsp.pitch import HOP, PITCH_SR


def onset_v1(y16: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y16, sr=PITCH_SR, hop_length=HOP)
