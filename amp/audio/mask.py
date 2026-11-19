"""보컬 없는 구간을 분리 결과(스템)에 적용한다. Qt 비의존.

구간 안에서는 보컬 트랙을 무음으로 만들고, 뺀 소리를 반주 트랙에 더한다. 그래서 보컬 + 반주 = 원래 두 트랙의 합이
샘플 단위로 그대로 유지된다(분리기가 보컬로 잘못 보낸 신스·기타 소리가 반주로 돌아간다).
구간 경계에는 짧은 페이드(구간 안쪽으로)를 넣어 끊김 소리를 막고, 구간 밖 샘플은 전혀 바꾸지 않는다.
디스크의 stems/*.wav는 바꾸지 않는다. 재생·파형·내보내기에 쓰는 메모리 사본만 바꾼다.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from amp.audio.data import AudioData

FADE_S = 0.02


def keep_gain(n_frames: int, sample_rate: int, ranges: Iterable[tuple[float, float]], fade_s: float = FADE_S) -> np.ndarray | None:
    """보컬에 곱할 이득 (1 = 그대로, 0 = 비움). 구간이 없거나 오디오 범위 밖이면 None."""
    g = None
    fade_n = max(1, int(round(fade_s * sample_rate)))
    for a, b in ranges:
        i = max(0, int(round(a * sample_rate)))
        j = min(n_frames, int(round(b * sample_rate)))
        if j <= i:
            continue
        if g is None:
            g = np.ones(n_frames, dtype=np.float32)
        f = min(fade_n, (j - i) // 2)
        seg = np.zeros(j - i, dtype=np.float32)
        if f > 0:
            ramp = (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, f))).astype(np.float32)  # 1 → 0 (양 끝 포함)
            if i > 0:  # 곡 처음·끝에 닿은 쪽은 페이드할 필요가 없다
                seg[:f] = ramp
            if j < n_frames:
                seg[len(seg) - f:] = ramp[::-1]
        g[i:j] = np.minimum(g[i:j], seg)
    return g


def apply_no_vocal(stems: dict[str, AudioData], ranges: Iterable[tuple[float, float]]) -> dict[str, AudioData]:
    """{'vocals', 'instrumental'} 스템에 보컬 없는 구간을 적용한 새 사본. 바꿀 것이 없으면 입력을 그대로 돌려준다."""
    vocals = stems.get("vocals")
    ranges = list(ranges)
    if vocals is None or not ranges:
        return stems
    g = keep_gain(vocals.frames, vocals.sample_rate, ranges)
    if g is None:
        return stems
    out = dict(stems)
    removed = vocals.samples * (1.0 - g)[:, None]
    out["vocals"] = AudioData((vocals.samples - removed).astype(np.float32), vocals.sample_rate)
    inst = stems.get("instrumental")
    if inst is not None and inst.sample_rate == vocals.sample_rate and inst.samples.shape == vocals.samples.shape:
        out["instrumental"] = AudioData((inst.samples + removed).astype(np.float32), inst.sample_rate)
    return out
