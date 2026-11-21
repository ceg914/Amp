"""음표 경계 보조 (v0.6, Qt 비의존). 피치 곡선은 바꾸지 않는다.

``trim_release``: 잔향 때문에 늦게 끝나는 음의 끝을 앞당기는 실험 기능. 기본은 꺼져 있다.
명세: "종료 경계 개선은 정답 시각이 있는 검증 자료를 확보한 뒤 적용 여부를 결정". 실제 곡에는 끝 시각 정답이 없다
(악보의 음 길이는 실제로 부른 길이가 아니다). 끝 시각을 아는 합성 사례에서만 쟀다 (보고서 7장).

규칙: 음의 가운데 60% 구간 20ms 음량(dB)의 중앙값을 그 음의 크기로 보고, 끝에서부터 그보다 drop_db 넘게 작은
프레임을 잘라 낸다. 줄이기만 하고 늘리지 않으며, 최소 길이와 원래 길이의 절반은 남긴다.
"""

from __future__ import annotations

import numpy as np

TRIM_DROP_DB = 15.0
MIN_KEEP_S = 0.08


def trim_release(notes, rms_fine: np.ndarray, frame_rate: float, start_time: float = 0.0,
                 drop_db: float = TRIM_DROP_DB, min_keep_s: float = MIN_KEEP_S) -> list:
    if rms_fine is None or len(rms_fine) == 0:
        return list(notes)
    db = 20 * np.log10(np.maximum(np.asarray(rms_fine, dtype=np.float64), 1e-9))
    n = len(db)
    out = []
    for note in notes:
        i0 = int(round((note.start - start_time) * frame_rate))
        i1 = int(round((note.end - start_time) * frame_rate))
        i0, i1 = max(0, i0), min(n, i1)
        if i1 - i0 < 5:
            out.append(note)
            continue
        m0, m1 = i0 + (i1 - i0) // 5, i1 - (i1 - i0) // 5
        level = float(np.median(db[m0:m1]))
        keep_from = i0 + max(int(round(min_keep_s * frame_rate)), (i1 - i0) // 2)
        j = i1
        while j - 1 > keep_from and db[j - 1] < level - drop_db:
            j -= 1
        if j < i1:
            out.append(note.with_(end=start_time + j / frame_rate))
        else:
            out.append(note)
    return out
