"""참조 악보 ↔ 음원 자동 정렬. 피치 곡선과 악보 음표 순서를 쓴다.

1단계(전체): 악보를 템포 배율 s로 늘리고 줄인 음이름(12개) 막대와 음원 피치 곡선의 음이름을 시간축으로 교차 상관해
   가장 잘 맞는 시작 오프셋·배율을 찾는다(긴 전주가 있어도 곡 전체에서 찾는다). 음이름을 쓰므로 옥타브 차이에 강하다.
2단계(구간): 1단계 결과 주변 ±BAND_S초 안에서 DTW로 프레임을 맞춰 템포 변화·루바토를 따라간다.
   음표 시작마다 대응점을 만들고 단조 증가·기울기 범위를 지키도록 다듬는다.

품질 지표는 '악보가 소리 나는 프레임 중 음원도 유성이고 음이름이 1반음 안인 비율'이다. 이 값이 낮으면 불확실로 표시하고
수동 조절을 권한다. 정렬 결과는 확정 전까지 미리보기다.
"""

from __future__ import annotations

import numpy as np

from amp.compare.reference import ReferenceNote, ScoreAlignment

FPS = 20.0
SCALES = tuple(np.round(np.arange(0.80, 1.2501, 0.01), 3))
BAND_S = 3.0
UNCERTAIN_BELOW = 0.5
MIN_MARGIN = 0.2  # 우연 일치 기준선(악보를 5반음 옮겨 같은 방식으로 정렬)보다 이만큼은 높아야 믿는다
MIN_SLOPE, MAX_SLOPE = 0.5, 2.0


def audio_pitch_track(curve, fps: float = FPS, min_conf: float = 0.05) -> np.ndarray:
    """피치 곡선 → fps 격자의 MIDI 값 (유성이 아니면 NaN)."""
    f0 = np.asarray(curve.f0_hz, dtype=float)
    conf = np.asarray(curve.confidence, dtype=float) if getattr(curve, "confidence", None) is not None else np.ones_like(f0)
    t = curve.start_time + np.arange(len(f0)) / curve.frame_rate
    dur = t[-1] if len(t) else 0.0
    grid = np.arange(0.0, dur, 1.0 / fps)
    idx = np.clip(np.round((grid - curve.start_time) * curve.frame_rate).astype(int), 0, max(len(f0) - 1, 0))
    if not len(f0):
        return np.full(0, np.nan)
    hz = f0[idx]
    ok = np.isfinite(hz) & (hz > 0) & (conf[idx] >= min_conf)
    out = np.full(len(grid), np.nan)
    out[ok] = 69 + 12 * np.log2(hz[ok] / 440.0)
    return out


def ref_pitch_track(notes: list[ReferenceNote], to_audio, n_frames: int, fps: float = FPS) -> np.ndarray:
    out = np.full(n_frames, np.nan)
    for n in notes:
        a, b = to_audio(n.start), to_audio(n.end)
        i0, i1 = max(0, int(round(a * fps))), min(n_frames, int(round(b * fps)))
        if i1 > i0:
            out[i0:i1] = n.pitch
    return out


def _chroma(track: np.ndarray) -> np.ndarray:
    m = np.zeros((len(track), 12))
    ok = np.isfinite(track)
    m[np.flatnonzero(ok), np.round(track[ok]).astype(int) % 12] = 1.0
    return m


def match_ratio(audio: np.ndarray, ref: np.ndarray) -> float:
    """악보가 소리 나는 프레임 중 음원이 유성이고 음이름이 1반음 안인 비율."""
    sounding = np.isfinite(ref)
    if not sounding.any():
        return 0.0
    both = sounding & np.isfinite(audio)
    d = np.abs(audio[both] - ref[both]) % 12
    d = np.minimum(d, 12 - d)
    return float((d <= 1.0).sum() / sounding.sum())


def coarse_align(notes: list[ReferenceNote], audio: np.ndarray, fps: float = FPS,
                 scales=SCALES) -> tuple[float, float, float]:
    """(오프셋 초, 배율, 일치 점수)."""
    if not notes or not np.isfinite(audio).any():
        return 0.0, 1.0, 0.0
    a_chroma = _chroma(audio)
    # 이웃 프레임까지 번지게 해 약간의 시간 어긋남을 허용
    a_chroma = np.maximum(a_chroma, np.vstack([a_chroma[1:], np.zeros((1, 12))]))
    a_chroma = np.maximum(a_chroma, np.vstack([np.zeros((1, 12)), a_chroma[:-1]]))
    first = min(n.start for n in notes)
    best = (0.0, 1.0, -1.0)
    na = len(audio)
    for s in scales:
        span = (max(n.end for n in notes) - first) * s
        nr = int(np.ceil(span * fps)) + 1
        r = ref_pitch_track(notes, lambda t, s=s: (t - first) * s, nr, fps)
        r_chroma = _chroma(r)
        total = max(1.0, np.isfinite(r).sum())
        n_fft = 1 << int(np.ceil(np.log2(na + nr)))
        acc = np.zeros(n_fft)
        for pc in range(12):
            if not r_chroma[:, pc].any() or not a_chroma[:, pc].any():
                continue
            fa = np.fft.rfft(a_chroma[:, pc], n_fft)
            fr = np.fft.rfft(r_chroma[:, pc], n_fft)
            acc += np.fft.irfft(fa * np.conj(fr), n_fft)
        lags = np.arange(n_fft)
        lags = np.where(lags > n_fft // 2, lags - n_fft, lags)
        valid = (lags >= -nr // 2) & (lags <= na)
        k = int(np.argmax(np.where(valid, acc, -np.inf)))
        score = acc[k] / total
        # 점수가 같으면 배율 1에 가까운 쪽
        if score > best[2] + 1e-9 or (abs(score - best[2]) <= 1e-9 and abs(s - 1) < abs(best[1] - 1)):
            best = (float(lags[k] / fps - first * s), float(s), float(score))
    return best


def _dtw_band(ref: np.ndarray, audio: np.ndarray, center: np.ndarray, band: int) -> np.ndarray:
    """ref 프레임 i → 음원 프레임 path[i] (띠 안 DTW)."""
    n, m = len(ref), len(audio)
    width = 2 * band + 1
    inf = 1e18
    cost_acc = np.full((n, width), inf)
    back = np.zeros((n, width), dtype=np.int8)  # 0: 대각, 1: i만, 2: j만
    js_all = center[:, None] - band + np.arange(width)[None, :]
    for i in range(n):
        js = js_all[i]
        ok = (js >= 0) & (js < m)
        a = audio[np.clip(js, 0, m - 1)]
        r = ref[i]
        if np.isfinite(r):
            d = np.abs(a - r) % 12
            d = np.minimum(d, 12 - d)
            c = np.where(np.isfinite(a), np.minimum(d / 3.0, 1.0), 0.6)
        else:
            c = np.where(np.isfinite(a), 0.4, 0.0)
        c = np.where(ok, c, inf)
        if i == 0:
            row = c.copy()
            # 같은 행에서 왼쪽으로(j만 증가)
            for k in range(1, width):
                if row[k - 1] + c[k] < row[k]:
                    row[k] = row[k - 1] + c[k]
                    back[0, k] = 2
            cost_acc[0] = row
            continue
        shift = center[i] - center[i - 1]  # 이전 행의 띠가 몇 칸 밀렸나
        prev = cost_acc[i - 1]
        idx = np.arange(width) + shift
        diag_src = idx - 1
        up_src = idx
        diag = np.where((diag_src >= 0) & (diag_src < width), prev[np.clip(diag_src, 0, width - 1)], inf)
        up = np.where((up_src >= 0) & (up_src < width), prev[np.clip(up_src, 0, width - 1)], inf)
        row = np.minimum(diag, up + 0.2) + c
        b = np.where(diag <= up + 0.2, 0, 1).astype(np.int8)
        for k in range(1, width):  # j만 증가 (음원이 더 김)
            alt = row[k - 1] + c[k] + 0.2
            if alt < row[k]:
                row[k] = alt
                b[k] = 2
        cost_acc[i] = row
        back[i] = b
    # 마지막 행의 최소에서 거꾸로
    k = int(np.argmin(cost_acc[-1]))
    path = np.zeros(n, dtype=int)
    i = n - 1
    while i >= 0:
        path[i] = js_all[i, k]
        step = back[i, k]
        if step == 2 and k > 0:
            k -= 1
            continue
        if i == 0:
            break
        shift = center[i] - center[i - 1]
        k = k + shift - (1 if step == 0 else 0)
        k = int(min(max(k, 0), width - 1))
        i -= 1
    return path


def auto_align(notes: list[ReferenceNote], curve, duration: float) -> ScoreAlignment:
    audio = audio_pitch_track(curve)
    if len(audio) == 0 or not notes:
        return ScoreAlignment(method="auto", quality={"match": 0.0, "uncertain": True, "reason": "피치 곡선이나 음표가 없음"})
    offset, scale, _score = coarse_align(notes, audio)
    coarse = ScoreAlignment(offset=offset, scale=scale, method="auto")
    r_coarse = ref_pitch_track(notes, coarse.audio_time, len(audio))
    q0 = match_ratio(audio, r_coarse)
    # 2단계: 악보 시간 격자(FPS)에서 DTW
    s0, s1 = min(n.start for n in notes), max(n.end for n in notes)
    st = np.arange(s0, s1, 1.0 / FPS)
    ref = np.full(len(st), np.nan)
    for n in notes:
        ref[(st >= n.start) & (st < n.end)] = n.pitch
    center = np.clip(np.round(np.array([coarse.audio_time(t) for t in st]) * FPS).astype(int), 0, len(audio) - 1)
    path = _dtw_band(ref, audio, center, int(BAND_S * FPS))
    a_times = path / FPS
    # 음표 시작마다 대응점, 단조 증가·기울기 제한
    pts: list[tuple[float, float]] = []
    for n in sorted(notes, key=lambda x: x.start):
        i = int(min(max(round((n.start - s0) * FPS), 0), len(st) - 1))
        pts.append((n.start, float(a_times[i])))
    pts.append((s1, float(a_times[-1]) + 1.0 / FPS))
    warp: list[tuple[float, float]] = []
    for s, a in pts:
        if warp:
            ps, pa = warp[-1]
            if s <= ps + 1e-6:
                continue
            a = min(max(a, pa + MIN_SLOPE * (s - ps)), pa + MAX_SLOPE * (s - ps))
        warp.append((s, a))
    fine = ScoreAlignment(offset=offset, scale=scale, warp=tuple(warp), method="auto")
    q1 = match_ratio(audio, ref_pitch_track(notes, fine.audio_time, len(audio)))
    use = fine if q1 >= q0 else ScoreAlignment(offset=offset, scale=scale, method="auto")
    q = max(q0, q1)
    # 우연 일치 기준선: DTW는 시간을 자유롭게 늘여 다른 선율과도 40~50% 맞춘다(시험으로 확인). 같은 리듬을 5반음 옮겨
    # 같은 방식으로 맞춘 값보다 충분히 높아야 정렬을 믿는다
    shifted = np.where(np.isfinite(ref), ref + 5, np.nan)
    null_path = _dtw_band(shifted, audio, center, int(BAND_S * FPS))
    null_track = np.full(len(audio), np.nan)
    for i, j in enumerate(null_path):
        if 0 <= j < len(audio) and np.isfinite(shifted[i]):
            null_track[j] = shifted[i]
    q_null = match_ratio(audio, null_track)
    margin = q - q_null
    uncertain = q < UNCERTAIN_BELOW or margin < MIN_MARGIN
    quality = {"match_coarse": round(q0, 3), "match_fine": round(q1, 3), "match": round(q, 3),
               "match_null": round(q_null, 3), "offset": round(offset, 3), "scale": scale, "uncertain": uncertain,
               "used": "구간 정렬" if use.warp else "전체 오프셋·배율"}
    if uncertain:
        quality["reason"] = (f"악보 음과 맞는 프레임 {q * 100:.0f}% (우연히 맞는 수준 {q_null * 100:.0f}%와 차이 "
                             f"{margin * 100:.0f}%p). 시작 위치·템포를 직접 맞추거나 보컬 성부를 맞게 골랐는지 확인하세요.")
    return ScoreAlignment(offset=use.offset, scale=use.scale, warp=use.warp, method="auto", confirmed=False,
                          quality=quality)


def alignment_quality(notes: list[ReferenceNote], alignment: ScoreAlignment, curve) -> float:
    audio = audio_pitch_track(curve)
    return match_ratio(audio, ref_pitch_track(notes, alignment.audio_time, len(audio)))
