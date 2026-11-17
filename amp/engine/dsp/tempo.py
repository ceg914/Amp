"""librosa 기반 BPM 추정.

박 추정은 2배·절반 템포를 헷갈리기 쉽다. 그래서 추정값 하나만 내지 않고
템포그램 자기상관 점수가 붙은 후보(추정값, ×2, ×½)를 함께 돌려준다.
마디의 첫 박(다운비트)은 추정하지 않는다. 박자는 4/4로 가정한다.
"""

from __future__ import annotations

import numpy as np

TEMPO_SR = 22050
HOP = 512
MIN_SECONDS = 6.0
MIN_BPM, MAX_BPM = 40.0, 240.0


def estimate_tempo(mono: np.ndarray, sr: int) -> dict:
    import librosa

    y = np.asarray(mono, dtype=np.float32)
    if sr != TEMPO_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TEMPO_SR, res_type="soxr_hq").astype(np.float32)
    duration = len(y) / TEMPO_SR
    if duration < MIN_SECONDS or float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) < 1e-4:
        return {"bpm": 120.0, "first_beat": 0.0, "candidates": [], "reliable": False,
                "reason": "오디오가 너무 짧거나 조용해 BPM을 추정하지 못했습니다. 기본값 120을 씁니다."}

    oenv = librosa.onset.onset_strength(y=y, sr=TEMPO_SR, hop_length=HOP)
    bpm = float(librosa.feature.tempo(onset_envelope=oenv, sr=TEMPO_SR, hop_length=HOP)[0])
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=TEMPO_SR, hop_length=HOP)
    ac = tg.mean(axis=1)
    if ac.max() > 0:
        ac = ac / ac.max()

    def score(b: float) -> float:
        lag = 60.0 * TEMPO_SR / (HOP * b)
        i = int(round(lag))
        return float(ac[i]) if 0 < i < len(ac) else 0.0

    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=TEMPO_SR, hop_length=HOP, bpm=bpm, units="time")
    refined = refine_with_beats(np.asarray(beats, dtype=np.float64), bpm)
    if refined is not None:
        bpm, first = refined
    else:
        first = float(beats[0]) if len(beats) else 0.0
    period = 60.0 / bpm
    first_beat = float(first % period)
    candidates = []
    for c in (bpm, bpm * 2.0, bpm / 2.0):
        if MIN_BPM <= c <= MAX_BPM:
            candidates.append({"bpm": round(c, 2), "score": round(score(c), 3)})
    return {
        "bpm": round(bpm, 2),
        "first_beat": round(first_beat, 4),
        "candidates": candidates,
        "reliable": len(beats) >= 8,
        "beats_found": int(len(beats)),
        "reason": "",
    }


def refine_with_beats(beats: np.ndarray, bpm: float) -> tuple[float, float] | None:
    """박 시각에 직선(박 = 시작 + 번호 × 간격)을 맞춰 BPM을 다듬는다.

    librosa 템포그램은 지연을 정수 프레임으로만 계산해서(512 hop에서 120BPM은 117.45나 123.0이 된다)
    값이 양자화된다. 박이 8개 이상이고 다듬은 값이 원래 값의 ±8% 안일 때만 쓴다.
    (bpm, 첫 박 시각) 또는 None.
    """
    if len(beats) < 8:
        return None
    period0 = float(np.median(np.diff(beats)))
    if period0 <= 0:
        return None
    # 박 번호: 간격마다 1씩(박을 건너뛴 곳은 그만큼) 늘린다. 전체를 중앙값으로 나눠 반올림하면
    # 간격이 0.511/0.488초처럼 번갈아 나올 때 뒤쪽 번호가 어긋난다.
    steps = np.maximum(1, np.round(np.diff(beats) / period0))
    idx = np.concatenate([[0.0], np.cumsum(steps)])
    slope, intercept = np.polyfit(idx, beats, 1)
    if slope <= 0:
        return None
    refined = 60.0 / slope
    if abs(refined - bpm) > 0.08 * bpm:
        return None
    return float(refined), float(intercept)
