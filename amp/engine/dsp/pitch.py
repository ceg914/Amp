"""librosa pYIN 기반 기본 주파수(f0) 추출.

- 16kHz로 낮춰 10ms 간격(100fps)으로 분석한다. 프레임 i는 i/100초를 중심으로 한다(center=True).
- 30초 단위로 나눠 처리하고 앞뒤 1초를 겹친다. pYIN은 곡 전체 확률 행렬을 만들어
  긴 곡에서 메모리를 크게 쓰는데, 청크로 나누면 상한이 생기고 청크마다 진행률과 취소를 확인할 수 있다.
- 같은 격자에서 온셋 강도와 RMS를 함께 계산한다(음표 분할·무성 판정에 쓴다).
  온셋은 32ms 창, 비중심(center=False)으로 계산한다. librosa 기본값(128ms 창, center=True)은
  합성 음원에서 실제 발성 시작보다 40~80ms 늦게 피크가 찍혔다 (STEP 2.5에서 측정, 기준선 v1).
  RMS는 64ms 창(``rms``)과 20ms 창(``rms_fine``) 두 가지다. 40ms 남짓한 음절 사이 틈은 20ms 창에서만 보인다.

v0.6 Hybrid: 청크 k의 pYIN 결과는 16kHz 신호의 [청크 시작 - 1초, 청크 끝 + 1초] 표본만으로 정해진다.
``reuse`` 로 받은 이전 결과(다른 분리 결과·이전 Hybrid)의 16kHz 신호가 그 창에서 비트 단위로 같으면
그 청크는 다시 계산하지 않고 이전 f0·확률을 옮긴다. 결과는 처음부터 계산한 것과 똑같다.
온셋·RMS는 곡 전체 기준값(top_db 등)이 있어 늘 새로 계산한다(전체 계산에 1초 남짓).

후처리(옥타브 오류·튐 보정)는 결과를 캐시한 뒤 GUI 쪽 ``amp.engine.pitch_post`` 가 한다.
그래서 후처리 방법을 바꿔도 pYIN을 다시 돌릴 필요가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

PITCH_SR = 16000
HOP = 160
FRAME_LENGTH = 1024
FRAME_RATE = PITCH_SR / HOP  # 100 프레임/초
FMIN_HZ = 55.0  # A1
FMAX_HZ = 1400.0  # F6 근처
CHUNK_S = 30.0
PAD_S = 1.0
ONSET_N_FFT = 512  # 32ms
ONSET_N_MELS = 40
FINE_FRAME_LENGTH = 320  # 20ms


@dataclass
class PitchFrames:
    f0_hz: np.ndarray  # float32, 무성 = NaN
    voiced_prob: np.ndarray  # float32, 0~1
    onset: np.ndarray  # float32, 온셋 강도
    rms: np.ndarray  # float32, 프레임 RMS (64ms 창)
    frame_rate: float = FRAME_RATE
    start_time: float = 0.0
    rms_fine: np.ndarray | None = None  # float32, 20ms 창 RMS. 피치 버전 1 캐시에는 없다

    def save(self, path) -> None:
        extra = {"rms_fine": self.rms_fine} if self.rms_fine is not None else {}
        np.savez_compressed(
            path, f0_hz=self.f0_hz, voiced_prob=self.voiced_prob, onset=self.onset, rms=self.rms,
            frame_rate=np.float64(self.frame_rate), start_time=np.float64(self.start_time), **extra,
        )

    @classmethod
    def load(cls, path) -> "PitchFrames":
        with np.load(path, allow_pickle=False) as z:
            fine = z["rms_fine"] if "rms_fine" in z.files else None
            return cls(z["f0_hz"], z["voiced_prob"], z["onset"], z["rms"], float(z["frame_rate"]), float(z["start_time"]),
                       fine)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).ravel()
    if a.shape[0] >= n:
        return a[:n]
    return np.concatenate([a, np.zeros(n - a.shape[0], dtype=np.float32)])


def to_pitch_sr(mono: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = np.asarray(mono, dtype=np.float32)
    if sr != PITCH_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=PITCH_SR, res_type="soxr_hq").astype(np.float32)
    return y


@dataclass
class ReuseSource:
    """다시 쓸 수 있는 이전 피치 결과: 그 결과를 만든 16kHz 신호와 f0·확률."""
    y16: np.ndarray
    f0_hz: np.ndarray
    voiced_prob: np.ndarray
    label: str = ""


def track_pitch(
    mono: np.ndarray,
    sr: int,
    progress: Callable[[float, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    fmin: float = FMIN_HZ,
    fmax: float = FMAX_HZ,
    reuse: list[ReuseSource] | None = None,
    stats: dict | None = None,
) -> PitchFrames:
    import librosa

    y = to_pitch_sr(mono, sr)
    n_frames = 1 + len(y) // HOP
    f0 = np.full(n_frames, np.nan, dtype=np.float32)
    prob = np.zeros(n_frames, dtype=np.float32)

    chunk = int(CHUNK_S * PITCH_SR) // HOP * HOP
    pad = int(PAD_S * PITCH_SR) // HOP * HOP
    starts = list(range(0, len(y), chunk)) if len(y) >= FRAME_LENGTH else []
    for k, start in enumerate(starts):
        if is_cancelled and is_cancelled():
            from amp.core.errors import AnalysisCancelled

            raise AnalysisCancelled("피치 분석을 취소했습니다.")
        a = max(0, start - pad)
        b = min(len(y), start + chunk + pad)
        seg = y[a:b]
        if float(np.max(np.abs(seg))) < 1e-6:  # 완전 무음 청크는 계산하지 않는다
            continue
        g0 = start // HOP
        g1 = n_frames if k == len(starts) - 1 else (start + chunk) // HOP
        src = next((r for r in reuse or () if len(r.y16) == len(y) and len(r.f0_hz) == n_frames
                    and np.array_equal(r.y16[a:b], seg)), None)
        if src is not None:  # 이 청크의 입력이 이전 결과와 같다: 같은 결과를 옮긴다
            f0[g0:g1] = src.f0_hz[g0:g1]
            prob[g0:g1] = src.voiced_prob[g0:g1]
            if stats is not None:
                stats["reused"] = stats.get("reused", 0) + 1
            if progress:
                progress((k + 1) / len(starts) * 0.9, "피치 추출 중 (바뀌지 않은 구간은 이전 결과 사용)")
            continue
        if stats is not None:
            stats["computed"] = stats.get("computed", 0) + 1
        f, voiced, vprob = librosa.pyin(
            seg, fmin=fmin, fmax=fmax, sr=PITCH_SR, frame_length=FRAME_LENGTH, hop_length=HOP, center=True
        )
        j0 = g0 - a // HOP
        take = slice(j0, j0 + (g1 - g0))
        fs = np.where(voiced[take], f[take], np.nan)
        f0[g0:g0 + len(fs)] = fs
        prob[g0:g0 + len(fs)] = vprob[take]
        if progress:
            progress((k + 1) / len(starts) * 0.9, "피치 추출 중 (pYIN)")

    onset = onset_envelope(y) if len(y) >= FRAME_LENGTH else []
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP, center=True)[0] if len(y) else []
    fine = librosa.feature.rms(y=y, frame_length=FINE_FRAME_LENGTH, hop_length=HOP, center=True)[0] if len(y) else []
    if progress:
        progress(1.0, "피치 추출 완료")
    return PitchFrames(f0, prob, _fit(onset, n_frames), _fit(rms, n_frames), rms_fine=_fit(fine, n_frames))


def onset_envelope(y16: np.ndarray) -> np.ndarray:
    """16kHz 신호의 온셋 강도. 프레임 i의 값은 i/100초 전후 스펙트럼 변화다."""
    import librosa

    return librosa.onset.onset_strength(y=y16, sr=PITCH_SR, hop_length=HOP, n_fft=ONSET_N_FFT,
                                        n_mels=ONSET_N_MELS, center=False)
