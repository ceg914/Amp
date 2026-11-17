"""검증용 합성 음원 (자동 테스트·벤치마크·검증 도구 자체 확인).

실제 노래 녹음을 저장소에 넣을 수 없어서, 성대 펄스열 + 포먼트 공명기로 모음 비슷한 소리를 만든다.
pYIN은 이 소리를 0.8 이상의 유성 확률로 잡고, htdemucs도 보컬로 분리한다(개발 중 확인).
실제 노래의 분리·채보 정확도를 보장하는 시험이 아니다.
"""

import numpy as np
from scipy.signal import lfilter

SR = 44100
VOWELS = [(730, 1090, 2440, 3400), (270, 2290, 3010, 3500), (570, 840, 2410, 3400), (300, 870, 2240, 3400)]


def formant_voice(f0: np.ndarray, sr: int = SR, seed: int = 0, vowel=VOWELS[0]) -> np.ndarray:
    """f0(Hz, 0이면 무음) 배열로 모음 소리를 만든다."""
    rng = np.random.default_rng(seed)
    jitter = 1 + 0.004 * np.convolve(rng.standard_normal(len(f0)), np.ones(200) / 200, mode="same")
    saw = np.cumsum(f0 * jitter / sr) % 1.0
    src = np.where(f0 > 0, saw - 0.5, 0.0)
    out = lfilter([1], [1, -0.97], np.diff(src, prepend=0))
    for F, B in zip(vowel, (90, 110, 160, 250)):
        r = np.exp(-np.pi * B / sr)
        th = 2 * np.pi * F / sr
        out = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], out)
    voiced_rms = np.sqrt(np.mean(out[f0 > 0] ** 2)) if np.any(f0 > 0) else 0.0
    breath = lfilter([1], [1, -0.5], rng.standard_normal(len(f0)))
    out = out + breath / (np.std(breath) + 1e-12) * 0.05 * voiced_rms * (f0 > 0)  # 약 -26dB 숨소리
    out = out / (np.abs(out).max() + 1e-12)
    return out.astype(np.float32)


def melody_f0(melody, duration: float, sr: int = SR, vibrato_cents: float = 0.0, vibrato_hz: float = 5.5,
              glide_s: float = 0.0) -> np.ndarray:
    """[(시작초, 끝초, MIDI), ...] → f0 배열. glide_s > 0이면 붙어 있는 음 사이를 미끄러지듯 잇는다."""
    t = np.arange(int(duration * sr)) / sr
    midi = np.full_like(t, np.nan)
    for a, b, m in melody:
        midi[(t >= a) & (t < b)] = m
    if glide_s > 0:
        for (a0, b0, m0), (a1, b1, m1) in zip(melody, melody[1:]):
            if abs(a1 - b0) < 1e-9:
                sel = (t >= b0 - glide_s / 2) & (t < b0 + glide_s / 2)
                midi[sel] = np.linspace(m0, m1, sel.sum())
    if vibrato_cents:
        midi = midi + vibrato_cents / 100 * np.sin(2 * np.pi * vibrato_hz * t)
    f0 = np.where(np.isfinite(midi), 440 * 2 ** ((np.nan_to_num(midi) - 69) / 12), 0.0)
    return f0


def envelope(melody, duration: float, sr: int = SR, attack: float = 0.03, release: float = 0.05) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    env = np.zeros_like(t)
    for a, b, _m in melody:
        sel = (t >= a) & (t < b)
        tt = t[sel] - a
        env[sel] = np.clip(tt / attack, 0, 1) * np.clip((b - a - tt) / release, 0, 1)
    return env


def sung(melody, duration: float, sr: int = SR, attack: float = 0.03, release: float = 0.05,
         same_vowel: bool = False, **kw) -> np.ndarray:
    """음마다 모음을 바꿔 부른다 (숨소리 포함). same_vowel이면 모든 음을 같은 모음으로."""
    f0 = melody_f0(melody, duration, sr, **kw)
    t = np.arange(len(f0)) / sr
    out = np.zeros(len(f0), dtype=np.float32)
    for k, (a, b, _m) in enumerate(melody):
        sel = (t >= a - 0.06) & (t < b + 0.06)  # 글라이드 구간까지 포함
        vowel = VOWELS[0] if same_vowel else VOWELS[k % len(VOWELS)]
        out += np.where(sel, formant_voice(np.where(sel, f0, 0.0), sr, seed=k, vowel=vowel), 0)
    env = envelope(melody, duration, sr, attack, release)
    return 0.5 * out / (np.abs(out).max() + 1e-12) * env


def ref_midi_frames(melody, duration: float, frame_rate: float = 100.0, **kw) -> np.ndarray:
    """프레임 격자(i/frame_rate 초)에서 정답 MIDI (무음 = NaN). sung()과 같은 인자를 받는다."""
    f0 = melody_f0(melody, duration, int(frame_rate), **{k: v for k, v in kw.items() if k in ("vibrato_cents", "vibrato_hz", "glide_s")})
    return np.where(f0 > 0, 69 + 12 * np.log2(np.where(f0 > 0, f0, 1) / 440.0), np.nan)


def breath_noise(duration: float, spans, level_db: float = -30.0, sr: int = SR, seed: int = 3) -> np.ndarray:
    """쉬는 구간의 숨소리(대역 잡음)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    x = lfilter([1], [1, -0.9], rng.standard_normal(len(t)))
    x = x / (np.abs(x).max() + 1e-12) * 10 ** (level_db / 20)
    mask = np.zeros(len(t))
    for a, b in spans:
        sel = (t >= a) & (t < b)
        mask[sel] = np.hanning(int(sel.sum()) + 2)[1:-1]
    return (x * mask).astype(np.float32)


def backing(duration: float, sr: int = SR, bpm: float = 120.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    out = 0.3 * np.sin(2 * np.pi * 55 * t) + sum(0.06 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    for b in np.arange(0, duration, 60.0 / bpm):
        i = int(b * sr)
        n = min(3000, len(t) - i)
        out[i:i + n] += rng.standard_normal(n) * np.exp(-np.arange(n) / 400) * 0.4
    return out.astype(np.float32)


def clicks(duration: float, bpm: float, sr: int = SR, offset: float = 0.25) -> np.ndarray:
    x = np.zeros(int(duration * sr), dtype=np.float32)
    for b in np.arange(offset, duration, 60.0 / bpm):
        i = int(b * sr)
        n = min(600, len(x) - i)
        x[i:i + n] = np.hanning(1200)[600:600 + n] * 0.8
    return x
