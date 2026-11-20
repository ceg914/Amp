"""음표 미리 듣기용 내장 합성음. Qt 비의존.

가수 음색이 아니라 음높이를 확인하는 소리다. 사인파에 약한 2·3배음을 더하고, 짧은 어택·릴리즈(올린 코사인)로
시작·끝의 딸깍 소리를 막는다. 음색은 Instrument 프로토콜 뒤에 두어 나중에 SoundFont 같은 다른 음원으로 바꿀 수 있다.
"""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

ATTACK_S = 0.008
RELEASE_S = 0.04


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((float(pitch) - 69.0) / 12.0)


class Instrument(Protocol):
    name: str

    def render(self, pitch: float, duration_s: float, sr: int, velocity: float = 0.8) -> np.ndarray:
        """모노 float32. 길이 = duration_s + 릴리즈."""
        ...


class SoftSynth:
    """사인 + 약한 배음. 높은 음일수록 배음을 줄여 날카롭지 않게 한다."""

    name = "내장 합성음 (사인 + 배음)"

    def __init__(self, harmonics: tuple[float, ...] = (1.0, 0.22, 0.07)) -> None:
        self.harmonics = harmonics

    def render(self, pitch: float, duration_s: float, sr: int, velocity: float = 0.8) -> np.ndarray:
        dur = max(float(duration_s), 0.02)
        n = int(round((dur + RELEASE_S) * sr))
        t = np.arange(n) / sr
        f = midi_to_hz(pitch)
        y = np.zeros(n)
        for k, amp in enumerate(self.harmonics, start=1):
            if f * k >= sr * 0.45:
                break
            y += amp * (0.6 if f * k > 2000 else 1.0) * np.sin(2 * np.pi * f * k * t)
        y *= velocity / sum(self.harmonics)
        return (y * envelope(n, int(round(dur * sr)), sr)).astype(np.float32)


def envelope(n: int, sustain_end: int, sr: int) -> np.ndarray:
    """0에서 올라가 sustain_end부터 RELEASE_S 동안 0으로 내려가는 이득 (올린 코사인)."""
    g = np.ones(n)
    a = max(1, int(ATTACK_S * sr))
    a = min(a, max(1, sustain_end // 2))
    g[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    r = n - sustain_end
    if r > 0:
        g[sustain_end:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
    return g


def render_sequence(events: Iterable[tuple[float, float, float]], sr: int, instrument: Instrument | None = None,
                    gain: float = 0.5, max_s: float = 60.0) -> np.ndarray:
    """[(시작초, 길이초, MIDI 음높이)] → 모노 float32 (첫 음 시작 = 0). 음 사이 간격도 그대로 둔다."""
    events = sorted(events)
    if not events:
        return np.zeros(0, np.float32)
    inst = instrument or SoftSynth()
    t0 = events[0][0]
    total = min(max(s - t0 + d for s, d, _p in events) + RELEASE_S, max_s)
    out = np.zeros(int(round(total * sr)) + 1, np.float32)
    for s, d, p in events:
        i = int(round((s - t0) * sr))
        if i >= out.size:
            break
        y = inst.render(p, d, sr)
        j = min(out.size, i + y.size)
        out[i:j] += y[: j - i]
    return np.clip(out * gain, -1.0, 1.0)


def render_track(notes: Iterable, total_frames: int, sr: int, channels: int = 2,
                 instrument: Instrument | None = None, gain: float = 0.35) -> np.ndarray:
    """음표 전체를 원본과 같은 시간축의 트랙으로 (frames, channels). 재생기에서 보컬과 함께 비교해 듣는 데 쓴다."""
    inst = instrument or SoftSynth()
    out = np.zeros(int(total_frames), np.float32)
    for n in notes:
        i = int(round(n.start * sr))
        if i >= out.size or n.end <= n.start:
            continue
        y = inst.render(n.pitch, n.end - n.start, sr)
        j = min(out.size, i + y.size)
        out[i:j] += y[: j - i]
    out = np.clip(out * gain, -1.0, 1.0)
    return np.repeat(out[:, None], channels, axis=1) if channels > 1 else out[:, None]
