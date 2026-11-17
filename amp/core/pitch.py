"""음높이 단위 변환 도우미."""

from __future__ import annotations

import numpy as np

A4_HZ = 440.0
A4_MIDI = 69
MIDI_MIN = 0
MIDI_MAX = 127

_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_hz(midi):
    """MIDI 음 번호(실수 허용)를 Hz로 바꾼다."""
    return A4_HZ * np.power(2.0, (np.asarray(midi, dtype=np.float64) - A4_MIDI) / 12.0)


def hz_to_midi(hz):
    """Hz를 MIDI 음 번호(실수)로 바꾼다. 0 이하나 NaN은 NaN이 된다."""
    hz = np.asarray(hz, dtype=np.float64)
    out = np.full(hz.shape, np.nan)
    valid = np.isfinite(hz) & (hz > 0)
    out[valid] = A4_MIDI + 12.0 * np.log2(hz[valid] / A4_HZ)
    return out if out.shape else float(out)


def note_name(midi: int) -> str:
    """60 → 'C4' 형식의 음 이름."""
    midi = int(midi)
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def is_black_key(midi: int) -> bool:
    return (int(midi) % 12) in (1, 3, 6, 8, 10)
