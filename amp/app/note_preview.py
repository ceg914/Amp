"""음표 미리 듣기: 본 재생기와 따로 쓰는 작은 출력(QAudioSink). 재생 중에도 겹쳐 들을 수 있다.

- 새 음을 누르면 이전 소리는 10ms 동안 줄이며 끝낸다. 빠르게 옮겨도 소리가 쌓이지 않는다(한 번에 하나).
- 선택한 여러 음표는 원래 간격대로 이어서 들려준다.
- 조용하면 2초 뒤 출력을 닫는다. 프로젝트를 닫을 때 shutdown()으로 정리한다.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PySide6 import QtMultimedia
from PySide6.QtCore import QIODevice, QObject, QTimer
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from amp.audio.pcm import SampleFormat
from amp.audio.synth import Instrument, SoftSynth, render_sequence

log = logging.getLogger(__name__)
_QtAudio = getattr(QtMultimedia, "QtAudio", None) or QtMultimedia.QAudio
PREVIEW_S = 0.45  # 클릭 한 번에 들려주는 길이
SWAP_FADE_S = 0.01
IDLE_CLOSE_MS = 2000
_FORMATS = {QAudioFormat.SampleFormat.Int16: SampleFormat.INT16, QAudioFormat.SampleFormat.Int32: SampleFormat.INT32,
            QAudioFormat.SampleFormat.Float: SampleFormat.FLOAT32}


def _quantize(block: np.ndarray, fmt: SampleFormat) -> bytes:
    block = np.clip(block, -1.0, 1.0)
    if fmt is SampleFormat.INT16:
        return (block * 32767.0).astype("<i2").tobytes()
    if fmt is SampleFormat.INT32:
        return (block.astype(np.float64) * 2147483647.0).astype("<i4").tobytes()
    return block.astype("<f4").tobytes()


class _Voice(QIODevice):
    """지금 소리 하나를 내보낸다. 소리가 끝나면 0을 낸다(출력은 열린 채로)."""

    def __init__(self, channels: int, fmt: SampleFormat, parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self.fmt = fmt
        self._buf = np.zeros(0, np.float32)
        self._pos = 0
        self._lock = threading.Lock()
        self.gain = 0.6

    def set_sound(self, mono: np.ndarray, fade: int) -> None:
        with self._lock:
            rest = self._buf[self._pos:self._pos + fade]
            new = mono.astype(np.float32).copy()
            if rest.size:  # 이전 소리는 짧게 줄여 새 소리 앞에 섞는다 (겹쳐 쌓이지 않는다)
                rest = rest * np.linspace(1.0, 0.0, rest.size, dtype=np.float32)
                k = min(rest.size, new.size)
                new[:k] += rest[:k]
            self._buf, self._pos = new, 0

    def silence(self, fade: int) -> None:
        self.set_sound(np.zeros(max(fade, 1), np.float32), fade)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pos < self._buf.size

    def readData(self, maxlen: int) -> bytes:  # noqa: N802
        bpf = self.channels * self.fmt.bytes_per_sample
        n = maxlen // bpf
        with self._lock:
            chunk = self._buf[self._pos:self._pos + n]
            self._pos += chunk.size
            gain = self.gain
        out = np.zeros(n, np.float32)
        out[:chunk.size] = chunk * gain
        return _quantize(np.repeat(out[:, None], self.channels, axis=1), self.fmt)

    def writeData(self, data) -> int:  # noqa: N802
        return -1

    def bytesAvailable(self) -> int:  # noqa: N802
        return 1 << 16

    def isSequential(self) -> bool:  # noqa: N802
        return True


class NotePreviewer(QObject):
    def __init__(self, parent=None, instrument: Instrument | None = None) -> None:
        super().__init__(parent)
        self.instrument = instrument or SoftSynth()
        self.enabled = True
        self._volume = 0.6
        self._sink: QAudioSink | None = None
        self._voice: _Voice | None = None
        self._rate = 44100
        self._idle = QTimer(self)
        self._idle.setInterval(IDLE_CLOSE_MS)
        self._idle.timeout.connect(self._close_if_idle)
        self.last_sound: np.ndarray | None = None  # 시험·진단용: 마지막으로 내보낸 소리
        self.triggers = 0

    @property
    def volume(self) -> float:
        return self._volume

    def set_volume(self, v: float) -> None:
        self._volume = min(max(float(v), 0.0), 1.0)
        if self._voice is not None:
            self._voice.gain = self._volume

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self.stop()

    def play_note(self, pitch: float, duration_s: float = PREVIEW_S) -> bool:
        if not self.enabled:
            return False
        return self._play(render_sequence([(0.0, duration_s, pitch)], self._rate, self.instrument, gain=0.8))

    def play_sequence(self, events: list[tuple[float, float, float]]) -> bool:
        """[(시작초, 길이초, 음높이)]를 원래 간격대로 (최대 60초)."""
        if not self.enabled or not events:
            return False
        return self._play(render_sequence(events, self._rate, self.instrument, gain=0.8))

    def stop(self) -> None:
        if self._voice is not None:
            self._voice.silence(int(SWAP_FADE_S * self._rate))

    def shutdown(self) -> None:
        self._idle.stop()
        sink, self._sink, self._voice = self._sink, None, None
        if sink is not None:
            sink.stop()
            sink.deleteLater()

    def _play(self, mono: np.ndarray) -> bool:
        self.last_sound = mono
        self.triggers += 1
        if not self._ensure_sink():
            return False
        self._voice.set_sound(mono, int(SWAP_FADE_S * self._rate))
        self._idle.start()
        return True

    def _ensure_sink(self) -> bool:
        if self._sink is not None:
            return True
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            return False
        fmt = QAudioFormat()
        fmt.setSampleRate(44100)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
            if fmt.sampleFormat() not in _FORMATS:
                return False
        self._rate = fmt.sampleRate()
        sink = QAudioSink(device, fmt, self)
        voice = _Voice(fmt.channelCount(), _FORMATS[fmt.sampleFormat()], parent=sink)
        voice.gain = self._volume
        voice.open(QIODevice.OpenModeFlag.ReadOnly)
        sink.setBufferSize(int(fmt.bytesForDuration(60_000)))  # 60ms: 누르고 바로 들리게
        sink.start(voice)
        if sink.error() != _QtAudio.Error.NoError:
            sink.deleteLater()
            log.warning("미리 듣기 출력을 열지 못했습니다: %s", sink.error())
            return False
        self._sink, self._voice = sink, voice
        return True

    def _close_if_idle(self) -> None:
        if self._voice is not None and not self._voice.busy:
            self.shutdown()
