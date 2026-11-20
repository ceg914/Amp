"""메모리에 올린 오디오를 QAudioSink로 재생한다.

트랙(원본·보컬·반주)을 여러 개 올려 트랙별 음량·음소거·솔로로 섞어 들을 수 있다.
모든 트랙은 원본과 같은 시간축이라 트랙을 바꾸거나 섞어도 재생 위치가 그대로다.
QMediaPlayer 대신 QAudioSink를 쓰는 이유도 이것이다 (numpy 배열을 직접 섞어 흘려보낸다).
"""

from __future__ import annotations

import logging
import threading
from enum import Enum

from PySide6 import QtMultimedia
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from amp.audio import windows_output
from amp.audio.data import AudioData
from amp.audio.pcm import PcmMixer, SampleFormat
from amp.audio.stream import PlaybackStream

log = logging.getLogger(__name__)

# PySide6 6.7 이후 QAudio 네임스페이스가 QtAudio로 바뀌었다
_QtAudio = getattr(QtMultimedia, "QtAudio", None) or QtMultimedia.QAudio

_FORMAT_MAP = {
    QAudioFormat.SampleFormat.Int16: SampleFormat.INT16,
    QAudioFormat.SampleFormat.Int32: SampleFormat.INT32,
    QAudioFormat.SampleFormat.Float: SampleFormat.FLOAT32,
}
BUFFER_USEC = 200_000  # 장치 버퍼 길이. UI가 잠깐 바빠도 소리가 끊기지 않을 만큼 둔다.
SLOW_WINDOW_S = 60.0  # 반복 구간 없이 느리게 재생할 때 한 번에 늘리는 길이 (곡 전체를 미리 늘리지 않는다)


class PlaybackState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class PcmDevice(QIODevice):
    """PcmConverter를 QIODevice로 감싼다. 오디오 스레드에서 읽힐 수 있어 잠금을 쓴다.

    재생을 멈출 때 장치를 닫지 않고 ``finish()`` 로 '더 줄 데이터 없음' 상태로만 바꾼다.
    Windows의 Qt 오디오(링 버퍼를 채우는 별도 스레드)는 sink.stop() 뒤에도 잠깐 장치를 읽을 수 있어,
    닫아 버리면 "QIODevice::read (PcmDevice): device not open" 경고가 나고(사용자 로그에서 확인),
    부모 없는 장치를 파이썬이 그 순간 지우면 해제된 객체를 읽을 위험도 있다. 장치는 sink의 자식으로 두어 sink와 함께 지운다.
    """

    def __init__(self, converter, start_frame: int = 0, parent: QObject | None = None) -> None:
        """converter는 read(start, count)·total_out_frames·bytes_per_frame를 가진 객체 (PcmConverter, PcmMixer)."""
        super().__init__(parent)
        self._conv = converter
        self._frame = start_frame
        self._lock = threading.Lock()
        self.at_end = False
        self.finished = False

    def finish(self) -> None:
        with self._lock:
            self.finished = True

    def readData(self, maxlen: int) -> bytes:  # noqa: N802 - Qt API
        bpf = self._conv.bytes_per_frame
        with self._lock:
            if self.finished:
                return b""
            data = self._conv.read(self._frame, maxlen // bpf)
            self._frame += len(data) // bpf
            if self._frame >= self._conv.total_out_frames:
                self.at_end = True
        return data

    def writeData(self, data) -> int:  # noqa: N802
        return -1

    def bytesAvailable(self) -> int:  # noqa: N802
        with self._lock:
            if self.finished:
                return 0
            remaining = (self._conv.total_out_frames - self._frame) * self._conv.bytes_per_frame
        # 구간 반복은 끝이 없다(LOOP_FRAMES). 아주 큰 값을 알리면 Qt 출력이 멈춰서 상한을 둔다 (이 환경에서 확인)
        return min(remaining, 1 << 30) + super().bytesAvailable()

    def isSequential(self) -> bool:  # noqa: N802
        return True


class AudioPlayer(QObject):
    """여러 트랙 재생 + 구간 반복 + 음높이 유지 느린 재생.

    느린 재생은 늘린 오디오가 필요하다. 없으면 stretchRequested(시작, 끝, 속도)를 내고 기다린다.
    바깥(ListeningController)이 작업 스레드에서 늘린 뒤 set_stretched()로 넘기면 그때 재생한다.
    재생 위치는 언제나 원본 시간이다.
    """

    positionChanged = Signal(float)
    stateChanged = Signal(object)  # PlaybackState
    errorOccurred = Signal(str)
    stretchRequested = Signal(float, float, float)  # 구간 시작, 끝(초), 속도
    playbackChanged = Signal()  # 반복 구간·반복 켜기·속도가 바뀜

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._audio: AudioData | None = None  # 기준 트랙 (원본). 길이·샘플레이트의 기준
        self._tracks: dict[str, AudioData] = {}
        self._track_state: dict[str, dict] = {}
        self._mixer: PcmMixer | None = None
        self._sink: QAudioSink | None = None
        self._device: PcmDevice | None = None
        self._state = PlaybackState.STOPPED
        self._pos = 0.0  # 재생 중이 아닐 때의 위치
        self._play_from = 0.0  # 정지하면 돌아갈 위치
        self._stream: PlaybackStream | None = None
        self._start_frame = 0  # 장치를 시작한 스트림 프레임
        self._region: tuple[float, float] | None = None  # 반복 구간 (원본 시간)
        self._loop_on = False
        self._speed = 1.0
        self._stretched: tuple[tuple[float, float, float, int], dict] | None = None  # ((a, b, speed, 버전), 트랙)
        self._tracks_version = 0  # 트랙이 바뀌면 늘린 오디오를 버린다
        self._pending_play = False
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._sink_device_id = b""  # 지금 재생 출력이 쓰는 장치 id
        self.device_switches = 0  # 출력 장치가 바뀌어 다시 연 횟수 (진단용)
        # 재생 중 기본 출력 장치가 바뀌거나(헤드폰 연결·해제) 쓰던 장치가 사라지면 새 장치로 다시 연다
        self._devices = QMediaDevices(self)
        self._devices.audioOutputsChanged.connect(self._on_outputs_changed)

    # --- 조회 --------------------------------------------------------------

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def duration(self) -> float:
        return self._audio.duration if self._audio else 0.0

    @property
    def has_audio(self) -> bool:
        return self._audio is not None

    def position(self) -> float:
        if self._state is PlaybackState.PLAYING and self._sink is not None and self._stream is not None:
            n = self._start_frame + int(round(self._sink.processedUSecs() * self._stream.out_rate / 1e6))
            return min(self._stream.time_at(n), self.duration)
        return self._pos

    # --- 반복·속도 -----------------------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def loop_region(self) -> tuple[float, float] | None:
        return self._region

    @property
    def loop_enabled(self) -> bool:
        return self._loop_on and self._region is not None

    @property
    def waiting_for_stretch(self) -> bool:
        return self._pending_play

    def set_loop_region(self, region: tuple[float, float] | None) -> None:
        if region is not None:
            a, b = sorted((max(0.0, float(region[0])), min(float(region[1]), self.duration or float(region[1]))))
            region = (a, b) if b - a >= 0.05 else None
        if region == self._region:
            return
        self._region = region
        self._restart_if_playing()
        self.playbackChanged.emit()

    def set_loop_enabled(self, on: bool) -> None:
        if bool(on) == self._loop_on:
            return
        self._loop_on = bool(on)
        self._restart_if_playing()
        self.playbackChanged.emit()

    def set_speed(self, speed: float) -> None:
        speed = min(max(float(speed), 0.5), 1.0)
        if abs(speed - self._speed) < 1e-6:
            return
        self._speed = speed
        self._restart_if_playing()
        self.playbackChanged.emit()

    def play_region(self) -> tuple[float, float] | None:
        """지금 재생에 쓰는 구간: 반복이 켜져 있으면 반복 구간, 느린 재생이면 위치부터 SLOW_WINDOW_S초, 아니면 None."""
        if self.loop_enabled:
            return self._region
        if self._speed < 1.0 and self._audio is not None:
            pos = self.position()
            if self._stretched is not None:
                (a, b, sp, ver), _t = self._stretched
                if abs(sp - self._speed) < 1e-6 and ver == self._tracks_version and a <= pos < b - 0.05:
                    return (a, b)
            return (pos, min(pos + SLOW_WINDOW_S, self.duration))
        return None

    def stretch_inputs(self) -> dict[str, AudioData]:
        return dict(self._tracks) or ({"original": self._audio} if self._audio is not None else {})

    @property
    def tracks_version(self) -> int:
        return self._tracks_version

    def set_stretched(self, a: float, b: float, speed: float, tracks: dict, version: int | None = None) -> None:
        """늘린 오디오를 넘긴다(트랙 이름 → (frames, ch) 배열). 기다리던 재생이 있으면 시작한다."""
        if version is not None and version != self._tracks_version:
            return  # 계산하는 동안 트랙이 바뀜: 버린다
        self._stretched = ((float(a), float(b), float(speed), self._tracks_version), dict(tracks))
        if self._pending_play:
            self._pending_play = False
            self.play()

    def _has_stretch(self, region: tuple[float, float]) -> bool:
        if self._stretched is None:
            return False
        (a, b, sp, ver), _t = self._stretched
        return abs(a - region[0]) < 1e-6 and abs(b - region[1]) < 1e-6 and abs(sp - self._speed) < 1e-6 \
            and ver == self._tracks_version

    def _restart_if_playing(self) -> None:
        if self._state is PlaybackState.PAUSED:  # 일시정지 중이면 다음 재생 때 새 설정으로 연다
            pos = self._pos
            self._teardown()
            self._pos = self._play_from = pos
            self._set_state(PlaybackState.STOPPED)
            return
        if self._state is PlaybackState.PLAYING:
            pos = self.position()
            self._teardown()
            self._pos = pos
            if self._start_sink(pos):
                self._timer.start()
            elif not self._pending_play:
                self._set_state(PlaybackState.STOPPED)

    @staticmethod
    def output_available() -> bool:
        return not QMediaDevices.defaultAudioOutput().isNull()

    _os_lookup_failed = False

    @classmethod
    def default_output(cls):
        """지금 쓸 기본 출력 장치.

        Windows에서는 Core Audio에 직접 물은 기본 장치와 id가 같은 Qt 장치를 쓴다. Qt의 '기본 장치' 표시는 장치를 바꾼 뒤
        Windows와 달라질 수 있어서다(7.4절). Windows가 아니거나, 묻지 못했거나, 그 id가 Qt 목록에 없으면 Qt의 기본 장치를 쓴다.
        """
        qt_default = QMediaDevices.defaultAudioOutput()
        if cls._os_lookup_failed:
            return qt_default
        try:
            os_id = windows_output.default_output_id()
        except Exception as e:  # noqa: BLE001 - 조회 실패는 Qt 기본 장치로 물러난다
            cls._os_lookup_failed = True
            log.warning("Windows 기본 출력 장치를 묻지 못해 Qt의 기본 장치를 씁니다: %s", e)
            return qt_default
        if not os_id:
            return qt_default
        want = os_id.encode("utf-8")
        for d in QMediaDevices.audioOutputs():
            if bytes(d.id()) == want:
                if qt_default.isNull() or bytes(qt_default.id()) != want:
                    log.info("Qt 기본 장치(%s) 대신 Windows 기본 장치(%s)를 씁니다",
                             "없음" if qt_default.isNull() else qt_default.description(), d.description())
                return d
        return qt_default

    # --- 제어 --------------------------------------------------------------

    def set_audio(self, audio: AudioData | None) -> None:
        """원본 하나만 올린다 (다른 트랙은 지운다). 위치를 처음으로 돌린다. 반복 구간·늘린 오디오도 지운다."""
        self._teardown()
        self._audio = audio
        self._region = None
        self._loop_on = False
        self._stretched = None
        self._pending_play = False
        self._tracks_version += 1
        self._tracks = {"original": audio} if audio is not None else {}
        self._pos = self._play_from = 0.0
        self._set_state(PlaybackState.STOPPED)
        self.positionChanged.emit(0.0)

    @property
    def tracks(self) -> dict[str, AudioData]:
        return dict(self._tracks)

    def set_track(self, name: str, audio: AudioData | None) -> None:
        """트랙을 더하거나(분리 결과) 뺀다. 재생 중이면 같은 위치에서 이어서 재생한다."""
        if audio is not None and self._audio is not None:
            if audio.frames != self._audio.frames or audio.sample_rate != self._audio.sample_rate:
                self.errorOccurred.emit(f"{name} 트랙의 길이·샘플레이트가 원본과 달라 재생 목록에 넣지 않았습니다.")
                return
        playing = self._state is PlaybackState.PLAYING
        pos = self.position()
        if playing:
            self._teardown()
        if audio is None:
            self._tracks.pop(name, None)
        else:
            self._tracks[name] = audio
        self._tracks_version += 1  # 늘린 오디오는 새 트랙으로 다시 만든다
        self._stretched = None
        if playing:
            self._pos = pos
            if self._start_sink(pos):
                self._timer.start()
            else:
                self._set_state(PlaybackState.STOPPED)

    def track_state(self, name: str) -> dict:
        return dict(self._track_state.get(name, {"gain": 1.0, "mute": False, "solo": False}))

    def set_track_state(self, name: str, *, gain: float | None = None, mute: bool | None = None,
                        solo: bool | None = None) -> None:
        st = self.track_state(name)
        if gain is not None:
            st["gain"] = float(gain)
        if mute is not None:
            st["mute"] = bool(mute)
        if solo is not None:
            st["solo"] = bool(solo)
        self._track_state[name] = st
        if self._mixer is not None:
            self._mixer.set_state(name, gain=gain, mute=mute, solo=solo)

    def toggle(self) -> None:
        if self._state is PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self._audio is None:
            return
        if self._state is PlaybackState.PAUSED and self._sink is not None:
            self._sink.resume()
            self._set_state(PlaybackState.PLAYING)
            self._timer.start()
            return
        if self._pos >= self.duration - 0.01:
            self._pos = 0.0
        if self.loop_enabled and not (self._region[0] <= self._pos < self._region[1]):
            self._pos = self._region[0]  # 반복 중에는 구간 안에서만 재생한다
        self._play_from = self._pos
        if self._start_sink(self._pos):
            self._set_state(PlaybackState.PLAYING)
            self._timer.start()

    def pause(self) -> None:
        if self._state is not PlaybackState.PLAYING or self._sink is None:
            return
        self._pos = self.position()
        self._sink.suspend()
        self._timer.stop()
        self._set_state(PlaybackState.PAUSED)
        self.positionChanged.emit(self._pos)

    def stop(self) -> None:
        """정지하고 마지막으로 재생을 시작한 위치로 돌아간다."""
        self._pending_play = False
        self._teardown()
        self._pos = self._play_from
        self._set_state(PlaybackState.STOPPED)
        self.positionChanged.emit(self._pos)

    def seek(self, seconds: float) -> None:
        seconds = min(max(seconds, 0.0), self.duration)
        if self.loop_enabled and not (self._region[0] <= seconds < self._region[1]):
            seconds = self._region[0]  # 반복 중 구간 밖으로 가면 구간 시작으로
        if self._state is PlaybackState.PLAYING:
            self._teardown()
            self._pos = seconds
            if self._start_sink(seconds):
                self._timer.start()  # _teardown이 멈춘 위치 갱신 타이머를 다시 켠다
            else:
                self._set_state(PlaybackState.STOPPED)
        else:
            if self._state is PlaybackState.PAUSED:
                self._teardown()
                self._set_state(PlaybackState.STOPPED)
            self._pos = seconds
            self._play_from = seconds
        self.positionChanged.emit(seconds)

    # --- 내부 --------------------------------------------------------------

    def _choose_format(self, device) -> tuple[QAudioFormat, SampleFormat] | None:
        assert self._audio is not None
        fmt = QAudioFormat()
        fmt.setSampleRate(self._audio.sample_rate)
        fmt.setChannelCount(min(self._audio.channels, 2))
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isFormatSupported(fmt):
            return fmt, SampleFormat.INT16
        pref = device.preferredFormat()
        if pref.sampleFormat() in _FORMAT_MAP:
            return pref, _FORMAT_MAP[pref.sampleFormat()]
        pref.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isFormatSupported(pref):
            return pref, SampleFormat.INT16
        return None

    def _start_sink(self, position: float) -> bool:
        assert self._audio is not None
        device = self.default_output()
        if device.isNull():
            self.errorOccurred.emit("오디오 출력 장치를 찾을 수 없습니다. 스피커나 헤드폰 연결을 확인해 주세요.")
            return False
        chosen = self._choose_format(device)
        if chosen is None:
            self.errorOccurred.emit("출력 장치가 지원하는 오디오 형식을 찾지 못했습니다.")
            return False
        fmt, sample_format = chosen
        region = self.play_region()
        sr = self._audio.sample_rate
        if self._speed < 1.0:
            if not self._has_stretch(region):
                self._pending_play = True  # 늘린 오디오가 오면 set_stretched()가 재생을 시작한다
                self.stretchRequested.emit(region[0], region[1], self._speed)
                return False
            tracks = dict(self._stretched[1])
        else:
            tracks = {name: a.samples for name, a in self._tracks.items()} or {"original": self._audio.samples}
        conv = PcmMixer(tracks, sr, fmt.sampleRate(), fmt.channelCount(), sample_format)
        for name in tracks:
            st = self.track_state(name)
            conv.set_state(name, gain=st["gain"], mute=st["mute"], solo=st["solo"])
        self._mixer = conv
        loop = self.loop_enabled
        if self._speed < 1.0:
            stream = PlaybackStream(conv, 0, conv.total_out_frames, region[0], self._speed, loop)
        elif loop:
            stream = PlaybackStream(conv, conv.out_frame_at(region[0]), conv.out_frame_at(region[1]), region[0], 1.0, True)
        else:
            stream = PlaybackStream(conv, 0, conv.total_out_frames, 0.0, 1.0, False)
        start = stream.frame_for_time(position)
        self._stream, self._start_frame = stream, start
        sink = QAudioSink(device, fmt, self)
        pcm = PcmDevice(stream, start, parent=sink)  # sink와 함께 지운다 (_teardown 참고)
        pcm.open(QIODevice.OpenModeFlag.ReadOnly)
        sink.setBufferSize(int(fmt.bytesForDuration(BUFFER_USEC)))
        # stateChanged 시그널은 PySide6 6.12에서 QAudio::State 인자 변환에 실패한다.
        # 대신 _tick 타이머(30ms)에서 sink.state()를 확인한다.
        self._sink, self._device = sink, pcm
        self._sink_device_id = bytes(device.id())
        sink.start(pcm)
        if sink.error() != _QtAudio.Error.NoError:
            self._teardown()
            self.errorOccurred.emit("오디오 장치를 열지 못했습니다.")
            return False
        return True

    def _teardown(self) -> None:
        self._timer.stop()
        sink, pcm = self._sink, self._device
        self._sink = self._device = None
        self._mixer = None
        self._stream = None
        if pcm is not None:
            pcm.finish()  # 닫지 않는다: 오디오 스레드가 늦게 읽어도 빈 데이터만 받는다
        if sink is not None:
            sink.stop()
            sink.deleteLater()  # 자식인 장치도 이때 함께 지워진다

    def _on_outputs_changed(self) -> None:
        """출력 장치 목록이 바뀌었다. 재생 중이고 쓰던 장치가 기본 장치가 아니게 됐거나 사라졌으면 같은 위치에서 다시 연다."""
        if self._sink is None or self._state is PlaybackState.STOPPED:
            return
        default = self.default_output()
        present = any(bytes(d.id()) == self._sink_device_id for d in QMediaDevices.audioOutputs())
        if present and not default.isNull() and bytes(default.id()) == self._sink_device_id:
            return
        was_playing = self._state is PlaybackState.PLAYING
        pos = self.position()
        log.info("출력 장치가 바뀌어 재생 출력을 다시 엽니다 (%.2f초, 기본 장치: %s)", pos,
                 "없음" if default.isNull() else default.description())
        self._teardown()
        self._pos = pos
        if default.isNull():
            self._set_state(PlaybackState.STOPPED)
            self.positionChanged.emit(pos)
            self.errorOccurred.emit("오디오 출력 장치가 없어 재생을 멈췄습니다. 장치를 연결한 뒤 다시 재생하세요.")
            return
        self.device_switches += 1
        if not was_playing:  # 일시정지 중이면 다음 재생 때 새 장치로 연다
            self._set_state(PlaybackState.STOPPED)
            self._play_from = pos
            self.positionChanged.emit(pos)
            return
        if self._start_sink(pos):
            self._timer.start()
        else:
            self._set_state(PlaybackState.STOPPED)

    def _check_sink(self) -> None:
        if self._sink is None:
            return
        state = self._sink.state()
        drained = self._device is not None and self._device.at_end
        # 장치에 따라 데이터가 끝나도 IdleState로 바로 바뀌지 않을 수 있어 처리 시간도 함께 본다
        end_t = self._stream.time_at(self._stream.total_out_frames) if self._stream is not None else self.duration
        if drained and (state == _QtAudio.State.IdleState or self.position() >= end_t - 0.005):
            # 끝까지 재생함 (느린 재생 창이면 창의 끝)
            self._teardown()
            self._pos = end_t
            self._set_state(PlaybackState.STOPPED)
            self.positionChanged.emit(self._pos)
        elif state == _QtAudio.State.StoppedState and self._sink.error() != _QtAudio.Error.NoError:
            err = self._sink.error()
            self._pos = self.position()
            self._teardown()
            self._set_state(PlaybackState.STOPPED)
            self.errorOccurred.emit(f"재생 중 오류가 발생했습니다 ({err.name}).")

    def _tick(self) -> None:
        self._check_sink()
        self.positionChanged.emit(self.position())

    def _set_state(self, state: PlaybackState) -> None:
        if state is not self._state:
            self._state = state
            self.stateChanged.emit(state)

    def shutdown(self) -> None:
        self._teardown()
