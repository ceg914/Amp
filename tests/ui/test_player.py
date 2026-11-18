import numpy as np
import pytest
from PySide6.QtCore import QIODevice

from amp.app.player import AudioPlayer, PcmDevice, PlaybackState
from amp.audio.data import AudioData
from amp.audio.pcm import PcmConverter, SampleFormat


def test_pcm_device_reads_until_end(qapp):
    conv = PcmConverter(np.zeros((100, 1), np.float32), 100, 100, 2, SampleFormat.INT16)
    dev = PcmDevice(conv, start_frame=90)
    dev.open(QIODevice.OpenModeFlag.ReadOnly)
    assert dev.bytesAvailable() == 10 * 4
    data = dev.read(1000)
    assert len(data) == 40
    assert len(dev.read(100)) == 0 and dev.at_end


def test_player_seek_while_stopped(qapp):
    p = AudioPlayer()
    p.set_audio(AudioData(np.zeros((1000, 1), np.float32), 100))
    positions = []
    p.positionChanged.connect(positions.append)
    p.seek(4.0)
    p.seek(99.0)
    assert positions == [4.0, 10.0]
    assert p.position() == 10.0
    assert p.state is PlaybackState.STOPPED


def test_play_without_audio_is_noop(qapp):
    p = AudioPlayer()
    p.play()
    assert p.state is PlaybackState.STOPPED


def test_play_without_device_reports_error(qapp, qtbot):
    p = AudioPlayer()
    p.set_audio(AudioData(np.zeros((1000, 1), np.float32), 100))
    if AudioPlayer.output_available():
        p.play()
        assert p.state is PlaybackState.PLAYING
        p.stop()
        return
    with qtbot.waitSignal(p.errorOccurred, timeout=1000) as blocker:
        p.play()
    assert "장치" in blocker.args[0]
    assert p.state is PlaybackState.STOPPED


needs_device = pytest.mark.skipif(not AudioPlayer.output_available(), reason="오디오 출력 장치 없음")


@needs_device
def test_seek_while_playing_keeps_updating(qtbot):
    """재생 중 탐색한 뒤에도 위치 갱신이 계속되고, 끝에 닿으면 멈춤 상태가 된다."""
    p = AudioPlayer()
    sr = 48000
    p.set_audio(AudioData(np.zeros((sr * 2, 2), np.float32), sr))
    p.play()
    qtbot.wait(200)
    p.seek(1.8)
    ticks = []
    p.positionChanged.connect(ticks.append)
    qtbot.wait(300)
    assert len(ticks) >= 3
    qtbot.waitUntil(lambda: p.state is PlaybackState.STOPPED, timeout=8000)
    assert p.position() == pytest.approx(2.0)
    p.shutdown()


def test_tracks_and_states_without_device(qapp):
    p = AudioPlayer()
    sr = 100
    orig = AudioData(np.zeros((1000, 2), np.float32), sr)
    p.set_audio(orig)
    p.seek(3.0)
    p.set_track("vocals", AudioData(np.zeros((1000, 2), np.float32), sr))
    assert set(p.tracks) == {"original", "vocals"} and p.position() == 3.0  # 위치 유지
    errors = []
    p.errorOccurred.connect(errors.append)
    p.set_track("bad", AudioData(np.zeros((999, 2), np.float32), sr))
    assert "bad" not in p.tracks and errors
    p.set_track_state("vocals", mute=True, gain=0.5)
    assert p.track_state("vocals") == {"gain": 0.5, "mute": True, "solo": False}
    p.set_track("vocals", None)
    assert set(p.tracks) == {"original"}


def test_pcm_device_finish_returns_no_data_without_closing(qapp):
    """정지 뒤 오디오 스레드가 늦게 읽어도 '열리지 않은 장치' 오류 대신 빈 데이터를 받는다 (Windows 로그)."""
    from PySide6.QtCore import QIODevice

    from amp.app.player import PcmDevice
    from amp.audio.pcm import PcmMixer, SampleFormat

    y = np.zeros((4410, 2), np.float32)
    dev = PcmDevice(PcmMixer({"original": y}, 44100, 44100, 2, SampleFormat.INT16))
    dev.open(QIODevice.OpenModeFlag.ReadOnly)
    assert len(dev.read(400)) == 400
    dev.finish()
    dev.readAll()  # QIODevice가 미리 읽어 둔 버퍼는 비운다
    assert dev.isOpen() and len(dev.read(400)) == 0 and dev.bytesAvailable() == 0


@needs_device
def test_teardown_keeps_device_alive_until_sink_is_deleted(qtbot):
    p = AudioPlayer()
    p.set_audio(AudioData(np.zeros((44100, 2), np.float32), 44100))
    p.play()
    dev, sink = p._device, p._sink
    assert dev.parent() is sink  # 장치는 sink의 자식 → sink와 함께 지워진다
    p.seek(0.5)  # 재생 중 이동 = 정리 후 새로 시작
    dev.readAll()
    assert dev.isOpen() and dev.finished and len(dev.read(64)) == 0
    p.stop()
