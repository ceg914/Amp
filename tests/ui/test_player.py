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
