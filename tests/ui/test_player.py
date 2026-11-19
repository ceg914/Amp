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


# --- STEP 3.5: 재생 중 출력 장치 바뀜 ---------------------------------------------------------------

def _pulse_null_sink(name):
    import shutil
    import subprocess

    if not shutil.which("pactl") or subprocess.run(["pactl", "info"], capture_output=True).returncode != 0:
        pytest.skip("PulseAudio(pactl)가 없어 가상 출력 장치를 만들 수 없다")
    r = subprocess.run(["pactl", "load-module", "module-null-sink", f"sink_name={name}",
                        f"sink_properties=device.description={name}"], capture_output=True, text=True)
    return r.stdout.strip()


def test_playback_moves_to_new_default_output_and_survives_unplug(qtbot):
    import subprocess

    old_default = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True).stdout.strip() \
        if AudioPlayer.output_available() else ""
    a_mod = _pulse_null_sink("amp_test_a")
    b_mod = _pulse_null_sink("amp_test_b")
    try:
        subprocess.run(["pactl", "set-default-sink", "amp_test_a"], check=True)
        qtbot.waitUntil(lambda: b"amp_test_a" in bytes(__import__("PySide6.QtMultimedia", fromlist=["QMediaDevices"])
                                                         .QMediaDevices.defaultAudioOutput().id()), timeout=5000)
        p = AudioPlayer()
        errors = []
        p.errorOccurred.connect(errors.append)
        sr = 8000
        y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr * 20) / sr)).astype(np.float32)
        p.set_audio(AudioData(np.stack([y, y], 1), sr))
        p.play()
        qtbot.wait(300)
        assert p._sink_device_id == b"amp_test_a"
        subprocess.run(["pactl", "set-default-sink", "amp_test_b"], check=True)  # 기본 장치 바꿈
        qtbot.waitUntil(lambda: p._sink_device_id == b"amp_test_b", timeout=5000)
        assert p.state is PlaybackState.PLAYING and p.device_switches == 1
        subprocess.run(["pactl", "unload-module", b_mod], check=True)  # 쓰던 장치를 뺌 (헤드폰 뽑기)
        b_mod = ""
        qtbot.waitUntil(lambda: p.device_switches == 2, timeout=5000)
        assert p.state is PlaybackState.PLAYING and p._sink_device_id != b"amp_test_b" and errors == []
        p.shutdown()
    finally:
        for m in (a_mod, b_mod):
            if m:
                subprocess.run(["pactl", "unload-module", m])
        if old_default:
            subprocess.run(["pactl", "set-default-sink", old_default])


def test_player_follows_os_default_when_qt_flag_is_stale(qtbot, monkeypatch):
    """Windows 점검에서 Qt가 본 기본 장치가 첫 장치에 머문 기록이 있다(보고서 7.4절).
    운영체제에 물은 기본 장치가 Qt 표시와 다르면 운영체제 쪽을 따른다. 여기서는 Windows 조회를 바꿔 끼워 흉내 낸다."""
    import subprocess

    from PySide6.QtMultimedia import QMediaDevices

    from amp.audio import windows_output

    old_default = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True).stdout.strip() \
        if AudioPlayer.output_available() else ""
    a_mod = _pulse_null_sink("amp_os_a")
    b_mod = _pulse_null_sink("amp_os_b")
    monkeypatch.setattr(AudioPlayer, "_os_lookup_failed", False)
    os_says = {"id": "amp_os_a"}
    monkeypatch.setattr(windows_output, "default_output_id", lambda: os_says["id"])
    try:
        subprocess.run(["pactl", "set-default-sink", "amp_os_a"], check=True)
        qtbot.waitUntil(lambda: bytes(QMediaDevices.defaultAudioOutput().id()) == b"amp_os_a", timeout=5000)
        p = AudioPlayer()
        errors = []
        p.errorOccurred.connect(errors.append)
        sr = 8000
        y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr * 20) / sr)).astype(np.float32)
        p.set_audio(AudioData(np.stack([y, y], 1), sr))
        p.play()
        qtbot.wait(200)
        assert p._sink_device_id == b"amp_os_a"
        os_says["id"] = "amp_os_b"  # Windows는 b로 바뀌었는데 Qt는 여전히 a를 기본으로 표시
        assert bytes(QMediaDevices.defaultAudioOutput().id()) == b"amp_os_a"
        p._on_outputs_changed()  # Qt 장치 변경 알림
        assert p._sink_device_id == b"amp_os_b" and p.device_switches == 1 and p.state is PlaybackState.PLAYING
        p._on_outputs_changed()  # 같은 상태에서 알림이 또 와도 다시 열지 않는다
        assert p.device_switches == 1
        os_says["id"] = "없는 장치 id"  # Qt 목록에 없는 id면 Qt 기본 장치(a)로
        assert bytes(AudioPlayer.default_output().id()) == b"amp_os_a"

        def boom():
            raise OSError("COM 실패")

        monkeypatch.setattr(windows_output, "default_output_id", boom)  # 조회가 실패하면 그 뒤로는 Qt만 쓴다
        assert bytes(AudioPlayer.default_output().id()) == b"amp_os_a" and AudioPlayer._os_lookup_failed
        assert errors == []
        p.shutdown()
    finally:
        for m in (a_mod, b_mod):
            if m:
                subprocess.run(["pactl", "unload-module", m])
        if old_default:
            subprocess.run(["pactl", "set-default-sink", old_default])


def test_no_output_left_stops_with_message(qapp, monkeypatch):
    from PySide6.QtMultimedia import QAudioDevice

    p = AudioPlayer()
    p.set_audio(AudioData(np.zeros((1000, 1), np.float32), 100))

    class Sink:
        def processedUSecs(self):
            return 500_000

        def stop(self):
            pass

        def deleteLater(self):
            pass

    p._sink, p._sink_device_id, p._state, p._base = Sink(), b"gone", PlaybackState.PLAYING, 1.0
    monkeypatch.setattr("amp.app.player.QMediaDevices.defaultAudioOutput", staticmethod(lambda: QAudioDevice()))
    monkeypatch.setattr("amp.app.player.QMediaDevices.audioOutputs", staticmethod(lambda: []))
    errors = []
    p.errorOccurred.connect(errors.append)
    p._on_outputs_changed()
    assert p.state is PlaybackState.STOPPED and abs(p.position() - 1.5) < 1e-9
    assert errors and "출력 장치가 없어" in errors[0]
