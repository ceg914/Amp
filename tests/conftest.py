import os

# Qt를 불러오기 전에 화면 없는 플랫폼을 지정한다 (CI·원격 환경에서도 테스트가 돌도록)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf

from amp.core.model import Note


@pytest.fixture(scope="session")
def qapp_args():
    return ["amp-tests"]


@pytest.fixture(scope="session", autouse=True)
def _qt_identity():
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("AmpTests")
    QCoreApplication.setApplicationName("AmpTests")


def make_tone(seconds=2.0, sr=22050, channels=2, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    x = 0.4 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x[:, None]


@pytest.fixture
def wav_file(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, make_tone(), 22050)
    return str(path)


@pytest.fixture
def mp3_file(tmp_path):
    path = tmp_path / "tone.mp3"
    sf.write(path, make_tone(8.0, 44100), 44100, format="MP3")
    return str(path)


def note(nid, start, end, pitch=60, **kw):
    return Note(id=nid, start=start, end=end, pitch=pitch, **kw)
