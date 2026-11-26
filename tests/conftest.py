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


@pytest.fixture(autouse=True)
def _optional_pronunciation_off(monkeypatch):
    """v0.7.0: 영어(CMUdict)·한자(MeCab) 변환기는 설치 여부에 따라 앱이 켠다. 시험 결과가 개발 PC에 무엇이 설치됐는지에
    따라 달라지지 않도록 기본은 '설치 안 됨'으로 보고, 시험이 끝나면 켠 변환기를 끈다. 켜서 보는 시험은 직접 켠다."""
    from amp.lyrics import pronunciation as pron
    from amp.lyrics.pronunciation import english, kanji

    monkeypatch.setattr(english, "find_installed", lambda find_spec=None: None)
    monkeypatch.setattr(kanji, "available", lambda find_spec=None: False)
    saved = list(pron._EXTRA)
    yield
    pron._EXTRA[:] = saved


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
