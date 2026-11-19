"""UI 시험 공용 픽스처 (메인 창·시험 곡)."""

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from amp.app.main_window import MainWindow
from amp.devtools.synth import SR, sung



@pytest.fixture(autouse=True)
def isolated_settings(tmp_path_factory):
    """창 설정(QSettings)을 시험마다 임시 폴더의 ini 파일로. 사용자 설정을 읽거나 남기지 않는다."""
    path = tmp_path_factory.mktemp("qsettings")
    old = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(path))
    yield
    QSettings.setDefaultFormat(old)

@pytest.fixture
def make_window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(MainWindow, "_show_error", lambda self, title, msg, details="": self.errors.append(msg))
    created = []

    def make(name="w"):
        w = MainWindow(recovery_dir=str(tmp_path / f"recovery-{name}"), autosave_interval_ms=3_600_000,
                       projects_root=str(tmp_path / "SONATLAS Projects"))
        w.errors = []
        qtbot.addWidget(w)
        w.show()
        created.append(w)
        return w

    yield make
    for w in created:
        if w.isVisible():
            w.doc.mark_saved(w.doc.file_path or "")
            w.close()


@pytest.fixture
def window(make_window):
    return make_window()


@pytest.fixture
def song(tmp_path):
    y = sung([(1.0, 1.8, 64), (1.9, 2.7, 67), (2.8, 4.0, 69)], 5.0, vibrato_cents=25)
    p = tmp_path / "원본 #노래.wav"
    sf.write(p, np.stack([y, y], 1), SR)
    return str(p)
