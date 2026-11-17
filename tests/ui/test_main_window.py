
import pytest
from PySide6.QtWidgets import QMessageBox

from amp.app.autosave import AutosaveManager
from amp.app.main_window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    # 대화상자가 테스트를 멈추지 않도록 기본 응답을 정한다
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(MainWindow, "_show_error", lambda self, title, msg, details="": self.errors.append(msg))
    w = MainWindow(recovery_dir=str(tmp_path / "recovery"), autosave_interval_ms=3_600_000)
    w.errors = []
    qtbot.addWidget(w)
    w.show()
    yield w
    w.doc.mark_saved(w.doc.file_path or "")  # 닫을 때 저장 확인 창을 띄우지 않도록
    w.close()


def load(qtbot, w, path):
    w.load_audio(path, new_project=True)
    qtbot.waitUntil(lambda: w.audio is not None, timeout=10_000)


def analyse(qtbot, w):
    w.run_sample_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and w._analysis is None, timeout=10_000)


def test_load_mp3_and_analyse(qtbot, window, mp3_file):
    # mp3_file은 8초짜리. 샘플 백엔드는 첫 마디(2.4초)를 쉬므로 그 뒤에 음표가 생긴다
    load(qtbot, window, mp3_file)
    assert window.peaks is not None
    assert window.doc.project.audio.path == mp3_file
    assert window.act_analyze.isEnabled()
    analyse(qtbot, window)
    assert window.doc.project.analysis.is_placeholder
    assert len(window.view.roll.items_by_id) == len(window.doc.score) > 0
    assert "샘플" in window.analysis_label.text()


def test_load_error_is_reported(qtbot, window, tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"nope" * 100)
    window.load_audio(str(bad), new_project=True)
    qtbot.waitUntil(lambda: bool(window.errors), timeout=10_000)
    assert "오디오를 읽지 못했습니다" in window.errors[0]
    assert window.audio is None


def test_save_and_reopen(qtbot, window, wav_file, tmp_path):
    load(qtbot, window, wav_file)
    window.doc.add_note(0.5, 1.0, 64)
    path = str(tmp_path / "song.ampproj")
    assert window._write(path)
    assert not window.doc.is_dirty and window.doc.file_path == path

    window.doc.reset(window.doc.project.__class__())
    assert window.open_project(path)
    qtbot.waitUntil(lambda: window.audio is not None, timeout=10_000)
    notes = window.doc.score.notes()
    assert [(n.start, n.pitch) for n in notes] == [(0.5, 64)]
    assert not window.doc.is_dirty


def test_open_invalid_project(qtbot, window, tmp_path):
    p = tmp_path / "x.ampproj"
    p.write_bytes(b"broken")
    assert not window.open_project(str(p))
    assert window.errors


def test_split_at_playhead_and_merge(qtbot, window, wav_file):
    load(qtbot, window, wav_file)
    nid = window.doc.add_note(0.2, 1.2, 60)
    window.view.roll.select_ids([nid])
    window.seek(0.7)
    window.split_selected_at_playhead()
    assert len(window.doc.score) == 2
    window.select_all()
    window.merge_selected()
    assert len(window.doc.score) == 1
    window.view.roll.select_ids([])
    window.split_selected_at_playhead()  # 선택이 없으면 상태 표시줄 메시지만 남긴다
    assert len(window.doc.score) == 1


def test_bpm_spin_is_undoable(qtbot, window):
    window.bpm_spin.setValue(90)
    assert window.doc.score.tempo.initial_bpm == 90
    window.doc.undo_stack.undo()
    assert window.bpm_spin.value() == 120


def test_recovery_after_crash(qtbot, window, wav_file, tmp_path):
    load(qtbot, window, wav_file)
    window.doc.add_note(0.5, 1.0, 64)
    assert window.autosave.save_now()
    window.autosave._lock.unlock()  # 비정상 종료 흉내: 잠금만 사라짐
    recovery_dir = window.autosave.directory

    other = MainWindow(recovery_dir=recovery_dir, autosave_interval_ms=3_600_000)
    qtbot.addWidget(other)
    other.check_recovery()
    assert [n.pitch for n in other.doc.score.notes()] == [64]
    assert other.doc.is_dirty
    qtbot.waitUntil(lambda: other.audio is not None, timeout=10_000)
    assert AutosaveManager.find_orphans(recovery_dir) == []
    other.doc.mark_saved("")
    other.close()
