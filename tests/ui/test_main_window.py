"""메인 창 통합 시험: 프로젝트 생성·저장·재열기·복구·분석·트랙·내보내기·동기화 (STEP 1·2·SONATLAS 회귀)."""

import json
import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QMessageBox

from amp.app import dialogs
from amp.app.autosave import AutosaveManager
from amp.app.main_window import MainWindow
from amp.core.model import AudioRef, Project, Score
from amp.engine.settings import AnalysisSettings
from amp.engine.transcription import TranscriptionPipeline
from amp.storage.project_file import save_project
from amp.devtools.synth import SR, sung
from tests.conftest import note


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


def create(qtbot, w, path, mode="copy"):
    w.create_project_from_audio(path, None, mode)
    qtbot.waitUntil(lambda: w.session.audio is not None, timeout=20_000)
    return w.session.folder


def test_create_project_and_sample_analysis(qtbot, window, mp3_file):
    folder = create(qtbot, window, mp3_file)
    assert (folder.root / "original.mp3").is_file() and (folder.root / "project.json").is_file()
    assert window.windowTitle().startswith(folder.name)
    window.run_sample_analysis()
    qtbot.waitUntil(lambda: window.doc.project.analysis is not None and not window.session.busy, timeout=10_000)
    assert len(window.view.roll.items_by_id) == len(window.doc.score) > 0
    assert "샘플" in window.analysis_label.text()
    assert window.tracks.rows["original"].property("available")
    assert not window.tracks.rows["vocals"].property("available")  # 분리 전


def test_broken_audio_is_reported(qtbot, window, tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"nope" * 100)
    window.create_project_from_audio(str(bad), None, "copy")
    qtbot.waitUntil(lambda: bool(window.errors), timeout=10_000)
    assert any("오디오를 읽지 못했습니다" in e for e in window.errors)
    assert window.session.audio is None
    assert bad.read_bytes() == b"nope" * 100  # 원본은 그대로


def test_save_close_and_reopen(qtbot, make_window, song):
    w = make_window("a")
    folder = create(qtbot, w, song)
    w.doc.add_note(0.5, 1.0, 64)
    w.seek(1.5)
    w.player.set_track_state("original", gain=0.7)
    assert w.save() and not w.doc.is_dirty
    root = str(folder.root)
    w.close()
    w2 = make_window("b")
    assert w2.open_project(os.path.join(root, "project.json"))
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    assert [(n.start, n.pitch) for n in w2.doc.score.notes()] == [(0.5, 64)]
    assert not w2.doc.is_dirty
    assert w2.current_time() == pytest.approx(1.5, abs=0.01)
    assert w2.player.track_state("original")["gain"] == pytest.approx(0.7)


def test_same_project_cannot_open_twice(qtbot, make_window, song):
    w = make_window("a")
    folder = create(qtbot, w, song)
    w2 = make_window("b")
    assert not w2.open_project(str(folder.root))  # 동시에 두 창에서 같은 프로젝트를 고치지 못하게 잠근다
    assert any("이미 열려 있습니다" in e and "다른 창" in e for e in w2.errors)


def test_split_at_playhead_merge_and_bpm(qtbot, window, song):
    create(qtbot, window, song)
    nid = window.doc.add_note(0.2, 1.2, 60)
    window.view.roll.select_ids([nid])
    window.seek(0.7)
    window.split_selected_at_playhead()
    assert len(window.doc.score) == 2
    window.select_all()
    window.merge_selected()
    assert len(window.doc.score) == 1
    window.bpm_spin.setValue(90)
    assert window.doc.score.tempo.initial_bpm == 90
    window.doc.undo_stack.undo()
    assert window.bpm_spin.value() == 120
    window.beat_offset.setValue(0.25)
    assert window.doc.score.tempo.first_beat_time == pytest.approx(0.25)


def test_recovery_after_crash(qtbot, make_window, song):
    w = make_window("a")
    folder = create(qtbot, w, song)
    w.doc.add_note(0.5, 1.0, 64)
    assert w.autosave.save_now()
    # 비정상 종료 흉내: 정상 종료 절차(close) 없이 잠금만 풀린다 (프로세스가 죽은 것과 같음)
    w.autosave._timer.stop()
    w.autosave._lock.unlock()
    w.session.close()
    recovery_dir = w.autosave.directory
    other = MainWindow(recovery_dir=recovery_dir, autosave_interval_ms=3_600_000,
                       projects_root=str(folder.root.parent))
    other.errors = []
    qtbot.addWidget(other)
    other.check_recovery()
    assert other.session.folder is not None and other.session.folder.root == folder.root
    assert [n.pitch for n in other.doc.score.notes()] == [64]
    assert other.doc.is_dirty
    assert AutosaveManager.find_orphans(recovery_dir) == []
    other.doc.mark_saved("")
    other.close()


class IdentityPipeline(TranscriptionPipeline):
    """실제 작업 프로세스를 쓰되 분리 모델만 시험용(_identity)으로 바꾼다 (htdemucs 없이 트랙 경로 시험)."""

    def __init__(self, *a, **k):
        super().__init__(*a, separation_model="_identity", **k)


def test_analysis_tracks_export_and_edit_protection(qtbot, window, song, monkeypatch):
    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    folder = create(qtbot, window, song)
    window.run_analysis()
    qtbot.waitUntil(lambda: window.doc.project.analysis is not None and not window.session.busy, timeout=120_000)
    qtbot.waitUntil(lambda: "vocals" in window.player.tracks, timeout=20_000)
    assert [n.pitch for n in window.doc.score.notes()] == [64, 67, 69]
    assert window.tracks.rows["vocals"].property("available")
    # 트랙 솔로·음소거
    window.tracks.rows["vocals"].solo.setChecked(True)
    window.tracks.rows["original"].mute.setChecked(True)
    assert window.player.track_state("vocals")["solo"] and window.player.track_state("original")["mute"]
    # 내보내기 (분리를 다시 하지 않고 stems/ 결과를 형식만 바꿔 저장)
    from amp.project.export import ExportOptions

    jobs = [("vocals", str(folder.paths.exports / "v.mp3")), ("instrumental", str(folder.paths.exports / "i.wav"))]
    h = window.session.export(jobs, ExportOptions("mp3", mp3_bitrate=192))
    qtbot.waitSignal(h.ended, timeout=30_000)
    qtbot.waitUntil(lambda: (folder.paths.exports / "i.wav").exists(), timeout=10_000)
    assert (folder.paths.exports / "v.mp3").stat().st_size > 0
    # 편집 후 음표만 다시 검출 → 편집 보존 (수정 유지하며 적용)
    first = window.doc.score.notes()[0].id
    window.doc.move_notes([first], 0.0, 2)
    monkeypatch.setattr(dialogs.ReanalysisDialog, "exec", lambda self: (setattr(self, "mode", "keep_edits"), 1)[1])
    window.inspector.analysis.set_settings(AnalysisSettings.for_sensitivity("simple"))
    window.redetect_notes()
    qtbot.waitUntil(lambda: not window.session.busy, timeout=20_000)
    qtbot.wait(50)
    pitches = [n.pitch for n in window.doc.score.notes()]
    assert 66 in pitches  # 사용자가 올린 음표가 남아 있다
    window.doc.undo_stack.undo()  # 재분석 반영도 실행 취소된다
    assert 66 in [n.pitch for n in window.doc.score.notes()]


def test_legacy_ampproj_is_migrated(qtbot, window, song, tmp_path):
    p = Project(audio=AudioRef(song, "fp", 5.0, SR, 2), score=Score(notes=[note("a", 0.5, 1.0, 62)]))
    old = tmp_path / "예전.ampproj"
    save_project(p, str(old))
    before = old.read_bytes()
    assert window.open_project(str(old))
    qtbot.waitUntil(lambda: window.session.audio is not None, timeout=20_000)
    assert [n.pitch for n in window.doc.score.notes()] == [62]
    assert old.read_bytes() == before
    assert any(f.name.endswith("예전.ampproj") for f in window.session.folder.paths.backup.iterdir())


def test_timeline_sync_between_views(qtbot, window, song):
    """요구 테스트 5: 파형·눈금자·피아노 롤·재생 커서가 같은 시간축을 쓴다."""
    create(qtbot, window, song)
    window.view.zoom_time(3.0, 2.0)
    window.view.horizontalScrollBar().setValue(int(window.view.mapper.time_to_x(1.0)))
    qtbot.wait(20)
    t = 2.345
    x_roll = window.view.mapper.time_to_x(t) - window.view.horizontalScrollBar().value()
    assert window.timeline.time_to_x(t) == pytest.approx(x_roll, abs=1.0)
    assert window.view.mapToScene(int(x_roll), 0).x() == pytest.approx(window.view.mapper.time_to_x(t), abs=1.5)
    window.seek(t)
    assert window.view.roll.playhead.pos().x() == pytest.approx(window.view.mapper.time_to_x(t), abs=0.5)
    window._show_track_waveform("original")
    assert window.current_time() == pytest.approx(t, abs=0.01)  # 트랙 표시를 바꿔도 위치 유지


def test_copied_open_project_opens_in_second_window(qtbot, make_window, song, tmp_path):
    """열려 있는 프로젝트 폴더를 탐색기로 복사한 상황: 복사본은 다른 창에서 독립적으로 열린다."""
    import shutil

    w = make_window("a")
    folder = create(qtbot, w, song)
    w.doc.add_note(0.5, 1.0, 64)
    assert w.save()
    copy_root = tmp_path / "복사본" / folder.root.name
    shutil.copytree(folder.root, copy_root)
    w2 = make_window("b")
    assert w2.open_project(str(copy_root / "project.json"))
    assert w2.session.folder.meta.id != w.session.folder.meta.id
    assert [n.pitch for n in w2.doc.score.notes()] == [64]
    w3 = make_window("c")
    assert not w3.open_project(str(folder.root))  # 원본은 여전히 보호된다


def test_analysis_runs_are_recorded_and_compared(qtbot, window, song, monkeypatch):
    """STEP 2.5: 성공한 분석만 analysis/runs/에 남고, 최근 두 실행 비교 보고서가 만들어진다."""
    from amp.project import runs

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    folder = create(qtbot, window, song)
    window.run_analysis()
    qtbot.waitUntil(lambda: window.doc.project.analysis is not None and not window.session.busy, timeout=120_000)
    first = window.doc.score.notes()[0].id
    window.doc.move_notes([first], 0.0, 2)  # 사용자 편집
    monkeypatch.setattr(dialogs.ReanalysisDialog, "exec", lambda self: (setattr(self, "mode", "keep_edits"), 1)[1])
    window.redetect_notes()
    qtbot.waitUntil(lambda: not window.session.busy, timeout=20_000)
    qtbot.wait(50)

    records = runs.list_runs(folder.paths.analysis)
    assert [r.kind for r in records] == ["full", "redetect"]
    full, again = records
    assert {"pitch", "tempo", "notes", "total"} <= set(full.timings_s) and full.timings_s["total"] > 0
    assert full.models["pitch"]["name"] == "pyin" and full.models["notes"]["version"] == "2"
    assert [n["pitch"] for n in full.notes] == [64, 67, 69]
    assert all(0.0 <= n["confidence"] <= 1.0 for n in full.notes)
    assert again.score_summary["modified"] == 1  # 실행 시점 편집 현황은 개수만 기록
    assert 66 not in [n["pitch"] for n in again.notes]  # 기록은 자동 분석 결과만

    before = [(n.start, n.end, n.pitch) for n in window.doc.score.notes()]
    window.compare_recent_runs()
    reports = list((folder.paths.analysis / "reports").glob("compare-*.md"))
    assert len(reports) == 1 and "일치: 3개" in reports[0].read_text(encoding="utf-8")
    assert [(n.start, n.end, n.pitch) for n in window.doc.score.notes()] == before  # 비교는 악보를 바꾸지 않는다


def test_failed_analysis_is_not_recorded(qtbot, window, song, monkeypatch):
    from amp.core.errors import AnalysisError
    from amp.project import runs

    class Failing:
        def run(self, *a, **k):
            raise AnalysisError("시험용 실패")

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", Failing)
    folder = create(qtbot, window, song)
    window.run_analysis()
    qtbot.waitUntil(lambda: not window.session.busy and bool(window.errors), timeout=20_000)
    assert runs.list_runs(folder.paths.analysis) == []
    assert window.doc.project.analysis is None


def test_save_conflict_asks_before_overwriting(qtbot, window, song, monkeypatch):
    folder = create(qtbot, window, song)
    assert window.save()
    data = json.loads(folder.paths.project_json.read_text(encoding="utf-8"))
    data["name"] = "다른 곳에서 바꾼 이름"
    folder.paths.project_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    asked = []

    def answer(choice):
        return lambda *a, **k: (asked.append(a[1]), choice)[1]

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.No))
    assert window.save() is False and asked == ["저장 충돌"]
    assert json.loads(folder.paths.project_json.read_text(encoding="utf-8"))["name"] == "다른 곳에서 바꾼 이름"
    assert not window.errors
    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Yes))
    assert window.save() is True


def test_run_record_failure_does_not_block_analysis(qtbot, window, song, monkeypatch):
    """실제 곡에서 발견: 실행 기록 저장이 예외를 내자 분석 결과가 반영되지 않았다."""
    def broken(*a, **k):
        raise ValueError("Out of range float values are not JSON compliant: nan")

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    monkeypatch.setattr("amp.project.runs.save_run", broken)
    create(qtbot, window, song)
    window.run_analysis()
    qtbot.waitUntil(lambda: window.doc.project.analysis is not None and not window.session.busy, timeout=120_000)
    assert [n.pitch for n in window.doc.score.notes()] == [64, 67, 69]
    assert window.save()


def test_no_vocal_range_removes_auto_notes_keeps_edits_and_is_saved(qtbot, make_window, song, monkeypatch):
    """보컬 없는 구간: 지정하면 음표를 다시 검출해 그 구간의 자동 음표를 없애고, 사용자가 고친 음표는 남긴다."""
    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    monkeypatch.setattr(dialogs.ReanalysisDialog, "exec", lambda self: (setattr(self, "mode", "keep_edits"), 1)[1])
    w = make_window("a")
    folder = create(qtbot, w, song)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    first, second, third = w.doc.score.notes()  # 64(1.0~1.8), 67(1.9~2.7), 69(2.8~4.0)
    w.doc.move_notes([third.id], 0.0, 1)  # 사용자가 고친 음표 (70)
    w.mark_no_vocal(first.start - 0.05, first.end + 0.05)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    qtbot.wait(50)
    assert w.inspector.analysis.settings().no_vocal_ranges and w.view.roll.no_vocal_ranges
    assert w.inspector.analysis.no_vocal_info.text().startswith("1개")
    w.mark_no_vocal(2.75, 4.1)  # 사용자가 고친 음표가 있는 곳도 지정
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    qtbot.wait(50)
    pitches = [n.pitch for n in w.doc.score.notes()]
    assert 64 not in pitches and 67 in pitches and 70 in pitches  # 자동 음표만 사라지고 고친 음표는 남는다
    assert w.doc.project.analysis.settings["no_vocal_ranges"]
    assert w.save()
    stored = json.loads(folder.paths.project_json.read_text(encoding="utf-8"))["analysis_settings"]["no_vocal_ranges"]
    assert len(stored) == 2
    assert w.settings.value("analysis/settings", {}).get("no_vocal_ranges", []) in ([], None) or \
        not AnalysisSettings.from_dict(w.settings.value("analysis/settings") or {}).no_vocal_ranges
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    w.close()
    w2 = make_window("b")
    assert w2.open_project(root)
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    assert len(w2.inspector.analysis.settings().no_vocal_ranges) == 2 and len(w2.view.roll.no_vocal_ranges) == 2


def test_no_vocal_ranges_do_not_carry_over_to_another_project(qtbot, window, song, tmp_path):
    folder = create(qtbot, window, song)
    window.inspector.analysis.set_no_vocal_ranges([(0.0, 1.0)])
    assert window.save()
    other = tmp_path / "다른 곡.wav"
    import shutil

    shutil.copy(song, other)
    window.doc.mark_saved(window.doc.file_path or "")
    create(qtbot, window, str(other))
    qtbot.waitUntil(lambda: window.session.folder is not None and window.session.folder.root != folder.root, timeout=20_000)
    assert window.inspector.analysis.settings().no_vocal_ranges == ()
    assert window.view.roll.no_vocal_ranges == ()


def test_no_vocal_dialog_parses_and_rejects_bad_rows(qtbot, monkeypatch):
    from PySide6.QtWidgets import QTableWidgetItem

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    dlg = dialogs.NoVocalRangesDialog([(0.0, 28.0)], duration=60.0, playhead=40.0)
    qtbot.addWidget(dlg)
    dlg._add()
    assert dlg.table.rowCount() == 2 and dlg.table.item(1, 0).text() == "0:40.00"
    dlg.accept()
    assert dlg.ranges() == ((0.0, 28.0), (40.0, 45.0))
    dlg.table.setItem(1, 1, QTableWidgetItem("0:39"))  # 끝이 시작보다 앞
    dlg.accept()
    assert shown and "끝" in shown[0]
