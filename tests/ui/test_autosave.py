import os

from amp.app.autosave import AutosaveManager
from amp.app.document import ProjectDocument
from amp.core.model import Project, Score
from amp.storage.project_file import load_project
from tests.conftest import note


def test_autosave_writes_only_when_dirty(qapp, tmp_path):
    doc = ProjectDocument()
    mgr = AutosaveManager(doc, str(tmp_path), interval_ms=3_600_000)
    assert not mgr.save_now()
    doc.reset(Project(score=Score(notes=[note("a", 0, 1)])), None, dirty=True)
    assert mgr.save_now()
    assert not mgr.save_now()  # 바뀐 것이 없으면 다시 쓰지 않는다
    assert os.path.exists(mgr.project_path)
    assert [n.id for n in load_project(mgr.project_path).score.notes()] == ["a"]
    doc.mark_saved(str(tmp_path / "saved.ampproj"))
    mgr.save_now()
    assert not os.path.exists(mgr.project_path)
    mgr.shutdown()


def test_active_session_is_not_orphan(qapp, tmp_path):
    doc = ProjectDocument()
    doc.reset(Project(score=Score(notes=[note("a", 0, 1)])), "/x/orig.ampproj", dirty=True)
    mgr = AutosaveManager(doc, str(tmp_path), interval_ms=3_600_000)
    mgr.save_now()
    assert AutosaveManager.find_orphans(str(tmp_path)) == []
    mgr.shutdown()


def test_orphan_after_crash(qapp, tmp_path):
    doc = ProjectDocument()
    doc.reset(Project(score=Score(notes=[note("a", 0, 1)])), "/x/orig.ampproj", dirty=True)
    mgr = AutosaveManager(doc, str(tmp_path), interval_ms=3_600_000)
    mgr.save_now()
    # 비정상 종료: 잠금만 풀리고 복구 파일은 남는다
    mgr._lock.unlock()
    entries = AutosaveManager.find_orphans(str(tmp_path))
    assert len(entries) == 1
    assert entries[0].original_path == "/x/orig.ampproj"
    assert entries[0].session_id == mgr.session_id
    AutosaveManager.remove_entry(entries[0])
    assert AutosaveManager.find_orphans(str(tmp_path)) == []
