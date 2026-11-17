"""프로젝트 잠금 (STEP 2.5 요구 테스트: 잠금 1~7, 자동 테스트 9·10).

허용 기준: 잠금 판단은 OS 파일 잠금만 따른다. 잠금 파일·소유자 정보가 복사돼 있어도 실제로 잡은
프로세스가 없으면 열 수 있어야 하고, 실제로 잡은 프로세스가 있으면 같은 프로세스든 다른 프로세스든 막아야 한다.
"""

import json
import os
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from amp.core.model import Project, Score
from amp.project.lock import LOCK_FILE, OWNER_FILE, ProjectLock, ProjectLockedError, is_locked, read_owner
from amp.project.workspace import COPY, create_project, open_folder
from tests.conftest import note

HOLDER = r"""
import sys, time
sys.path.insert(0, sys.argv[2])
from amp.project.lock import ProjectLock
lock = ProjectLock(sys.argv[1]); lock.acquire("other-process")
print("LOCKED", flush=True)
time.sleep(60)
"""


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "원본.wav"
    sf.write(src, np.zeros((4410, 2), np.float32), 44100)
    folder = create_project(src, tmp_path / "P", "곡 #1", COPY)
    p = Project(audio=folder.audio_ref(), score=Score(notes=[note("u1", 0.1, 0.4, 64)]))
    folder.save(p)
    return folder


def _holder(root):
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(root), os.getcwd()], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "LOCKED"
    return proc


def test_same_project_twice_in_one_process(project):
    a = ProjectLock(project.root)
    a.acquire()
    with pytest.raises(ProjectLockedError) as e:
        ProjectLock(project.root).acquire()
    assert "다른 창" in str(e.value)
    assert is_locked(project.root) is not None
    a.release()
    assert is_locked(project.root) is None
    with ProjectLock(project.root):  # 해제 뒤에는 다시 열 수 있다
        pass


def test_other_process_is_blocked_then_released(project):
    proc = _holder(project.root)
    try:
        with pytest.raises(ProjectLockedError) as e:
            ProjectLock(project.root).acquire()
        assert e.value.owner.pid == proc.pid and e.value.owner.project_id == "other-process"
        assert "다른 SONATLAS 실행" in str(e.value)
    finally:
        proc.terminate()
        proc.wait(10)
    with ProjectLock(project.root):  # 정상 종료하면 풀린다
        pass


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL은 POSIX 전용 (Windows는 TerminateProcess로 같은 동작)")
def test_crashed_process_leaves_no_active_lock(project):
    proc = _holder(project.root)
    os.kill(proc.pid, signal.SIGKILL)  # 비정상 종료: 정리 코드가 돌지 않는다
    proc.wait(10)
    assert (project.root / OWNER_FILE).exists()  # 안내 정보 파일은 남아 있지만
    lock = ProjectLock(project.root)
    lock.acquire()  # OS가 잠금을 풀었으므로 바로 열린다
    assert read_owner(project.root).pid == os.getpid()  # 남은 정보는 새 소유자로 덮어쓴다
    lock.release()


def test_copy_while_open_then_open_copy_independently(project, tmp_path):
    lock = ProjectLock(project.root)
    lock.acquire(project.meta.id)
    copy_root = tmp_path / "탐색기로 복사한 곳" / "곡 #1 - 복사본"
    shutil.copytree(project.root, copy_root)  # 잠금 파일·소유자 정보까지 복사된다
    assert (copy_root / LOCK_FILE).exists() and (copy_root / OWNER_FILE).exists()
    copy_lock = ProjectLock(copy_root)
    copy_lock.acquire()  # 원본이 열려 있어도 복사본은 열린다
    copied, report = open_folder(copy_root)
    assert copied.meta.id != project.meta.id  # 독립된 프로젝트로 새 ID
    assert copied.meta.copied_from["id"] == project.meta.id
    assert any("복사된 프로젝트" in n for n in report.notes)
    assert copied.load_project()[0].score.get("u1").pitch == 64  # 편집 데이터 보존
    original, _ = open_folder(project.root)
    assert original.meta.id == project.meta.id  # 원본 ID는 그대로
    copy_lock.release()
    lock.release()


def test_lock_file_copied_from_another_computer(project, tmp_path):
    other = tmp_path / "다른 PC에서 가져옴"
    shutil.copytree(project.root, other)
    (other / OWNER_FILE).write_text(json.dumps({"instance_id": "x", "pid": os.getpid(), "host": "OTHER-PC"}), "utf-8")
    with ProjectLock(other):  # 살아 있는 PID처럼 보여도 OS 잠금이 없으면 연다
        assert read_owner(other).host != "OTHER-PC"


def test_moved_project_keeps_id_and_data(project, tmp_path):
    with ProjectLock(project.root):
        pass
    project_id = project.meta.id
    moved = tmp_path / "새 위치" / "옮긴 곡"
    moved.parent.mkdir()
    shutil.move(str(project.root), str(moved))
    folder, report = open_folder(moved)
    assert folder.meta.id == project_id  # 이동은 같은 프로젝트
    assert any("위치가 바뀌었습니다" in n for n in report.notes)
    assert folder.meta.last_path == str(moved.resolve())
    assert folder.load_project()[0].score.get("u1").pitch == 64
    with ProjectLock(moved):
        pass


def test_release_is_idempotent_and_owner_removed(project):
    lock = ProjectLock(project.root)
    lock.acquire()
    lock.release()
    lock.release()
    assert not (project.root / OWNER_FILE).exists()
    assert (project.root / LOCK_FILE).exists()  # 잠금 파일 자체는 지우지 않는다


def test_leftover_temp_files_cleaned_on_open(project):
    (project.paths.stems / ".tmp-abcd").mkdir()
    (project.paths.stems / ".tmp-abcd" / "vocals.wav").write_bytes(b"partial")
    (project.paths.analysis / ".sonatlas-x.tmp").write_bytes(b"partial")
    (project.paths.exports / ".v.123456.part.mp3").write_bytes(b"partial")
    (project.paths.exports / "내보낸 파일.wav").write_bytes(b"keep")
    with ProjectLock(project.root):
        removed = project.cleanup_temp()
    assert len(removed) == 3
    assert not (project.paths.stems / ".tmp-abcd").exists()
    assert (project.paths.exports / "내보낸 파일.wav").read_bytes() == b"keep"
    assert project.load_project()[0].score.get("u1").pitch == 64
