"""STEP 1·2의 단일 파일 프로젝트(.ampproj) → 프로젝트 폴더 변환.

- 원래 .ampproj 파일은 수정하지 않는다. 새 폴더의 backup/에 사본도 남긴다.
- 원본 오디오는 '연결' 방식으로 참조한다 (기존 파일이 있던 위치를 그대로 쓴다).
- 분석 원본·사용자 편집 음표·템포를 그대로 옮긴다. 편집 음표를 초기화하지 않는다.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from amp.core.model import AudioRef
from amp.project.workspace import (
    LINK,
    OriginalInfo,
    ProjectFolder,
    ProjectMeta,
    _relpath,
    backup_file,
    create_project,
    now_iso,
)
from amp.project.layout import ProjectPaths, unique_folder
from amp.storage.project_file import load_project


def import_legacy(ampproj: str | os.PathLike, parent: str | os.PathLike, progress=None) -> tuple[ProjectFolder, list[str]]:
    """(새 프로젝트 폴더, 안내 메시지 목록)."""
    src = Path(ampproj)
    project = load_project(str(src))  # 형식 오류면 ProjectFormatError
    notes: list[str] = []
    audio_path = Path(project.audio.path) if project.audio else None
    if audio_path is not None and audio_path.is_file():
        info = {"duration": project.audio.duration, "sample_rate": project.audio.sample_rate,
                "channels": project.audio.channels}
        folder = create_project(audio_path, parent, src.stem, LINK, info, progress)
    else:
        root = unique_folder(parent, src.stem)
        root.mkdir(parents=True)
        try:
            ProjectPaths(root).ensure_dirs()
            a = project.audio
            original = OriginalInfo(LINK, str(audio_path) if audio_path else "", audio_path.name if audio_path else "",
                                    "", 0, relative_path=_relpath(audio_path, root) if audio_path else None,
                                    duration=a.duration if a else 0.0, sample_rate=a.sample_rate if a else 0,
                                    channels=a.channels if a else 0)
            folder = ProjectFolder(root, ProjectMeta(uuid.uuid4().hex, root.name, now_iso(), now_iso(), original))
            folder.write_meta()
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        notes.append("원본 오디오 파일을 찾지 못했습니다. 프로젝트를 연 뒤 '원본 다시 연결'로 지정해 주세요.")
    backup = backup_file(folder, src, "legacy")
    notes.append(f"원래 프로젝트 파일은 그대로 두었고, 사본을 {backup.relative_to(folder.root)}에 남겼습니다.")
    if project.analysis is not None and project.analysis.is_placeholder:
        notes.append("이전 프로젝트의 분석 결과는 샘플 데이터입니다.")
    status = folder.resolve_original(check_content=False)
    project.audio = folder.audio_ref(status) if status.path else project.audio or AudioRef("")
    folder.save(project)
    return folder, notes

