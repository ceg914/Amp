"""프로젝트 폴더 생성·열기·저장 (Qt 비의존).

원칙
- 원본 파일은 읽기만 한다. 복사 방식이면 프로젝트 안에 사본을 만들고, 연결 방식이면 경로만 기록한다.
- 같은 이름의 폴더가 있으면 "이름 (2)"처럼 새 폴더를 만든다. 기존 프로젝트를 덮어쓰지 않는다.
- 프로젝트 안의 파일은 상대 경로로 기록해 폴더를 옮기거나 복사해도 열린다.
- project.json·notes.json·tempo.json은 원자적으로 교체하고 직전 파일을 .bak으로 남긴다.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from amp import __version__
from amp.core.errors import AnalysisCancelled, ProjectFormatError, StorageError
from amp.core.fileio import atomic_write_json, long_path, read_json_with_backup, storage_error
from amp.core.model import AudioRef, Project
from amp.engine import stems as stem_store
from amp.engine.cache import AnalysisCache
from amp.engine.merge import summarize
from amp.project import schema
from amp.project.analysis_io import LoadReport, load_analysis_files, save_analysis_files
from amp.project.layout import BACKUP_DIR, CACHE_DIR, LOCK_FILE, ProjectPaths, unique_folder

COPY = "copy"
LINK = "link"
TRACKS = ("original", "vocals", "instrumental")
_CHUNK = 1 << 20

ProgressFn = Callable[[float, str], None]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class OriginalInfo:
    mode: str  # copy | link
    path: str  # 복사: 프로젝트 기준 상대 경로 / 연결: 외부 절대 경로
    file_name: str  # 원래 파일 이름 (표시용)
    sha256: str
    size: int
    mtime: float = 0.0
    relative_path: str | None = None  # 연결 방식에서 프로젝트 기준 상대 경로 (함께 옮긴 경우 대비)
    source_path: str | None = None  # 복사 방식에서 복사해 온 원래 위치 (기록용)
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0


@dataclass
class ProjectMeta:
    id: str
    name: str
    created: str
    modified: str
    original: OriginalInfo
    app_version: str = __version__
    separation: dict = field(default_factory=lambda: {"status": "none"})
    analysis: dict = field(default_factory=lambda: {"status": "none"})
    edit_state: dict = field(default_factory=dict)
    analysis_settings: dict = field(default_factory=dict)
    playback: dict = field(default_factory=lambda: {"position": 0.0, "tracks": {}})
    ui: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["files"] = {"stems": "stems/stems.json", "pitch": "analysis/pitch.json", "notes": "analysis/notes.json",
                      "tempo": "analysis/tempo.json", "cache": "cache/", "exports": "exports/"}
        return schema.stamp(schema.PROJECT, d)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectMeta":
        try:
            o = d["original"]
            original = OriginalInfo(
                mode=str(o["mode"]), path=str(o["path"]), file_name=str(o.get("file_name", "")),
                sha256=str(o.get("sha256", "")), size=int(o.get("size", 0)), mtime=float(o.get("mtime", 0.0)),
                relative_path=o.get("relative_path"), source_path=o.get("source_path"),
                duration=float(o.get("duration", 0.0)), sample_rate=int(o.get("sample_rate", 0)),
                channels=int(o.get("channels", 0)),
            )
            return cls(
                id=str(d["id"]), name=str(d["name"]), created=str(d.get("created", "")),
                modified=str(d.get("modified", "")), original=original, app_version=str(d.get("app_version", "")),
                separation=dict(d.get("separation") or {"status": "none"}),
                analysis=dict(d.get("analysis") or {"status": "none"}),
                edit_state=dict(d.get("edit_state") or {}), analysis_settings=dict(d.get("analysis_settings") or {}),
                playback=dict(d.get("playback") or {"position": 0.0, "tracks": {}}), ui=dict(d.get("ui") or {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProjectFormatError(f"project.json 내용이 올바르지 않습니다: {e}") from e


@dataclass
class OriginalStatus:
    path: Path | None
    state: str  # ok | missing | changed
    message: str = ""


def sha256_file(path: str | os.PathLike, progress: ProgressFn | None = None,
                is_cancelled: Callable[[], bool] | None = None) -> str:
    h = hashlib.sha256()
    total = max(1, os.path.getsize(long_path(path)))
    done = 0
    with open(long_path(path), "rb") as f:
        while block := f.read(_CHUNK):
            if is_cancelled and is_cancelled():
                raise AnalysisCancelled("취소했습니다.")
            h.update(block)
            done += len(block)
            if progress:
                progress(done / total, "원본 파일 확인 중")
    return h.hexdigest()


def _copy_with_hash(src: Path, dst: Path, progress: ProgressFn | None, is_cancelled) -> str:
    """원본을 읽기 전용으로 열어 복사하면서 해시를 계산한다. 임시 이름에 쓴 뒤 이름을 바꾼다."""
    tmp = dst.with_name(f".{dst.name}.part")
    h = hashlib.sha256()
    total = max(1, src.stat().st_size)
    done = 0
    try:
        with open(long_path(src), "rb") as fin, open(long_path(tmp), "wb") as fout:
            while block := fin.read(_CHUNK):
                if is_cancelled and is_cancelled():
                    raise AnalysisCancelled("프로젝트 생성을 취소했습니다.")
                fout.write(block)
                h.update(block)
                done += len(block)
                if progress:
                    progress(done / total, "원본 파일 복사 중")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(long_path(tmp), long_path(dst))
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise storage_error(e, dst) from e
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    shutil.copystat(long_path(src), long_path(dst), follow_symlinks=True)
    return h.hexdigest()


class ProjectFolder:
    def __init__(self, root: str | os.PathLike, meta: ProjectMeta) -> None:
        self.paths = ProjectPaths(root)
        self.meta = meta
        self._analysis_written: object | None = None  # 마지막으로 pitch.json에 쓴 분석 결과 객체

    @property
    def root(self) -> Path:
        return self.paths.root

    @property
    def name(self) -> str:
        return self.meta.name

    # --- 원본 ---------------------------------------------------------------

    def resolve_original(self, check_content: bool = True) -> OriginalStatus:
        o = self.meta.original
        candidates: list[Path] = []
        if o.mode == COPY:
            candidates.append(self.root / o.path)
        else:
            candidates.append(Path(o.path))
            if o.relative_path:
                candidates.append((self.root / o.relative_path).resolve())
        for p in candidates:
            if p.is_file():
                if check_content and o.sha256:
                    st = p.stat()
                    if st.st_size != o.size or (o.mtime and abs(st.st_mtime - o.mtime) > 2):
                        if sha256_file(p) != o.sha256:
                            return OriginalStatus(p, "changed", "원본 파일 내용이 프로젝트를 만들 때와 다릅니다. "
                                                  "분리·분석 결과가 맞지 않을 수 있습니다.")
                return OriginalStatus(p, "ok")
        where = candidates[0]
        return OriginalStatus(None, "missing", f"원본 파일을 찾을 수 없습니다: {where}")

    def relink(self, new_path: str, sha256: str | None = None) -> bool:
        """외부 원본을 다시 연결한다. 내용이 같으면 True."""
        p = Path(new_path).resolve()
        sha = sha256 or sha256_file(p)
        same = sha == self.meta.original.sha256
        st = p.stat()
        o = self.meta.original
        self.meta.original = OriginalInfo(
            mode=LINK if o.mode == LINK or not same else o.mode, path=str(p), file_name=p.name, sha256=sha,
            size=st.st_size, mtime=st.st_mtime, relative_path=_relpath(p, self.root),
            duration=o.duration, sample_rate=o.sample_rate, channels=o.channels,
        )
        return same

    def audio_ref(self, status: OriginalStatus | None = None) -> AudioRef:
        status = status or self.resolve_original(check_content=False)
        o = self.meta.original
        path = str(status.path) if status.path else (str(self.root / o.path) if o.mode == COPY else o.path)
        return AudioRef(path, o.sha256, o.duration, o.sample_rate, o.channels)

    # --- 분리 결과 ------------------------------------------------------------

    def stems_status(self) -> tuple[stem_store.StemManifest | None, bool, str]:
        m = stem_store.read_manifest(self.paths.stems) if self.paths.stems.exists() else None
        ok, reason = stem_store.validate(self.paths.stems, m, self.meta.original.sha256)
        return m, ok, reason

    # --- 캐시 ---------------------------------------------------------------

    def cache(self) -> AnalysisCache:
        return AnalysisCache(self.paths.cache)

    def clear_cache(self) -> None:
        """cache/만 지운다. 원본·stems/·analysis/·exports/는 건드리지 않는다."""
        self.cache().clear()
        self.paths.cache.mkdir(exist_ok=True)

    # --- 저장·불러오기 ----------------------------------------------------------

    def write_meta(self) -> None:
        self.meta.modified = now_iso()
        self.meta.app_version = __version__
        atomic_write_json(self.paths.project_json, self.meta.to_dict(), backup=True)

    def save(self, project: Project, *, ui: dict | None = None, playback: dict | None = None) -> None:
        """분석 파일과 project.json을 저장한다. 피치는 분석 결과가 바뀌었을 때만 다시 쓴다."""
        self.paths.ensure_dirs()
        write_pitch = project.analysis is not None and project.analysis is not self._analysis_written
        save_analysis_files(self.paths, project, write_pitch=write_pitch)
        if write_pitch:
            self._analysis_written = project.analysis
        s = summarize(project)
        self.meta.edit_state = {"has_user_edits": s.has_edits, "notes_total": len(project.score),
                                "original": s.original, "modified": s.modified, "user_added": s.user,
                                "tempo_edited": s.tempo_edited, "saved": now_iso()}
        a = project.analysis
        if a is None:
            self.meta.analysis = {"status": "none"}
        else:
            self.meta.analysis = {
                "status": "sample" if a.is_placeholder else "done",
                "last_run": a.created_at,
                "settings": a.settings,
                "models": {k: {"name": m.name, "version": m.version} for k, m in a.models.items()},
                "warnings": list(a.warnings),
                "notes": len(a.notes),
            }
        m, ok, reason = self.stems_status()
        self.meta.separation = ({"status": "done", "model": m.model, "created": m.created, "settings": m.settings,
                                 "device": m.model_info.get("device"), "manifest": "stems/stems.json"}
                                if ok and m else {"status": "invalid" if m else "none", "reason": reason})
        if ui is not None:
            self.meta.ui = ui
        if playback is not None:
            self.meta.playback = playback
        self.write_meta()

    def load_project(self) -> tuple[Project, LoadReport]:
        report = LoadReport()
        analysis, score = load_analysis_files(self.paths, report)
        self._analysis_written = analysis
        return Project(audio=self.audio_ref(), analysis=analysis, score=score), report


def _relpath(target: Path, start: Path) -> str | None:
    try:
        return os.path.relpath(target, start)
    except ValueError:  # Windows에서 드라이브가 다르면 상대 경로를 만들 수 없다
        return None


def open_folder(root: str | os.PathLike) -> tuple[ProjectFolder, LoadReport]:
    paths = ProjectPaths(root)
    report = LoadReport()
    if not paths.project_json.is_file():
        raise ProjectFormatError(f"SONATLAS 프로젝트 폴더가 아닙니다 (project.json 없음): {root}")
    data, from_bak = read_json_with_backup(paths.project_json, what="프로젝트 파일")
    data, migrated = schema.migrate(schema.PROJECT, data)
    if from_bak:
        report.warnings.append("project.json이 손상되어 직전 백업(.bak)에서 불러왔습니다.")
    if migrated:
        report.migrated.append("project.json")
    return ProjectFolder(paths.root, ProjectMeta.from_dict(data)), report


def create_project(
    source: str | os.PathLike,
    parent: str | os.PathLike,
    name: str | None = None,
    mode: str = COPY,
    audio_info: dict | None = None,
    progress: ProgressFn | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ProjectFolder:
    """새 프로젝트 폴더를 만든다. 실패하거나 취소하면 이번에 만든 폴더를 지운다."""
    src = Path(source).resolve()
    if not src.is_file():
        raise StorageError(f"원본 파일을 찾을 수 없습니다: {src}")
    if mode not in (COPY, LINK):
        raise ValueError(mode)
    parent = Path(parent)
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise storage_error(e, parent) from e
    root = unique_folder(parent, name or src.stem)
    try:
        root.mkdir()
    except OSError as e:
        raise storage_error(e, root) from e
    try:
        paths = ProjectPaths(root)
        paths.ensure_dirs()
        st = src.stat()
        if mode == COPY:
            dst = root / f"original{src.suffix.lower()}"
            sha = _copy_with_hash(src, dst, progress, is_cancelled)
            original = OriginalInfo(COPY, dst.name, src.name, sha, st.st_size, dst.stat().st_mtime, source_path=str(src))
        else:
            sha = sha256_file(src, progress, is_cancelled)
            original = OriginalInfo(LINK, str(src), src.name, sha, st.st_size, st.st_mtime, relative_path=_relpath(src, root))
        info = audio_info or {}
        original.duration = float(info.get("duration", 0.0))
        original.sample_rate = int(info.get("sample_rate", 0))
        original.channels = int(info.get("channels", 0))
        meta = ProjectMeta(id=uuid.uuid4().hex, name=root.name, created=now_iso(), modified=now_iso(), original=original)
        folder = ProjectFolder(root, meta)
        folder.write_meta()
        save_analysis_files(paths, Project(audio=folder.audio_ref()), write_pitch=False)
        return folder
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)  # 이번 호출에서 새로 만든 폴더만 지운다
        raise


def find_projects_with_source(parent: str | os.PathLike, sha256: str) -> list[Path]:
    """parent 바로 아래 프로젝트 중 같은 원본(내용 해시)을 쓰는 것."""
    out = []
    parent = Path(parent)
    if not parent.is_dir():
        return out
    for d in sorted(parent.iterdir()):
        pj = d / "project.json"
        if not pj.is_file():
            continue
        try:
            data, _ = read_json_with_backup(pj)
            if isinstance(data, dict) and (data.get("original") or {}).get("sha256") == sha256:
                out.append(d)
        except Exception:  # noqa: BLE001 - 다른 프로젝트가 깨져 있어도 검색은 계속한다
            continue
    return out


def copy_project(folder: ProjectFolder, parent: str | os.PathLike, name: str,
                 progress: ProgressFn | None = None) -> ProjectFolder:
    """'다른 이름으로 저장': 프로젝트 폴더를 새 위치로 복사한다 (cache/ 제외). 원래 폴더는 그대로 둔다."""
    dst = unique_folder(parent, name)
    ignore = shutil.ignore_patterns(CACHE_DIR, LOCK_FILE, "*.part", ".sonatlas-*.tmp")
    try:
        shutil.copytree(folder.root, dst, ignore=ignore)
    except OSError as e:
        shutil.rmtree(dst, ignore_errors=True)
        raise storage_error(e, dst) from e
    (dst / CACHE_DIR).mkdir(exist_ok=True)
    new_meta = ProjectMeta.from_dict(folder.meta.to_dict())
    new_meta.id = uuid.uuid4().hex
    new_meta.name = dst.name
    new_meta.created = now_iso()
    o = new_meta.original
    if o.mode == LINK:
        o.relative_path = _relpath(Path(o.path), dst)
    copied = ProjectFolder(dst, new_meta)
    copied.write_meta()
    if progress:
        progress(1.0, "프로젝트 복사 완료")
    return copied


def backup_file(folder: ProjectFolder, src: str | os.PathLike, label: str) -> Path:
    """마이그레이션 전 백업: backup/<시각>-<label>-<파일이름>."""
    d = folder.root / BACKUP_DIR
    d.mkdir(exist_ok=True)
    stamp_ = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = d / f"{stamp_}-{label}-{Path(src).name}"
    shutil.copy2(long_path(src), long_path(dst))
    return dst
