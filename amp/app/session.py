"""열린 프로젝트 하나의 작업 흐름 (명세 10장의 controllers 역할).

생성·열기·잠금·오디오/분리 결과 불러오기·저장·다른 이름으로 저장·원본 재연결·분석 실행·내보내기를 맡는다.
무거운 일은 TaskRunner로 보내고, 결과는 시그널로 메인 창에 알린다.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QStandardPaths, Signal

from amp.app.document import ProjectDocument
from amp.app.player import AudioPlayer
from amp.app.workers import TaskHandle, TaskRunner
from amp.audio.data import AudioData
from amp.audio.decoder import load_audio
from amp.audio.peaks import WaveformPeaks
from amp.audio.mask import apply_no_vocal
from amp.core.errors import AmpError, AnalysisError, SaveConflictError
from amp import __version__
from amp.engine import stems as stem_store
from amp.engine.merge import summarize
from amp.engine.settings import AnalysisSettings
from amp.engine.transcription import TranscriptionPipeline, TranscriptionRequest, redetect_notes
from amp.engine.worker_client import run_worker
from amp.project import runs
from amp.project.export import ExportOptions, export_audio
from amp.project.layout import DEFAULT_ROOT_NAME
from amp.project.lock import ProjectLock, ProjectLockedError
from amp.project.legacy import import_legacy
from amp.project.workspace import (
    OriginalStatus,
    ProjectFolder,
    copy_project,
    create_project,
    find_projects_with_source,
    open_folder,
    sha256_file,
)

log = logging.getLogger(__name__)
RECENT_KEY = "projects/recent"
ROOT_KEY = "projects/root"
MAX_RECENT = 10


def default_projects_root() -> str:
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation) or os.path.expanduser("~")
    return os.path.join(docs, DEFAULT_ROOT_NAME)


def app_data_dir(*parts: str) -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation) or os.path.expanduser("~/.sonatlas")
    return os.path.join(base, *parts)


class ProjectSession(QObject):
    opened = Signal()
    closed = Signal()
    audioReady = Signal()
    tracksChanged = Signal()
    originalChanged = Signal()
    message = Signal(str, int)
    error = Signal(str, str, str)  # 제목, 메시지, 상세
    notice = Signal(str, str)  # 제목, 안내 (오류가 아닌 알림)
    analysisReady = Signal(object)  # AnalysisResult (반영 방식은 메인 창이 묻는다)
    recognitionReady = Signal(object, bool)  # Recognition, 캐시 재사용 여부
    alignmentReady = Signal(object)  # amp.lyrics.engine.AlignOutput
    busyChanged = Signal(bool)
    taskStarted = Signal(object, str)  # TaskHandle, 표시 문구
    stemsImported = Signal(object)  # amp.engine.stem_import.ImportReport

    def __init__(self, doc: ProjectDocument, player: AudioPlayer, runner: TaskRunner,
                 settings: QSettings, projects_root: str | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.player = player
        self.runner = runner
        self.settings = settings
        self._root_override = projects_root
        self.folder: ProjectFolder | None = None
        self.audio: AudioData | None = None
        self.stems: dict[str, AudioData] = {}  # 재생·내보내기에 쓰는 스템 (보컬 없는 구간 적용본)
        self.raw_stems: dict[str, AudioData] = {}  # 분리 결과 그대로 (stems/*.wav와 같음)
        self.mask_ranges: tuple = ()  # 스템에 적용할 보컬 없는 구간 (끄면 빈 값)
        self._applied_mask: tuple = ()
        self.peaks: dict[str, WaveformPeaks] = {}
        self.manifest: stem_store.StemManifest | None = None
        self.stems_reason = "분리 전"
        self.model_files: dict = {}  # RoFormer {"checkpoint", "config"} (프로그램 설정에서 채운다)
        self.original: OriginalStatus | None = None
        self.analysis_handle: TaskHandle | None = None
        self._lock: ProjectLock | None = None
        self._seq = 0
        self.last_conflict: list[str] = []  # 마지막 저장이 충돌로 멈췄을 때 바뀐 파일 이름

    # --- 설정 ---------------------------------------------------------------

    @property
    def projects_root(self) -> str:
        return self._root_override or str(self.settings.value(ROOT_KEY, "") or default_projects_root())

    def set_projects_root(self, path: str) -> None:
        self._root_override = None
        self.settings.setValue(ROOT_KEY, path)

    def recent(self) -> list[str]:
        items = self.settings.value(RECENT_KEY, []) or []
        if isinstance(items, str):
            items = [items]
        return [p for p in items if Path(p, "project.json").is_file()]

    def _remember(self, root: Path) -> None:
        items = [str(root)] + [p for p in self.recent() if Path(p) != root]
        self.settings.setValue(RECENT_KEY, items[:MAX_RECENT])

    @property
    def is_open(self) -> bool:
        return self.folder is not None

    @property
    def busy(self) -> bool:
        return self.analysis_handle is not None

    def _submit(self, label: str, fn) -> TaskHandle:
        handle = self.runner.submit(label, fn)
        self.taskStarted.emit(handle, label)
        handle.failed.connect(lambda msg, tb, l=label: self._failed(l, msg, tb))
        handle.cancelled.connect(lambda l=label: self.message.emit(f"{l}을(를) 취소했습니다.", 5000))
        return handle

    def _failed(self, label: str, message: str, detail: str) -> None:
        log.error("%s 실패: %s\n%s", label, message, detail)  # 원인 확인용 (로그 파일·로그 패널)
        self.error.emit(f"{label} 실패", message, detail)

    # --- 생성·열기·닫기 ----------------------------------------------------------

    def existing_projects_for(self, audio_path: str) -> list[Path]:
        try:
            return find_projects_with_source(self.projects_root, sha256_file(audio_path))
        except OSError:
            return []

    def create_from_audio(self, audio_path: str, name: str | None = None, mode: str = "copy",
                          parent: str | None = None) -> TaskHandle:
        parent = parent or self.projects_root

        def job(progress, token):
            info = {}
            try:
                import soundfile as sf

                i = sf.info(audio_path)
                info = {"duration": i.duration, "sample_rate": i.samplerate, "channels": i.channels}
            except Exception:  # noqa: BLE001 - 정보가 없어도 프로젝트는 만든다 (오디오를 불러올 때 채운다)
                pass
            return create_project(audio_path, parent, name, mode, info, progress, lambda: token.cancelled)

        handle = self._submit("프로젝트 만들기", job)
        handle.finished.connect(lambda folder: self.open_folder(str(folder.root)))
        return handle

    def open_folder(self, root: str) -> bool:
        root_p = Path(root)
        if self.folder is not None and os.path.normcase(os.path.realpath(self.folder.root)) == os.path.normcase(os.path.realpath(root_p)):
            self.message.emit("이미 열려 있는 프로젝트입니다.", 5000)
            return True
        if not root_p.is_dir():
            self.error.emit("프로젝트를 열 수 없습니다", f"폴더가 없습니다: {root_p}", "")
            return False
        lock = ProjectLock(root_p)
        try:
            lock.acquire()
        except ProjectLockedError as e:
            log.info("잠긴 프로젝트 열기 거부: %s (%s)", root_p, e.owner)
            self.error.emit("프로젝트를 열 수 없습니다", str(e), "")
            return False
        except OSError as e:
            self.error.emit("프로젝트를 열 수 없습니다", f"잠금 파일을 만들 수 없습니다 (읽기 전용 위치일 수 있습니다): {e}", "")
            return False
        try:
            folder, rep = open_folder(root_p)
            removed = folder.cleanup_temp()
            project, rep2 = folder.load_project()
        except (AmpError, OSError) as e:
            lock.release()
            self.error.emit("프로젝트를 열 수 없습니다", str(e), "")
            return False
        if removed:
            log.info("비정상 종료로 남은 임시 파일 정리: %s", removed)
        self.close()
        self._lock = lock
        self.folder = folder
        self.doc.reset(project, str(root_p))
        warnings = rep.warnings + rep2.warnings
        if warnings:
            self.notice.emit("프로젝트 복구 안내", "\n".join(warnings))
        for note in rep.notes + rep2.notes:
            log.info(note)
            self.message.emit(note, 10000)
        self.original = folder.resolve_original()
        self.originalChanged.emit()
        if self.original.state == "changed":
            self.notice.emit("원본 파일 변경", self.original.message)
        self._remember(root_p)
        self.opened.emit()
        if self.original.path is not None:
            self.load_audio(str(self.original.path))
        else:
            self.message.emit(self.original.message, 10000)
        return True

    def import_legacy(self, ampproj: str) -> bool:
        try:
            folder, notes = import_legacy(ampproj, self.projects_root)
        except (AmpError, OSError) as e:
            self.error.emit("이전 형식 프로젝트를 변환하지 못했습니다", str(e), "")
            return False
        ok = self.open_folder(str(folder.root))
        if ok:
            self.notice.emit("프로젝트 변환 완료", f"새 프로젝트 폴더: {folder.root}\n" + "\n".join(notes))
        return ok

    def close(self) -> None:
        self._seq += 1
        self.cancel_analysis()
        self.player.set_audio(None)
        self.audio = None
        self.stems = {}
        self.raw_stems = {}
        self._applied_mask = ()
        self.peaks = {}
        self.manifest = None
        self.original = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None
        if self.folder is not None:
            self.folder = None
            self.closed.emit()

    # --- 오디오·분리 결과 ---------------------------------------------------------

    def load_audio(self, path: str) -> TaskHandle:
        self._seq += 1
        seq = self._seq

        def job(progress, token):
            audio = load_audio(path, progress, lambda: token.cancelled)
            progress(1.0, "파형 계산 중")
            return audio, WaveformPeaks.build(audio.mono(), audio.sample_rate)

        handle = self._submit("오디오 불러오기", job)

        def done(result) -> None:
            if seq != self._seq:
                return
            self.audio, peaks = result
            self.peaks = {"original": peaks}
            self.player.set_audio(self.audio)
            if self.folder is not None:
                o = self.folder.meta.original
                o.duration, o.sample_rate, o.channels = self.audio.duration, self.audio.sample_rate, self.audio.channels
                self.doc.project.audio = self.folder.audio_ref(self.original)
            self.audioReady.emit()
            self.message.emit(f"{Path(path).name} — {self.audio.duration:.1f}초, {self.audio.sample_rate}Hz, "
                              f"{self.audio.channels}채널", 8000)
            self.load_stems()

        handle.finished.connect(done)
        return handle

    def load_stems(self) -> TaskHandle | None:
        """stems/의 분리 결과가 유효하면 불러와 트랙에 올린다."""
        if self.folder is None:
            return None
        manifest, ok, reason = self.folder.stems_status()
        self.manifest = manifest if ok else None
        self.stems_reason = "" if ok else ("분리 전" if manifest is None else f"분리 결과 무효: {reason}")
        if not ok:
            self._set_stems({})
            return None
        folder, seq, ranges = self.folder, self._seq, self.mask_ranges

        def job(progress, token):
            raw = stem_store.load(folder.paths.stems, manifest)
            eff = apply_no_vocal(raw, ranges)
            return raw, eff, {n: WaveformPeaks.build(a.mono(), a.sample_rate) for n, a in eff.items()}, ranges

        handle = self._submit("분리 결과 불러오기", job)
        handle.finished.connect(lambda r: seq == self._seq and self._set_stems(*r))
        return handle

    def _set_stems(self, raw: dict[str, AudioData], effective: dict[str, AudioData] | None = None,
                   peaks: dict | None = None, applied: tuple = ()) -> None:
        """raw: 분리 결과 그대로, effective: 보컬 없는 구간을 적용한 사본(재생·파형·내보내기용), applied: 적용한 구간."""
        effective = raw if effective is None else effective
        for name in ("vocals", "instrumental"):
            if name in effective and self.audio is not None:
                self.player.set_track(name, effective[name])
            else:
                self.player.set_track(name, None)
        self.raw_stems = {n: a for n, a in raw.items() if n in self.player.tracks}
        self.stems = {n: a for n, a in effective.items() if n in self.player.tracks}
        self._applied_mask = tuple(applied) if self.stems else ()
        for name in ("vocals", "instrumental"):
            self.peaks.pop(name, None)
        self.peaks.update(peaks or {})
        self.tracksChanged.emit()
        if self.raw_stems and self._applied_mask != self.mask_ranges:
            self._remask()  # 작업 중에 구간이 바뀌었다

    def import_stems(self, vocals_path: str, instrumental_path: str | None = None, tool: str = "") -> TaskHandle | None:
        """다른 프로그램으로 만든 분리 결과를 stems/models/로 가져온다. 기존 분리 결과는 각자 모델 폴더에 그대로 남는다.

        가져온 결과는 manifest model='external'이고, 다음 분석은 Demucs 대신 이 결과로 피치·음표를 계산한다.
        """
        if self.folder is None or self.audio is None or self.busy:
            return None
        from amp.engine import stem_import

        folder, audio, seq, ranges = self.folder, self.audio, self._seq, self.mask_ranges
        sha = folder.meta.original.sha256

        def job(progress, token):
            vocals, inst, rep = stem_import.prepare(audio, vocals_path, instrumental_path, tool, progress)
            token.raise_if_cancelled()
            progress(0.8, "분리 결과 저장 중")  # 기존 결과는 stems/models/<키>/에 그대로 남는다 (모델 비교에서 다시 고를 수 있음)
            manifest = stem_import.write_stems(folder.paths.stems, vocals, inst, sha, rep, vocals_path, instrumental_path)
            raw = stem_store.load(folder.paths.stems, manifest)  # 저장한 그대로(24bit) 다시 읽어 재생·분석과 같게
            eff = apply_no_vocal(raw, ranges)
            peaks = {n: WaveformPeaks.build(a.mono(), a.sample_rate) for n, a in eff.items()}
            return manifest, raw, eff, peaks, ranges, rep

        handle = self._submit("외부 분리 결과 가져오기", job)
        self._track_analysis(handle)  # ended 연결 순서상 바쁨 해제가 아래 announce보다 먼저 실행된다
        imported = []

        def done(r) -> None:
            manifest, raw, eff, peaks, applied, rep = r
            if seq != self._seq:
                return
            self.manifest = manifest
            self.stems_reason = ""
            self._set_stems(raw, eff, peaks, applied)
            log.info("외부 분리 결과 가져옴: %s (%s)", rep.vocals_file, rep.summary())
            imported.append(rep)

        def announce() -> None:
            # 작업이 완전히 끝난 뒤(바쁨 해제 뒤)에 알린다. 받는 쪽이 곧바로 재분석을 시작할 수 있게
            if imported and seq == self._seq:
                self.stemsImported.emit(imported[0])

        handle.finished.connect(done)
        handle.ended.connect(announce)
        return handle

    def set_no_vocal_mask(self, ranges) -> None:
        """보컬 없는 구간을 보컬·반주 트랙(재생·파형·내보내기)에 적용한다. 빈 값이면 분리 결과 그대로."""
        ranges = tuple(tuple(float(x) for x in r) for r in ranges)
        if ranges == self.mask_ranges:
            return
        self.mask_ranges = ranges
        if self.raw_stems and ranges != self._applied_mask:
            self._remask()

    def _remask(self) -> TaskHandle:
        raw, ranges, seq = dict(self.raw_stems), self.mask_ranges, self._seq

        def job(progress, token):
            eff = apply_no_vocal(raw, ranges)
            return raw, eff, {n: WaveformPeaks.build(a.mono(), a.sample_rate) for n, a in eff.items()}, ranges

        def done(r) -> None:
            if seq != self._seq:
                return
            self._set_stems(*r)
            if ranges:
                self.message.emit(f"보컬 없는 구간 {len(ranges)}곳을 보컬 트랙에서 비우고 그 소리를 반주에 더했습니다 "
                                  "(재생·내보내기. 분리 결과 파일은 그대로).", 8000)

        handle = self._submit("보컬 없는 구간 적용", job)
        handle.finished.connect(done)
        return handle

    def track_audio(self, name: str) -> AudioData | None:
        return self.audio if name == "original" else self.stems.get(name)

    def track_path(self, name: str) -> Path | None:
        if name == "original":
            return self.original.path if self.original else None
        if self.manifest is not None and self.folder is not None:
            return self.folder.paths.stems / self.manifest.files[name]["path"]
        return None

    # --- 저장 -----------------------------------------------------------------

    def save(self, ui: dict | None = None, playback: dict | None = None, analysis_settings: dict | None = None,
             force: bool = False) -> bool:
        """저장 충돌(다른 곳에서 파일이 바뀜)이면 False를 돌려주고 last_conflict에 바뀐 파일 이름을 둔다.

        이때는 오류 창을 띄우지 않는다. 덮어쓸지는 메인 창이 묻고, 덮어쓰기로 하면 force=True로 다시 부른다.
        """
        self.last_conflict: list[str] = []
        if self.folder is None:
            self.error.emit("저장할 수 없습니다", "열린 프로젝트가 없습니다. 먼저 오디오를 열어 프로젝트를 만드세요.", "")
            return False
        try:
            if analysis_settings is not None:
                self.folder.meta.analysis_settings = analysis_settings
            self.folder.save(self.doc.project, ui=ui, playback=playback, force=force)
        except SaveConflictError as e:
            log.warning("저장 충돌: %s", ", ".join(e.changed))
            self.last_conflict = e.changed
            return False
        except (ValueError, TypeError) as e:  # 직렬화할 수 없는 값. 원자적 쓰기라 기존 파일은 그대로다
            log.exception("저장 중 직렬화 오류")
            self.error.emit("저장하지 못했습니다", f"저장할 수 없는 값이 있습니다 ({e}). 기존 파일은 바뀌지 않았습니다.", "")
            return False
        except (AmpError, OSError) as e:
            self.error.emit("저장하지 못했습니다", str(e), "")
            return False
        self.doc.mark_saved(str(self.folder.root))
        self.message.emit(f"저장했습니다: {self.folder.root}", 5000)
        return True

    def save_as(self, parent: str, name: str, ui: dict | None = None, playback: dict | None = None) -> TaskHandle | None:
        if self.folder is None:
            return None
        src = self.folder

        def job(progress, token):
            return copy_project(src, parent, name, progress)

        handle = self._submit("다른 이름으로 저장", job)

        def done(new_folder: ProjectFolder) -> None:
            project = self.doc.project
            try:
                new_folder.save(project, ui=ui, playback=playback)
            except (AmpError, OSError) as e:
                self.error.emit("저장하지 못했습니다", str(e), "")
                return
            lock = ProjectLock(new_folder.root)
            try:
                lock.acquire(new_folder.meta.id)
            except (ProjectLockedError, OSError) as e:
                self.error.emit("새 프로젝트를 잠그지 못했습니다", str(e), "")
                return
            if self._lock is not None:
                self._lock.release()
            self._lock = lock
            self.folder = new_folder
            self._remember(new_folder.root)
            self.doc.mark_saved(str(new_folder.root))
            self.opened.emit()
            self.load_stems()
            self.message.emit(f"새 프로젝트로 저장했습니다: {new_folder.root}", 8000)

        handle.finished.connect(done)
        return handle

    def relink(self, path: str) -> bool:
        if self.folder is None:
            return False
        try:
            same = self.folder.relink(path)
            self.folder.write_meta()
        except (AmpError, OSError) as e:
            self.error.emit("원본을 연결하지 못했습니다", str(e), "")
            return False
        if not same:
            self.notice.emit("원본 내용이 다릅니다",
                             "연결한 파일의 내용이 프로젝트를 만들 때의 원본과 다릅니다. 분리·분석 결과는 다시 만들어야 합니다.")
        self.original = self.folder.resolve_original()
        self.originalChanged.emit()
        self.load_audio(path)
        return same

    # --- 분석 -----------------------------------------------------------------

    def _request(self, settings: AnalysisSettings, force_separation: bool) -> TranscriptionRequest:
        if self.folder is None or self.audio is None or self.original is None or self.original.path is None:
            raise AnalysisError("분석할 오디오가 없습니다. 원본 파일을 확인하세요.")
        return TranscriptionRequest(
            audio=self.audio, source_path=str(self.original.path), source_sha256=self.folder.meta.original.sha256,
            settings=settings, stems_dir=str(self.folder.paths.stems), cache_root=str(self.folder.paths.cache),
            force_separation=force_separation, numba_cache_dir=app_data_dir("numba"),
            syllable_starts=self.syllable_starts(), model_files=dict(self.model_files),
        )

    # --- 베이스 TAB (v0.6.3) -----------------------------------------------------

    def run_bass(self, options, device: str = "auto", use_original: bool = False,
                 allow_separation: bool = True) -> TaskHandle | None:
        """베이스 stem을 찾거나(없으면 htdemucs로 분리해 보관 결과에 더함) 베이스 피치·음표를 구한다 (작업 스레드).
        결과(BassOutput)는 메인 창이 받아 베이스 작업본에 반영한다. 보컬 분석 결과·분리 결과 파일은 바꾸지 않는다."""
        if self.busy:
            return None
        if self.folder is None or self.audio is None or self.original is None or self.original.path is None:
            self.error.emit("베이스를 분석할 수 없습니다", "분석할 오디오가 없습니다. 원본 파일을 확인하세요.", "")
            return None
        from amp.engine.bass_pipeline import BassPipeline, BassRequest

        req = BassRequest(str(self.original.path), self.folder.meta.original.sha256, self.audio.duration, options,
                          stems_dir=str(self.folder.paths.stems), cache_root=str(self.folder.paths.cache),
                          device=device, use_original=use_original, allow_separation=allow_separation,
                          numba_cache_dir=app_data_dir("numba"))

        def job(progress, token):
            return BassPipeline().run(req, progress, token)

        handle = self._submit("베이스 분석", job)
        self._track_analysis(handle)
        return handle

    def extract_instruments(self, model: str, device: str = "auto", names: tuple | None = None,
                            source: str = "original") -> TaskHandle | None:
        """v0.6.4: 악기 소리(드럼·베이스·기타·피아노·나머지)를 분리 보관소에 넣는다 (작업 스레드).
        보컬·반주·이미 있는 악기 파일과 지금 쓰는 분리 결과는 바꾸지 않는다."""
        if self.busy:
            return None
        if self.folder is None or self.original is None or self.original.path is None:
            self.error.emit("악기 소리를 분리할 수 없습니다", "분석할 오디오가 없습니다. 원본 파일을 확인하세요.", "")
            return None
        from amp.engine import instruments

        folder, src = self.folder, str(self.original.path)

        def job(progress, token):
            return instruments.extract(folder.paths.stems, folder.meta.original.sha256, src, model, run_worker,
                                       progress, token, device, names, app_data_dir("numba"), source)

        handle = self._submit("악기 소리 분리", job)
        self._track_analysis(handle)
        return handle

    # --- 피치 모델 (v0.7.0) --------------------------------------------------------

    def pitch_source(self) -> dict | None:
        """피치 모델이 분석할 소리: 지금 보컬 분석에 쓰는 분리 결과의 보컬, 없으면 원본. 해시는 작업 스레드에서 계산한다."""
        if self.folder is None or self.audio is None:
            return None
        m = self.manifest
        if m is not None and "vocals" in m.files:
            from amp.engine.hybrid import HYBRID
            from amp.engine.model_store import MODEL_LABELS
            from amp.engine.stem_import import EXTERNAL

            kind = "hybrid" if m.model == HYBRID else "external" if m.model == EXTERNAL else "vocals"
            return {"path": str(self.folder.paths.stems / m.files["vocals"]["path"]), "kind": kind,
                    "input_id": f"stems:{m.stems_id}", "label": f"보컬 ({MODEL_LABELS.get(m.model, m.model)})"}
        if self.original is None or self.original.path is None:
            return None
        return {"path": str(self.original.path), "kind": "original", "input_id": "", "label": "원본 (분리하지 않음)"}

    def run_pitch_models(self, models: tuple, analysis_range: tuple | None = None, force: bool = False,
                         device: str = "auto") -> TaskHandle | None:
        """v0.7.0: 피치 모델을 차례로 실행해 analysis/pitch_models/에 따로 저장한다. 음표·가사는 바꾸지 않는다."""
        if self.busy:
            return None
        src = self.pitch_source()
        if src is None:
            self.error.emit("피치 모델 분석을 할 수 없습니다", "프로젝트와 오디오를 먼저 여세요.", "")
            return None
        from amp.engine import pitch_runs as PR
        from amp.engine.model_store import file_sha256

        folder, duration = self.folder, float(self.audio.duration)

        def job(progress, token):
            inp = PR.PitchInput(src["path"], src["kind"], src["input_id"], file_sha256(src["path"]),
                                folder.meta.original.sha256, duration, src["label"])
            req = PR.PitchRunRequest(inp, str(folder.paths.analysis), tuple(models), device=device,
                                     analysis_range=analysis_range, force=force, cache_root=str(folder.paths.cache),
                                     numba_cache_dir=app_data_dir("numba"))
            PR.PitchRunStore(folder.paths.analysis).mark_stale(inp)
            return PR.run_models(req, run_worker, progress, token)

        handle = self._submit("피치 모델 분석", job)
        self._track_analysis(handle)
        return handle

    def _pitch_input_job(self):
        src = self.pitch_source()
        if src is None:
            self.error.emit("분석할 수 없습니다", "프로젝트와 오디오를 먼저 여세요.", "")
            return None
        from amp.engine import pitch_runs as PR
        from amp.engine.model_store import file_sha256

        folder, duration = self.folder, float(self.audio.duration)
        return lambda: PR.PitchInput(src["path"], src["kind"], src["input_id"], file_sha256(src["path"]),
                                     folder.meta.original.sha256, duration, src["label"])

    def run_multipitch(self, max_voices: int = 2) -> TaskHandle | None:
        """v0.7.0: 다중 피치 → 성부 (analysis/voice_tracks.json·multi_pitch.npz). 이전 결과는 보관하고 사용자 성부는 남긴다."""
        if self.busy:
            return None
        make = self._pitch_input_job()
        if make is None:
            return None
        from amp.engine import voices as V

        folder = self.folder

        def job(progress, token):
            return V.run_analysis(make(), folder.paths.analysis, run_worker, max_voices, progress, token,
                                  app_data_dir("numba"))

        handle = self._submit("다중 피치 분석", job)
        self._track_analysis(handle)
        return handle

    def run_voice_band(self, vs, vid: str, t0: float, t1: float, fmin: float, fmax: float) -> TaskHandle | None:
        if self.busy:
            return None
        make = self._pitch_input_job()
        if make is None:
            return None
        from amp.engine import voices as V

        def job(progress, token):
            return V.run_band(make(), vs, vid, t0, t1, fmin, fmax, run_worker, progress, token, app_data_dir("numba"))

        handle = self._submit("성부 음역 다시 분석", job)
        self._track_analysis(handle)
        return handle

    def run_acoustics(self, snap) -> TaskHandle | None:
        """v0.7.0: 비브라토·호흡·프레이즈·발성·발성 경계 분석. snap은 acoustics.run.Snapshot (음표·가사 스냅샷).
        결과는 analysis/에 따로 저장하고 음표·가사는 바꾸지 않는다."""
        if self.busy:
            return None
        make = self._pitch_input_job()
        if make is None:
            return None
        from amp.engine.acoustics import run as AR

        folder = self.folder

        def job(progress, token):
            inp = make()
            AR.mark_stale(folder.paths.analysis, f"{inp.kind}:{inp.input_id}:{inp.input_hash[:16]}", inp.source_sha256)
            return AR.run(inp, folder.paths.analysis, run_worker, snap, str(folder.paths.cache), progress, token,
                          app_data_dir("numba"))

        handle = self._submit("음향 분석", job)
        self._track_analysis(handle)
        return handle

    # --- 모델 비교 (STEP 5) -----------------------------------------------------

    def model_store(self):
        """프로젝트의 모델별 분리 결과 보관소 (stems/models/). 예전 평면 결과는 처음 열 때 옮긴다."""
        if self.folder is None:
            return None
        from amp.engine.model_store import ModelStore

        return ModelStore(self.folder.paths.stems, self.folder.meta.original.sha256)

    def run_model_compare(self, settings: AnalysisSettings, specs: list, force: bool = False) -> TaskHandle | None:
        """고른 모델을 하나씩 분리하고 같은 설정으로 채보한다. 지금 분석에 쓰는 결과는 바꾸지 않는다."""
        if self.busy:
            return None
        try:
            req = self._request(settings, False)
        except AnalysisError as e:
            self.error.emit("모델을 비교할 수 없습니다", str(e), "")
            return None
        from amp.engine import model_compare

        def job(progress, token):
            return model_compare.run_compare(req, list(specs), progress, token, force=force)

        handle = self._submit("분리 모델 비교", job)
        self._track_analysis(handle)
        handle.finished.connect(self._record_compare_runs)
        return handle

    def transcribe_entries(self, settings: AnalysisSettings, keys: list[str]) -> TaskHandle | None:
        """보관한 분리 결과를 같은 설정으로 채보만 한다 (가져온 결과 비교, 프로젝트를 다시 연 뒤 등)."""
        if self.busy:
            return None
        try:
            req = self._request(settings, False)
        except AnalysisError as e:
            self.error.emit("모델별 채보를 할 수 없습니다", str(e), "")
            return None
        from amp.engine import model_compare

        def job(progress, token):
            return model_compare.transcribe_keys(req, list(keys), progress, token)

        handle = self._submit("모델별 채보", job)
        self._track_analysis(handle)
        return handle

    def _record_compare_runs(self, run) -> None:
        for oc in getattr(run, "outcomes", ()):
            out = getattr(oc, "output", None)
            if out is not None:  # 모델별 자동 분석도 실행 기록으로 남긴다 (나중에 비교·복구용)
                self._record_run(out.result, f"compare:{oc.spec.model}", out.timings, out.reused, out.resources)

    def render_hybrid(self, plan, current_keys: dict | None = None) -> TaskHandle | None:
        """v0.6: Hybrid 구간 계획대로 보컬·반주를 합쳐 보관소 항목으로 만든다 (작업 스레드). 원본 stem은 읽기만 한다."""
        if self.busy:
            return None
        store = self.model_store()
        if store is None:
            self.error.emit("Hybrid 트랙을 만들 수 없습니다", "열린 프로젝트가 없습니다.", "")
            return None
        from amp.engine import hybrid

        def job(progress, token):
            return hybrid.render(plan, store, progress, lambda: token.cancelled, current_keys)

        handle = self._submit("Hybrid 트랙 만들기", job)
        self._track_analysis(handle)
        return handle

    def current_store_keys(self) -> dict[str, str]:
        """모델 → 지금 설정(Demucs 버전·RoFormer 체크포인트·설정 파일)으로 분리하면 쓰일 보관소 키.
        Hybrid 구간이 이전 체크포인트 결과를 쓰는지 판단할 때 쓴다. 체크포인트를 고르지 않았으면 RoFormer는 뺀다."""
        store = self.model_store()
        if store is None:
            return {}
        from amp.engine.model_store import identity_for, store_key

        out = {}
        for model in ("htdemucs", "mel_roformer"):
            if model == "mel_roformer" and not (self.model_files or {}).get("checkpoint"):
                continue
            try:
                ident = identity_for(model, store.source_sha256, dict(self.model_files))
            except OSError:
                continue
            found = store.find(ident)
            out[model] = found.key if found is not None else store_key(ident)
        return out

    def activate_stems(self, key: str) -> None:
        """이 분리 결과를 분석에 쓴다 (stems/stems.json만 바꿈, 다른 결과 파일은 그대로). 재생 트랙을 다시 불러온다."""
        store = self.model_store()
        if store is None:
            raise AnalysisError("열린 프로젝트가 없습니다.")
        manifest = store.activate(key)
        log.info("분석 소스 변경: %s (%s)", key, manifest.model)
        self.load_stems()

    def syllable_starts(self) -> tuple[float, ...]:
        """음표 검출에 넘길 음절 시작: 정렬 모델·사용자·음표에서 온 시간만 (인식 단어 시각 추정값은 쓰지 않는다)."""
        al = self.doc.lyrics.alignment
        return tuple(sorted(t.start for t in al.timings.values()
                            if t.aligned and not t.stale and (t.user_edited or t.start_source in ("model", "user", "notes"))))

    def run_analysis(self, settings: AnalysisSettings, force_separation: bool = False) -> TaskHandle | None:
        if self.busy:
            return None
        try:
            req = self._request(settings, force_separation)
        except AnalysisError as e:
            self.error.emit("분석할 수 없습니다", str(e), "")
            return None
        audio, seq, ranges = self.audio, self._seq, self.mask_ranges

        def job(progress, token):
            # 다시 분리해도 다른 모델·가져온 결과는 stems/models/<키>/에 그대로 남는다 (같은 모델·설정 결과만 바뀜)
            out = TranscriptionPipeline().run(req, progress, token)
            # 파형 요약은 작업 스레드에서 만든다. 화면 스레드에서 만들면 5분 곡 스템 둘에 0.7초쯤 화면이 멈췄다
            # (사용자 Windows 로그: 분석 실행 직후 0.65~1.05초 멈춤)
            eff = apply_no_vocal(out.stems, ranges) if out.stems else {}
            return out, eff, {name: WaveformPeaks.build(a.mono(), a.sample_rate) for name, a in eff.items()}

        handle = self._submit("보컬 분석", job)
        self._track_analysis(handle)

        def done(result) -> None:
            out, eff, peaks = result
            if seq != self._seq or self.audio is not audio:
                return
            self.manifest = out.manifest
            self.stems_reason = ""
            if out.stems:
                self._set_stems(out.stems, eff, peaks, ranges)
            reused = [k for k, v in out.reused.items() if v]
            if reused:
                self.message.emit("재사용한 단계: " + ", ".join({"separation": "보컬 분리", "pitch": "피치 추출",
                                                                "tempo": "BPM"}[k] for k in reused), 8000)
            self._record_run(out.result, "full", out.timings, out.reused, out.resources)
            self.analysisReady.emit(out.result)

        handle.finished.connect(done)
        return handle

    def redetect(self, settings: AnalysisSettings) -> TaskHandle | None:
        result = self.doc.project.analysis
        if result is None or result.pitch_curve is None or self.busy:
            self.error.emit("음표를 다시 검출할 수 없습니다", "피치 분석 결과가 없습니다. 먼저 보컬 분석을 실행하세요.", "")
            return None
        starts = self.syllable_starts()

        def job(progress, token):
            t0 = time.perf_counter()
            new = redetect_notes(result, settings, starts)
            return new, time.perf_counter() - t0

        handle = self._submit("음표 다시 검출", job)
        self._track_analysis(handle)

        def done(out) -> None:
            new, seconds = out
            self._record_run(new, "redetect", {"notes": seconds, "total": seconds})
            self.analysisReady.emit(new)

        handle.finished.connect(done)
        return handle

    # --- 실행 기록 ----------------------------------------------------------------

    def _record_run(self, result, kind: str, timings: dict | None = None, reused: dict | None = None,
                    resources: dict | None = None) -> None:
        """성공한 분석만 analysis/runs/에 남긴다. 실패해도 분석 결과 반영은 막지 않는다."""
        if self.folder is None:
            return
        try:
            record = runs.record_from_result(
                result, kind, timings=timings, reused=reused, resources=resources,
                score_summary=dataclasses.asdict(summarize(self.doc.project)), app_version=__version__,
                quality=self._run_quality())
            path = runs.save_run(self.folder.paths.analysis, record)
            log.info("분석 실행 기록 저장: %s (음표 %d개, %s)", path.name, len(record.notes),
                     ", ".join(f"{k} {v:.2f}s" for k, v in record.timings_s.items()))
        except Exception as e:  # noqa: BLE001 - 기록 실패가 분석 결과 반영을 막으면 안 된다
            log.exception("분석 실행 기록을 저장하지 못했습니다: %s", e)
            self.message.emit("분석 실행 기록을 저장하지 못했습니다 (분석 결과에는 영향이 없습니다).", 8000)

    def _run_quality(self) -> dict:
        """실행 기록에 남길 v0.6 지표: 교정 제안 대기·적용 수 (정답 없는 지표)."""
        st = self.doc.project.suggestions
        if st is None:
            return {}
        c = st.counts()
        return {"suggestions_pending": c.get("pending", 0), "suggestions_applied": c.get("accepted", 0)}

    def compare_recent_runs(self) -> tuple[Path, runs.RunComparison]:
        """가장 최근 두 실행을 비교해 analysis/reports/에 Markdown 보고서를 쓴다."""
        if self.folder is None:
            raise AmpError("열린 프로젝트가 없습니다.")
        records = runs.list_runs(self.folder.paths.analysis)
        if len(records) < 2:
            raise AmpError(f"비교하려면 분석 실행 기록이 2개 이상 필요합니다 (현재 {len(records)}개).")
        a, b = records[-2], records[-1]
        try:
            path = runs.write_report(self.folder.paths.analysis, a, b)
        except runs.RunCompareError as e:  # 원본을 다른 파일로 다시 연결한 뒤의 실행 등
            raise AmpError(str(e)) from e
        log.info("실행 비교 보고서: %s", path)
        return path, runs.compare_runs(a, b)

    def run_sample(self) -> TaskHandle | None:
        if self.audio is None or self.busy:
            return None
        from amp.engine.pipeline import AnalysisPipeline, PipelineConfig
        from amp.engine.registry import default_registry

        audio = self.audio
        sha = self.folder.meta.original.sha256 if self.folder else ""
        pipe = AnalysisPipeline(default_registry(), PipelineConfig())
        handle = self._submit("샘플 데이터 생성", lambda p, t: pipe.run(audio, p, t, sha))
        self._track_analysis(handle)
        handle.finished.connect(lambda r: self.audio is audio and self.analysisReady.emit(r))
        return handle

    # --- 가사 ----------------------------------------------------------------

    def lyrics_request(self, lyric_settings: dict, analysis: AnalysisSettings):
        """가사 인식·정렬 입력. 분리한 보컬을 쓴다(분리 안 함 설정이면 원본)."""
        from amp.lyrics.engine import LyricsRequest

        if self.folder is None or self.audio is None:
            raise AnalysisError("열린 프로젝트가 없습니다.")
        if self.manifest is not None and not (self.folder.paths.stems / self.manifest.files["vocals"]["path"]).is_file():
            m, ok, _why = self.folder.stems_status()  # 보관소로 옮겨졌을 수 있다 (STEP 5): 목록을 다시 읽는다
            self.manifest = m if ok else None
        if self.manifest is not None:
            path = str(self.folder.paths.stems / self.manifest.files["vocals"]["path"])
            input_id = "stems-" + self.manifest.stems_id
        elif analysis.separation == "none" and self.original is not None and self.original.path is not None:
            path = str(self.original.path)
            input_id = "orig-" + self.folder.meta.original.sha256[:16]
        else:
            raise AnalysisError("분리된 보컬이 없습니다. 먼저 '보컬 분석 실행'으로 보컬을 분리하세요. "
                                "(반주 없는 음원이면 분석 설정에서 '분리 안 함'을 고르면 원본으로 인식합니다.)")
        return LyricsRequest(
            vocals_path=path, input_id=input_id, cache_root=str(self.folder.paths.cache),
            whisper_root=app_data_dir("models", "whisper"), mms_root=app_data_dir("models", "mms"),
            device=analysis.device, model=str(lyric_settings.get("model", "auto")),
            language=str(lyric_settings.get("language", "auto")),
            no_vocal_ranges=tuple(tuple(r) for r in analysis.no_vocal_ranges), use_cache=analysis.use_cache,
            numba_cache_dir=app_data_dir("numba"))

    @staticmethod
    def _staged(progress, stage: str):
        """진행 메시지 앞에 연속 작업 단계('[1/3]')를 붙인다."""
        if not stage:
            return progress
        return lambda f, msg="": progress(f, f"{stage} {msg}" if msg else stage)

    def run_recognition(self, lyric_settings: dict, analysis: AnalysisSettings, stage: str = "") -> TaskHandle | None:
        if self.busy:
            return None
        try:
            req = self.lyrics_request(lyric_settings, analysis)
        except AnalysisError as e:
            self.error.emit("가사를 인식할 수 없습니다", str(e), "")
            return None
        from amp.lyrics.engine import LyricsEngine

        seq = self._seq
        handle = self._submit(f"{stage} 가사 인식".strip(),
                              lambda p, t: LyricsEngine().recognize(req, self._staged(p, stage), t))
        self._track_analysis(handle)
        handle.finished.connect(lambda out: seq == self._seq and self.recognitionReady.emit(out[0], out[1]))
        return handle

    def run_alignment(self, lyric_settings: dict, analysis: AnalysisSettings, mode: str = "all",
                      ids: list[str] | None = None, stage: str = "") -> TaskHandle | None:
        if self.busy:
            return None
        state = self.doc.lyrics
        if state.text.is_empty:
            self.error.emit("정렬할 가사가 없습니다", "가사를 입력하거나 자동 인식한 뒤 정렬하세요.", "")
            return None
        try:
            req = self.lyrics_request(lyric_settings, analysis)
        except AnalysisError as e:
            self.error.emit("가사를 정렬할 수 없습니다", str(e), "")
            return None
        from amp.lyrics.engine import LyricsEngine

        seq = self._seq
        duration = self.audio.duration if self.audio else None
        handle = self._submit(f"{stage} 가사 시간 정렬".strip(),
                              lambda p, t: LyricsEngine().align(req, state, mode=mode, ids=ids, duration=duration,
                                                                progress=self._staged(p, stage), token=t))
        self._track_analysis(handle)
        handle.finished.connect(lambda out: seq == self._seq and self.alignmentReady.emit(out))
        return handle

    def registry_context(self):
        """모델 등록 정보를 찾을 곳 (v0.7.0). 앱 모델 폴더·사용자가 고른 RoFormer 파일·저장해 둔 해시."""
        from amp.app.lite import lite_mode
        from amp.engine import model_registry as R

        path = app_data_dir(R.FILE_NAME)
        return R.RegistryContext(models_dir=app_data_dir("models"), roformer_files=dict(self.model_files or {}),
                                 lite=lite_mode(), hash_cache=R.load_hash_cache(path)), path

    def model_registry(self, hash_files: bool = False, save: bool = True) -> list:
        """모델 등록 정보를 살펴(필요하면 해시 계산) 앱 데이터 폴더의 model_registry.json에 적는다.
        hash_files=True는 큰 파일을 읽으므로 작업 스레드에서 부른다. 파일을 바꾸거나 내려받지 않는다."""
        from amp.engine import model_registry as R

        ctx, path = self.registry_context()
        statuses = R.scan(ctx, hash_files=hash_files)
        if save:
            try:
                R.save(path, statuses, ctx)
            except OSError:
                pass  # 기록을 못 해도 등록 정보는 쓴다
        return statuses

    def lyrics_install_status(self) -> dict:
        """GUI 프로세스에서 무거운 라이브러리를 불러오지 않고 설치·내려받기 상태만 본다."""
        import importlib.util

        from amp.lyrics import mms

        whisper_dir = Path(app_data_dir("models", "whisper"))
        mms_dir = app_data_dir("models", "mms")
        downloaded = sorted(p.stem for p in whisper_dir.glob("*.pt")) if whisper_dir.is_dir() else []
        return {"whisper": importlib.util.find_spec("whisper") is not None,
                "torch": importlib.util.find_spec("torch") is not None,
                "whisper_models": downloaded, "mms": mms.is_downloaded(mms_dir),
                "models_dir": str(Path(mms_dir).parent)}

    def _track_analysis(self, handle: TaskHandle) -> None:
        self.analysis_handle = handle
        self.busyChanged.emit(True)

        def ended() -> None:
            if self.analysis_handle is handle:
                self.analysis_handle = None
                self.busyChanged.emit(False)

        handle.ended.connect(ended)

    def cancel_analysis(self) -> None:
        if self.analysis_handle is not None:
            self.analysis_handle.cancel()

    def clear_cache(self) -> None:
        if self.folder is not None:
            self.folder.clear_cache()
            self.message.emit("캐시를 비웠습니다. 원본·분리 결과·분석 결과·내보낸 파일은 그대로입니다.", 6000)

    def cache_size(self) -> int | None:
        return self.folder.cache().size_bytes() if self.folder else None

    def probe(self) -> TaskHandle:
        return self._submit("GPU 확인", lambda p, t: run_worker("probe", {}, p, t))

    def check_model_files(self, files: dict) -> TaskHandle:
        """RoFormer 모델 파일을 작업 프로세스에서 읽어 구조를 확인한다 (GUI 프로세스는 torch를 불러오지 않는다)."""
        args = {"checkpoint": files.get("checkpoint", ""), "config": files.get("config", "")}
        return self._submit("모델 파일 확인", lambda p, t: run_worker("check_model", args, p, t))

    # --- 내보내기 --------------------------------------------------------------

    def export(self, jobs: list[tuple[str, str]], options: ExportOptions) -> TaskHandle:
        sources = {track: self.track_audio(track) for track, _ in jobs}

        def job(progress, token):
            results = []
            for i, (track, path) in enumerate(jobs):
                audio = sources[track]
                if audio is None:
                    raise AnalysisError(f"{track} 트랙이 없습니다.")
                export_audio(audio, path, options,
                             lambda f, m, i=i: progress((i + f) / len(jobs), m), lambda: token.cancelled)
                results.append(path)
            return results

        handle = self._submit("내보내기", job)
        handle.finished.connect(lambda paths: self.message.emit(f"{len(paths)}개 파일을 내보냈습니다: "
                                                                + ", ".join(Path(p).name for p in paths), 10000))
        return handle

    def export_file(self, src: str, dst: str, options: ExportOptions, label: str = "내보내기") -> TaskHandle:
        """v0.6.3: 트랙에 올리지 않은 분리 파일(베이스 stem)을 WAV·MP3로 내보낸다. 원래 파일은 읽기만 한다."""
        import soundfile as sf

        def job(progress, token):
            data, sr = sf.read(src, dtype="float32", always_2d=True)
            export_audio(AudioData(data, sr), dst, options, progress, lambda: token.cancelled)
            return dst

        handle = self._submit(label, job)
        handle.finished.connect(lambda p: self.message.emit(f"내보냈습니다: {Path(p).name}", 10000))
        return handle
