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
    busyChanged = Signal(bool)
    taskStarted = Signal(object, str)  # TaskHandle, 표시 문구

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
        self.stems: dict[str, AudioData] = {}
        self.peaks: dict[str, WaveformPeaks] = {}
        self.manifest: stem_store.StemManifest | None = None
        self.stems_reason = "분리 전"
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
        folder, seq = self.folder, self._seq

        def job(progress, token):
            stems = stem_store.load(folder.paths.stems, manifest)
            return stems, {n: WaveformPeaks.build(a.mono(), a.sample_rate) for n, a in stems.items()}

        handle = self._submit("분리 결과 불러오기", job)
        handle.finished.connect(lambda r: seq == self._seq and self._set_stems(r[0], r[1]))
        return handle

    def _set_stems(self, stems: dict[str, AudioData], peaks: dict | None = None) -> None:
        for name in ("vocals", "instrumental"):
            if name in stems and self.audio is not None:
                self.player.set_track(name, stems[name])
            else:
                self.player.set_track(name, None)
        self.stems = {n: a for n, a in stems.items() if n in self.player.tracks}
        for name in ("vocals", "instrumental"):
            self.peaks.pop(name, None)
        self.peaks.update(peaks or {})
        self.tracksChanged.emit()

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
        )

    def run_analysis(self, settings: AnalysisSettings, force_separation: bool = False) -> TaskHandle | None:
        if self.busy:
            return None
        try:
            req = self._request(settings, force_separation)
        except AnalysisError as e:
            self.error.emit("분석할 수 없습니다", str(e), "")
            return None
        audio, seq = self.audio, self._seq
        handle = self._submit("보컬 분석", lambda p, t: TranscriptionPipeline().run(req, p, t))
        self._track_analysis(handle)

        def done(out) -> None:
            if seq != self._seq or self.audio is not audio:
                return
            peaks = {}
            for name, a in out.stems.items():
                peaks[name] = WaveformPeaks.build(a.mono(), a.sample_rate)
            self.manifest = out.manifest
            self.stems_reason = ""
            if out.stems:
                self._set_stems(out.stems, peaks)
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
        def job(progress, token):
            t0 = time.perf_counter()
            new = redetect_notes(result, settings)
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
                score_summary=dataclasses.asdict(summarize(self.doc.project)), app_version=__version__)
            path = runs.save_run(self.folder.paths.analysis, record)
            log.info("분석 실행 기록 저장: %s (음표 %d개, %s)", path.name, len(record.notes),
                     ", ".join(f"{k} {v:.2f}s" for k, v in record.timings_s.items()))
        except Exception as e:  # noqa: BLE001 - 기록 실패가 분석 결과 반영을 막으면 안 된다
            log.exception("분석 실행 기록을 저장하지 못했습니다: %s", e)
            self.message.emit("분석 실행 기록을 저장하지 못했습니다 (분석 결과에는 영향이 없습니다).", 8000)

    def compare_recent_runs(self) -> tuple[Path, runs.RunComparison]:
        """가장 최근 두 실행을 비교해 analysis/reports/에 Markdown 보고서를 쓴다."""
        if self.folder is None:
            raise AmpError("열린 프로젝트가 없습니다.")
        records = runs.list_runs(self.folder.paths.analysis)
        if len(records) < 2:
            raise AmpError(f"비교하려면 분석 실행 기록이 2개 이상 필요합니다 (현재 {len(records)}개).")
        a, b = records[-2], records[-1]
        path = runs.write_report(self.folder.paths.analysis, a, b)
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
