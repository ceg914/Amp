"""열려 있는 프로젝트 문서.

UI 위젯은 데이터를 직접 고치지 않고 이 클래스의 편집 메서드를 호출한다.
편집 메서드는 ``amp.core.editing`` 으로 새 음표를 계산한 뒤 QUndoStack에 명령을 넣는다.
명령이 실행되면 시그널이 나가고, 피아노 롤 같은 뷰가 그 시그널을 받아 화면을 고친다.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QUndoCommand, QUndoStack

from amp.core import editing
from amp.core.errors import EditError
from amp.core.model import AnalysisResult, AudioRef, Note, Project, Score
from amp.core.tempo import TempoMap
from amp.lyrics import linking, pronunciation
from amp.lyrics.model import LyricsState


def _adopt_sample_lyrics(current: LyricsState, result: AnalysisResult) -> LyricsState:
    """샘플 분석기가 만든 가사는 프로젝트에 가사가 없을 때만 받아들인다 (사용자 가사를 덮지 않는다)."""
    if result.lyrics is None or not current.is_empty:
        return current
    from amp.lyrics import io as lyrics_io
    from amp.storage.serialization import lyrics_to_dict

    from dataclasses import replace

    new = lyrics_io.from_legacy(lyrics_to_dict(result.lyrics), dict(result.lyric_links))
    return pronunciation.sync(replace(new, pron_dictionary=current.pron_dictionary))


class ReplaceNotesCommand(QUndoCommand):
    """음표 집합 교체. 이동·길이 변경·분할·병합·추가·삭제를 모두 이 명령 하나로 표현한다.

    음표가 사라지면(나누기·합치기·지우기) 가사 연결을 다시 잇는다. 음절 텍스트·시간은 바꾸지 않는다.
    실행 취소하면 연결도 명령 전 상태로 정확히 돌아간다.
    """

    def __init__(self, doc: "ProjectDocument", removed: list[Note], added: list[Note], text: str) -> None:
        super().__init__(text)
        self._doc = doc
        self._removed = list(removed)
        self._added = list(added)
        self._old_lyrics = doc.project.lyrics
        self._new_lyrics = linking.reconcile(self._old_lyrics, self._removed, self._added)

    def redo(self) -> None:
        self._doc._apply_notes([n.id for n in self._removed], self._added)
        self._doc._apply_lyrics(self._new_lyrics)
        self._doc.selectionRequested.emit([n.id for n in self._added])

    def undo(self) -> None:
        self._doc._apply_notes([n.id for n in self._added], self._removed)
        self._doc._apply_lyrics(self._old_lyrics)
        self._doc.selectionRequested.emit([n.id for n in self._removed])


class SetLyricsCommand(QUndoCommand):
    """가사 상태 교체 (텍스트·음절 시간·연결 편집 모두). 상태가 불변 객체라 스냅샷 교체로 정확히 되돌린다."""

    def __init__(self, doc: "ProjectDocument", old: LyricsState, new: LyricsState, text: str) -> None:
        super().__init__(text)
        self._doc = doc
        self._old = old
        self._new = new

    def redo(self) -> None:
        self._doc._apply_lyrics(self._new)

    def undo(self) -> None:
        self._doc._apply_lyrics(self._old)


class SetFieldCommand(QUndoCommand):
    """STEP 5: 프로젝트의 검토 상태·참조 악보 교체 (불변 객체 스냅샷 교체)."""

    def __init__(self, doc: "ProjectDocument", name: str, old, new, text: str) -> None:
        super().__init__(text)
        self._doc, self._name, self._old, self._new = doc, name, old, new

    def redo(self) -> None:
        self._doc._apply_field(self._name, self._new)

    def undo(self) -> None:
        self._doc._apply_field(self._name, self._old)


class SetTempoCommand(QUndoCommand):
    ID = 1001

    def __init__(self, doc: "ProjectDocument", old: TempoMap, new: TempoMap) -> None:
        super().__init__("템포 변경")
        self._doc = doc
        self._old = old
        self._new = new

    def id(self) -> int:  # noqa: A003 - Qt API 이름
        return self.ID

    def mergeWith(self, other: QUndoCommand) -> bool:  # 연속된 BPM 조정을 하나로 묶는다
        if not isinstance(other, SetTempoCommand):
            return False
        self._new = other._new
        return True

    def redo(self) -> None:
        self._doc._apply_tempo(self._new)

    def undo(self) -> None:
        self._doc._apply_tempo(self._old)


class ApplyAnalysisCommand(QUndoCommand):
    """새 분석 결과를 반영한다: 원본 분석 교체 + 음표 교체(보호 음표는 유지) + 템포. 실행 취소 가능."""

    def __init__(self, doc: "ProjectDocument", new: AnalysisResult, plan, text: str) -> None:
        super().__init__(text)
        self._doc = doc
        self._old_analysis = doc.project.analysis
        self._new_analysis = new
        self._removed = list(plan.removed)
        self._added = list(plan.added)
        self._old_tempo = doc.score.tempo
        self._new_tempo = plan.tempo
        self._old_lyrics = doc.project.lyrics
        lyrics = linking.reconcile(self._old_lyrics, self._removed, self._added)
        self._new_lyrics = _adopt_sample_lyrics(lyrics, new)

    def redo(self) -> None:
        self._doc._apply_analysis_state(self._new_analysis, [n.id for n in self._removed], self._added,
                                        self._new_tempo, self._new_lyrics)

    def undo(self) -> None:
        self._doc._apply_analysis_state(self._old_analysis, [n.id for n in self._added], self._removed,
                                        self._old_tempo, self._old_lyrics)


class ProjectDocument(QObject):
    projectReset = Signal()  # 프로젝트 전체가 바뀜 (새로 만들기·열기·분석 적용)
    lyricsChanged = Signal()  # 가사 텍스트·음절 시간·음표 연결 중 무엇이든 바뀜
    analysisChanged = Signal()  # 원본 분석이 바뀜 (재분석 적용·되돌리기) — 피아노 롤 전체 다시 그림
    notesReplaced = Signal(list, list)  # 지운 id 목록, 추가/교체된 Note 목록
    tempoChanged = Signal()
    reviewChanged = Signal()  # STEP 5: 검토 상태
    referenceChanged = Signal()  # STEP 5: 참조 악보·정렬
    hybridChanged = Signal()  # v0.6: Hybrid 보컬 구간 계획
    suggestionsChanged = Signal()  # v0.6: 교정 제안 상태
    selectionRequested = Signal(list)  # 선택할 음표 id 목록
    dirtyChanged = Signal(bool)
    filePathChanged = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project = Project()
        self.file_path: str | None = None
        self.undo_stack = QUndoStack(self)
        self.revision = 0  # 내용이 바뀔 때마다 증가. 자동 저장이 변경 여부를 판단할 때 쓴다.
        self._extra_dirty = False  # 실행 취소 기록 밖의 변경 (분석 결과 적용 등)
        self._last_dirty = False
        self.undo_stack.cleanChanged.connect(self._on_clean_changed)

    # --- 상태 -----------------------------------------------------------

    @property
    def score(self) -> Score:
        return self.project.score

    @property
    def lyrics(self) -> LyricsState:
        return self.project.lyrics

    @property
    def is_dirty(self) -> bool:
        return self._extra_dirty or not self.undo_stack.isClean()

    @Slot(bool)
    def _on_clean_changed(self, _clean: bool) -> None:
        # 람다 대신 메서드로 연결해야 문서가 삭제될 때 연결도 함께 끊긴다
        self._emit_dirty()

    def _emit_dirty(self) -> None:
        d = self.is_dirty
        if d != self._last_dirty:
            self._last_dirty = d
            self.dirtyChanged.emit(d)

    def _touch(self) -> None:
        self.revision += 1

    def note_status(self, note: Note) -> str:
        return self.project.note_status(note)

    # --- 문서 전체 교체 --------------------------------------------------

    def reset(self, project: Project, file_path: str | None = None, *, dirty: bool = False) -> None:
        self.project = project
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self._extra_dirty = dirty
        self._touch()
        self._set_path(file_path)
        self.projectReset.emit()
        self._emit_dirty()

    def set_audio(self, audio: AudioRef) -> None:
        """새 오디오로 빈 프로젝트를 시작한다."""
        self.reset(Project(audio=audio), None, dirty=True)

    def set_analysis(self, result: AnalysisResult) -> None:
        """분석 원본을 저장하고 작업본을 원본 복사본으로 새로 만든다. 실행 취소 기록은 지운다."""
        self.project.analysis = result
        self.project.score = Score.from_analysis(result)
        lyrics, _ = linking.prune(self.project.lyrics, [n.id for n in self.project.score.notes()])
        self.project.lyrics = _adopt_sample_lyrics(lyrics, result)
        self.undo_stack.clear()
        self._extra_dirty = True
        self._touch()
        self.projectReset.emit()
        self._emit_dirty()

    def mark_modified(self) -> None:
        """실행 취소 기록 밖에서 프로젝트 내용이 바뀜 (v0.6.3 베이스 작업본은 베이스 문서가 따로 기록한다).
        저장·자동 저장 대상이 된다."""
        self._extra_dirty = True
        self._touch()
        self._emit_dirty()

    def relink_audio(self, audio: AudioRef) -> None:
        """프로젝트의 오디오 참조만 바꾼다 (파일을 옮겼거나 다시 연결할 때)."""
        self.project.audio = audio
        self._extra_dirty = True
        self._touch()
        self._emit_dirty()

    def mark_saved(self, path: str) -> None:
        self._extra_dirty = False
        self.undo_stack.setClean()
        self._set_path(path)
        self._emit_dirty()

    def _set_path(self, path: str | None) -> None:
        self.file_path = path
        self.filePathChanged.emit(path or "")

    # --- 명령이 호출하는 내부 반영 함수 ---------------------------------------

    def _apply_notes(self, removed_ids: list[str], added: list[Note]) -> None:
        self.score.replace_notes(removed_ids, added)
        self._touch()
        self.notesReplaced.emit(removed_ids, added)

    def _apply_lyrics(self, state: LyricsState) -> None:
        if state is self.project.lyrics:
            return
        self.project.lyrics = state
        self._touch()
        self.lyricsChanged.emit()

    def set_review_state(self, new, text: str = "검토 상태") -> bool:
        if new == self.project.review:
            return False
        self.undo_stack.push(SetFieldCommand(self, "review", self.project.review, new, text))
        return True

    def set_reference(self, new, text: str = "참조 악보") -> bool:
        if new == self.project.reference:
            return False
        self.undo_stack.push(SetFieldCommand(self, "reference", self.project.reference, new, text))
        return True

    def set_hybrid(self, new, text: str = "Hybrid 구간") -> bool:
        """v0.6: Hybrid 구간 계획 교체 (실행 취소 가능). None이면 계획을 지운다."""
        if new == self.project.hybrid:
            return False
        self.undo_stack.push(SetFieldCommand(self, "hybrid", self.project.hybrid, new, text))
        return True

    def set_suggestions(self, new, text: str = "교정 제안") -> bool:
        """v0.6: 교정 제안 목록·처리 상태 교체 (실행 취소 가능)."""
        if new == self.project.suggestions:
            return False
        self.undo_stack.push(SetFieldCommand(self, "suggestions", self.project.suggestions, new, text))
        return True

    def apply_suggestions(self, ids: Iterable[str]) -> tuple[int, list[str]]:
        """v0.6: 고른 교정 제안을 적용한다. 한 번에 실행 취소된다. (적용 수, 적용하지 못한 이유 목록).

        제안마다 지금 음표 기준으로 다시 계산하므로, 앞 제안이 뒤 제안의 대상 음표를 바꿨으면 뒤 제안은 건너뛴다.
        적용한 음표는 사용자 수정 음표가 되어 재분석이 덮어쓰지 않는다. 가사 연결은 음표 교체 명령이 다시 잇는다.
        """
        from amp.compare import suggestions as S

        state = self.project.suggestions
        if state is None:
            return 0, ["제안이 없습니다."]
        items = sorted((s for s in (state.get(i) for i in dict.fromkeys(ids)) if s is not None),
                       key=lambda s: s.start_time)
        applied, skipped, done = 0, [], []
        ready = []
        for s in items:  # 지금 상태로 적용할 수 있는 것이 하나도 없으면 빈 실행 취소 기록을 만들지 않는다
            try:
                S.plan_apply(self.project, s)
                ready.append(s)
            except EditError as e:
                skipped.append(f"{s.type_label} {s.start_time:.2f}s: {e}")
        items = ready
        if not items:
            return 0, skipped
        self.undo_stack.beginMacro(f"교정 제안 {len(items)}개 적용")
        try:
            for s in items:
                try:
                    removed, added = S.plan_apply(self.project, s)
                except EditError as e:
                    skipped.append(f"{s.type_label} {s.start_time:.2f}s: {e}")
                    continue
                self._push(removed, added, f"교정 제안: {s.type_label}")
                done.append(s.suggestion_id)
                applied += 1
            if done:
                self.undo_stack.push(SetFieldCommand(self, "suggestions", state,
                                                     state.with_status(done, S.ACCEPTED), "교정 제안 상태"))
        finally:
            self.undo_stack.endMacro()
        return applied, skipped

    def _apply_field(self, name: str, value) -> None:
        setattr(self.project, name, value)  # Project는 다른 곳도 같은 객체를 들고 있어 자리에서 바꾼다
        self._touch()
        {"review": self.reviewChanged, "reference": self.referenceChanged, "hybrid": self.hybridChanged,
         "suggestions": self.suggestionsChanged}[name].emit()

    def push_lyrics(self, new: LyricsState, text: str) -> bool:
        """가사 편집 결과를 실행 취소 기록에 넣는다. 바뀐 것이 없으면 넣지 않는다.

        텍스트가 바뀌었으면 발음도 같은 명령 안에서 맞춘다(자동 발음은 새로, 사용자 발음은 지키고 재검토 표시).
        """
        old = self.project.lyrics
        if new is not old and new.text != old.text:
            new = pronunciation.sync(new)
        if new is old or (new.text == old.text and new.alignment == old.alignment and new.recognition == old.recognition
                          and new.pronunciation == old.pronunciation and new.pron_dictionary == old.pron_dictionary):
            return False
        self.undo_stack.push(SetLyricsCommand(self, old, new, text))
        return True

    def _apply_tempo(self, tempo: TempoMap) -> None:
        self.score.tempo = tempo
        self._touch()
        self.tempoChanged.emit()

    def _push(self, removed: list[Note], added: list[Note], text: str) -> None:
        self.undo_stack.push(ReplaceNotesCommand(self, removed, added, text))

    def _notes(self, ids: Iterable[str]) -> list[Note]:
        return [self.score.get(i) for i in dict.fromkeys(ids)]

    # --- 편집 API (모두 실행 취소 가능) -------------------------------------

    def add_note(self, start: float, end: float, pitch: int) -> str:
        note = editing.create_note(start, end, pitch)
        self._push([], [note], "음표 추가")
        return note.id

    def delete_notes(self, ids: Iterable[str]) -> int:
        notes = self._notes(ids)
        if notes:
            self._push(notes, [], f"음표 {len(notes)}개 삭제")
        return len(notes)

    def move_notes(self, ids: Iterable[str], dt: float, dpitch: int) -> bool:
        notes = self._notes(ids)
        dt, dpitch = editing.clamp_move(notes, dt, dpitch)
        if not notes or (abs(dt) < 1e-9 and dpitch == 0):
            return False
        self._push(notes, editing.move_notes(notes, dt, dpitch), "음표 이동")
        return True

    def resize_notes(self, ids: Iterable[str], edge: str, delta: float) -> bool:
        notes = self._notes(ids)
        delta = editing.clamp_resize(notes, edge, delta)
        if not notes or abs(delta) < 1e-9:
            return False
        self._push(notes, editing.resize_notes(notes, edge, delta), "음표 길이 변경")
        return True

    def split_note(self, note_id: str, at: float) -> tuple[str, str]:
        note = self.score.get(note_id)
        left, right = editing.split_note(note, at)
        self._push([note], [left, right], "음표 분할")
        return left.id, right.id

    def split_notes_at(self, ids: Iterable[str], at: float) -> int:
        """선택한 음표 중 ``at`` 을 지나는 음표를 모두 나눈다 (한 번에 실행 취소)."""
        targets = [
            n for n in self._notes(ids)
            if n.start + editing.MIN_NOTE_DURATION <= at <= n.end - editing.MIN_NOTE_DURATION
        ]
        if not targets:
            raise EditError("재생 위치를 지나는 선택 음표가 없습니다.")
        added: list[Note] = []
        for n in targets:
            added.extend(editing.split_note(n, at))
        self._push(targets, added, f"음표 {len(targets)}개 분할")
        return len(targets)

    def merge_notes(self, ids: Iterable[str]) -> str:
        notes = self._notes(ids)
        merged = editing.merge_notes(notes)
        self._push(notes, [merged], f"음표 {len(notes)}개 병합")
        return merged.id

    def revert_to_analysis(self) -> None:
        """작업본 음표를 원본 분석 결과로 되돌린다. 이 동작도 실행 취소할 수 있다."""
        if self.project.analysis is None:
            raise EditError("되돌릴 분석 결과가 없습니다.")
        original = Score.from_analysis(self.project.analysis).notes()
        self.undo_stack.beginMacro("분석 원본으로 되돌리기")
        self._push(self.score.notes(), original, "음표 되돌리기")
        if self.score.tempo != self.project.analysis.tempo:
            self.undo_stack.push(SetTempoCommand(self, self.score.tempo, self.project.analysis.tempo))
        self.undo_stack.endMacro()

    def _apply_analysis_state(self, analysis, removed_ids, added, tempo, lyrics) -> None:
        self.project.analysis = analysis
        self.score.replace_notes(removed_ids, added)
        self.score.tempo = tempo
        changed = lyrics is not self.project.lyrics
        self.project.lyrics = lyrics
        self._touch()
        self.analysisChanged.emit()
        self.tempoChanged.emit()
        if changed:
            self.lyricsChanged.emit()

    def apply_analysis(self, result: AnalysisResult, mode: str = "keep_edits", text: str = "분석 결과 적용"):
        """재분석 결과 반영. keep_edits면 사용자가 고친 음표·템포를 지킨다. 반영 계획을 돌려준다."""
        from amp.engine.merge import carry_note_ids, plan_reanalysis

        result = carry_note_ids(self.project, result)  # 그대로인 자동 음표는 id를 이어받아 가사 연결·검토 상태를 지킨다
        plan = plan_reanalysis(self.project, result, mode)
        self.undo_stack.push(ApplyAnalysisCommand(self, result, plan, text))
        return plan

    def set_tempo(self, tempo: TempoMap) -> None:
        """BPM·첫 박 위치 등 템포 전체를 바꾼다 (실행 취소 가능, 연속 변경은 하나로 합침)."""
        if tempo == self.score.tempo:
            return
        self.undo_stack.push(SetTempoCommand(self, self.score.tempo, tempo))

    def set_bpm(self, bpm: float) -> None:
        old = self.score.tempo
        if abs(old.initial_bpm - bpm) < 1e-6:
            return
        self.undo_stack.push(SetTempoCommand(self, old, old.with_initial_bpm(bpm)))
