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


class ReplaceNotesCommand(QUndoCommand):
    """음표 집합 교체. 이동·길이 변경·분할·병합·추가·삭제를 모두 이 명령 하나로 표현한다."""

    def __init__(self, doc: "ProjectDocument", removed: list[Note], added: list[Note], text: str) -> None:
        super().__init__(text)
        self._doc = doc
        self._removed = list(removed)
        self._added = list(added)

    def redo(self) -> None:
        self._doc._apply_notes([n.id for n in self._removed], self._added)
        self._doc.selectionRequested.emit([n.id for n in self._added])

    def undo(self) -> None:
        self._doc._apply_notes([n.id for n in self._added], self._removed)
        self._doc.selectionRequested.emit([n.id for n in self._removed])


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


class ProjectDocument(QObject):
    projectReset = Signal()  # 프로젝트 전체가 바뀜 (새로 만들기·열기·분석 적용)
    notesReplaced = Signal(list, list)  # 지운 id 목록, 추가/교체된 Note 목록
    tempoChanged = Signal()
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
        self.undo_stack.clear()
        self._extra_dirty = True
        self._touch()
        self.projectReset.emit()
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

    def set_bpm(self, bpm: float) -> None:
        old = self.score.tempo
        if abs(old.initial_bpm - bpm) < 1e-6:
            return
        self.undo_stack.push(SetTempoCommand(self, old, old.with_initial_bpm(bpm)))
