"""재분석 결과를 작업본에 반영하는 계획 (Qt 비의존).

사용자가 고친 음표를 재분석이 함부로 지우지 않도록, 먼저 무엇을 지우고 무엇을 넣을지 계산한다.
실제 반영과 실행 취소는 UI 쪽 명령(ApplyAnalysisCommand)이 맡는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from amp.core.model import AnalysisResult, Note, Project, Score
from amp.core.tempo import TempoMap

KEEP_EDITS = "keep_edits"
REPLACE_ALL = "replace_all"
OVERLAP_LIMIT = 0.2  # 새 음표 길이의 20%보다 많이 보존 음표와 겹치면 넣지 않는다


@dataclass(frozen=True)
class ScoreSummary:
    original: int
    modified: int
    user: int
    tempo_edited: bool

    @property
    def has_edits(self) -> bool:
        return self.modified > 0 or self.user > 0 or self.tempo_edited


@dataclass(frozen=True)
class ReanalysisPlan:
    removed: tuple[Note, ...]  # 작업본에서 지울 음표
    added: tuple[Note, ...]  # 넣을 새 음표
    kept: tuple[Note, ...]  # 보존한 사용자 음표
    dropped: tuple[Note, ...]  # 보존 음표와 겹쳐 넣지 않은 새 음표
    tempo: TempoMap


def summarize(project: Project) -> ScoreSummary:
    counts = {"original": 0, "modified": 0, "user": 0}
    for n in project.score.notes():
        counts[project.note_status(n)] += 1
    tempo_edited = project.analysis is not None and project.score.tempo != project.analysis.tempo
    return ScoreSummary(counts["original"], counts["modified"], counts["user"], tempo_edited)


def _overlap(a: Note, b: Note) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def plan_reanalysis(project: Project, new: AnalysisResult, mode: str = KEEP_EDITS) -> ReanalysisPlan:
    current = project.score.notes()
    fresh = Score.from_analysis(new).notes()
    if mode == REPLACE_ALL:
        return ReanalysisPlan(tuple(current), tuple(fresh), (), (), new.tempo)
    if mode != KEEP_EDITS:
        raise ValueError(f"알 수 없는 반영 방식: {mode}")

    protected = [n for n in current if project.note_status(n) != "original"]
    removable = [n for n in current if project.note_status(n) == "original"]
    added, dropped = [], []
    for n in fresh:
        if any(_overlap(n, p) > OVERLAP_LIMIT * n.duration for p in protected):
            dropped.append(n)
        else:
            added.append(n)
    tempo_edited = project.analysis is not None and project.score.tempo != project.analysis.tempo
    tempo = project.score.tempo if (tempo_edited or project.analysis is None and protected) else new.tempo
    return ReanalysisPlan(tuple(removable), tuple(added), tuple(protected), tuple(dropped), tempo)
