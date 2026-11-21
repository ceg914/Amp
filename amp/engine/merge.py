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


CARRY_TOL_S = 1e-6


def carry_note_ids(project: Project, new: AnalysisResult) -> AnalysisResult:
    """새 분석에서 이전 자동 음표와 시작·끝·음높이가 같은 음표는 이전 id를 이어받는다 (v0.6).

    Hybrid 구간 하나만 바꿔 다시 분석하면 나머지 구간의 음표는 그대로 나온다. id가 같으면 가사 연결·검토 상태·
    제안이 그 음표를 계속 가리킨다. 작업본에서 사용자가 고친 음표(수정·추가)가 쓰는 id는 이어받지 않는다
    (같은 id의 음표가 두 개가 되지 않게).
    """
    old = project.analysis
    if old is None or not old.notes or not new.notes:
        return new
    reserved = {n.id for n in project.score.notes() if project.note_status(n) != "original"}
    by_key: dict[tuple, list[Note]] = {}
    for n in old.notes:
        if n.id not in reserved:
            by_key.setdefault((n.pitch, round(n.start, 4), round(n.end, 4)), []).append(n)
    used: set[str] = set()
    fresh_ids = {n.id for n in new.notes}
    out = []
    changed = False
    for n in new.notes:
        cands = by_key.get((n.pitch, round(n.start, 4), round(n.end, 4)), [])
        hit = next((o for o in cands if o.id not in used and abs(o.start - n.start) < CARRY_TOL_S
                    and abs(o.end - n.end) < CARRY_TOL_S and (o.id == n.id or o.id not in fresh_ids)), None)
        if hit is not None and hit.id != n.id:
            used.add(hit.id)
            out.append(n.with_(id=hit.id))
            changed = True
        else:
            if hit is not None:
                used.add(hit.id)
            out.append(n)
    if not changed:
        return new
    import dataclasses

    return dataclasses.replace(new, notes=tuple(out))


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
