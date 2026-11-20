"""분석 소스(분리 결과) 전환의 영향 미리 보기. Qt 비의존, 읽기 전용.

실제 반영(ProjectDocument.apply_analysis → ApplyAnalysisCommand)과 같은 함수(plan_reanalysis, linking.reconcile)를
돌려 숫자를 낸다. 그래서 미리 보기와 실제 결과가 어긋나지 않는다.

- 자동 음표: 사용자가 손대지 않은 음표만 새 결과로 바뀐다. 수정·추가한 음표는 그대로 (keep_edits).
- 음절↔음표 연결: 바뀌는 음표에 걸린 연결은 시간이 겹치는 새 음표로 옮긴다. 옮길 곳이 없으면 그 음절은 '음표 없는
  음절'이 되고 검토 목록에 나온다(조용히 지우지 않고 알린다). 사용자가 확정한 연결이 끊기는 수는 따로 센다.
- 사용자 멜리스마 판정: 지우지 않는다. 음표를 옮길 수 없으면 '재확인 필요'가 켜진다.
- 음절 시간 정렬: 자동으로 다시 정렬하지 않는다. 이전 보컬로 정렬한 자동 시간은 그대로 남으므로 다시 정렬을 권한다.
- 가사 텍스트·발음(사용자 수정 포함)은 음표와 무관해 바뀌지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SwitchPreview:
    auto_replaced: int = 0
    new_notes: int = 0
    added: int = 0
    kept_user: int = 0
    dropped_new: int = 0  # 보존 음표와 겹쳐 넣지 않는 새 음표
    links_moved: int = 0  # 다른 음표로 옮겨지는 연결이 있는 음절
    unlinked: list[str] = field(default_factory=list)  # 연결이 모두 끊기는 음절 id
    confirmed_unlinked: int = 0  # 그중 사용자가 확정했거나 직접 만든 연결이 있던 음절
    melisma_review: list[str] = field(default_factory=list)  # 새로 '재확인 필요'가 되는 사용자 멜리스마 판정
    melisma_kept: int = 0
    auto_aligned: int = 0  # 이전 보컬로 자동 정렬한 음절 시간 (그대로 남음)
    user_timings: int = 0  # 사용자가 고친 음절 시간 (그대로 남음)
    tempo_kept: bool = False
    pitch_curve_changes: bool = True

    def lines(self) -> list[str]:
        out = [f"자동 검출 음표: 손대지 않은 {self.auto_replaced}개를 새 결과 {self.added}개로 바꿉니다 "
               f"(새 결과 전체 {self.new_notes}개 중 사용자 음표와 겹치는 {self.dropped_new}개는 넣지 않음).",
               f"사용자가 수정·추가한 음표 {self.kept_user}개는 그대로 둡니다.",
               "피치 곡선: 새 분리 결과로 추출한 곡선으로 바뀝니다 (이전 곡선은 실행 기록과 캐시에 남습니다).",
               f"음절↔음표 연결: {self.links_moved}개 음절의 연결을 겹치는 새 음표로 옮깁니다.",
               ]
        if self.unlinked:
            out.append(f"  ⚠ {len(self.unlinked)}개 음절은 옮길 음표가 없어 '음표 없는 음절'이 됩니다"
                       + (f" (그중 사용자가 확정한 연결 {self.confirmed_unlinked}개)" if self.confirmed_unlinked else "")
                       + ". 검토 목록에 표시됩니다.")
        out.append(f"사용자 멜리스마 판정 {self.melisma_kept}개는 지우지 않습니다"
                   + (f" ({len(self.melisma_review)}개는 음표가 바뀌어 '재확인 필요'로 표시)" if self.melisma_review else "")
                   + ". 자동 멜리스마 판정은 새 음표로 다시 계산됩니다.")
        out.append(f"음절 시간 정렬: 자동 정렬 {self.auto_aligned}개·사용자 수정 {self.user_timings}개 모두 그대로 둡니다. "
                   "새 보컬로 다시 맞추려면 가사 시간 정렬(F7)을 다시 실행하세요.")
        out.append("가사 텍스트와 발음(사용자 수정 포함)은 바뀌지 않습니다.")
        if self.tempo_kept:
            out.append("사용자가 고친 템포는 그대로 둡니다.")
        return out


def preview(project, result) -> SwitchPreview:
    from amp.engine.merge import KEEP_EDITS, plan_reanalysis
    from amp.lyrics import linking

    plan = plan_reanalysis(project, result, KEEP_EDITS)
    pv = SwitchPreview(auto_replaced=len(plan.removed), new_notes=len(result.notes), added=len(plan.added),
                       kept_user=len(plan.kept), dropped_new=len(plan.dropped), tempo_kept=plan.tempo != result.tempo)
    old = project.lyrics
    if old is None:
        return pv
    new = linking.reconcile(old, plan.removed, plan.added)
    ola, nla = old.alignment, new.alignment
    for sid, links in ola.links.items():
        if not links:
            continue
        after = nla.links.get(sid, ())
        if not after:
            pv.unlinked.append(sid)
            if any(lk.confirmed or lk.source in linking.PROTECTED_SOURCES for lk in links):
                pv.confirmed_unlinked += 1
        elif tuple(lk.note_id for lk in after) != tuple(lk.note_id for lk in links):
            pv.links_moved += 1
    for sid, d in nla.melisma.items():
        pv.melisma_kept += 1
        before = ola.melisma.get(sid)
        if d.needs_review and not (before is not None and before.needs_review):
            pv.melisma_review.append(sid)
    for t in ola.timings.values():
        if not t.aligned:
            continue
        if t.user_edited:
            pv.user_timings += 1
        else:
            pv.auto_aligned += 1
    return pv
