"""주선율·화음 후보와 성부 사이 음정 (v0.7.0 Feature C, 명세 6장). Qt 비의존.

성부 결과(amp/engine/voices.VoiceSet)에서
- 동시 발성 구간(두 성부가 함께 유성 100ms 이상)과 그 사이 음정(반음, 프레임 중앙값)
- 음정 종류: 유니즌·단3도·장3도·4도·5도·단6도·장6도·옥타브·기타 (옥타브를 넘으면 옥타브를 뺀 나머지로 분류, 옥타브 수 기록)
- 주선율 후보·화음 후보·모름: 근거 점수를 더해 정하되 '가장 큰 소리 = 주선율'로 가정하지 않는다.
  근거: 유성 지속(곡에서 소리 내는 시간), 이어짐(끊김이 적음), 가사 음절 시작과 맞는 정도(있으면), 선율 반복(있으면 생략),
  다른 성부와 같이 날 때 위에 있는 비율(참고만), 배음 합 크기(참고만, 가중치 낮음)
결과는 추정값이다. 사용자가 역할을 바꾸면(성부 편집) 그 값이 우선한다. 음정 관계만으로 음악적 역할을 확정하지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

INTERVALS = {0: "유니즌", 3: "단3도", 4: "장3도", 5: "4도", 7: "5도", 8: "단6도", 9: "장6도", 12: "옥타브"}
MIN_OVERLAP_S = 0.1


def interval_name(semitones: float) -> tuple[str, int]:
    """(이름, 더한 옥타브 수). 반올림한 반음 기준."""
    s = int(round(abs(semitones)))
    octs = 0
    if s > 12:
        octs, s = (s - 1) // 12, s - 12 * ((s - 1) // 12)
    name = INTERVALS.get(s, "기타")
    return name, octs


@dataclass
class Overlap:
    upper: str  # voice_track_id (구간 중앙값이 높은 쪽)
    lower: str
    start: float
    end: float
    semitones: float
    interval: str
    octaves: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"], d["semitones"] = round(self.start, 3), round(self.end, 3), round(self.semitones, 2)
        return d


@dataclass
class RoleEstimate:
    voice_track_id: str
    role: str  # main_candidate / harmony_candidate / unknown
    score: float
    evidence: list = field(default_factory=list)
    user_set: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 3)
        return d


def overlaps(vs, min_s: float = MIN_OVERLAP_S) -> list[Overlap]:
    fr = vs.frame_rate
    out = []
    vv = [v for v in vs.voices if v.voiced().any()]
    for i in range(len(vv)):
        for j in range(i + 1, len(vv)):
            a, b = vv[i], vv[j]
            both = a.voiced() & b.voiced()
            d = np.diff(np.r_[0, both.astype(np.int8), 0])
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if (e - s) / fr < min_s:
                    continue
                st = 12 * np.log2(a.f0_hz[s:e] / b.f0_hz[s:e])
                med = float(np.median(st))
                up, lo = (a, b) if med >= 0 else (b, a)
                name, octs = interval_name(med)
                out.append(Overlap(up.voice_track_id, lo.voice_track_id, (vs.start_frame + s) / fr,
                                   (vs.start_frame + e) / fr, abs(med), name, octs))
    out.sort(key=lambda o: o.start)
    return out


def interval_summary(ov: list[Overlap]) -> dict:
    tot: dict[str, float] = {}
    for o in ov:
        tot[o.interval] = tot.get(o.interval, 0.0) + (o.end - o.start)
    return {k: round(v, 2) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}


def estimate_roles(vs, syllable_starts: list[float] | None = None) -> list[RoleEstimate]:
    voices = [v for v in vs.voices if v.voiced().any()]
    if not voices:
        return []
    fr = vs.frame_rate
    n = vs.n
    raw = {}
    for v in voices:
        on = v.voiced()
        dur = on.sum() / fr
        runs = int(np.count_nonzero(np.diff(np.r_[0, on.astype(np.int8), 0]) == 1))
        cont = dur / max(1, runs)
        sync = None
        if syllable_starts:
            onsets = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8)]) == 1) / fr + vs.start_frame / fr
            if len(onsets):
                near = [np.min(np.abs(onsets - t)) <= 0.12 for t in syllable_starts if vs.start_frame / fr <= t <= (vs.start_frame + n) / fr]
                sync = float(np.mean(near)) if near else None
        others = [o for o in voices if o is not v]
        top = None
        if others:
            both = on & np.any([o.voiced() for o in others], axis=0)
            if both.any():
                hi = np.all([~o.voiced()[both] | (v.f0_hz[both] >= np.nan_to_num(o.f0_hz[both])) for o in others], axis=0)
                top = float(hi.mean())
        sal = float(np.nanmean(v.confidence[on])) if on.any() else 0.0
        raw[v.voice_track_id] = (dur, cont, sync, top, sal)
    maxdur = max(r[0] for r in raw.values()) or 1.0
    maxcont = max(r[1] for r in raw.values()) or 1.0
    out = []
    for v in voices:
        dur, cont, sync, top, sal = raw[v.voice_track_id]
        score = 0.4 * dur / maxdur + 0.25 * cont / maxcont + 0.05 * sal
        ev = [f"유성 {dur:.1f}초 (가장 긴 성부 대비 {dur / maxdur:.0%})", f"한 번에 평균 {cont:.2f}초 이어짐"]
        if sync is not None:
            score += 0.25 * sync
            ev.append(f"가사 음절 시작과 맞음 {sync:.0%}")
        if top is not None:
            score += 0.05 * top
            ev.append(f"같이 날 때 위에 있는 비율 {top:.0%} (참고)")
        out.append(RoleEstimate(v.voice_track_id, "unknown", float(score), ev))
    out.sort(key=lambda r: -r.score)
    if len(out) == 1:
        out[0].role = "main_candidate"
    else:
        lead, second = out[0], out[1]
        if lead.score - second.score >= 0.1:
            lead.role = "main_candidate"
            for r in out[1:]:
                r.role = "harmony_candidate"
        else:
            for r in out:
                r.evidence.append("근거 점수 차이가 작아 정하지 않음")
    user = {v.voice_track_id: v.role for v in vs.voices if v.edited and v.role in ("main_candidate", "harmony_candidate", "user")}
    for r in out:
        if r.voice_track_id in user:
            r.role, r.user_set = user[r.voice_track_id], True
    return out
