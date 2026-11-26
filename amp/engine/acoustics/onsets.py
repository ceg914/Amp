"""발성 경계 정밀 분석 (v0.7.0 Feature G, 명세 10장). Qt 비의존.

음표와 따로 관리하는 경계 후보. 음표마다 다섯 시각을 잰다 (잴 수 없으면 None = 미측정):
- acoustic_onset: 유성 시작 앞 150ms 안에서 음량이 바닥 + 6dB를 처음 넘은 곳 (자음·숨 시작 포함)
- voiced_onset: 음표 둘레(시작 ±80ms ~ 끝)에서 처음 유성인 프레임
- stable_pitch_onset: 유성 시작 뒤, 2Hz 저역 곡선이 음 가운데 높이 ±30센트 안에 50ms 이상 머물기 시작한 곳
- voiced_end: 음표 둘레에서 마지막 유성 프레임
- acoustic_tail_end: 유성 끝 뒤 400ms 안에서 음량이 바닥 + 6dB 아래로 내려간 곳 (잔향 꼬리 포함)

한 문턱값만 쓰지 않도록 바닥은 곡 전체의 10번째 백분위로 정하고, 안정 판정은 음마다 그 음의 가운데 높이로 한다.
결과는 음표·가사에 자동으로 적용하지 않는다. 사용자가 고를 때 교정 제안(시작·끝)으로만 만든다.
잔향 자르기(기존 trim_release)는 그대로 기본 꺼짐이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from amp.engine.acoustics.phonation import smooth_midi

ANALYSIS_VERSION = "1"
FIELDS = ("acoustic_onset", "voiced_onset", "stable_pitch_onset", "voiced_end", "acoustic_tail_end")
FIELD_LABELS = {"acoustic_onset": "소리 시작", "voiced_onset": "유성 시작", "stable_pitch_onset": "목표 음높이 도달",
                "voiced_end": "유성 끝", "acoustic_tail_end": "소리 꼬리 끝"}


@dataclass
class NoteBoundary:
    note_id: str
    note_start: float  # 분석할 때의 음표 (바뀌면 오래된 결과)
    note_end: float
    note_pitch: int
    acoustic_onset: float | None = None
    voiced_onset: float | None = None
    stable_pitch_onset: float | None = None
    voiced_end: float | None = None
    acoustic_tail_end: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d

    @property
    def measured(self) -> int:
        return sum(1 for f in FIELDS if getattr(self, f) is not None)


def analyze(notes, f0_hz: np.ndarray, rms_db: np.ndarray, frame_rate: float = 100.0) -> list[NoteBoundary]:
    fr = float(frame_rate)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    db = np.asarray(rms_db, dtype=np.float64)
    n = min(len(f0), len(db))
    f0, db = f0[:n], db[:n]
    voiced = np.isfinite(f0)
    floor = max(float(np.nanpercentile(db, 10)), -80.0) + 6.0
    sm = smooth_midi(f0, fr)
    out = []
    ordered = sorted(notes, key=lambda x: x.start)
    for k, note in enumerate(ordered):
        nb = NoteBoundary(note.id, float(note.start), float(note.end), int(note.pitch))
        lo = int(max(0, (note.start - 0.08) * fr))
        if k:
            lo = max(lo, int(ordered[k - 1].end * fr))  # 앞 음 안으로 들어가지 않는다
        hi = int(min(n, note.end * fr + 1))
        if hi - lo < 2:
            out.append(nb)
            continue
        v = np.flatnonzero(voiced[lo:hi])
        if len(v):
            vo, ve = lo + int(v[0]), lo + int(v[-1])
            nb.voiced_onset, nb.voiced_end = vo / fr, (ve + 1) / fr
            back = max(0, vo - int(0.15 * fr))
            if k:
                back = max(back, int(ordered[k - 1].end * fr))
            j = vo
            while j - 1 >= back and db[j - 1] > floor:
                j -= 1
            nb.acoustic_onset = j / fr
            mid = sm[vo + (ve - vo) // 3: ve - (ve - vo) // 3 + 1]
            mid = mid[np.isfinite(mid)]
            if mid.size:
                target = float(np.median(mid))
                near = np.abs(sm[vo:ve + 1] - target) * 100 <= 30.0
                need = max(2, int(0.05 * fr))
                run = 0
                for i, ok in enumerate(near):
                    run = run + 1 if ok else 0
                    if run >= need:
                        nb.stable_pitch_onset = (vo + i - need + 1) / fr
                        break
            nxt = int(ordered[k + 1].start * fr) if k + 1 < len(ordered) else n
            limit = min(n, ve + 1 + int(0.4 * fr), max(ve + 1, nxt))
            j = ve + 1
            while j < limit and db[j] > floor:
                j += 1
            nb.acoustic_tail_end = j / fr if j < limit or j == n else None
        out.append(nb)
    return out


def stale(b: NoteBoundary, note) -> bool:
    return note is None or abs(note.start - b.note_start) > 1e-3 or abs(note.end - b.note_end) > 1e-3 \
        or int(note.pitch) != b.note_pitch


def suggestions(project, boundaries: list[NoteBoundary], start_field: str = "voiced_onset",
                end_field: str = "voiced_end", min_shift_s: float = 0.05):
    """경계 결과 → 교정 제안('boundary' 종류). 사용자가 직접 고친 음표·경계 결과가 오래된 음표는 건너뛴다.
    적용은 교정 제안 탭에서 사용자가 고를 때만 한다."""
    from amp.compare import suggestions as S

    notes = {n.id: n for n in project.score.notes()}
    out = []
    for b in boundaries:
        n = notes.get(b.note_id)
        if n is None or stale(b, n) or project.note_status(n) != "original":
            continue
        changes, ev = {}, []
        s_new, e_new = getattr(b, start_field), getattr(b, end_field)
        if s_new is not None and abs(s_new - n.start) >= min_shift_s and s_new < n.end - 0.06:
            changes["start"] = round(s_new, 3)
            ev.append(f"발성 경계 분석: {FIELD_LABELS[start_field]} {s_new:.3f}s (음표 시작과 {(s_new - n.start) * 1000:+.0f}ms)")
        if e_new is not None and abs(e_new - n.end) >= min_shift_s and e_new > changes.get("start", n.start) + 0.06:
            changes["end"] = round(e_new, 3)
            ev.append(f"발성 경계 분석: {FIELD_LABELS[end_field]} {e_new:.3f}s (음표 끝과 {(e_new - n.end) * 1000:+.0f}ms)")
        if changes:
            shift = max(abs(changes.get("start", n.start) - n.start), abs(changes.get("end", n.end) - n.end))
            out.append(S._make("boundary", [n], n.start, n.end, changes, ev, 0.3 + 0.5 * min(1.0, shift / 0.2)))  # noqa: SLF001
    return out
