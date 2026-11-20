"""내보내기 공통: 옵션, 내보낼 음표 고르기, 음표별 가사(멜리스마 연장 포함), 초 ↔ 틱 변환.

원본 분석 결과와 실제 내보낸 데이터가 어떻게 다른지(사용자 수정 포함 여부, 구간, 형식상 빠진 정보)를 ExportReport에 적는다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

TICKS_PER_BEAT = 480
EXTEND = "+~"  # OpenUtau: 앞 음절을 이어 부르는 음 (OpenUtau MusicXML 가져오기와 같은 표기)


@dataclass(frozen=True)
class ExportOptions:
    source: str = "user"  # user: 사용자 수정 포함(지금 악보) / auto: 자동 분석 원본 / reference: 참조 악보
    range: tuple[float, float] | None = None  # None = 전체, (시작, 끝) = 이 구간에서 시작하는 음표만
    lyrics: bool = True
    pitch: str = "none"  # none / simple / detailed (USTX·MIDI 피치)
    lrc_words: bool = False  # LRC 음절 단위 시각(확장 표기)
    csv_bom: bool = True  # Windows Excel이 UTF-8 한글을 바로 읽게


@dataclass
class ExportReport:
    format: str
    path: str = ""
    notes: int = 0
    lyrics: int = 0
    extended: int = 0
    gaps_closed: int = 0
    omitted: list[str] = field(default_factory=list)  # 형식상 담지 못한 정보
    warnings: list[str] = field(default_factory=list)
    source: str = "user"
    range: tuple[float, float] | None = None

    def summary(self) -> str:
        src = {"user": "사용자 수정 포함", "auto": "자동 분석 원본", "reference": "참조 악보"}.get(self.source, self.source)
        rng = "전체" if self.range is None else f"{self.range[0]:.2f}~{self.range[1]:.2f}초"
        lines = [f"{self.format}: 음표 {self.notes}개 ({src}, {rng})"]
        if self.lyrics:
            lines.append(f"가사 {self.lyrics}개" + (f", 이어 부르기 음표 {self.extended}개" if self.extended else ""))
        if self.gaps_closed:
            lines.append(f"멜리스마 음표 사이 빈틈 {self.gaps_closed}곳을 앞 음표를 늘여 메웠습니다(이어 부르기 표기 조건)")
        lines += [f"담지 못한 정보: {x}" for x in self.omitted]
        lines += [f"주의: {x}" for x in self.warnings]
        return "\n".join(lines)


@dataclass(frozen=True)
class ExportNote:
    id: str
    start: float
    end: float
    pitch: int
    lyric: str | None = None  # 이 음표에서 시작하는 음절(들)
    extend: bool = False  # 앞 음표의 음절을 이어 부르는 음 (같은 음절의 두 번째 이후 음표)
    syllable_ids: tuple[str, ...] = ()
    status: str = ""  # original / modified / user / reference
    confidence: float | None = None


def pick_notes(project, options: ExportOptions) -> tuple[list[ExportNote], list[str]]:
    """옵션에 맞는 음표와 가사. (음표, 경고)."""
    warnings: list[str] = []
    if options.source == "reference":
        ref = project.reference
        if ref is None:
            raise ValueError("참조 악보가 없습니다.")
        if not ref.alignment.confirmed:
            warnings.append("참조 악보 정렬을 확정하지 않았습니다(미리보기 위치로 내보냄).")
        out = []
        last_lyric_note = None
        for n, a, b in ref.aligned_notes():
            lyric = n.lyric if options.lyrics else None
            ext = bool(options.lyrics and not n.lyric and n.lyric_extend and last_lyric_note is not None)
            out.append(ExportNote(n.id, a, b, n.pitch, lyric, ext, (), "reference"))
            if lyric:
                last_lyric_note = n
        return _in_range(out, options.range), warnings
    if options.source == "auto":
        if project.analysis is None:
            raise ValueError("자동 분석 결과가 없습니다.")
        notes = list(project.analysis.notes)
        status = {n.id: "original" for n in notes}
    else:
        notes = list(project.score.notes())
        status = {n.id: project.note_status(n) for n in notes}
    notes.sort(key=lambda n: (n.start, n.pitch))
    lyric_of: dict[str, list[str]] = {}
    sids_of: dict[str, list[str]] = {}
    extend: set[str] = set()
    ly = project.lyrics
    if options.lyrics and not ly.is_empty:
        by_id = {n.id: n for n in notes}
        al = ly.alignment
        for s in ly.text.syllables:
            linked = sorted((by_id[lk.note_id] for lk in al.links.get(s.id, ()) if lk.note_id in by_id),
                            key=lambda n: n.start)
            if not linked:
                continue
            first = linked[0]
            lyric_of.setdefault(first.id, []).append(s.text)
            sids_of.setdefault(first.id, []).append(s.id)
            for n in linked[1:]:
                extend.add(n.id)
        many = [k for k, v in lyric_of.items() if len(v) > 1]
        if many:
            warnings.append(f"음표 {len(many)}개에 음절이 둘 이상 연결돼 한 음표 가사로 붙였습니다(예: '{''.join(lyric_of[many[0]])}').")
    out = [ExportNote(n.id, n.start, n.end, n.pitch, "".join(lyric_of[n.id]) if n.id in lyric_of else None,
                      n.id in extend and n.id not in lyric_of, tuple(sids_of.get(n.id, ())), status.get(n.id, ""),
                      n.confidence) for n in notes]
    return _in_range(out, options.range), warnings


def _in_range(notes: list[ExportNote], rng) -> list[ExportNote]:
    if rng is None:
        return notes
    a, b = rng
    return [n for n in notes if a <= n.start < b]


class TickMap:
    """초 ↔ 틱 (4분음표 = 480틱). 음원 0초 = 0틱. 템포 변화를 구간별로 적분해 틱을 정확히 계산한다.

    SONATLAS 템포는 시각 기준 (시각, BPM) 점들이다. 첫 점 앞은 첫 BPM을 쓴다. 음표 시작·끝을 각각 절대 시각에서 바로
    틱으로 바꾸므로 반올림 오차가 쌓이지 않는다(최대 0.5틱).
    """

    def __init__(self, tempo_points: list[tuple[float, float]], beat_unit: int = 4) -> None:
        pts = sorted((max(0.0, float(t)), float(b)) for t, b in tempo_points) or [(0.0, 120.0)]
        if pts[0][0] > 0:
            pts.insert(0, (0.0, pts[0][1]))
        merged: list[tuple[float, float]] = []
        for t, b in pts:
            if merged and abs(merged[-1][0] - t) < 1e-9:
                merged[-1] = (t, b)
            elif not merged or abs(merged[-1][1] - b) > 1e-9:
                merged.append((t, b))
        self.points = merged
        self.ticks = [0.0]
        for (t0, b0), (t1, _b1) in zip(merged[:-1], merged[1:]):
            self.ticks.append(self.ticks[-1] + (t1 - t0) * b0 / 60.0 * TICKS_PER_BEAT)

    def tick_float(self, t: float) -> float:
        i = max(0, bisect.bisect_right([p[0] for p in self.points], t) - 1)
        t0, bpm = self.points[i]
        return self.ticks[i] + (t - t0) * bpm / 60.0 * TICKS_PER_BEAT

    def tick(self, t: float) -> int:
        return int(round(self.tick_float(max(0.0, t))))

    def seconds(self, tick: float) -> float:
        i = max(0, bisect.bisect_right(self.ticks, tick) - 1)
        t0, bpm = self.points[i]
        return t0 + (tick - self.ticks[i]) / TICKS_PER_BEAT * 60.0 / bpm

    def tempo_ticks(self) -> list[tuple[int, float]]:
        return [(int(round(tk)), b) for tk, (_t, b) in zip(self.ticks, self.points)]


def tempo_points(project) -> tuple[list[tuple[float, float]], int, int]:
    """프로젝트 템포 (사용자 템포가 있으면 그것). ((시각, BPM) 목록, 박자 분자, 분모)."""
    tm = project.score.tempo
    pts = [(float(p.time), float(p.bpm)) for p in tm.points]
    return pts, int(tm.beats_per_bar), int(getattr(tm, "beat_unit", 4) or 4)
