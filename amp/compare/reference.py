"""참조 악보(외부에서 가져온 MIDI·MusicXML)의 데이터. 자동 채보 음표·사용자 음표와 따로 둔다.

- 음표 시간은 '악보 시간'(파일의 템포로 계산한 초)으로 둔다. 음원 시간은 정렬(ScoreAlignment)로 바꾼다.
- 정렬은 처음엔 시작 오프셋·템포 배율 두 값이고, 자동 정렬을 하면 구간별 대응점(warp)이 생긴다.
  사용자가 확정하기 전에는 '미리보기'다.
- 가져온 원본 파일은 프로젝트 폴더 scores/에 복사해 두어 프로젝트를 옮겨도 다시 읽을 수 있다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace

SCHEMA = "sonatlas.reference_score"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReferenceNote:
    id: str
    start: float  # 악보 시간 (초)
    end: float
    pitch: int  # MIDI
    lyric: str | None = None  # 파일에 있는 가사만 (없으면 None, 지어내지 않는다)
    lyric_extend: bool = False  # 앞 음절을 늘여 부르는 음(MusicXML <extend/>)
    beat: float = 0.0  # 악보 시작부터 4분음표 단위 위치 (내보내기·표시용)
    beats: float = 0.0  # 길이 (4분음표 단위)
    voice: str = ""
    tied: bool = False  # 붙임줄로 이어 붙인 음


@dataclass(frozen=True)
class PartInfo:
    id: str
    name: str
    notes: int
    low: int = 0
    high: int = 0
    has_lyrics: bool = False
    polyphonic: bool = False  # 같은 성부 안에서 음이 겹친다 (화음)
    drums: bool = False


@dataclass(frozen=True)
class ScoreAlignment:
    offset: float = 0.0  # 악보 0초가 음원의 몇 초인가
    scale: float = 1.0  # 음원 시간 / 악보 시간 (템포 배율의 역수)
    warp: tuple[tuple[float, float], ...] = ()  # 자동 정렬 대응점 ((악보 초, 음원 초), …), 단조 증가
    method: str = "manual"  # manual / auto
    confirmed: bool = False
    quality: dict = field(default_factory=dict, compare=False)

    def audio_time(self, t: float) -> float:
        if len(self.warp) >= 2:
            xs = [p[0] for p in self.warp]
            ys = [p[1] for p in self.warp]
            i = bisect.bisect_right(xs, t)
            if i <= 0:
                i = 1
            elif i >= len(xs):
                i = len(xs) - 1
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            slope = (y1 - y0) / (x1 - x0) if x1 > x0 else self.scale
            return y0 + (t - x0) * slope
        return self.offset + t * self.scale

    def nudged(self, d_offset: float = 0.0, scale: float | None = None) -> "ScoreAlignment":
        """수동 조정: 오프셋을 더하고(대응점도 함께 민다) 템포 배율을 바꾼다. 확정은 풀린다."""
        warp = tuple((s, a + d_offset) for s, a in self.warp)
        return replace(self, offset=self.offset + d_offset, scale=self.scale if scale is None else scale,
                       warp=warp if scale is None else (), confirmed=False,
                       method=self.method if scale is None else "manual")


@dataclass(frozen=True)
class ReferenceScore:
    source_name: str
    source_sha256: str
    kind: str  # midi / musicxml
    stored_path: str = ""  # 프로젝트 폴더 기준 상대 경로 (scores/…)
    parts: tuple[PartInfo, ...] = ()
    part_id: str = ""
    notes: tuple[ReferenceNote, ...] = ()
    tempo: tuple[tuple[float, float], ...] = ()  # (악보 초, BPM)
    time_signatures: tuple[tuple[float, int, int], ...] = ()  # (악보 초, 분자, 분모)
    warnings: tuple[str, ...] = ()
    alignment: ScoreAlignment = field(default_factory=ScoreAlignment)
    imported: str = ""

    def audio_span(self, n: ReferenceNote) -> tuple[float, float]:
        return self.alignment.audio_time(n.start), self.alignment.audio_time(n.end)

    def aligned_notes(self) -> list[tuple[ReferenceNote, float, float]]:
        return [(n, *self.audio_span(n)) for n in self.notes]

    def with_alignment(self, alignment: ScoreAlignment) -> "ReferenceScore":
        return replace(self, alignment=alignment)

    @property
    def has_lyrics(self) -> bool:
        return any(n.lyric for n in self.notes)


def to_dict(ref: ReferenceScore) -> dict:
    a = ref.alignment
    return {
        "source": {"name": ref.source_name, "sha256": ref.source_sha256, "kind": ref.kind, "stored_path": ref.stored_path,
                   "imported": ref.imported},
        "parts": [vars(p) | {} for p in ref.parts],
        "part_id": ref.part_id,
        # 반올림하지 않는다: 저장 → 다시 열기가 메모리 값과 같아야 한다 (6자리 반올림으로 실제 곡 악보 73음이 0.5µs씩 달라졌다)
        "notes": [{"id": n.id, "start": float(n.start), "end": float(n.end), "pitch": n.pitch, "lyric": n.lyric,
                   "lyric_extend": n.lyric_extend, "beat": float(n.beat), "beats": float(n.beats),
                   "voice": n.voice, "tied": n.tied} for n in ref.notes],
        "tempo": [list(t) for t in ref.tempo],
        "time_signatures": [list(t) for t in ref.time_signatures],
        "warnings": list(ref.warnings),
        "alignment": {"offset": float(a.offset), "scale": float(a.scale), "warp": [[float(x), float(y)] for x, y in a.warp],
                      "method": a.method,
                      "confirmed": a.confirmed, "quality": a.quality},
    }


def from_dict(d: dict) -> ReferenceScore:
    src = d.get("source") or {}
    al = d.get("alignment") or {}
    return ReferenceScore(
        source_name=str(src.get("name", "")), source_sha256=str(src.get("sha256", "")), kind=str(src.get("kind", "")),
        stored_path=str(src.get("stored_path", "")), imported=str(src.get("imported", "")),
        parts=tuple(PartInfo(**{k: p[k] for k in PartInfo.__dataclass_fields__ if k in p}) for p in d.get("parts", [])),
        part_id=str(d.get("part_id", "")),
        notes=tuple(ReferenceNote(id=str(n["id"]), start=float(n["start"]), end=float(n["end"]), pitch=int(n["pitch"]),
                                  lyric=n.get("lyric"), lyric_extend=bool(n.get("lyric_extend", False)),
                                  beat=float(n.get("beat", 0.0)), beats=float(n.get("beats", 0.0)),
                                  voice=str(n.get("voice", "")), tied=bool(n.get("tied", False)))
                    for n in d.get("notes", [])),
        tempo=tuple((float(a), float(b)) for a, b in d.get("tempo", [])),
        time_signatures=tuple((float(a), int(b), int(c)) for a, b, c in d.get("time_signatures", [])),
        warnings=tuple(str(w) for w in d.get("warnings", [])),
        alignment=ScoreAlignment(offset=float(al.get("offset", 0.0)), scale=float(al.get("scale", 1.0)),
                                 warp=tuple((float(a), float(b)) for a, b in al.get("warp", [])),
                                 method=str(al.get("method", "manual")), confirmed=bool(al.get("confirmed", False)),
                                 quality=dict(al.get("quality") or {})),
    )
