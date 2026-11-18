"""가사 데이터 모델 (불변 객체).

텍스트(``LyricsText``)와 시간·연결(``Alignment``)을 나눠 둔다.
- 텍스트를 고쳐도 바뀌지 않은 음절은 id가 그대로라 시간·연결이 유지된다.
- 음표를 나누거나 합쳐도 음절 텍스트는 바뀌지 않는다. 연결만 다시 계산한다.

시각 출처(``start_source`` / ``end_source``)
    model      정렬 모델이 직접 검출한 경계
    asr_word   자동 인식 모델의 단어 시각 (단어 안 음절 경계는 estimated)
    estimated  단어·줄 경계 안에서 글자 수로 나눈 추정값. 검출값이 아니다
    inferred   음절 끝: 다음 음절 시작이나 보컬이 끊긴 지점으로 정한 값 (CTC 모델은 끝을 직접 주지 않는다)
    notes      연결한 음표의 시작·끝에서 가져온 값
    user       사용자가 직접 정한 값
    legacy     STEP 1 형식에서 옮겨 온 값
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Iterable, Mapping

SOURCES = ("model", "asr_word", "estimated", "inferred", "notes", "user", "legacy")
SOURCE_LABELS = {
    "model": "모델 검출", "asr_word": "인식 모델 단어 시각", "estimated": "추정(나눔)", "inferred": "추정(다음 음절·보컬 끝)",
    "notes": "음표에서 가져옴", "user": "사용자 지정", "legacy": "이전 형식", "": "없음",
}
LINK_SOURCES = ("auto", "user", "remapped", "sequential", "legacy")
ORIGINS = ("manual", "paste", "file", "asr", "repeat", "legacy")


@dataclass(frozen=True)
class LyricSyllable:
    id: str
    text: str
    kind: str = ""
    reading: str = ""  # 정렬용 로마자. 모르면 ""
    note: str = ""  # 텍스트 처리 단계의 검토 안내 (한자 읽기 모름 등)


@dataclass(frozen=True)
class LyricWord:
    id: str
    text: str
    syllables: tuple[LyricSyllable, ...] = ()


@dataclass(frozen=True)
class LyricLine:
    id: str
    text: str
    words: tuple[LyricWord, ...] = ()
    origin: str = "manual"
    review: str = ""  # 검토가 필요한 이유. 사용자가 고치거나 확인하면 ""
    asr_segment: str | None = None  # 자동 인식 구간에서 온 줄이면 그 id

    @property
    def syllables(self) -> tuple[LyricSyllable, ...]:
        return tuple(s for w in self.words for s in w.syllables)


@dataclass(frozen=True)
class LyricsText:
    language: str = ""  # ko / en / ja / "" (정하지 않음)
    lines: tuple[LyricLine, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @cached_property
    def syllables(self) -> tuple[LyricSyllable, ...]:
        return tuple(s for ln in self.lines for s in ln.syllables)

    @cached_property
    def index(self) -> dict[str, tuple[int, int, int]]:
        """음절 id → (줄, 단어, 음절) 위치."""
        out = {}
        for li, ln in enumerate(self.lines):
            for wi, w in enumerate(ln.words):
                for si, s in enumerate(w.syllables):
                    out[s.id] = (li, wi, si)
        return out

    @cached_property
    def order(self) -> dict[str, int]:
        """음절 id → 전체 순번."""
        return {s.id: i for i, s in enumerate(self.syllables)}

    def syllable(self, sid: str) -> LyricSyllable:
        li, wi, si = self.index[sid]
        return self.lines[li].words[wi].syllables[si]

    def line_of(self, sid: str) -> LyricLine:
        return self.lines[self.index[sid][0]]

    def word_of(self, sid: str) -> LyricWord:
        li, wi, _ = self.index[sid]
        return self.lines[li].words[wi]

    @property
    def is_empty(self) -> bool:
        return not self.syllables


@dataclass(frozen=True)
class SyllableTiming:
    start: float | None = None
    end: float | None = None
    confidence: float | None = None  # 0~1. 정렬 모델의 확신 정도 (사용자 지정이면 None)
    start_source: str = ""
    end_source: str = ""
    user_edited: bool = False
    stale: bool = False  # 텍스트가 바뀌어 다시 정렬이 필요한 시간 (값은 이전 것을 남겨 둠)

    @property
    def aligned(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def detected(self) -> bool:
        """시작 경계를 모델이 직접 검출했거나 사용자가 정했는가 (추정값이 아닌가)."""
        return self.start_source in ("model", "user", "notes", "legacy")


@dataclass(frozen=True)
class NoteLink:
    note_id: str
    confidence: float | None = None
    source: str = "auto"
    confirmed: bool = False  # 사용자가 확정한 연결 (자동 재계산이 바꾸지 않는다)


@dataclass(frozen=True)
class AsrWord:
    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True)
class AsrSegment:
    id: str
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    review: tuple[str, ...] = ()  # 검토 사유 (낮은 확신, 보컬 근거 부족, 반복, 흔한 환각 문구 …)
    words: tuple[AsrWord, ...] = ()


@dataclass(frozen=True)
class Recognition:
    """자동 인식 원본. 사용자가 편집하지 않는다(편집은 LyricsText에)."""

    model: str
    model_version: str = ""
    language: str = ""
    language_probability: float | None = None
    device: str = ""
    created: str = ""
    settings: dict = field(default_factory=dict)
    input_id: str = ""  # 입력 보컬 식별자 (분리 결과 id 등)
    segments: tuple[AsrSegment, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments)


@dataclass(frozen=True)
class Alignment:
    timings: Mapping[str, SyllableTiming] = field(default_factory=dict)
    links: Mapping[str, tuple[NoteLink, ...]] = field(default_factory=dict)
    model: dict = field(default_factory=dict)  # 마지막 자동 정렬 모델 (name, version, device, created)
    settings: dict = field(default_factory=dict)

    def timing(self, sid: str) -> SyllableTiming:
        return self.timings.get(sid) or SyllableTiming()

    def linked_notes(self, sid: str) -> tuple[str, ...]:
        return tuple(lk.note_id for lk in self.links.get(sid, ()))

    @cached_property
    def note_to_syllables(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for sid, links in self.links.items():
            for lk in links:
                out.setdefault(lk.note_id, []).append(sid)
        return {k: tuple(v) for k, v in out.items()}

    def with_timings(self, updates: Mapping[str, SyllableTiming | None]) -> "Alignment":
        t = dict(self.timings)
        for sid, v in updates.items():
            if v is None or v == SyllableTiming():
                t.pop(sid, None)
            else:
                t[sid] = v
        return replace(self, timings=t)

    def with_links(self, updates: Mapping[str, Iterable[NoteLink] | None]) -> "Alignment":
        lk = dict(self.links)
        for sid, v in updates.items():
            v = tuple(v or ())
            if v:
                lk[sid] = v
            else:
                lk.pop(sid, None)
        return replace(self, links=lk)

    def restricted_to(self, syllable_ids: Iterable[str]) -> "Alignment":
        keep = set(syllable_ids)
        return replace(self, timings={k: v for k, v in self.timings.items() if k in keep},
                       links={k: v for k, v in self.links.items() if k in keep})


@dataclass(frozen=True)
class HistoryEntry:
    time: str
    action: str
    detail: str = ""


MAX_HISTORY = 200


@dataclass(frozen=True)
class LyricsState:
    text: LyricsText = field(default_factory=LyricsText)
    alignment: Alignment = field(default_factory=Alignment)
    recognition: Recognition | None = None
    history: tuple[HistoryEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.text.is_empty and self.recognition is None

    def with_history(self, action: str, detail: str = "", time: str = "") -> "LyricsState":
        if not time:
            from datetime import datetime, timezone

            time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        h = (self.history + (HistoryEntry(time, action, detail),))[-MAX_HISTORY:]
        return replace(self, history=h)

    def unaligned(self) -> list[str]:
        return [s.id for s in self.text.syllables if not self.alignment.timing(s.id).aligned]

    def unlinked(self) -> list[str]:
        return [s.id for s in self.text.syllables if not self.alignment.links.get(s.id)]


def finite_or_none(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
