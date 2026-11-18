"""채보 데이터 모델.

설계 원칙
- ``AnalysisResult`` 는 분석 엔진이 만든 원본이다. 만든 뒤에는 바꾸지 않는다.
- ``Score`` 는 사용자가 편집하는 작업본이다. 분석을 실행하면 원본을 복사해 만든다.
- 작업본의 음표는 ``origin_ids`` 로 어떤 원본 음표에서 왔는지 기록한다.
  분할하면 두 조각이 같은 출처를 물려받고, 병합하면 출처를 합친다.
  사용자가 새로 그린 음표는 출처가 비어 있다.
- 가사(텍스트·음절 시간·음절↔음표 연결)는 ``Project.lyrics`` (``amp.lyrics``)가 따로 가진다.
  음표는 가사를 모르며, 연결은 alignment 쪽 한 곳에만 있다 (STEP 3에서 음표의 ``syllable_id`` 를 옮김).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Iterable

import numpy as np

from amp.core.errors import EditError
from amp.core.pitch import MIDI_MAX, MIDI_MIN, hz_to_midi, note_name
from amp.core.tempo import TempoMap
from amp.lyrics.model import LyricsState

DEFAULT_BPM = 120.0


@dataclass(frozen=True)
class Vibrato:
    """음표 하나에 대한 비브라토 분석값."""

    rate_hz: float
    extent_cents: float  # 중심에서 위아래로 흔들리는 폭


@dataclass(frozen=True)
class Note:
    id: str
    start: float  # 초
    end: float  # 초
    pitch: int  # MIDI 음 번호
    velocity: int = 100
    confidence: float | None = None  # 분석기가 준 신뢰도 0~1
    vibrato: Vibrato | None = None
    origin_ids: tuple[str, ...] = ()
    ornament: bool = False  # 최소 길이보다 짧지만 신뢰도가 높아 남긴 장식음

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise EditError("음표 시간은 유한한 값이어야 합니다.")
        if self.end <= self.start:
            raise EditError(f"음표 끝({self.end:.3f})이 시작({self.start:.3f})보다 뒤여야 합니다.")
        if not (MIDI_MIN <= self.pitch <= MIDI_MAX):
            raise EditError(f"음높이는 {MIDI_MIN}~{MIDI_MAX} 범위여야 합니다: {self.pitch}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        """음 이름 (예: C4)."""
        return note_name(self.pitch)

    def with_(self, **changes) -> "Note":
        return replace(self, **changes)


@dataclass(frozen=True)
class PitchCurve:
    """일정 간격으로 샘플링한 기본 주파수(f0) 곡선.

    무성 구간은 ``f0_hz`` 가 NaN이다.
    """

    start_time: float
    frame_rate: float  # 초당 프레임 수
    f0_hz: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        f0 = np.asarray(self.f0_hz, dtype=np.float32)
        conf = np.asarray(self.confidence, dtype=np.float32)
        if f0.ndim != 1 or conf.shape != f0.shape:
            raise ValueError("f0_hz와 confidence는 길이가 같은 1차원 배열이어야 합니다.")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate는 0보다 커야 합니다.")
        f0.flags.writeable = False
        conf.flags.writeable = False
        object.__setattr__(self, "f0_hz", f0)
        object.__setattr__(self, "confidence", conf)

    def __len__(self) -> int:
        return int(self.f0_hz.shape[0])

    @property
    def end_time(self) -> float:
        return self.start_time + len(self) / self.frame_rate

    def times(self) -> np.ndarray:
        return self.start_time + np.arange(len(self)) / self.frame_rate

    def midi(self) -> np.ndarray:
        return hz_to_midi(self.f0_hz)

    def frame_range(self, t0: float, t1: float) -> tuple[int, int]:
        """[t0, t1] 을 덮는 프레임 인덱스 범위 [i0, i1)."""
        n = len(self)
        i0 = int(math.floor((t0 - self.start_time) * self.frame_rate))
        i1 = int(math.ceil((t1 - self.start_time) * self.frame_rate)) + 1
        return max(0, min(n, i0)), max(0, min(n, i1))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PitchCurve):
            return NotImplemented
        return (
            self.start_time == other.start_time
            and self.frame_rate == other.frame_rate
            and np.array_equal(self.f0_hz, other.f0_hz, equal_nan=True)
            and np.array_equal(self.confidence, other.confidence, equal_nan=True)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Syllable:
    """STEP 1 형식의 음절 (샘플 분석기 출력·이전 파일 읽기 전용). 편집용 가사는 ``amp.lyrics`` 를 쓴다."""

    id: str
    text: str
    phonetic: str | None = None  # 발음 기호(IPA 또는 로마자 표기)
    line: int = 0  # 가사 줄 번호
    start: float | None = None  # 정렬 결과 (없으면 None)
    end: float | None = None


@dataclass(frozen=True)
class LyricsTrack:
    language: str = "ko"
    syllables: tuple[Syllable, ...] = ()

    def by_id(self) -> dict[str, Syllable]:
        return {s.id: s for s in self.syllables}


@dataclass(frozen=True)
class ModelInfo:
    """분석 단계를 수행한 모델/알고리즘 정보."""

    name: str
    version: str = ""
    placeholder: bool = False  # 실제 분석이 아닌 샘플/더미 구현이면 True
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    """분석 엔진이 만든 원본 결과. 편집하지 않는다."""

    duration: float
    tempo: TempoMap
    notes: tuple[Note, ...]
    pitch_curve: PitchCurve | None = None
    lyrics: LyricsTrack | None = None
    models: dict[str, ModelInfo] = field(default_factory=dict)
    created_at: str = ""
    audio_fingerprint: str = ""
    # STEP 2에서 추가한 필드 (모두 기본값이 있어 이전 데이터도 읽힌다)
    onset_envelope: np.ndarray | None = field(default=None, compare=False)  # 피치 곡선과 같은 프레임 격자
    tempo_info: dict = field(default_factory=dict)  # 알고리즘, 후보(2배·절반), 신뢰 여부, 첫 박
    warnings: tuple[str, ...] = ()
    settings: dict = field(default_factory=dict)  # 분석에 쓴 AnalysisSettings
    # 샘플 분석기가 만든 (음표 id, 음절 id) 연결. 반영할 때 프로젝트 가사로 옮기고 파일에는 쓰지 않는다
    lyric_links: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    @property
    def tempo_candidates(self) -> tuple[float, ...]:
        return tuple(float(c["bpm"]) for c in self.tempo_info.get("candidates", ()))

    @property
    def is_placeholder(self) -> bool:
        return any(m.placeholder for m in self.models.values())

    @cached_property
    def note_map(self) -> dict[str, Note]:
        """id → 원본 음표. 결과가 불변이므로 한 번만 만든다."""
        return {n.id: n for n in self.notes}


class Score:
    """사용자가 편집하는 작업본.

    음표는 id → Note 사전에 보관한다. Note가 불변이므로 수정은 교체로 이뤄진다.
    """

    def __init__(
        self,
        tempo: TempoMap | None = None,
        notes: Iterable[Note] = (),
    ) -> None:
        self.tempo = tempo or TempoMap.constant(DEFAULT_BPM)
        self._notes: dict[str, Note] = {}
        for n in notes:
            if n.id in self._notes:
                raise EditError(f"중복된 음표 id: {n.id}")
            self._notes[n.id] = n

    @classmethod
    def from_analysis(cls, result: AnalysisResult) -> "Score":
        notes = [n.with_(origin_ids=(n.id,)) for n in result.notes]
        return cls(result.tempo, notes)

    def __len__(self) -> int:
        return len(self._notes)

    def __contains__(self, note_id: object) -> bool:
        return note_id in self._notes

    def get(self, note_id: str) -> Note:
        try:
            return self._notes[note_id]
        except KeyError:
            raise EditError(f"음표를 찾을 수 없습니다: {note_id}") from None

    def notes(self) -> list[Note]:
        """시작 시간 순으로 정렬한 음표 목록."""
        return sorted(self._notes.values(), key=lambda n: (n.start, n.pitch, n.id))

    def replace_notes(self, removed_ids: Iterable[str], added: Iterable[Note]) -> None:
        """음표 일부를 지우고 새 음표를 넣는다. 모든 편집이 이 한 경로를 지난다.

        먼저 전부 검사한 뒤 반영하므로 실패하면 작업본이 바뀌지 않는다.
        """
        removed_ids = list(removed_ids)
        added = list(added)
        for rid in removed_ids:
            if rid not in self._notes:
                raise EditError(f"지울 음표가 없습니다: {rid}")
        remaining = set(self._notes) - set(removed_ids)
        new_ids = [n.id for n in added]
        if len(set(new_ids)) != len(new_ids) or remaining & set(new_ids):
            raise EditError("추가하려는 음표 id가 기존 음표와 겹칩니다.")
        for rid in removed_ids:
            del self._notes[rid]
        for n in added:
            self._notes[n.id] = n

    def end_time(self) -> float:
        return max((n.end for n in self._notes.values()), default=0.0)

    def copy(self) -> "Score":
        return Score(self.tempo, self._notes.values())


@dataclass(frozen=True)
class AudioRef:
    """프로젝트가 참조하는 원본 오디오 파일 정보."""

    path: str
    fingerprint: str = ""
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0


@dataclass
class Project:
    audio: AudioRef | None = None
    analysis: AnalysisResult | None = None
    score: Score = field(default_factory=Score)
    lyrics: LyricsState = field(default_factory=LyricsState)

    def note_status(self, note: Note) -> str:
        """'original'(원본 그대로), 'modified'(수정됨), 'user'(사용자 추가) 중 하나."""
        if not note.origin_ids:
            return "user"
        if self.analysis is None or len(note.origin_ids) != 1:
            return "modified"
        src = self.analysis.note_map.get(note.origin_ids[0])
        if src is None:
            return "modified"
        same = (
            abs(src.start - note.start) < 1e-9
            and abs(src.end - note.end) < 1e-9
            and src.pitch == note.pitch
        )
        return "original" if same else "modified"
