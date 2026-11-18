"""분석 단계별 인터페이스.

실제 AI 모델(예: 음원 분리, 피치 추적, 음성 인식)은 여기 있는 추상 클래스를
구현해 ``registry`` 에 등록하면 파이프라인에 연결된다.
각 백엔드는 ``info`` 로 모델 이름·버전을 알려야 한다. 이 정보는 원본 분석 결과에 함께 저장된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from amp.audio.data import AudioData
from amp.core.model import LyricsTrack, ModelInfo, Note, PitchCurve
from amp.core.tempo import TempoMap
from amp.engine.context import StageContext


@dataclass(frozen=True)
class Stems:
    vocals: AudioData
    accompaniment: AudioData | None = None


@dataclass(frozen=True)
class AlignmentResult:
    lyrics: LyricsTrack
    notes: tuple[Note, ...]
    links: tuple[tuple[str, str], ...] = ()  # (음표 id, 음절 id)


class Stage(ABC):
    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...


class SourceSeparator(Stage):
    """믹스에서 보컬과 반주를 분리한다."""

    @abstractmethod
    def separate(self, mix: AudioData, ctx: StageContext) -> Stems: ...


class TempoEstimator(Stage):
    @abstractmethod
    def estimate(self, mix: AudioData, ctx: StageContext) -> TempoMap: ...


class PitchTracker(Stage):
    """보컬 트랙에서 프레임별 f0 곡선을 뽑는다."""

    @abstractmethod
    def track(self, vocals: AudioData, ctx: StageContext) -> PitchCurve: ...


class NoteTranscriber(Stage):
    """f0 곡선을 음표로 나눈다. 비브라토·멜리스마 처리는 이 단계 책임이다."""

    @abstractmethod
    def transcribe(
        self, pitch: PitchCurve, vocals: AudioData, tempo: TempoMap, ctx: StageContext
    ) -> list[Note]: ...


class LyricsRecognizer(Stage):
    @abstractmethod
    def recognize(self, vocals: AudioData, ctx: StageContext) -> LyricsTrack: ...


class LyricsAligner(Stage):
    """음절을 음표에 연결하고 음절 시간 범위를 정한다."""

    @abstractmethod
    def align(
        self,
        lyrics: LyricsTrack,
        notes: list[Note],
        pitch: PitchCurve | None,
        vocals: AudioData,
        ctx: StageContext,
    ) -> AlignmentResult: ...
