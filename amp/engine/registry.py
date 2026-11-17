"""분석 백엔드 등록소.

단계 종류마다 이름 → 생성 함수를 보관한다.
새 모델을 붙일 때는 백엔드 클래스를 만들고 ``register`` 로 등록한 뒤
``PipelineConfig`` 에서 그 이름을 고르면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from amp.engine.interfaces import (
    LyricsAligner,
    LyricsRecognizer,
    NoteTranscriber,
    PitchTracker,
    SourceSeparator,
    Stage,
    TempoEstimator,
)


class StageKind(str, Enum):
    SEPARATOR = "separator"
    TEMPO = "tempo"
    PITCH = "pitch"
    NOTES = "notes"
    LYRICS = "lyrics"
    ALIGNER = "aligner"


STAGE_BASES: dict[StageKind, type[Stage]] = {
    StageKind.SEPARATOR: SourceSeparator,
    StageKind.TEMPO: TempoEstimator,
    StageKind.PITCH: PitchTracker,
    StageKind.NOTES: NoteTranscriber,
    StageKind.LYRICS: LyricsRecognizer,
    StageKind.ALIGNER: LyricsAligner,
}


@dataclass(frozen=True)
class BackendSpec:
    kind: StageKind
    name: str
    factory: Callable[[], Stage]
    description: str = ""


class BackendRegistry:
    def __init__(self) -> None:
        self._specs: dict[StageKind, dict[str, BackendSpec]] = {k: {} for k in StageKind}

    def register(self, kind: StageKind, name: str, factory: Callable[[], Stage], description: str = "") -> None:
        if name in self._specs[kind]:
            raise ValueError(f"이미 등록된 백엔드입니다: {kind.value}/{name}")
        self._specs[kind][name] = BackendSpec(kind, name, factory, description)

    def available(self, kind: StageKind) -> list[BackendSpec]:
        return list(self._specs[kind].values())

    def create(self, kind: StageKind, name: str) -> Stage:
        try:
            spec = self._specs[kind][name]
        except KeyError:
            raise KeyError(f"등록되지 않은 백엔드입니다: {kind.value}/{name}") from None
        stage = spec.factory()
        base = STAGE_BASES[kind]
        if not isinstance(stage, base):
            raise TypeError(f"{kind.value}/{name} 백엔드는 {base.__name__}를 구현해야 합니다.")
        return stage


def default_registry() -> BackendRegistry:
    """현재 버전에서 쓸 수 있는 백엔드를 등록한 등록소."""
    from amp.engine.backends import sample

    reg = BackendRegistry()
    sample.register(reg)
    return reg
