"""분석 단계를 순서대로 실행해 ``AnalysisResult`` 를 만든다.

이 모듈은 스레드나 프로세스를 직접 다루지 않는다.
UI 쪽 작업 실행기가 ``run`` 을 백그라운드에서 호출하고, 취소는 CancelToken으로 전달한다.
입력·출력이 모두 피클 가능한 데이터여서 나중에 별도 프로세스로 옮기기 쉽다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from amp.audio.data import AudioData
from amp.core.errors import AmpError, AnalysisCancelled, AnalysisError
from amp.core.model import AnalysisResult, LyricsTrack
from amp.engine.context import CancelToken, ProgressFn, StageContext
from amp.engine.registry import BackendRegistry, StageKind

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """단계별 백엔드 이름. lyrics/aligner를 None으로 두면 가사 단계를 건너뛴다."""

    separator: str = "passthrough"
    tempo: str = "sample"
    pitch: str = "sample"
    notes: str = "sample"
    lyrics: str | None = "sample"
    aligner: str | None = "sample"


# 단계별 진행률 비중 (합 1.0)
_WEIGHTS = {
    StageKind.SEPARATOR: 0.30,
    StageKind.TEMPO: 0.10,
    StageKind.PITCH: 0.30,
    StageKind.NOTES: 0.10,
    StageKind.LYRICS: 0.10,
    StageKind.ALIGNER: 0.10,
}


class AnalysisPipeline:
    def __init__(self, registry: BackendRegistry, config: PipelineConfig | None = None) -> None:
        self.registry = registry
        self.config = config or PipelineConfig()

    def run(
        self,
        mix: AudioData,
        progress: ProgressFn | None = None,
        token: CancelToken | None = None,
        audio_fingerprint: str = "",
    ) -> AnalysisResult:
        token = token or CancelToken()
        report = progress or (lambda f, m: None)
        cfg = self.config
        models = {}
        offset = 0.0

        def ctx_for(kind: StageKind) -> StageContext:
            nonlocal offset
            ctx = StageContext(report, token, offset, _WEIGHTS[kind])
            offset += _WEIGHTS[kind]
            return ctx

        def build(kind: StageKind, name: str):
            stage = self.registry.create(kind, name)
            models[kind.value] = stage.info
            return stage

        try:
            token.raise_if_cancelled()
            separator = build(StageKind.SEPARATOR, cfg.separator)
            stems = separator.separate(mix, ctx_for(StageKind.SEPARATOR))

            tempo = build(StageKind.TEMPO, cfg.tempo).estimate(mix, ctx_for(StageKind.TEMPO))
            pitch = build(StageKind.PITCH, cfg.pitch).track(stems.vocals, ctx_for(StageKind.PITCH))
            notes = build(StageKind.NOTES, cfg.notes).transcribe(
                pitch, stems.vocals, tempo, ctx_for(StageKind.NOTES)
            )

            lyrics: LyricsTrack | None = None
            links: tuple[tuple[str, str], ...] = ()
            if cfg.lyrics:
                lyrics = build(StageKind.LYRICS, cfg.lyrics).recognize(stems.vocals, ctx_for(StageKind.LYRICS))
                if cfg.aligner:
                    aligned = build(StageKind.ALIGNER, cfg.aligner).align(
                        lyrics, notes, pitch, stems.vocals, ctx_for(StageKind.ALIGNER)
                    )
                    lyrics, notes, links = aligned.lyrics, list(aligned.notes), aligned.links
            token.raise_if_cancelled()
        except (AnalysisCancelled, AnalysisError):
            raise
        except AmpError as e:
            raise AnalysisError(str(e)) from e
        except Exception as e:  # 백엔드 내부 오류를 사용자 메시지로 감싼다
            log.exception("분석 단계 실패")
            raise AnalysisError(f"분석 중 오류가 발생했습니다: {e}") from e

        report(1.0, "분석 완료")
        return AnalysisResult(
            duration=mix.duration,
            tempo=tempo,
            notes=tuple(sorted(notes, key=lambda n: (n.start, n.pitch))),
            pitch_curve=pitch,
            lyrics=lyrics,
            lyric_links=links,
            models=models,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            audio_fingerprint=audio_fingerprint,
        )
