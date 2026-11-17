"""샘플(가짜) 분석 백엔드.

실제 AI 모델을 붙이기 전까지 파이프라인과 UI를 끝까지 돌려 보기 위한 구현이다.
입력 오디오의 **길이만** 사용하고 소리 내용은 보지 않는다.
결과는 오디오 길이와 시드로 정해지는 고정 패턴이며, 모든 ModelInfo에 placeholder=True를 단다.

여러 단계가 같은 ``build_plan`` 결과를 공유하므로 단계별로 따로 호출해도 서로 맞아떨어진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from amp.audio.data import AudioData
from amp.core.ids import new_id
from amp.core.model import LyricsTrack, ModelInfo, Note, PitchCurve, Syllable, Vibrato
from amp.core.pitch import midi_to_hz
from amp.core.tempo import TempoMap
from amp.engine.context import StageContext
from amp.engine.interfaces import (
    AlignmentResult,
    LyricsAligner,
    LyricsRecognizer,
    NoteTranscriber,
    PitchTracker,
    SourceSeparator,
    Stems,
    TempoEstimator,
)

SAMPLE_BPM = 100.0
SAMPLE_SEED = 7
FRAME_RATE = 100.0  # 피치 곡선 프레임/초
_SCALE = (55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76)  # G3 ~ E5, C장조
_TEXT = (("샘", "saem"), ("플", "peul"), ("가", "ga"), ("사", "sa"), ("입", "ip"), ("니", "ni"), ("다", "da"))


def _info(name: str = "sample") -> ModelInfo:
    return ModelInfo(name=name, version="1", placeholder=True, params={"seed": SAMPLE_SEED, "bpm": SAMPLE_BPM})


@dataclass(frozen=True)
class PlannedNote:
    start: float
    end: float
    pitch: int
    vibrato: Vibrato | None
    group: int  # 같은 group = 같은 음절 (2개 이상이면 멜리스마)
    detune_cents: float


@dataclass(frozen=True)
class SamplePlan:
    bpm: float
    notes: tuple[PlannedNote, ...]
    group_lines: tuple[int, ...]  # group → 가사 줄(프레이즈) 번호


def build_plan(duration: float, bpm: float = SAMPLE_BPM, seed: int = SAMPLE_SEED) -> SamplePlan:
    rng = np.random.default_rng(seed)
    tempo = TempoMap.constant(bpm)
    notes: list[PlannedNote] = []
    group_lines: list[int] = []
    deg = 5
    beat = 4.0  # 첫 마디는 쉼
    phrase = 0
    end_limit = duration - 0.5
    while tempo.time_at(beat + 1) < end_limit:
        p = 0.0
        while p < 7.5:
            dur = float(rng.choice([0.5, 0.5, 1.0, 1.0, 1.5, 2.0]))
            dur = min(dur, 8.0 - p)
            if tempo.time_at(beat + p + dur) > end_limit:
                break
            group = len(group_lines)
            group_lines.append(phrase)
            if dur >= 1.0 and rng.random() < 0.18:
                # 멜리스마: 한 음절을 음 3개에 걸쳐 부른다
                sub = dur / 3
                d = deg
                for k in range(3):
                    t0 = tempo.time_at(beat + p + k * sub)
                    t1 = tempo.time_at(beat + p + (k + 1) * sub) - (0.04 if k == 2 else 0.0)
                    notes.append(PlannedNote(t0, t1, _SCALE[d], None, group, float(rng.normal(0, 6))))
                    d = int(np.clip(d + rng.choice([-1, 1]), 0, len(_SCALE) - 1))
                deg = d
            else:
                t0 = tempo.time_at(beat + p)
                t1 = tempo.time_at(beat + p + dur) - 0.04
                vib = None
                if t1 - t0 >= 0.6:
                    vib = Vibrato(rate_hz=round(5.2 + rng.random() * 0.8, 2), extent_cents=round(30 + rng.random() * 25, 1))
                notes.append(PlannedNote(t0, t1, _SCALE[deg], vib, group, float(rng.normal(0, 8))))
                deg = int(np.clip(deg + rng.integers(-2, 3), 0, len(_SCALE) - 1))
            p += dur
        beat += 8.0 + (4.0 if phrase % 2 else 0.0)
        phrase += 1
    return SamplePlan(bpm, tuple(notes), tuple(group_lines))


def render_pitch_curve(plan: SamplePlan, duration: float, ctx: StageContext | None = None) -> PitchCurve:
    """계획된 음표를 비브라토·포르타멘토가 섞인 f0 곡선으로 그린다."""
    rng = np.random.default_rng(SAMPLE_SEED + 1)
    n = int(math.ceil(duration * FRAME_RATE))
    midi = np.full(n, np.nan, dtype=np.float64)
    conf = np.zeros(n, dtype=np.float32)
    prev: PlannedNote | None = None
    total = max(len(plan.notes), 1)
    for k, note in enumerate(plan.notes):
        i0 = int(math.ceil(note.start * FRAME_RATE))
        i1 = min(int(math.ceil(note.end * FRAME_RATE)), n)
        if i1 <= i0:
            continue
        t = np.arange(i0, i1) / FRAME_RATE - note.start
        curve = note.pitch + note.detune_cents / 100.0 + np.zeros_like(t)
        if prev is not None and note.start - prev.end < 0.08:
            glide = np.clip(t / 0.07, 0.0, 1.0)
            glide = glide * glide * (3 - 2 * glide)  # smoothstep
            curve += (prev.pitch - note.pitch) * (1.0 - glide)
        if note.vibrato is not None:
            ramp = np.clip((t - 0.15) / 0.25, 0.0, 1.0)
            curve += note.vibrato.extent_cents / 100.0 * ramp * np.sin(2 * np.pi * note.vibrato.rate_hz * t)
        curve += rng.normal(0.0, 0.04, size=t.shape)
        midi[i0:i1] = curve
        conf[i0:i1] = 0.9
        prev = note
        if ctx is not None and k % 200 == 0:
            ctx.progress(k / total, "샘플 피치 곡선 생성 중")
    return PitchCurve(0.0, FRAME_RATE, midi_to_hz(midi).astype(np.float32), conf)


class PassthroughSeparator(SourceSeparator):
    """분리하지 않고 믹스를 모노로 바꿔 보컬 자리에 넘긴다."""

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(name="passthrough", version="1", placeholder=True)

    def separate(self, mix: AudioData, ctx: StageContext) -> Stems:
        ctx.progress(1.0, "음원 분리 건너뜀 (분리 모델 없음)")
        return Stems(vocals=AudioData(mix.mono(), mix.sample_rate), accompaniment=None)


class SampleTempoEstimator(TempoEstimator):
    @property
    def info(self) -> ModelInfo:
        return _info()

    def estimate(self, mix: AudioData, ctx: StageContext) -> TempoMap:
        ctx.progress(1.0, "샘플 템포 설정")
        return TempoMap.constant(SAMPLE_BPM)


class SamplePitchTracker(PitchTracker):
    @property
    def info(self) -> ModelInfo:
        return _info()

    def track(self, vocals: AudioData, ctx: StageContext) -> PitchCurve:
        plan = build_plan(vocals.duration)
        curve = render_pitch_curve(plan, vocals.duration, ctx)
        ctx.progress(1.0, "샘플 피치 곡선 생성 완료")
        return curve


class SampleNoteTranscriber(NoteTranscriber):
    """피치 곡선을 실제로 해석하지 않고 계획된 음표를 그대로 돌려준다."""

    @property
    def info(self) -> ModelInfo:
        return _info()

    def transcribe(self, pitch: PitchCurve, vocals: AudioData, tempo: TempoMap, ctx: StageContext) -> list[Note]:
        plan = build_plan(vocals.duration)
        rng = np.random.default_rng(SAMPLE_SEED + 2)
        notes = [
            Note(
                id=new_id(),
                start=round(p.start, 4),
                end=round(p.end, 4),
                pitch=p.pitch,
                velocity=96,
                confidence=round(float(0.75 + 0.25 * rng.random()), 3),
                vibrato=p.vibrato,
            )
            for p in plan.notes
        ]
        ctx.progress(1.0, "샘플 음표 생성 완료")
        return notes


class SampleLyricsRecognizer(LyricsRecognizer):
    @property
    def info(self) -> ModelInfo:
        return _info()

    def recognize(self, vocals: AudioData, ctx: StageContext) -> LyricsTrack:
        plan = build_plan(vocals.duration)
        syllables = []
        for g, line in enumerate(plan.group_lines):
            text, phon = _TEXT[g % len(_TEXT)]
            syllables.append(Syllable(id=new_id(), text=text, phonetic=phon, line=line))
        ctx.progress(1.0, "샘플 가사 생성 완료")
        return LyricsTrack(language="ko", syllables=tuple(syllables))


class SampleLyricsAligner(LyricsAligner):
    """샘플 계획의 음절 그룹대로 음표에 음절을 연결한다."""

    @property
    def info(self) -> ModelInfo:
        return _info()

    def align(self, lyrics, notes, pitch, vocals, ctx) -> AlignmentResult:
        plan = build_plan(vocals.duration)
        ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
        syls = list(lyrics.syllables)
        if len(ordered) == len(plan.notes):
            groups = [p.group for p in plan.notes]
        else:  # 계획과 다르면 음표 하나에 음절 하나씩
            groups = list(range(len(ordered)))
        linked: list[Note] = []
        spans: dict[int, list[float]] = {}
        for note, g in zip(ordered, groups):
            if g < len(syls):
                linked.append(note.with_(syllable_id=syls[g].id))
                span = spans.setdefault(g, [note.start, note.end])
                span[0] = min(span[0], note.start)
                span[1] = max(span[1], note.end)
            else:
                linked.append(note)
        timed = tuple(
            Syllable(s.id, s.text, s.phonetic, s.line, *(spans.get(g) or (None, None)))
            for g, s in enumerate(syls)
        )
        ctx.progress(1.0, "샘플 가사 정렬 완료")
        return AlignmentResult(LyricsTrack(lyrics.language, timed), tuple(linked))


def register(registry) -> None:
    from amp.engine.registry import StageKind

    registry.register(StageKind.SEPARATOR, "passthrough", PassthroughSeparator, "분리하지 않음 (믹스를 그대로 사용)")
    registry.register(StageKind.TEMPO, "sample", SampleTempoEstimator, "샘플: 고정 100 BPM")
    registry.register(StageKind.PITCH, "sample", SamplePitchTracker, "샘플: 합성 피치 곡선")
    registry.register(StageKind.NOTES, "sample", SampleNoteTranscriber, "샘플: 고정 패턴 음표")
    registry.register(StageKind.LYRICS, "sample", SampleLyricsRecognizer, "샘플: 고정 가사")
    registry.register(StageKind.ALIGNER, "sample", SampleLyricsAligner, "샘플: 계획된 음절 연결")
