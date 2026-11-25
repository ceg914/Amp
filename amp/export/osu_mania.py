"""멜로디 음표 → osu!mania 비트맵 (.osu / .osz) (v0.6.8). 기본 4K, Easy·Normal·Hard 세 난이도.

만드는 방법
- 음표: 보컬 작업본(사용자가 고친 음표 포함). 시작·끝을 박 격자에 맞추고, 난이도마다 음표 사이 최소 간격으로 솎는다
- 열: 멜로디가 올라가면 오른쪽, 내려가면 왼쪽으로 옮긴다(5반음 이상이면 두 칸). 끝 열에 닿으면 음역 안 위치로 다시 맞춘다.
  같은 음을 짧게 반복하면 같은 열 연타(잭)를 피해 옆 열로 옮긴다
- 롱노트: 길게 끄는 음. 다음 음 한 칸 전에 놓도록 끝을 자르고, 그래도 난이도별 최소 길이보다 길면 롱노트
- 타이밍: 프로젝트 템포(BPM·첫 박)를 빨간 선(uninherited)으로. 첫 선은 첫 박에서 마디 단위로 앞당겨 0ms 이후 첫 마디에 둔다
- 음원: .osz 안에 OGG(Vorbis)로 넣는다. 분석에 쓴 같은 디코딩 결과를 그대로 인코딩해 시간축이 같다(MP3는 인코더 지연을
  디코더마다 다르게 처리할 수 있어 쓰지 않음)

osu! 파일 형식(v14)의 mania 규칙: Mode 3, CircleSize = 키 수, 열 = floor(x × 키 수 / 512), 롱노트 type 128과 끝 시각.
"""

from __future__ import annotations

import io
import math
import os
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np

from amp.core.model import Note

KEYS = 4


@dataclass(frozen=True)
class Difficulty:
    name: str
    snap: float  # 박 격자 (박)
    min_gap: float  # 음표 시작 사이 최소 간격 (박)
    hold_min: float  # 롱노트 최소 길이 (박)
    jack_gap: float  # 이보다 가까우면 같은 열을 피한다 (박)
    od: float
    hp: float
    ornaments: bool  # 장식음을 넣는지


DIFFICULTIES = {
    "Easy": Difficulty("Easy", 0.5, 1.0, 2.0, 2.0, 6.0, 6.0, False),
    "Normal": Difficulty("Normal", 0.5, 0.5, 1.0, 1.0, 7.0, 7.0, False),
    "Hard": Difficulty("Hard", 0.25, 0.25, 1.0, 0.5, 8.0, 8.0, True),
}


@dataclass(frozen=True)
class ManiaNote:
    beat: float
    column: int
    end_beat: float | None  # None이면 단타
    pitch: int


@dataclass
class Meta:
    title: str = ""  # ASCII (osu! Title)
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = "SONATLAS"
    source: str = ""
    tags: str = "sonatlas auto-generated melody"
    offset_ms: float = 0.0  # 전체 시각 보정 (osu!에서 확인한 뒤 맞출 때)


@dataclass
class ChartSummary:
    name: str
    notes: int = 0
    holds: int = 0
    dropped: int = 0  # 간격 때문에 솎은 음표
    columns: list[int] = field(default_factory=lambda: [0] * KEYS)


# --- 채보 -------------------------------------------------------------------------

def _column_for(pitch: float, lo: float, hi: float, keys: int) -> int:
    if hi <= lo:
        return (keys - 1) // 2
    return int(min(keys - 1, max(0, round((pitch - lo) / (hi - lo) * (keys - 1)))))


def chart(notes: list[Note], tempo, diff: Difficulty, keys: int = KEYS) -> tuple[list[ManiaNote], ChartSummary]:
    """음표(초) → mania 음표(박). 한 번에 한 음(멜로디)으로 만든다."""
    summary = ChartSummary(diff.name, columns=[0] * keys)
    src = sorted((n for n in notes if diff.ornaments or not getattr(n, "ornament", False)), key=lambda n: n.start)
    snapped: list[tuple[float, float, int]] = []
    for n in src:
        b0 = round(tempo.beat_at(n.start) / diff.snap) * diff.snap
        b1 = round(tempo.beat_at(n.end) / diff.snap) * diff.snap
        if b1 <= b0:
            b1 = b0 + diff.snap
        if snapped and abs(snapped[-1][0] - b0) < 1e-9:
            summary.dropped += 1  # 같은 칸에 둘: 앞 것만
            continue
        snapped.append((b0, b1, int(n.pitch)))
    kept: list[tuple[float, float, int]] = []
    for s in snapped:
        if kept and s[0] - kept[-1][0] < diff.min_gap - 1e-9:
            summary.dropped += 1
            continue
        kept.append(s)
    if not kept:
        return [], summary
    pitches = np.array([p for _b0, _b1, p in kept], dtype=float)
    lo, hi = float(np.percentile(pitches, 10)), float(np.percentile(pitches, 90))
    out: list[ManiaNote] = []
    prev_col, prev_pitch, prev_beat = None, None, None
    for i, (b0, b1, pitch) in enumerate(kept):
        nxt = kept[i + 1][0] if i + 1 < len(kept) else None
        end = b1 if nxt is None else min(b1, nxt - diff.snap)
        hold = end if end - b0 >= diff.hold_min - 1e-9 else None
        if prev_col is None:
            col = _column_for(pitch, lo, hi, keys)
        else:
            d = pitch - prev_pitch
            step = 0 if d == 0 else (1 if 0 < d < 5 else 2 if d >= 5 else -1 if -5 < d < 0 else -2)
            col = prev_col + step
            if not 0 <= col < keys:  # 끝에 닿음: 음역 안 위치로 다시 맞추되 같은 열은 피한다
                col = _column_for(pitch, lo, hi, keys)
                if col == prev_col:
                    col = prev_col - (1 if step > 0 else -1)
            gap = b0 - prev_beat
            if col == prev_col and gap < diff.jack_gap - 1e-9:  # 짧은 같은 열 연타: 가운데 쪽 옆 열로
                col = prev_col + (1 if prev_col < (keys - 1) / 2 else -1)
            col = int(min(keys - 1, max(0, col)))
        out.append(ManiaNote(b0, col, hold, pitch))
        summary.notes += 1
        summary.holds += hold is not None
        summary.columns[col] += 1
        prev_col, prev_pitch, prev_beat = col, pitch, b0
    return out, summary


# --- .osu 텍스트 ----------------------------------------------------------------------

def ascii_text(s: str) -> str:
    """osu! Title·Artist용 ASCII. 한글은 로마자로 읽고(uroman과 같은 표), 그 밖의 글자는 뺀다."""
    from amp.lyrics.text import hangul_reading

    out = []
    for ch in s or "":
        if ord(ch) < 128:
            out.append(ch)
        else:
            r = hangul_reading(ch)
            if r:
                out.append(r)
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    return " ".join(w[:1].upper() + w[1:] for w in text.split(" ")) if text else ""


def timing_points(tempo, offset_ms: float = 0.0) -> list[str]:
    bar = tempo.beats_per_bar
    lines = []
    for k, p in enumerate(tempo.points):
        beat_ms = 60000.0 / p.bpm
        t = p.time * 1000.0 + offset_ms
        if k == 0:  # 첫 박에서 마디 단위로 앞당겨 0ms 이후 첫 마디에 둔다 (첫 음 앞 마디선도 맞게)
            t -= math.floor(t / (beat_ms * bar)) * beat_ms * bar if t > 0 else 0.0
        lines.append(f"{int(round(t))},{beat_ms!r},{bar},1,0,60,1,0")
    return lines


def ms(tempo, beat: float, offset_ms: float = 0.0) -> int:
    return int(round(tempo.time_at(beat) * 1000.0 + offset_ms))


def osu_text(notes: list[ManiaNote], tempo, diff: Difficulty, meta: Meta, audio_name: str, keys: int = KEYS) -> str:
    times = [ms(tempo, n.beat, meta.offset_ms) for n in notes]
    preview = times[0] if times else -1
    lines = [
        "osu file format v14", "",
        "[General]", f"AudioFilename: {audio_name}", "AudioLeadIn: 0", f"PreviewTime: {preview}", "Countdown: 0",
        "SampleSet: Soft", "StackLeniency: 0.7", "Mode: 3", "LetterboxInBreaks: 0", "SpecialStyle: 0",
        "WidescreenStoryboard: 0", "",
        "[Editor]", "DistanceSpacing: 1", f"BeatDivisor: {int(round(1 / diff.snap))}", "GridSize: 4", "TimelineZoom: 1", "",
        "[Metadata]", f"Title:{meta.title}", f"TitleUnicode:{meta.title_unicode or meta.title}",
        f"Artist:{meta.artist}", f"ArtistUnicode:{meta.artist_unicode or meta.artist}", f"Creator:{meta.creator}",
        f"Version:{diff.name}", f"Source:{meta.source}", f"Tags:{meta.tags}", "BeatmapID:0", "BeatmapSetID:-1", "",
        "[Difficulty]", f"HPDrainRate:{diff.hp:g}", f"CircleSize:{keys}", f"OverallDifficulty:{diff.od:g}",
        "ApproachRate:5", "SliderMultiplier:1.4", "SliderTickRate:1", "",
        "[Events]", "//Background and Video events", "//Break Periods", "",
        "[TimingPoints]", *timing_points(tempo, meta.offset_ms), "", "",
        "[HitObjects]",
    ]
    for n, t in zip(notes, times):
        x = int(math.floor((n.column * 512 + 256) / keys))  # 4K: 64, 192, 320, 448
        if n.end_beat is None:
            lines.append(f"{x},192,{t},1,0,0:0:0:0:")
        else:
            lines.append(f"{x},192,{t},128,0,{ms(tempo, n.end_beat, meta.offset_ms)}:0:0:0:0:")
    return "\r\n".join(lines) + "\r\n"


def parse_hit_objects(text: str, keys: int = KEYS) -> list[tuple[int, int, int | None]]:
    """시험·검증용: [HitObjects]의 (열, 시작 ms, 끝 ms 또는 None)."""
    out, on = [], False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            on = line == "[HitObjects]"
            continue
        if not on or not line:
            continue
        parts = line.split(",")
        col = int(int(parts[0]) * keys / 512)
        typ = int(parts[3])
        end = int(parts[5].split(":")[0]) if typ & 128 else None
        out.append((col, int(parts[2]), end))
    return out


# --- .osz ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "", name).strip() or "SONATLAS"


def encode_ogg(audio) -> bytes:
    """분석에 쓴 디코딩 결과 그대로 OGG Vorbis로 (osu!가 읽는 형식, 시간축 그대로)."""
    import soundfile as sf

    buf = io.BytesIO()
    data = audio.samples[:, :2]
    # 한 번에 길게 쓰면 libsndfile Vorbis 인코더가 죽는다(60초 이상 한 번에 쓸 때 segfault 확인). 1초씩 나눠 쓴다
    with sf.SoundFile(buf, "w", samplerate=audio.sample_rate, channels=data.shape[1], format="OGG",
                      subtype="VORBIS") as f:
        step = int(audio.sample_rate)
        for i in range(0, data.shape[0], step):
            f.write(np.clip(data[i:i + step], -1.0, 1.0))
    return buf.getvalue()


def build_osz(notes: list[Note], tempo, audio, meta: Meta, difficulties=("Easy", "Normal", "Hard"),
              keys: int = KEYS) -> tuple[bytes, list[ChartSummary]]:
    audio_name = "audio.ogg"
    files: dict[str, bytes] = {audio_name: encode_ogg(audio)}
    summaries = []
    meta.title = meta.title or "Untitled"
    meta.artist = meta.artist or "Unknown"  # osu!는 빈 제목·아티스트를 싫어한다
    base = _safe(f"{meta.artist} - {meta.title} ({meta.creator})")
    for name in difficulties:
        diff = DIFFICULTIES[name]
        chart_notes, s = chart(notes, tempo, diff, keys)
        files[f"{base} [{name}].osu"] = osu_text(chart_notes, tempo, diff, meta, audio_name, keys).encode("utf-8")
        summaries.append(s)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, data in files.items():
            z.writestr(fname, data)
    return buf.getvalue(), summaries


def write_osz(path: str | os.PathLike, notes: list[Note], tempo, audio, meta: Meta,
              difficulties=("Easy", "Normal", "Hard"), keys: int = KEYS) -> list[ChartSummary]:
    from amp.core.fileio import atomic_write_bytes

    data, summaries = build_osz(notes, tempo, audio, meta, difficulties, keys)
    atomic_write_bytes(path, data)
    return summaries
