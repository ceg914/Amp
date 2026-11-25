"""멜로디 음표 → osu!mania 비트맵 (.osu / .osz) (v0.6.8~). 기본 4K, 기본 난이도 Normal·Hard·Insane (Easy도 고를 수 있음).

만드는 방법
- 음표: 보컬 작업본(사용자가 고친 음표 포함). 시작·끝을 박 격자에 맞추고, 난이도마다 음표 사이 최소 간격으로 솎는다
- 열: 멜로디가 올라가면 오른쪽, 내려가면 왼쪽으로 옮긴다(5반음 이상이면 두 칸). 끝 열에 닿으면 음역 안 위치로 다시 맞춘다.
  같은 음을 짧게 반복하면 같은 열 연타(잭)를 피해 옆 열로 옮긴다
- 롱노트: 길게 끄는 음. 다음 음 한 칸 전에 놓도록 끝을 자르고, 그래도 난이도별 최소 길이보다 길면 롱노트
- 난이도 올리기 (v0.6.9): 마디 첫 박(Insane은 강한 타격도)에 멜로디와 먼 열을 함께 누르는 동시치기, 멜로디 사이에 드럼(없으면
  반주) 타격을 따온 채우기 노트. 롱노트를 잡고 있는 동안에도 다른 열에 채우기가 온다
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
    snap: float  # 멜로디 박 격자 (박)
    min_gap: float  # 멜로디 음표 시작 사이 최소 간격 (박)
    hold_min: float  # 롱노트 최소 길이 (박)
    jack_gap: float  # 이보다 가까우면 같은 열을 피한다 (박)
    od: float
    hp: float
    ornaments: bool  # 장식음을 넣는지
    # v0.6.9 난이도 올리기: 멜로디에 동시치기(코드)와 리듬 채우기(드럼·반주의 타격)를 더한다
    chord: str = "none"  # none / downbeat(마디 첫 박) / strong(마디 첫 박 + 강한 타격)
    chord_top: float = 0.0  # strong: 타격 세기 상위 몇 %에서 동시치기
    fill_snap: float = 0.0  # 리듬 채우기 격자 (0이면 안 함)
    fill_gap: float = 1.0  # 채우기 노트와 앞 줄 사이 최소 간격 (박)
    fill_top: float = 0.0  # 타격 세기 상위 몇 %를 채우기로 쓰는지
    chord_size: int = 1  # 동시치기 때 멜로디에 더하는 노트 수 (Insane 마디 첫 박은 2 = 세 열)
    fill_jump_top: float = 0.0  # 채우기 칸 중 가장 센 몇 %는 두 열을 함께 (점프)


# 조각 나비(94 BPM)에서 rosu-pp 별점: Easy 0.72 · Normal 1.18 · Hard 2.08 · Insane 약 2.6 · Expert 3.0 (드럼 소리로 채우기).
# 별점은 곡 BPM·드럼 밀도에 따라 달라진다 (느린 곡은 같은 밀도에서 별점이 낮다)
DIFFICULTIES = {
    "Easy": Difficulty("Easy", 0.5, 1.0, 2.0, 2.0, 6.0, 6.0, False),
    "Normal": Difficulty("Normal", 0.25, 0.5, 1.0, 1.0, 7.0, 7.0, False, chord="downbeat",
                         fill_snap=0.5, fill_gap=0.5, fill_top=0.35),
    "Hard": Difficulty("Hard", 0.25, 0.25, 1.0, 0.5, 8.0, 7.5, True, chord="downbeat",
                       fill_snap=0.25, fill_gap=0.25, fill_top=0.45, fill_jump_top=0.1),
    "Insane": Difficulty("Insane", 0.25, 0.25, 1.0, 0.5, 8.5, 8.0, True, chord="strong", chord_top=0.5, chord_size=2,
                         fill_snap=0.25, fill_gap=0.25, fill_top=0.7, fill_jump_top=0.3),
    "Expert": Difficulty("Expert", 0.25, 0.25, 1.0, 0.25, 9.0, 8.5, True, chord="strong", chord_top=0.7,
                         chord_size=2, fill_snap=0.25, fill_gap=0.25, fill_top=0.85, fill_jump_top=0.5),
}
DEFAULT_DIFFICULTIES = ("Normal", "Hard", "Insane", "Expert")


@dataclass(frozen=True)
class ManiaNote:
    beat: float
    column: int
    end_beat: float | None  # None이면 단타
    pitch: int  # 멜로디 음높이. 동시치기·채우기 노트는 -1
    kind: str = "melody"  # melody / chord / fill


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
    dropped: int = 0  # 간격 때문에 솎은 멜로디 음표
    chords: int = 0
    fills: int = 0
    columns: list[int] = field(default_factory=lambda: [0] * KEYS)


# --- 리듬 (드럼·반주의 타격) ------------------------------------------------------------

def rhythm_onsets(mono: np.ndarray, sr: int) -> np.ndarray:
    """(시각 초, 세기) 배열: 11.6ms마다의 타격 세기(onset strength). 95% 지점을 1로 맞춘다.
    드럼 소리가 있으면 그것을, 없으면 반주를 넣는다. 격자 칸마다 가장 센 값을 쓰므로 검출한 타격 봉우리만이 아니라
    드럼이 바쁜 구간의 모든 칸이 후보가 된다 (봉우리만 쓰면 조각 나비에서 초당 2.1개뿐이라 어려운 난이도를 못 만들었다)."""
    import librosa

    y = np.asarray(mono, dtype=np.float32)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
    if len(env) == 0:
        return np.zeros((0, 2))
    t = librosa.frames_to_time(np.arange(len(env)), sr=22050, hop_length=hop)
    return np.stack([t, env / (np.percentile(env, 95) + 1e-9)], axis=1)


# --- 채보 -------------------------------------------------------------------------

def _column_for(pitch: float, lo: float, hi: float, keys: int) -> int:
    if hi <= lo:
        return (keys - 1) // 2
    return int(min(keys - 1, max(0, round((pitch - lo) / (hi - lo) * (keys - 1)))))


def _melody(notes, tempo, diff: Difficulty, summary: ChartSummary) -> list[tuple[float, float, int]]:
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
    return kept


def _strengths(onsets, tempo, step: float) -> dict[float, float]:
    """격자 칸(박) → 그 칸 ±반 칸 안에서 가장 센 타격 세기."""
    out: dict[float, float] = {}
    if onsets is None or step <= 0:
        return out
    arr = np.asarray(onsets, dtype=float).reshape(-1, 2)
    if not len(arr):
        return out
    beats = np.array([tempo.beat_at(float(t)) for t in arr[:, 0]])
    slots = np.round(beats / step) * step
    for b, s in zip(slots, arr[:, 1]):
        b = float(b)
        if s > out.get(b, -1.0):
            out[b] = float(s)
    return out


def chart(notes: list[Note], tempo, diff: Difficulty, keys: int = KEYS,
          onsets=None) -> tuple[list[ManiaNote], ChartSummary]:
    """음표(초) → mania 음표(박). 멜로디가 뼈대이고, 난이도에 따라 동시치기와 리듬 채우기를 더한다.

    onsets: rhythm_onsets()의 (시각, 세기). 없으면 동시치기는 마디 첫 박만, 채우기는 하지 않는다."""
    summary = ChartSummary(diff.name, columns=[0] * keys)
    melody = _melody(notes, tempo, diff, summary)
    fill_step = diff.fill_snap
    fine = _strengths(onsets, tempo, diff.snap)  # 멜로디 칸의 타격 세기 (동시치기)
    fills = _strengths(onsets, tempo, fill_step) if fill_step > 0 else {}
    fill_thr = float(np.quantile(list(fills.values()), 1 - diff.fill_top)) if fills and diff.fill_top > 0 else math.inf
    jump_thr = (float(np.quantile(list(fills.values()), 1 - diff.fill_jump_top))
                if fills and diff.fill_jump_top > 0 else math.inf)
    chord_thr = (float(np.quantile(list(fine.values()), 1 - diff.chord_top))
                 if fine and diff.chord == "strong" and diff.chord_top > 0 else math.inf)
    mel_at = {m[0]: (i, m) for i, m in enumerate(melody)}
    slots = sorted(set(mel_at) | {b for b, v in fills.items() if v >= fill_thr})
    if not slots:
        return [], summary
    pitches = np.array([p for _b0, _b1, p in melody], dtype=float) if melody else np.array([60.0])
    lo, hi = float(np.percentile(pitches, 10)), float(np.percentile(pitches, 90))
    last_hit = [-math.inf] * keys
    hold_until = [-math.inf] * keys
    out: list[ManiaNote] = []
    prev_col, prev_pitch, last_row, prev_row_cols = None, None, -math.inf, []
    bpb = tempo.beats_per_bar

    def ok(c: int, b: float) -> bool:
        return b - hold_until[c] >= diff.snap - 1e-9 and b - last_hit[c] >= diff.jack_gap - 1e-9

    def place(b: float, c: int, end: float | None, pitch: int, kind: str) -> None:
        out.append(ManiaNote(b, c, end, pitch, kind))
        last_hit[c] = b if end is None else end
        if end is not None:
            hold_until[c] = end
        summary.notes += 1
        summary.holds += end is not None
        summary.columns[c] += 1
        summary.chords += kind == "chord"
        summary.fills += kind == "fill"

    for b in slots:
        row: list[int] = []
        if b in mel_at:
            i, (b0, b1, pitch) = mel_at[b]
            nxt = melody[i + 1][0] if i + 1 < len(melody) else None
            end = b1 if nxt is None else min(b1, nxt - diff.snap)
            hold = end if end - b0 >= diff.hold_min - 1e-9 else None
            if prev_col is None:
                want = _column_for(pitch, lo, hi, keys)
            else:
                d = pitch - prev_pitch
                step = 0 if d == 0 else (1 if 0 < d < 5 else 2 if d >= 5 else -1 if -5 < d < 0 else -2)
                want = prev_col + step
                if not 0 <= want < keys:  # 끝에 닿음: 음역 안 위치로 다시 맞춘다
                    want = _column_for(pitch, lo, hi, keys)
            want = int(min(keys - 1, max(0, want)))
            # 롱노트가 잡고 있거나 짧은 같은 열 연타가 되면 가까운 열로 (가운데 쪽 먼저)
            order = sorted(range(keys), key=lambda c: (abs(c - want), abs(c - (keys - 1) / 2)))
            col = next((c for c in order if ok(c, b)), None)
            if col is None:
                summary.dropped += 1
                continue
            place(b, col, hold, pitch, "melody")
            row.append(col)
            prev_col, prev_pitch = col, pitch
            strong = (diff.chord in ("downbeat", "strong") and abs(b / bpb - round(b / bpb)) < 1e-9) or \
                     (diff.chord == "strong" and fine.get(b, 0.0) >= chord_thr)
            if strong:  # 동시치기: 멜로디와 먼 열 (최근에 안 친 열 먼저). 마디 첫 박은 chord_size개까지
                downbeat = abs(b / bpb - round(b / bpb)) < 1e-9
                for _k in range(diff.chord_size if downbeat else 1):
                    cands = [c for c in range(keys) if c not in row and ok(c, b)]
                    if not cands:
                        break
                    c = max(cands, key=lambda c: (min(abs(c - r) for r in row), -last_hit[c]))
                    place(b, c, None, -1, "chord")
                    row.append(c)
        else:
            if b - last_row < diff.fill_gap - 1e-9:
                continue
            cands = [c for c in range(keys) if ok(c, b) and c not in prev_row_cols]
            if not cands:
                continue
            c = min(cands, key=lambda c: (last_hit[c], -min((abs(c - p) for p in prev_row_cols), default=0)))
            place(b, c, None, -1, "fill")
            row.append(c)
            if fills.get(b, 0.0) >= jump_thr:  # 점프: 첫 노트와 먼 쪽, 최근에 안 친 열
                more = [x for x in cands if x != c and ok(x, b)]
                if more:
                    c2 = max(more, key=lambda x: (abs(x - c), -last_hit[x]))
                    place(b, c2, None, -1, "fill")
                    row.append(c2)
        last_row, prev_row_cols = b, row
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
    notes = sorted(notes, key=lambda n: (n.beat, n.column))  # 시각 순, 같은 시각이면 왼쪽 열부터
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


def build_osz(notes: list[Note], tempo, audio, meta: Meta, difficulties=DEFAULT_DIFFICULTIES,
              keys: int = KEYS, onsets=None) -> tuple[bytes, list[ChartSummary]]:
    audio_name = "audio.ogg"
    files: dict[str, bytes] = {audio_name: encode_ogg(audio)}
    summaries = []
    meta.title = meta.title or "Untitled"
    meta.artist = meta.artist or "Unknown"  # osu!는 빈 제목·아티스트를 싫어한다
    base = _safe(f"{meta.artist} - {meta.title} ({meta.creator})")
    for name in difficulties:
        diff = DIFFICULTIES[name]
        chart_notes, s = chart(notes, tempo, diff, keys, onsets)
        files[f"{base} [{name}].osu"] = osu_text(chart_notes, tempo, diff, meta, audio_name, keys).encode("utf-8")
        summaries.append(s)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, data in files.items():
            z.writestr(fname, data)
    return buf.getvalue(), summaries


def write_osz(path: str | os.PathLike, notes: list[Note], tempo, audio, meta: Meta,
              difficulties=DEFAULT_DIFFICULTIES, keys: int = KEYS, onsets=None) -> list[ChartSummary]:
    from amp.core.fileio import atomic_write_bytes

    data, summaries = build_osz(notes, tempo, audio, meta, difficulties, keys, onsets)
    atomic_write_bytes(path, data)
    return summaries
