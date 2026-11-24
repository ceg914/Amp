"""베이스 TAB (v0.6.3). Qt 비의존.

    htdemucs 베이스 stem ─(pYIN, 낮은 음역)→ 피치 곡선 ─(음표 검출 + 뜯는 시점에서 나눔)→ 베이스 음표
        ─(BPM·첫 박으로 박자 격자에 맞춤)→ 박 단위 음표 ─(줄·프렛 배치)→ TAB (GP5·텍스트)

측정 (조각 나비, Guitar Pro 정답 575음, 시작 ±100ms·음높이 일치):
- 보컬용 음표 검출 그대로: F1 0.73. 같은 음을 8분음표로 반복하는 부분 192음이 한 음으로 합쳐졌다.
- 20ms 음량이 직전 30ms 최저보다 4dB 넘게 다시 오르는 지점(뜯음)에서 나누면: F1 0.89 (정밀도 0.90, 재현율 0.88).
- 줄·프렛 배치: 개방현 벌점 2.0이면 정답 TAB과 같은 줄·프렛 67.8% (벌점이 없으면 29.9%, 개방현을 172번 씀).
나누는 기준과 배치 가중치는 이 한 곡에서 골랐다. 다른 곡에서 같은 정확도가 나오는지는 확인하지 않았다.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from amp.core.ids import new_id
from amp.core.model import Note
from amp.core.pitch import midi_to_hz, note_name

BASS_NOTES_VERSION = "1"
# 조율: 낮은 줄부터 개방현 MIDI
TUNINGS = {
    "4_standard": ("4현 표준 (E A D G)", (28, 33, 38, 43)),
    "5_standard": ("5현 표준 (B E A D G)", (23, 28, 33, 38, 43)),
    "4_drop_d": ("4현 드롭 D (D A D G)", (26, 33, 38, 43)),
    "4_half_down": ("4현 반음 내림 (Eb Ab Db Gb)", (27, 32, 37, 42)),
    "6_standard": ("6현 (B E A D G C)", (23, 28, 33, 38, 43, 48)),
}
OPEN_PREFS = {"few": ("개방현 적게", 2.0), "normal": ("보통", 0.5), "many": ("개방현 많이", 0.0)}
GRIDS = {0.25: "16분음표", 0.5: "8분음표"}
SPLIT_RISE_DB = 4.0
SPLIT_FLOOR_DB = -50.0
SPLIT_LOOKBACK = 3  # 프레임 (30ms)
SPLIT_MIN_GAP_S = 0.08
SPLIT_MIN_PIECE_S = 0.06
SNAP_MAX_S = 0.15  # 음 시작을 앞쪽 뜯는 시점으로 당기는 최대 거리
HAND_SPAN = 4  # 손가락 넷이 닿는 프렛 수


class BassError(ValueError):
    pass


@dataclass(frozen=True)
class BassOptions:
    tuning: str = "4_standard"
    max_fret: int = 20
    open_strings: str = "few"
    grid: float = 0.25
    split_rise_db: float = SPLIT_RISE_DB

    def __post_init__(self) -> None:
        if self.tuning not in TUNINGS:
            raise BassError(f"알 수 없는 조율: {self.tuning}")
        if self.open_strings not in OPEN_PREFS:
            raise BassError(f"알 수 없는 개방현 설정: {self.open_strings}")
        if not 5 <= int(self.max_fret) <= 24:
            raise BassError("최대 프렛은 5~24 사이여야 합니다.")
        if float(self.grid) not in GRIDS:
            raise BassError("박자 격자는 16분음표(0.25박) 또는 8분음표(0.5박)입니다.")

    @property
    def open_midis(self) -> tuple[int, ...]:
        return TUNINGS[self.tuning][1]

    def midi_range(self) -> tuple[int, int]:
        """연주할 수 있는 음역 (가장 낮은 개방현 ~ 가장 높은 줄의 최대 프렛)."""
        return min(self.open_midis), max(self.open_midis) + int(self.max_fret)

    def hz_range(self) -> tuple[float, float]:
        """피치 추적 범위: 연주 음역보다 1반음 넓게, 위는 400Hz까지 (베이스 배음을 기음으로 잡지 않게)."""
        lo, hi = self.midi_range()
        return float(midi_to_hz(lo - 1)), float(min(400.0, midi_to_hz(hi + 1)))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BassOptions":
        if not d:
            return cls()
        names = {f.name for f in dataclasses.fields(cls)}
        try:
            return cls(**{k: v for k, v in d.items() if k in names})
        except (BassError, TypeError, ValueError):
            return cls()


@dataclass(frozen=True)
class BassState:
    """프로젝트의 베이스 TAB 상태 (analysis/bass.json). 자동 분석 원본과 사용자 작업본을 나눠 둔다.

    analysis: 베이스 자동 분석 결과 (AnalysisResult: 음표·피치 곡선·모델 정보). 분석할 때만 바뀐다.
    notes: 사용자 작업본 음표. 다시 분석해도 고친 음표·추가한 음표는 지킨다 ('수정 유지').
    박 위치·템포는 보컬 프로젝트의 템포를 함께 쓴다."""
    options: BassOptions = field(default_factory=BassOptions)
    analysis: object = None  # AnalysisResult | None
    notes: tuple = ()
    source: dict = field(default_factory=dict)  # 베이스를 어디서 읽었는지 ({"kind", "store_key", "path"} 등)


# --- 음표 ------------------------------------------------------------------------

def pluck_onsets(rms_fine: np.ndarray, frame_rate: float, rise_db: float = SPLIT_RISE_DB) -> list[float]:
    """뜯는 시점(초): 20ms 음량이 직전 30ms의 최저보다 rise_db 넘게 처음 오른 프레임. 그 뒤 SPLIT_MIN_GAP_S 동안은 새로 잡지 않는다.

    베이스는 줄을 뜯을 때마다 소리가 커졌다가 줄어든다. 같은 음을 반복해도 음량이 다시 오르므로 음높이 변화가 없어도 나눌 수 있다.
    봉우리(국소 최대)만 잡던 규칙은 여러 프레임에 걸쳐 천천히 오르는 퉁김을 놓쳤다. 조각 나비 베이스(기준 GP5 575음)에서
    봉우리 규칙 F1 0.889 → 처음 넘는 프레임 0.911 (±100ms), 같은 음 반복이 합쳐진 경우 39 → 15. 한 곡에서 고른 규칙이다."""
    if rms_fine is None or len(rms_fine) < SPLIT_LOOKBACK + 3:
        return []
    db = 20 * np.log10(np.asarray(rms_fine, dtype=np.float64) + 1e-9)
    k = SPLIT_LOOKBACK
    out: list[float] = []
    min_gap = SPLIT_MIN_GAP_S
    for i in range(k, len(db)):
        if db[i] <= SPLIT_FLOOR_DB:
            continue
        if db[i] - db[i - k:i].min() >= rise_db and (not out or i / frame_rate - out[-1] > min_gap):
            out.append(i / frame_rate)
    return out


def split_at_onsets(notes: list[Note], onsets: list[float], min_piece_s: float = SPLIT_MIN_PIECE_S) -> list[Note]:
    out: list[Note] = []
    arr = np.asarray(onsets, dtype=np.float64)
    for n in sorted(notes, key=lambda n: n.start):
        lo, hi = np.searchsorted(arr, n.start + min_piece_s), np.searchsorted(arr, n.end - min_piece_s, side="right")
        cuts = [float(t) for t in arr[lo:hi]]
        if not cuts:
            out.append(n)
            continue
        bounds = [n.start, *cuts, n.end]
        for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
            out.append(n.with_(id=n.id if k == 0 else new_id(), start=a, end=b))
    return out


def snap_starts_to_onsets(notes: list[Note], onsets: list[float], max_shift_s: float = SNAP_MAX_S,
                          min_piece_s: float = SPLIT_MIN_PIECE_S) -> list[Note]:
    """음 시작을 바로 앞(max_shift_s 이내)의 뜯는 시점으로 당긴다. 앞 음이 그 시점을 넘으면 거기서 자른다.

    pYIN은 크게 내려가는 도약(예: A2 → E1)에서 새 높이를 늦게 잡는다. 베이스는 뜯는 순간 높이가 바뀌므로 음량이 오른
    시점이 실제 시작에 가깝다. 앞 음이 너무 짧아지면(min_piece_s 미만) 당기지 않는다."""
    arr = np.asarray(onsets, dtype=np.float64)
    out = sorted(notes, key=lambda n: n.start)
    for i, n in enumerate(out):
        k = np.searchsorted(arr, n.start + 1e-9) - 1
        if k < 0 or n.start - arr[k] > max_shift_s or n.start - arr[k] < 0.005:
            continue
        t = float(arr[k])
        if i > 0:
            p = out[i - 1]
            if p.end > t:
                if t - p.start < min_piece_s:
                    continue
                out[i - 1] = p.with_(end=t)
        out[i] = n.with_(start=t)
    return out


def detect_bass_notes(frames, options: BassOptions) -> tuple[list[Note], object, np.ndarray]:
    """피치 결과(PitchFrames) → (베이스 음표, 피치 곡선, 재발음 지표). 피치 곡선은 바꾸지 않는다."""
    from amp.engine.notes import detect_notes
    from amp.engine.pitch_post import articulation_envelope, build_curve
    from amp.engine.settings import AnalysisSettings

    lo, hi = options.midi_range()
    curve = build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate, frames.start_time,
                        rms_fine=frames.rms_fine, fix_octaves=False)  # 베이스의 옥타브 도약(E1→E2)을 지키려고
    art = articulation_envelope(frames.onset, frames.rms, frames.frame_rate, rms_fine=frames.rms_fine,
                                voiced_prob=frames.voiced_prob, f0_hz=frames.f0_hz)
    st = AnalysisSettings(sensitivity="standard", min_note_ms=80.0, pitch_change_semitones=1.0,
                          pitch_low=max(0, lo - 1), pitch_high=min(127, hi + 1))
    params = dataclasses.replace(st.note_params(), merge_approach=False)  # 베이스에는 끌어올림 흡수가 맞지 않는다
    notes = detect_notes(curve, art, params)
    onsets = [t + frames.start_time for t in pluck_onsets(frames.rms_fine, frames.frame_rate, options.split_rise_db)]
    return snap_starts_to_onsets(split_at_onsets(notes, onsets), onsets), curve, art


# --- 줄·프렛 ------------------------------------------------------------------------

def positions(pitch: int, options: BassOptions) -> list[tuple[int, int]]:
    """(줄 번호, 프렛). 줄 번호 0 = 가장 낮은 줄."""
    return [(s, pitch - o) for s, o in enumerate(options.open_midis) if 0 <= pitch - o <= options.max_fret]


def assign_fingering(pitches: list[int], options: BassOptions, w_high: float = 0.02, w_string: float = 0.3,
                     beam: int = 60) -> list[tuple[int, int] | None]:
    """손 위치 이동이 적은 줄·프렛 (동적 계획). 칠 수 없는 음(음역 밖)은 None.

    비용 = 손 위치(검지 프렛) 이동 + 개방현 벌점 + 높은 프렛 × w_high + 줄 건너뛰기 × w_string.
    손이 닿는 범위(검지 ~ 검지+3) 안의 프렛은 손을 옮기지 않는다. 개방현은 손 위치를 바꾸지 않는다."""
    w_open = OPEN_PREFS[options.open_strings][1]
    out: list[tuple[int, int] | None] = [None] * len(pitches)
    # 연주할 수 있는 음끼리 이어서 계산한다 (음역 밖 음은 건너뜀)
    idx = [i for i, p in enumerate(pitches) if positions(p, options)]
    if not idx:
        return out
    states: dict = {}
    for o in positions(pitches[idx[0]], options):
        hand = max(1, o[1]) if o[1] else 1
        states[(o, hand)] = (w_high * o[1] + (w_open if o[1] == 0 else 0.0), [o])
    for i in idx[1:]:
        nxt: dict = {}
        for o in positions(pitches[i], options):
            for (q, hand), (c, path) in states.items():
                if o[1] == 0:
                    nh, move = hand, w_open
                elif hand <= o[1] <= hand + HAND_SPAN - 1:
                    nh, move = hand, 0.0
                else:
                    nh = o[1] if o[1] < hand else o[1] - HAND_SPAN + 1
                    move = float(abs(nh - hand))
                cost = c + move + w_high * o[1] + w_string * abs(o[0] - q[0])
                key = (o, nh)
                if key not in nxt or cost < nxt[key][0]:
                    nxt[key] = (cost, path + [o])
        states = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:beam])
    path = min(states.values(), key=lambda v: v[0])[1]
    for i, o in zip(idx, path):
        out[i] = o
    return out


# --- 박자 -------------------------------------------------------------------------

@dataclass(frozen=True)
class TabNote:
    start_beat: float  # 첫 박(1마디 1박) 기준
    beats: float  # 길이 (박)
    pitch: int
    string: int | None  # 0 = 가장 낮은 줄. None = 칠 수 없음 (음역 밖)
    fret: int | None
    note_id: str = ""


@dataclass
class Tab:
    options: BassOptions
    bpm: float
    beats_per_bar: int
    notes: list[TabNote] = field(default_factory=list)
    pickup_bars: int = 0  # 첫 박 앞에 음이 있어 앞에 붙인 마디 수
    unplayable: int = 0

    @property
    def bars(self) -> int:
        if not self.notes:
            return 0
        end = max(n.start_beat + n.beats for n in self.notes)
        return max(1, int(math.ceil(end / self.beats_per_bar - 1e-9)))


def quantize(notes: list[Note], tempo, options: BassOptions) -> Tab:
    """음표(초) → 박 격자에 맞춘 TAB. 겹치면 앞 음을 다음 음 시작에서 자른다(한 번에 한 음)."""
    grid = float(options.grid)
    bpb = int(tempo.beats_per_bar)
    q: list[tuple[float, float, Note]] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        s = round(tempo.beat_at(n.start) / grid) * grid
        e = round(tempo.beat_at(n.end) / grid) * grid
        if e <= s:
            e = s + grid
        q.append((s, e, n))
    clean: list[tuple[float, float, Note]] = []
    for s, e, n in q:
        if clean and abs(clean[-1][0] - s) < 1e-9:
            continue  # 같은 격자에 둘: 앞 것만 (베이스는 한 번에 한 음)
        if clean and clean[-1][1] > s:
            ps, _pe, pn = clean[-1]
            clean[-1] = (ps, s, pn)
        clean.append((s, e, n))
    first = min((s for s, _e, _n in clean), default=0.0)
    pickup = int(math.ceil(-first / bpb)) if first < 0 else 0
    shift = pickup * bpb
    fingers = assign_fingering([n.pitch for _s, _e, n in clean], options)
    out = []
    for (s, e, n), f in zip(clean, fingers):
        out.append(TabNote(round(s + shift, 6), round(e - s, 6), n.pitch, f[0] if f else None, f[1] if f else None, n.id))
    return Tab(options, float(tempo.initial_bpm), bpb, out, pickup, sum(1 for f in fingers if f is None))


# --- 텍스트 TAB ---------------------------------------------------------------------

def string_names(options: BassOptions) -> list[str]:
    """줄 이름 (낮은 줄부터). 예: E A D G."""
    return [note_name(m).rstrip("0123456789-") for m in options.open_midis]


def text_tab(tab: Tab, bars_per_line: int = 4, title: str = "") -> str:
    """ASCII TAB. 위 줄이 가장 높은 줄. 한 칸 = 박자 격자 하나 (프렛 두 자리를 위해 칸 너비 3)."""
    opts = tab.options
    grid = float(opts.grid)
    per_bar = max(1, int(round(tab.beats_per_bar / grid)))
    names = string_names(opts)
    width = max(len(n) for n in names)
    head = [title] if title else []
    head.append(f"{TUNINGS[opts.tuning][0]} · ♩={tab.bpm:.1f} · {tab.beats_per_bar}/4 · 한 칸 = {GRIDS.get(grid, f'{grid}박')}")
    if tab.pickup_bars:
        head.append(f"(첫 박보다 앞에 음이 있어 앞에 {tab.pickup_bars}마디를 붙였습니다)")
    if tab.unplayable:
        head.append(f"(이 조율·최대 프렛으로 칠 수 없는 음 {tab.unplayable}개는 빠졌습니다)")
    cells: dict[tuple[int, int], str] = {}
    for n in tab.notes:
        if n.string is None:
            continue
        slot = int(round(n.start_beat / grid))
        cells[(slot, n.string)] = str(n.fret)
    lines = head + [""]
    bars = tab.bars
    for b0 in range(0, bars, bars_per_line):
        b1 = min(bars, b0 + bars_per_line)
        lines.append(f"{b0 + 1}")
        for s in reversed(range(len(opts.open_midis))):
            row = names[s].ljust(width) + "|"
            for b in range(b0, b1):
                for k in range(per_bar):
                    cell = cells.get((b * per_bar + k, s), "")
                    row += (cell + "-" * 3)[:3] if cell else "---"
                row += "|"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def summary(tab: Tab) -> dict:
    played = [n for n in tab.notes if n.string is not None]
    frets = [n.fret for n in played]
    return {"notes": len(tab.notes), "bars": tab.bars, "unplayable": tab.unplayable,
            "open_strings": sum(1 for f in frets if f == 0),
            "fret_range": (min(frets), max(frets)) if frets else None}
