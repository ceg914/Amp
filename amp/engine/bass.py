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
GRIDS = {0.25: "16분음표", 0.5: "8분음표", 0.125: "32분음표"}
TPB = 24  # 박당 눈금: 8·16·32분음표와 8·16분 셋잇단을 모두 정수 눈금으로 (v0.7.0)
GRID_TICKS = {0.5: 12, 0.25: 6, 0.125: 3}
TUPLETS = {"off": "쓰지 않음", "auto": "박마다 자동 (셋잇단이 더 맞을 때만)"}
TRIPLET_GAIN = 0.5  # 셋잇단 오차가 곧은 격자 오차의 이 비율보다 작을 때만 그 박을 셋잇단으로
DUPLICATE_BEATS = 1.0 / 16  # 앞 음 시작과 64분음표(1/16박)보다 가까우면 같은 소리를 두 번 잡은 것으로 본다
PICKUPS = {"bar": "온마디를 붙임", "partial": "못갖춘마디 (필요한 박만)"}
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
    tuplets: str = "off"  # v0.7.0 셋잇단: off / auto (박마다 곧은 격자와 비교)
    meter: tuple = ()  # v0.7.0 박자표 변화: ((마디 번호, 박 수), ...). 1마디 = 템포 첫 박이 든 마디. 비우면 곡 박자
    pickup: str = "bar"  # v0.7.0 첫 박 앞 음: bar(온마디를 붙임) / partial(못갖춘마디)
    riff: bool = False  # v0.7.0 반복 리프는 같은 운지로 (실험)

    def __post_init__(self) -> None:
        if self.tuning not in TUNINGS:
            raise BassError(f"알 수 없는 조율: {self.tuning}")
        if self.open_strings not in OPEN_PREFS:
            raise BassError(f"알 수 없는 개방현 설정: {self.open_strings}")
        if not 5 <= int(self.max_fret) <= 24:
            raise BassError("최대 프렛은 5~24 사이여야 합니다.")
        if float(self.grid) not in GRIDS:
            raise BassError("박자 격자는 8분음표(0.5박)·16분음표(0.25박)·32분음표(0.125박)입니다. 셋잇단은 tuplets로 켭니다.")
        if self.tuplets not in TUPLETS:
            raise BassError(f"알 수 없는 셋잇단 설정: {self.tuplets}")
        if self.pickup not in PICKUPS:
            raise BassError(f"알 수 없는 못갖춘마디 설정: {self.pickup}")
        try:
            meter = tuple(sorted({int(b): int(n) for b, n in (self.meter or ())}.items()))
        except (TypeError, ValueError) as e:
            raise BassError(f"박자표 변화 형식이 잘못되었습니다: {self.meter}") from e
        if any(b < 1 or not 1 <= n <= 16 for b, n in meter):
            raise BassError("박자표 변화는 1마디부터, 마디당 1~16박입니다.")
        object.__setattr__(self, "meter", meter)
        object.__setattr__(self, "riff", bool(self.riff))

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
    fingering: dict = field(default_factory=dict)  # v0.7.0 사용자가 정한 줄 {음 id: 줄 번호(0 = 가장 낮은 줄)}


def parse_meter(text: str) -> tuple:
    """'9:3, 17:4' → ((9, 3), (17, 4)) (마디 번호: 박 수). 빈 글은 (). 형식이 틀리면 BassError."""
    out = []
    for part in str(text or "").replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            bar, beats = (int(x) for x in part.replace("/", ":").split(":")[:2])
        except ValueError as e:
            raise BassError(f"박자표 변화는 '마디:박 수'로 적습니다 (예: 9:3, 17:4). 읽지 못함: {part}") from e
        out.append((bar, beats))
    return BassOptions(meter=tuple(out)).meter


def format_meter(meter: tuple) -> str:
    return ", ".join(f"{b}:{n}" for b, n in meter)


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


RIFF_LEN = 4  # 같은 음높이 4개가 다시 나오면 같은 리프로 본다
RIFF_BONUS = 0.6  # 리프 첫 운지와 같은 자리면 비용에서 뺀다 (실험, BassOptions.riff)


def _fingering_dp(pitches: list[int], options: BassOptions, w_high: float, w_string: float, beam: int,
                  fixed: dict | None = None, prefer: dict | None = None) -> list[tuple[int, int] | None]:
    w_open = OPEN_PREFS[options.open_strings][1]
    out: list[tuple[int, int] | None] = [None] * len(pitches)
    fixed = fixed or {}
    prefer = prefer or {}

    def cands(i):
        ps = positions(pitches[i], options)
        if i in fixed:
            on = [o for o in ps if o[0] == fixed[i]]
            return on or ps  # 그 줄로 칠 수 없으면 (음역 밖) 지정을 무시한다
        return ps

    def bonus(i, o):
        return -RIFF_BONUS if prefer.get(i) == o else 0.0

    # 연주할 수 있는 음끼리 이어서 계산한다 (음역 밖 음은 건너뜀)
    idx = [i for i, p in enumerate(pitches) if positions(p, options)]
    if not idx:
        return out
    states: dict = {}
    for o in cands(idx[0]):
        hand = max(1, o[1]) if o[1] else 1
        states[(o, hand)] = (w_high * o[1] + (w_open if o[1] == 0 else 0.0) + bonus(idx[0], o), [o])
    for i in idx[1:]:
        nxt: dict = {}
        for o in cands(i):
            for (q, hand), (c, path) in states.items():
                if o[1] == 0:
                    nh, move = hand, w_open
                elif hand <= o[1] <= hand + HAND_SPAN - 1:
                    nh, move = hand, 0.0
                else:
                    nh = o[1] if o[1] < hand else o[1] - HAND_SPAN + 1
                    move = float(abs(nh - hand))
                cost = c + move + w_high * o[1] + w_string * abs(o[0] - q[0]) + bonus(i, o)
                key = (o, nh)
                if key not in nxt or cost < nxt[key][0]:
                    nxt[key] = (cost, path + [o])
        states = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:beam])
    path = min(states.values(), key=lambda v: v[0])[1]
    for i, o in zip(idx, path):
        out[i] = o
    return out


def assign_fingering(pitches: list[int], options: BassOptions, w_high: float = 0.02, w_string: float = 0.3,
                     beam: int = 60, fixed: dict | None = None) -> list[tuple[int, int] | None]:
    """손 위치 이동이 적은 줄·프렛 (동적 계획). 칠 수 없는 음(음역 밖)은 None.

    비용 = 손 위치(검지 프렛) 이동 + 개방현 벌점 + 높은 프렛 × w_high + 줄 건너뛰기 × w_string.
    손이 닿는 범위(검지 ~ 검지+3) 안의 프렛은 손을 옮기지 않는다. 개방현은 손 위치를 바꾸지 않는다.
    fixed: {음 순번: 줄 번호} 사용자가 정한 줄 (v0.7.0). 그 줄로 칠 수 없으면 지정을 무시한다.
    options.riff (실험): 같은 음높이 4개 묶음이 다시 나오면 처음 나온 곳의 운지를 따르도록 비용을 줄여 한 번 더 푼다."""
    path = _fingering_dp(pitches, options, w_high, w_string, beam, fixed)
    if not options.riff or len(pitches) < 2 * RIFF_LEN:
        return path
    first: dict = {}
    prefer: dict = {}
    for i in range(RIFF_LEN - 1, len(pitches)):
        key = tuple(pitches[i - RIFF_LEN + 1:i + 1])
        if key not in first:
            first[key] = i
            continue
        j = first[key]
        for k in range(RIFF_LEN):
            if path[j - k] is not None:
                prefer.setdefault(i - k, path[j - k])
    return _fingering_dp(pitches, options, w_high, w_string, beam, fixed, prefer) if prefer else path


# --- 박자 -------------------------------------------------------------------------

@dataclass(frozen=True)
class TabNote:
    start_beat: float  # 첫 박(1마디 1박) 기준
    beats: float  # 길이 (박)
    pitch: int
    string: int | None  # 0 = 가장 낮은 줄. None = 칠 수 없음 (음역 밖)
    fret: int | None
    note_id: str = ""
    # v0.7.0: 양자화하기 전 실제 분석 시간 (초, TAB 박) — 양자화 결과와 따로 보관
    start_s: float | None = None
    end_s: float | None = None
    raw_start_beat: float | None = None
    flags: tuple = ()  # moved(앞 음과 같은 칸이라 더 잘게 옮김) · triplet(셋잇단 박) · fixed(사용자 지정 줄) · fixed_ignored


@dataclass
class Tab:
    options: BassOptions
    bpm: float
    beats_per_bar: int
    notes: list[TabNote] = field(default_factory=list)
    pickup_bars: int = 0  # 첫 박 앞에 음이 있어 앞에 붙인 마디 수
    unplayable: int = 0
    # v0.7.0
    bar_lengths: list = field(default_factory=list)  # 마디마다 박 수 (비면 beats_per_bar로 계산)
    tempo_points: list = field(default_factory=list)  # [(TAB 박, BPM)] 템포가 바뀌는 곳 (첫 값 포함)
    dropped: list = field(default_factory=list)  # [(음 id, 이유)] 같은 소리를 두 번 잡은 것으로 보고 뺀 음
    moved: int = 0  # 같은 칸에 들어와 더 잘게 옮긴 음 수
    triplet_beats: list = field(default_factory=list)  # 셋잇단으로 맞춘 박 (TAB 박 번호)
    pickup_beats: int = 0  # 못갖춘마디 박 수 (options.pickup == "partial"일 때)

    def _lengths(self) -> list[int]:
        if self.bar_lengths:
            return list(self.bar_lengths)
        if not self.notes:
            return []
        end = max(n.start_beat + n.beats for n in self.notes)
        return [self.beats_per_bar] * max(1, int(math.ceil(end / self.beats_per_bar - 1e-9)))

    @property
    def bars(self) -> int:
        if not self.notes:
            return 0
        return len(self._lengths())

    def bar_starts(self) -> list[float]:
        out, t = [], 0.0
        for n in self._lengths():
            out.append(t)
            t += n
        return out


def _tick(x: float) -> int:
    return int(round(x * TPB))


def _bar_lengths(first_bars: list[int], bpb: int, meter: tuple, end_beats: float, start_beats: float) -> list[int]:
    """첫 마디들(못갖춘마디·붙인 마디) + 1마디부터 박자표 변화를 따른 마디들로 end_beats까지 덮는다."""
    out = list(first_bars)
    t = start_beats
    changes = dict(meter)
    cur = bpb
    k = 1
    while t < end_beats - 1e-9 or k == 1:
        cur = changes.get(k, cur)
        out.append(cur)
        t += cur
        k += 1
    return out


def quantize(notes: list[Note], tempo, options: BassOptions, fingering: dict | None = None) -> Tab:
    """음표(초) → 박 격자에 맞춘 TAB. 겹치면 앞 음을 다음 음 시작에서 자른다(한 번에 한 음).

    v0.7.0
    - 템포 맵(구간별 템포)대로 박을 센다. 박자표 변화(options.meter)·못갖춘마디(options.pickup)
    - 셋잇단(options.tuplets="auto"): 박마다 곧은 격자와 셋잇단 중 오차가 훨씬 작은 쪽
    - 같은 칸에 들어온 음: 앞 음과 64분음표보다 가까우면 같은 소리를 두 번 잡은 것으로 보고 빼되 tab.dropped에 남긴다.
      그보다 떨어져 있으면 지우지 않고 한 단계 잘게(16분 → 32분) 옮긴다 (flags에 moved)
    - 실제 분석 시간(start_s·end_s·raw_start_beat)을 양자화 결과와 함께 둔다
    - fingering: {음 id: 줄 번호} 사용자가 정한 줄"""
    base = GRID_TICKS[float(options.grid)]
    trip_step = 8 if base >= 6 else 4
    bpb = int(tempo.beats_per_bar)
    raw = sorted(((tempo.beat_at(n.start), tempo.beat_at(n.end), n) for n in notes), key=lambda x: (x[0], x[2].pitch))
    triplets: set[int] = set()
    if options.tuplets == "auto":
        by_beat: dict[int, list[float]] = {}
        for rs, _re, _n in raw:
            by_beat.setdefault(math.floor(rs + 1e-9), []).append(rs)

        def err(xs, st):
            return sum(((x * TPB - round(x * TPB / st) * st) / TPB) ** 2 for x in xs)

        for bt, xs in by_beat.items():
            if len(xs) >= 2 and err(xs, trip_step) < TRIPLET_GAIN * err(xs, base):
                triplets.add(bt)

    def step_at(x: float) -> int:
        return trip_step if math.floor(x + 1e-9) in triplets else base

    def snap(x: float) -> int:
        st = step_at(x)
        return int(round(x * TPB / st)) * st

    placed: list[list] = []  # [시작 눈금, 끝 눈금, 음, 원래 시작 박, 표시]
    dropped: list[tuple[str, str]] = []
    moved = 0
    for rs, re_, n in raw:
        st = step_at(rs)
        s0 = snap(rs)
        e0 = snap(re_)
        flags: list[str] = ["triplet"] if math.floor(rs + 1e-9) in triplets else []
        if placed and s0 <= placed[-1][0]:
            prev = placed[-1]
            if abs(rs - prev[3]) < DUPLICATE_BEATS:
                dropped.append((n.id, f"앞 음({prev[2].id})과 {abs(rs - prev[3]) * tempo.seconds_per_beat_at(n.start) * 1000:.0f}ms "
                                      "차이로 같은 칸 (같은 소리를 두 번 잡은 것으로 봄)"))
                continue
            fine = st // 2 if st % 2 == 0 and st // 2 >= 3 else st
            s0 = prev[0] + fine
            flags.append("moved")
            moved += 1
        if e0 <= s0:
            e0 = s0 + (st // 2 if "moved" in flags and st // 2 >= 3 else st)
        if placed and placed[-1][1] > s0:
            placed[-1][1] = s0
        placed.append([s0, e0, n, rs, flags])
    first = min((p[0] for p in placed), default=0)
    first_bars: list[int] = []
    shift = 0
    pickup = 0
    if first < 0:
        if options.pickup == "partial":
            need = int(math.ceil(-first / TPB - 1e-9))
            first_bars = [need]
            shift = need * TPB
        else:
            pickup = int(math.ceil(-first / (bpb * TPB) - 1e-9))
            first_bars = [bpb] * pickup
            shift = pickup * bpb * TPB
    ids = [p[2].id for p in placed]
    fixed = {i: int(fingering[nid]) for i, nid in enumerate(ids) if fingering and nid in fingering}
    fingers = assign_fingering([p[2].pitch for p in placed], options, fixed=fixed)
    out = []
    for i, ((s0, e0, n, rs, flags), f) in enumerate(zip(placed, fingers)):
        fl = list(flags)
        if i in fixed:
            fl.append("fixed" if f is not None and f[0] == fixed[i] else "fixed_ignored")
        out.append(TabNote(round((s0 + shift) / TPB, 6), round((e0 - s0) / TPB, 6), n.pitch, f[0] if f else None,
                           f[1] if f else None, n.id, round(n.start, 4), round(n.end, 4), round(rs + shift / TPB, 4),
                           tuple(fl)))
    end = max((n.start_beat + n.beats for n in out), default=0.0)
    lengths = _bar_lengths(first_bars, bpb, options.meter, end, float(sum(first_bars))) if out else []
    tpts: list[tuple[float, float]] = [(0.0, float(tempo.initial_bpm))]
    for pt in tempo.points[1:]:  # 첫 템포 지점 앞(붙인 마디)도 첫 템포로 센다
        b = round(tempo.beat_at(pt.time) + shift / TPB, 6)
        if b > end + 1e-9 or float(pt.bpm) == tpts[-1][1]:
            continue
        if b <= tpts[-1][0] + 1e-9:
            tpts[-1] = (tpts[-1][0], float(pt.bpm))
        else:
            tpts.append((b, float(pt.bpm)))
    trip_tab = sorted(bt + shift // TPB for bt in triplets)
    return Tab(options, float(tempo.initial_bpm), bpb, out, pickup, sum(1 for f in fingers if f is None), lengths,
               tpts, dropped, moved, trip_tab, first_bars[0] if options.pickup == "partial" and first_bars else 0)


# --- 텍스트 TAB ---------------------------------------------------------------------

def string_names(options: BassOptions) -> list[str]:
    """줄 이름 (낮은 줄부터). 예: E A D G."""
    return [note_name(m).rstrip("0123456789-") for m in options.open_midis]


CELL_LABELS = {1: "4분음표", 2: "8분음표", 4: "16분음표", 8: "32분음표", 3: "8분 셋잇단", 6: "16분 셋잇단",
               12: "1/12박 (16분·셋잇단 공통)", 24: "1/24박"}


def _cells_per_beat(ticks: list[int], base: int) -> int:
    """이 눈금들을 모두 칸 경계에 놓을 수 있는 가장 큰 칸 (박당 칸 수). 기본 격자보다 굵게는 하지 않는다."""
    g = base
    for t in ticks:
        g = math.gcd(g, t % TPB)
    return TPB // max(1, g)


def text_tab(tab: Tab, bars_per_line: int = 4, title: str = "") -> str:
    """ASCII TAB. 위 줄이 가장 높은 줄. 칸 너비 3 (프렛 두 자리).

    마디마다 칸 크기가 다를 수 있다: 격자(기본 16분음표)로 놓을 수 없는 음(32분으로 옮긴 음·셋잇단)이 있는 마디만 칸을 잘게 한다."""
    opts = tab.options
    base = GRID_TICKS.get(float(opts.grid), 6)
    names = string_names(opts)
    width = max(len(n) for n in names)
    lengths = tab._lengths()  # noqa: SLF001
    starts = tab.bar_starts()
    meters = sorted({n for n in lengths})
    head = [title] if title else []
    meter_txt = f"{tab.beats_per_bar}/4" if meters in ([], [tab.beats_per_bar]) else "박자 " + "·".join(f"{m}/4" for m in meters)
    tempo_txt = f"♩={tab.bpm:.1f}"
    if len(tab.tempo_points) > 1:
        tempo_txt += f" (템포 바뀜 {len(tab.tempo_points) - 1}곳)"
    head.append(f"{TUNINGS[opts.tuning][0]} · {tempo_txt} · {meter_txt} · 한 칸 = {GRIDS.get(float(opts.grid), f'{opts.grid}박')}")
    if tab.pickup_bars:
        head.append(f"(첫 박보다 앞에 음이 있어 앞에 {tab.pickup_bars}마디를 붙였습니다)")
    elif tab.pickup_beats:
        head.append(f"(첫 마디는 {tab.pickup_beats}박 못갖춘마디입니다)")
    if tab.unplayable:
        head.append(f"(이 조율·최대 프렛으로 칠 수 없는 음 {tab.unplayable}개는 빠졌습니다)")
    if tab.moved:
        head.append(f"(앞 음과 같은 칸에 들어온 음 {tab.moved}개는 한 단계 잘게 옮겼습니다. 그 마디는 칸이 더 촘촘합니다)")
    if tab.dropped:
        head.append(f"(같은 소리를 두 번 잡은 것으로 보고 뺀 음 {len(tab.dropped)}개)")
    for b_, bpm in tab.tempo_points[1:]:
        k = max(0, next((i for i, s0 in enumerate(starts) if s0 > b_ + 1e-9), len(starts)) - 1)
        head.append(f"  {k + 1}마디 {b_ - starts[k] + 1:.2f}박부터 ♩={bpm:.1f}")
    per_bar_notes: dict[int, list[TabNote]] = {}
    for n in tab.notes:
        if n.string is None:
            continue
        k = max(0, next((i for i, s0 in enumerate(starts) if s0 > n.start_beat + 1e-9), len(starts)) - 1)
        per_bar_notes.setdefault(k, []).append(n)
    lines = head + [""]
    bars = len(lengths)
    for b0 in range(0, bars, bars_per_line):
        b1 = min(bars, b0 + bars_per_line)
        label = f"{b0 + 1}"
        fine = [k for k in range(b0, b1) if _cells_per_beat([_tick(n.start_beat - starts[k]) for n in per_bar_notes.get(k, [])],
                                                              base) != TPB // base]
        if fine:
            label += "  (" + ", ".join(f"{k + 1}마디 한 칸 = {CELL_LABELS.get(_cells_per_beat([_tick(n.start_beat - starts[k]) for n in per_bar_notes.get(k, [])], base), '')}"
                                       for k in fine) + ")"
        lines.append(label)
        rows = {s_: names[s_].ljust(width) + "|" for s_ in range(len(opts.open_midis))}
        for k in range(b0, b1):
            ns = per_bar_notes.get(k, [])
            cpb = _cells_per_beat([_tick(n.start_beat - starts[k]) for n in ns], base)
            cells = {(int(round((n.start_beat - starts[k]) * cpb)), n.string): str(n.fret) for n in ns}
            for s_ in rows:
                row = ""
                for c in range(int(lengths[k] * cpb)):
                    cell = cells.get((c, s_), "")
                    row += (cell + "-" * 3)[:3] if cell else "---"
                rows[s_] += row + "|"
        for s_ in reversed(range(len(opts.open_midis))):
            lines.append(rows[s_])
        lines.append("")
    return "\n".join(lines)


def summary(tab: Tab) -> dict:
    played = [n for n in tab.notes if n.string is not None]
    frets = [n.fret for n in played]
    return {"notes": len(tab.notes), "bars": tab.bars, "unplayable": tab.unplayable,
            "open_strings": sum(1 for f in frets if f == 0),
            "fret_range": (min(frets), max(frets)) if frets else None,
            "moved": tab.moved, "dropped": len(tab.dropped), "triplet_beats": len(tab.triplet_beats),
            "tempo_changes": max(0, len(tab.tempo_points) - 1)}
