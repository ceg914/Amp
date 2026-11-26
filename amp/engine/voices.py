"""성부 추적과 성부 데이터 (v0.7.0 Feature B, 명세 5.4~5.6). Qt 비의존.

다중 피치 후보(amp/engine/dsp/multipitch.py)를 목소리별 경로로 잇는다.
1. 짧은 이음(contour): 프레임마다 후보를 진행 중인 이음의 예상 높이(마지막 높이 + 기울기)에 가장 가깝게 붙인다
   (프레임당 80센트 넘게 뛰면 새 이음). 60ms 넘게 비면 이음을 닫고, 50ms보다 짧은 이음은 버린다
2. 성부: 시작 시각이 비슷한(50ms 안) 이음을 묶어, 비어 있는 성부에 높이 차이 + 쉰 시간이 가장 작게 함께 나눈다.
   성부 수가 max_voices를 넘으면 그 이음은 성부에 넣지 않고 '불확실'로 남긴다
3. 불확실: 두 성부가 50센트 안으로 붙은 프레임(유니즌·교차), 가장 좋은 성부와 다음 성부의 비용 차이가 1반음보다 작은 이음
4. 성부는 높이 중앙값이 높은 순서로 '성부 1, 2, 3'. 역할(main/harmony)은 아직 정하지 않는다 (Feature C)

수동 편집(되돌릴 수 있게 새 VoiceSet을 돌려줌): 성부 추가·이름·숨기기·역할·확정, 구간 바꾸기·옮기기·지우기,
구간에서 다른 후보 고르기, 나누기·합치기. 원본 오디오는 바꾸지 않는다. 피치만 있는 성부이며 소리 파일이 아니다.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from amp.core import result_meta as RM
from amp.engine.dsp.multipitch import Candidates

SCHEMA = "sonatlas.voice_tracks"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "1"
JSON_NAME = "voice_tracks.json"
NPZ_NAME = "multi_pitch.npz"
HISTORY_DIR = "multi_pitch_runs"
ROLES = {"main_candidate": "주선율 후보", "harmony_candidate": "화음 후보", "unknown": "모름", "user": "사용자 지정"}

MAX_STEP_CENTS = 80.0
MAX_GAP_FRAMES = 6
MIN_CONTOUR_FRAMES = 5
BATCH_FRAMES = 5
NEW_VOICE_COST = 6.0
AMBIGUOUS_COST = 1.0
UNISON_CENTS = 50.0
NEW_VOICE_MIN_FRAMES = 15  # 새 성부를 여는 이음은 150ms 이상이고
NEW_VOICE_MIN_CONF = 0.3  # 평균 신뢰도가 이 이상이어야 한다 (남은 배음·반주 조각이 성부를 차지하지 않게)
MIN_VOICE_FRAMES = 20  # 이보다 짧은 성부는 성부로 두지 않는다


@dataclass
class Contour:
    frames: list = field(default_factory=list)
    cents: list = field(default_factory=list)
    conf: list = field(default_factory=list)
    slope: float = 0.0
    shared: list = field(default_factory=list)  # 다른 이음과 같은 후보를 나눠 가진 프레임 (유니즌·교차)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def head(self, n: int = 5) -> float:
        return float(np.median(self.cents[:n]))

    def tail(self, n: int = 10) -> float:
        return float(np.median(self.cents[-n:]))

    def end_fit(self, n: int = 30) -> tuple[float, float]:
        """끝부분 직선 (끝 프레임에서의 높이, 기울기 센트/프레임). 비브라토가 고르게 지워지도록 0.3초로 맞춘다."""
        fr = np.asarray(self.frames[-n:], dtype=float)
        ce = np.asarray(self.cents[-n:], dtype=float)
        if len(fr) < 8:
            return float(np.median(ce)), 0.0
        k, b = np.polyfit(fr - fr[-1], ce, 1)
        return float(b), float(np.clip(k, -20.0, 20.0))

    @property
    def quality(self) -> float:
        return len(self.frames) * float(np.mean(self.conf))

    @property
    def can_open_voice(self) -> bool:
        return len(self.frames) >= NEW_VOICE_MIN_FRAMES and float(np.mean(self.conf)) >= NEW_VOICE_MIN_CONF


def _cents(f: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(f / 440.0) + 6900.0


def build_contours(c: Candidates) -> list[Contour]:
    cents = _cents(c.f0_hz)
    first = np.maximum(c.salience[:, :1], 1e-9)
    conf = np.clip(c.salience / first, 0, 1) * np.clip(c.evidence, 0, 1)
    active: list[Contour] = []
    done: list[Contour] = []
    for t in range(c.n):
        cand = [(k, cents[t, k]) for k in range(c.k) if np.isfinite(cents[t, k])]
        pairs = []
        for ai, a in enumerate(active):
            gap = t - a.end
            pred = a.cents[-1] + a.slope * gap
            lim = MAX_STEP_CENTS + 10.0 * (gap - 1)
            for k, x in cand:
                d = abs(x - pred)
                if d <= lim:
                    pairs.append((d, ai, k))
        pairs.sort()
        used_a, used_k = set(), set()
        taker: dict[int, int] = {}
        for d, ai, k in pairs:
            if ai in used_a or k in used_k:
                continue
            taker[k] = ai
            used_a.add(ai)
            used_k.add(k)
        # 후보가 하나로 합쳐진 구간(유니즌·교차): 예상 높이가 그 후보에 가까운 다른 이음도 같은 점을 나눠 갖고,
        # 둘 다 기울기를 바꾸지 않는다 (갈라진 뒤 각자 가던 방향으로 이어지게)
        frozen = set()
        for ai, a in enumerate(active):
            if ai in used_a:
                continue
            gap = t - a.end
            pred = a.cents[-1] + a.slope * gap
            for k, owner in taker.items():
                if abs(cents[t, k] - pred) <= UNISON_CENTS * 1.5:
                    a.frames.append(t)
                    a.cents.append(cents[t, k])
                    a.conf.append(conf[t, k] * 0.5)
                    a.shared.append(t)
                    active[owner].shared.append(t)
                    frozen.update((ai, owner))
                    used_a.add(ai)
                    break
        for k, ai in taker.items():
            a = active[ai]
            x = cents[t, k]
            if ai not in frozen:
                step = (x - a.cents[-1]) / max(1, t - a.end)
                a.slope = float(np.clip(0.6 * a.slope + 0.4 * step, -30, 30))
            a.frames.append(t)
            a.cents.append(x)
            a.conf.append(conf[t, k])
        for k, x in cand:
            if k not in used_k:
                active.append(Contour([t], [x], [conf[t, k]]))
        keep = []
        for a in active:
            (keep if t - a.end <= MAX_GAP_FRAMES else done).append(a)
        active = keep
    done.extend(active)
    return sorted((x for x in done if len(x.frames) >= MIN_CONTOUR_FRAMES), key=lambda x: (x.start, -x.head()))


@dataclass
class VoiceTrack:
    voice_track_id: str
    display_name: str
    f0_hz: np.ndarray
    confidence: np.ndarray
    role: str = "unknown"
    user_confirmed: bool = False
    hidden: bool = False
    edited: bool = False
    source_id: str = ""
    analysis_version: str = ANALYSIS_VERSION

    def voiced(self) -> np.ndarray:
        return np.isfinite(self.f0_hz)

    def voiced_intervals(self, start_frame: int = 0, fr: float = 100.0, min_frames: int = 1) -> list[tuple[float, float]]:
        v = self.voiced().astype(np.int8)
        d = np.diff(np.r_[0, v, 0])
        out = []
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s >= min_frames:
                out.append((round((start_frame + s) / fr, 3), round((start_frame + e) / fr, 3)))
        return out

    def median_midi(self) -> float | None:
        v = self.voiced()
        if not v.any():
            return None
        return float(np.median(69 + 12 * np.log2(self.f0_hz[v] / 440.0)))

    def meta_dict(self) -> dict:
        return {"voice_track_id": self.voice_track_id, "display_name": self.display_name, "role": self.role,
                "user_confirmed": self.user_confirmed, "hidden": self.hidden, "edited": self.edited,
                "source_id": self.source_id, "analysis_version": self.analysis_version}


@dataclass
class VoiceSet:
    meta: RM.ResultMeta
    voices: list[VoiceTrack]
    candidates: Candidates
    uncertain: np.ndarray  # (T,) 성부 연결을 확신할 수 없는 프레임
    start_frame: int = 0
    frame_rate: float = 100.0
    unassigned_frames: int = 0  # 성부 수를 넘어 넣지 못한 이음의 프레임 수
    history: list = field(default_factory=list)  # 편집 기록 (사람이 읽는 문장)

    @property
    def n(self) -> int:
        return self.candidates.n

    def voice(self, vid: str) -> VoiceTrack | None:
        return next((v for v in self.voices if v.voice_track_id == vid), None)

    def frame(self, t: float) -> int:
        return int(round(t * self.frame_rate)) - self.start_frame

    def frames(self, t0: float, t1: float) -> slice:
        a = max(0, self.frame(min(t0, t1)))
        b = min(self.n, self.frame(max(t0, t1)) + 1)
        return slice(a, max(a, b))

    def copy(self) -> "VoiceSet":
        return VoiceSet(self.meta, [replace(v, f0_hz=v.f0_hz.copy(), confidence=v.confidence.copy()) for v in self.voices],
                        self.candidates, self.uncertain.copy(), self.start_frame, self.frame_rate,
                        self.unassigned_frames, list(self.history))

    @property
    def has_user_edits(self) -> bool:
        return any(v.edited or v.user_confirmed for v in self.voices)


def track_voices(c: Candidates, max_voices: int = 2, source_id: str = "") -> tuple[list[VoiceTrack], np.ndarray, int]:
    """(성부들, 불확실 프레임, 성부에 넣지 못한 프레임 수)."""
    max_voices = int(max(1, min(3, max_voices)))
    contours = build_contours(c)
    T = c.n
    voices: list[dict] = []  # {"contours": [...], "end": int, "last": float}
    uncertain = np.zeros(T, dtype=bool)
    unassigned = 0
    i = 0
    while i < len(contours):
        batch = [contours[i]]
        j = i + 1
        while j < len(contours) and contours[j].start - contours[i].start <= BATCH_FRAMES:
            batch.append(contours[j])
            j += 1
        i = j
        free = [vi for vi, v in enumerate(voices) if v["end"] < min(b.start for b in batch) + 3]
        options = free + [None] * max(0, min(len(batch), max_voices - len(voices)))
        best = _assign(batch, voices, options)
        for b, choice, ambiguous in best:
            if choice == "skip":
                unassigned += len(b.frames)
                uncertain[b.frames] = True
                continue
            last, slope = b.end_fit()
            if choice is None:
                voices.append({"contours": [b], "end": b.end, "last": last, "slope": slope})
            else:
                v = voices[choice]
                v["contours"].append(b)
                v["end"], v["last"], v["slope"] = max(v["end"], b.end), last, slope
            if ambiguous:
                uncertain[b.frames] = True
            if b.shared:
                uncertain[b.shared] = True
    tracks = []
    for v in voices:
        f0 = np.full(T, np.nan)
        conf = np.zeros(T)
        for b in v["contours"]:
            f0[b.frames] = 440.0 * 2 ** ((np.asarray(b.cents) - 6900.0) / 1200.0)
            conf[b.frames] = np.asarray(b.conf)
        tracks.append((f0, conf))
    small = [fc for fc in tracks if np.isfinite(fc[0]).sum() < MIN_VOICE_FRAMES]
    for f0, _c in small:
        unassigned += int(np.isfinite(f0).sum())
    tracks = [fc for fc in tracks if np.isfinite(fc[0]).sum() >= MIN_VOICE_FRAMES]
    tracks.sort(key=lambda fc: -(np.nanmedian(fc[0]) if np.isfinite(fc[0]).any() else 0))
    out = []
    for n, (f0, conf) in enumerate(tracks, 1):
        out.append(VoiceTrack(f"v{n}", f"성부 {n}", f0, conf, source_id=source_id))
    _mark_unison(out, uncertain)
    for vt in out:
        vt.confidence = np.where(uncertain & vt.voiced(), vt.confidence * 0.5, vt.confidence)
    return out, uncertain, unassigned


def _mark_unison(tracks: list[VoiceTrack], uncertain: np.ndarray) -> None:
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            fa, fb = tracks[a].f0_hz, tracks[b].f0_hz
            both = np.isfinite(fa) & np.isfinite(fb)
            with np.errstate(divide="ignore", invalid="ignore"):
                close = both & (np.abs(1200 * np.log2(fa / fb)) < UNISON_CENTS)
            uncertain |= close


def _assign(batch, voices, options):
    """batch 이음을 성부(기존 번호 또는 None=새 성부)에 나눈다. 품질(길이 × 신뢰도)이 높은 이음을 먼저 넣고,
    같은 수를 넣는 조합 중 총비용이 가장 작은 것. 넣을 곳이 없으면 'skip'.
    애매함: 기존 성부 중 가장 좋은 곳과 다음 곳의 비용 차이가 1반음보다 작을 때만 (새 성부끼리는 따지지 않음)."""
    import itertools

    def cost(b, choice):
        if choice is None:
            return NEW_VOICE_COST if b.can_open_voice else None
        v = voices[choice]
        g = max(0, b.start - v["end"])
        pred = v["last"] + v.get("slope", 0.0) * min(g, 30)  # 기울기로 0.3초까지 이어 본다 (교차 구간)
        return abs(b.head() - pred) / 100.0 + 0.5 * min(g / 100.0, 4.0)

    order = sorted(range(len(batch)), key=lambda i: -batch[i].quality)
    best, best_key = None, None
    slots = list(range(len(options)))
    for k in range(min(len(batch), len(slots)), -1, -1):
        for chosen in itertools.combinations(order, k):
            for perm in itertools.permutations(slots, k):
                costs = [cost(batch[ci], options[p]) for ci, p in zip(chosen, perm)]
                if any(c is None for c in costs):
                    continue
                key = (-sum(batch[ci].quality for ci in chosen), sum(costs))
                if best_key is None or key < best_key:
                    best_key, best = key, (chosen, perm)
        if best is not None:
            break
    result = []
    chosen, perm = best if best is not None else ((), ())
    mapping = dict(zip(chosen, perm))
    existing = [s for s in slots if options[s] is not None]
    for ci, b in enumerate(batch):
        if ci not in mapping:
            result.append((b, "skip", True))
            continue
        choice = options[mapping[ci]]
        ambiguous = False
        if choice is not None and len(existing) > 1:
            cs = sorted(cost(b, options[s]) for s in existing)
            ambiguous = cs[1] - cs[0] < AMBIGUOUS_COST
        result.append((b, choice, ambiguous))
    return result


def analyze(c: Candidates, meta: RM.ResultMeta, max_voices: int = 2) -> VoiceSet:
    tracks, unc, unassigned = track_voices(c, max_voices, meta.analysis_source_id)
    return VoiceSet(meta, tracks, c, unc, c.start_frame, c.frame_rate, unassigned)


# --- 수동 편집 (새 VoiceSet을 돌려준다) ---------------------------------------------------

def _touch(vs: VoiceSet, vids, note: str) -> None:
    for v in vs.voices:
        if v.voice_track_id in vids:
            v.edited = True
    vs.history.append(note)


def _new_id(vs: VoiceSet) -> str:
    n = 1
    ids = {v.voice_track_id for v in vs.voices}
    while f"v{n}" in ids:
        n += 1
    return f"v{n}"


def add_voice(vs: VoiceSet, name: str = "") -> tuple[VoiceSet, str]:
    out = vs.copy()
    vid = _new_id(out)
    out.voices.append(VoiceTrack(vid, name or f"성부 {vid[1:]}", np.full(out.n, np.nan), np.zeros(out.n), role="user",
                                 edited=True, source_id=vs.meta.analysis_source_id))
    out.history.append(f"{vid} 추가")
    return out, vid


def remove_voice(vs: VoiceSet, vid: str) -> VoiceSet:
    out = vs.copy()
    out.voices = [v for v in out.voices if v.voice_track_id != vid]
    out.history.append(f"{vid} 삭제")
    return out


def set_attr(vs: VoiceSet, vid: str, **kw) -> VoiceSet:
    """display_name·role·hidden·user_confirmed. 이름·역할·확정은 사용자 편집으로 본다(재분석에서 보호)."""
    out = vs.copy()
    v = out.voice(vid)
    if v is None:
        return vs
    for k, val in kw.items():
        if k not in ("display_name", "role", "hidden", "user_confirmed"):
            raise ValueError(k)
        if k == "role" and val not in ROLES:
            raise ValueError(val)
        setattr(v, k, val)
    if set(kw) - {"hidden"}:
        v.edited = True
    out.history.append(f"{vid} {', '.join(f'{k}={val}' for k, val in kw.items())}")
    return out


def swap_range(vs: VoiceSet, a: str, b: str, t0: float, t1: float) -> VoiceSet:
    out = vs.copy()
    va, vb = out.voice(a), out.voice(b)
    sl = out.frames(t0, t1)
    for arr in ("f0_hz", "confidence"):
        x, y = getattr(va, arr)[sl].copy(), getattr(vb, arr)[sl].copy()
        getattr(va, arr)[sl], getattr(vb, arr)[sl] = y, x
    _touch(out, {a, b}, f"{a}↔{b} {t0:.2f}–{t1:.2f}s 바꿈")
    return out


def move_range(vs: VoiceSet, src: str, dst: str, t0: float, t1: float) -> tuple[VoiceSet, int]:
    """src의 유성 프레임을 dst로 옮긴다. dst가 이미 유성인 프레임은 그대로 두고 그 수를 돌려준다."""
    out = vs.copy()
    vsrc, vdst = out.voice(src), out.voice(dst)
    sl = out.frames(t0, t1)
    s_on = np.isfinite(vsrc.f0_hz[sl])
    d_on = np.isfinite(vdst.f0_hz[sl])
    mv = s_on & ~d_on
    idx = np.arange(sl.start, sl.stop)[mv]
    vdst.f0_hz[idx], vdst.confidence[idx] = vsrc.f0_hz[idx], vsrc.confidence[idx]
    vsrc.f0_hz[idx], vsrc.confidence[idx] = np.nan, 0.0
    _touch(out, {src, dst}, f"{src}→{dst} {t0:.2f}–{t1:.2f}s 옮김")
    return out, int((s_on & d_on).sum())


def clear_range(vs: VoiceSet, vid: str, t0: float, t1: float) -> VoiceSet:
    out = vs.copy()
    v = out.voice(vid)
    sl = out.frames(t0, t1)
    v.f0_hz[sl], v.confidence[sl] = np.nan, 0.0
    _touch(out, {vid}, f"{vid} {t0:.2f}–{t1:.2f}s 지움")
    return out


def pick_candidate(vs: VoiceSet, vid: str, t0: float, t1: float, target_midi: float,
                   max_cents: float = 150.0) -> tuple[VoiceSet, int]:
    """구간에서 target_midi에 가장 가까운 후보(±max_cents)로 바꾼다. 바꾼 프레임 수."""
    out = vs.copy()
    v = out.voice(vid)
    sl = out.frames(t0, t1)
    c = out.candidates
    cents = _cents(c.f0_hz[sl])
    d = np.abs(cents - target_midi * 100.0)
    d = np.where(np.isfinite(d), d, np.inf)
    k = np.argmin(d, axis=1)
    ok = d[np.arange(len(k)), k] <= max_cents
    idx = np.arange(sl.start, sl.stop)[ok]
    v.f0_hz[idx] = c.f0_hz[idx, k[ok]]
    first = np.maximum(c.salience[idx, 0], 1e-9)
    v.confidence[idx] = np.clip(c.salience[idx, k[ok]] / first, 0, 1)
    _touch(out, {vid}, f"{vid} {t0:.2f}–{t1:.2f}s 후보 다시 고름 (MIDI {target_midi:.1f} 근처)")
    return out, int(ok.sum())


def split_voice(vs: VoiceSet, vid: str, t: float) -> tuple[VoiceSet, str]:
    out, new = add_voice(vs, "")
    src, dst = out.voice(vid), out.voice(new)
    i = max(0, min(out.n, out.frame(t)))
    dst.f0_hz[i:], dst.confidence[i:] = src.f0_hz[i:].copy(), src.confidence[i:].copy()
    src.f0_hz[i:], src.confidence[i:] = np.nan, 0.0
    dst.display_name = f"{src.display_name} (나눔)"
    _touch(out, {vid, new}, f"{vid} {t:.2f}s에서 나눔 → {new}")
    return out, new


def merge_voices(vs: VoiceSet, keep: str, other: str) -> tuple[VoiceSet, int]:
    """other를 keep에 합친다. 둘 다 유성인 프레임은 keep 값을 남기고 그 수를 돌려준다."""
    out = vs.copy()
    a, b = out.voice(keep), out.voice(other)
    take = np.isfinite(b.f0_hz) & ~np.isfinite(a.f0_hz)
    conflicts = int((np.isfinite(b.f0_hz) & np.isfinite(a.f0_hz)).sum())
    a.f0_hz[take], a.confidence[take] = b.f0_hz[take], b.confidence[take]
    out.voices = [v for v in out.voices if v.voice_track_id != other]
    _touch(out, {keep}, f"{other}를 {keep}에 합침 (겹친 {conflicts}프레임은 {keep} 값)")
    return out, conflicts


def replace_range_from(vs: VoiceSet, vid: str, region: Candidates, t0: float, t1: float) -> tuple[VoiceSet, int]:
    """특정 음역만 다시 분석한 후보(region, 한 성부로 이음)로 vid의 구간을 바꾼다. 바꾼 프레임 수."""
    tracks, _u, _n = track_voices(region, 1)
    out = vs.copy()
    v = out.voice(vid)
    sl = out.frames(t0, t1)
    off = region.start_frame - out.start_frame
    newf = np.full(out.n, np.nan)
    newc = np.zeros(out.n)
    if tracks:
        a, b = max(0, off), min(out.n, off + region.n)
        newf[a:b] = tracks[0].f0_hz[a - off:b - off]
        newc[a:b] = tracks[0].confidence[a - off:b - off]
    v.f0_hz[sl], v.confidence[sl] = newf[sl], newc[sl]
    _touch(out, {vid}, f"{vid} {t0:.2f}–{t1:.2f}s 음역을 정해 다시 분석")
    return out, int(np.isfinite(newf[sl]).sum())


# --- 저장 ----------------------------------------------------------------------------

def save(analysis_dir: str | Path, vs: VoiceSet) -> None:
    """multi_pitch.npz(후보·성부 곡선)를 먼저, voice_tracks.json을 마지막에 바꿔 넣는다."""
    d = Path(analysis_dir)
    d.mkdir(parents=True, exist_ok=True)
    c = vs.candidates
    V = len(vs.voices)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=d)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, f0_hz=c.f0_hz.astype(np.float32), salience=c.salience.astype(np.float32),
                                evidence=c.evidence.astype(np.float32), odd_even=c.odd_even.astype(np.float32),
                                frame_db=c.frame_db.astype(np.float32), start_frame=np.int64(c.start_frame),
                                frame_rate=np.float64(c.frame_rate),
                                voice_f0=np.stack([v.f0_hz for v in vs.voices]).astype(np.float32) if V else np.zeros((0, c.n), np.float32),
                                voice_conf=np.stack([v.confidence for v in vs.voices]).astype(np.float32) if V else np.zeros((0, c.n), np.float32),
                                uncertain=vs.uncertain.astype(bool))
        os.replace(tmp, d / NPZ_NAME)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    from amp.core.fileio import atomic_write_json

    atomic_write_json(d / JSON_NAME, {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "meta": vs.meta.to_dict(),
                                      "voices": [v.meta_dict() for v in vs.voices], "start_frame": vs.start_frame,
                                      "frame_rate": vs.frame_rate, "unassigned_frames": vs.unassigned_frames,
                                      "history": vs.history[-200:], "pitch_only": True})


def load(analysis_dir: str | Path) -> VoiceSet | None:
    d = Path(analysis_dir)
    try:
        j = json.loads((d / JSON_NAME).read_text(encoding="utf-8"))
        meta = RM.meta_from_dict(j.get("meta"))
        if j.get("schema") != SCHEMA or meta is None or int(j.get("schema_version", 0)) > SCHEMA_VERSION:
            return None
        with np.load(d / NPZ_NAME, allow_pickle=False) as z:
            c = Candidates(z["f0_hz"].astype(np.float64), z["salience"].astype(np.float64),
                           z["evidence"].astype(np.float64), z["odd_even"].astype(np.float64),
                           z["frame_db"].astype(np.float64), int(z["start_frame"]), float(z["frame_rate"]))
            vf, vc, unc = z["voice_f0"].astype(np.float64), z["voice_conf"].astype(np.float64), z["uncertain"].astype(bool)
        vm = j.get("voices") or []
        if len(vm) != len(vf):
            return None
        voices = [VoiceTrack(m["voice_track_id"], m.get("display_name", m["voice_track_id"]), vf[i], vc[i],
                             m.get("role", "unknown"), bool(m.get("user_confirmed")), bool(m.get("hidden")),
                             bool(m.get("edited")), str(m.get("source_id", "")), str(m.get("analysis_version", "")))
                  for i, m in enumerate(vm)]
        return VoiceSet(meta, voices, c, unc, int(j.get("start_frame", 0)), float(j.get("frame_rate", 100.0)),
                        int(j.get("unassigned_frames", 0)), list(j.get("history") or []))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def archive(analysis_dir: str | Path, reason: str) -> Path | None:
    """지금 성부 결과를 multi_pitch_runs/<시각>/로 옮겨 보관한다 (지우지 않음). 옮긴 폴더."""
    from datetime import datetime

    d = Path(analysis_dir)
    if not (d / JSON_NAME).is_file():
        return None
    dst = d / HISTORY_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dst.mkdir(parents=True, exist_ok=True)
    for name in (NPZ_NAME, JSON_NAME):
        if (d / name).is_file():
            os.replace(d / name, dst / name)
    (dst / "reason.txt").write_text(reason, encoding="utf-8")
    return dst


def merge_reanalysis(old: VoiceSet | None, new: VoiceSet) -> VoiceSet:
    """다시 분석: 사용자가 확정했거나 고친 성부는 그대로 남기고(새 결과에 덧붙임), 나머지만 새 결과로 바꾼다."""
    if old is None or not old.has_user_edits or old.n != new.n:
        return new
    out = new.copy()
    kept = [copy.deepcopy(v) for v in old.voices if v.edited or v.user_confirmed]
    ids = {v.voice_track_id for v in kept}
    for v in out.voices:
        if v.voice_track_id in ids:
            v.voice_track_id = _new_id(VoiceSet(out.meta, out.voices + kept, out.candidates, out.uncertain))
            v.display_name = f"{v.display_name} (새 분석)"
    out.voices = kept + out.voices
    out.history = old.history + [f"다시 분석: 사용자 성부 {len(kept)}개 유지"]
    return out


# --- 실행 (작업 프로세스 호출) -------------------------------------------------------------

def run_analysis(inp, analysis_dir: str | Path, runner, max_voices: int = 2, progress=None, token=None,
                 numba_cache_dir: str | None = None) -> VoiceSet:
    """곡 전체 다중 피치 → 성부. 이전 결과는 multi_pitch_runs/로 옮겨 보관하고, 사용자가 고치거나 확정한 성부는
    새 결과에 그대로 남긴다. inp는 pitch_runs.PitchInput."""
    from amp.engine.dsp import multipitch as MP

    report = progress or (lambda f, m: None)
    with tempfile.TemporaryDirectory(prefix="sonatlas-mp-") as scratch:
        runner("multipitch", {"input": inp.path, "max_voices": int(max_voices), "out_dir": scratch},
               lambda f, m: report(0.95 * f, m), token, numba_cache_dir)
        cand = Candidates.load(Path(scratch) / "multipitch.npz")
        info = json.loads((Path(scratch) / "info.json").read_text(encoding="utf-8"))
    meta = RM.make_meta(SCHEMA, SCHEMA_VERSION, inp.source_sha256, input_kind=inp.kind, input_id=inp.input_id,
                        input_hash=inp.input_hash, model_id="harmonic_salience", model_version=str(MP.CODE_VERSION),
                        analysis_version=ANALYSIS_VERSION,
                        analysis_settings={"max_voices": int(max_voices), "fmin": info.get("fmin"), "fmax": info.get("fmax")})
    new = analyze(cand, meta, max_voices)
    if not any(v.voiced().any() for v in new.voices):
        new.meta = new.meta.with_status(RM.UNCERTAIN, "목소리 후보를 찾지 못했습니다.")
    old = load(analysis_dir)
    out = merge_reanalysis(old, new)
    if old is not None:
        archive(analysis_dir, "다시 분석하기 전 결과")
    save(analysis_dir, out)
    report(1.0, f"성부 {len(out.voices)}개")
    return out


def run_band(inp, vs: VoiceSet, vid: str, t0: float, t1: float, fmin: float, fmax: float, runner, progress=None,
             token=None, numba_cache_dir: str | None = None) -> tuple[VoiceSet, int]:
    """한 성부의 구간을 음역(fmin~fmax)을 정해 다시 분석한다. 새 VoiceSet과 바꾼 프레임 수 (저장은 부르는 쪽)."""
    with tempfile.TemporaryDirectory(prefix="sonatlas-mpb-") as scratch:
        runner("multipitch", {"input": inp.path, "max_voices": 1, "start": float(t0), "end": float(t1),
                              "fmin": float(fmin), "fmax": float(fmax), "out_dir": scratch},
               progress, token, numba_cache_dir)
        region = Candidates.load(Path(scratch) / "multipitch.npz")
    return replace_range_from(vs, vid, region, t0, t1)
