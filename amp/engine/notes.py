"""피치 곡선 → 음표 (Qt 비의존, numpy·scipy).

순서
1. 유성 판정: f0가 있고 유성 확률이 임계 이상이며 분석 음역 안에 있는 프레임.
   사용자가 지정한 보컬 없는 구간(no_vocal_ranges)의 프레임은 무성으로 본다(분리 결과에 섞인 반주).
   짧은 무성 틈(자음·피치 놓침)은 같은 프레이즈로 잇는다. 단, 틈 끝에 뚜렷한 재발음이 있으면
   (표준·정밀) 잇지 않는다. 음절 사이에서 잠깐 쉬고 다시 부른 것이다.
2. 중심선: 약 비브라토 한 주기 길이의 중앙값 필터. 비브라토는 상쇄되고, 음 사이 경계는 날카롭게,
   포르타멘토는 경사로 남는다.
3. 변화점: 중심선이 현재 음에서 기준(기본 1반음, 허용 오차 0.3) 이상 벗어나고, 새 높이가 hold 시간
   동안 안정적으로 유지될 때만 나눈다. 경계는 두 음의 중간을 지나는 지점이다.
   그래서 포르타멘토는 가운데에서 한 번만 나뉘고, 중간 음들이 따로 음표가 되지 않는다.
4. 재발음: 재발음 지표(온셋 × 음량 하락, pitch_post.articulation_envelope) 피크가 split_strength 이상이면
   뚜렷한 발성 시작으로 본다. 음높이 경계 ±60ms 안에 있으면 경계를 그리로 옮기고, 같은 높이가 이어지는
   구간 안에 있으면 나눈다(정밀·표준 민감도).
5. 짧은 조각: 최소 길이보다 짧으면 가까운 높이의 이웃에 합치고, 이웃이 없으면 버린다.
6. 미끄러짐 흡수: 두 음 사이에 낀 짧은 조각들이 높이가 두 음 사이에 있고 안에서 한 방향으로 계속 움직이면
   (포르타멘토가 계단처럼 잡힌 것) 따로 음표로 두지 않고 앞뒤 음에 나눠 붙인다.
   경계는 조각 안의 재발음 지점, 없으면 두 음의 중간 높이를 지나는 지점이다.
7. 틈이 아주 짧은 같은 높이의 이웃 음표를 합친다.
8. 짧은 이탈(표준·정밀): 음표 안에서 중심선 창보다 짧게 기준 이상 벗어났다 돌아오는 구간을 떼어 낸다.
   최소 길이 미만이면서 신뢰도가 높으면 장식음(ornament), 최소 길이 이상이면 일반 음표가 된다.
   비브라토 폭이 기준에 가까운 음은 건너뛴다.
9. 접근음 흡수(STEP 3.5): 노래를 시작할 때 목표 음보다 낮은 데서 출발해 올라가는 짧은 구간(끌어올림)이
   따로 음표로 잡혔으면 뒤의 목표 음에 합친다. 다음을 모두 만족할 때만 합친다.
   - 짧은 음이 프레이즈 첫 음이거나 재발음으로 시작한다 (노래가 시작되는 자리).
   - 길이 APPROACH_MAX_S 이하, 장식음으로 판정되지 않았다.
   - 바로 뒤 음(틈 APPROACH_GAP_S 이하)이 1~7반음 위다. 4반음 이상이면 짧은 음 안에서 1반음 이상 올라가야 한다.
   - 짧은 음 안에서 음높이가 위로 움직인다(기울기 APPROACH_MIN_SLOPE 이상). 그리고 0.3반음 폭 안에 머문 구간이
     APPROACH_PLATEAU_S 미만이다. 평평하게 머문 음은 한 모음 안에서 따로 부른 음(멜리스마)일 수 있다.
   - 뒤 음 시작에 재발음 근거가 없다(재발음 지표 APPROACH_ONSET_MAX 미만). 재발음 지표가 없으면 합치지 않는다.
   - 뒤 음이 짧은 음보다 짧지 않다 (목표 음에 머문다).
   - 가사 음절 시작 시각이 주어졌으면(정렬한 가사가 있을 때) 뒤 음 시작 ±APPROACH_SYLLABLE_TOL 안에서 다른 음절이
     시작하지 않는다. 뒤 음이 다음 음절의 음이면 짧은 음은 앞 음절의 실제 음이다.
   피치 곡선은 바꾸지 않는다. 합친 음표의 높이는 뒤 음의 높이다.
10. 음표별 비브라토(3.5~9Hz, 20cent 이상)를 추정한다. 음의 앞뒤 가장자리는 빼고 본다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from amp.core.ids import new_id
from amp.core.model import Note, PitchCurve, Vibrato
from amp.engine.pitch_post import voiced_runs
from amp.engine.settings import NoteDetectionParams

DETECT_TOLERANCE = 0.3  # 반음. 가수의 음정 오차를 감안해 기준보다 조금 작은 변화도 인정한다
MIN_ISLAND_S = 0.03  # 이보다 짧은 유성 조각은 잡음으로 본다
MIN_VOICED_FRACTION = 0.6  # 음표 구간 중 실제 유성 프레임이 이 비율 미만이면 버린다
SNAP_BEFORE_S, SNAP_AFTER_S = 0.06, 0.06  # 음높이 경계를 온셋으로 옮길 수 있는 범위
GAP_ONSET_BEFORE, GAP_ONSET_AFTER = 2, 4  # 틈 끝(프레임) 기준으로 재발음 피크를 찾는 범위
GLIDE_PIECE_MAX_S = 0.25  # 미끄러짐 조각 하나의 최대 길이
GLIDE_RUN_MAX_S = 0.4  # 미끄러짐 조각 묶음 전체의 최대 길이
GLIDE_MIN_DRIFT = 0.4  # 반음. 조각 안에서 이만큼 이상 한 방향으로 움직여야 미끄러짐으로 본다
REF_WINDOW_S = 0.5  # 현재 음 높이를 추정할 때 볼 최근 구간
# 접근음(끌어올림) 흡수. 값은 원리에서 정했다: 16분음표(90~100BPM에서 0.15~0.17초)보다 조금 긴 길이까지,
# 재발음 지표는 표준 민감도의 같은 음 분할 임계(4.0)의 절반 미만이면 '다시 부른 근거 없음'으로 본다.
APPROACH_MAX_S = 0.25
APPROACH_GAP_S = 0.08
APPROACH_MAX_INTERVAL = 7
APPROACH_BIG_INTERVAL = 4  # 이 이상 떨어져 있으면 짧은 음 안에서 실제로 1반음 이상 올라가야 한다
APPROACH_MIN_SLOPE = 1.5  # 반음/초
APPROACH_ONSET_MAX = 2.0
APPROACH_PLATEAU_S = 0.12  # 이보다 오래 한 높이에 머물렀으면 접근이 아니라 음이다 (합성 사례 a_legato_step 뒤에 추가)
APPROACH_PLATEAU_BAND = 0.3  # 반음
APPROACH_SYLLABLE_TOL = 0.08  # 초
APPROACH_ONSET_WINDOW = (3, 4)  # 뒤 음 시작 기준 앞·뒤 프레임


@dataclass
class _Seg:
    s: int
    e: int
    onset_start: bool = False  # 온셋으로 나뉜 시작점 (같은 음이어도 합치지 않는다)
    pitch: int = 0
    est: float = 0.0
    conf: float = 0.0
    ornament: bool = False


def _blocked_frames(ranges, start_time: float, fr: float, n: int) -> np.ndarray:
    """보컬 없는 구간 [시작, 끝](초)에 드는 프레임이면 True. 프레임 i의 시각은 start_time + i/fr."""
    blocked = np.zeros(n, dtype=bool)
    for a, b in ranges or ():
        i0 = max(0, int(np.ceil((a - start_time) * fr - 1e-6)))
        i1 = min(n, int(np.floor((b - start_time) * fr + 1e-6)) + 1)
        if i1 > i0:
            blocked[i0:i1] = True
    return blocked


def _fill_gaps(midi: np.ndarray, voiced: np.ndarray, bridged: np.ndarray) -> np.ndarray:
    """잇기로 한 짧은 무성 틈은 앞뒤 높이를 직선으로 이어 채운다."""
    out = np.where(voiced, midi, np.nan)
    for a, b in voiced_runs(bridged):
        seg = out[a:b]
        ok = np.isfinite(seg)
        if ok.any() and not ok.all():
            x = np.arange(b - a)
            seg[~ok] = np.interp(x[~ok], x[ok], seg[ok])
        out[a:b] = seg
    return out


def _segment_region(center: np.ndarray, params: NoteDetectionParams, fr: float) -> list[int]:
    """한 프레이즈 안의 경계 인덱스 목록 (0과 길이 포함)."""
    L = len(center)
    hold = max(2, int(round(params.hold_s * fr)))
    detect = max(0.5, params.change_semitones - DETECT_TOLERANCE)
    stable_spread = 0.45  # 새 높이가 hold 동안 이 범위 안에 머물러야 안정된 음으로 본다
    max_drift = 0.3  # hold 동안 한 방향으로 흘러가면(포르타멘토) 아직 도착하지 않은 것으로 본다
    ref_win = int(REF_WINDOW_S * fr)
    bounds = [0]
    seg_start = 0
    ref_from = 0  # 현재 음 높이를 잴 때 쓰는 첫 프레임. 분할 직후에는 안정 구간 시작점이다
    i = 1
    while i < L:
        lo = max(ref_from + min(hold // 2, max(0, i - ref_from - 1)), i - ref_win)
        ref = float(np.median(center[lo:i])) if i > lo else float(center[ref_from])
        d = center[i] - ref
        if abs(d) >= detect and i + hold <= L:
            w = center[i:i + hold]
            same_side = np.all(np.sign(w - ref) == np.sign(d)) and np.all(np.abs(w - ref) >= detect)
            if same_side and (w.max() - w.min()) < stable_spread and abs(w[-1] - w[0]) < max_drift:
                new = float(np.median(w))
                if round(new) != round(ref):
                    k = i
                    while k - 1 > seg_start and abs(center[k - 1] - new) < abs(center[k - 1] - ref):
                        k -= 1
                    bounds.append(k)
                    seg_start = k
                    # 경계(k)는 글라이드 한가운데일 수 있으므로, 새 음 높이는 도착한 뒤(i)부터 잰다
                    ref_from = i
                    i = i + hold
                    continue
        i += 1
    bounds.append(L)
    return bounds


def _evaluate(seg: _Seg, midi: np.ndarray, filled: np.ndarray, center: np.ndarray, conf: np.ndarray,
              voiced: np.ndarray) -> None:
    s, e = seg.s, seg.e
    n = e - s
    trim = int(n * 0.2) if n >= 10 else 0
    core = slice(s + trim, e - trim)
    vals = midi[core][voiced[core]]
    if vals.size == 0:
        vals = filled[s:e][np.isfinite(filled[s:e])]
    seg.est = float(np.median(vals)) if vals.size else float("nan")
    seg.pitch = int(round(seg.est)) if np.isfinite(seg.est) else 0
    v = voiced[s:e]
    conf_mean = float(conf[s:e][v].mean()) if v.any() else 0.0
    # 같은 음 병합은 짧은 무성 틈(중심선 NaN)을 넘어 합칠 수 있으므로 NaN은 빼고 잰다.
    # (v1은 np.std가 NaN을 내어 신뢰도가 NaN이 되었고, 실제 곡에서 프로젝트 저장이 실패했다)
    cc = center[core]
    cc = cc[np.isfinite(cc)]
    spread = float(np.std(cc)) if cc.size else 0.0
    stab = float(np.clip(1.0 - spread / 0.5, 0.0, 1.0))
    seg.conf = round(conf_mean * (0.5 + 0.5 * stab) * float(np.sqrt(v.mean())), 3)


def _vibrato(filled: np.ndarray, center: np.ndarray, s: int, e: int, fr: float) -> Vibrato | None:
    edge = max(int(0.08 * fr), int((e - s) * 0.15))
    a, b = s + edge, e - edge
    dur = (b - a) / fr
    if dur < 0.3:
        return None
    # 중앙값 중심선은 창이 주기와 딱 맞지 않으면 조금 따라 흔들리므로, 원래 곡선에서 직선 추세만 뺀다
    r = filled[a:b]
    r = r[np.isfinite(r)]
    if r.size < 10:
        return None
    x = np.arange(r.size)
    r = r - np.polyval(np.polyfit(x, r, 1), x)  # 느린 흐름(음정 처짐 등) 제거
    r = np.convolve(r, np.ones(3) / 3, mode="same")  # 프레임 잡음이 교차 횟수를 늘리지 않게
    sig = r[np.abs(r) > 0.05]
    if sig.size < 6:
        return None
    crossings = int(np.count_nonzero(np.diff(np.sign(sig)) != 0))
    rate = crossings / 2.0 / dur
    extent = float((np.percentile(r, 95) - np.percentile(r, 5)) / 2.0 * 100.0)  # 진폭(위아래 한쪽 폭)
    if 3.5 <= rate <= 9.0 and extent >= 20.0:
        return Vibrato(rate_hz=round(rate, 2), extent_cents=round(extent, 1))
    return None


def detect_notes(curve: PitchCurve, onset: np.ndarray | None, params: NoteDetectionParams) -> list[Note]:
    fr = curve.frame_rate
    midi = curve.midi().astype(np.float64)
    conf = np.asarray(curve.confidence, dtype=np.float64)
    n = len(midi)
    if n == 0:
        return []
    voiced = np.isfinite(midi) & (conf >= params.voiced_threshold)
    voiced &= (np.nan_to_num(midi, nan=-1e9) >= params.pitch_low - 0.5) & (np.nan_to_num(midi, nan=1e9) <= params.pitch_high + 0.5)
    blocked = _blocked_frames(params.no_vocal_ranges, curve.start_time, fr, n)
    voiced &= ~blocked
    island = max(2, int(round(MIN_ISLAND_S * fr)))
    for a, b in voiced_runs(voiced):  # 흩어진 잡음 프레임이 틈 잇기로 이어지지 않게
        if b - a < island:
            voiced[a:b] = False

    onset_ok = None
    if onset is not None and params.split_strength is not None and len(onset) == n:
        o = np.asarray(onset, dtype=np.float64)
        peak = np.r_[False, (o[1:-1] >= o[:-2]) & (o[1:-1] >= o[2:]), False] if n >= 3 else np.zeros(n, bool)
        onset_ok = peak & (o >= params.split_strength)

    bridged = voiced.copy()
    gap_max = int(round(params.bridge_gap_s * fr))
    runs = voiced_runs(voiced)
    for (a0, b0), (a1, _b1) in zip(runs, runs[1:]):
        if a1 - b0 <= gap_max:
            if onset_ok is not None and onset_ok[max(0, a1 - GAP_ONSET_BEFORE):a1 + GAP_ONSET_AFTER].any():
                continue  # 쉬었다가 다시 부름
            bridged[b0:a1] = True
    bridged &= ~blocked  # 짧은 보컬 없는 구간도 틈 잇기로 건너지 않는다
    filled = _fill_gaps(midi, voiced, bridged)

    center = np.full(n, np.nan)
    min_len = max(1, int(round(params.min_note_s * fr)))
    win = max(1, int(round(params.center_window_s * fr)))

    segs: list[_Seg] = []
    region_of: list[int] = []
    for r_idx, (a, b) in enumerate(voiced_runs(bridged)):
        size = min(win, b - a) | 1
        c = median_filter(filled[a:b], size=size, mode="nearest")
        center[a:b] = c
        bounds = _segment_region(c, params, fr)
        # 프레이즈가 뚜렷한 재발음으로 시작하면 앞 음과 같은 높이여도 합치지 않는다
        region_onset = onset_ok is not None and bool(onset_ok[max(0, a - GAP_ONSET_BEFORE):a + GAP_ONSET_AFTER].any())
        snapped = [False] * len(bounds)
        if onset_ok is not None:
            bounds, snapped = _snap_to_onsets(bounds, a, np.asarray(onset, dtype=np.float64), onset_ok, fr)
        for i_b, (s, e) in enumerate(zip(bounds, bounds[1:])):
            pieces = [(a + s, a + e, False)]
            if onset_ok is not None:
                pieces = _split_on_onsets(a + s, a + e, np.asarray(onset, dtype=np.float64), onset_ok, min_len)
            for ps, pe, onset_start in pieces:
                first = ps == a + s
                articulated = onset_start or (first and snapped[i_b]) or (ps == a and region_onset)
                segs.append(_Seg(ps, pe, articulated))
                region_of.append(r_idx)

    for seg in segs:
        _evaluate(seg, midi, filled, center, conf, voiced)
    orn_len = max(2, int(round(params.ornament_min_s * fr))) if params.ornament_min_s else None
    detect = max(0.5, params.change_semitones - DETECT_TOLERANCE)
    segs, region_of = _absorb_short(segs, region_of, min_len, midi, filled, center, conf, voiced, orn_len, detect)
    segs, region_of = _absorb_glides(segs, region_of, fr, midi, filled, center, conf, voiced)
    segs = _merge_same_pitch(segs, region_of, int(round(params.merge_gap_s * fr)), midi, filled, center, conf, voiced,
                             blocked)
    if params.ornament_min_s:
        segs = _carve_ornaments(segs, params, fr, min_len, midi, filled, center, conf, voiced)
    if params.merge_approach and onset is not None and len(onset) == n:
        syl = np.sort(np.asarray([(t - curve.start_time) * fr for t in params.syllable_starts], dtype=np.float64))
        segs = _absorb_approaches(segs, np.asarray(onset, dtype=np.float64), fr, midi, filled, center, conf, voiced,
                                  syl)

    notes = []
    for seg in segs:
        if not (params.pitch_low <= seg.pitch <= params.pitch_high):
            continue
        if voiced[seg.s:seg.e].mean() < MIN_VOICED_FRACTION:
            continue  # 대부분 틈 잇기로 메운 구간 = 실제로 노래한 음이 아니다
        notes.append(Note(
            id=new_id(),
            start=round(curve.start_time + seg.s / fr, 4),
            end=round(curve.start_time + seg.e / fr, 4),
            pitch=seg.pitch,
            confidence=seg.conf,
            vibrato=None if seg.ornament else _vibrato(filled, center, seg.s, seg.e, fr),
            ornament=seg.ornament,
        ))
    return notes


def _snap_to_onsets(bounds: list[int], offset: int, onset: np.ndarray, ok: np.ndarray,
                    fr: float) -> tuple[list[int], list[bool]]:
    """음높이 변화로 찾은 경계 근처에 뚜렷한 온셋(발성 시작)이 있으면 경계를 그리로 옮긴다.

    음높이 중간점은 앞 음에서 미끄러져 올라가는 도중이라 실제 발성보다 이를 수 있다.
    두 번째 값은 경계마다 온셋으로 옮겼는지(= 발성이 확인된 경계인지)다.
    """
    before, after = int(SNAP_BEFORE_S * fr), int(SNAP_AFTER_S * fr)
    out = list(bounds)
    snapped = [False] * len(out)
    for i in range(1, len(out) - 1):
        k = offset + out[i]
        lo = max(offset + out[i - 1] + 1, k - before)
        hi = min(offset + out[i + 1] - 1, k + after)
        if hi <= lo:
            continue
        cand = [j for j in range(lo, hi) if ok[j]]
        if cand:
            out[i] = max(cand, key=lambda j: onset[j]) - offset
            snapped[i] = True
    return out, snapped


def _split_on_onsets(s: int, e: int, onset: np.ndarray, ok: np.ndarray, min_len: int) -> list[tuple[int, int, bool]]:
    """같은 높이 구간 안의 뚜렷한 온셋(재발음)에서 나눈다.

    구간 양 끝 가까이의 온셋은 그 음 자신의 시작·끝이므로 나누지 않는다.
    """
    edge = min_len
    if e - s < 2 * edge:
        return [(s, e, False)]
    cuts = []
    last = s
    for k in range(s + edge, e - edge + 1):
        if ok[k] and k - last >= min_len:
            cuts.append(k)
            last = k
    if not cuts:
        return [(s, e, False)]
    edges = [s] + cuts + [e]
    return [(edges[i], edges[i + 1], i > 0) for i in range(len(edges) - 1)]


def _absorb_short(segs, region_of, min_len, midi, filled, center, conf, voiced, orn_len=None, detect=0.7):
    """최소 길이보다 짧은 조각을 가까운 높이의 이웃에 합치거나, 이웃이 없으면 버린다.

    orn_len이 있으면(표준·정밀) 합칠 이웃과의 경계가 뚜렷한 재발음이고, 길이가 orn_len 이상이며,
    높이가 이웃과 기준 이상 다른 조각은 장식음으로 남긴다. 따로 발음한 꾸밈음을 지우지 않기 위해서다.
    """
    segs = list(segs)
    region_of = list(region_of)
    while True:
        short = [i for i, g in enumerate(segs) if (g.e - g.s) < min_len and not g.ornament]
        if not short:
            break
        i = min(short, key=lambda j: segs[j].e - segs[j].s)
        g = segs[i]
        nbrs = [j for j in (i - 1, i + 1)
                if 0 <= j < len(segs) and region_of[j] == region_of[i] and (segs[j].e == g.s or segs[j].s == g.e)]
        if not nbrs:
            del segs[i], region_of[i]
            continue
        j = min(nbrs, key=lambda k: abs(segs[k].est - g.est))
        # 두 이웃 사이 높이에 있는 짧은 조각은 꾸밈음보다 음과 음 사이를 미끄러지는 중간일 가능성이 크다
        between = len(nbrs) == 2 and min(segs[k].est for k in nbrs) < g.est < max(segs[k].est for k in nbrs)
        if (orn_len is not None and (g.e - g.s) >= orn_len and not between
                and all(abs(segs[k].est - g.est) >= detect for k in nbrs)):
            articulated = segs[j].onset_start if j > i else g.onset_start
            if articulated:
                g.ornament = True
                continue
        tgt = segs[j]
        tgt.s, tgt.e = min(tgt.s, g.s), max(tgt.e, g.e)
        if j > i:
            tgt.onset_start = g.onset_start
        _evaluate(tgt, midi, filled, center, conf, voiced)
        del segs[i], region_of[i]
    return segs, region_of


def _is_glide_piece(g: _Seg, direction: float, filled: np.ndarray, fr: float) -> bool:
    if g.ornament or (g.e - g.s) > GLIDE_PIECE_MAX_S * fr or g.e - g.s < 2:
        return False
    trim = max(1, (g.e - g.s) // 5)  # 경계 프레임은 앞뒤 음으로 넘어가는 중이라 빼고 본다
    seg = filled[g.s + trim:g.e - trim]
    seg = seg[np.isfinite(seg)]
    if seg.size < 3:
        return False
    seg = median_filter(seg, size=3, mode="nearest")
    return (seg[-1] - seg[0]) * direction >= GLIDE_MIN_DRIFT


def _absorb_glides(segs, region_of, fr, midi, filled, center, conf, voiced, max_pieces: int = 4):
    segs, region_of = list(segs), list(region_of)
    i = 0
    while i < len(segs):
        p = segs[i]
        for size in range(max_pieces, 0, -1):  # 긴 묶음부터 본다
            j = i + size + 1  # q = segs[j], 미끄러짐 조각 = segs[i+1:j]
            if j >= len(segs) or any(region_of[k] != region_of[i] or segs[k].s != segs[k - 1].e for k in range(i + 1, j + 1)):
                continue
            q, run = segs[j], segs[i + 1:j]
            lo, hi = sorted((p.est, q.est))
            direction = float(np.sign(q.est - p.est))
            # 첫 조각이 재발음으로 시작하면 미끄러짐이 아니라 따로 부른 음이다
            if (q.pitch == p.pitch or run[0].onset_start or not all(lo < g.est < hi for g in run)
                    or run[-1].e - run[0].s > GLIDE_RUN_MAX_S * fr
                    or not all(_is_glide_piece(g, direction, filled, fr) for g in run)):
                continue
            cut = next((g.s for g in run if g.onset_start), None)
            if cut is None:
                mid = (p.est + q.est) / 2
                cut = next((k for k in range(run[0].s, run[-1].e)
                            if np.isfinite(filled[k]) and (filled[k] - mid) * direction >= 0), run[-1].e)
            q.onset_start = q.onset_start or any(g.onset_start for g in run)
            p.e, q.s = cut, cut
            _evaluate(p, midi, filled, center, conf, voiced)
            _evaluate(q, midi, filled, center, conf, voiced)
            del segs[i + 1:j], region_of[i + 1:j]
            break
        i += 1
    return segs, region_of


def _toward(run_pitch: int, own: int, neighbor: int) -> bool:
    """run_pitch가 이웃 음 높이이거나, 자기 음과 이웃 음 사이에 있다."""
    return run_pitch == neighbor or min(own, neighbor) < run_pitch < max(own, neighbor)


def _carve_ornaments(segs, params, fr, min_len, midi, filled, center, conf, voiced):
    """음표 안의 짧고 또렷한 이탈 구간을 떼어 낸다.

    중심선(중앙값 필터)은 창 길이의 절반보다 짧은 음을 지워 버리므로, 그런 음도 여기서 되살린다.
    최소 길이 미만이면 장식음(ornament), 이상이면 일반 음표가 된다.
    """
    orn_len = max(2, int(round(params.ornament_min_s * fr)))
    max_len = max(min_len, int(round(params.center_window_s * fr)))
    detect = max(0.5, params.change_semitones - DETECT_TOLERANCE)
    out: list[_Seg] = []
    for idx, g in enumerate(segs):
        prev_pitch = segs[idx - 1].pitch if idx > 0 and segs[idx - 1].e == g.s else None
        next_pitch = segs[idx + 1].pitch if idx + 1 < len(segs) and segs[idx + 1].s == g.e else None
        vib = _vibrato(filled, center, g.s, g.e, fr)
        if vib is not None and vib.extent_cents / 100.0 >= detect - 0.15:
            out.append(g)  # 비브라토 꼭대기·골짜기를 장식음으로 오인하지 않도록
            continue
        raw = filled[g.s:g.e]
        smooth = median_filter(raw, size=3, mode="nearest") if raw.size >= 3 else raw
        dev = smooth - g.est
        mask = (np.abs(dev) >= detect) & np.isfinite(dev)
        pieces = []
        for a, b in voiced_runs(mask):
            run = smooth[a:b]
            if not (orn_len <= b - a < max_len) or (run.max() - run.min()) >= 0.5:
                continue
            run_pitch = round(float(np.median(run)))
            if run_pitch == g.pitch:
                continue
            # 가장자리 조각이 이웃 음과 같은 높이면 경계가 조금 어긋난 것이지 새 음이 아니다.
            # 이웃 음 쪽으로 가는 중간 높이여도 마찬가지다(음과 음 사이를 미끄러지는 중).
            if a == 0 and prev_pitch is not None and _toward(run_pitch, g.pitch, prev_pitch):
                continue
            if b == len(raw) and next_pitch is not None and _toward(run_pitch, g.pitch, next_pitch):
                continue
            probe = _Seg(g.s + a, g.s + b)
            _evaluate(probe, midi, filled, center, conf, voiced)
            short = (b - a) < min_len
            if short and probe.conf < params.ornament_conf:
                continue
            probe.ornament = short
            pieces.append(probe)
        if not pieces:
            out.append(g)
            continue
        cursor = g.s
        for orn in pieces:
            if orn.s - cursor >= 2:
                part = _Seg(cursor, orn.s, g.onset_start if cursor == g.s else False)
                _evaluate(part, midi, filled, center, conf, voiced)
                out.append(part)
            else:
                orn.s = cursor  # 아주 짧은 자투리는 장식음에 붙인다
            out.append(orn)
            cursor = orn.e
        if g.e - cursor >= 2:
            tail = _Seg(cursor, g.e)
            _evaluate(tail, midi, filled, center, conf, voiced)
            out.append(tail)
        elif out[-1].ornament:
            out[-1].e = g.e
    return out


def approach_slope(filled: np.ndarray, s: int, e: int, fr: float) -> tuple[float, float]:
    """(기울기 반음/초, 처음→끝 상승 반음). 구간 안 유성 프레임에 직선을 맞춘다.

    끝쪽 1/5은 뒤 음으로 넘어가는 경계라 뺀다(_is_glide_piece와 같은 생각). 평평하게 머물다 뛰어오른 음이
    경계 프레임 때문에 올라가는 것처럼 보이지 않게 한다.
    """
    e = e - max(1, (e - s) // 5)
    y = filled[s:e]
    x = np.arange(len(y), dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return 0.0, 0.0
    k, c = np.polyfit(x[ok], y[ok], 1)
    return float(k * fr), float(k * (x[ok][-1] - x[ok][0]))


def longest_plateau(filled: np.ndarray, s: int, e: int, band: float = APPROACH_PLATEAU_BAND) -> int:
    """[s, e) 안에서 높이가 band 폭 안에 머문 가장 긴 연속 프레임 수."""
    y = filled[s:e]
    best = 0
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            continue
        lo = hi = y[i]
        j = i
        while j < len(y) and np.isfinite(y[j]):
            lo, hi = min(lo, y[j]), max(hi, y[j])
            if hi - lo > band:
                break
            j += 1
        best = max(best, j - i)
    return best


def is_approach(a: _Seg, b: _Seg, prev_end: int | None, onset: np.ndarray, filled: np.ndarray, fr: float) -> bool:
    """a가 b로 올라가는 접근 구간(끌어올림)인가. 규칙은 모듈 설명 9번."""
    if a.ornament or b.ornament:
        return False
    if not (prev_end is None or prev_end < a.s or a.onset_start):
        return False  # 앞 음에서 이어지는 중간 음은 미끄러짐 흡수(6번)가 다룬다
    if (a.e - a.s) > APPROACH_MAX_S * fr or (b.e - b.s) < (a.e - a.s):
        return False
    if b.s - a.e > APPROACH_GAP_S * fr or b.s < a.e:
        return False
    interval = b.pitch - a.pitch
    if not (1 <= interval <= APPROACH_MAX_INTERVAL):
        return False
    slope, rise = approach_slope(filled, a.s, a.e, fr)
    if slope < APPROACH_MIN_SLOPE:
        return False
    if interval >= APPROACH_BIG_INTERVAL and rise < 1.0:
        return False
    if longest_plateau(filled, a.s, a.e) >= APPROACH_PLATEAU_S * fr:
        return False
    lo, hi = max(0, b.s - APPROACH_ONSET_WINDOW[0]), min(len(onset), b.s + APPROACH_ONSET_WINDOW[1])
    return not (hi > lo and float(np.nanmax(onset[lo:hi])) >= APPROACH_ONSET_MAX)


def _new_syllable_at(syl: np.ndarray, a: int, b: int, tol: int) -> bool:
    """b 근처(±tol)에서 시작하는 음절 중, a 근처에서 시작한 그 음절이 아닌 것이 있는가."""
    if syl.size == 0:
        return False
    lo, hi = np.searchsorted(syl, b - tol), np.searchsorted(syl, b + tol, side="right")
    return any(abs(t - a) > tol for t in syl[lo:hi])


def _absorb_approaches(segs, onset, fr, midi, filled, center, conf, voiced, syllable_frames=None):
    syl = syllable_frames if syllable_frames is not None else np.zeros(0)
    tol = int(round(APPROACH_SYLLABLE_TOL * fr))
    out: list[_Seg] = []
    i = 0
    while i < len(segs):
        a = segs[i]
        if i + 1 < len(segs):
            b = segs[i + 1]
            prev_end = out[-1].e if out else None
            # 뒤 음에서 다른 음절이 시작하면 합치지 않는다
            if not _new_syllable_at(syl, a.s, b.s, tol) and is_approach(a, b, prev_end, onset, filled, fr):
                pitch, est = b.pitch, b.est
                merged = _Seg(a.s, b.e, a.onset_start)
                _evaluate(merged, midi, filled, center, conf, voiced)
                merged.pitch, merged.est = pitch, est  # 높이는 목표 음
                out.append(merged)
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def _merge_same_pitch(segs, region_of, gap_max, midi, filled, center, conf, voiced, blocked=None):
    out: list[_Seg] = []
    out_region: list[int] = []
    for g, r in zip(segs, region_of):
        if out and not g.onset_start and g.pitch == out[-1].pitch:
            prev = out[-1]
            if blocked is not None and blocked[prev.e:g.s].any():  # 보컬 없는 구간을 넘어 합치지 않는다
                out.append(g)
                out_region.append(r)
                continue
            same_region = out_region[-1] == r and prev.e == g.s
            if same_region or g.s - prev.e <= gap_max:
                prev.e = g.e
                _evaluate(prev, midi, filled, center, conf, voiced)
                continue
        out.append(g)
        out_region.append(r)
    return out
