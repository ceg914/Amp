"""피치 곡선 → 음표 (Qt 비의존, numpy·scipy).

순서
1. 유성 판정: f0가 있고 유성 확률이 임계 이상이며 분석 음역 안에 있는 프레임.
   짧은 무성 틈(자음·피치 놓침)은 같은 프레이즈로 잇는다.
2. 중심선: 약 비브라토 한 주기 길이의 중앙값 필터. 비브라토는 상쇄되고, 음 사이 경계는 날카롭게,
   포르타멘토는 경사로 남는다.
3. 변화점: 중심선이 현재 음에서 기준(기본 1반음, 허용 오차 0.3) 이상 벗어나고, 새 높이가 hold 시간
   동안 안정적으로 유지될 때만 나눈다. 경계는 두 음의 중간을 지나는 지점이다.
   그래서 포르타멘토는 가운데에서 한 번만 나뉘고, 중간 음들이 따로 음표가 되지 않는다.
4. 온셋: 음높이 경계 근처에 뚜렷한 온셋(발성 시작)이 있으면 경계를 그리로 옮기고,
   같은 높이가 이어지는 구간 안의 뚜렷한 온셋에서는 재발음으로 보고 나눈다(정밀·표준 민감도).
5. 짧은 조각: 최소 길이보다 짧으면 가까운 높이의 이웃에 합치고, 이웃이 없으면 버린다.
6. 틈이 아주 짧은 같은 높이의 이웃 음표를 합친다.
7. 짧은 이탈(표준·정밀): 음표 안에서 중심선 창보다 짧게 기준 이상 벗어났다 돌아오는 구간을 떼어 낸다.
   최소 길이 미만이면서 신뢰도가 높으면 장식음(ornament), 최소 길이 이상이면 일반 음표가 된다.
   비브라토 폭이 기준에 가까운 음은 건너뛴다.
8. 음표별 비브라토(3.5~9Hz, 20cent 이상)를 추정한다. 음의 앞뒤 가장자리는 빼고 본다.
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
SNAP_BEFORE_S, SNAP_AFTER_S = 0.04, 0.15  # 음높이 경계를 온셋으로 옮길 수 있는 범위
EDGE_GUARD_FRAMES = 15  # 재발음 분할은 구간 끝에서 150ms(100fps 기준) 이상 떨어진 온셋에서만
REF_WINDOW_S = 0.5  # 현재 음 높이를 추정할 때 볼 최근 구간


@dataclass
class _Seg:
    s: int
    e: int
    onset_start: bool = False  # 온셋으로 나뉜 시작점 (같은 음이어도 합치지 않는다)
    pitch: int = 0
    est: float = 0.0
    conf: float = 0.0
    ornament: bool = False


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad + 1e-9)


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
    spread = float(np.std(center[core])) if (e - trim) > (s + trim) else 0.0
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
    island = max(2, int(round(MIN_ISLAND_S * fr)))
    for a, b in voiced_runs(voiced):  # 흩어진 잡음 프레임이 틈 잇기로 이어지지 않게
        if b - a < island:
            voiced[a:b] = False

    bridged = voiced.copy()
    gap_max = int(round(params.bridge_gap_s * fr))
    runs = voiced_runs(voiced)
    for (a0, b0), (a1, _b1) in zip(runs, runs[1:]):
        if a1 - b0 <= gap_max:
            bridged[b0:a1] = True
    filled = _fill_gaps(midi, voiced, bridged)

    center = np.full(n, np.nan)
    onset_ok = None
    if onset is not None and params.onset_split_z is not None and len(onset) == n and bridged.any():
        o = np.asarray(onset, dtype=np.float64)
        ov = o[bridged]
        z = _robust_z(o)
        # 표준점수만 보면 무음이 대부분인 곡에서 지속음의 작은 흔들림도 커 보인다.
        # 그래서 노래하는 구간의 강한 온셋(상위 1%)에 견줄 만한 피크만 재발음으로 본다.
        strong = (params.onset_split_rel or 0.0) * float(np.percentile(ov, 99))
        onset_ok = (z >= params.onset_split_z) & (o >= strong)
    min_len = max(1, int(round(params.min_note_s * fr)))
    win = max(1, int(round(params.center_window_s * fr)))

    segs: list[_Seg] = []
    region_of: list[int] = []
    for r_idx, (a, b) in enumerate(voiced_runs(bridged)):
        size = min(win, b - a) | 1
        c = median_filter(filled[a:b], size=size, mode="nearest")
        center[a:b] = c
        bounds = _segment_region(c, params, fr)
        if onset_ok is not None:
            bounds = _snap_to_onsets(bounds, a, np.asarray(onset, dtype=np.float64), onset_ok, fr)
        for s, e in zip(bounds, bounds[1:]):
            pieces = [(a + s, a + e, False)]
            if onset_ok is not None:
                pieces = _split_on_onsets(a + s, a + e, np.asarray(onset, dtype=np.float64), onset_ok, min_len)
            for ps, pe, onset_start in pieces:
                segs.append(_Seg(ps, pe, onset_start))
                region_of.append(r_idx)

    for seg in segs:
        _evaluate(seg, midi, filled, center, conf, voiced)
    segs, region_of = _absorb_short(segs, region_of, min_len, midi, filled, center, conf, voiced)
    segs = _merge_same_pitch(segs, region_of, int(round(params.merge_gap_s * fr)), midi, filled, center, conf, voiced)
    if params.ornament_min_s:
        segs = _carve_ornaments(segs, params, fr, min_len, midi, filled, center, conf, voiced)

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


def _snap_to_onsets(bounds: list[int], offset: int, onset: np.ndarray, ok: np.ndarray, fr: float) -> list[int]:
    """음높이 변화로 찾은 경계 근처에 뚜렷한 온셋(발성 시작)이 있으면 경계를 그리로 옮긴다.

    음높이 중간점은 앞 음에서 미끄러져 올라가는 도중이라 실제 발성보다 이를 수 있다.
    """
    before, after = int(SNAP_BEFORE_S * fr), int(SNAP_AFTER_S * fr)
    out = list(bounds)
    for i in range(1, len(out) - 1):
        k = offset + out[i]
        lo = max(offset + out[i - 1] + 1, k - before)
        hi = min(offset + out[i + 1] - 1, k + after)
        if hi <= lo:
            continue
        cand = [j for j in range(lo, hi) if ok[j]]
        if cand:
            out[i] = max(cand, key=lambda j: onset[j]) - offset
    return out


def _split_on_onsets(s: int, e: int, onset: np.ndarray, ok: np.ndarray, min_len: int) -> list[tuple[int, int, bool]]:
    """같은 높이 구간 안의 뚜렷한 온셋(재발음)에서 나눈다.

    구간 양 끝 가까이의 온셋은 그 음 자신의 시작·끝이므로 나누지 않는다.
    """
    edge = max(min_len, EDGE_GUARD_FRAMES)
    if e - s < 2 * edge:
        return [(s, e, False)]
    local_med = float(np.median(onset[s:e]))
    cuts = []
    last = s
    for k in range(s + edge, e - edge):
        if not ok[k] or onset[k] < 3.0 * local_med:
            continue
        if onset[k] >= onset[k - 1] and onset[k] >= onset[k + 1] and k - last >= min_len:
            cuts.append(k)
            last = k
    if not cuts:
        return [(s, e, False)]
    edges = [s] + cuts + [e]
    return [(edges[i], edges[i + 1], i > 0) for i in range(len(edges) - 1)]


def _absorb_short(segs, region_of, min_len, midi, filled, center, conf, voiced):
    """최소 길이보다 짧은 조각을 가까운 높이의 이웃에 합치거나, 이웃이 없으면 버린다."""
    segs = list(segs)
    region_of = list(region_of)
    while True:
        short = [i for i, g in enumerate(segs) if (g.e - g.s) < min_len]
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
        tgt = segs[j]
        tgt.s, tgt.e = min(tgt.s, g.s), max(tgt.e, g.e)
        if j > i:
            tgt.onset_start = g.onset_start
        _evaluate(tgt, midi, filled, center, conf, voiced)
        del segs[i], region_of[i]
    return segs, region_of


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
            # 가장자리 조각이 이웃 음과 같은 높이면 경계가 조금 어긋난 것이지 새 음이 아니다
            if (a == 0 and run_pitch == prev_pitch) or (b == len(raw) and run_pitch == next_pitch):
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


def _merge_same_pitch(segs, region_of, gap_max, midi, filled, center, conf, voiced):
    out: list[_Seg] = []
    out_region: list[int] = []
    for g, r in zip(segs, region_of):
        if out and not g.onset_start and g.pitch == out[-1].pitch:
            prev = out[-1]
            same_region = out_region[-1] == r and prev.e == g.s
            if same_region or g.s - prev.e <= gap_max:
                prev.e = g.e
                _evaluate(prev, midi, filled, center, conf, voiced)
                continue
        out.append(g)
        out_region.append(r)
    return out
