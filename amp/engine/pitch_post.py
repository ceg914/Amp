"""pYIN 결과 후처리 (Qt 비의존, numpy·scipy).

1. 에너지 게이트: 보컬 트랙이 아주 조용한 프레임(분리 잔여음, 호흡 끝)은 무성으로 본다.
   20ms RMS가 있으면 주변 ±250ms 최대 음량보다 30dB 넘게 낮은 프레임도 무성으로 본다.
   pYIN의 HMM은 40ms 남짓한 음절 사이 틈을 유성으로 이어 버리는데, 이 게이트가 그 틈을 끊는다.
2. 짧은 유성 조각 제거: 40ms 미만으로 튀어나온 유성 구간은 잡음으로 본다.
3. 옥타브 오류 보정: 시간상 ±175ms 안 유성 프레임의 중앙값과 약 ±12반음 차이 나는 프레임을 되돌린다.
   구간 전체가 앞뒤 구간과 한 옥타브 어긋난 경우도 구간 단위로 보정한다.
4. 튐 제거: 5프레임 중앙값에서 1.5반음 넘게 벗어난 프레임을 중앙값으로 바꾼다.
5. 무성 경계를 넘지 않는 3프레임 중앙값 평활.

무성 프레임은 끝까지 NaN으로 남긴다. 화면에서 곡선을 끊어 그리는 기준이 된다.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from amp.core.model import PitchCurve
from amp.core.pitch import hz_to_midi, midi_to_hz

MIN_ISLAND_S = 0.04
OCTAVE_WINDOW_S = 0.35
OCTAVE_TOL = 1.5  # 반음
SPIKE_TOL = 1.5  # 반음
ENERGY_FLOOR_DB = -60.0  # 절대 하한 (dBFS)
ENERGY_REL_DB = -40.0  # 곡 안 최대 RMS 대비 하한
GAP_DB = 30.0  # 20ms RMS가 주변 최대보다 이만큼 낮으면 소리 사이 틈
GAP_WINDOW_S = 0.25
DIP_MIN_DB, DIP_FULL_DB = 3.0, 12.0  # 재발음 지표: 이 깊이부터 반영하기 시작해 이 깊이에서 온셋을 그대로 쓴다
DIP_REF_S = 0.15  # 하락 전후 음량을 볼 범위
DIP_RECENT_S = 0.07  # 온셋 직전 최저 음량을 볼 범위
# v0.6 (음표 검출 v4): 음량은 거의 그대로인데 자음 때문에 유성 확률이 잠깐 떨어지는 재발음.
# 온셋 ±VOICING_DIP_S 안의 최저 유성 확률이 0.5 안팎이면 음량 하락과 같은 근거로 본다 (0.45 이하 = 그대로, 0.55 이상 = 없음).
# 경계값은 탐색하지 않았다 (확률의 가운데 값). 측정(보고서 7장): 합성 -7dB 하락 재발음(h_dip7) F1 0.40 → 1.00,
# 합성 24사례 중 나빠진 사례 없음. 실제 두 곡(악보 정답)은 표준·단순에서 v3와 같은 결과, 정밀에서 F1 +0.002.
# 실제 곡의 같은 음 반복 누락은 대부분 온셋 근거 자체가 약했다(온셋 피크 1~3).
# 프레이즈 시작(무성 → 유성)은 하락이 아니다: 앞뒤 DIP_REF_S 안의 최대 유성 확률이 둘 다 VOICING_AROUND 이상일 때만
# 쓴다. 이 조건이 없으면 프레이즈 첫머리 끌어올림이 '재발음'으로 보여 접근음 흡수가 막혔다 (여래아 2곳에서 확인).
VOICING_DIP_S = 0.06
VOICING_DIP_LOW, VOICING_DIP_HIGH = 0.45, 0.55
VOICING_AROUND_LOW, VOICING_AROUND_HIGH = 0.6, 0.8
# 같은 높이 재발음에만 쓴다: 앞뒤(하락 창 바깥) 90ms 중앙 음높이 차가 이 값 미만일 때. 다른 높이로 넘어가는 자리에서도
# 유성 확률이 잠깐 떨어져, 이 조건이 없으면 정밀 민감도에서 음 끝에 70ms 같은 높이 조각이 생겼다 (합성 음계·포르타멘토).
VOICING_SAME_PITCH = 0.5


def voiced_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 구간의 [시작, 끝) 목록."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def energy_gate(rms: np.ndarray) -> np.ndarray:
    """소리가 충분히 큰 프레임이면 True."""
    rms = np.asarray(rms, dtype=np.float64)
    if rms.size == 0 or rms.max() <= 0:
        return np.zeros(rms.shape, dtype=bool)
    floor = max(10 ** (ENERGY_FLOOR_DB / 20), rms.max() * 10 ** (ENERGY_REL_DB / 20))
    return rms >= floor


def _db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.asarray(x, dtype=np.float64) + 1e-7)


def gap_gate(rms_fine: np.ndarray, frame_rate: float) -> np.ndarray:
    """주변보다 GAP_DB 넘게 조용하지 않은 프레임이면 True (= 틈이 아니다)."""
    from scipy.ndimage import maximum_filter1d

    db = _db(rms_fine)
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    w = max(1, int(GAP_WINDOW_S * frame_rate))
    return db >= maximum_filter1d(db, size=2 * w + 1, mode="nearest") - GAP_DB


def _fix_octaves(midi: np.ndarray, frame_rate: float) -> np.ndarray:
    voiced = np.isfinite(midi)
    if voiced.sum() < 3:
        return midi
    idx = np.flatnonzero(voiced)
    vals = midi[idx]
    win = max(3, int(OCTAVE_WINDOW_S * frame_rate) | 1)
    # 기준 = 시간상 ±창/2 안의 유성 프레임 중앙값. (v1은 유성 프레임만 이어 붙인 목록에서 창을 잡아,
    # 쉼 뒤 새 프레이즈 첫 프레임이 몇 초 전 프레이즈 높이에 끌려 옥타브가 틀어졌다 — 실제 곡에서 발견)
    from numpy.lib.stride_tricks import sliding_window_view

    half = win // 2
    padded = np.pad(midi, half, constant_values=np.nan)
    ref = np.nanmedian(sliding_window_view(padded, win)[idx], axis=1)  # 자기 자신이 있어 빈 창은 없다
    d = vals - ref
    k = np.round(d / 12.0)
    fix = (k != 0) & (np.abs(d - 12.0 * k) < OCTAVE_TOL)
    vals = vals - 12.0 * k * fix
    out = midi.copy()
    out[idx] = vals
    # 구간 단위: 짧은 구간의 중앙값이 앞뒤 구간과 한 옥타브 어긋나면 옮긴다
    runs = voiced_runs(np.isfinite(out))
    meds = [float(np.median(out[a:b])) for a, b in runs]
    for i, (a, b) in enumerate(runs):
        neighbors = [meds[j] for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        if not neighbors or (b - a) > 1.0 * frame_rate:
            continue
        ref_n = float(np.median(neighbors))
        k = round((meds[i] - ref_n) / 12.0)
        if k != 0 and abs(meds[i] - ref_n - 12.0 * k) < OCTAVE_TOL:
            out[a:b] -= 12.0 * k
    return out


def clean_midi(midi: np.ndarray, frame_rate: float, gate: np.ndarray | None = None,
               fix_octaves: bool = True) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float64).copy()
    if gate is not None:
        midi[~gate] = np.nan
    min_island = max(1, int(round(MIN_ISLAND_S * frame_rate)))
    for a, b in voiced_runs(np.isfinite(midi)):
        if b - a < min_island:
            midi[a:b] = np.nan
    if fix_octaves:
        midi = _fix_octaves(midi, frame_rate)
    for a, b in voiced_runs(np.isfinite(midi)):
        seg = midi[a:b]
        if seg.size >= 5:
            med = median_filter(seg, size=5, mode="nearest")
            spike = np.abs(seg - med) > SPIKE_TOL
            seg = np.where(spike, med, seg)
        if seg.size >= 3:
            seg = median_filter(seg, size=3, mode="nearest")
        midi[a:b] = seg
    return midi


def build_curve(f0_hz: np.ndarray, voiced_prob: np.ndarray, rms: np.ndarray | None,
                frame_rate: float, start_time: float = 0.0, rms_fine: np.ndarray | None = None,
                fix_octaves: bool = True) -> PitchCurve:
    """작업 프로세스가 낸 원시 결과를 정리해 PitchCurve로 만든다.

    fix_octaves=False: 주변 높이와 한 옥타브 어긋난 구간을 옮기지 않는다 (베이스는 옥타브 도약이 흔하다, v0.6.3)."""
    midi = hz_to_midi(np.asarray(f0_hz, dtype=np.float64))
    gate = energy_gate(rms) if rms is not None and len(rms) == len(midi) else None
    if rms_fine is not None and len(rms_fine) == len(midi):
        gap = gap_gate(rms_fine, frame_rate)
        gate = gap if gate is None else gate & gap
    clean = clean_midi(midi, frame_rate, gate, fix_octaves)
    conf = np.where(np.isfinite(clean), np.asarray(voiced_prob, dtype=np.float32), 0.0).astype(np.float32)
    f0 = np.where(np.isfinite(clean), midi_to_hz(np.nan_to_num(clean)), np.nan).astype(np.float32)
    return PitchCurve(start_time, frame_rate, f0, conf)


def articulation_envelope(onset: np.ndarray, rms: np.ndarray, frame_rate: float,
                          rms_fine: np.ndarray | None = None, voiced_prob: np.ndarray | None = None,
                          f0_hz: np.ndarray | None = None) -> np.ndarray:
    """재발음 지표 = 온셋 강도 × (직전 음량 하락 깊이 비율).

    실제로 다시 부르는 지점은 자음·끊김 때문에 음량이 잠깐 떨어졌다가 오른다. 비브라토만 있는 지속음은
    스펙트럼이 흔들려 온셋 피크가 생겨도 음량은 거의 떨어지지 않는다(분리된 보컬에서 확인).
    하락 깊이 = min(하락 전 최대, 하락 뒤 최대) − 직전 최저 (dB). DIP_MIN_DB 미만이면 0,
    DIP_FULL_DB 이상이면 온셋을 그대로 쓴다. 20ms RMS가 있으면 그것으로 깊이를 잰다.
    온셋 강도는 멜 대역 평균 dB 증가량이라 곡 음량과 무관하다.

    voiced_prob과 f0_hz를 주면(음표 검출 v4) 앞뒤가 같은 높이인 곳에서 유성 확률이 잠깐 떨어진 지점도 근거로 쓴다:
    가중치 = max(음량 하락, 유성 하락). 주지 않으면 v3와 같다 (비교·이전 결과 재현용).
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    onset = np.asarray(onset, dtype=np.float64)
    env = rms_fine if rms_fine is not None and len(rms_fine) == len(onset) else rms
    env = np.asarray(env, dtype=np.float64)
    if onset.shape != env.shape or onset.size == 0:
        return onset.astype(np.float32)
    db = _db(env)
    w = max(3, int(DIP_REF_S * frame_rate))
    r = max(3, int(DIP_RECENT_S * frame_rate))
    # origin > 0이면 창이 과거 쪽으로, < 0이면 미래 쪽으로 밀린다
    before = maximum_filter1d(db, size=w, origin=w // 2, mode="nearest")  # [k−w+1, k]
    after = maximum_filter1d(db, size=w, origin=-(w // 2), mode="nearest")  # [k, k+w−1]
    recent_min = minimum_filter1d(db, size=r, origin=r // 2 - 1, mode="nearest")  # [k−r+2, k+1]
    depth = np.minimum(before, after) - recent_min
    weight = np.clip((depth - DIP_MIN_DB) / (DIP_FULL_DB - DIP_MIN_DB), 0.0, 1.0)
    if voiced_prob is not None and f0_hz is not None and len(voiced_prob) == len(onset) == len(f0_hz):
        vp = np.asarray(voiced_prob, dtype=np.float64)
        k = max(1, int(round(VOICING_DIP_S * frame_rate)))
        vmin = minimum_filter1d(vp, size=2 * k + 1, mode="nearest")
        # 하락 창(±k) 바깥의 앞·뒤 w 프레임: 앞 = [i−k−w, i−k−1], 뒤 = [i+k+1, i+k+w]
        vb = maximum_filter1d(vp, size=w, origin=w // 2, mode="nearest")  # [i−w+1, i]
        va = maximum_filter1d(vp, size=w, origin=-(w // 2), mode="nearest")  # [i, i+w−1]
        vbefore = np.r_[np.full(k + 1, vb[0]), vb[:-(k + 1)]] if len(vb) > k + 1 else np.full_like(vb, vb[0])
        vafter = np.r_[va[k + 1:], np.full(k + 1, va[-1])] if len(va) > k + 1 else np.full_like(va, va[-1])
        around = np.clip((np.minimum(vbefore, vafter) - VOICING_AROUND_LOW) / (VOICING_AROUND_HIGH - VOICING_AROUND_LOW),
                         0.0, 1.0)
        from scipy.ndimage import median_filter

        m = hz_to_midi(np.asarray(f0_hz, dtype=np.float64))
        sm = median_filter(np.nan_to_num(m, nan=-1000.0), size=max(3, int(round(0.09 * frame_rate))) | 1, mode="nearest")
        sh = k + max(1, int(round(0.05 * frame_rate)))
        mb = np.r_[np.full(sh, -1000.0), sm[:-sh]] if len(sm) > sh else np.full_like(sm, -1000.0)
        ma = np.r_[sm[sh:], np.full(sh, -1000.0)] if len(sm) > sh else np.full_like(sm, -1000.0)
        same = (mb > 0) & (ma > 0) & (np.abs(ma - mb) < VOICING_SAME_PITCH)
        vweight = np.clip((VOICING_DIP_HIGH - vmin) / (VOICING_DIP_HIGH - VOICING_DIP_LOW), 0.0, 1.0) * around * same
        weight = np.maximum(weight, vweight)
    return (onset * weight).astype(np.float32)
