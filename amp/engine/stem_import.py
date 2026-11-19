"""외부 분리 결과 가져오기 (Qt 비의존).

다른 프로그램(예: Ultimate Vocal Remover의 RoFormer 모델)으로 만든 보컬·반주 파일을 프로젝트의 stems/로 가져온다.
가져온 결과는 manifest의 model이 'external'이며, 다음 분석은 Demucs를 다시 돌리지 않고 이 결과를 쓴다.

원본과 맞추는 순서
1. 앱과 같은 디코더로 읽는다(WAV·FLAC·MP3 …). 샘플레이트가 다르면 원본에 맞춰 지연 없이 리샘플한다(resample_poly).
2. 채널을 원본과 같게 한다(모노 → 복사, 3채널 이상 → 앞 두 채널).
3. 원본과의 시간 어긋남을 찾는다: 8kHz로 줄인 신호의 교차 상관으로 ±2초 안에서 찾고, 원래 샘플레이트에서 ±8샘플로 다듬는다.
   MP3로 저장한 파일은 인코더 지연 때문에 수십 ms 어긋나는 일이 흔하다.
4. 어긋난 만큼 옮기고 원본 프레임 수에 맞춰 자르거나 0으로 채운다.
5. 반주 파일이 없으면 '원본 − 보컬'로 만든다.
6. 원본과 맞는지 검사한다: 반주가 있으면 (원본 − 보컬 − 반주)의 크기, 없으면 보컬과 원본의 상관. 다른 곡이면 거부한다.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from amp.audio.data import AudioData
from amp.audio.decoder import load_audio
from amp.core.errors import AnalysisError

EXTERNAL = "external"
MAX_LAG_S = 2.0
MAX_LENGTH_DIFF_S = 3.0
COARSE_SR = 8000
REFINE = 8  # 원래 샘플레이트에서 다듬는 범위(샘플)
RESIDUAL_MAX_DB = -6.0  # 원본 − 보컬 − 반주가 원본보다 이만큼 작지 않으면 맞지 않는 파일
MIN_CORR = 0.08  # 반주 없이 보컬만 가져올 때 원본과의 최소 상관

ProgressFn = Callable[[float, str], None]


@dataclass
class ImportReport:
    vocals_file: str
    instrumental_file: str = ""
    tool: str = ""
    lag_ms: float = 0.0
    resampled_from: int | None = None
    length_diff_ms: float = 0.0  # 가져온 파일 길이 − 원본 길이 (어긋남을 맞추기 전)
    instrumental: str = "imported"  # imported / original_minus_vocals
    residual_db: float | None = None
    correlation: float | None = None
    channels_from: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"시간 어긋남 {self.lag_ms:+.1f}ms 보정" if abs(self.lag_ms) >= 0.05 else "시간 어긋남 없음"]
        if self.resampled_from:
            parts.append(f"{self.resampled_from}Hz → 원본 샘플레이트로 변환")
        if abs(self.length_diff_ms) >= 1:
            parts.append(f"길이 차이 {self.length_diff_ms:+.0f}ms 맞춤")
        parts.append("반주는 원본 − 보컬로 만듦" if self.instrumental == "original_minus_vocals" else "반주 파일 사용")
        if self.residual_db is not None:
            parts.append(f"원본과의 차이 {self.residual_db:.1f}dB")
        return ", ".join(parts)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _match_format(a: AudioData, sample_rate: int, channels: int) -> tuple[np.ndarray, int | None]:
    """(channels, frames) float32. 리샘플했으면 원래 샘플레이트를 함께 돌려준다."""
    x = a.samples.T.astype(np.float32)
    if x.shape[0] == 1 and channels > 1:
        x = np.repeat(x, channels, axis=0)
    elif x.shape[0] > channels:
        x = x[:channels] if channels > 1 else x.mean(axis=0, keepdims=True)
    elif x.shape[0] < channels:
        x = np.concatenate([x] + [x[-1:]] * (channels - x.shape[0]), axis=0)
    if a.sample_rate == sample_rate:
        return x, None
    from scipy.signal import resample_poly

    g = math.gcd(a.sample_rate, sample_rate)
    return resample_poly(x, sample_rate // g, a.sample_rate // g, axis=-1).astype(np.float32), a.sample_rate


def _mono_8k(x: np.ndarray, sr: int, max_s: float) -> np.ndarray:
    from scipy.signal import resample_poly

    m = x.mean(axis=0)[: int(max_s * sr)]
    g = math.gcd(sr, COARSE_SR)
    return resample_poly(m, COARSE_SR // g, sr // g).astype(np.float32)


def find_lag(reference: np.ndarray, candidate: np.ndarray, sr: int, max_lag_s: float = MAX_LAG_S) -> int:
    """candidate를 몇 샘플 옮겨야(양수 = 뒤로 밀기) reference와 맞는가. 둘 다 (channels, frames)."""
    from scipy.signal import correlate

    span = 120.0 + max_lag_s
    r, c = _mono_8k(reference, sr, span), _mono_8k(candidate, sr, span)
    n = min(len(r), len(c))
    if n < COARSE_SR:  # 1초보다 짧으면 어긋남을 재지 않는다
        return 0
    r, c = r[:n] - r[:n].mean(), c[:n] - c[:n].mean()
    full = correlate(r, c, mode="full", method="fft")
    lags = np.arange(-n + 1, n)
    keep = np.abs(lags) <= int(max_lag_s * COARSE_SR)
    coarse = int(lags[keep][np.argmax(full[keep])])
    guess = int(round(coarse * sr / COARSE_SR))
    # 원래 샘플레이트에서 다듬기: 가운데 20초로 ±REFINE 샘플을 직접 비교
    rm, cm = reference.mean(axis=0), candidate.mean(axis=0)
    mid = min(len(rm), len(cm)) // 2
    half = int(10 * sr)
    lo, hi = guess - REFINE, guess + REFINE
    # cm[a0 - lag : a1 - lag]가 모든 후보 lag에서 범위 안에 있어야 한다 (두 파일 길이가 달라도)
    a0 = max(mid - half, hi, 0)
    a1 = min(mid + half, len(rm), len(cm) + lo)
    if a1 - a0 < sr:
        return guess
    seg = rm[a0:a1]
    best, best_v = guess, -np.inf
    for lag in range(guess - REFINE, guess + REFINE + 1):
        v = float(np.dot(seg, cm[a0 - lag:a1 - lag]))
        if v > best_v:
            best, best_v = lag, v
    return best


def _shift(x: np.ndarray, lag: int, frames: int) -> np.ndarray:
    """lag만큼 뒤로 밀고(음수면 앞으로 당기고) 원본 프레임 수에 맞춘다."""
    if lag > 0:
        x = np.concatenate([np.zeros((x.shape[0], lag), dtype=x.dtype), x], axis=1)
    elif lag < 0:
        x = x[:, -lag:]
    if x.shape[1] >= frames:
        return x[:, :frames]
    return np.concatenate([x, np.zeros((x.shape[0], frames - x.shape[1]), dtype=x.dtype)], axis=1)


def _db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def prepare(original: AudioData, vocals_path: str, instrumental_path: str | None = None, tool: str = "",
            progress: ProgressFn | None = None) -> tuple[AudioData, AudioData, ImportReport]:
    """가져올 파일을 원본에 맞춘 (보컬, 반주, 보고). 맞지 않으면 AnalysisError."""
    report = progress or (lambda f, m: None)
    sr, ch, frames = original.sample_rate, original.channels, original.frames
    rep = ImportReport(os.path.basename(vocals_path), os.path.basename(instrumental_path or ""), tool.strip())
    report(0.05, "보컬 파일 읽는 중")
    va = load_audio(vocals_path)
    rep.channels_from = va.channels
    v, rep.resampled_from = _match_format(va, sr, ch)
    ia = None
    if instrumental_path:
        report(0.25, "반주 파일 읽는 중")
        ia = load_audio(instrumental_path)
        i, _ = _match_format(ia, sr, ch)
    rep.length_diff_ms = (v.shape[1] - frames) / sr * 1000
    if abs(rep.length_diff_ms) > MAX_LENGTH_DIFF_S * 1000:
        raise AnalysisError(f"보컬 파일 길이가 원본과 {rep.length_diff_ms / 1000:+.1f}초 다릅니다. "
                            "같은 곡을 자르지 않고 분리한 파일인지 확인하세요.")
    if ia is not None and abs(i.shape[1] - v.shape[1]) > MAX_LENGTH_DIFF_S * sr:
        raise AnalysisError("보컬과 반주 파일의 길이가 크게 다릅니다. 같은 분리 결과인지 확인하세요.")
    ref = original.samples.T.astype(np.float32)
    report(0.45, "원본과 시간 맞추는 중")
    probe = v + i[:, : v.shape[1]] if ia is not None and i.shape[1] >= v.shape[1] else v
    lag = find_lag(ref, probe, sr)
    rep.lag_ms = lag / sr * 1000
    v = _shift(v, lag, frames)
    if ia is not None:
        i = _shift(i, lag, frames)
        rep.residual_db = round(_db(ref - v - i) - _db(ref), 1)
        if rep.residual_db > RESIDUAL_MAX_DB:
            raise AnalysisError(f"보컬 + 반주가 원본과 맞지 않습니다(차이 {rep.residual_db:.1f}dB). "
                                "다른 곡이거나 음량·효과를 바꾼 파일일 수 있습니다.")
        if rep.residual_db > -20:
            rep.warnings.append(f"보컬 + 반주가 원본과 {rep.residual_db:.1f}dB만큼 다릅니다(분리 모델이 음량을 바꿨을 수 있음).")
    else:
        rep.instrumental = "original_minus_vocals"
        rm, vm = ref.mean(axis=0), v.mean(axis=0)
        denom = float(np.linalg.norm(rm) * np.linalg.norm(vm))
        rep.correlation = round(float(np.dot(rm, vm)) / denom, 3) if denom > 0 else 0.0
        if rep.correlation < MIN_CORR:
            raise AnalysisError(f"보컬 파일이 원본과 거의 맞지 않습니다(상관 {rep.correlation:.2f}). 같은 곡의 분리 결과인지 확인하세요.")
        i = ref - v
    if abs(rep.lag_ms) > 200:
        rep.warnings.append(f"원본과 {rep.lag_ms:+.0f}ms 어긋나 있어 맞췄습니다. 파일 앞에 무음이 붙었거나 잘렸을 수 있습니다.")
    report(0.7, "맞춤 완료")
    return (AudioData(np.ascontiguousarray(v.T, dtype=np.float32), sr),
            AudioData(np.ascontiguousarray(i.T, dtype=np.float32), sr), rep)


def write_stems(stems_dir: str | os.PathLike, vocals: AudioData, instrumental: AudioData, source_sha256: str,
                rep: ImportReport, vocals_path: str, instrumental_path: str | None):
    """stems/에 PCM 24bit로 쓰고 model='external' 목록을 남긴다. 목록을 먼저 지워 교체 중 중단에 대비한다."""
    import json
    import uuid

    import soundfile as sf

    from amp.engine import stems as stem_store

    stems_dir = Path(stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)
    tmp = stems_dir / f".tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for name, a in (("vocals", vocals), ("instrumental", instrumental)):
            sf.write(str(tmp / f"{name}.wav"), np.clip(a.samples, -1.0, 1.0), a.sample_rate, subtype="PCM_24")
        info = {
            "model": EXTERNAL, "sample_rate": vocals.sample_rate, "frames": vocals.frames, "source": EXTERNAL,
            "tool": rep.tool, "vocals_file": rep.vocals_file, "vocals_sha256": _sha256(vocals_path),
            "instrumental_file": rep.instrumental_file,
            "instrumental_sha256": _sha256(instrumental_path) if instrumental_path else "",
            "instrumental": rep.instrumental, "lag_ms": round(rep.lag_ms, 2), "resampled_from": rep.resampled_from,
            "length_diff_ms": round(rep.length_diff_ms, 1), "residual_db": rep.residual_db,
            "correlation": rep.correlation, "warnings": list(rep.warnings),
        }
        (tmp / "info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        return stem_store.commit(tmp, stems_dir, source_sha256, {"imported": True})
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
