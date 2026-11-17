"""실제 음원 검증 도구 (STEP 2.5).

앱과 같은 작업 프로세스(보컬 분리·pYIN·BPM)를 거쳐 음원 하나를 분석하고, 사람이 판단할 자료를 모은다.

    python scripts/validate_audio.py <오디오 파일> --type A|B|C [--ref 정답.csv] [--device auto|cpu|cuda]
    python scripts/validate_audio.py --self-check        합성 음원으로 도구 동작만 확인 (실제 음악 검증 아님)

출력 폴더 <out>/<파일 이름>-<시각>/
    report.md, report.json
    stems/vocals.wav, stems/instrumental.wav   직접 들어 볼 분리 결과

이 도구가 재는 것은 수치로 잴 수 있는 것뿐이다. 분리 품질(반주가 얼마나 남았는지, 목소리가 상했는지)은
분리 결과를 직접 들어 보고 report.md의 확인 항목에 적어야 한다. 수치만 보고 "분리가 잘 됐다"고 판단하지 않는다.

정답 CSV(선택): 한 줄에 `시작초,끝초,음높이`. 음높이는 MIDI 번호(64) 또는 음 이름(E4, C#5). 첫 줄이 머리글이어도 된다.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

TYPES = {
    "A": ("또렷한 보컬, 가벼운 반주", [
        "보컬 트랙에 반주(기타·피아노)가 들리는 정도: 거의 없음 / 작게 들림 / 뚜렷함",
        "보컬 음색 손상(물속 소리, 금속성 잡음, 끊김)이 있는가",
        "가사 첫 음과 프레이즈 끝에서 음표 시작·끝이 귀로 들은 위치와 맞는가",
    ]),
    "B": ("비브라토·긴 음", [
        "비브라토가 걸린 긴 음이 음표 하나로 나왔는가 (같은 높이 음표가 잘게 나뉘지 않았는가)",
        "긴 음 끝(음이 꺼지는 부분)에서 음표가 너무 일찍 끝나거나 늦게 끝나지 않는가",
        "비브라토 폭·빠르기 표시가 들리는 것과 비슷한가",
    ]),
    "C": ("빠른 음높이 변화·꾸밈음", [
        "빠른 패시지의 음이 빠지거나 둘씩 합쳐지지 않았는가",
        "꾸밈음(장식음)이 따로 표시되었는가, 미끄러지는 음(포르타멘토)이 여러 음으로 쪼개지지 않았는가",
        "같은 높이로 다시 부른 음절이 따로 나왔는가",
    ]),
}
TYPES["-"] = ("유형 미지정 (A·B·C 항목을 모두 확인)", [item for k in "ABC" for item in TYPES[k][1]])
SENSITIVITIES = ("simple", "standard", "precise")
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_pitch(text: str) -> int:
    text = text.strip()
    if re.fullmatch(r"-?\d+(\.0+)?", text):
        return int(float(text))
    m = _NOTE_RE.match(text)
    if not m:
        raise ValueError(f"음높이를 읽을 수 없습니다: {text!r}")
    pc = _PC[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return pc + (int(m.group(3)) + 1) * 12


def read_reference(path: str):
    from amp.devtools.evaluation import RefNote

    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            row = [c for c in row if c.strip()]
            if not row:
                continue
            try:
                out.append(RefNote(float(row[0]), float(row[1]), parse_pitch(row[2])))
            except (ValueError, IndexError):
                if i == 0:
                    continue  # 머리글
                raise ValueError(f"{path} {i + 1}번째 줄을 읽을 수 없습니다: {row}") from None
    return sorted(out, key=lambda r: r.start)


# --- 지표 -----------------------------------------------------------------------------

def _db(num: float, den: float) -> float | None:
    return None if den <= 0 or num <= 0 else round(10 * np.log10(num / den), 1)


def _lag_samples(a: np.ndarray, b: np.ndarray, sr: int, max_ms: float = 50.0, window_s: float = 20.0) -> int:
    """b가 a보다 늦으면 양수. 가장 소리가 큰 구간 window_s초로 잰다."""
    from scipy.signal import correlate

    n = min(len(a), len(b))
    w = min(n, int(window_s * sr))
    if w < sr // 10:
        return 0
    hop = max(1, w // 4)
    starts = range(0, max(1, n - w + 1), hop)
    s0 = max(starts, key=lambda s: float(np.sum(a[s:s + w] ** 2)))
    x, y = a[s0:s0 + w].astype(np.float64), b[s0:s0 + w].astype(np.float64)
    c = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-w + 1, w)
    m = int(max_ms / 1000 * sr)
    sel = np.abs(lags) <= m
    return int(lags[sel][np.argmax(c[sel])])


def separation_metrics(mix: np.ndarray, vocals: np.ndarray, instr: np.ndarray, sr: int) -> dict:
    """수치로 잴 수 있는 것만. 분리 품질 판단은 들어 보고 한다."""
    n = min(len(mix), len(vocals), len(instr))
    mix, vocals, instr = mix[:n], vocals[:n], instr[:n]
    if vocals.shape[1] != mix.shape[1] or instr.shape[1] != mix.shape[1]:
        mix, vocals, instr = (x.mean(axis=1, keepdims=True) for x in (mix, vocals, instr))
    e_mix = float(np.mean(mix.astype(np.float64) ** 2))
    resid = mix.astype(np.float64) - vocals - instr
    mm, vm, im = mix.mean(axis=1), vocals.mean(axis=1), instr.mean(axis=1)
    win = int(0.05 * sr)
    k = n // win
    active = None
    if k > 0:
        rv = np.sqrt(np.mean(vm[:k * win].astype(np.float64).reshape(k, win) ** 2, axis=1))
        rm = np.sqrt(np.mean(mm[:k * win].astype(np.float64).reshape(k, win) ** 2, axis=1))
        active = round(float(np.mean(rv > np.maximum(rm * 10 ** (-20 / 20), 1e-4))), 3)
    return {
        "same_length": len(vocals) == len(instr) == len(mix),
        "vocal_to_mix_db": _db(float(np.mean(vocals.astype(np.float64) ** 2)), e_mix),
        "instrumental_to_mix_db": _db(float(np.mean(instr.astype(np.float64) ** 2)), e_mix),
        "reconstruction_residual_db": _db(float(np.mean(resid ** 2)), e_mix),
        "lag_vocals_plus_instr_samples": _lag_samples(mm, vm + im, sr),
        "lag_vocals_samples": _lag_samples(mm, vm, sr),
        "clipping_fraction": {"vocals": round(float(np.mean(np.abs(vocals) >= 0.999)), 6),
                              "instrumental": round(float(np.mean(np.abs(instr) >= 0.999)), 6)},
        "vocal_active_fraction": active,
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    from amp.engine.pitch_post import voiced_runs

    return voiced_runs(mask)


def _octave_jumps(m: np.ndarray) -> int:
    d = np.diff(m)
    ok = np.isfinite(d)
    return int(np.sum(np.abs(np.abs(d[ok]) - 12.0) < 1.0))


def _spikes(m: np.ndarray) -> int:
    """1~3프레임만 주변 중앙값에서 1.5반음 넘게 벗어난 조각 수."""
    from scipy.ndimage import median_filter

    count = 0
    for a, b in _runs(np.isfinite(m)):
        seg = m[a:b]
        if seg.size < 7:
            continue
        dev = np.abs(seg - median_filter(seg, size=7, mode="nearest")) > 1.5
        count += sum(1 for s, e in _runs(dev) if e - s <= 3)
    return count


def pitch_metrics(frames, curve) -> dict:
    from amp.core.pitch import hz_to_midi

    raw = hz_to_midi(np.asarray(frames.f0_hz, dtype=np.float64))
    post = curve.midi().astype(np.float64)
    voiced = np.isfinite(raw)
    return {
        "frames": int(len(raw)),
        "voiced_fraction_raw": round(float(voiced.mean()), 3) if len(raw) else 0.0,
        "voiced_fraction_post": round(float(np.isfinite(post).mean()), 3) if len(post) else 0.0,
        "uncertain_fraction": round(float(np.mean(frames.voiced_prob[voiced] < 0.3)), 3) if voiced.any() else 0.0,
        "octave_jumps_raw": _octave_jumps(raw),
        "octave_jumps_post": _octave_jumps(post),
        "spikes_raw": _spikes(raw),
        "spikes_post": _spikes(post),
    }


def note_stats(notes, curve) -> dict:
    m = curve.midi().astype(np.float64)
    fr = curve.frame_rate
    durs = np.array([n.end - n.start for n in notes]) if notes else np.zeros(0)
    dev, offset = [], []
    for n in notes:
        a, b = int((n.start - curve.start_time) * fr), int((n.end - curve.start_time) * fr)
        trim = (b - a) // 5
        seg = m[a + trim:b - trim]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 3:
            offset.append((float(np.median(seg)) - n.pitch) * 100)
            dev.append(float(np.median(np.abs(seg - np.median(seg)))) * 100)
    same_close = sum(1 for x, y in zip(notes, notes[1:]) if x.pitch == y.pitch and y.start - x.end <= 0.05)
    return {
        "count": len(notes),
        "mean_duration_s": round(float(durs.mean()), 3) if durs.size else None,
        "shorter_than_100ms": int(np.sum(durs < 0.1)),
        "ornaments": sum(1 for n in notes if n.ornament),
        "with_vibrato": sum(1 for n in notes if n.vibrato is not None),
        "adjacent_same_pitch": same_close,  # 다시 부른 음절이거나 긴 음이 잘못 나뉜 것 — 들어 보고 확인
        "pitch_offset_cents_median": round(float(np.median(offset)), 1) if offset else None,
        "in_note_wobble_cents_median": round(float(np.median(dev)), 1) if dev else None,
    }


def reference_scores(ref, notes, curve) -> dict:
    from amp.devtools.evaluation import score_notes

    s = score_notes(ref, notes)
    m = curve.midi().astype(np.float64)
    fr = curve.frame_rate
    cents = []
    for r in ref:
        a, b = int((r.start - curve.start_time) * fr), int((r.end - curve.start_time) * fr)
        trim = (b - a) // 5
        seg = m[a + trim:b - trim]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            cents.append((float(np.median(seg)) - r.pitch) * 100)
    c = np.abs(np.array(cents)) if cents else np.zeros(0)
    return {**s.as_dict(),
            "ref_notes_with_pitch": int(c.size),
            "median_abs_cents": round(float(np.median(c)), 1) if c.size else None,
            "within_50_cents": round(float(np.mean(c <= 50)), 3) if c.size else None,
            "octave_errors": int(np.sum(np.abs(c - 1200) < 100)) if c.size else 0}


# --- 실행 -----------------------------------------------------------------------------

@dataclass
class ValidationResult:
    report_dir: Path
    data: dict = field(default_factory=dict)

    @property
    def report_md(self) -> Path:
        return self.report_dir / "report.md"


def _worker(command: str, args: dict, timings: dict, resources: dict, stage: str) -> dict:
    from amp.engine.worker_client import run_worker

    t0 = time.perf_counter()
    out = run_worker(command, args)
    timings[stage] = round(time.perf_counter() - t0, 2)
    resources[stage] = out.get("resources", {})
    return out


def validate(audio_path: str, kind: str = "A", ref_path: str | None = None, out_root: str = "validation",
             device: str = "auto", separation: str = "htdemucs", label: str = "") -> ValidationResult:
    """음원 하나를 검증한다. separation: htdemucs | none | (시험용) _identity."""
    from amp.audio.decoder import load_audio
    from amp.devtools import benchmark
    from amp.engine import notes as notes_v2
    from amp.engine import pitch_post as post_v2
    from amp.engine.baselines import notes_v1, pitch_post_v1
    from amp.engine.baselines.settings_v1 import note_params_v1
    from amp.engine.dsp.pitch import PitchFrames
    from amp.engine.settings import AnalysisSettings

    if kind not in TYPES:
        raise ValueError(f"유형은 A·B·C 또는 -(미지정)입니다: {kind}")
    src = Path(audio_path).resolve()  # 작업 프로세스는 저장소 폴더에서 실행되므로 절대 경로로 넘긴다
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root).resolve() / f"{src.stem}-{stamp}"
    stems_dir = out / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    resources: dict[str, dict] = {}
    data: dict = {"file": str(src), "type": kind, "type_label": TYPES[kind][0], "label": label,
                  "created": datetime.now().isoformat(timespec="seconds"), "separation": separation, "device": device}

    t0 = time.perf_counter()
    audio = load_audio(str(src))
    timings["decode"] = round(time.perf_counter() - t0, 2)
    data["audio"] = {"duration_s": round(audio.duration, 3), "sample_rate": audio.sample_rate,
                     "channels": audio.channels, "size_bytes": src.stat().st_size, "format": src.suffix.lower()}

    if separation != "none":
        info = _worker("separate", {"input": str(src), "out_dir": str(stems_dir), "model": separation, "device": device},
                       timings, resources, "separation")
        vocals = load_audio(str(stems_dir / "vocals.wav"))
        instr = load_audio(str(stems_dir / "instrumental.wav"))
        data["separation_info"] = {k: info.get(k) for k in ("model", "device", "fallback", "silent_input") if k in info}
        data["separation_metrics"] = separation_metrics(audio.samples, vocals.samples, instr.samples, audio.sample_rate)
        pitch_input, tempo_input = str(stems_dir / "vocals.wav"), str(stems_dir / "instrumental.wav")
        vocal_mono = vocals.mono()
    else:
        pitch_input = tempo_input = str(src)
        vocal_mono = audio.mono()

    pdir, tdir = out / "work-pitch", out / "work-tempo"
    pdir.mkdir()
    tdir.mkdir()
    _worker("pitch", {"input": pitch_input, "out_dir": str(pdir)}, timings, resources, "pitch")
    frames = PitchFrames.load(pdir / "pitch.npz")
    tempo = _worker("tempo", {"input": tempo_input, "out_dir": str(tdir)}, timings, resources, "tempo")
    data["tempo"] = {k: tempo.get(k) for k in ("bpm", "first_beat", "candidates", "reliable", "reason")}

    t0 = time.perf_counter()
    curve = post_v2.build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate, rms_fine=frames.rms_fine)
    artic = post_v2.articulation_envelope(frames.onset, frames.rms, frames.frame_rate, rms_fine=frames.rms_fine)
    data["pitch"] = pitch_metrics(frames, curve)
    ref = read_reference(ref_path) if ref_path else None
    data["notes"], data["reference"], keep = {}, {}, {}
    for sens in SENSITIVITIES:
        found = notes_v2.detect_notes(curve, artic, AnalysisSettings.for_sensitivity(sens).note_params())
        keep[sens] = found
        data["notes"][sens] = note_stats(found, curve)
        if ref is not None:
            data["reference"][sens] = reference_scores(ref, found, curve)
    timings["notes_all_sensitivities"] = round(time.perf_counter() - t0, 3)

    # 같은 pYIN 결과 위의 기준선 v1 (STEP 2) — 변경 전후 비교
    onset1 = benchmark.v1_onset(vocal_mono, audio.sample_rate, len(frames.f0_hz))
    curve1 = pitch_post_v1.build_curve(frames.f0_hz, frames.voiced_prob, frames.rms, frames.frame_rate)
    v1 = notes_v1.detect_notes(curve1, pitch_post_v1.articulation_envelope(onset1, frames.rms, frames.frame_rate),
                               note_params_v1(AnalysisSettings()))
    data["notes"]["standard_v1"] = note_stats(v1, curve1)
    data["pitch"]["voiced_fraction_post_v1"] = round(float(np.isfinite(curve1.midi()).mean()), 3)
    if ref is not None:
        data["reference"]["standard_v1"] = reference_scores(ref, v1, curve1)
    data["timings_s"] = timings
    data["resources"] = resources
    data["notes_standard"] = [{"start": n.start, "end": n.end, "pitch": n.pitch, "confidence": n.confidence,
                               "ornament": n.ornament} for n in keep["standard"]]

    for d in (pdir, tdir):  # 중간 파일은 보고서에 필요 없다 (분리 결과는 들어 보도록 남긴다)
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    (out / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(data), encoding="utf-8")
    return ValidationResult(out, data)


def markdown(d: dict) -> str:
    t = d["type"]
    lines = [f"# 실제 음원 검증: {Path(d['file']).name}", ""]
    if d.get("label"):
        lines += [f"> {d['label']}", ""]
    a = d["audio"]
    lines += [f"- 유형 {t if t != '-' else '미지정'}: {d['type_label']}", f"- 길이 {a['duration_s']}초, {a['sample_rate']}Hz, {a['channels']}채널, "
              f"{a['format']}, {a['size_bytes']:,}바이트", f"- 분리: {d['separation']} (장치 설정 {d['device']})",
              f"- 생성: {d['created']}", ""]
    lines += ["## 처리 시간과 메모리", "", "| 단계 | 시간 | 작업 프로세스 최대 RAM | GPU 최대 |", "|---|---|---|---|"]
    for stage, sec in d["timings_s"].items():
        r = d["resources"].get(stage, {})
        rss = f"{r['peak_rss_mb']} MB" if r.get("peak_rss_mb") is not None else "-"
        gpu = f"{r['gpu_peak_mb']} MB" if r.get("gpu_peak_mb") is not None else "-"
        lines.append(f"| {stage} | {sec}s | {rss} | {gpu} |")
    if "separation_metrics" in d:
        m = d["separation_metrics"]
        lines += ["", "## 보컬 분리 (수치)", "",
                  "수치는 분리 품질을 보장하지 않는다. stems/의 두 파일을 직접 들어 보고 아래 확인 항목을 채운다.", "",
                  f"- 모델·장치: {d.get('separation_info', {})}",
                  f"- 길이 일치: {m['same_length']}",
                  f"- 보컬/원본 에너지: {m['vocal_to_mix_db']} dB, 반주/원본: {m['instrumental_to_mix_db']} dB",
                  f"- 재구성 잔차(원본 − 보컬 − 반주)/원본: {m['reconstruction_residual_db']} dB",
                  f"- 정렬 지연(샘플): 보컬+반주 {m['lag_vocals_plus_instr_samples']}, 보컬 {m['lag_vocals_samples']}",
                  f"- 클리핑 비율: {m['clipping_fraction']}",
                  f"- 보컬 활성 구간 비율 (보컬 RMS가 원본 대비 −20dB 이상인 50ms 구간): {m['vocal_active_fraction']}"]
    p = d["pitch"]
    lines += ["", "## 피치", "",
              f"- 유성 프레임 비율: pYIN {p['voiced_fraction_raw']}, 후처리 v2 {p['voiced_fraction_post']}, "
              f"v1 {p.get('voiced_fraction_post_v1')}",
              f"- 불확실(유성 확률 0.3 미만) 비율: {p['uncertain_fraction']}",
              f"- 옥타브 도약(이웃 프레임 ±12반음): pYIN {p['octave_jumps_raw']} → 후처리 {p['octave_jumps_post']}",
              f"- 짧은 튐(1~3프레임, 1.5반음 이상): pYIN {p['spikes_raw']} → 후처리 {p['spikes_post']}"]
    lines += ["", "## 음표 (민감도별)", "",
              "| 민감도 | 개수 | 평균 길이 | 100ms 미만 | 꾸밈음 | 비브라토 | 붙은 같은 음 | 음 안 흔들림(cent) | 음높이 치우침(cent) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sens, s in d["notes"].items():
        lines.append(f"| {sens} | {s['count']} | {s['mean_duration_s']} | {s['shorter_than_100ms']} | {s['ornaments']} | "
                     f"{s['with_vibrato']} | {s['adjacent_same_pitch']} | {s['in_note_wobble_cents_median']} | "
                     f"{s['pitch_offset_cents_median']} |")
    lines += ["", "'붙은 같은 음'은 50ms 안에 이어진 같은 높이 음표 쌍이다. 다시 부른 음절이면 맞고, "
              "긴 음이 잘못 나뉜 것이면 오류다. 들어 보고 판단한다."]
    if d.get("reference"):
        lines += ["", "## 정답 대비 (사용자 제공 정답 CSV)", "",
                  "| 민감도 | 정답/추정/일치 | F1 | 시작 오차 ms | 끝 오차 ms | 피치 중앙 오차 cent | 50cent 이내 | 옥타브 오류 |",
                  "|---|---|---|---|---|---|---|---|"]
        for sens, s in d["reference"].items():
            lines.append(f"| {sens} | {s['ref']}/{s['est']}/{s['matched']} | {s['f1']} | {s['onset_mae_ms']} | "
                         f"{s['offset_mae_ms']} | {s['median_abs_cents']} | {s['within_50_cents']} | {s['octave_errors']} |")
    tp = d["tempo"]
    lines += ["", "## BPM", "", f"- 추정 {tp.get('bpm')} (신뢰 {tp.get('reliable')}), 첫 박 {tp.get('first_beat')}초",
              f"- 후보: {tp.get('candidates')}"]
    if tp.get("reason"):
        lines.append(f"- 참고: {tp['reason']}")
    lines += ["", "## 직접 확인할 항목 (들어 보고 적는다)", ""]
    lines += [f"- [ ] {item}: " for item in TYPES[t][1]]
    lines += ["- [ ] 보컬 트랙과 반주 트랙을 함께 재생했을 때 박자가 어긋나는 곳이 있는가: ",
              "- [ ] BPM이 들리는 박과 맞는가 (2배·절반 후보 중 무엇이 맞는가): ", ""]
    return "\n".join(lines)


def self_check(out_root: str = "validation", separation: str = "htdemucs", device: str = "cpu") -> ValidationResult:
    """합성 음원(모음 합성 + 화음·드럼 반주)으로 도구가 끝까지 도는지 확인한다. 실제 음악 검증이 아니다."""
    import soundfile as sf

    from amp.devtools import synth

    melody = [(1.0, 1.8, 64), (1.9, 2.7, 67), (2.8, 3.1, 69), (3.14, 3.44, 69), (3.5, 4.6, 71), (4.7, 5.5, 69),
              (5.6, 5.67, 74), (5.67, 6.6, 72)]
    dur = 8.0
    voice = synth.sung(melody, dur, vibrato_cents=25)
    mix = voice + 0.35 * synth.backing(dur)
    mix = (0.8 * mix / np.max(np.abs(mix))).astype(np.float32)
    base = Path(out_root)
    base.mkdir(parents=True, exist_ok=True)
    wav = base / "self-check-synthetic.wav"
    sf.write(wav, np.stack([mix, mix], 1), synth.SR)
    ref = base / "self-check-reference.csv"
    with open(ref, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "pitch"])
        for a, b, m in melody:
            w.writerow([a, b, m])
    return validate(str(wav), "C", str(ref), out_root, device, separation,
                    label="도구 동작 확인용 합성 음원(모음 합성 + 화음·드럼 반주). 실제 음악 검증이 아니다.")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="validate_audio", description="실제 음원 검증 도구 (STEP 2.5)")
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--type", default="A", choices=sorted(TYPES))
    ap.add_argument("--ref", help="정답 CSV (시작초,끝초,음높이)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--separation", default="htdemucs", help="htdemucs | none")
    ap.add_argument("--out", default="validation")
    ap.add_argument("--self-check", action="store_true", help="합성 음원으로 도구 동작만 확인")
    ns = ap.parse_args(argv)
    if ns.self_check:
        res = self_check(ns.out, ns.separation, ns.device if ns.device != "auto" else "cpu")
    elif ns.audio:
        res = validate(ns.audio, ns.type, ns.ref, ns.out, ns.device, ns.separation)
    else:
        ap.error("오디오 파일을 주거나 --self-check를 쓰세요.")
        return 2
    print(res.report_md)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
