"""분석 작업 프로세스.

GUI가 ``python -m amp.engine.worker <명령> --args-file <JSON 파일>`` 로 실행한다 (``--args-json <JSON>`` 도 받는다).
보컬 분리(PyTorch)·피치 추출(pYIN)·BPM 추정(librosa)은 계산 중 GIL을 오래 잡거나(측정: pYIN 30초 처리 중
다른 스레드가 최대 0.78초 멈춤) Qt와 DLL 충돌을 일으킬 수 있어 GUI 프로세스 밖에서 돌린다.

표준 출력으로는 한 줄에 하나씩 JSON 이벤트만 보낸다 (라이브러리 출력은 표준 오류로 돌린다).
    {"event": "progress", "fraction": 0.4, "message": "..."}
    {"event": "result", ..., "resources": {"peak_rss_mb": ..., "gpu_peak_mb": ...}}
    {"event": "error", "code": "...", "message": "...", "detail": "..."}

명령
    probe      PyTorch·Demucs·CUDA 설치 상태
    check_model {"checkpoint", "config"} → RoFormer 모델 파일 구조 확인 (분리하지 않음)
    separate   {"input", "out_dir", "model", "device", "model_files", "cpu_fallback"} → out_dir/vocals.wav, instrumental.wav, info.json
               (model_files: RoFormer 체크포인트·설정 경로, cpu_fallback=False면 GPU 메모리 부족 때 oom_gpu 오류)
    pitch      {"input", "out_dir", "fmin", "fmax"} → out_dir/pitch.npz, meta.json
    tempo      {"input", "out_dir"} → out_dir/tempo.json
    asr        {"input", "out_dir", "model", "language", "device", "download_root", "no_vocal_ranges"}
               → out_dir/asr.json (Whisper 가사 인식)
    emissions  {"input", "out_dir", "device", "model_dir"} → out_dir/emissions.npz (MMS 정렬 모델 출력·보컬 크기)
    ctc_align  {"emissions", "units", "anchors", "no_vocal_ranges"} → 음절별 시간 (numpy만 사용)
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import threading
import traceback

_PROTO = sys.stdout


_EMIT_LOCK = threading.Lock()


def emit(event: str, **data) -> None:
    line = json.dumps({"event": event, **data}, ensure_ascii=True) + "\n"
    with _EMIT_LOCK:  # 가사 인식은 경과 시간 알림을 다른 스레드에서도 보낸다 (줄이 섞이지 않게)
        _PROTO.write(line)
        _PROTO.flush()


def _progress(fraction: float, message: str = "") -> None:
    emit("progress", fraction=round(float(fraction), 4), message=message)


def _load(path: str):
    from amp.audio.decoder import load_audio

    return load_audio(path)


def _write_wav(path: str, data, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    # 정수 PCM으로 쓸 때 ±1을 넘는 값이 뒤집히지 않도록 자른다
    sf.write(path, np.clip(data, -1.0, 1.0), sr, format="WAV", subtype="PCM_24")


def _versions() -> dict:
    out = {}
    for mod in ("numpy", "librosa", "torch", "demucs", "soundfile"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", "")
    return out


def cmd_separate(args: dict) -> dict:
    from amp.engine.dsp.separation import separate

    audio = _load(args["input"])
    _progress(0.02, "오디오 디코딩 완료")
    vocals, instrumental, info = separate(audio.samples, audio.sample_rate, args.get("model", "htdemucs"),
                                          args.get("device", "auto"), _progress, args.get("model_files"),
                                          bool(args.get("cpu_fallback", True)))
    out = args["out_dir"]
    _write_wav(os.path.join(out, "vocals.wav"), vocals, audio.sample_rate)
    _write_wav(os.path.join(out, "instrumental.wav"), instrumental, audio.sample_rate)
    info.update(frames=audio.frames, sample_rate=audio.sample_rate, output_channels=int(vocals.shape[1]),
                versions=_versions())
    with open(os.path.join(out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
    return info


def cmd_pitch(args: dict) -> dict:
    from amp.engine.dsp.pitch import FMAX_HZ, FMIN_HZ, PitchFrames, ReuseSource, to_pitch_sr, track_pitch

    audio = _load(args["input"])
    reuse = []
    for r in args.get("reuse") or ():  # v0.6 Hybrid: 바뀌지 않은 청크는 이전 결과를 옮긴다 (입력 신호가 같을 때만)
        try:
            ra = _load(r["audio"])
            if ra.sample_rate != audio.sample_rate or ra.frames != audio.frames:
                continue
            old = PitchFrames.load(r["pitch"])
            reuse.append(ReuseSource(to_pitch_sr(ra.mono(), ra.sample_rate), old.f0_hz, old.voiced_prob,
                                     str(r.get("label", ""))))
            del ra
        except Exception as e:  # noqa: BLE001 - 이전 결과를 못 읽으면 그 청크를 새로 계산하면 된다
            print(f"이전 피치 결과를 쓰지 못함: {e}", file=sys.stderr)
    stats: dict = {}
    frames = track_pitch(audio.mono(), audio.sample_rate, _progress,
                         fmin=float(args.get("fmin", FMIN_HZ)), fmax=float(args.get("fmax", FMAX_HZ)),
                         reuse=reuse, stats=stats)
    frames.save(os.path.join(args["out_dir"], "pitch.npz"))
    meta = {"tracker": "pyin", "frames": int(frames.f0_hz.shape[0]), "frame_rate": frames.frame_rate,
            "fmin": float(args.get("fmin", FMIN_HZ)), "fmax": float(args.get("fmax", FMAX_HZ)), "versions": _versions(),
            "chunks": {"computed": int(stats.get("computed", 0)), "reused": int(stats.get("reused", 0))}}
    with open(os.path.join(args["out_dir"], "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def cmd_pitch_region(args: dict) -> dict:
    """v0.6.1: 한 구간만 좁은 음역·완화한 유성 판정으로 다시 추적한다 (여러 소리가 겹친 구간)."""
    import numpy as np

    from amp.engine.dsp.pitch import track_region

    audio = _load(args["input"])
    _progress(0.1, "구간 피치 다시 찾는 중")
    stats: dict = {}
    f0, prob, g0 = track_region(audio.mono(), audio.sample_rate, float(args["start"]), float(args["end"]),
                                float(args["fmin"]), float(args["fmax"]), stats=stats)
    np.savez_compressed(os.path.join(args["out_dir"], "region.npz"), f0_hz=f0, voiced_prob=prob, g0=np.int64(g0))
    meta = {"frames": int(len(f0)), "g0": int(g0), "voiced": float(np.isfinite(f0).mean()) if len(f0) else 0.0,
            "subharmonic_frames": int(stats.get("subharmonic_frames", 0)), "versions": _versions()}
    with open(os.path.join(args["out_dir"], "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    _progress(1.0, "구간 피치 완료")
    return meta


def cmd_tempo(args: dict) -> dict:
    from amp.engine.dsp.tempo import estimate_tempo

    audio = _load(args["input"])
    _progress(0.3, "BPM 추정 중")
    result = estimate_tempo(audio.mono(), audio.sample_rate)
    result["versions"] = _versions()
    with open(os.path.join(args["out_dir"], "tempo.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    _progress(1.0, "BPM 추정 완료")
    return result


def _vocals_16k(path: str):
    import numpy as np
    from scipy.signal import resample_poly

    audio = _load(path)
    y = audio.mono().astype(np.float32)
    if audio.sample_rate != 16000:
        from math import gcd

        g = gcd(int(audio.sample_rate), 16000)
        y = resample_poly(y, 16000 // g, int(audio.sample_rate) // g).astype(np.float32)
    return y


def _frame_rms_db(y, hop: int = 320):
    import numpy as np

    n = max(1, len(y) // hop)
    seg = np.asarray(y[: n * hop], dtype=np.float64).reshape(n, hop)
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2, axis=1)) + 1e-9)


def _ranges_mask(ranges, n: int, dt: float):
    import numpy as np

    m = np.zeros(n, dtype=bool)
    for a, b in ranges or ():
        m[max(0, int(float(a) / dt)): min(n, int(float(b) / dt) + 1)] = True
    return m


def cmd_asr(args: dict) -> dict:
    import numpy as np

    from amp.lyrics import asr
    from amp.lyrics.ctc import activity_mask

    y = _vocals_16k(args["input"])
    _progress(0.01, "보컬 불러옴")
    ranges = args.get("no_vocal_ranges") or ()
    blocked = _ranges_mask(ranges, len(y) // 320, 0.02)
    y_in = y.copy()
    for a, b in ranges:  # 보컬 없는 구간은 소리를 지워 넣는다 (반주가 섞인 전주에서 가사를 지어내지 않게)
        y_in[int(float(a) * 16000): int(float(b) * 16000)] = 0.0
    result = asr.transcribe(y_in, model_name=args.get("model", "auto"), language=args.get("language", "auto"),
                            device=args.get("device", "auto"), download_root=args["download_root"], report=_progress,
                            options=args.get("options"))
    db = _frame_rms_db(y)
    act = activity_mask(db, floor_db=20.0)
    ref = float(np.percentile(db, 95)) if len(db) else 0.0
    for seg in result["segments"]:  # 환각 검사 근거: 그 구간에 실제로 보컬이 있는가
        a, b = int(seg["start"] / 0.02), max(int(seg["start"] / 0.02) + 1, int(seg["end"] / 0.02))
        seg["activity"] = float(np.mean(act[a:b])) if a < len(act) else 0.0
        seg["level_db"] = float(np.median(db[a:b]) - ref) if a < len(db) else -120.0
        seg["in_no_vocal"] = float(np.mean(blocked[a:b])) if a < len(blocked) else 0.0
    result["versions"] = _versions()
    result.setdefault("stats", {}).update(_resources())  # 최대 RAM·GPU 메모리 (진단·실행 기록용)
    with open(os.path.join(args["out_dir"], "asr.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return {"segments": len(result["segments"]), "language": result["language"], "device": result["device"],
            "model": result["model"]}


def cmd_emissions(args: dict) -> dict:
    import numpy as np

    from amp.lyrics import mms
    from amp.engine.dsp.separation import choose_device

    y = _vocals_16k(args["input"])
    _progress(0.01, "보컬 불러옴")
    model_dir = args["model_dir"]
    actual, note = choose_device(args.get("device", "auto"))  # torch가 없으면 ImportError → 내려받기 전에 멈춘다
    if not mms.is_downloaded(model_dir):
        mms.download(model_dir, lambda f, m: _progress(0.02 + 0.3 * f, m))
    import time

    t0 = time.monotonic()
    model = mms.load(model_dir, actual)
    load_s = time.monotonic() - t0
    t1 = time.monotonic()
    try:
        lp = mms.emissions(model, y, actual, lambda f, m: _progress(0.35 + 0.63 * f, m))
    except Exception as e:  # noqa: BLE001
        if actual == "cuda" and "out of memory" in str(e).lower():
            import torch

            torch.cuda.empty_cache()
            model = model.to("cpu")
            actual, note = "cpu", "GPU 메모리가 부족해 CPU로 다시 계산했습니다."
            lp = mms.emissions(model, y, "cpu", lambda f, m: _progress(0.35 + 0.63 * f, m))
        else:
            raise
    rms_db = _frame_rms_db(y)[: len(lp)]
    if len(rms_db) < len(lp):
        rms_db = np.pad(rms_db, (0, len(lp) - len(rms_db)), constant_values=-120.0)
    np.savez_compressed(os.path.join(args["out_dir"], "emissions.npz"), logp=lp.astype(np.float16),
                        rms_db=rms_db.astype(np.float32), frame_s=np.float64(mms.HOP / mms.SAMPLE_RATE))
    info = {"model": mms.MODEL_NAME, "license": mms.LICENSE, "device": actual, "note": note or "",
            "frames": int(len(lp)), "versions": _versions(),
            "stats": {"load_s": round(load_s, 2), "emissions_s": round(time.monotonic() - t1, 2),
                      "audio_s": round(len(y) / 16000.0, 2), **_resources()}}
    with open(os.path.join(args["out_dir"], "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
    _progress(1.0, "정렬 모델 계산 완료")
    return info


def cmd_ctc_align(args: dict) -> dict:
    import numpy as np

    from amp.lyrics import ctc, mms

    with np.load(args["emissions"], allow_pickle=False) as z:
        logp = z["logp"].astype(np.float32)
        rms_db = z["rms_db"].astype(np.float64)
        dt = float(z["frame_s"])
    act = ctc.activity_mask(rms_db)
    act &= ~_ranges_mask(args.get("no_vocal_ranges"), len(act), dt)
    units = [ctc.Unit(u["id"], tuple(mms.LABEL_INDEX[c] for c in u.get("reading", "") if c in mms.LABEL_INDEX),
                      int(u.get("line", 0))) for u in args["units"]]
    anchors = {int(i): (float(a), float(b)) for i, a, b in args.get("anchors") or ()}
    soft = {int(i) for i in args.get("soft_anchors") or ()}
    _progress(0.1, f"음절 {len(units)}개 정렬 중")
    results, warnings, released = ctc.align(logp, units, dt=dt, star=mms.STAR, blank=mms.BLANK, anchors=anchors,
                                            active=act, soft=soft)
    _progress(1.0, "정렬 완료")
    return {"results": [{"id": r.syllable_id, "start": r.start, "end": r.end, "confidence": r.confidence,
                         "start_source": r.start_source, "end_source": r.end_source, "detected_end": r.detected_end}
                        for r in results], "warnings": warnings,
            "released": [units[i].syllable_id for i in released]}


def cmd_check_model(args: dict) -> dict:
    """RoFormer 모델 파일 확인: 텐서만 읽어 구조를 맞춰 보고 요약을 돌려준다 (분리는 하지 않음)."""
    from amp.engine.dsp import roformer
    from amp.engine.dsp.separation import SeparationError

    _progress(0.1, "모델 파일 읽는 중")
    try:
        model, cfg, notes = roformer.load_model(args["checkpoint"], args.get("config") or None)
    except roformer.RoformerError as e:
        raise SeparationError(e.code, str(e)) from e
    _progress(0.9, "확인 완료")
    return {"params": int(sum(p.numel() for p in model.parameters())), "notes": notes,
            "config": {k: v for k, v in cfg.to_dict().items() if k in (
                "dim", "depth", "num_bands", "num_stems", "stereo", "sample_rate", "stft_n_fft", "stft_hop_length",
                "chunk_size", "num_overlap", "target")},
            "identity": roformer.file_identity(args["checkpoint"])}


def cmd_probe(args: dict) -> dict:
    from amp.engine.dsp.separation import probe

    return probe()


COMMANDS = {"separate": cmd_separate, "pitch": cmd_pitch, "pitch_region": cmd_pitch_region, "tempo": cmd_tempo, "probe": cmd_probe,
            "check_model": cmd_check_model, "asr": cmd_asr,
            "emissions": cmd_emissions, "ctc_align": cmd_ctc_align}


def main(argv: list[str] | None = None) -> int:
    global _PROTO
    parser = argparse.ArgumentParser(prog="amp.engine.worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--args-json", default="{}")
    # 인자가 길면(가사 정렬: 음절 수백 개) Windows 명령줄 한도(32,767자)를 넘으므로 GUI는 파일로 넘긴다
    parser.add_argument("--args-file", default=None)
    ns = parser.parse_args(argv)
    _PROTO = sys.stdout
    sys.stdout = sys.stderr  # 라이브러리가 print해도 프로토콜 줄을 망치지 않게
    try:
        if ns.args_file:
            with open(ns.args_file, encoding="utf-8") as f:
                args = json.load(f)
        else:
            args = json.loads(ns.args_json)
        result = COMMANDS[ns.command](args)
    except Exception as e:  # noqa: BLE001 - 모든 오류를 코드와 함께 부모에게 알린다
        code, message = _classify(e)
        emit("error", code=code, message=message, detail=traceback.format_exc()[-4000:])
        return 1
    result = dict(result)
    result["resources"] = _resources()
    emit("result", **result)
    return 0


def _peak_rss_mb() -> float | None:
    """이 프로세스의 최대 상주 메모리(MB). 얻지 못하면 None."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.PeakWorkingSetSize / 2**20, 1)
        if os.path.exists("/proc/self/status"):
            # Linux: ru_maxrss는 exec 전 부모 프로세스의 최대치를 물려받는다(GUI·검증 도구의 메모리가 섞임).
            # VmHWM은 지금 주소 공간만의 최대치다.
            with open("/proc/self/status", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return round(int(line.split()[1]) / 1024, 1)  # kB
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS: 바이트
        return round(peak / (2**20 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001 - 측정 실패가 분석 실패가 되면 안 된다
        return None


def _resources() -> dict:
    out: dict = {"peak_rss_mb": _peak_rss_mb()}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                out["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                out["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    return out


def _classify(e: Exception) -> tuple[str, str]:
    from amp.core.errors import AudioLoadError
    from amp.engine.dsp.separation import SeparationError
    from amp.lyrics.asr import AsrError

    if isinstance(e, (SeparationError, AsrError)):
        return e.code, str(e)
    if isinstance(e, KeyboardInterrupt):
        return "cancelled", "취소되었습니다."
    if isinstance(e, AudioLoadError):
        return "decode", str(e)
    if isinstance(e, ImportError):
        return "missing_dependency", f"필요한 라이브러리를 불러오지 못했습니다 ({getattr(e, 'name', '') or e})."
    if isinstance(e, MemoryError):
        return "oom", "메모리가 부족해 분석을 마치지 못했습니다."
    if isinstance(e, OSError) and (e.errno == errno.ENOSPC or getattr(e, "winerror", None) in (39, 112)):
        return "disk_full", f"디스크 공간이 부족해 결과를 쓰지 못했습니다: {e.filename or ''}"
    if isinstance(e, PermissionError):
        return "permission", f"파일에 쓸 권한이 없습니다 (읽기 전용 폴더일 수 있습니다): {e.filename or ''}"
    return "internal", f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    sys.exit(main())
