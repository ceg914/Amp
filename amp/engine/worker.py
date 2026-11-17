"""분석 작업 프로세스.

GUI가 ``python -m amp.engine.worker <명령> --args-json <JSON>`` 으로 실행한다.
보컬 분리(PyTorch)·피치 추출(pYIN)·BPM 추정(librosa)은 계산 중 GIL을 오래 잡거나(측정: pYIN 30초 처리 중
다른 스레드가 최대 0.78초 멈춤) Qt와 DLL 충돌을 일으킬 수 있어 GUI 프로세스 밖에서 돌린다.

표준 출력으로는 한 줄에 하나씩 JSON 이벤트만 보낸다 (라이브러리 출력은 표준 오류로 돌린다).
    {"event": "progress", "fraction": 0.4, "message": "..."}
    {"event": "result", ..., "resources": {"peak_rss_mb": ..., "gpu_peak_mb": ...}}
    {"event": "error", "code": "...", "message": "...", "detail": "..."}

명령
    probe      PyTorch·Demucs·CUDA 설치 상태
    separate   {"input", "out_dir", "model", "device"} → out_dir/vocals.wav, instrumental.wav, info.json
    pitch      {"input", "out_dir", "fmin", "fmax"} → out_dir/pitch.npz, meta.json
    tempo      {"input", "out_dir"} → out_dir/tempo.json
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import traceback

_PROTO = sys.stdout


def emit(event: str, **data) -> None:
    _PROTO.write(json.dumps({"event": event, **data}, ensure_ascii=True) + "\n")
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
                                          args.get("device", "auto"), _progress)
    out = args["out_dir"]
    _write_wav(os.path.join(out, "vocals.wav"), vocals, audio.sample_rate)
    _write_wav(os.path.join(out, "instrumental.wav"), instrumental, audio.sample_rate)
    info.update(frames=audio.frames, sample_rate=audio.sample_rate, output_channels=int(vocals.shape[1]),
                versions=_versions())
    with open(os.path.join(out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
    return info


def cmd_pitch(args: dict) -> dict:
    from amp.engine.dsp.pitch import FMAX_HZ, FMIN_HZ, track_pitch

    audio = _load(args["input"])
    frames = track_pitch(audio.mono(), audio.sample_rate, _progress,
                         fmin=float(args.get("fmin", FMIN_HZ)), fmax=float(args.get("fmax", FMAX_HZ)))
    frames.save(os.path.join(args["out_dir"], "pitch.npz"))
    meta = {"tracker": "pyin", "frames": int(frames.f0_hz.shape[0]), "frame_rate": frames.frame_rate,
            "fmin": float(args.get("fmin", FMIN_HZ)), "fmax": float(args.get("fmax", FMAX_HZ)), "versions": _versions()}
    with open(os.path.join(args["out_dir"], "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
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


def cmd_probe(args: dict) -> dict:
    from amp.engine.dsp.separation import probe

    return probe()


COMMANDS = {"separate": cmd_separate, "pitch": cmd_pitch, "tempo": cmd_tempo, "probe": cmd_probe}


def main(argv: list[str] | None = None) -> int:
    global _PROTO
    parser = argparse.ArgumentParser(prog="amp.engine.worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--args-json", default="{}")
    ns = parser.parse_args(argv)
    _PROTO = sys.stdout
    sys.stdout = sys.stderr  # 라이브러리가 print해도 프로토콜 줄을 망치지 않게
    try:
        result = COMMANDS[ns.command](json.loads(ns.args_json))
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

    if isinstance(e, SeparationError):
        return e.code, str(e)
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
