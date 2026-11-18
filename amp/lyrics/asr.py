"""자동 가사 인식 (OpenAI Whisper, 작업 프로세스 안에서만 import).

- 입력: 분리된 보컬 (16kHz 모노로 바꿔 넣는다). '보컬 없는 구간'은 0으로 지워 넣는다.
- 언어: 자동이면 보컬이 가장 많은 30초 창에서 언어를 고른다(전주가 긴 곡에서 첫 30초로 고르면 틀리기 쉽다).
- 환각 줄이기: 앞 문장을 다음 문장의 힌트로 쓰지 않고(condition_on_previous_text=False),
  긴 무음은 건너뛰며(hallucination_silence_threshold), 단어 시각을 받는다(word_timestamps).
- Whisper는 노래를 '말소리 아님'(no_speech 0.8~0.95)으로 보는 일이 많아, 기본 무음 판정을 쓰면 30초 창이
  통째로 빠진다(개발 중 실제 곡에서 확인). 그래서 무음 판정으로 창을 건너뛰지 않고(no_speech_threshold=None),
  대신 분리 보컬의 실제 음량과 '보컬 없는 구간'으로 환각 여부를 따로 검사한다(``amp.lyrics.review``).
- GPU 메모리가 부족하면 CPU로 다시 돌리고 그 사실을 결과에 남긴다.

Whisper 코드·가중치 라이선스: MIT.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
# 이름 → (설명, 대략 크기 MB, 권장 GPU 메모리 GB)
MODELS = {
    "tiny": ("가장 빠름, 정확도 낮음", 75, 1),
    "base": ("빠름", 145, 1),
    "small": ("빠름, 노래에서 빠지는 구간이 많음", 465, 2),
    "medium": ("느림 (GPU 약 3GB)", 1460, 3),
    "turbo": ("기본값 (large-v3-turbo, GPU 약 3GB)", 1550, 3),
    "large-v3": ("가장 정확, 가장 느림 (GPU 약 5GB, CPU로는 수십 분)", 2950, 5),
}
LANGUAGES = {"auto": "자동 감지", "ko": "한국어", "en": "영어", "ja": "일본어"}


class AsrError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


AUTO_LARGE_FREE_GB = 5.5  # 자동일 때 GPU 여유 메모리가 이만큼 있으면 large-v3


def resolve_model(name: str, device: str, gpu_free_bytes: int | None = None) -> str:
    """자동: GPU 여유 메모리가 넉넉하면 large-v3, 아니면 turbo (CPU도 turbo).

    실제 곡 60초 구간(분리 보컬, 빔 5)에서 글자 오류율 turbo 0.088→large-v3 0.044(02번), 0.098→0.081(05번).
    CPU에서는 large-v3가 2배 넘게 느려 turbo를 쓴다. small은 CER 0.47에 30초 넘는 구간을 빠뜨려 기본으로 쓰지 않는다."""
    if name in ("", "auto"):
        if device == "cuda" and gpu_free_bytes is not None and gpu_free_bytes >= AUTO_LARGE_FREE_GB * 2**30:
            return "large-v3"
        return "turbo"
    if name not in MODELS:
        raise AsrError("internal", f"알 수 없는 인식 모델입니다: {name}")
    return name


def model_downloaded(name: str, root: str) -> bool:
    try:
        import whisper
    except ImportError:
        return False
    url = whisper._MODELS.get(name)  # noqa: SLF001 - 공개 상수가 없다
    return bool(url) and os.path.isfile(os.path.join(root, os.path.basename(url)))


def _patch_progress(report: Callable[[float, str], None], label: Callable[[float, float], str]):
    """whisper가 쓰는 tqdm을 진행률 보고로 바꾼다 (작업 프로세스 안에서만)."""
    import tqdm

    class _Bar:
        def __init__(self, *a, total=None, **k):
            self.total = float(total or 0)
            self.n = 0.0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            self.n += n
            if self.total > 0:
                report(min(1.0, self.n / self.total), label(self.n, self.total))

        def close(self):
            pass

    old = tqdm.tqdm
    tqdm.tqdm = _Bar
    return lambda: setattr(tqdm, "tqdm", old)


def busiest_window(y: np.ndarray, sr: int = SAMPLE_RATE, win_s: float = 30.0) -> tuple[int, int]:
    """보컬 에너지가 가장 큰 win_s초 창 (언어 감지용)."""
    hop = sr // 2
    if len(y) <= win_s * sr:
        return 0, len(y)
    frames = np.add.reduceat(y.astype(np.float64) ** 2, np.arange(0, len(y), hop))
    k = int(win_s * sr // hop)
    cs = np.convolve(frames, np.ones(k), mode="valid")
    i = int(np.argmax(cs))
    return i * hop, min(len(y), i * hop + int(win_s * sr))


TUNABLE = ("hallucination_silence_threshold", "condition_on_previous_text", "no_speech_threshold", "beam_size")
# beam_size 5: 실제 두 곡(CPU, turbo)에서 CER 0.239→0.200, 0.116→0.108, WER 0.436→0.396, 0.247→0.217 (시간 약 1.5배)
DEFAULT_OPTIONS = {"hallucination_silence_threshold": 2.0, "condition_on_previous_text": False,
                   "no_speech_threshold": None, "beam_size": 5}
# 빈 구간 다시 듣기: 보컬은 들리는데 1차 인식에 쓸 만한 글자가 없는 구간(3초 이상)만 다른 디코딩으로 한 번 더 인식한다.
# 05번 곡에서 환각("한글자막 by …")이 삼킨 첫 줄을 되찾았다. 다시 들은 결과도 의심 문구면 받지 않는다.
GAP_MIN_S = 3.0
GAP_JOIN_S = 0.5
GAP_PAD_S = 0.3
GAP_MAX_TOTAL_S = 150.0
# GPU에 올릴 때 대략 필요한 메모리(GB): 16비트 가중치 + 추론 작업 공간. 가중치 크기에서 잡은 여유값이다(실측 아님)
GPU_NEED_GB = {"tiny": 0.6, "base": 0.8, "small": 1.5, "medium": 3.0, "turbo": 3.0, "large-v3": 5.0}
# CPU로 넘기면 곡 하나에 수십 분 걸릴 수 있어 자동으로 넘기지 않는 모델
HEAVY_MODELS = ("medium", "large-v3")
HEARTBEAT_S = 5.0


def frame_activity(y16: np.ndarray, hop: int = 320, floor_db: float = 20.0) -> np.ndarray:
    """20ms 프레임마다 보컬이 들리는가 (곡 안 상위 음량보다 floor_db 이상 작지 않음)."""
    from amp.lyrics.ctc import activity_mask

    n = len(y16) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    db = 20 * np.log10(np.sqrt(np.mean(np.asarray(y16[: n * hop], np.float64).reshape(n, hop) ** 2, axis=1)) + 1e-9)
    return activity_mask(db, floor_db=floor_db, dilate_frames=0)


def segment_suspect(seg: dict) -> bool:
    """다시 듣기 판단용: 글자가 없거나, 흔한 환각 문구·반복·확신 낮음·비정상 속도인 구간."""
    from amp.lyrics import review

    text = str(seg.get("text", ""))
    if not review._norm(text):  # noqa: SLF001
        return True
    reasons = review.review_segment(text, float(seg["start"]), float(seg["end"]), avg_logprob=seg.get("avg_logprob"),
                                    compression_ratio=seg.get("compression_ratio"))
    return any(r in (review.R_PHRASE, review.R_REPETITIVE, review.R_LOW_CONF, review.R_RATE) for r in reasons)


def find_gaps(active: np.ndarray, segments: list[dict], dt: float = 0.02) -> list[tuple[float, float]]:
    """보컬은 들리는데 믿을 만한 인식 결과가 없는 구간 (GAP_MIN_S 이상, 짧은 끊김은 이어 붙임)."""
    n = len(active)
    covered = np.zeros(n, dtype=bool)
    for seg in segments:
        if not segment_suspect(seg):
            covered[max(0, int(seg["start"] / dt)): min(n, int(seg["end"] / dt) + 1)] = True
    need = np.asarray(active, dtype=bool) & ~covered
    join = max(1, int(GAP_JOIN_S / dt))
    gaps: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not need[i]:
            i += 1
            continue
        j = i
        while j < n and (need[j] or need[j:j + join].any()):
            j += 1
        if (j - i) * dt >= GAP_MIN_S:
            gaps.append((round(max(0.0, i * dt - GAP_PAD_S), 3), round(min(n * dt, j * dt + GAP_PAD_S), 3)))
        i = j
    chosen, total = [], 0.0
    for a, b in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True):  # 너무 많으면 긴 구간부터
        if total + (b - a) <= GAP_MAX_TOTAL_S:
            chosen.append((a, b))
            total += b - a
    return sorted(chosen)


def merge_second_pass(first: list[dict], second: list[dict]) -> list[dict]:
    """다시 들은 결과를 1차 결과에 넣는다. 의심스러운 결과는 받지 않고, 받은 결과와 겹치는 1차 의심 구간은 뺀다."""

    def overlap(a: dict, b: dict) -> float:
        return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

    accepted = [s for s in second if not segment_suspect(s)]
    out = [s for s in first if not (segment_suspect(s) and any(overlap(s, a) > 0 for a in accepted))]
    for a in accepted:
        good = [s for s in out if not segment_suspect(s)]
        if any(overlap(a, s) > 0.3 * max(a["end"] - a["start"], 1e-3) for s in good):
            continue
        out.append(dict(a, second_pass=True))
    return sorted(out, key=lambda s: s["start"])


def _segments(out: dict) -> list[dict]:
    segs = []
    for s in out.get("segments") or ():
        words = [{"text": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"]),
                  "probability": float(w.get("probability", 0.0))} for w in s.get("words") or ()]
        segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")).strip(),
                     "avg_logprob": float(s.get("avg_logprob", 0.0)), "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
                     "compression_ratio": float(s.get("compression_ratio", 0.0)), "words": words})
    return segs


def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def half_heavy_layers(model):
    """Linear·Conv1d·Embedding 가중치만 16비트로 바꾼다. Whisper의 이 층들은 계산할 때 가중치를 입력 dtype으로 맞추고,
    LayerNorm은 32비트로 계산하므로 32비트로 남긴다. large-v3 기준 GPU 가중치가 약 6.2GB → 3.1GB가 된다."""
    from torch import nn

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding)):
            m.half()
    return model


def load_model(name: str, device: str, download_root: str):
    """whisper.load_model 대신 쓴다. 원래 함수는 16비트 체크포인트를 GPU에 둔 채 32비트 모델을 GPU로 하나 더 옮겨,
    large-v3에서 순간 약 9.3GB가 필요했다(8GB GPU에서 메모리 부족 → CPU로 넘어가 멈춘 듯 보임, 사용자 보고).
    여기서는 체크포인트를 CPU에서 읽고, GPU면 무거운 층만 16비트로 바꿔 올린다."""
    import torch
    import whisper
    from whisper.model import ModelDimensions, Whisper

    path = whisper._download(whisper._MODELS[name], download_root, False)  # noqa: SLF001 - 공개 함수가 없다
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Whisper(ModelDimensions(**ckpt["dims"]))
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.set_alignment_heads(whisper._ALIGNMENT_HEADS[name])  # noqa: SLF001
    if device == "cuda":
        half_heavy_layers(model)
    return model.to(device)


def _gpu_free_bytes() -> int | None:
    try:
        import torch

        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:  # noqa: BLE001
        return None


def _oom_model_error(name: str, free: int | None = None) -> "AsrError":
    have = f"GPU 여유 메모리가 {free / 2**30:.1f}GB라 " if free is not None else "GPU 메모리가 부족해 "
    return AsrError("oom_model",
                    f"{have}{name} 모델(약 {GPU_NEED_GB.get(name, 0):.0f}GB 필요)을 GPU로 돌릴 수 없습니다. "
                    "가사 탭에서 모델을 turbo로 바꿔 다시 실행하세요. "
                    "CPU로 돌리면 곡 하나에 수십 분 걸릴 수 있어 자동으로 넘기지 않았습니다.")


class _Progress:
    """마지막 진행률을 기억했다가, 오래 걸리는 구간에서도 몇 초마다 경과 시간을 다시 알린다(멈춘 것처럼 보이지 않게)."""

    def __init__(self, report: Callable[[float, str], None]) -> None:
        import threading
        import time

        self._report = report
        self._lock = threading.Lock()
        self._time = time.monotonic
        self.t0 = self._time()
        self.fraction, self.message = 0.0, ""

    def __call__(self, fraction: float, message: str) -> None:
        with self._lock:
            self.fraction, self.message = fraction, message
        self._report(fraction, message)

    def beat(self) -> None:
        with self._lock:
            f, m = self.fraction, self.message
        self._report(f, f"{m} · {int(self._time() - self.t0)}초 경과")


def transcribe(y16: np.ndarray, *, model_name: str, language: str, device: str, download_root: str,
               report: Callable[[float, str], None], options: dict | None = None) -> dict:
    import threading

    try:
        import torch
        import whisper
    except ImportError as e:
        raise AsrError("missing_dependency",
                       f"가사 인식 라이브러리가 설치되어 있지 않습니다 ({e.name}). install_lyrics.bat을 실행하세요.") from e
    from amp.engine.dsp.separation import choose_device

    prog = _Progress(report)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_S):
            prog.beat()

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        return _transcribe(y16, prog, torch, whisper, choose_device, model_name=model_name, language=language,
                           device=device, download_root=download_root, options=options)
    finally:
        stop.set()


def _transcribe(y16, report, torch, whisper, choose_device, *, model_name, language, device, download_root, options):
    actual, note = choose_device(device)
    free = _gpu_free_bytes() if actual == "cuda" else None
    name = resolve_model(model_name, actual, free)
    os.makedirs(download_root, exist_ok=True)
    if actual == "cuda":
        if free is not None and free < GPU_NEED_GB.get(name, 0) * 2**30:
            if name in HEAVY_MODELS:
                raise _oom_model_error(name, free)
            actual, note = "cpu", f"GPU 여유 메모리({free / 2**30:.1f}GB)가 부족해 CPU로 인식합니다."
            report(0.0, note)
    if not model_downloaded(name, download_root):
        report(0.0, f"인식 모델({name}) 내려받는 중")
    restore = _patch_progress(lambda f, m: report(0.02 + 0.13 * f, m),
                              lambda n, t: f"인식 모델({name}) 내려받는 중 {n / 2**20:.0f}/{t / 2**20:.0f}MB")
    try:
        report(0.15, f"인식 모델({name}) 불러오는 중 ({actual.upper()})")
        try:
            model = load_model(name, actual, download_root)
        except Exception as e:  # noqa: BLE001
            if actual == "cuda" and is_oom(e):
                torch.cuda.empty_cache()
                if name in HEAVY_MODELS:
                    raise _oom_model_error(name) from e
                actual, note = "cpu", "GPU 메모리가 부족해 CPU로 인식합니다 (시간이 더 걸립니다)."
                report(0.15, note)
                model = load_model(name, "cpu", download_root)
            elif isinstance(e, (OSError, RuntimeError)) and not model_downloaded(name, download_root):
                raise AsrError("model_download", f"인식 모델({name})을 내려받지 못했습니다: {e}") from e
            else:
                raise
    finally:
        restore()

    lang_prob = None
    if language in ("", "auto"):
        a, b = busiest_window(y16)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(y16[a:b].astype(np.float32))),
                                          model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)
        lang_prob = float(probs[language])
        report(0.17, f"언어 감지: {language} ({lang_prob:.2f})")

    opts = dict(DEFAULT_OPTIONS)
    opts.update({k: v for k, v in (options or {}).items() if k in TUNABLE})  # 평가용 조정

    def run(m, fp16: bool, clip=None, override: dict | None = None, span=(0.18, 0.85), label="가사 인식 중"):
        a, b = span
        report(a, f"{label} ({actual.upper()}, {name})")
        restore2 = _patch_progress(lambda f, msg: report(a + (b - a) * f, msg),
                                   lambda n, t: f"{label} ({actual.upper()}, {name}) {n / 100:.0f}/{t / 100:.0f}초")
        try:
            o = dict(opts)
            for k, v in (override or {}).items():
                if v is None:
                    o.pop(k, None)
                else:
                    o[k] = v
            extra = {"clip_timestamps": [x for g in clip for x in g]} if clip else {}
            return m.transcribe(
                y16.astype(np.float32), language=language, task="transcribe", verbose=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
                logprob_threshold=-1.0, word_timestamps=True, fp16=fp16, **o, **extra,
            )
        finally:
            restore2()

    try:
        out = run(model, actual == "cuda")
    except Exception as e:  # noqa: BLE001
        if actual == "cuda" and is_oom(e):
            del model
            torch.cuda.empty_cache()
            if name in HEAVY_MODELS:
                raise _oom_model_error(name) from e
            actual, note = "cpu", "GPU 메모리가 부족해 CPU로 다시 인식했습니다 (시간이 더 걸립니다)."
            report(0.18, note)
            model = load_model(name, "cpu", download_root)
            out = run(model, False)
        else:
            raise
    segments = _segments(out)
    gaps: list[tuple[float, float]] = []
    if (options or {}).get("gap_fill", True):
        gaps = find_gaps(frame_activity(y16), segments)
        if gaps:  # 1차와 다른 디코딩(빔 ↔ 탐욕)으로 그 구간만 다시 인식
            alt = {"beam_size": None} if opts.get("beam_size") else {"beam_size": 5}
            out2 = run(model, actual == "cuda", clip=gaps, override=alt, span=(0.85, 0.98),
                       label=f"빈 구간 {len(gaps)}곳 다시 듣는 중")
            segments = merge_second_pass(segments, _segments(out2))
    report(1.0, "가사 인식 완료")
    return {"model": name, "model_version": getattr(whisper, "__version__", ""), "language": language,
            "language_probability": lang_prob, "device": actual, "note": note or "", "segments": segments,
            "gaps": [list(g) for g in gaps], "options": {k: v for k, v in opts.items()}}
