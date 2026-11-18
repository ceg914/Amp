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
    "medium": ("느림, 더 정확", 1460, 5),
    "turbo": ("기본값 (large-v3-turbo)", 1550, 6),
    "large-v3": ("가장 정확, 가장 느림", 2950, 10),
}
LANGUAGES = {"auto": "자동 감지", "ko": "한국어", "en": "영어", "ja": "일본어"}


class AsrError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def resolve_model(name: str, device: str) -> str:
    """자동이면 turbo. 개발 중 실제 노래(05번 트랙, CPU)에서 small은 CER 0.47·30초 넘는 구간 누락,
    turbo는 CER 0.24로 차이가 커서 CPU에서도 turbo를 쓴다(276초 곡 CPU 4코어 165초)."""
    if name in ("", "auto"):
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
DEFAULT_OPTIONS = {"hallucination_silence_threshold": 2.0, "condition_on_previous_text": False,
                   "no_speech_threshold": None}


def transcribe(y16: np.ndarray, *, model_name: str, language: str, device: str, download_root: str,
               report: Callable[[float, str], None], options: dict | None = None) -> dict:
    try:
        import torch
        import whisper
    except ImportError as e:
        raise AsrError("missing_dependency",
                       f"가사 인식 라이브러리가 설치되어 있지 않습니다 ({e.name}). install_lyrics.bat을 실행하세요.") from e
    from amp.engine.dsp.separation import choose_device

    actual, note = choose_device(device)
    name = resolve_model(model_name, actual)
    os.makedirs(download_root, exist_ok=True)
    if not model_downloaded(name, download_root):
        report(0.0, f"인식 모델({name}) 내려받는 중")
    restore = _patch_progress(lambda f, m: report(0.02 + 0.13 * f, m),
                              lambda n, t: f"인식 모델({name}) 내려받는 중 {n / 2**20:.0f}/{t / 2**20:.0f}MB")
    try:
        try:
            model = whisper.load_model(name, device=actual, download_root=download_root)
        except Exception as e:  # noqa: BLE001
            if "out of memory" in str(e).lower() and actual == "cuda":
                torch.cuda.empty_cache()
                actual, note = "cpu", "GPU 메모리가 부족해 CPU로 인식했습니다."
                model = whisper.load_model(name, device="cpu", download_root=download_root)
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

    def run(m, fp16: bool):
        restore2 = _patch_progress(lambda f, msg: report(0.18 + 0.8 * f, msg),
                                   lambda n, t: f"가사 인식 중 ({actual.upper()}) {n / 100:.0f}/{t / 100:.0f}초")
        try:
            opts = dict(DEFAULT_OPTIONS)
            opts.update({k: v for k, v in (options or {}).items() if k in TUNABLE})  # 평가용 조정
            return m.transcribe(
                y16.astype(np.float32), language=language, task="transcribe", verbose=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
                logprob_threshold=-1.0, word_timestamps=True, fp16=fp16, **opts,
            )
        finally:
            restore2()

    try:
        out = run(model, actual == "cuda")
    except Exception as e:  # noqa: BLE001
        if actual == "cuda" and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            model = model.to("cpu")
            actual, note = "cpu", "GPU 메모리가 부족해 CPU로 다시 인식했습니다."
            report(0.18, note)
            out = run(model, False)
        else:
            raise
    segments = []
    for s in out.get("segments") or ():
        words = [{"text": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"]),
                  "probability": float(w.get("probability", 0.0))} for w in s.get("words") or ()]
        segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")).strip(),
                         "avg_logprob": float(s.get("avg_logprob", 0.0)), "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
                         "compression_ratio": float(s.get("compression_ratio", 0.0)), "words": words})
    report(1.0, "가사 인식 완료")
    return {"model": name, "model_version": getattr(whisper, "__version__", ""), "language": language,
            "language_probability": lang_prob, "device": actual, "note": note or "", "segments": segments}
