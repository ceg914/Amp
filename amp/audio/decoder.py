"""오디오 파일 디코딩.

기본 백엔드는 soundfile(libsndfile 1.1 이상, MP3 지원)이다.
soundfile이 파일을 열지 못하면 선택 설치 패키지인 miniaudio로 한 번 더 시도한다.
긴 파일도 블록 단위로 읽으면서 진행률을 알리고 취소 요청을 확인한다.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable

import numpy as np

from amp.audio.data import AudioData
from amp.core.errors import AnalysisCancelled, AudioLoadError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg")
BLOCK_FRAMES = 1 << 18  # 약 6초 (44.1kHz)

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


def file_fingerprint(path: str, chunk: int = 1 << 16) -> str:
    """파일 크기와 앞·뒤 64KB로 만든 빠른 지문. 오디오 파일이 바뀌었는지 확인할 때 쓴다."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return f"{size}:{h.hexdigest()}"


def load_audio(
    path: str,
    progress: ProgressFn | None = None,
    is_cancelled: CancelFn | None = None,
) -> AudioData:
    if not os.path.isfile(path):
        raise AudioLoadError(f"파일을 찾을 수 없습니다: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"지원하지 않는 형식입니다({ext or '확장자 없음'}). 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    try:
        audio = _load_with_soundfile(path, progress, is_cancelled)
    except AnalysisCancelled:
        raise
    except Exception as first_error:  # noqa: BLE001 - 백엔드마다 예외 종류가 다르다
        log.warning("soundfile 디코딩 실패, miniaudio로 재시도: %s", first_error)
        try:
            audio = _load_with_miniaudio(path, progress)
        except ImportError:
            raise AudioLoadError(_unreadable(path, first_error)) from first_error
        except Exception as second_error:  # noqa: BLE001
            raise AudioLoadError(_unreadable(path, first_error, second_error)) from second_error
    if audio.frames == 0:
        raise AudioLoadError("오디오 데이터가 비어 있습니다.")
    return audio


def _unreadable(path: str, *errors: Exception) -> str:
    detail = " / ".join(str(e) for e in errors)
    return (
        f"오디오를 읽지 못했습니다: {os.path.basename(path)}\n"
        f"파일이 손상되었거나 지원하지 않는 인코딩일 수 있습니다.\n(상세: {detail})"
    )


def _load_with_soundfile(path: str, progress: ProgressFn | None, is_cancelled: CancelFn | None) -> AudioData:
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr, ch = f.samplerate, f.channels
        expected = max(int(f.frames), 0) if f.frames and f.frames > 0 else 0
        buf = np.empty((expected, ch), dtype=np.float32)
        pos = 0
        extra: list[np.ndarray] = []
        while True:
            if is_cancelled and is_cancelled():
                raise AnalysisCancelled("불러오기를 취소했습니다.")
            block = f.read(BLOCK_FRAMES, dtype="float32", always_2d=True)
            n = block.shape[0]
            if n == 0:
                break
            fit = max(0, min(n, expected - pos))
            if fit:
                buf[pos : pos + fit] = block[:fit]
            if fit < n:  # 헤더의 길이 정보보다 실제 데이터가 길 때
                extra.append(block[fit:])
            pos += n
            if progress and expected:
                progress(min(pos / expected, 1.0), "오디오 디코딩 중")
        if extra:
            data = np.concatenate([buf] + extra, axis=0)
        else:
            data = buf[: min(pos, expected)]
    return AudioData(data, sr)


def _load_with_miniaudio(path: str, progress: ProgressFn | None) -> AudioData:
    import miniaudio  # 선택 의존성

    decoded = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32)
    data = np.frombuffer(decoded.samples, dtype=np.float32).reshape(-1, decoded.nchannels)
    if progress:
        progress(1.0, "오디오 디코딩 중")
    return AudioData(data.copy(), decoded.sample_rate)
