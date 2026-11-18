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
        if expected and pos < expected * 0.98:
            if path.lower().endswith(".mp3") and not mp3_has_length_header(path):
                # 길이 정보(Xing·Info·VBRI)가 없는 MP3는 libsndfile이 파일 크기(앨범 아트 포함)로 길이를 어림한다.
                # 실제 곡 4개 중 1개가 이 경우였고, 프레임을 직접 세면 디코딩 길이와 같았다.
                log.info("MP3 길이 어림값(%d 프레임)과 실제 디코딩 길이(%d 프레임)가 다릅니다 (%s)", expected, pos, path)
            else:
                log.warning("헤더 길이보다 짧게 읽혔습니다: 헤더 %d 프레임, 실제 %d 프레임 (%s). 파일 끝이 잘렸을 수 있습니다.",
                            expected, pos, path)
    return AudioData(data, sr)


def mp3_has_length_header(path: str) -> bool:
    """첫 MPEG 프레임에 전체 길이 정보(Xing·Info·VBRI)가 있는지. 읽을 수 없으면 False."""
    try:
        with open(path, "rb") as f:
            head = f.read(10)
            start = 0
            if head[:3] == b"ID3" and len(head) == 10:
                start = 10 + ((head[6] << 21) | (head[7] << 14) | (head[8] << 7) | head[9])
            f.seek(start)
            data = f.read(64 * 1024)
    except OSError:
        return False
    for i in range(len(data) - 1):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:  # 첫 프레임 동기 신호
            frame = data[i:i + 200]
            return any(tag in frame for tag in (b"Xing", b"Info", b"VBRI"))
    return False


def _load_with_miniaudio(path: str, progress: ProgressFn | None) -> AudioData:
    import miniaudio  # 선택 의존성

    decoded = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32)
    data = np.frombuffer(decoded.samples, dtype=np.float32).reshape(-1, decoded.nchannels)
    if progress:
        progress(1.0, "오디오 디코딩 중")
    return AudioData(data.copy(), decoded.sample_rate)
