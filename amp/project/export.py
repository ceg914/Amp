"""오디오 내보내기 (WAV / MP3).

이미 있는 결과만 쓴다. 원본은 원본 파일에서, 보컬·반주는 stems/의 WAV에서 읽어 형식만 바꾼다.
보컬 분리를 다시 실행하지 않는다.

MP3는 soundfile(libsndfile + LAME)로 쓴다. 고정 비트레이트는 압축 수준 = (320 − kbps) / 288 로
지정하며, 128·192·256·320kbps가 정확히 나오는 것을 개발 중 확인했다.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from amp.audio.data import AudioData
from amp.core.errors import AnalysisCancelled, StorageError
from amp.core.fileio import long_path, storage_error

WAV_SUBTYPES = {"PCM_16": "16-bit PCM", "PCM_24": "24-bit PCM", "FLOAT": "32-bit float"}
MP3_BITRATES = (128, 192, 256, 320)
MP3_RATES = (32000, 44100, 48000)  # MPEG-1 Layer III가 지원하는 샘플레이트
BLOCK_S = 2.0


@dataclass(frozen=True)
class ExportOptions:
    fmt: str = "wav"  # wav | mp3
    wav_subtype: str = "PCM_24"
    mp3_bitrate: int = 320

    def __post_init__(self) -> None:
        if self.fmt not in ("wav", "mp3"):
            raise ValueError(f"지원하지 않는 형식: {self.fmt}")
        if self.wav_subtype not in WAV_SUBTYPES:
            raise ValueError(f"지원하지 않는 WAV 형식: {self.wav_subtype}")
        if self.mp3_bitrate not in MP3_BITRATES:
            raise ValueError(f"지원하지 않는 비트레이트: {self.mp3_bitrate}")

    @property
    def extension(self) -> str:
        return "." + self.fmt


def default_name(project_name: str, track: str, options: ExportOptions) -> str:
    return f"{project_name}_{track}{options.extension}"


def export_audio(
    audio: AudioData,
    dst: str | os.PathLike,
    options: ExportOptions,
    progress: Callable[[float, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """audio를 dst로 저장한다. 같은 폴더의 임시 파일에 쓴 뒤 바꿔치기한다 (덮어쓰기 확인은 호출한 쪽 몫)."""
    import soundfile as sf

    dst = Path(dst)
    if options.fmt == "mp3" and audio.sample_rate not in MP3_RATES:
        raise StorageError(f"MP3는 {', '.join(map(str, MP3_RATES))}Hz만 지원합니다 (지금 {audio.sample_rate}Hz). WAV로 내보내 주세요.")
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise storage_error(e, dst.parent) from e
    tmp = dst.with_name(f".{dst.stem}.{uuid.uuid4().hex[:6]}.part{dst.suffix}")
    channels = min(audio.channels, 2) if options.fmt == "mp3" else audio.channels
    kwargs = dict(samplerate=audio.sample_rate, channels=channels)
    if options.fmt == "wav":
        kwargs.update(format="WAV", subtype=options.wav_subtype)
    else:
        kwargs.update(format="MP3", subtype="MPEG_LAYER_III", bitrate_mode="CONSTANT",
                      compression_level=(320 - options.mp3_bitrate) / 288)
    data = audio.samples[:, :channels]
    block = int(BLOCK_S * audio.sample_rate)
    try:
        with sf.SoundFile(long_path(tmp), "w", **kwargs) as f:
            for start in range(0, audio.frames, block):
                if is_cancelled and is_cancelled():
                    raise AnalysisCancelled("내보내기를 취소했습니다.")
                chunk = data[start:start + block]
                if options.wav_subtype != "FLOAT" or options.fmt == "mp3":
                    chunk = np.clip(chunk, -1.0, 1.0)  # 정수 형식에서 값이 뒤집히지 않게
                f.write(chunk)
                if progress:
                    progress(min(1.0, (start + len(chunk)) / audio.frames), f"{dst.name} 쓰는 중")
        os.replace(long_path(tmp), long_path(dst))
    except OSError as e:
        Path(tmp).unlink(missing_ok=True)
        raise storage_error(e, dst) from e
    except Exception as e:
        Path(tmp).unlink(missing_ok=True)
        if isinstance(e, (AnalysisCancelled, StorageError)):
            raise
        raise StorageError(f"{dst.name}을(를) 쓰지 못했습니다: {e}") from e
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"path": str(dst), "size": dst.stat().st_size}
