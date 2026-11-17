"""실험용 오디오 파일을 만든다 (가지고 있는 노래 파일이 없을 때).

사용법: python scripts/make_test_audio.py [출력 경로] [길이(초)]
확장자가 .mp3면 MP3로, 아니면 WAV로 저장한다.
사람 목소리가 아니라 음이 바뀌는 사인파이므로 분석 품질 확인용으로는 쓸 수 없다.
"""

import sys

import numpy as np
import soundfile as sf


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "test_tone.mp3"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    scale = np.array([0, 2, 4, 5, 7, 9, 11, 12])
    midi = 60 + scale[(t // 0.6).astype(int) % len(scale)]
    freq = 440.0 * 2 ** ((midi - 69) / 12) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(freq) / sr)
    stereo = np.stack([x, x], axis=1).astype(np.float32)
    fmt = "MP3" if out.lower().endswith(".mp3") else "WAV"
    sf.write(out, stereo, sr, format=fmt)
    print(f"{out} ({seconds:.0f}초, {fmt}) 저장")


if __name__ == "__main__":
    main()
