"""재생 스트림: 출력 장치 프레임 ↔ 원본 시간. 구간 반복과 느린 재생(늘린 오디오)을 한 곳에서 처리한다. Qt 비의존.

장치는 프레임 번호를 0부터 계속 늘려 가며 읽는다. 스트림은 그 번호를 '내용' 프레임으로 바꾼다.
- 보통 재생: 내용 = 원본 전체 믹서, 시작 위치부터.
- 구간 반복: 내용 = 구간 [a, b)를 되풀이한다. 이음매 앞뒤로 짧게 페이드해 딸깍 소리를 막는다.
- 느린 재생: 내용 = 구간을 늘린 오디오(StretchCache 결과)로 만든 믹서. 출력 1초 = 원본 speed초.

화면의 재생 위치는 늘린 오디오의 샘플 위치가 아니라 time_at()으로 바꾼 원본 시간을 쓴다.
분석 데이터의 시간값은 여기서 바뀌지 않는다.
"""

from __future__ import annotations

import numpy as np

SEAM_FADE_S = 0.006  # 반복 이음매 페이드 (끝 6ms 줄이고 처음 6ms 키움)
LOOP_FRAMES = 1 << 60  # 반복 재생의 '끝 없음'


class PlaybackStream:
    def __init__(self, mixer, content_start: int, content_end: int, origin_time: float, speed: float = 1.0,
                 loop: bool = False) -> None:
        """mixer: read_float·quantize·out_rate·bytes_per_frame를 가진 PcmMixer.
        내용 = mixer 출력 프레임 [content_start, content_end). 그 첫 프레임의 원본 시간이 origin_time이다.
        speed: 출력 1초에 원본이 몇 초 지나가는가 (늘린 오디오면 늘린 배수, 아니면 1)."""
        self.mixer = mixer
        self.c0 = int(content_start)
        self.c1 = max(int(content_end), self.c0)
        self.length = self.c1 - self.c0
        self.origin = float(origin_time)
        self.speed = float(speed)
        self.loop = bool(loop) and self.length > 0
        self.out_rate = mixer.out_rate
        self.bytes_per_frame = mixer.bytes_per_frame
        self.total_out_frames = LOOP_FRAMES if self.loop else self.length
        self.fade = max(1, int(SEAM_FADE_S * self.out_rate)) if self.loop and self.length > 8 * SEAM_FADE_S * self.out_rate else 0

    def frame_for_time(self, t: float) -> int:
        """원본 시간 t에 해당하는 스트림 프레임 (내용 밖이면 처음·끝으로)."""
        f = int(round((t - self.origin) / self.speed * self.out_rate))
        return min(max(f, 0), max(self.length - 1, 0))

    def time_at(self, n: int) -> float:
        """스트림 프레임 n(장치가 읽은 누적 프레임)의 원본 시간."""
        if self.length == 0:
            return self.origin
        k = (n % self.length) if self.loop else min(n, self.length)
        return self.origin + k / self.out_rate * self.speed

    def read(self, start: int, count: int) -> bytes:
        start, count = max(0, int(start)), max(0, int(count))
        if not self.loop:
            count = min(count, self.length - start)
            if count <= 0:
                return b""
            return self.mixer.quantize(self.mixer.read_float(self.c0 + start, count))
        if count == 0:
            return b""
        parts = []
        n = start
        left = count
        while left > 0:
            k = n % self.length
            take = min(left, self.length - k)
            block = self.mixer.read_float(self.c0 + k, take)
            if block.shape[0] < take:  # 믹서 끝(정상이라면 없음): 0으로 채운다
                block = np.vstack([block, np.zeros((take - block.shape[0], block.shape[1]), np.float32)])
            if self.fade:
                block = block.copy()
                f = self.fade
                # 이 블록이 덮는 내용 프레임 [k, k+take) 중 처음 f개(키움)와 끝 f개(줄임)
                head = np.arange(k, k + take)
                g = np.ones(take, np.float32)
                m = head < f
                g[m] = np.minimum(g[m], (head[m] + 1) / f)
                m = head >= self.length - f
                g[m] = np.minimum(g[m], (self.length - head[m]) / f)
                block *= g[:, None]
            parts.append(block)
            n += take
            left -= take
        return self.mixer.quantize(np.vstack(parts))
