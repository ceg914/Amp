"""스펙트로그램 레인. 파형·피아노 롤과 같은 시간축(TimelineState)을 쓴다.

화면에 보이는 시간 구간의 타일만 작업 스레드에서 계산하고(TileCache), 그린 그림도 따로 캐시한다.
확대 수준에 맞춰 hop을 바꾸므로 5분 넘는 곡도 전체를 고해상도로 만들지 않는다.
아래 가장자리를 끌어 높이를 바꾼다. 피치 곡선을 겹쳐 그릴 수 있다(주파수 축에 맞춰).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from amp.app import theme
from amp.app.timeline import TimelineState
from amp.audio import spectrogram as S

HANDLE_PX = 6
MIN_H, MAX_H = 60, 520
FREQ_TICKS = (100, 200, 500, 1000, 2000, 5000, 10000)


class _Done(QObject):
    tileReady = Signal(object)


class _TileJob(QRunnable):
    def __init__(self, owner: "SpectrogramView", key: tuple, holder: dict, sr: int, params: S.SpecParams, tile: int):
        super().__init__()
        self.owner, self.key, self.holder, self.sr, self.params, self.tile = owner, key, holder, sr, params, tile

    def run(self) -> None:  # 작업 스레드
        try:
            mono = self.holder.get("mono")
            if mono is None:
                with self.holder["lock"]:
                    mono = self.holder.get("mono")
                    if mono is None:
                        mono = self.holder["audio"].mono().astype(np.float32, copy=False)
                        self.holder["mono"] = mono
            tile = S.compute_tile(mono, self.sr, self.params, self.tile)
            self.owner.cache.put(self.key, tile)
        finally:
            self.owner._signals.tileReady.emit(self.key)


class SpectrogramView(QWidget):
    seekRequested = Signal(float)
    heightChanged = Signal(int)

    def __init__(self, timeline: TimelineState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = timeline
        self.cache = S.TileCache()
        self._images: OrderedDict[tuple, QImage] = OrderedDict()
        self._lut = S.colormap()
        self._signals = _Done()
        self._signals.tileReady.connect(self._on_tile)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._pending: set[tuple] = set()
        self._holder: dict | None = None
        self._track_id = ""
        self._label = ""
        self._sr = 44100
        self._playhead = 0.0
        self.region: tuple[float, float] | None = None
        self.curve = None  # PitchCurve (겹쳐 그리기)
        self.show_curve = False
        # 표시 설정
        self.log_axis = True
        self.fmin, self.fmax = 50.0, 8000.0
        self.db_floor, self.db_ceil = -90.0, -10.0
        self.resolution = "balanced"
        self._resizing = False
        self.setMinimumHeight(MIN_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(140)
        self.setMouseTracking(True)
        timeline.changed.connect(self.update)

    # --- 설정 --------------------------------------------------------------

    def set_source(self, name: str, audio) -> None:
        """표시할 트랙. 다른 트랙이어도 시간축은 같다."""
        if audio is None:
            self._holder, self._track_id = None, ""
        else:
            tid = f"{name}:{id(audio)}"
            if tid != self._track_id:
                self._holder = {"audio": audio, "lock": threading.Lock()}
                self._track_id = tid
                self._sr = audio.sample_rate
        self._label = name
        self.update()

    def set_options(self, *, log_axis: bool | None = None, fmin: float | None = None, fmax: float | None = None,
                    db_floor: float | None = None, db_ceil: float | None = None, resolution: str | None = None) -> None:
        if log_axis is not None:
            self.log_axis = bool(log_axis)
        if fmin is not None:
            self.fmin = float(fmin)
        if fmax is not None:
            self.fmax = float(fmax)
        if db_floor is not None:
            self.db_floor = float(db_floor)
        if db_ceil is not None:
            self.db_ceil = float(db_ceil)
        if resolution is not None and resolution in S.RESOLUTIONS:
            self.resolution = resolution
        self._images.clear()
        self.update()

    def set_curve(self, curve, visible: bool | None = None) -> None:
        self.curve = curve
        if visible is not None:
            self.show_curve = visible
        self.update()

    def set_region(self, region) -> None:
        self.region = region
        self.update()

    def set_playhead(self, t: float) -> None:
        old = self.timeline.time_to_x(self._playhead)
        self._playhead = t
        new = self.timeline.time_to_x(t)
        self.update(int(old) - 2, 0, 5, self.height())
        self.update(int(new) - 2, 0, 5, self.height())

    def params(self) -> S.SpecParams:
        n_fft = S.RESOLUTIONS[self.resolution]
        return S.SpecParams(n_fft, S.choose_hop(n_fft, self._sr, self.timeline.px_per_sec))

    # --- 그리기 --------------------------------------------------------------

    def visible_tiles(self) -> list[tuple]:
        if self._holder is None:
            return []
        p = self.params()
        t0, t1 = self.timeline.x_to_time(0), self.timeline.x_to_time(self.width())
        dur = self._holder["audio"].duration
        t0, t1 = max(0.0, t0), min(t1, dur + p.hop / self._sr)
        if t1 <= t0:
            return []
        return [(self._track_id, p.n_fft, p.hop, k) for k in S.tiles_for_range(t0, t1, p.hop, self._sr)]

    def _tile_image(self, key: tuple, tile: np.ndarray, rows: int) -> QImage:
        ikey = (key, rows, self.fmin, self.fmax, self.log_axis, self.db_floor, self.db_ceil)
        img = self._images.get(ikey)
        if img is not None:
            self._images.move_to_end(ikey)
            return img
        p = S.SpecParams(key[1], key[2])
        db = S.rows_from_bins(tile, p, self._sr, rows, self.fmin, self.fmax, self.log_axis)
        idx = S.to_indices(db, self.db_floor, self.db_ceil)
        rgb = np.ascontiguousarray(self._lut[idx])  # (rows, frames, 3)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()
        self._images[ikey] = img
        while len(self._images) > 96:
            self._images.popitem(last=False)
        return img

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        body_h = h - HANDLE_PX
        p.fillRect(self.rect(), theme.WAVE_BG)
        if self._holder is None:
            p.setPen(theme.TEXT_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "표시할 트랙이 없습니다")
            p.end()
            return
        rows = max(8, body_h)
        missing = 0
        for key in self.visible_tiles():
            tile = self.cache.get(key)
            k0 = key[3] * S.TILE_FRAMES
            hop = key[2]
            left_t = (k0 - 0.5) * hop / self._sr  # 프레임 중심 ± hop/2
            x0 = self.timeline.time_to_x(left_t)
            width = S.TILE_FRAMES * hop / self._sr * self.timeline.px_per_sec
            if tile is None:
                missing += 1
                self._request(key)
                continue
            p.drawImage(QRectF(x0, 0, width, body_h), self._tile_image(key, tile, rows))
        if self.region is not None:
            a, b = (self.timeline.time_to_x(t) for t in self.region)
            p.fillRect(QRectF(a, 0, b - a, body_h), QColor(173, 150, 245, 40))
        if self.show_curve and self.curve is not None:
            self._draw_curve(p, body_h)
        self._draw_ticks(p, body_h)
        p.setPen(theme.TEXT_DIM)
        info = f"{self._label} · n_fft {S.RESOLUTIONS[self.resolution]}"
        if missing:
            info += " · 계산 중…"
        p.drawText(QRectF(6, 2, w - 12, 16), Qt.AlignmentFlag.AlignRight, info)
        x = self.timeline.time_to_x(self._playhead)
        p.setPen(QPen(theme.PLAYHEAD, 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, body_h))
        p.fillRect(QRectF(0, body_h, w, HANDLE_PX), theme.BORDER)
        p.end()

    def _draw_ticks(self, p: QPainter, body_h: int) -> None:
        p.setFont(theme.mono_font(9))
        for hz in FREQ_TICKS:
            if not (self.fmin <= hz <= self.fmax):
                continue
            y = float(S.freq_to_y(hz, body_h, self.fmin, self.fmax, self.log_axis))
            p.setPen(QPen(QColor(240, 236, 246, 50), 1))
            p.drawLine(QPointF(0, y), QPointF(8, y))
            p.setPen(theme.TEXT_DIM)
            p.drawText(QPointF(10, y + 4), f"{hz // 1000}k" if hz >= 1000 else str(hz))

    def _draw_curve(self, p: QPainter, body_h: int) -> None:
        c = self.curve
        t0, t1 = self.timeline.x_to_time(0), self.timeline.x_to_time(self.width())
        i0 = max(0, int((t0 - c.start_time) * c.frame_rate) - 1)
        i1 = min(len(c.f0_hz), int((t1 - c.start_time) * c.frame_rate) + 2)
        if i1 <= i0:
            return
        f = np.asarray(c.f0_hz[i0:i1], dtype=float)
        ts = c.start_time + np.arange(i0, i1) / c.frame_rate
        xs = ts * self.timeline.px_per_sec - self.timeline.scroll_x  # TimelineState.time_to_x와 같은 식 (배열)
        ys = S.freq_to_y(f, body_h, self.fmin, self.fmax, self.log_axis)
        path = QPainterPath()
        pen_down = False
        for x, y, hz in zip(xs.tolist(), ys.tolist(), f.tolist()):
            if not np.isfinite(hz) or hz <= 0:
                pen_down = False
                continue
            if pen_down:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                pen_down = True
        p.setPen(QPen(theme.PITCH_CURVE, 1.6))
        p.drawPath(path)

    def _request(self, key: tuple) -> None:
        if key in self._pending or self._holder is None:
            return
        self._pending.add(key)
        self._pool.start(_TileJob(self, key, self._holder, self._sr, S.SpecParams(key[1], key[2]), key[3]))

    def _on_tile(self, key) -> None:
        self._pending.discard(key)
        self.update()

    def wait_idle(self, msecs: int = 10000) -> bool:
        """시험용: 요청한 타일 계산이 끝날 때까지 기다린다."""
        return self._pool.waitForDone(msecs)

    # --- 마우스 ----------------------------------------------------------------

    def _on_handle(self, y: float) -> bool:
        return y >= self.height() - HANDLE_PX - 1

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._on_handle(event.position().y()):
            self._resizing = True
            return
        self.seekRequested.emit(max(0.0, self.timeline.x_to_time(event.position().x())))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        y = event.position().y()
        if self._resizing:
            h = int(min(max(y + HANDLE_PX / 2, MIN_H), MAX_H))
            if h != self.height():
                self.setFixedHeight(h)
                self.heightChanged.emit(h)
            return
        self.setCursor(Qt.CursorShape.SizeVerCursor if self._on_handle(y) else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._resizing = False

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.timeline.zoomRequested.emit(1.25 if delta > 0 else 0.8, self.timeline.x_to_time(event.position().x()))
        else:
            self.timeline.scrollRequested.emit(-float(delta))
        event.accept()
