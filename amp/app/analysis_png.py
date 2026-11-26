"""분석 결과 그림 내보내기 (v0.7.0 명세 18 PNG): 선택 구간 피치 그래프, 스펙트로그램, 비브라토 그래프.

QImage에 직접 그린다 (화면 위젯을 캡처하지 않음). Obsidian Violet 색·Pretendard 글꼴을 쓴다.
그림은 분석 결과를 보여 줄 뿐 새 악보를 만들지 않는다.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from amp.app import theme
from amp.app.theme import tokens as T

W, H = 1400, 560
PAD_L, PAD_R, PAD_T, PAD_B = 70, 20, 40, 40
VOICE_COLORS = ("#66C8BE", "#F2B866", "#E58FB0")


def _canvas(w=W, h=H) -> tuple[QImage, QPainter]:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(T.BG_PRIMARY))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setFont(theme.ui_font(13))
    return img, p


def _frame(p: QPainter, title: str, w=W, h=H) -> QRectF:
    plot = QRectF(PAD_L, PAD_T, w - PAD_L - PAD_R, h - PAD_T - PAD_B)
    p.fillRect(plot, QColor(T.BG_PANEL))
    p.setPen(QColor(T.TEXT))
    p.drawText(QRectF(PAD_L, 8, w - PAD_L, 26), Qt.AlignmentFlag.AlignVCenter, title)
    return plot


def _time_axis(p: QPainter, plot: QRectF, t0: float, t1: float) -> None:
    span = max(t1 - t0, 1e-6)
    step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60) if span / s <= 14)
    p.setFont(theme.ui_font(11))
    k = math.ceil(t0 / step)
    while k * step <= t1 + 1e-9:
        t = k * step
        x = plot.left() + (t - t0) / span * plot.width()
        p.setPen(QPen(QColor(T.BORDER), 1))
        p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        p.setPen(QColor(T.TEXT_2))
        p.drawText(QRectF(x - 30, plot.bottom() + 4, 60, 18), Qt.AlignmentFlag.AlignHCenter, f"{t:.1f}s")
        k += 1


def _hz_to_midi(hz):
    hz = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((hz > 0) & np.isfinite(hz), 69 + 12 * np.log2(hz / 440.0), np.nan)


def _path(xs, ys) -> QPainterPath:
    path = QPainterPath()
    pen_down = False
    for x, y in zip(xs, ys):
        if not np.isfinite(y):
            pen_down = False
            continue
        if pen_down:
            path.lineTo(x, y)
        else:
            path.moveTo(x, y)
            pen_down = True
    return path


def pitch_png(path: str, curve, t0: float, t1: float, notes=(), voices=None, title: str = "") -> None:
    """피치 곡선(MIDI) + 음표 + 성부(피치만). 세로축은 구간 안 음높이 범위."""
    img, p = _canvas()
    plot = _frame(p, title or f"피치 {t0:.2f}~{t1:.2f}초")
    series = []
    if curve is not None:
        n = len(curve.f0_hz)
        ts = curve.start_time + np.arange(n) / curve.frame_rate
        m = (ts >= t0) & (ts <= t1)
        series.append(("pitch", ts[m], _hz_to_midi(np.asarray(curve.f0_hz)[m]), QColor(T.ACCENT)))
    if voices is not None:
        for i, v in enumerate(voices.voices):
            ts = (voices.start_frame + np.arange(len(v.f0_hz))) / voices.frame_rate
            m = (ts >= t0) & (ts <= t1)
            series.append((v.display_name, ts[m], _hz_to_midi(np.asarray(v.f0_hz)[m]), QColor(VOICE_COLORS[i % 3])))
    notes = [n for n in notes if n.end >= t0 and n.start <= t1]
    vals = np.concatenate([s[2][np.isfinite(s[2])] for s in series] + [np.array([n.pitch for n in notes], float)]) \
        if series or notes else np.array([])
    lo, hi = (np.floor(vals.min()) - 2, np.ceil(vals.max()) + 2) if len(vals) else (55.0, 72.0)
    span, tspan = max(hi - lo, 1.0), max(t1 - t0, 1e-6)
    X = lambda t: plot.left() + (np.asarray(t) - t0) / tspan * plot.width()  # noqa: E731
    Y = lambda m: plot.bottom() - (np.asarray(m) - lo) / span * plot.height()  # noqa: E731
    p.setFont(theme.ui_font(11))
    for midi in range(int(lo), int(hi) + 1):
        if midi % 12 in (0, 5) or span <= 14:
            y = float(Y(midi))
            p.setPen(QPen(QColor(T.BORDER), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            from amp.core.pitch import note_name

            p.setPen(QColor(T.TEXT_2))
            p.drawText(QRectF(4, y - 9, PAD_L - 8, 18), Qt.AlignmentFlag.AlignRight, note_name(midi))
    _time_axis(p, plot, t0, t1)
    for n in notes:
        r = QRectF(float(X(max(n.start, t0))), float(Y(n.pitch + 0.4)), max(2.0, float(X(min(n.end, t1)) - X(max(n.start, t0)))),
                   float(Y(n.pitch - 0.4) - Y(n.pitch + 0.4)))
        c = QColor(T.ACCENT)
        c.setAlpha(70)
        p.fillRect(r, c)
    for name, ts, midi, color in series:
        p.setPen(QPen(color, 2.0 if name == "pitch" else 1.6))
        p.drawPath(_path(X(ts), Y(midi)))
    y = plot.top() + 6
    for name, _ts, _m, color in series:  # 범례 (색과 이름을 함께)
        p.setPen(QPen(color, 3))
        p.drawLine(QPointF(plot.right() - 160, y + 8), QPointF(plot.right() - 135, y + 8))
        p.setPen(QColor(T.TEXT))
        p.drawText(QRectF(plot.right() - 128, y, 125, 18), Qt.AlignmentFlag.AlignVCenter,
                   "단성 피치" if name == "pitch" else f"{name} (피치만)")
        y += 20
    p.end()
    img.save(path, "PNG")


def vibrato_png(path: str, curve, segments, t0: float, t1: float, title: str = "") -> None:
    """피치를 센트(구간 중앙값 기준)로 그리고 비브라토 구간을 칠하고 속도·폭을 적는다."""
    img, p = _canvas()
    plot = _frame(p, title or f"비브라토 {t0:.2f}~{t1:.2f}초 (센트, 구간 중앙값 기준)")
    tspan = max(t1 - t0, 1e-6)
    X = lambda t: plot.left() + (np.asarray(t) - t0) / tspan * plot.width()  # noqa: E731
    cents = np.array([])
    ts = np.array([])
    if curve is not None:
        n = len(curve.f0_hz)
        ts = curve.start_time + np.arange(n) / curve.frame_rate
        m = (ts >= t0) & (ts <= t1)
        ts = ts[m]
        midi = _hz_to_midi(np.asarray(curve.f0_hz)[m])
        ref = np.nanmedian(midi) if np.isfinite(midi).any() else 0.0
        cents = (midi - ref) * 100.0
    lim = max(100.0, float(np.nanmax(np.abs(cents))) * 1.1) if np.isfinite(cents).any() else 100.0
    Y = lambda c: plot.center().y() - np.asarray(c) / lim * plot.height() / 2  # noqa: E731
    for s in segments:
        if s.end < t0 or s.start > t1:
            continue
        r = QRectF(float(X(max(s.start, t0))), plot.top(), float(X(min(s.end, t1)) - X(max(s.start, t0))), plot.height())
        c = QColor(T.ACCENT_2)
        c.setAlpha(45 if s.status == "completed" else 20)
        p.fillRect(r, c)
        p.setPen(QColor(T.TEXT))
        p.setFont(theme.ui_font(11))
        p.drawText(r.adjusted(4, 4, -2, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                   f"{s.rate_hz:.2f}Hz\n한쪽 {s.extent_cents:.0f}c\n봉우리 간 {s.extent_pp_cents:.0f}c"
                   + ("" if s.status == "completed" else "\n(불확실)"))
    p.setFont(theme.ui_font(11))
    for c in (-lim, -lim / 2, 0, lim / 2, lim):
        y = float(Y(c))
        p.setPen(QPen(QColor(T.BORDER), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        p.setPen(QColor(T.TEXT_2))
        p.drawText(QRectF(4, y - 9, PAD_L - 8, 18), Qt.AlignmentFlag.AlignRight, f"{c:+.0f}c")
    _time_axis(p, plot, t0, t1)
    if len(ts):
        p.setPen(QPen(QColor(T.ACCENT), 1.8))
        p.drawPath(_path(X(ts), Y(cents)))
    p.end()
    img.save(path, "PNG")


def spectrogram_png(path: str, mono: np.ndarray, sr: int, t0: float, t1: float, title: str = "",
                    fmin: float = 50.0, fmax: float = 8000.0, rows: int = 400) -> None:
    from amp.audio import spectrogram as S

    a, b = max(0, int(t0 * sr)), min(len(mono), int(t1 * sr))
    seg = np.asarray(mono[a:b], dtype=np.float32)
    params = S.SpecParams(2048, 256)
    frames = max(1, int(math.ceil(len(seg) / params.hop)))
    tiles = [S.compute_tile(seg, sr, params, k) for k in range(int(math.ceil(frames / S.TILE_FRAMES)))]
    db = np.concatenate(tiles, axis=1)[:, :frames] if tiles else np.full((params.n_fft // 2 + 1, 1), -200.0)
    fmax = min(fmax, sr / 2)
    rr = S.rows_from_bins(db, params, sr, rows, fmin, fmax, True)
    top = float(np.percentile(rr[rr > -199], 99)) if np.any(rr > -199) else 0.0
    idx = S.to_indices(rr, top - 80.0, top)
    cmap = S.colormap()
    rgb = cmap[idx[::-1]]  # 위가 높은 주파수
    h, w = rgb.shape[:2]
    argb = np.empty((h, w, 4), np.uint8)
    argb[..., 0], argb[..., 1], argb[..., 2], argb[..., 3] = rgb[..., 2], rgb[..., 1], rgb[..., 0], 255
    spec_img = QImage(np.ascontiguousarray(argb).data, w, h, 4 * w, QImage.Format.Format_ARGB32).copy()
    img, p = _canvas()
    plot = _frame(p, title or f"스펙트로그램 {t0:.2f}~{t1:.2f}초 (로그 주파수 {fmin:.0f}~{fmax:.0f}Hz)")
    p.drawImage(plot, spec_img)
    p.setFont(theme.ui_font(11))
    for hz in (100, 200, 500, 1000, 2000, 4000, 8000):
        if fmin <= hz <= fmax:
            y = plot.bottom() - (math.log(hz / fmin) / math.log(fmax / fmin)) * plot.height()
            p.setPen(QColor(T.TEXT_2))
            p.drawText(QRectF(4, y - 9, PAD_L - 8, 18), Qt.AlignmentFlag.AlignRight, f"{hz}Hz" if hz < 1000 else f"{hz // 1000}kHz")
    _time_axis(p, plot, t0, t1)
    p.end()
    img.save(path, "PNG")


def region_wav(path: str, audio, t0: float, t1: float) -> None:
    """선택 구간 오디오 (WAV 24bit). 원본·분리 파일은 바꾸지 않는다."""
    import soundfile as sf

    sr = audio.sample_rate
    a, b = max(0, int(t0 * sr)), min(len(audio.samples), int(t1 * sr))
    sf.write(path, np.asarray(audio.samples)[a:b], sr, subtype="PCM_24")
