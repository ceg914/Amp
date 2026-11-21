"""청취 도구: 구간 반복, 음높이 유지 느린 재생, 음표 미리 듣기, 음표 소리 트랙, 스펙트로그램 레인 표시.

메인 창이 커지지 않게 이 컨트롤러가 도구막대·메뉴 동작·단축키를 만든다.
- 반복 구간은 재생기(AudioPlayer)가 들고 있다. 파형 끌기, 선택 음표, 선택 음절, 시간 직접 입력으로 정한다.
  음표·음절로 정할 때만 앞뒤 여유를 더한다(직접 고른 구간은 그대로).
- 느린 재생에 필요한 늘린 오디오는 작업 스레드에서 만들고 StretchCache에 둔다. 분석 데이터 시간값은 바꾸지 않는다.
- 음표 소리 트랙('MIDI 음표')은 재생기에 트랙으로 넣어 보컬과 같은 시간축으로 섞어 듣는다.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel, QSlider, QToolBar

from amp.app import theme
from amp.app.note_preview import NotePreviewer
from amp.app.player import PlaybackState
from amp.audio.data import AudioData
from amp.audio.spectrogram import RESOLUTION_LABELS
from amp.audio.stretch import SPEEDS, StretchCache
from amp.audio.synth import render_track

log = logging.getLogger(__name__)
MIDI_TRACK = "midi"
LANE_MODES = {"hidden": "숨기기", "wave": "파형만", "spec": "스펙트로그램만", "both": "파형 + 스펙트로그램",
              "spec_curve": "스펙트로그램 + 피치 곡선"}
FREQ_RANGES = {"voice": (50.0, 8000.0, "목소리 (50Hz~8kHz)"), "low": (50.0, 2000.0, "기본음 (50Hz~2kHz)"),
               "full": (30.0, 16000.0, "넓게 (30Hz~16kHz)")}


class ListeningController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.player = window.player
        self.cache = StretchCache()
        self.preview = NotePreviewer(self)
        self.padding = 0.25  # 음표·음절로 구간을 정할 때 앞뒤 여유(초)
        self.midi_on = False
        self.midi_gain = 0.6
        self._midi_dirty = True
        self._midi_timer = QTimer(self)
        self._midi_timer.setSingleShot(True)
        self._midi_timer.setInterval(350)
        self._midi_timer.timeout.connect(self._render_midi)
        self._stretch_seq = 0
        self.lane_mode = "wave"
        self._build_actions()
        self.player.stretchRequested.connect(self._on_stretch_requested)
        self.player.playbackChanged.connect(self._sync_widgets)
        self.player.stateChanged.connect(self._on_player_state)

    # --- 동작·도구막대 ---------------------------------------------------------

    def _build_actions(self) -> None:
        w = self.w

        def A(text, slot, shortcut=None, checkable=False):  # noqa: N802
            a = QAction(text, w)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setCheckable(checkable)
            w.addAction(a)
            return a

        self.act_loop = A("구간 반복 재생", self._on_loop_toggled, "Alt+L", checkable=True)
        self.act_set_start = A("반복 구간 시작 = 재생 위치", self.set_start_here, "Alt+[")
        self.act_set_end = A("반복 구간 끝 = 재생 위치", self.set_end_here, "Alt+]")
        self.act_region_notes = A("선택 음표 구간 반복", self.loop_selected_notes, "Ctrl+Shift+R")
        self.act_clear_region = A("반복 구간 지우기", self.clear_region)
        self.act_slower = A("재생 속도 낮추기", lambda: self.step_speed(-1), "Alt+,")
        self.act_faster = A("재생 속도 높이기", lambda: self.step_speed(+1), "Alt+.")
        self.act_normal = A("정상 속도로", lambda: self.set_speed(1.0), "Alt+/")
        self.act_preview = A("음표 미리 듣기", self._on_preview_toggled, checkable=True)
        self.act_preview.setChecked(True)
        self.act_play_notes = A("선택 음표 소리 듣기", self.play_selected_notes, "Alt+P")
        self.act_midi = A("음표 소리 함께 재생 (보컬과 비교)", self._on_midi_toggled, "Alt+M", checkable=True)
        # 레인 표시 (보기 메뉴)
        self.lane_group = QActionGroup(w)
        self.lane_actions = {}
        for key, label in LANE_MODES.items():
            a = QAction(label, w, checkable=True)
            a.triggered.connect(lambda _c=False, k=key: self.set_lane_mode(k))
            self.lane_group.addAction(a)
            self.lane_actions[key] = a
        self.lane_actions[self.lane_mode].setChecked(True)
        self.act_log_axis = QAction("로그 주파수 축", w, checkable=True)
        self.act_log_axis.setChecked(True)
        self.act_log_axis.toggled.connect(lambda on: w.spectrogram.set_options(log_axis=on))
        self.res_group = QActionGroup(w)
        self.res_actions = {}
        for key, label in RESOLUTION_LABELS.items():
            a = QAction(label, w, checkable=True)
            a.triggered.connect(lambda _c=False, k=key: w.spectrogram.set_options(resolution=k))
            self.res_group.addAction(a)
            self.res_actions[key] = a
        self.res_actions["balanced"].setChecked(True)
        self.range_group = QActionGroup(w)
        self.range_actions = {}
        for key, (lo, hi, label) in FREQ_RANGES.items():
            a = QAction(label, w, checkable=True)
            a.triggered.connect(lambda _c=False, lo=lo, hi=hi: w.spectrogram.set_options(fmin=lo, fmax=hi))
            self.range_group.addAction(a)
            self.range_actions[key] = a
        self.range_actions["voice"].setChecked(True)
        self.act_brighter = QAction("스펙트로그램 밝게 (+6dB)", w)
        self.act_brighter.triggered.connect(lambda: self._shift_db(-6))
        self.act_darker = QAction("스펙트로그램 어둡게 (−6dB)", w)
        self.act_darker.triggered.connect(lambda: self._shift_db(+6))

    def build_toolbar(self) -> QToolBar:
        w = self.w
        w.addToolBarBreak()
        tb = w.addToolBar("청취")
        tb.setObjectName("listening")
        tb.setMovable(False)
        tb.addWidget(QLabel(" 속도 "))
        self.speed_combo = QComboBox()
        for s in SPEEDS:
            self.speed_combo.addItem(f"{s:g}×", s)
        self.speed_combo.setCurrentIndex(len(SPEEDS) - 1)
        self.speed_combo.setToolTip("음높이를 유지한 채 느리게 재생합니다 (Alt+, / Alt+. / Alt+/).\n"
                                    "느린 재생은 반복 구간(없으면 재생 위치부터 60초)만 미리 늘려 둡니다.")
        self.speed_combo.currentIndexChanged.connect(lambda _i: self.set_speed(self.speed_combo.currentData()))
        tb.addWidget(self.speed_combo)
        tb.addSeparator()
        self.act_loop.setIcon(_icon("repeat"))
        tb.addAction(self.act_loop)
        self.start_spin = self._time_spin("반복 구간 시작(초)")
        self.end_spin = self._time_spin("반복 구간 끝(초)")
        self.start_spin.editingFinished.connect(self._on_spin_edited)
        self.end_spin.editingFinished.connect(self._on_spin_edited)
        tb.addWidget(self.start_spin)
        tb.addWidget(QLabel("~"))
        tb.addWidget(self.end_spin)
        tb.addWidget(QLabel(" 여유 "))
        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0.0, 3.0)
        self.pad_spin.setSingleStep(0.1)
        self.pad_spin.setDecimals(2)
        self.pad_spin.setSuffix(" s")
        self.pad_spin.setValue(self.padding)
        self.pad_spin.setFont(theme.mono_font())
        self.pad_spin.setToolTip("음표·음절로 반복 구간을 정할 때 앞뒤에 더하는 시간")
        self.pad_spin.valueChanged.connect(lambda v: setattr(self, "padding", float(v)))
        tb.addWidget(self.pad_spin)
        tb.addSeparator()
        self.act_preview.setIcon(_icon("music"))
        tb.addAction(self.act_preview)
        self.preview_vol = self._slider("미리 듣기 음량", int(self.preview.volume * 100), self.preview.set_volume)
        tb.addWidget(self.preview_vol)
        self.act_midi.setIcon(_icon("list-music"))
        tb.addAction(self.act_midi)
        self.midi_vol = self._slider("음표 소리 트랙 음량", int(self.midi_gain * 100), self._set_midi_gain)
        tb.addWidget(self.midi_vol)
        self.status = QLabel("")
        self.status.setProperty("role", "secondary")
        tb.addWidget(self.status)
        self._sync_widgets()
        return tb

    def build_menu(self, play_menu, view_menu) -> None:
        play_menu.addSeparator()
        for a in (self.act_loop, self.act_set_start, self.act_set_end, self.act_region_notes, self.act_clear_region):
            play_menu.addAction(a)
        play_menu.addSeparator()
        for a in (self.act_slower, self.act_faster, self.act_normal):
            play_menu.addAction(a)
        play_menu.addSeparator()
        for a in (self.act_preview, self.act_play_notes, self.act_midi):
            play_menu.addAction(a)
        lanes = view_menu.addMenu("파형·스펙트로그램")
        for a in self.lane_actions.values():
            lanes.addAction(a)
        lanes.addSeparator()
        lanes.addAction(self.act_log_axis)
        res = lanes.addMenu("시간·주파수 해상도")
        for a in self.res_actions.values():
            res.addAction(a)
        rng = lanes.addMenu("주파수 범위")
        for a in self.range_actions.values():
            rng.addAction(a)
        lanes.addAction(self.act_brighter)
        lanes.addAction(self.act_darker)

    @staticmethod
    def _time_spin(tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.0, 99999.0)
        s.setDecimals(2)
        s.setSingleStep(0.1)
        s.setSuffix(" s")
        s.setKeyboardTracking(False)
        s.setFont(theme.mono_font())
        s.setToolTip(tip)
        s.setMinimumWidth(92)
        return s

    @staticmethod
    def _slider(tip: str, value: int, slot) -> QSlider:
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 100)
        sl.setValue(value)
        sl.setFixedWidth(70)
        sl.setToolTip(tip)
        sl.valueChanged.connect(lambda v: slot(v / 100.0))
        return sl

    # --- 반복 구간 -------------------------------------------------------------

    def set_region(self, a: float, b: float, *, enable: bool | None = None, play: bool = False,
                   pad: bool = False) -> None:
        if pad:
            a, b = a - self.padding, b + self.padding
        dur = self.player.duration
        a, b = max(0.0, min(a, b)), min(max(a, b), dur if dur > 0 else max(a, b))
        if b - a < 0.05:
            self.w.statusBar().showMessage("반복 구간이 너무 짧습니다 (0.05초 이상).", 4000)
            return
        self.player.set_loop_region((a, b))
        if enable is not None:
            self.player.set_loop_enabled(enable)
        if play:
            self.w.seek(a)
            if self.player.state is not PlaybackState.PLAYING:
                self.player.play()

    def clear_region(self) -> None:
        self.player.set_loop_enabled(False)
        self.player.set_loop_region(None)

    def set_start_here(self) -> None:
        t = self.player.position()
        r = self.player.loop_region
        end = r[1] if r is not None and r[1] > t + 0.05 else min(t + 2.0, self.player.duration)
        self.set_region(t, end)

    def set_end_here(self) -> None:
        t = self.player.position()
        r = self.player.loop_region
        start = r[0] if r is not None and r[0] < t - 0.05 else max(0.0, t - 2.0)
        self.set_region(start, t)

    def loop_selected_notes(self) -> None:
        ids = self.w.view.roll.selected_ids()
        notes = [self.w.doc.score.get(i) for i in ids]
        notes = [n for n in notes if n is not None]
        if not notes:
            self.w.statusBar().showMessage("반복할 음표를 먼저 고르세요.", 4000)
            return
        self.set_region(min(n.start for n in notes), max(n.end for n in notes), enable=True, play=True, pad=True)

    def loop_range(self, a: float, b: float) -> None:
        """가사 음절 등 다른 곳에서 고른 구간 (앞뒤 여유를 더해 반복 재생)."""
        self.set_region(a, b, enable=True, play=True, pad=True)

    def _on_loop_toggled(self, on: bool) -> None:
        if on and self.player.loop_region is None:
            self.loop_selected_notes() if self.w.view.roll.selected_ids() else self.set_start_here()
        self.player.set_loop_enabled(on)
        self._sync_widgets()

    def _on_spin_edited(self) -> None:
        a, b = self.start_spin.value(), self.end_spin.value()
        if b > a + 0.05:
            self.set_region(a, b)

    # --- 속도 ---------------------------------------------------------------

    def set_speed(self, speed: float) -> None:
        self.player.set_speed(float(speed))

    def step_speed(self, direction: int) -> None:
        cur = self.player.speed
        speeds = list(SPEEDS)
        i = min(range(len(speeds)), key=lambda k: abs(speeds[k] - cur))
        self.set_speed(speeds[min(max(i + direction, 0), len(speeds) - 1)])

    def _on_stretch_requested(self, a: float, b: float, speed: float) -> None:
        tracks = self.player.stretch_inputs()
        if not tracks:
            return
        sr = next(iter(tracks.values())).sample_rate
        version = self.player.tracks_version
        self._stretch_seq += 1
        seq = self._stretch_seq
        inputs = {name: (f"{name}:{audio.token}", audio.samples) for name, audio in tracks.items()}
        start, end = int(round(a * sr)), int(round(b * sr))
        self.status.setText(f"느린 재생 준비 중 ({b - a:.1f}초 구간, {speed:g}×)…")
        handle = self.w.runner.submit("느린 재생 준비", lambda p, t: self.cache.stretch_tracks(inputs, start, end, speed, sr))

        def done(out: dict) -> None:
            if seq != self._stretch_seq:
                return
            self.status.setText("")
            self.player.set_stretched(a, b, speed, out, version)

        handle.finished.connect(done)
        handle.failed.connect(lambda msg, _d: self.status.setText(f"느린 재생 준비 실패: {msg}"))

    # --- 미리 듣기 ------------------------------------------------------------

    def _on_preview_toggled(self, on: bool) -> None:
        self.preview.set_enabled(on)

    def preview_pitch(self, pitch: int) -> None:
        self.preview.play_note(pitch)

    def play_selected_notes(self) -> None:
        notes = sorted((self.w.doc.score.get(i) for i in self.w.view.roll.selected_ids()), key=lambda n: n.start)
        notes = [n for n in notes if n is not None]
        if not notes:
            self.w.statusBar().showMessage("들어 볼 음표를 먼저 고르세요.", 4000)
            return
        if not self.preview.enabled:
            self.act_preview.setChecked(True)
            self.preview.set_enabled(True)
        self.preview.play_sequence([(n.start, n.end - n.start, n.pitch) for n in notes])

    # --- 음표 소리 트랙 ---------------------------------------------------------

    def _on_midi_toggled(self, on: bool) -> None:
        self.midi_on = bool(on)
        if on:
            self._render_midi()
        else:
            self.player.set_track(MIDI_TRACK, None)

    def _set_midi_gain(self, v: float) -> None:
        self.midi_gain = v
        self.player.set_track_state(MIDI_TRACK, gain=v)

    def notes_changed(self) -> None:
        self._midi_dirty = True
        if self.midi_on:
            self._midi_timer.start()

    def _render_midi(self) -> None:
        audio = self.player._audio
        if not self.midi_on or audio is None:
            return
        if self.player.state is PlaybackState.PLAYING and self.player.speed < 1.0:
            return  # 느린 재생 중에 바꾸면 다시 늘려야 하므로 멈춘 뒤 반영한다 (_on_player_state)
        notes = list(self.w.doc.score.notes())
        sr, frames, ch = audio.sample_rate, audio.frames, min(audio.channels, 2)
        handle = self.w.runner.submit("음표 소리 만들기", lambda p, t: render_track(notes, frames, sr, ch))

        def done(arr) -> None:
            if not self.midi_on or self.player._audio is not audio:
                return
            self.player.set_track_state(MIDI_TRACK, gain=self.midi_gain)
            self.player.set_track(MIDI_TRACK, AudioData(arr, sr))
            self._midi_dirty = False

        handle.finished.connect(done)

    def _on_player_state(self, state) -> None:
        if state is PlaybackState.STOPPED and self.midi_on and self._midi_dirty:
            self._midi_timer.start()

    # --- 레인 -------------------------------------------------------------------

    def set_lane_mode(self, mode: str) -> None:
        if mode not in LANE_MODES:
            return
        self.lane_mode = mode
        self.lane_actions[mode].setChecked(True)
        w = self.w
        wave = mode in ("wave", "both")
        spec = mode in ("spec", "both", "spec_curve")
        w.waveform.setVisible(wave)
        w.wave_label.setVisible(wave)
        w.spectrogram.setVisible(spec)
        w.spec_label.setVisible(spec)
        analysis = w.doc.project.analysis
        w.spectrogram.set_curve(analysis.pitch_curve if analysis is not None else None, visible=mode == "spec_curve")

    def _shift_db(self, delta: float) -> None:
        sp = self.w.spectrogram
        sp.set_options(db_floor=min(sp.db_floor + delta, sp.db_ceil - 12), db_ceil=sp.db_ceil + delta)

    # --- 공통 ---------------------------------------------------------------------

    def _sync_widgets(self) -> None:
        if not hasattr(self, "speed_combo"):
            return
        r = self.player.loop_region
        for spin, v in ((self.start_spin, r[0] if r else 0.0), (self.end_spin, r[1] if r else 0.0)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self.act_loop.blockSignals(True)
        self.act_loop.setChecked(self.player.loop_enabled)
        self.act_loop.blockSignals(False)
        i = self.speed_combo.findData(min(SPEEDS, key=lambda s: abs(s - self.player.speed)))
        self.speed_combo.blockSignals(True)
        self.speed_combo.setCurrentIndex(i)
        self.speed_combo.blockSignals(False)
        for view in (self.w.waveform, self.w.spectrogram):
            view.set_region(r)
        self.w.view.roll.set_loop_region(r if self.player.loop_enabled else None)

    def shutdown(self) -> None:
        self.preview.shutdown()
        self.cache.clear()

    def project_closed(self) -> None:
        self.preview.stop()
        self.cache.clear()
        if self.midi_on:
            self.act_midi.setChecked(False)
            self.midi_on = False


def _icon(name: str):
    from amp.app.theme.icons import icon

    return icon(name)
