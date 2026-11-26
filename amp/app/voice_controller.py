"""v0.7.0 '보컬 정밀 → 다중 보컬' (Feature B 화면).

- 분석: 지금 분석 소스(보컬 분리 결과의 보컬, 없으면 원본)에서 다중 피치 후보 → 성부 (2성부, 3성부는 실험)
- 성부 목록: 보이기(피아노 롤에 겹쳐 그림), 이름 바꾸기(두 번 눌러 고침), 역할, 사용자 확정
- 편집(반복 구간 = 대상 구간, 재생 위치 = 나누는 곳): 두 성부 구간 바꾸기·옮기기, 구간 지우기, 후보 다시 고르기(MIDI 근처),
  나누기·합치기, 성부 추가, 음역을 정해 그 구간만 다시 분석. 편집마다 바로 저장하고 되돌리기·다시 하기가 된다
- 피치만 있는 성부다. 소리 파일이 아니며 원본 오디오를 바꾸지 않는다
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from amp.engine import voices as V

VOICE_COLORS = ("#F0A3C8", "#8FD3E8", "#E8D28F", "#B8A4F0")  # 성부 1·2·3·4 (모델 색·상태색과 다름)
MAX_UNDO = 50


def _secondary(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("role", "secondary")
    return lab


def _small_button(text: str, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    b.setToolTip(tip)
    b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return b


class MultiVoicePage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(_secondary("같은 시간에 다른 높이로 부르는 목소리를 성부로 나눠 피치를 따라갑니다. 피치만 있는 성부이며 "
                                 "소리 파일이 아닙니다. 연결을 확신할 수 없는 프레임은 '불확실'로 셉니다."))
        form = QFormLayout()
        self.source = QLabel("—")
        self.source.setWordWrap(True)
        form.addRow("분석할 소리", self.source)
        self.max_voices = QComboBox()
        self.max_voices.addItem("2성부", 2)
        self.max_voices.addItem("3성부 (실험)", 3)
        form.addRow("최대 성부", self.max_voices)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.run = _small_button("분석", "다시 분석하면 이전 결과는 analysis/multi_pitch_runs/에 보관하고, "
                                       "직접 고치거나 확정한 성부는 그대로 둡니다.")
        self.undo = _small_button("되돌리기")
        self.redo = _small_button("다시 하기")
        for b in (self.run, self.undo, self.redo):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.message = _secondary("")
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.message)
        self.voices = QTreeWidget()
        self.voices.setHeaderLabels(["성부 (피치만)", "역할", "확정", "유성", "높이"])
        self.voices.setRootIsDecorated(False)
        lay.addWidget(self.voices, 2)
        grid = QGridLayout()
        grid.addWidget(QLabel("성부 A"), 0, 0)
        self.voice_a = QComboBox()
        grid.addWidget(self.voice_a, 0, 1)
        grid.addWidget(QLabel("성부 B"), 0, 2)
        self.voice_b = QComboBox()
        grid.addWidget(self.voice_b, 0, 3)
        lay.addLayout(grid)
        self.range_label = _secondary("")
        lay.addWidget(self.range_label)
        ops = QGridLayout()
        self.swap = _small_button("A↔B 바꾸기", "반복 구간에서 두 성부의 피치를 서로 바꿉니다 (연결이 뒤바뀐 곳).")
        self.move = _small_button("A→B 옮기기", "반복 구간에서 A의 피치를 B로 옮깁니다 (B가 비어 있는 프레임만).")
        self.clear = _small_button("A 지우기", "반복 구간에서 A의 피치를 지웁니다.")
        self.split = _small_button("A 나누기", "재생 위치에서 A를 두 성부로 나눕니다.")
        self.merge = _small_button("B를 A에 합치기", "겹친 프레임은 A 값을 남깁니다.")
        self.add = _small_button("성부 추가")
        for i, b in enumerate((self.swap, self.move, self.clear, self.split, self.merge, self.add)):
            ops.addWidget(b, i // 3, i % 3)
        lay.addLayout(ops)
        pick = QHBoxLayout()
        self.target = QDoubleSpinBox()
        self.target.setRange(24, 108)
        self.target.setDecimals(1)
        self.target.setValue(64.0)
        self.target.setSuffix(" (MIDI)")
        self.pick = _small_button("A를 이 높이 후보로", "반복 구간에서 이 높이(±150센트) 근처의 다른 후보로 A를 바꿉니다.")
        pick.addWidget(self.target)
        pick.addWidget(self.pick)
        pick.addStretch(1)
        lay.addLayout(pick)
        band = QHBoxLayout()
        self.band_lo = QDoubleSpinBox()
        self.band_hi = QDoubleSpinBox()
        for sp, v in ((self.band_lo, 60.0), (self.band_hi, 67.0)):
            sp.setRange(24, 108)
            sp.setDecimals(0)
            sp.setValue(v)
        self.band = _small_button("A 음역 다시 분석", "반복 구간에서 이 음역(MIDI)만 다시 분석해 A를 바꿉니다.")
        band.addWidget(self.band_lo)
        band.addWidget(QLabel("–"))
        band.addWidget(self.band_hi)
        band.addWidget(self.band)
        band.addStretch(1)
        lay.addLayout(band)
        role_row = QHBoxLayout()
        self.role = QComboBox()
        for k, label in V.ROLES.items():
            self.role.addItem(label, k)
        self.set_role = _small_button("A 역할 정하기")
        self.confirm = QCheckBox("A 확정 (다시 분석해도 유지)")
        role_row.addWidget(self.role)
        role_row.addWidget(self.set_role)
        role_row.addWidget(self.confirm)
        role_row.addStretch(1)
        lay.addLayout(role_row)


class VoiceController(QObject):
    def __init__(self, window, page: MultiVoicePage) -> None:
        super().__init__(window)
        self.w = window
        self.page = p = page
        self.vs: V.VoiceSet | None = None
        self.undo_stack: list[V.VoiceSet] = []
        self.redo_stack: list[V.VoiceSet] = []
        p.run.clicked.connect(self.run)
        p.undo.clicked.connect(self.undo)
        p.redo.clicked.connect(self.redo)
        p.swap.clicked.connect(lambda: self._range_op("swap"))
        p.move.clicked.connect(lambda: self._range_op("move"))
        p.clear.clicked.connect(lambda: self._range_op("clear"))
        p.pick.clicked.connect(lambda: self._range_op("pick"))
        p.band.clicked.connect(self.band)
        p.split.clicked.connect(self.split)
        p.merge.clicked.connect(self.merge)
        p.add.clicked.connect(self.add)
        p.set_role.clicked.connect(self.set_role)
        p.confirm.toggled.connect(self.set_confirmed)
        p.voices.itemChanged.connect(self._item_changed)
        p.voice_a.currentIndexChanged.connect(lambda _i: self._sync_confirm())
        window.player.playbackChanged.connect(self._update_range)
        window.session.busyChanged.connect(lambda _b: self.update_actions())

    # --- 상태 -----------------------------------------------------------------------

    def reset(self) -> None:
        self.vs = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.refresh()

    def refresh(self) -> None:
        s = self.w.session
        src = s.pitch_source()
        self.page.source.setText(src["label"] if src else "프로젝트·오디오 없음")
        if s.folder is not None and self.vs is None:
            self.vs = V.load(s.folder.paths.analysis)
        self._fill()

    def _fill(self) -> None:
        from amp.core.result_meta import STATUS_LABELS

        p = self.page
        vs = self.vs
        prev_a, prev_b = p.voice_a.currentData(), p.voice_b.currentData()
        p.voices.blockSignals(True)
        p.voices.clear()
        p.voice_a.blockSignals(True)
        p.voice_b.blockSignals(True)
        p.voice_a.clear()
        p.voice_b.clear()
        if vs is not None:
            for i, v in enumerate(vs.voices):
                med = v.median_midi()
                voiced = 100.0 * v.voiced().mean() if vs.n else 0.0
                it = QTreeWidgetItem([v.display_name, V.ROLES.get(v.role, v.role), "예" if v.user_confirmed else "",
                                      f"{voiced:.0f}%", f"{med:.1f}" if med is not None else "—"])
                it.setData(0, Qt.ItemDataRole.UserRole, v.voice_track_id)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
                it.setCheckState(0, Qt.CheckState.Unchecked if v.hidden else Qt.CheckState.Checked)
                from PySide6.QtGui import QColor

                it.setForeground(0, QColor(VOICE_COLORS[i % len(VOICE_COLORS)]))
                p.voices.addTopLevelItem(it)
                p.voice_a.addItem(v.display_name, v.voice_track_id)
                p.voice_b.addItem(v.display_name, v.voice_track_id)
            for combo, prev, default in ((p.voice_a, prev_a, 0), (p.voice_b, prev_b, 1)):
                i = combo.findData(prev)
                combo.setCurrentIndex(i if i >= 0 else min(default, combo.count() - 1))
            status = STATUS_LABELS.get(vs.meta.status, vs.meta.status)
            unc = int(vs.uncertain.sum())
            if not p.message.text() or p.message.property("auto"):
                p.message.setText(f"성부 {len(vs.voices)}개 · {status} · 불확실 프레임 {unc} · 성부에 넣지 못한 프레임 "
                                  f"{vs.unassigned_frames}")
                p.message.setProperty("auto", True)
        p.voices.blockSignals(False)
        p.voice_a.blockSignals(False)
        p.voice_b.blockSignals(False)
        self._sync_confirm()
        self._update_range()
        self.update_overlays()

    def _sync_confirm(self) -> None:
        v = self.vs.voice(self.page.voice_a.currentData()) if self.vs else None
        self.page.confirm.blockSignals(True)
        self.page.confirm.setChecked(bool(v and v.user_confirmed))
        self.page.confirm.blockSignals(False)
        self.update_actions()

    def _update_range(self) -> None:
        r = self.w.player.loop_region
        self.page.range_label.setText(f"대상 구간: 반복 구간 {r[0]:.2f}–{r[1]:.2f}s · 나누는 곳: 재생 위치" if r
                                      else "대상 구간: 반복 구간을 먼저 정하세요 · 나누는 곳: 재생 위치")
        self.update_actions()

    def update_actions(self) -> None:
        p = self.page
        s = self.w.session
        ready = s.folder is not None and s.audio is not None and not s.busy
        p.run.setEnabled(ready)
        has = self.vs is not None and len(self.vs.voices) > 0
        two = self.vs is not None and len(self.vs.voices) > 1 and p.voice_a.currentData() != p.voice_b.currentData()
        rng = self.w.player.loop_region is not None
        for b in (p.clear, p.pick):
            b.setEnabled(ready and has and rng)
        p.band.setEnabled(ready and has and rng)
        for b in (p.swap, p.move):
            b.setEnabled(ready and two and rng)
        p.merge.setEnabled(ready and two)
        p.split.setEnabled(ready and has)
        p.add.setEnabled(ready and self.vs is not None)
        p.set_role.setEnabled(ready and has)
        p.confirm.setEnabled(ready and has)
        p.undo.setEnabled(bool(self.undo_stack) and not s.busy)
        p.redo.setEnabled(bool(self.redo_stack) and not s.busy)

    # --- 겹쳐 그리기 -------------------------------------------------------------------

    def update_overlays(self) -> None:
        layers = {}
        if self.vs is not None:
            for i, v in enumerate(self.vs.voices):
                if v.hidden:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    midi = 69.0 + 12.0 * np.log2(v.f0_hz / 440.0)
                layers[f"voice:{v.voice_track_id}"] = (self.vs.start_frame, self.vs.frame_rate, midi,
                                                      VOICE_COLORS[i % len(VOICE_COLORS)])
        self.w.precision.set_layers("voices", layers)

    # --- 실행·편집 -----------------------------------------------------------------------

    def run(self) -> None:
        w = self.w
        handle = w.session.run_multipitch(int(self.page.max_voices.currentData()))
        if handle is None:
            return
        self._say("분석 중…")
        handle.finished.connect(self._on_analyzed)
        handle.failed.connect(lambda msg, _tb: self._say(f"분석하지 못했습니다: {msg}"))
        handle.cancelled.connect(lambda: self._say("분석을 취소했습니다."))
        self.update_actions()

    def _on_analyzed(self, vs) -> None:
        if self.vs is not None:
            self._push_undo()
        self.vs = vs
        kept = sum(1 for v in vs.voices if v.edited or v.user_confirmed)
        self.page.message.setProperty("auto", True)
        self.page.message.setText("")
        self._fill()
        if kept:
            self._say(f"성부 {len(vs.voices)}개 (직접 고치거나 확정한 성부 {kept}개는 그대로 둠) · 불확실 프레임 "
                      f"{int(vs.uncertain.sum())}", auto=True)

    def _say(self, text: str, auto: bool = False) -> None:
        self.page.message.setText(text)
        self.page.message.setProperty("auto", auto)
        self.w.statusBar().showMessage(text, 6000)

    def _push_undo(self) -> None:
        self.undo_stack.append(self.vs)
        del self.undo_stack[:-MAX_UNDO]
        self.redo_stack.clear()

    def _apply(self, new: V.VoiceSet, text: str) -> None:
        self._push_undo()
        self.vs = new
        self._save()
        self._say(text)
        self._fill()

    def _save(self) -> None:
        folder = self.w.session.folder
        if folder is not None and self.vs is not None:
            V.save(folder.paths.analysis, self.vs)

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self.vs)
        self.vs = self.undo_stack.pop()
        self._save()
        self._say("되돌렸습니다.")
        self._fill()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self.vs)
        self.vs = self.redo_stack.pop()
        self._save()
        self._say("다시 했습니다.")
        self._fill()

    def _ab(self):
        return self.page.voice_a.currentData(), self.page.voice_b.currentData()

    def _range_op(self, kind: str) -> None:
        r = self.w.player.loop_region
        if self.vs is None or r is None:
            return
        a, b = self._ab()
        t0, t1 = float(r[0]), float(r[1])
        if kind == "swap":
            self._apply(V.swap_range(self.vs, a, b, t0, t1), f"{t0:.2f}–{t1:.2f}s에서 두 성부를 바꿨습니다.")
        elif kind == "move":
            new, conflicts = V.move_range(self.vs, a, b, t0, t1)
            self._apply(new, "옮겼습니다." + (f" B가 이미 있는 {conflicts}프레임은 그대로 두었습니다." if conflicts else ""))
        elif kind == "clear":
            self._apply(V.clear_range(self.vs, a, t0, t1), f"{t0:.2f}–{t1:.2f}s를 지웠습니다.")
        elif kind == "pick":
            new, n = V.pick_candidate(self.vs, a, t0, t1, float(self.page.target.value()))
            if n == 0:
                self._say("그 높이 근처(±150센트)에 후보가 없습니다.")
                return
            self._apply(new, f"{n}프레임을 다른 후보로 바꿨습니다.")

    def split(self) -> None:
        if self.vs is None:
            return
        t = float(self.w.player.position())
        new, vid = V.split_voice(self.vs, self._ab()[0], t)
        self._apply(new, f"{t:.2f}s에서 나눴습니다 ({vid}).")

    def merge(self) -> None:
        a, b = self._ab()
        if self.vs is None or a == b:
            return
        new, conflicts = V.merge_voices(self.vs, a, b)
        self._apply(new, "합쳤습니다." + (f" 겹친 {conflicts}프레임은 A 값을 남겼습니다." if conflicts else ""))

    def add(self) -> None:
        if self.vs is None:
            return
        new, vid = V.add_voice(self.vs)
        self._apply(new, f"성부를 더했습니다 ({vid}). 다른 성부에서 구간을 옮겨 채우세요.")

    def set_role(self) -> None:
        if self.vs is None:
            return
        self._apply(V.set_attr(self.vs, self._ab()[0], role=self.page.role.currentData()), "역할을 정했습니다.")

    def set_confirmed(self, on: bool) -> None:
        if self.vs is None or self.vs.voice(self._ab()[0]) is None:
            return
        self._apply(V.set_attr(self.vs, self._ab()[0], user_confirmed=bool(on)),
                    "확정했습니다. 다시 분석해도 이 성부는 그대로 둡니다." if on else "확정을 풀었습니다.")

    def _item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        if self.vs is None:
            return
        vid = item.data(0, Qt.ItemDataRole.UserRole)
        v = self.vs.voice(vid)
        if v is None:
            return
        hidden = item.checkState(0) != Qt.CheckState.Checked
        if hidden != v.hidden:
            self.vs = V.set_attr(self.vs, vid, hidden=hidden)  # 보이기만 바꿈: 되돌리기 기록에 넣지 않는다
            self._save()
            self.update_overlays()
            return
        name = item.text(0).strip()
        if name and name != v.display_name:
            self._apply(V.set_attr(self.vs, vid, display_name=name), f"이름을 '{name}'(으)로 바꿨습니다.")

    def band(self) -> None:
        r = self.w.player.loop_region
        if self.vs is None or r is None:
            return
        lo, hi = sorted((float(self.page.band_lo.value()), float(self.page.band_hi.value())))
        fmin, fmax = 440.0 * 2 ** ((lo - 69) / 12), 440.0 * 2 ** ((hi - 69) / 12)
        vid = self._ab()[0]
        handle = self.w.session.run_voice_band(self.vs, vid, float(r[0]), float(r[1]), fmin, fmax)
        if handle is None:
            return
        self._say("음역을 정해 다시 분석 중…")
        handle.finished.connect(lambda res: self._apply(res[0], f"{res[1]}프레임을 다시 찾았습니다 (MIDI {lo:.0f}–{hi:.0f})."))
        handle.failed.connect(lambda msg, _tb: self._say(f"다시 분석하지 못했습니다: {msg}"))
