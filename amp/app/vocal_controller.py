"""발음·가창 길이·멜리스마 화면 연결 (STEP 4).

분석은 ``amp.lyrics.vocal``·``amp.lyrics.pronunciation`` (Qt 비의존)이 하고, 여기서는
- 문서가 바뀔 때 판정을 다시 계산해 피아노 롤·가사 레인·가사 패널에 넘기고 (캐시)
- 패널·레인 메뉴의 발음/멜리스마 편집을 실행 취소 가능한 가사 편집(push_lyrics)으로 바꾸고
- 보기 메뉴의 표시 방식(가사만·발음만·가사+발음·가사+IPA·전체, 멜리스마 강조, 음절 길이)을 관리한다.
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QAction, QActionGroup

from amp.lyrics import pronunciation, vocal

log = logging.getLogger(__name__)

DISPLAY_MODES = (("lyrics", "가사만"), ("pron", "발음만"), ("lyrics_pron", "가사 + 발음"), ("lyrics_ipa", "가사 + IPA"),
                 ("all", "전체 (가사·발음·IPA)"))
LANE_SUB = {"lyrics": "none", "pron": "pron", "lyrics_pron": "pron", "lyrics_ipa": "ipa", "all": "pron"}
VOCAL_ACTIONS = {"pron_confirm", "pron_reset", "pron_regen", "pron_regen_all", "mel_set", "mel_not", "mel_clear", "mel_add",
                 "mel_remove", "mel_reviewed"}


class VocalDetailController(QObject):
    analysisUpdated = Signal()

    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.doc = window.doc
        self.panel = window.lyrics_panel
        self.lane = window.lyric_lane
        self.scene = window.view.roll
        self.settings: QSettings = window.settings
        self.groups: dict = {}
        self.durations: dict = {}
        self.last_refresh_ms = 0.0
        mode = str(self.settings.value("vocal/display", "lyrics"))
        self.mode = mode if mode in dict(DISPLAY_MODES) else "lyrics"
        self.emphasis = str(self.settings.value("vocal/melisma_emphasis", "true")) != "false"
        self.show_lengths = str(self.settings.value("vocal/show_lengths", "false")) == "true"
        self._build_actions()
        self._connect()
        self._apply_display()
        self.refresh()

    # --- 구성 -----------------------------------------------------------------

    def _build_actions(self) -> None:
        self.mode_group = QActionGroup(self)
        self.mode_actions: dict[str, QAction] = {}
        for key, label in DISPLAY_MODES:
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(key == self.mode)
            a.triggered.connect(lambda _c=False, k=key: self.set_mode(k))
            self.mode_group.addAction(a)
            self.mode_actions[key] = a
        self.act_emphasis = QAction("멜리스마 강조 (괄호선·판정 불확실 점선)", self)
        self.act_emphasis.setCheckable(True)
        self.act_emphasis.setChecked(self.emphasis)
        self.act_emphasis.toggled.connect(self.set_emphasis)
        self.act_lengths = QAction("음절 구간 길이 표시 (가사 레인, ms)", self)
        self.act_lengths.setCheckable(True)
        self.act_lengths.setChecked(self.show_lengths)
        self.act_lengths.toggled.connect(self.set_show_lengths)
        self.act_regen_all = QAction("발음 전체 다시 생성 (자동 발음만)", self)
        self.act_regen_all.triggered.connect(lambda: self.on_action("pron_regen_all", []))

    def build_menu(self, view_menu, lyrics_menu=None) -> None:
        m = view_menu.addMenu("가사·발음 표시")
        for key, _label in DISPLAY_MODES:
            m.addAction(self.mode_actions[key])
        m.addSeparator()
        m.addAction(self.act_emphasis)
        m.addAction(self.act_lengths)
        self.display_menu = m
        if lyrics_menu is not None:
            lyrics_menu.addSeparator()
            lyrics_menu.addAction(self.act_regen_all)

    def _connect(self) -> None:
        d = self.doc
        d.lyricsChanged.connect(self.refresh)
        d.notesReplaced.connect(lambda *_a: self.refresh())
        d.analysisChanged.connect(self.refresh)
        d.projectReset.connect(self.refresh)
        self.w.inspector.analysis.settingsChanged.connect(lambda _s: self.refresh())
        self.panel.pronunciationEdited.connect(self.set_pronunciation)
        self.panel.vocalActionRequested.connect(self.on_action)

    # --- 표시 -----------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.settings.setValue("vocal/display", mode)
        self._apply_display()

    def set_emphasis(self, on: bool) -> None:
        self.emphasis = on
        self.settings.setValue("vocal/melisma_emphasis", "true" if on else "false")
        self._apply_display()

    def set_show_lengths(self, on: bool) -> None:
        self.show_lengths = on
        self.settings.setValue("vocal/show_lengths", "true" if on else "false")
        self._apply_display()

    def _apply_display(self) -> None:
        self.mode_actions[self.mode].setChecked(True)
        self.scene.set_display(self.mode, self.emphasis)
        self.lane.set_display(LANE_SUB[self.mode], self.show_lengths)

    # --- 분석 (캐시) ------------------------------------------------------------

    def refresh(self) -> None:
        """음표·가사·피치에서 판정과 길이를 다시 계산한다. 음표·가사 데이터는 읽기만 한다."""
        t0 = time.perf_counter()
        ly = self.doc.lyrics
        notes = self.doc.score.notes()
        an = self.doc.project.analysis
        curve = an.pitch_curve if an is not None else None
        onset = an.onset_envelope if an is not None else None
        ranges = self.w.inspector.analysis.settings().no_vocal_ranges
        try:
            self.groups = vocal.analyze(ly, notes, curve, onset, ranges)
            self.durations = vocal.all_durations(ly, {n.id: n for n in notes}, curve, ranges)
        except Exception:  # noqa: BLE001 - 판정이 실패해도 편집 화면은 살린다 (데이터는 바꾸지 않았다)
            log.exception("멜리스마·길이 분석 실패")
            self.groups, self.durations = {}, {}
        self.scene.set_vocal_groups(self.groups)
        self.lane.set_vocal_groups(self.groups)
        self.panel.set_vocal(self.groups, self.durations)
        self.last_refresh_ms = (time.perf_counter() - t0) * 1000
        self.analysisUpdated.emit()

    def melisma_records(self) -> list[dict]:
        return vocal.melisma_records(self.groups)

    # --- 편집 -----------------------------------------------------------------

    def _push(self, new, text: str) -> bool:
        return self.doc.push_lyrics(new, text)

    def set_pronunciation(self, sid: str, value: str, ipa) -> None:
        if sid not in self.doc.lyrics.text.index:
            return
        self._push(pronunciation.set_user(self.doc.lyrics, sid, value, ipa), "발음 수정")

    def _selected_notes(self) -> list[str]:
        try:
            return [n for n in self.w.view.selected_ids() if n in self.doc.score]
        except RuntimeError:
            return []

    def on_action(self, kind: str, ids: list) -> None:
        st = self.doc.lyrics
        ids = [i for i in ids if i in st.text.index]
        status = self.w.statusBar()
        if kind == "pron_regen_all":
            new, n, kept = pronunciation.regenerate(st)
            self._push(new, "발음 다시 생성")
            status.showMessage(f"자동 발음 {n}개를 다시 만들었습니다. 직접 고치거나 확정한 발음 {kept}개는 그대로 두었습니다.", 8000)
            return
        if not ids:
            self.w._show_status_error("음절을 먼저 고르세요.")
            return
        notes_by_id = {n.id: n for n in self.doc.score.notes()}
        sid = ids[0]
        try:
            if kind == "pron_confirm":
                self._push(pronunciation.confirm(st, ids), "발음 확정")
            elif kind == "pron_reset":
                self._push(pronunciation.reset(st, ids), "기본 발음으로 되돌림")
            elif kind == "pron_regen":
                new, n, kept = pronunciation.regenerate(st, ids)
                self._push(new, "발음 다시 생성")
                status.showMessage(f"자동 발음 {n}개를 다시 만들었습니다"
                                   + (f". 직접 고친 발음 {kept}개는 그대로 두었습니다('기본 발음으로'로 되돌릴 수 있음)." if kept else "."),
                                   8000)
            elif kind == "mel_set":
                chosen = self._selected_notes()
                if len(chosen) < 2:
                    g = self.groups.get(sid)
                    chosen = list(g.note_ids) if g is not None else []
                if len(chosen) < 2:
                    self.w._show_status_error("멜리스마로 정하려면 음표가 2개 이상 필요합니다. 피아노 롤에서 음표를 고르세요.")
                    return
                self._push(vocal.set_melisma(st, sid, chosen, notes_by_id), "멜리스마 지정")
            elif kind == "mel_not":
                self._push(vocal.set_not_melisma(st, sid), "멜리스마 해제")
            elif kind == "mel_clear":
                self._push(vocal.clear_decision(st, ids), "멜리스마 자동 판정으로")
            elif kind in ("mel_add", "mel_remove"):
                chosen = self._selected_notes()
                if not chosen:
                    self.w._show_status_error("피아노 롤에서 더하거나 뺄 음표를 고르세요.")
                    return
                fn = vocal.add_notes if kind == "mel_add" else vocal.remove_notes
                self._push(fn(st, sid, chosen, notes_by_id, self.groups.get(sid)),
                           "멜리스마 음표 추가" if kind == "mel_add" else "멜리스마 음표 빼기")
            elif kind == "mel_reviewed":
                self._push(vocal.mark_reviewed(st, ids), "멜리스마 판정 확인")
        except ValueError as e:
            self.w._show_status_error(str(e))
