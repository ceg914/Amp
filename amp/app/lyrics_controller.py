"""가사 기능을 메인 창에 잇는 컨트롤러.

메인 창이 커지지 않도록 가사 패널·가사 레인·피아노 롤 강조·재생 연동·가사 메뉴를 여기서 다룬다.
모든 가사 편집은 ``ProjectDocument.push_lyrics`` 로 실행 취소 기록에 들어간다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from amp.app.document import ProjectDocument
from amp.app.player import PlaybackState
from amp.core.errors import EditError
from amp.lyrics import engine as lyr_engine
from amp.lyrics import linking, ops

log = logging.getLogger(__name__)
TXT_MAX_BYTES = 2 * 2**20


def read_text_file(path: str) -> str:
    """TXT 가사 읽기: UTF-8(BOM 포함) → UTF-16(BOM) → CP949(한국어 윈도 메모장 기본) 순서로 시도한다."""
    data = Path(path).read_bytes()
    if len(data) > TXT_MAX_BYTES:
        raise EditError("가사 파일이 너무 큽니다 (2MB 초과).")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise EditError("가사 파일의 문자 인코딩을 알 수 없습니다. UTF-8로 저장해 주세요.")


class LyricsController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.w = window
        self.doc: ProjectDocument = window.doc
        self.session = window.session
        self.panel = window.lyrics_panel
        self.lane = window.lyric_lane
        self.scene = window.view.roll
        self.settings: QSettings = window.settings
        self.loop: tuple[float, float] | None = None
        self._build_actions()
        self._connect()
        self.panel.set_settings({k: self.settings.value(f"lyrics/{k}", d) for k, d in
                                 (("language", "auto"), ("model", "auto"))}
                                | {"link_after": str(self.settings.value("lyrics/link_after", "true")) != "false"})
        self.refresh_install_status()
        self.panel.set_state(self.doc.lyrics, force_editor=True)

    # --- 구성 -----------------------------------------------------------------

    def _build_actions(self) -> None:
        A = self.w._action
        self.act_recognize = A("가사 자동 인식 (Whisper)", self.recognize, "F6")
        self.act_align = A("가사 시간 정렬 (전체)", lambda: self.align("all"), "F7")
        self.act_align_changed = A("바뀐 부분만 다시 정렬", lambda: self.align("changed"), "Shift+F7")
        self.act_auto_link = A("음표와 자동 연결", self.auto_link)
        self.act_import = A("가사 TXT 가져오기…", self.import_txt)
        self.act_offset = A("가사 전체 시간 이동 (오프셋 보정)…", self.shift_all)
        self.act_loop = A("선택 음절 구간 반복 재생", lambda: self.on_syllable_action("loop", self.lane.selected), "Ctrl+R")
        self.act_loop_off = A("구간 반복 끄기", self.stop_loop)
        self.act_show = A("가사 표시", self.set_visible, "Ctrl+L", checkable=True, checked=True)
        self.act_models = A("모델 파일 폴더 열기", self.open_models)

    def build_menu(self, menubar, before=None) -> None:
        m = menubar.addMenu("가사(&Y)") if before is None else menubar.insertMenu(before, None)
        for a in (self.act_recognize, self.act_align, self.act_align_changed, self.act_auto_link, None, self.act_import,
                  self.act_offset, None, self.act_loop, self.act_loop_off, None, self.act_show, self.act_models):
            m.addSeparator() if a is None else m.addAction(a)
        self.menu = m

    def _connect(self) -> None:
        d, p, lane = self.doc, self.panel, self.lane
        d.lyricsChanged.connect(self.on_lyrics_changed)
        d.projectReset.connect(lambda: self.panel.set_state(self.doc.lyrics, force_editor=True))
        self.session.recognitionReady.connect(self.on_recognition)
        self.session.alignmentReady.connect(self.on_alignment)
        self.session.busyChanged.connect(lambda _b: self.update_actions())
        p.recognizeRequested.connect(self.recognize)
        p.applyRecognitionRequested.connect(self.apply_recognition)
        p.applyTextRequested.connect(self.apply_text)
        p.importTxtRequested.connect(self.import_txt)
        p.repeatLinesRequested.connect(self.repeat_lines)
        p.reviewDoneRequested.connect(self.review_done)
        p.alignRequested.connect(self.align)
        p.autoLinkRequested.connect(self.auto_link)
        p.syllableActionRequested.connect(self.on_syllable_action)
        p.syllableActivated.connect(self.activate_syllable)
        p.openModelFolderRequested.connect(self.open_models)
        p.settingsChanged.connect(self._save_settings)
        p.link_after.toggled.connect(lambda _v: self._save_settings())
        lane.syllablesSelected.connect(self.on_syllables_selected)
        lane.seekRequested.connect(self.w.seek)
        lane.playFromRequested.connect(self.play_from)
        lane.actionRequested.connect(self.on_syllable_action)
        lane.editFailed.connect(self.w._show_status_error)
        self.w.view.roll.selectionChanged.connect(self.on_notes_selected)

    def _save_settings(self) -> None:
        for k, v in self.panel.settings().items():
            self.settings.setValue(f"lyrics/{k}", "true" if v is True else "false" if v is False else v)

    def refresh_install_status(self) -> None:
        st = self.session.lyrics_install_status()
        parts = []
        if not st["whisper"]:
            parts.append("가사 인식 라이브러리(openai-whisper)가 설치되어 있지 않습니다. install_lyrics.bat을 한 번 실행하세요.")
        elif st["whisper_models"]:
            parts.append("내려받은 인식 모델: " + ", ".join(st["whisper_models"]))
        else:
            parts.append("처음 인식할 때 모델을 내려받습니다 (기본 turbo 약 1.5GB).")
        if not st["torch"]:
            parts.append("PyTorch가 없어 인식·정렬을 쓸 수 없습니다 (install_separation.bat).")
        parts.append("정렬 모델: " + ("내려받음" if st["mms"] else "처음 정렬할 때 약 1.2GB를 내려받습니다")
                     + " · MMS_FA 가중치는 CC-BY-NC 4.0(비상업적 이용) 라이선스입니다.")
        self.panel.set_install_status("\n".join(parts))

    def update_actions(self) -> None:
        is_open = self.session.is_open
        busy = self.session.busy
        has_text = not self.doc.lyrics.text.is_empty
        for a in (self.act_recognize, self.panel.recognize):
            a.setEnabled(is_open and not busy)
        for a in (self.act_align, self.act_align_changed, self.panel.align_all, self.panel.align_changed):
            a.setEnabled(is_open and not busy and has_text)
        for a in (self.act_auto_link, self.panel.auto_link, self.act_offset):
            a.setEnabled(has_text)
        self.act_import.setEnabled(is_open)
        self.act_loop_off.setEnabled(self.loop is not None)

    # --- 상태 반영 -------------------------------------------------------------

    def on_lyrics_changed(self) -> None:
        self.panel.set_state(self.doc.lyrics)
        self.update_actions()
        self.on_syllables_selected(self.lane.selected, select_notes=False)

    def set_visible(self, on: bool) -> None:
        self.scene.set_lyrics_visible(on)
        self.lane.setVisible(on)
        self.w.lyric_label.setVisible(on)

    def on_position(self, t: float) -> None:
        sid = self.lane.set_playhead(t)
        notes = set(self.doc.lyrics.alignment.linked_notes(sid)) if sid else set()
        self.scene.set_highlights(current=notes)
        if self.loop is not None and self.w.player.state() is PlaybackState.PLAYING and t >= self.loop[1]:
            self.w.seek(self.loop[0])

    def on_syllables_selected(self, ids: list, select_notes: bool = True) -> None:
        self.panel.show_syllables(list(ids))
        notes = set()
        for sid in ids:
            notes.update(self.doc.lyrics.alignment.linked_notes(sid))
        self.scene.set_highlights(notes=notes)

    def on_notes_selected(self) -> None:
        """음표를 고르면 연결된 음절을 가사 레인에서 강조한다."""
        try:
            ids = self.w.view.selected_ids()
        except RuntimeError:  # 창을 닫는 중 장면이 먼저 사라진 경우
            return
        n2s = self.doc.lyrics.alignment.note_to_syllables
        sids = list(dict.fromkeys(s for n in ids for s in n2s.get(n, ())))
        if sids:
            self.lane.select(sids, emit=False)
            self.panel.show_syllables(sids)

    def activate_syllable(self, sid: str) -> None:
        self.lane.select([sid])
        t = self.doc.lyrics.alignment.timing(sid)
        if t.aligned:
            self.w.seek(t.start)
            self.w.reveal_time(t.start)

    def play_from(self, t: float) -> None:
        self.w.seek(t)
        if self.w.player.state() is not PlaybackState.PLAYING:
            self.w.toggle_play()

    def stop_loop(self) -> None:
        self.loop = None
        self.update_actions()
        self.w.statusBar().showMessage("구간 반복을 껐습니다.", 3000)

    # --- 가사 인식 -------------------------------------------------------------

    def _analysis_settings(self):
        return self.w.inspector.analysis.settings()

    def recognize(self) -> None:
        self._save_settings()
        self.session.run_recognition(self.panel.settings(), self._analysis_settings())

    def on_recognition(self, rec, reused: bool) -> None:
        state = self.doc.lyrics
        flagged = sum(1 for s in rec.segments if s.review)
        summary = (f"인식 구간 {len(rec.segments)}개, 검토 필요 {flagged}개 · 모델 {rec.model} · 언어 {rec.language}"
                   f" · 장치 {rec.device.upper()}" + (" · 캐시 재사용" if reused else ""))
        if state.text.is_empty:
            new = lyr_engine.text_from_recognition(state, rec, replace_text=True)
            self.doc.push_lyrics(new, "자동 인식 결과로 가사 만들기")
        else:
            box = QMessageBox(self.w)
            box.setWindowTitle("가사 인식 완료")
            box.setText(summary + "\n\n이미 가사가 있습니다. 인식 결과로 바꿀까요?\n"
                        "(바꿔도 '실행 취소'로 되돌릴 수 있고, 인식 원본은 가사 탭에 남습니다.)")
            keep = box.addButton("인식 원본만 보관", QMessageBox.ButtonRole.AcceptRole)
            rep = box.addButton("인식 결과로 바꾸기", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("취소", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(keep)
            box.exec()
            if box.clickedButton() is rep:
                self.doc.push_lyrics(lyr_engine.text_from_recognition(state, rec, replace_text=True),
                                     "자동 인식 결과로 가사 바꾸기")
            elif box.clickedButton() is keep:
                self.doc.push_lyrics(lyr_engine.text_from_recognition(state, rec, replace_text=False), "인식 원본 보관")
            else:
                return
        msg = summary
        if rec.warnings:
            msg += " · " + " ".join(rec.warnings)
        self.w.statusBar().showMessage(msg, 15000)
        if flagged:
            self.panel.filter.setCurrentIndex(self.panel.filter.findData("review"))

    def apply_recognition(self) -> None:
        rec = self.doc.lyrics.recognition
        if rec is None:
            return
        if not self.doc.lyrics.text.is_empty:
            ans = QMessageBox.question(self.w, "인식 결과로 가사 만들기",
                                       "지금 가사를 인식 결과로 바꿉니다. '실행 취소'로 되돌릴 수 있습니다. 계속할까요?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.doc.push_lyrics(lyr_engine.text_from_recognition(self.doc.lyrics, rec, replace_text=True),
                             "자동 인식 결과로 가사 만들기")

    # --- 텍스트 ---------------------------------------------------------------

    def apply_text(self, text: str, origin: str = "manual") -> None:
        try:
            new = ops.apply_text(self.doc.lyrics, text, origin=origin)
        except EditError as e:
            self.w._show_status_error(str(e))
            return
        if not self.doc.push_lyrics(new, "가사 수정"):
            self.panel.set_state(self.doc.lyrics, force_editor=True)
            return
        stale = sum(1 for s in new.text.syllables if new.alignment.timing(s.id).stale or not new.alignment.timing(s.id).aligned)
        if stale and any(new.alignment.timing(s.id).aligned for s in new.text.syllables):
            self.w.statusBar().showMessage(f"가사를 반영했습니다. 다시 정렬할 음절 {stale}개 — '바뀐 부분만'으로 정렬하세요.", 10000)

    def import_txt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.w, "가사 TXT 가져오기", "", "텍스트 (*.txt *.lrc);;모든 파일 (*)")
        if not path:
            return
        try:
            text = read_text_file(path)
        except (OSError, EditError) as e:
            self.w._show_error("가사 파일을 읽지 못했습니다", str(e), "")
            return
        if path.lower().endswith(".lrc"):  # [mm:ss.xx] 시각 표시는 떼고 글자만 쓴다 (시각을 실제 경계로 단정하지 않는다)
            import re

            text = "\n".join(re.sub(r"^\s*(\[[^\]]*\])+", "", ln) for ln in text.splitlines())
        if not self.doc.lyrics.text.is_empty:
            ans = QMessageBox.question(self.w, "가사 가져오기",
                                       f"{os.path.basename(path)}의 가사로 바꿉니다. 같은 글자는 시간·음표 연결을 그대로 남깁니다. 계속할까요?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.apply_text(text, origin="file")

    def _line_ids(self, numbers: list[int]) -> list[str]:
        if self.panel._dirty_editor:
            self.w._show_status_error("편집기 내용을 먼저 '적용'하세요.")
            return []
        lines = self.doc.lyrics.text.lines
        return [lines[i].id for i in numbers if 0 <= i < len(lines)]

    def repeat_lines(self, numbers: list, times: int) -> None:
        ids = self._line_ids(numbers)
        if ids:
            self._push(lambda s: ops.repeat_lines(s, ids, times), "반복 가사 지정")

    def review_done(self, numbers: list) -> None:
        ids = self._line_ids(numbers)
        if ids:
            self._push(lambda s: ops.set_line_review(s, ids, ""), "검토 완료")

    def _push(self, fn, label: str) -> bool:
        try:
            return self.doc.push_lyrics(fn(self.doc.lyrics), label)
        except EditError as e:
            self.w._show_status_error(str(e))
            return False

    # --- 정렬·연결 -------------------------------------------------------------

    def align(self, mode: str = "all", ids: list[str] | None = None) -> None:
        self.session.run_alignment(self.panel.settings(), self._analysis_settings(), mode, ids)

    def on_alignment(self, out) -> None:
        state = self.doc.lyrics
        duration = self.session.audio.duration if self.session.audio else None
        new, aligned, failed = lyr_engine.apply_results(state, out.results, out.targets, out.info, out.settings, duration)
        if self.panel.link_after.isChecked() and len(self.doc.score) and aligned:
            new = linking.auto_link(new, self.doc.score.notes(), only=aligned)
        self.doc.push_lyrics(new, "가사 시간 정렬")
        low = sum(1 for sid in aligned if (new.alignment.timing(sid).confidence or 0) < 0.1)
        msg = f"정렬 {len(aligned)}음절 · 실패 {len(failed)}음절 · 신뢰도 낮음 {low}음절 · 장치 {out.info.get('device', '').upper()}"
        if out.reused:
            msg += " · 정렬 모델 출력 재사용"
        if out.warnings:
            msg += " · " + " / ".join(out.warnings[:3])
        self.w.statusBar().showMessage(msg, 15000)
        if failed:
            self.panel.filter.setCurrentIndex(self.panel.filter.findData("unaligned"))

    def auto_link(self) -> None:
        if not len(self.doc.score):
            self.w._show_status_error("연결할 음표가 없습니다. 먼저 보컬 분석으로 음표를 만드세요.")
            return
        self._push(lambda s: linking.auto_link(s, self.doc.score.notes()), "음표 자동 연결")

    def shift_all(self) -> None:
        ms, ok = QInputDialog.getDouble(self.w, "가사 전체 시간 이동",
                                        "모든 음절을 이만큼 옮깁니다 (밀리초, 음수는 앞으로):", 0.0, -5000.0, 5000.0, 0)
        if ok and ms:
            dur = self.session.audio.duration if self.session.audio else None
            self._push(lambda s: ops.shift_times(s, None, ms / 1000.0, duration=dur), "가사 전체 시간 보정")

    def on_syllable_action(self, kind: str, ids: list) -> None:
        ids = [i for i in ids if i in self.doc.lyrics.text.index]
        if not ids:
            self.w._show_status_error("음절을 먼저 고르세요.")
            return
        st = self.doc.lyrics
        notes_sel = [self.doc.score.get(n) for n in self.w.view.selected_ids() if n in self.doc.score]
        if kind == "loop":
            ts = [st.alignment.timing(i) for i in ids if st.alignment.timing(i).aligned]
            if not ts:
                self.w._show_status_error("정렬된 음절이 아니라 반복할 구간이 없습니다.")
                return
            self.loop = (min(t.start for t in ts), max(t.end for t in ts))
            self.update_actions()
            self.play_from(self.loop[0])
            self.w.statusBar().showMessage(f"구간 반복 {self.loop[0]:.2f}~{self.loop[1]:.2f}초 (끄기: 가사 메뉴)", 6000)
        elif kind in ("link_replace", "link_add"):
            if len(ids) != 1 or not notes_sel:
                self.w._show_status_error("음절 하나와 피아노 롤의 음표(여러 개 가능)를 함께 고르세요.")
                return
            mode = "replace" if kind == "link_replace" else "add"
            self._push(lambda s: linking.set_links(s, ids[0], [n.id for n in notes_sel], mode=mode), "음표 연결 편집")
        elif kind == "link_sequential":
            if not notes_sel:
                self.w._show_status_error("피아노 롤에서 이을 음표를 고르세요.")
                return

            def seq(s):
                new = linking.link_sequential(s, ids, notes_sel, set_times=False)
                untimed = [i for i in ids if not s.alignment.timing(i).aligned and new.alignment.links.get(i)]
                if untimed:
                    new = linking.times_from_links(new, untimed, {n.id: n for n in notes_sel})
                return new

            self._push(seq, "음표와 순서대로 연결")
        elif kind == "confirm":
            self._push(lambda s: linking.confirm_links(s, ids), "연결 확정")
        elif kind == "unlink":
            self._push(lambda s: linking.clear_links(s, ids), "음표 연결 지우기")
        elif kind == "times_from_notes":
            notes = {n.id: n for n in self.doc.score.notes()}
            self._push(lambda s: linking.times_from_links(s, ids, notes), "음표 시간 가져오기")
        elif kind == "split":
            syl = st.text.syllable(ids[0])
            pos, ok = QInputDialog.getInt(self.w, "음절 나누기", f"'{syl.text}'에서 앞 음절의 글자 수:", max(1, len(syl.text) // 2),
                                          1, max(1, len(syl.text) - 1))
            if ok:
                self._push(lambda s: ops.split_syllable(s, ids[0], pos)[0], "음절 나누기")
        elif kind == "merge":
            self._push(lambda s: ops.merge_syllables(s, ids)[0], "음절 합치기")
        elif kind == "realign":
            self.align("selected", ids)
        elif kind == "clear_times":
            self._push(lambda s: ops.clear_times(s, ids), "음절 시간 지우기")

    def open_models(self) -> None:
        path = self.session.lyrics_install_status()["models_dir"]
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

