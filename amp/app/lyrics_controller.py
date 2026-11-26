"""가사 기능을 메인 창에 잇는 컨트롤러.

메인 창이 커지지 않도록 가사 패널·가사 레인·피아노 롤 강조·재생 연동·가사 메뉴를 여기서 다룬다.
모든 가사 편집은 ``ProjectDocument.push_lyrics`` 로 실행 취소 기록에 들어간다.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from amp.core.install_hints import script as install_script
from amp.app.document import ProjectDocument
from amp.app.player import PlaybackState
from amp.core.errors import EditError
from amp.lyrics import engine as lyr_engine
from amp.lyrics import linking, ops, review

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


def device_detail(device: str, stats: dict | None) -> str:
    """'CUDA, NVIDIA GeForce RTX 3070 Ti, VRAM 최대 4.1GB' 처럼 실제로 쓴 장치를 적는다."""
    parts = [(device or "?").upper()]
    stats = stats or {}
    if stats.get("gpu_name"):
        parts.append(str(stats["gpu_name"]))
    peak = stats.get("gpu_peak_bytes")
    if peak is None and stats.get("gpu_peak_mb") is not None:
        peak = float(stats["gpu_peak_mb"]) * 2**20
    if peak:
        parts.append(f"VRAM 최대 {peak / 2**30:.1f}GB")
    return ", ".join(parts)


@dataclass
class LyricsChain:
    """F6 한 번으로 이어지는 인식 → 정렬 → 음표 연결의 진행 상태 (단계별 시간·장치)."""

    align_after: bool
    save_after: bool  # 시작할 때 저장하지 않은 편집이 없었으면 끝난 뒤 프로젝트를 저장한다
    undo_index: int  # 시작할 때 실행 취소 위치 (끝날 때 이 작업의 변경만 있는지 본다)
    stage: str = "recognize"
    pushes: int = 0
    stage_t0: float = field(default_factory=time.monotonic)
    times: dict = field(default_factory=dict)
    devices: dict = field(default_factory=dict)

    def finish_stage(self, stage: str, device: str, stats: dict | None = None) -> None:
        now = time.monotonic()
        self.times[stage] = now - self.stage_t0
        self.devices[stage] = device_detail(device, stats)
        self.stage_t0 = now

    def describe(self, stage: str) -> str:
        if stage not in self.times:
            return "—"
        dev = self.devices.get(stage) or "?"
        return f"{self.times[stage]:.0f}초({dev})"


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
        self.chain: LyricsChain | None = None  # F6 연속 작업 (인식 → 정렬 → 연결)
        self._build_actions()
        self._connect()
        self.panel.set_settings({k: self.settings.value(f"lyrics/{k}", d) for k, d in
                                 (("language", "auto"), ("model", "auto"))}
                                | {k: str(self.settings.value(f"lyrics/{k}", "true")) != "false"
                                   for k in ("link_after", "align_after")})
        self.refresh_install_status()
        self.panel.set_state(self.doc.lyrics, force_editor=True)

    # --- 구성 -----------------------------------------------------------------

    def _build_actions(self) -> None:
        A = self.w._action
        self.act_recognize = A("가사 자동 인식 (Whisper)", self.recognize, "F6")
        self.act_recognize_region = A("선택 구간만 다시 인식", self.recognize_region, "Shift+F6")
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
        for a in (self.act_recognize, self.act_recognize_region, self.act_align, self.act_align_changed, self.act_auto_link,
                  None, self.act_import,
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
        p.align_after.toggled.connect(lambda _v: self._save_settings())
        lane.syllablesSelected.connect(self.on_syllables_selected)
        lane.seekRequested.connect(self.w.seek)
        lane.playFromRequested.connect(self.play_from)
        lane.actionRequested.connect(self.on_syllable_action)
        lane.editFailed.connect(self.w._show_status_error)
        p.relink_mode.currentIndexChanged.connect(lambda _i: setattr(lane, "relink_mode", p.relink_mode.currentData()))
        p.snap_notes.toggled.connect(lambda on: setattr(lane, "snap_enabled", bool(on)))
        self.w.view.roll.selectionChanged.connect(self.on_notes_selected)

    def _save_settings(self) -> None:
        for k, v in self.panel.settings().items():
            self.settings.setValue(f"lyrics/{k}", "true" if v is True else "false" if v is False else v)

    def refresh_install_status(self) -> None:
        st = self.session.lyrics_install_status()
        parts = []
        if not st["whisper"]:
            parts.append(f"가사 인식 라이브러리(openai-whisper)가 설치되어 있지 않습니다. {install_script('lyrics')}을 한 번 실행하세요.")
        elif st["whisper_models"]:
            parts.append("내려받은 인식 모델: " + ", ".join(st["whisper_models"]))
        else:
            parts.append("처음 인식할 때 모델을 내려받습니다 (turbo 약 1.5GB, GPU에서 자동으로 고르는 large-v3 약 2.9GB).")
        if not st["torch"]:
            parts.append(f"PyTorch가 없어 인식·정렬을 쓸 수 없습니다 ({install_script('separation')}).")
        parts.append("정렬 모델: " + ("내려받음" if st["mms"] else "처음 정렬할 때 약 1.2GB를 내려받습니다")
                     + " · MMS_FA 가중치는 CC-BY-NC 4.0(비상업적 이용) 라이선스입니다.")
        self.panel.set_install_status("\n".join(parts))

    def update_actions(self) -> None:
        is_open = self.session.is_open
        busy = self.session.busy
        has_text = not self.doc.lyrics.text.is_empty
        for a in (self.act_recognize, self.panel.recognize):
            a.setEnabled(is_open and not busy)
        self.act_recognize_region.setEnabled(is_open and not busy)
        for a in (self.act_align, self.act_align_changed, self.panel.align_all, self.panel.align_changed):
            a.setEnabled(is_open and not busy and has_text)
        for a in (self.act_auto_link, self.panel.auto_link, self.act_offset):
            a.setEnabled(has_text)
        self.act_import.setEnabled(is_open)
        self.act_loop_off.setEnabled(self.w.player.loop_enabled)

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

    @property
    def loop(self) -> tuple[float, float] | None:
        """지금 켜진 반복 구간 (재생기가 들고 있다. STEP 5부터 재생 스트림 안에서 끊김 없이 되돌아간다)."""
        return self.w.player.loop_region if self.w.player.loop_enabled else None

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
        if self.w.player.state is not PlaybackState.PLAYING:
            self.w.toggle_play()

    def stop_loop(self) -> None:
        self.w.listening.clear_region()
        self.update_actions()
        self.w.statusBar().showMessage("구간 반복을 껐습니다.", 3000)

    # --- 가사 인식 -------------------------------------------------------------

    def _analysis_settings(self):
        return self.w.inspector.analysis.settings()

    def recognize(self) -> None:
        """F6: 가사 인식. '인식한 뒤 바로 시간 정렬'이 켜져 있으면 정렬 → 음표 연결까지 이어서 한다.

        각 단계는 앞 단계가 끝난 뒤에 시작한다. 어느 단계가 실패하거나 취소되어도 그 전에 반영된 결과와
        사용자 편집은 그대로 두고, 어디서부터 다시 하면 되는지 알린다.
        """
        self._save_settings()
        if self.session.busy:
            return
        align_after = self.panel.align_after.isChecked()
        chain = LyricsChain(align_after=align_after, save_after=not self.doc.is_dirty,
                            undo_index=self.doc.undo_stack.index())
        handle = self.session.run_recognition(self.panel.settings(), self._analysis_settings(),
                                              stage="[1/3]" if align_after else "")
        if handle is None:
            return
        self.chain = chain
        self._chain_info("[1/3] 가사 인식 중…" if align_after else "가사 인식 중…")
        handle.failed.connect(lambda msg, _tb, c=chain: self._chain_stopped(c, "recognize", f"실패했습니다 ({msg})"))
        handle.cancelled.connect(lambda c=chain: self._chain_stopped(c, "recognize", "취소했습니다"))

    def region_for_recognition(self) -> tuple[float, float] | None:
        """다시 인식할 구간: 켜 둔 반복 구간 → 고른 음절들의 시간 구간 순서."""
        if self.loop is not None:
            a, b = self.loop
            return (float(a), float(b)) if b > a else None
        al = self.doc.lyrics.alignment
        spans = [al.timing(sid) for sid in self.lane.selected if al.timing(sid).aligned]
        spans = [t for t in spans if t.end is not None and t.start is not None and t.end > t.start]
        if spans:
            return (min(t.start for t in spans), max(t.end for t in spans))
        return None

    def recognize_region(self, region: tuple[float, float] | None = None) -> None:
        """v0.7.0: 선택 구간만 다시 인식해 인식 원본의 그 구간만 바꾼다. 지금 가사 텍스트·시간·연결은 바꾸지 않는다."""
        self._save_settings()
        region = region or self.region_for_recognition()
        if region is None:
            self.w._show_status_error("다시 인식할 구간이 없습니다. 반복 구간을 켜거나 시간 정렬된 음절을 고르세요.")
            return
        handle = self.session.run_recognition_region(self.panel.settings(), self._analysis_settings(), region)
        if handle is None:
            return
        handle.finished.connect(lambda out, r=region: self.on_region_recognition(out[0], r, out[1]))

    def on_region_recognition(self, rec, region: tuple[float, float], reused: bool = False) -> None:
        from dataclasses import replace

        st = self.doc.lyrics
        old_n = len(st.recognition.segments) if st.recognition is not None else 0
        merged = lyr_engine.merge_region(st.recognition, rec, region)
        info = merged.settings["regions"][-1]
        new = replace(st, recognition=merged).with_history("구간 다시 인식", f"{region[0]:.2f}~{region[1]:.2f}초")
        if not self.doc.push_lyrics(new, "구간 다시 인식"):
            return
        text_note = "" if st.text.is_empty else " 지금 가사 텍스트·시간·연결은 그대로입니다 (인식 원본만 바꿈)."
        self.w.statusBar().showMessage(
            f"{region[0]:.1f}~{region[1]:.1f}초를 다시 인식했습니다: 이전 인식 {info['replaced']}줄 → 새 인식 {info['added']}줄"
            f" (전체 {old_n} → {len(merged.segments)}줄){' · 캐시 사용' if reused else ''}.{text_note}", 12000)
        if info["added"]:
            self.panel.filter.setCurrentIndex(self.panel.filter.findData("review"))

    def on_recognition(self, rec, reused: bool) -> None:
        chain = self.chain
        if chain is not None:
            chain.finish_stage("recognize", rec.device, (rec.settings or {}).get("stats"))
        state = self.doc.lyrics
        flagged = sum(1 for s in rec.segments if s.review)
        summary = (f"인식 구간 {len(rec.segments)}개, 검토 필요 {flagged}개 · 모델 {rec.model} · 언어 {rec.language}"
                   f" · 장치 {rec.device.upper()}" + (" · 캐시 재사용" if reused else ""))
        skipped = [seg for seg in rec.segments if review.excluded(seg)]
        if not any(seg.text.strip() for seg in rec.segments if not review.excluded(seg)):
            # 빈 인식 결과로 가사를 지우지 않는다. 인식 원본(빈 것)만 보관한다
            if self.doc.push_lyrics(lyr_engine.text_from_recognition(state, rec, replace_text=False), "인식 원본 보관"):
                self._count_push(chain)
            why = (f"결과가 비어 있습니다 (자막·인사 문구 {len(skipped)}줄만 나와 가사에 넣지 않았습니다)" if skipped else
                   "결과가 비어 있습니다 (보컬이 거의 없거나 '보컬 없는 구간'에 덮였을 수 있습니다)")
            self._chain_stopped(chain, "recognize", why, error=False)
            return
        replaced = False
        if state.text.is_empty:
            new = lyr_engine.text_from_recognition(state, rec, replace_text=True)
            replaced = self.doc.push_lyrics(new, "자동 인식 결과로 가사 만들기")
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
                replaced = self.doc.push_lyrics(lyr_engine.text_from_recognition(state, rec, replace_text=True),
                                                "자동 인식 결과로 가사 바꾸기")
            elif box.clickedButton() is keep:
                if self.doc.push_lyrics(lyr_engine.text_from_recognition(state, rec, replace_text=False), "인식 원본 보관"):
                    self._count_push(chain)
                self._chain_end(chain, "인식 원본만 보관했습니다. 지금 가사·시간·연결은 그대로입니다.")
                return
            else:
                self._chain_end(chain, "인식 결과를 반영하지 않았습니다. 지금 가사·시간·연결은 그대로입니다.")
                return
        if replaced:
            self._count_push(chain)
        msg = summary
        if skipped and replaced:
            msg += (f" · 노래와 상관없는 자막·인사 문구 {len(skipped)}줄(" + ", ".join(f"'{x.text}'" for x in skipped[:2])
                    + ")은 가사에 넣지 않았습니다 (인식 원본에는 남아 있음)")
        if rec.warnings:
            msg += " · " + " ".join(rec.warnings)
        if replaced and not self.panel.align_after.isChecked():
            msg += " · 음절 시간은 단어 시각을 나눈 추정값입니다. [전체 정렬](F7)로 맞추세요."
        self.w.statusBar().showMessage(msg, 15000)
        if flagged:
            self.panel.filter.setCurrentIndex(self.panel.filter.findData("review"))
        if replaced and self.panel.align_after.isChecked():
            self._align_after_recognition(chain)
        else:
            self._chain_end(chain, msg)

    def _align_after_recognition(self, chain: "LyricsChain | None" = None) -> None:
        """인식 결과로 가사를 만든 뒤 정렬까지 잇는다. 인식 작업이 끝났다고 알려진 뒤(busy 해제)에 시작한다."""
        if not self.panel.align_after.isChecked() or self.doc.lyrics.text.is_empty:
            self._chain_end(chain, "")
            return
        if not self.session.busy:
            self._start_chain_alignment(chain)
            return

        def go(busy: bool) -> None:
            if busy:
                return
            self.session.busyChanged.disconnect(go)
            if not self.doc.lyrics.text.is_empty:
                self._start_chain_alignment(chain)

        self.session.busyChanged.connect(go)

    def _start_chain_alignment(self, chain: "LyricsChain | None") -> None:
        if chain is None:  # 인식 결과를 직접 다시 적용한 경우 등
            chain = LyricsChain(align_after=True, save_after=False, undo_index=self.doc.undo_stack.index())
            chain.stage = "recognize"
            self.chain = chain
        chain.stage = "align"
        chain.stage_t0 = time.monotonic()
        handle = self.session.run_alignment(self.panel.settings(), self._analysis_settings(), "all", None, stage="[2/3]")
        if handle is None:
            self._chain_stopped(chain, "align", "을 시작하지 못했습니다")
            return
        self._chain_info("[2/3] 시간 정렬 중…")
        handle.failed.connect(lambda msg, _tb, c=chain: self._chain_stopped(c, "align", f"이 실패했습니다 ({msg})"))
        handle.cancelled.connect(lambda c=chain: self._chain_stopped(c, "align", "을 취소했습니다"))

    # --- 연속 작업 상태 -------------------------------------------------------------

    def _chain_info(self, text: str) -> None:
        self.panel.set_align_info(text)

    def _count_push(self, chain: "LyricsChain | None") -> None:
        if chain is not None:
            chain.pushes += 1

    def _chain_stopped(self, chain: "LyricsChain | None", stage: str, reason: str, error: bool = True) -> None:
        """단계가 실패·취소되면 그 전 결과는 두고, 어디서부터 다시 하면 되는지 알린다."""
        if chain is None or chain is not self.chain:
            return
        self.chain = None
        if stage == "recognize":
            text = (f"가사 인식이 {reason}. 가사·시간·음표 연결은 바뀌지 않았습니다. 다시 하려면 [가사 인식](F6).")
        elif stage == "align":
            text = (f"시간 정렬{reason}. 인식한 가사는 남아 있고, 음절 시간은 인식 모델의 단어 시각을 나눈 추정값입니다"
                    "(가사 레인의 점선 테두리). 정렬부터 다시 하려면 [전체 정렬](F7).")
        else:
            text = (f"음표 연결이 {reason}. 시간 정렬 결과는 반영했습니다. 다시 하려면 [음표 자동 연결].")
        log.info("가사 연속 작업 멈춤 (%s): %s", stage, text)
        self._chain_info(text)
        self.w.statusBar().showMessage(text, 20000)

    def _chain_end(self, chain: "LyricsChain | None", message: str) -> None:
        if chain is None or chain is not self.chain:
            return
        self.chain = None
        if message:
            self._chain_info(message)

    def _chain_done(self, chain: "LyricsChain", aligned: int, linked: int, total: int) -> None:
        self.chain = None
        parts = [f"전체 작업 완료 · 인식 {chain.describe('recognize')} · 정렬 {chain.describe('align')}"]
        if "link" in chain.times:
            parts.append(f"연결 {chain.times['link']:.1f}초")
        parts.append(f"음절 {total}개 중 정렬 {aligned}, 연결 {linked}")
        ours_only = self.doc.undo_stack.index() == chain.undo_index + chain.pushes
        if chain.save_after and ours_only:
            parts.append("프로젝트 저장함" if self.w.save() else "저장하지 못함 — Ctrl+S로 다시 저장하세요")
        else:
            self.w.autosave.save_now()
            parts.append("저장 안 함 (작업 전후에 저장하지 않은 다른 편집이 있어 Ctrl+S로 직접 저장하세요. 복구 파일은 써 둠)")
        text = " · ".join(parts)
        log.info("가사 연속 작업: %s", text)
        self._chain_info(text)
        self.w.statusBar().showMessage(text, 20000)

    def apply_recognition(self) -> None:
        rec = self.doc.lyrics.recognition
        if rec is None:
            return
        if not self.doc.lyrics.text.is_empty:
            ans = QMessageBox.question(self.w, "인식 결과로 가사 만들기",
                                       "지금 가사를 인식 결과로 바꿉니다. '실행 취소'로 되돌릴 수 있습니다. 계속할까요?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        if self.doc.push_lyrics(lyr_engine.text_from_recognition(self.doc.lyrics, rec, replace_text=True),
                                "자동 인식 결과로 가사 만들기"):
            self._align_after_recognition(None)

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
        chain = self.chain if self.chain is not None and self.chain.stage == "align" else None
        if chain is not None:
            chain.finish_stage("align", str(out.info.get("device", "")), out.info.get("stats"))
        state = self.doc.lyrics
        duration = self.session.audio.duration if self.session.audio else None
        new, aligned, failed = lyr_engine.apply_results(state, out.results, out.targets, out.info, out.settings, duration)
        link_error = None
        if self.panel.link_after.isChecked() and len(self.doc.score) and aligned:
            if chain is not None:
                self._chain_info("[3/3] 음표 연결 중…")
            t0 = time.monotonic()
            try:
                new = linking.auto_link(new, self.doc.score.notes(), only=aligned)
            except Exception as e:  # noqa: BLE001 - 연결 실패가 정렬 결과 반영을 막지 않게
                log.exception("음표 자동 연결 실패")
                link_error = str(e) or type(e).__name__
            if chain is not None:
                chain.times["link"] = time.monotonic() - t0
        if self.doc.push_lyrics(new, "가사 시간 정렬"):
            self._count_push(chain)
        low = sum(1 for sid in aligned if (new.alignment.timing(sid).confidence or 0) < 0.1)
        msg = f"정렬 {len(aligned)}음절 · 실패 {len(failed)}음절 · 신뢰도 낮음 {low}음절 · 장치 {out.info.get('device', '').upper()}"
        if out.reused:
            msg += " · 정렬 모델 출력 재사용"
        if out.warnings:
            msg += " · " + " / ".join(out.warnings[:3])
        self.w.statusBar().showMessage(msg, 15000)
        if failed:
            self.panel.filter.setCurrentIndex(self.panel.filter.findData("unaligned"))
        if link_error is not None:
            if chain is None:
                chain = LyricsChain(align_after=True, save_after=False, undo_index=0)
                self.chain = chain
            self._chain_stopped(chain, "link", f"실패했습니다 ({link_error})")
        elif chain is not None:
            ly = self.doc.lyrics
            total = len(ly.text.syllables)
            self._chain_done(chain, total - len(ly.unaligned()), total - len(ly.unlinked()), total)

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
        if kind.startswith(("pron_", "mel_")):  # STEP 4: 발음·멜리스마는 VocalDetailController가 맡는다
            self.w.vocal.on_action(kind, ids)
            return
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
            self.w.listening.loop_range(min(t.start for t in ts), max(t.end for t in ts))
            self.update_actions()
            if self.loop is not None:
                a, b = self.loop
                self.w.statusBar().showMessage(f"구간 반복 {a:.2f}~{b:.2f}초 (앞뒤 여유 {self.w.listening.padding:g}초 포함, "
                                               "끄기: Alt+L)", 6000)
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
        elif kind == "toggle_approach":
            links = st.alignment.links.get(ids[0], ())
            marked = [lk.note_id for lk in links if lk.role == "approach"]
            if marked:
                self._push(lambda s: linking.set_approach(s, ids[0], marked[0], False), "접근음 표시 해제")
            else:
                notes = sorted((self.doc.score.get(lk.note_id) for lk in links if lk.note_id in self.doc.score),
                               key=lambda n: n.start)
                if len(notes) < 2:
                    self.w._show_status_error("음표가 두 개 이상 연결된 음절에서만 접근음을 지정할 수 있습니다.")
                    return
                self._push(lambda s: linking.set_approach(s, ids[0], notes[0].id, True), "접근음 표시")
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

