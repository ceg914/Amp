"""가사 패널 (오른쪽 '가사' 탭).

- 자동 인식: 언어·모델 고르기, 실행, 인식 원본 보기(검토 필요 구간 표시), 인식 결과로 가사 만들기
- 가사 편집: 전체 가사 편집기, 붙여넣기, TXT 가져오기, 줄 반복, 검토 완료, 적용(바뀐 줄만 반영)
- 시간 정렬: 전체 / 바뀐 부분만, 정렬 후 음표 자동 연결
- 선택 음절 속성과 미정렬·연결 없음·확신 낮음 목록
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from amp.app import theme
from amp.app.widgets import shrinkable_combos
from amp.lyrics import linking, review, vocal
from amp.lyrics import pronunciation as pron
from amp.lyrics.asr import LANGUAGES, MODELS
from amp.lyrics.model import SOURCE_LABELS, LyricsState

LANG_ITEMS = list(LANGUAGES.items())
MODEL_ITEMS = [("auto", "자동 (GPU 여유 5.5GB 이상: large-v3, 아니면 turbo)")] + [(k, f"{k} — {v[0]}") for k, v in MODELS.items()]
FILTERS = (("unaligned", "정렬 안 된 음절"), ("unlinked", "음표 연결 없는 음절"), ("lowconf", "정렬 신뢰도 낮음"),
           ("review", "검토 필요한 줄"), ("stale", "다시 정렬 필요"), ("pron_check", "발음 불확실·변환 불가"),
           ("pron_stale", "발음 재검토 필요"), ("melisma", "멜리스마"), ("mel_check", "멜리스마 판정 불확실·재확인"))
PRON_HINT = "발음은 가사 텍스트에서 표준 발음 규칙으로 추정한 값입니다. 가수의 실제 발음을 인식한 것이 아닙니다."
LENGTH_HINT = ("음절 구간 길이: 정렬된 시작~끝 (끝은 대부분 다음 음절·보컬 끊김으로 정한 추정값).\n"
               "유성 가창 길이: 그 구간에서 피치(유성음)가 잡힌 시간. 무성 자음(ㅅ·ㅎ 등)은 빠지고, 보컬 분리에 남은 잔향은 들어갈 수 있습니다.")
LOW_CONF = 0.1  # 실제 두 곡: 0.1 미만 음절의 오차율 8~9% vs 나머지 1~2% (약한 지표)


def _mono() -> QLabel:
    lab = QLabel("—")
    lab.setFont(theme.mono_font())
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


class LyricsPanel(QScrollArea):
    recognizeRequested = Signal()
    applyRecognitionRequested = Signal()
    applyTextRequested = Signal(str)  # 편집기 내용
    importTxtRequested = Signal()
    repeatLinesRequested = Signal(list, int)  # 줄 번호 목록, 횟수
    reviewDoneRequested = Signal(list)  # 줄 번호 목록
    alignRequested = Signal(str)  # all / changed
    autoLinkRequested = Signal()
    syllableActionRequested = Signal(str, list)  # 가사 레인 메뉴와 같은 종류
    syllableActivated = Signal(str)  # 목록에서 고른 음절
    pronunciationEdited = Signal(str, str, object)  # 음절 id, 발음, IPA(None이면 발음에서 만듦)
    vocalActionRequested = Signal(str, list)  # 발음·멜리스마 동작, 음절 id 목록
    openModelFolderRequested = Signal()
    settingsChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        self.setWidget(body)
        self._state = LyricsState()
        self._groups: dict = {}  # 음절 id → vocal.VocalGroup
        self._durations: dict = {}  # 음절 id → vocal.SyllableDuration
        self._selected: list[str] = []
        self._dirty_editor = False
        self._loading = False

        # --- 자동 인식 ---
        g = QGroupBox("자동 가사 인식 (Whisper)")
        self.recognition_group = g
        f = QFormLayout(g)
        self.language = QComboBox()
        for k, v in LANG_ITEMS:
            self.language.addItem(v, k)
        self.model = QComboBox()
        for k, v in MODEL_ITEMS:
            self.model.addItem(v, k)
        f.addRow("언어", self.language)
        f.addRow("모델", self.model)
        row = QHBoxLayout()
        self.recognize = QPushButton("가사 인식")
        self.recognize.setProperty("accent", True)
        self.apply_rec = QPushButton("인식 결과로 가사 만들기")
        row.addWidget(self.recognize)
        row.addWidget(self.apply_rec)
        f.addRow(row)
        self.align_after = QCheckBox("인식한 뒤 바로 시간 정렬 (권장)")
        self.align_after.setChecked(True)
        self.align_after.setToolTip("인식 모델의 단어 시각만 쓰면 음절 시작이 악보보다 중앙값 0.24초 어긋났습니다"
                                    "(한국어 실제 곡 2개). 정렬 모델로 음절마다 다시 맞추고, 아래 설정에 따라 음표와 잇습니다.")
        f.addRow(self.align_after)
        hint = QLabel("노래 인식은 틀리기 쉽습니다(한국어 실제 곡 2개에서 글자 10~18% 틀림). 정확한 가사가 있으면 "
                      "아래 편집기에 붙여넣고 [전체 정렬]을 쓰세요.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        f.addRow(hint)
        self.install_label = QLabel("")
        self.install_label.setWordWrap(True)
        self.install_label.setProperty("role", "secondary")
        f.addRow(self.install_label)
        self.rec_view = QTextEdit()
        self.rec_view.setReadOnly(True)
        self.rec_view.setPlaceholderText("인식 원본이 여기에 표시됩니다. 원본은 고치지 않고, 아래 편집기에서 고칩니다.")
        self.rec_view.setMaximumHeight(130)
        f.addRow(self.rec_view)
        lay.addWidget(g)

        # --- 편집 ---
        g = QGroupBox("가사 편집")
        v = QVBoxLayout(g)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("가사를 직접 입력하거나 붙여넣으세요. 줄바꿈은 구간을 나누는 참고로만 씁니다.")
        self.editor.setFont(theme.ui_font(13))
        self.editor.setMinimumHeight(220)
        self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.editor)
        row = QGridLayout()  # 2×2: 좁은 패널에서 버튼 네 개가 한 줄을 넘지 않게
        self.apply_text = QPushButton("적용")
        self.apply_text.setProperty("accent", True)
        self.apply_text.setToolTip("바뀐 줄만 다시 나눕니다. 그대로인 음절의 시간·음표 연결은 남습니다.")
        self.revert_text = QPushButton("되돌리기")
        self.paste = QPushButton("붙여넣기")
        self.paste.setToolTip("클립보드 가사를 선택 영역(없으면 커서 위치)에 넣습니다. '적용'을 눌러야 반영됩니다.")
        self.import_txt = QPushButton("TXT 가져오기…")
        for i, b in enumerate((self.apply_text, self.revert_text, self.paste, self.import_txt)):
            row.addWidget(b, i // 2, i % 2)
        v.addLayout(row)
        row = QHBoxLayout()
        self.repeat = QPushButton("선택 줄 반복 추가")
        self.repeat.setToolTip("편집기에서 고른 줄(후렴 등)을 그 뒤에 다시 넣습니다. 새 줄은 따로 정렬됩니다.")
        self.repeat_times = QSpinBox()
        self.repeat_times.setRange(1, 8)
        self.repeat_times.setSuffix("번")
        self.review_done = QPushButton("선택 줄 검토 완료")
        row.addWidget(self.repeat)
        row.addWidget(self.repeat_times)
        row.addWidget(self.review_done)
        v.addLayout(row)
        self.text_status = QLabel("")
        self.text_status.setProperty("role", "secondary")
        self.text_status.setWordWrap(True)
        v.addWidget(self.text_status)
        lay.addWidget(g)

        # 가벼운 모드 안내 (자동 인식·정렬 대신 직접 연결)
        self.lite_hint = QLabel("가벼운 모드: 자동 인식·정렬은 쓰지 않습니다. 가사를 입력해 적용한 뒤, 가사 레인에서 음절을 고르고 "
                                "피아노 롤에서 음표를 고른 다음 오른쪽 클릭 [음표와 순서대로 연결]을 누르면 음절 시간이 음표에 맞춰집니다.")
        self.lite_hint.setWordWrap(True)
        self.lite_hint.setProperty("role", "secondary")
        self.lite_hint.hide()
        lay.addWidget(self.lite_hint)

        # --- 정렬 ---
        g = QGroupBox("시간 정렬 (MMS 강제 정렬)")
        self.alignment_group = g
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.align_all = QPushButton("전체 정렬")
        self.align_all.setProperty("accent", True)
        self.align_all.setToolTip("사용자가 직접 고친 음절 시간은 고정하고 나머지를 모두 다시 정렬합니다.")
        self.align_changed = QPushButton("바뀐 부분만")
        self.align_changed.setToolTip("미정렬·텍스트가 바뀐 음절·추정 시간만 정렬합니다. 나머지는 그대로 둡니다.")
        self.auto_link = QPushButton("음표 자동 연결")
        for b in (self.align_all, self.align_changed, self.auto_link):
            row.addWidget(b)
        v.addLayout(row)
        self.link_after = QCheckBox("정렬한 뒤 음표와 자동 연결 (확정한 연결은 유지)")
        self.link_after.setChecked(True)
        v.addWidget(self.link_after)
        # v0.6: 레인에서 음절 시간을 끌어 고친 뒤의 연결 처리와 음표 경계 붙이기
        row = QHBoxLayout()
        row.addWidget(QLabel("시간 수정 뒤 연결"))
        self.relink_mode = QComboBox()
        from amp.app.widgets.lyric_lane import RELINK_MODES

        for k, label in RELINK_MODES.items():
            self.relink_mode.addItem(label, k)
        self.relink_mode.setToolTip("가사 레인에서 음절 시간을 고친 뒤 음표 연결을 어떻게 할지. 확정한 연결은 바꾸지 않습니다")
        row.addWidget(self.relink_mode, 1)
        v.addLayout(row)
        self.snap_notes = QCheckBox("끌 때 음표 경계에 붙이기 (Alt: 이번만 끄기)")
        self.snap_notes.setChecked(True)
        v.addWidget(self.snap_notes)
        self.align_info = QLabel("")
        self.align_info.setWordWrap(True)
        self.align_info.setProperty("role", "secondary")
        v.addWidget(self.align_info)
        row = QHBoxLayout()
        self.models_btn = QPushButton("모델 폴더 열기")
        row.addWidget(self.models_btn)
        row.addStretch(1)
        v.addLayout(row)
        lay.addWidget(g)

        # --- 선택 음절 ---
        g = QGroupBox("선택 음절")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.fields: dict[str, QLabel] = {}
        for key, label in (("text", "글자"), ("where", "위치"), ("time", "시작~끝"), ("len", "길이"), ("conf", "신뢰도"),
                           ("src", "시각 출처"), ("state", "상태"), ("links", "연결 음표"), ("rel", "관계"),
                           ("note", "안내")):
            lab = _mono() if key in ("time", "len", "conf") else QLabel("—")
            lab.setWordWrap(True)
            self.fields[key] = lab
            f.addRow(label, lab)
        row = QHBoxLayout()
        self.btn_confirm = QPushButton("연결 확정")
        self.btn_link = QPushButton("선택 음표로 연결")
        self.btn_unlink = QPushButton("연결 지우기")
        for b in (self.btn_confirm, self.btn_link, self.btn_unlink):
            row.addWidget(b)
        f.addRow(row)
        lay.addWidget(g)

        # --- 발음·길이·멜리스마 (STEP 4) ---
        g = QGroupBox("발음 · 가창 길이 · 멜리스마")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for key, label in (("pron", "발음"), ("ipa", "IPA"), ("pron_info", "발음 근거"), ("span", "음절 구간 길이"),
                           ("voiced", "유성 가창 길이"), ("union", "연결 음표 구간"), ("mel", "멜리스마"),
                           ("mel_pitch", "음높이 변화"), ("mel_info", "판정 근거")):
            lab = _mono() if key in ("span", "voiced", "union") else QLabel("—")
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.fields[key] = lab
            f.addRow(label, lab)
        hint = QLabel(PRON_HINT)
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        f.addRow(hint)
        self.pron_edit = QLineEdit()
        self.pron_edit.setPlaceholderText("발음 (예: 가치). 비우면 '앞 음절과 합쳐 발음'")
        self.ipa_edit = QLineEdit()
        self.ipa_edit.setPlaceholderText("IPA (비우면 발음에서 만듦)")
        f.addRow("발음 고치기", self.pron_edit)
        f.addRow("IPA 고치기", self.ipa_edit)
        row = QHBoxLayout()
        self.btn_pron_apply = QPushButton("발음 적용")
        self.btn_pron_apply.setProperty("accent", True)
        self.btn_pron_confirm = QPushButton("지금 발음 확정")
        self.btn_pron_reset = QPushButton("기본 발음으로")
        for b in (self.btn_pron_apply, self.btn_pron_confirm, self.btn_pron_reset):
            row.addWidget(b)
        f.addRow(row)
        row = QHBoxLayout()
        self.btn_pron_regen = QPushButton("선택 음절 다시 생성")
        self.btn_pron_regen_all = QPushButton("전체 다시 생성")
        self.btn_pron_regen_all.setToolTip("자동 발음만 지금 규칙으로 다시 만듭니다. 직접 고치거나 확정한 발음은 그대로 둡니다.")
        row.addWidget(self.btn_pron_regen)
        row.addWidget(self.btn_pron_regen_all)
        f.addRow(row)
        row = QHBoxLayout()  # v0.7.0 프로젝트 발음 사전
        self.btn_pron_dict = QPushButton("발음 사전에 추가")
        self.btn_pron_dict.setToolTip("고른 음절이 든 단어의 지금 발음을 이 프로젝트의 발음 사전에 넣습니다. 같은 단어의 자동 발음에 "
                                      "쓰입니다 (직접 고친 발음은 그대로). 실행 취소할 수 있습니다.")
        self.btn_pron_dict_edit = QPushButton("발음 사전…")
        self.btn_pron_dict_edit.setToolTip("이 프로젝트의 발음 사전을 보고 고칩니다. 다른 프로젝트와는 파일로 가져오기·내보내기를 합니다.")
        row.addWidget(self.btn_pron_dict)
        row.addWidget(self.btn_pron_dict_edit)
        row.addStretch(1)
        f.addRow(row)
        len_hint = QLabel(LENGTH_HINT)
        len_hint.setWordWrap(True)
        len_hint.setProperty("role", "secondary")
        f.addRow(len_hint)
        row = QHBoxLayout()
        self.btn_mel_set = QPushButton("멜리스마로 확정")
        self.btn_mel_set.setToolTip("피아노 롤에서 음표를 2개 이상 골랐으면 그 음표로, 아니면 연결된 음표로 확정합니다.")
        self.btn_mel_not = QPushButton("멜리스마 아님")
        self.btn_mel_clear = QPushButton("자동 판정으로")
        for b in (self.btn_mel_set, self.btn_mel_not, self.btn_mel_clear):
            row.addWidget(b)
        f.addRow(row)
        row = QHBoxLayout()
        self.btn_mel_add = QPushButton("선택 음표 추가")
        self.btn_mel_remove = QPushButton("선택 음표 빼기")
        self.btn_mel_ok = QPushButton("판정 확인")
        self.btn_mel_ok.setToolTip("음표가 바뀌어 '재확인 필요'가 붙은 사용자 판정을 확인했다고 표시합니다.")
        for b in (self.btn_mel_add, self.btn_mel_remove, self.btn_mel_ok):
            row.addWidget(b)
        f.addRow(row)
        lay.addWidget(g)

        # --- 목록 ---
        g = QGroupBox("확인할 음절")
        v = QVBoxLayout(g)
        self.filter = QComboBox()
        for k, label in FILTERS:
            self.filter.addItem(label, k)
        v.addWidget(self.filter)
        self.listing = QListWidget()
        self.listing.setMinimumHeight(120)
        v.addWidget(self.listing)
        lay.addWidget(g)
        lay.addStretch(1)
        # 좁은 창: 긴 콤보 항목('자동 (GPU 여유 5.5GB 이상: …)')이 패널 최소 폭을 456px로 밀어 가로 스크롤이 생겼다
        shrinkable_combos(body)
        for form in body.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.recognize.clicked.connect(self.recognizeRequested)
        self.apply_rec.clicked.connect(self.applyRecognitionRequested)
        self.apply_text.clicked.connect(lambda: self.applyTextRequested.emit(self.editor.toPlainText()))
        self.revert_text.clicked.connect(lambda: self.set_state(self._state, force_editor=True))
        self.paste.clicked.connect(self._paste)
        self.import_txt.clicked.connect(self.importTxtRequested)
        self.repeat.clicked.connect(lambda: self.repeatLinesRequested.emit(self._selected_lines(), self.repeat_times.value()))
        self.review_done.clicked.connect(lambda: self.reviewDoneRequested.emit(self._selected_lines()))
        self.align_all.clicked.connect(lambda: self.alignRequested.emit("all"))
        self.align_changed.clicked.connect(lambda: self.alignRequested.emit("changed"))
        self.auto_link.clicked.connect(self.autoLinkRequested)
        self.models_btn.clicked.connect(self.openModelFolderRequested)
        self.btn_confirm.clicked.connect(lambda: self.syllableActionRequested.emit("confirm", list(self._selected)))
        self.btn_link.clicked.connect(lambda: self.syllableActionRequested.emit("link_replace", list(self._selected)))
        self.btn_unlink.clicked.connect(lambda: self.syllableActionRequested.emit("unlink", list(self._selected)))
        self.btn_pron_apply.clicked.connect(self._apply_pron)
        self.pron_edit.returnPressed.connect(self._apply_pron)
        for btn, kind in ((self.btn_pron_confirm, "pron_confirm"), (self.btn_pron_reset, "pron_reset"),
                          (self.btn_pron_regen, "pron_regen"), (self.btn_pron_regen_all, "pron_regen_all"),
                          (self.btn_pron_dict, "pron_to_dict"), (self.btn_pron_dict_edit, "pron_dict_edit"),
                          (self.btn_mel_set, "mel_set"), (self.btn_mel_not, "mel_not"), (self.btn_mel_clear, "mel_clear"),
                          (self.btn_mel_add, "mel_add"), (self.btn_mel_remove, "mel_remove"),
                          (self.btn_mel_ok, "mel_reviewed")):
            btn.clicked.connect(lambda _c=False, k=kind: self.vocalActionRequested.emit(k, list(self._selected)))
        self.filter.currentIndexChanged.connect(lambda _i: self._fill_list())
        self.listing.itemActivated.connect(self._on_item)
        self.listing.itemClicked.connect(self._on_item)
        self.editor.textChanged.connect(self._on_editor_changed)
        self.language.currentIndexChanged.connect(lambda _i: self.settingsChanged.emit())
        self.model.currentIndexChanged.connect(lambda _i: self.settingsChanged.emit())

    # --- 설정 ---------------------------------------------------------------

    def settings(self) -> dict:
        return {"language": self.language.currentData(), "model": self.model.currentData(),
                "link_after": self.link_after.isChecked(), "align_after": self.align_after.isChecked()}

    def set_settings(self, d: dict) -> None:
        for combo, key in ((self.language, "language"), (self.model, "model")):
            i = combo.findData(d.get(key, "auto"))
            if i >= 0:
                combo.setCurrentIndex(i)
        self.link_after.setChecked(bool(d.get("link_after", True)))
        self.align_after.setChecked(bool(d.get("align_after", True)))

    def set_lite(self, on: bool) -> None:
        """가벼운 모드: 자동 인식(Whisper)·자동 정렬(MMS) 상자를 숨기고 직접 연결 방법을 안내한다."""
        self.recognition_group.setVisible(not on)
        self.alignment_group.setVisible(not on)
        self.lite_hint.setVisible(on)

    def set_install_status(self, text: str) -> None:
        self.install_label.setText(text)

    def set_align_info(self, text: str) -> None:
        self.align_info.setText(text)

    # --- 상태 반영 -------------------------------------------------------------

    def _on_editor_changed(self) -> None:
        if self._loading:
            return
        self._dirty_editor = self.editor.toPlainText() != self._state.text.text
        self._update_text_status()

    def _update_text_status(self) -> None:
        ly = self._state
        n = len(ly.text.syllables)
        msg = f"{len(ly.text.lines)}줄 · 음절 {n}개 · 정렬 {n - len(ly.unaligned())} · 연결 {n - len(ly.unlinked())}"
        if self._dirty_editor:
            msg = "편집 중 — '적용'을 눌러야 반영됩니다. " + msg
        review = sum(1 for ln in ly.text.lines if ln.review)
        if review:
            msg += f" · 검토 필요 {review}줄 (주황 배경)"
        self.text_status.setText(msg)

    def set_state(self, state: LyricsState, force_editor: bool = False) -> None:
        self._state = state
        if force_editor or not self._dirty_editor:
            self._loading = True
            cur = self.editor.textCursor().position()
            self.editor.setPlainText(state.text.text)
            c = self.editor.textCursor()
            c.setPosition(min(cur, len(state.text.text)))
            self.editor.setTextCursor(c)
            self._loading = False
            self._dirty_editor = False
            self._mark_review_lines()
        rec = state.recognition
        if rec is None:
            self.rec_view.setPlainText("")
            self.apply_rec.setEnabled(False)
        else:
            parts = [f"[{rec.model} · {rec.language} · {rec.device}] 구간 {len(rec.segments)}개"]
            for s in rec.segments:
                if review.excluded(s):
                    mark, why = "✕ ", f"   ← {', '.join(s.review)} · 가사에 넣지 않음"
                else:
                    mark = "⚠ " if s.review else ""
                    why = f"   ← {', '.join(s.review)}" if s.review else ""
                parts.append(f"{mark}{s.start:6.1f}s  {s.text}{why}")
            parts.extend(f"참고: {w}" for w in rec.warnings)
            self.rec_view.setPlainText("\n".join(parts))
            self.apply_rec.setEnabled(True)
        model = state.alignment.model
        if model:
            info = f"마지막 정렬: {model.get('name', '')} · {model.get('device', '')} · {model.get('created', '')}"
            if model.get("license"):
                info += f"\n가중치 라이선스: {model['license']}"
            self.align_info.setText(info)
        self._update_text_status()
        self.show_syllables(self._selected)
        self._fill_list()

    def _mark_review_lines(self) -> None:
        sels = []
        doc = self.editor.document()
        for i, ln in enumerate(self._state.text.lines):
            if not ln.review:
                continue
            block = doc.findBlockByNumber(i)
            if not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            fmt = QTextCharFormat()
            bg = QColor(theme.LYRIC_WARN)
            bg.setAlpha(55)
            fmt.setBackground(bg)
            fmt.setProperty(QTextFormat.Property.FullWidthSelection, True)
            fmt.setToolTip(ln.review)
            sel.format = fmt
            sel.cursor = QTextCursor(block)
            sels.append(sel)
        self.editor.setExtraSelections(sels)

    def _selected_lines(self) -> list[int]:
        c = self.editor.textCursor()
        doc = self.editor.document()
        a = doc.findBlock(c.selectionStart()).blockNumber()
        b = doc.findBlock(c.selectionEnd()).blockNumber()
        return list(range(min(a, b), max(a, b) + 1))

    def _paste(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = QApplication.clipboard().text()
        if text:
            self.editor.insertPlainText(text)

    def show_syllables(self, ids: list[str]) -> None:
        self._selected = [i for i in ids if i in self._state.text.index]
        f = self.fields
        for b in (self.btn_confirm, self.btn_link, self.btn_unlink):
            b.setEnabled(bool(self._selected))
        if not self._selected:
            for lab in f.values():
                lab.setText("—")
            f["note"].setText("가사 레인이나 목록에서 음절을 고르세요")
            self._show_vocal(None)
            return
        ly = self._state
        if len(self._selected) > 1:
            for lab in f.values():
                lab.setText("—")
            f["text"].setText(" ".join(ly.text.syllable(s).text for s in self._selected[:20]))
            f["note"].setText(f"음절 {len(self._selected)}개 선택")
            self._show_vocal(None)
            return
        sid = self._selected[0]
        s = ly.text.syllable(sid)
        li, wi, si = ly.text.index[sid]
        t = ly.alignment.timing(sid)
        f["text"].setText(s.text + (f"  ({s.reading})" if s.reading and s.reading != s.text.lower() else ""))
        f["where"].setText(f"{li + 1}줄 {wi + 1}번째 단어 '{ly.text.lines[li].words[wi].text}'")
        if t.aligned:
            f["time"].setText(f"{t.start:.3f} ~ {t.end:.3f} s")
            f["len"].setText(f"{(t.end - t.start) * 1000:.0f} ms")
        else:
            f["time"].setText("미정렬")
            f["len"].setText("—")
        f["conf"].setText("—" if t.confidence is None else f"{t.confidence:.2f}")
        f["src"].setText(f"시작 {SOURCE_LABELS.get(t.start_source, t.start_source)} / 끝 {SOURCE_LABELS.get(t.end_source, t.end_source)}")
        state = []
        if t.user_edited:
            state.append("사용자 수정")
        if t.stale:
            state.append("다시 정렬 필요")
        links = ly.alignment.links.get(sid, ())
        if links and all(lk.confirmed for lk in links):
            state.append("연결 확정")
        f["state"].setText(", ".join(state) or "자동")
        f["links"].setText(", ".join(f"{lk.note_id[:6]}({lk.source}{', 확정' if lk.confirmed else ''}"
                                     f"{', 접근음' if lk.role == 'approach' else ''}"
                                     f"{'' if lk.confidence is None else f', {lk.confidence:.2f}'})" for lk in links) or "없음")
        kind = linking.relation_kind(ly, sid)
        rel = {"none": "연결 없음", "1:1": "1:1", "1:N": "1:N (음표 여러 개 — 멜리스마 여부는 아래 판정)", "N:1": "N:1 (여러 음절이 한 음표)",
               "N:M": "여러 음표·여러 음절"}[kind]
        n_app = sum(1 for lk in links if lk.role == "approach")
        f["rel"].setText(rel + (f" + 접근음 {n_app}" if n_app else ""))
        line = ly.text.lines[li]
        f["note"].setText("; ".join(x for x in (s.note, line.review) if x) or "—")
        self._show_vocal(sid)

    # --- 발음·길이·멜리스마 (STEP 4) ------------------------------------------------

    def set_vocal(self, groups: dict, durations: dict) -> None:
        self._groups, self._durations = groups, durations
        self.show_syllables(self._selected)
        if self.filter.currentData() in ("pron_check", "pron_stale", "melisma", "mel_check"):
            self._fill_list()

    def _apply_pron(self) -> None:
        if len(self._selected) != 1:
            return
        ipa = self.ipa_edit.text().strip()
        self.pronunciationEdited.emit(self._selected[0], self.pron_edit.text(), ipa or None)

    def _vocal_buttons(self, one: bool, any_sel: bool) -> None:
        for b in (self.btn_pron_apply, self.btn_mel_set, self.btn_mel_not, self.btn_mel_add, self.btn_mel_remove):
            b.setEnabled(one)
        for b in (self.btn_pron_confirm, self.btn_pron_reset, self.btn_pron_regen, self.btn_pron_dict, self.btn_mel_clear,
                  self.btn_mel_ok):
            b.setEnabled(any_sel)
        self.pron_edit.setEnabled(one)
        self.ipa_edit.setEnabled(one)

    def _show_vocal(self, sid: str | None) -> None:
        f = self.fields
        keys = ("pron", "ipa", "pron_info", "span", "voiced", "union", "mel", "mel_pitch", "mel_info")
        self._vocal_buttons(sid is not None, bool(self._selected))
        if sid is None:
            for k in keys:
                f[k].setText("—")
            return
        e = self._state.pronunciation.get(sid)
        if e is None:
            f["pron"].setText("없음")
            f["ipa"].setText("—")
            f["pron_info"].setText("—")
            self.pron_edit.setText("")
            self.ipa_edit.setText("")
        else:
            badge = [pron.SOURCE_LABELS.get(e.source, e.source) if e.user_edited else pron.STATUS_LABELS.get(e.status, e.status)]
            if e.stale:
                badge.append("재검토 필요")
            shown = e.text if e.text else ("(앞 음절과 합쳐 발음)" if e.user_edited else "—")
            units = f" · 발음 {e.units}음절" if e.units != 1 and e.user_edited else ""
            f["pron"].setText(f"[{shown}]  {' · '.join(badge)}{units}")
            f["ipa"].setText(f"/{e.ipa}/" if e.ipa else "— (만들 수 없음)")
            info = list(e.rules)
            if e.user_edited and e.auto_text != e.text:
                info.append(f"자동 발음은 [{e.auto_text or '—'}]")
            info.extend(e.notes)
            f["pron_info"].setText("; ".join(info) or "규칙 변화 없음 (글자대로)")
            self.pron_edit.setText(e.text)
            self.ipa_edit.setText(e.ipa if e.ipa_edited else "")
        d = self._durations.get(sid)
        if d is None:
            for k in ("span", "voiced", "union"):
                f[k].setText("—")
        else:
            f["span"].setText("—  (" + vocal.SPAN_STATUS_LABELS[d.span_status] + ")" if d.span is None else
                              f"{d.span * 1000:.0f} ms  ({vocal.SPAN_STATUS_LABELS[d.span_status]})")
            f["voiced"].setText(f"미측정 — {d.voiced_reason}" if d.voiced is None else
                                f"{d.voiced * 1000:.0f} ms  ({d.voiced_reason})")
            f["union"].setText("—" if d.note_union is None else f"{d.note_union * 1000:.0f} ms  (겹친 부분은 한 번만)")
        g = self._groups.get(sid)
        if g is None:
            f["mel"].setText("연결된 음표 없음")
            f["mel_pitch"].setText("—")
            f["mel_info"].setText("—")
            return
        if g.user_kind:
            head = ("멜리스마" if g.is_melisma else "멜리스마 아님") + " (사용자 확정)"
            if g.kind != ("melisma" if g.is_melisma else g.kind):
                head += f" · 자동 판정: {vocal.KIND_LABELS[g.kind]}"
        else:
            head = f"{vocal.KIND_LABELS[g.kind]} (자동 판정, 규칙 점수 {g.confidence:.2f})"
        if g.needs_review:
            head += " · 재확인 필요"
        f["mel"].setText(head)
        if g.pitch_sequence:
            dur = "" if g.duration is None else f", {g.start:.3f}~{g.end:.3f} s · {g.duration * 1000:.0f} ms"
            f["mel_pitch"].setText(f"{' → '.join(g.pitch_sequence)}  (음표 {g.note_count}개, 범위 {g.pitch_range}반음, "
                                   f"{vocal.DIRECTION_LABELS[g.direction]}{dur})")
        else:
            f["mel_pitch"].setText("—")
        f["mel_info"].setText("; ".join(g.reasons) or "—")

    def _fill_list(self) -> None:
        kind = self.filter.currentData()
        ly = self._state
        self.listing.clear()
        rows = []
        if kind == "review":
            for li, ln in enumerate(ly.text.lines):
                if ln.review and ln.syllables:
                    rows.append((ln.syllables[0].id, f"{li + 1}줄: {ln.text}  — {ln.review}"))
        else:
            for s in ly.text.syllables:
                t = ly.alignment.timing(s.id)
                if kind == "unaligned" and t.aligned:
                    continue
                if kind == "unlinked" and ly.alignment.links.get(s.id):
                    continue
                if kind == "lowconf" and not (t.confidence is not None and t.start_source == "model" and t.confidence < LOW_CONF):
                    continue
                if kind == "stale" and not t.stale:
                    continue
                e = ly.pronunciation.get(s.id)
                if kind == "pron_check" and not (e is not None and not e.user_edited and e.status != "valid"):
                    continue
                if kind == "pron_stale" and not (e is not None and e.stale):
                    continue
                g = self._groups.get(s.id)
                if kind == "melisma" and not (g is not None and g.is_melisma):
                    continue
                if kind == "mel_check" and not (g is not None and (g.needs_review or (g.kind == "uncertain" and not g.user_kind))):
                    continue
                li = ly.text.index[s.id][0]
                when = f"{t.start:7.2f}s" if t.aligned else "   —   "
                rows.append((s.id, f"{when}  '{s.text}'  {li + 1}줄 {ly.text.lines[li].text[:24]}"))
        for sid, label in rows[:2000]:
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, sid)
            self.listing.addItem(it)
        if len(rows) > 2000:
            self.listing.addItem(f"… {len(rows) - 2000}개 더")

    def _on_item(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid:
            self.syllableActivated.emit(sid)
