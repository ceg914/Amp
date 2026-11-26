"""모델 관리 창 (v0.7.0 명세 13.2·16.3). 모델 등록 정보(설치 여부·버전·크기·파일 위치·체크포인트 해시·장치·의존성·
라이선스·상태)와 프로필별로 켤 수 있는 기능을 보여 준다.

- 읽기 전용이다. 모델을 내려받거나 지우거나 바꾸지 않는다 (사용자 MelBandRoformer.ckpt 포함)
- 체크포인트 해시는 큰 파일을 읽으므로 '해시 계산'을 누를 때만 작업 스레드에서 계산한다
- GUI 프로세스에서 torch를 불러오지 않는다 (등록 정보는 find_spec·파일 크기만 본다)
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from amp.engine import model_registry as R
from amp.engine import profiles as P

TYPE_LABELS = {"separation": "분리", "pitch": "피치", "asr": "가사 인식", "alignment": "정렬", "pronunciation": "발음 자료"}
COLUMNS = ("모델", "종류", "상태", "버전", "크기", "장치")


def _size(n: int | None) -> str:
    if not n:
        return "—"
    return f"{n / 2**20:.1f} MB" if n < 2**30 else f"{n / 2**30:.2f} GB"


def detail_text(st: R.ModelStatus) -> str:
    lines = [f"{st.name} ({st.model_id})", f"상태: {st.state_label}" + (f" — {st.reason}" if st.reason else ""),
             f"종류: {TYPE_LABELS.get(st.model_type, st.model_type)} · 기능: {', '.join(st.features) or '—'}",
             f"버전: {st.version or '—'}", f"파일: {st.path or '—'}", f"크기: {_size(st.size_bytes)}",
             f"체크포인트 SHA-256: {st.checkpoint_sha256 or '계산하지 않음'}",
             f"실행 장치: {', '.join(st.devices)}",
             "의존성: " + (", ".join(f"{k} {v or '(없음)'}" for k, v in st.dependencies.items()) or "—"),
             f"코드 라이선스: {st.license or '확인하지 않음'}", f"가중치·자료 라이선스: {st.weights_license or '확인하지 않음'}"]
    if st.experimental:
        lines.append("실험 기능: 실제 곡에서 검증하지 않았습니다.")
    return "\n".join(lines)


class ModelManagerDialog(QDialog):
    def __init__(self, window, statuses: list[R.ModelStatus]) -> None:
        super().__init__(window)
        self.w = window
        self.statuses = list(statuses)
        self.setWindowTitle("모델 관리")
        self.resize(820, 560)
        lay = QVBoxLayout(self)
        hint = QLabel("설치된 분석 모델과 자료입니다. 이 창은 모델을 내려받거나 지우거나 바꾸지 않습니다. "
                      "모델이 없거나 실패한 기능만 꺼지고 나머지 분석은 그대로 쓸 수 있습니다.")
        hint.setWordWrap(True)
        hint.setProperty("role", "secondary")
        lay.addWidget(hint)
        split = QSplitter(Qt.Orientation.Vertical)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        split.addWidget(self.tree)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        split.addWidget(self.detail)
        lay.addWidget(split, 1)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("프로필"))
        self.profile = QComboBox()
        for k in P.PROFILES:
            self.profile.addItem(P.PROFILE_LABELS[k], k)
        lite = bool(getattr(window, "lite", False))
        self.profile.setCurrentIndex(self.profile.findData(P.default_profile(lite)))
        self.profile.setEnabled(not lite)
        prow.addWidget(self.profile)
        prow.addStretch(1)
        lay.addLayout(prow)
        self.features = QPlainTextEdit()
        self.features.setReadOnly(True)
        self.features.setMaximumHeight(150)
        lay.addWidget(self.features)
        brow = QHBoxLayout()
        self.btn_hash = QPushButton("체크포인트 해시 계산")
        self.btn_hash.setToolTip("모델 파일을 끝까지 읽어 SHA-256을 계산합니다 (큰 파일은 시간이 걸립니다). 파일은 바꾸지 않습니다.")
        self.btn_refresh = QPushButton("다시 살펴보기")
        brow.addWidget(self.btn_hash)
        brow.addWidget(self.btn_refresh)
        brow.addStretch(1)
        lay.addLayout(brow)
        self.message = QLabel("")
        self.message.setProperty("role", "secondary")
        lay.addWidget(self.message)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        box.button(QDialogButtonBox.StandardButton.Close).setText("닫기")
        lay.addWidget(box)
        self.tree.currentItemChanged.connect(lambda _c, _p: self._show_detail())
        self.profile.currentIndexChanged.connect(lambda _i: self._fill_features())
        self.btn_hash.clicked.connect(lambda: self.rescan(True))
        self.btn_refresh.clicked.connect(lambda: self.rescan(False))
        self._fill()

    def _fill(self) -> None:
        keep = self.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) if self.tree.currentItem() else None
        self.tree.clear()
        for st in self.statuses:
            it = QTreeWidgetItem([st.name + (" (실험)" if st.experimental else ""), TYPE_LABELS.get(st.model_type, st.model_type),
                                  ("✓ " if st.usable else "✕ ") + st.state_label, st.version or "—", _size(st.size_bytes),
                                  ", ".join(st.devices)])
            it.setData(0, Qt.ItemDataRole.UserRole, st.model_id)
            it.setToolTip(2, st.reason)
            self.tree.addTopLevelItem(it)
            if st.model_id == keep:
                self.tree.setCurrentItem(it)
        if self.tree.currentItem() is None and self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._fill_features()

    def _show_detail(self) -> None:
        it = self.tree.currentItem()
        st = R.status_of(self.statuses, it.data(0, Qt.ItemDataRole.UserRole)) if it else None
        self.detail.setPlainText(detail_text(st) if st else "")

    def _fill_features(self) -> None:
        lite = bool(getattr(self.w, "lite", False))
        res = P.resolve(self.profile.currentData(), self.statuses, lite)
        lines = []
        for fs in res.values():
            mark = "✓" if fs.enabled else "✕"
            extra = f" — {fs.reason}" if fs.reason else (f" ({', '.join(fs.models)})" if fs.models else "")
            lines.append(f"{mark} {fs.feature.label}{' (실험)' if fs.feature.experimental else ''}{extra}")
        self.features.setPlainText("\n".join(lines))

    def rescan(self, hash_files: bool) -> None:
        """작업 스레드에서 다시 살핀다 (해시 계산은 파일을 읽기만 한다)."""
        self.btn_hash.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.message.setText("해시 계산 중…" if hash_files else "살펴보는 중…")
        session = self.w.session
        handle = self.w.runner.submit("모델 등록 정보", lambda p, t: session.model_registry(hash_files=hash_files))
        handle.finished.connect(self._on_scanned)
        handle.failed.connect(lambda msg, _tb: self.message.setText(f"살펴보지 못했습니다: {msg}"))
        handle.ended.connect(lambda: (self.btn_hash.setEnabled(True), self.btn_refresh.setEnabled(True)))

    def _on_scanned(self, statuses) -> None:
        self.statuses = list(statuses)
        self._fill()
        hashed = sum(1 for s in self.statuses if s.checkpoint_sha256)
        self.message.setText(f"모델 {len(self.statuses)}개 · 해시를 아는 파일 {hashed}개")
