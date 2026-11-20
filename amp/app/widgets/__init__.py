"""피아노 롤 주변 위젯 (파형, 눈금자, 건반)."""


def shrinkable_combos(root, min_chars: int = 8) -> None:
    """root 안 콤보박스가 항목 글자 길이와 상관없이 줄어들게 한다 (좁은 패널에서 가로 스크롤 방지).
    줄어든 글자는 목록을 펼치거나 툴팁에서 전부 보인다."""
    from PySide6.QtWidgets import QComboBox

    for c in root.findChildren(QComboBox):
        c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(min_chars)
        if not c.toolTip():
            c.setToolTip(c.currentText())
            c.currentTextChanged.connect(c.setToolTip)
