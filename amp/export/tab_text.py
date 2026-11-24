"""베이스 TAB → 텍스트 파일 (v0.6.3). 메모장·카페 글에 붙여 넣는 용도. UTF-8(BOM)로 써서 Windows 메모장에서도 한글이 깨지지 않는다."""

from __future__ import annotations

import os

from amp.core.fileio import atomic_write_bytes
from amp.engine.bass import Tab, text_tab


def write_text_tab(tab: Tab, path: str | os.PathLike, title: str = "", bars_per_line: int = 4) -> None:
    text = text_tab(tab, bars_per_line=bars_per_line, title=title)
    atomic_write_bytes(path, text.replace("\n", "\r\n").encode("utf-8-sig"))
