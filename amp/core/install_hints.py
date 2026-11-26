"""설치 안내 문구 (운영체제별). Windows는 .bat, 크롬북·리눅스는 셸 스크립트를 알려 준다. 표준 라이브러리만 쓴다."""

from __future__ import annotations

import os

_WINDOWS = {"run": "run.bat", "separation": "install_separation.bat", "lyrics": "install_lyrics.bat",
            "pitch": "install_pitch.bat", "pron": "install_lyrics.bat --dictionaries-only",
            "pron_japanese": "install_lyrics.bat --dictionaries-only --japanese"}
_POSIX = {"run": "./run.sh", "separation": "./install_chromebook.sh --separation",
          "lyrics": "./install_chromebook.sh --lyrics", "pitch": "./install_chromebook.sh --pitch",
          "pron": "./install_chromebook.sh --pron", "pron_japanese": "./install_chromebook.sh --pron --japanese"}


def script(kind: str, os_name: str | None = None) -> str:
    """kind: run / separation / lyrics / pitch / pron / pron_japanese. 리눅스에서는 리눅스 명령을 먼저, Windows 파일 이름을 괄호로 함께 적는다."""
    if (os_name or os.name) == "nt":
        return _WINDOWS[kind]
    return f"{_POSIX[kind]} (Windows는 {_WINDOWS[kind]})"
