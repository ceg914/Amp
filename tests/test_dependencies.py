"""설치 목록 점검: 앱 코드가 import하는 외부 패키지는 requirements.txt·시작 확인 목록·run.bat에 모두 있어야 한다.

STEP 5에서 mido(MIDI)·PyYAML(USTX)을 쓰기 시작했지만 설치 목록에 넣지 않아, 개발 환경에서는 통과하고 사용자 PC에서
'No module named mido'로 내보내기가 실패했다. 이런 빠짐을 시험으로 잡는다.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 별도 설치(보컬 분리·가사 인식)이거나, 없으면 다른 방법으로 넘어가거나 안내를 띄우는 선택 패키지
OPTIONAL = {"torch", "demucs", "whisper", "tqdm", "safetensors", "miniaudio"}
BUNDLED = {"shiboken6": "PySide6"}  # PySide6와 함께 설치된다
DIST_NAME = {"yaml": "pyyaml", "guitarpro": "pyguitarpro"}  # import 이름 → pip 패키지 이름


def _third_party_imports() -> set[str]:
    found: set[str] = set()
    for p in (ROOT / "amp").rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                names = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module.split(".")[0]]
            else:
                continue
            found.update(m for m in names if m not in sys.stdlib_module_names and m not in ("amp", "__future__"))
    return found


def _required() -> set[str]:
    return set(_third_party_imports()) - OPTIONAL - set(BUNDLED)


def test_every_required_import_is_in_requirements_txt():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    listed = {re.split(r"[<>=!~\[ ;]", ln.strip(), maxsplit=1)[0].lower() for ln in lines if ln.strip() and not ln.startswith("#")}
    missing = sorted(m for m in _required() if DIST_NAME.get(m, m).lower() not in listed)
    assert not missing, f"requirements.txt에 없음: {missing}"


def test_startup_check_and_run_bat_cover_the_same_packages():
    from amp.__main__ import REQUIRED_MODULES

    assert set(REQUIRED_MODULES) == _required()
    bat = (ROOT / "run.bat").read_text(encoding="utf-8", errors="replace")
    line = next(ln for ln in bat.splitlines() if '-c "import ' in ln)
    imported = {m.strip() for m in line.split('-c "import ', 1)[1].split('"', 1)[0].split(",")}
    assert imported == _required(), f"run.bat 확인 목록: {sorted(imported)}"
