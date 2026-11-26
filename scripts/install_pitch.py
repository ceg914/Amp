"""외부 피치 모델(CREPE·FCPE) 가중치 설치 (v0.7.0). PyTorch가 먼저 있어야 한다(install_separation.bat).

torchcrepe 0.0.24(MIT, 72MB)와 torchfcpe 0.0.4(MIT, 40MB)를 의존 패키지 없이(--no-deps) 설치한다.
SONATLAS는 두 패키지를 불러오지 않고, 그 안의 가중치 파일만 읽어 torch로 계산한다(torchaudio 등이 필요 없음).
이 스크립트를 실행할 때만 내려받는다. 앱은 모델 파일을 스스로 내려받지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys

PACKAGES = ("torchcrepe==0.0.24", "torchfcpe==0.0.4")
# 설치 후 확인할 가중치 (패키지 이름, 파일, sha256). 2026-11 PyPI 휠에서 잰 값
WEIGHTS = (("torchcrepe", "full.pth", "133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986"),
           ("torchcrepe", "tiny.pth", "d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432"),
           ("torchfcpe", "fcpe_c_v001.pt", "b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015"))


def commands(python: str = sys.executable) -> list[list[str]]:
    return [[python, "-m", "pip", "install", "--no-deps", *PACKAGES]]


def has_torch(python: str = sys.executable) -> bool:
    try:
        r = subprocess.run([python, "-c", "import torch"], capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(find_spec=importlib.util.find_spec) -> tuple[bool, list[str]]:
    """가중치 파일이 있고 해시가 맞는지. 패키지를 불러오지 않는다 (이 프로세스에서 확인할 때는 새로 설치한 것을 보도록 캐시 비움)."""
    importlib.invalidate_caches()
    ok, lines = True, []
    for pkg, name, want in WEIGHTS:
        spec = find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            ok = False
            lines.append(f"{pkg}: 설치되지 않음")
            continue
        p = os.path.join(list(spec.submodule_search_locations)[0], "assets", name)
        if not os.path.isfile(p):
            ok = False
            lines.append(f"{pkg}/{name}: 파일 없음")
            continue
        got = _sha256(p)
        if got != want:
            lines.append(f"{pkg}/{name}: 해시가 예상과 다릅니다 ({got[:12]}…). 다른 판일 수 있어 결과를 비교해 보세요.")
        else:
            lines.append(f"{pkg}/{name}: 확인 ({os.path.getsize(p) / 2**20:.0f}MB)")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="실행할 명령만 보여 줌")
    args = ap.parse_args(argv)
    if not args.dry_run and not has_torch():
        print("[SONATLAS] PyTorch가 없습니다. install_separation.bat 을 먼저 실행해 주세요 (GPU판을 맞춰 설치합니다).")
        return 1
    for cmd in commands():
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not args.dry_run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print("[SONATLAS] 설치에 실패했습니다. 인터넷 연결과 위쪽 오류 메시지를 확인하세요.")
                return r.returncode
    if not args.dry_run:
        ok, lines = verify()
        for line in lines:
            print(f"[SONATLAS] {line}")
        if not ok:
            return 1
        print("[SONATLAS] 'pip check'가 torchaudio 등이 없다고 알릴 수 있습니다. SONATLAS는 그 패키지를 쓰지 않습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
