"""보컬 분리(Demucs htdemucs)에 필요한 PyTorch와 demucs를 설치한다.

- NVIDIA GPU가 있으면 nvidia-smi가 알려 주는 드라이버의 CUDA 버전에 맞는 PyTorch를 고른다.
- GPU가 없으면 CPU판을 설치한다 (Windows는 PyPI 기본 휠이 CPU판).
- 지금 실행한 Python(가상 환경이라면 그 안)에 설치한다.

사용법: python scripts/install_separation.py [--cpu] [--dry-run]
PyTorch가 제공하는 CUDA 빌드 목록은 바뀐다. 이 표는 2026-11 기준(cu132/cu130/cu126)이다.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/"
# (드라이버가 지원하는 최소 CUDA 버전, 인덱스 이름) — 위에서부터 고른다
CUDA_BUILDS = [((13, 2), "cu132"), ((13, 0), "cu130"), ((12, 6), "cu126")]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_cuda_version(nvidia_smi_output: str) -> tuple[int, int] | None:
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    return (int(m.group(1)), int(m.group(2))) if m else None


def detect_cuda() -> tuple[int, int] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_cuda_version(out)


def choose_build(cuda: tuple[int, int] | None, force_cpu: bool = False) -> str:
    if cuda is not None and not force_cpu:
        for minimum, name in CUDA_BUILDS:
            if cuda >= minimum:
                return name
    return "cpu"


def commands(build: str, python: str = sys.executable, platform: str = sys.platform) -> list[list[str]]:
    torch = [python, "-m", "pip", "install", "--upgrade", "torch"]
    if build != "cpu" or not platform.startswith("win"):
        # Windows CPU판은 PyPI 기본 휠. 그 밖에는 PyTorch 인덱스를 지정한다 (Linux PyPI 휠은 CUDA 포함이라 매우 크다)
        torch += ["--index-url", TORCH_INDEX + build]
    return [torch, [python, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements-separation.txt")]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU판을 설치")
    ap.add_argument("--dry-run", action="store_true", help="실행할 명령만 보여 줌")
    args = ap.parse_args(argv)
    cuda = detect_cuda()
    build = choose_build(cuda, args.cpu)
    print(f"[SONATLAS] NVIDIA 드라이버 CUDA: {'.'.join(map(str, cuda)) if cuda else '없음'} → PyTorch 빌드: {build}")
    if cuda and build == "cpu" and not args.cpu:
        print("[SONATLAS] 드라이버가 오래되어 맞는 CUDA 빌드가 없습니다. GPU 드라이버를 업데이트하면 GPU를 쓸 수 있습니다.")
    for cmd in commands(build):
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not args.dry_run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print("[SONATLAS] 설치에 실패했습니다. 인터넷 연결과 위쪽 오류 메시지를 확인하세요.")
                return r.returncode
    if not args.dry_run:
        print("[SONATLAS] 설치를 마쳤습니다. 프로그램의 분석 설정 패널에서 [GPU 확인]으로 확인할 수 있습니다.")
        print("[SONATLAS] 처음 분석할 때 htdemucs 모델(약 80MB)을 내려받습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
