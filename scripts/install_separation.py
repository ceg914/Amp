"""보컬 분리(Demucs htdemucs)에 필요한 PyTorch와 demucs를 설치한다.

- NVIDIA GPU가 있으면 nvidia-smi가 알려 주는 드라이버의 CUDA 버전에 맞는 PyTorch를 고른다.
  nvidia-smi가 PATH에 없으면 Windows 표준 위치(System32, NVSMI)도 찾는다.
- GPU가 없으면 CPU판을 설치한다 (Windows는 PyPI 기본 휠이 CPU판).
- GPU판을 설치할 때 CPU판 PyTorch가 이미 있으면 먼저 지운다 (같은 버전 번호라 그대로 남는 것을 막는다).
- 설치 뒤 PyTorch가 실제로 GPU를 인식하는지 확인해 알려 준다.
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


def find_nvidia_smi(which=shutil.which, isfile=os.path.isfile, nt: bool | None = None) -> str | None:
    exe = which("nvidia-smi")
    if exe:
        return exe
    if (os.name == "nt") if nt is None else nt:
        for p in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
                  os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI",
                               "nvidia-smi.exe")):
            if isfile(p):
                return p
    return None


def installed_torch(python: str = sys.executable) -> tuple[str, str | None] | None:
    """(버전, CUDA 버전 또는 None). 설치돼 있지 않으면 None."""
    code = "import torch;print(torch.__version__);print(torch.version.cuda or '')"
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines:
        return None
    return lines[0].strip(), (lines[1].strip() or None) if len(lines) > 1 else None


def detect_cuda() -> tuple[int, int] | None:
    exe = find_nvidia_smi()
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


def commands(build: str, python: str = sys.executable, platform: str = sys.platform,
             existing: tuple[str, str | None] | None = None) -> list[list[str]]:
    """existing = 이미 설치된 (torch 버전, CUDA 버전). GPU판을 설치하는데 CPU판이 있으면 먼저 지운다."""
    cmds = []
    if build != "cpu" and existing is not None and existing[1] is None:
        cmds.append([python, "-m", "pip", "uninstall", "-y", "torch"])
    torch = [python, "-m", "pip", "install", "--upgrade", "torch"]
    if build != "cpu" or not platform.startswith("win"):
        # Windows CPU판은 PyPI 기본 휠. 그 밖에는 PyTorch 인덱스를 지정한다 (Linux PyPI 휠은 CUDA 포함이라 매우 크다)
        torch += ["--index-url", TORCH_INDEX + build]
    cmds.append(torch)
    cmds.append([python, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements-separation.txt")])
    return cmds


def verify(python: str = sys.executable) -> tuple[bool, str]:
    """(GPU 사용 가능, 설명)."""
    code = ("import torch;ok=torch.cuda.is_available();"
            "print(ok);print(torch.__version__);print(torch.cuda.get_device_name(0) if ok else '')")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"PyTorch를 확인하지 못했습니다: {e}"
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or len(lines) < 2:
        return False, "PyTorch를 불러오지 못했습니다: " + (out.stderr.strip().splitlines() or ["알 수 없는 오류"])[-1]
    ok = lines[0].strip() == "True"
    return ok, (f"GPU 사용 가능: {lines[2].strip()} (PyTorch {lines[1].strip()})" if ok and len(lines) > 2
                else f"GPU를 쓸 수 없습니다 (PyTorch {lines[1].strip()}). CPU로 분리합니다.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU판을 설치")
    ap.add_argument("--dry-run", action="store_true", help="실행할 명령만 보여 줌")
    args = ap.parse_args(argv)
    smi = find_nvidia_smi()
    cuda = detect_cuda()
    build = choose_build(cuda, args.cpu)
    print(f"[SONATLAS] nvidia-smi: {smi or '찾지 못함'}")
    print(f"[SONATLAS] NVIDIA 드라이버 CUDA: {'.'.join(map(str, cuda)) if cuda else '없음'} → PyTorch 빌드: {build}")
    if cuda and build == "cpu" and not args.cpu:
        print("[SONATLAS] 드라이버가 오래되어 맞는 CUDA 빌드가 없습니다. GPU 드라이버를 업데이트하면 GPU를 쓸 수 있습니다.")
    if smi and cuda is None:
        print("[SONATLAS] nvidia-smi는 있지만 CUDA 버전을 읽지 못했습니다. GPU 드라이버를 다시 설치하거나 업데이트하세요.")
    existing = installed_torch()
    if existing:
        print(f"[SONATLAS] 지금 설치된 PyTorch: {existing[0]} ({'CUDA ' + existing[1] if existing[1] else 'CPU판'})")
    for cmd in commands(build, existing=existing):
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not args.dry_run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print("[SONATLAS] 설치에 실패했습니다. 인터넷 연결과 위쪽 오류 메시지를 확인하세요.")
                return r.returncode
    if not args.dry_run:
        ok, msg = verify()
        print(f"[SONATLAS] {msg}")
        if build != "cpu" and not ok:
            print("[SONATLAS] GPU판을 설치했는데 GPU를 인식하지 못합니다. GPU 드라이버를 업데이트한 뒤 다시 실행해 보세요.")
        print("[SONATLAS] 설치를 마쳤습니다. 프로그램의 분석 설정 패널에서 [GPU 확인]으로 확인할 수 있습니다.")
        print("[SONATLAS] 처음 분석할 때 htdemucs 모델(약 80MB)을 내려받습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
