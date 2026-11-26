"""가사 인식 라이브러리(openai-whisper)와 발음 사전 설치. whisper는 PyTorch가 먼저 있어야 한다(install_separation.bat).

whisper는 PyPI에 소스 배포만 있어 설치할 때 빌드한다(순수 파이썬). 의존: numba(이미 librosa가 씀), tiktoken, tqdm.
리눅스 x86_64에서만 triton을 요구한다(Windows에서는 설치하지 않는다).

v0.7.0 발음 사전 (PyTorch 필요 없음):
- 영어: cmudict 1.1.3의 data/cmudict.dict (자료 BSD-2-Clause 계열). 패키지 코드(GPL-3.0)는 불러오지 않으므로 --no-deps
- 일본어 한자 읽기 (--japanese): fugashi(MIT·BSD-3, MeCab) + unidic-lite(BSD-3, 설치 약 250MB)
- --dictionaries-only: whisper 없이 발음 사전만
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CMUDICT = "cmudict==1.1.3"
JAPANESE = ("fugashi==1.5.2", "unidic-lite==1.0.8")


def commands(python: str = sys.executable, japanese: bool = False, dictionaries_only: bool = False) -> list[list[str]]:
    out = [] if dictionaries_only else [[python, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements-lyrics.txt")]]
    out.append([python, "-m", "pip", "install", "--no-deps", CMUDICT])
    if japanese:
        out.append([python, "-m", "pip", "install", "--no-deps", *JAPANESE])
    return out


def has_torch(python: str = sys.executable) -> bool:
    try:
        r = subprocess.run([python, "-c", "import torch"], capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def verify(python: str = sys.executable) -> tuple[bool, str]:
    code = "import whisper, torch;print(whisper.__version__);print(torch.cuda.is_available())"
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"설치를 확인하지 못했습니다: {e}"
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or len(lines) < 2:
        return False, "whisper를 불러오지 못했습니다: " + (out.stderr.strip().splitlines() or ["알 수 없는 오류"])[-1]
    gpu = "GPU로 인식합니다" if lines[1].strip() == "True" else "GPU를 쓸 수 없어 CPU로 인식합니다 (느림)"
    return True, f"whisper {lines[0].strip()} 설치됨 · {gpu}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="실행할 명령만 보여 줌")
    ap.add_argument("--japanese", action="store_true", help="일본어 한자 읽기 사전(MeCab+UniDic, 약 250MB)도 설치")
    ap.add_argument("--dictionaries-only", action="store_true", help="whisper 없이 발음 사전만 설치 (PyTorch 필요 없음)")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.dictionaries_only and not has_torch():
        print("[SONATLAS] PyTorch가 없습니다. install_separation.bat 을 먼저 실행해 주세요 (GPU판을 맞춰 설치합니다).")
        print("[SONATLAS] 발음 사전만 설치하려면 --dictionaries-only 를 붙이세요.")
        return 1
    for cmd in commands(japanese=args.japanese, dictionaries_only=args.dictionaries_only):
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not args.dry_run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print("[SONATLAS] 설치에 실패했습니다. 인터넷 연결과 위쪽 오류 메시지를 확인하세요.")
                return r.returncode
    if not args.dry_run and args.dictionaries_only:
        print("[SONATLAS] 발음 사전을 설치했습니다. SONATLAS를 다시 시작하면 쓰입니다.")
        return 0
    if not args.dry_run:
        ok, msg = verify()
        print(f"[SONATLAS] {msg}")
        if not ok:
            return 1
        print("[SONATLAS] 처음 인식할 때 모델을 내려받습니다 (기본 turbo 약 1.5GB).")
        print("[SONATLAS] 처음 정렬할 때 정렬 모델 MMS_FA(약 1.2GB, CC-BY-NC 4.0 비상업 라이선스)를 내려받습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
