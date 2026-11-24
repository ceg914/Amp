"""가벼운 모드 (크롬북 등 메모리·CPU가 작은 컴퓨터).

켜는 방법: ``python -m amp --lite`` 또는 환경 변수 ``SONATLAS_LITE=1``. ``SONATLAS_LITE=auto`` (run.sh 기본값)는
PyTorch가 설치되어 있지 않을 때만 켠다. Windows run.bat은 이 변수를 쓰지 않으므로 기존 동작 그대로다.

가벼운 모드에서 숨기는 것: 모델 비교, Hybrid 보컬, Mel-Band RoFormer, GPU 확인, 가사 자동 인식(Whisper)·자동 정렬(MMS).
남는 것: 피치·음표 검출, 편집, 가사 직접 입력과 음표 순서 연결, 재생·반복·느린 재생, 검토·교정 제안·품질, 내보내기,
외부 분리 결과 가져오기. 프로젝트 파일 형식은 같아 Windows에서 만든 프로젝트를 그대로 열 수 있다
(이미 분리해 둔 결과가 있으면 다시 분리하지 않고 쓴다).
"""

from __future__ import annotations

import importlib.util
import os

ENV = "SONATLAS_LITE"


def lite_mode(env: dict | None = None, find_spec=importlib.util.find_spec) -> bool:
    value = str((os.environ if env is None else env).get(ENV, "")).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value == "auto":
        return find_spec("torch") is None
    return False


def separation_installed(find_spec=importlib.util.find_spec) -> bool:
    return find_spec("torch") is not None and find_spec("demucs") is not None
