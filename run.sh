#!/usr/bin/env bash
# SONATLAS 실행 (크롬북 리눅스·데비안·우분투). 처음이면 ./install_chromebook.sh 를 먼저 실행하세요.
# 기본은 SONATLAS_LITE=auto: PyTorch가 없으면 가벼운 모드로 켭니다. 항상 전체 기능: ./run.sh --full
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv/bin/python ]; then
  echo "[SONATLAS] 가상 환경(.venv)이 없습니다. 먼저 ./install_chromebook.sh 를 실행하세요." >&2
  exit 1
fi
export SONATLAS_LITE="${SONATLAS_LITE:-auto}"
exec .venv/bin/python -m amp "$@"
