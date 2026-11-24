#!/usr/bin/env bash
# SONATLAS 설치 — 크롬북 리눅스 개발 환경(Debian 12·13, x86_64)과 일반 데비안·우분투.
#
#   ./install_chromebook.sh               기본 설치 (가벼운 모드: 피치·음표·편집·가사 직접 입력·내보내기)
#   ./install_chromebook.sh --separation  보컬 분리(CPU용 PyTorch + Demucs, 약 1.2GB 추가)도 설치
#   ./install_chromebook.sh --lyrics      가사 자동 인식(Whisper)도 설치 (분리 설치 필요, 모델은 처음 쓸 때 1.5~3GB 내려받음)
#   ./install_chromebook.sh --no-apt      시스템 패키지 설치(sudo apt-get)를 건너뜀
#
# 같은 명령을 여러 번 실행해도 됩니다 (이미 설치된 것은 건너뜀).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"
WITH_SEP=0; WITH_LYR=0; APT=1
for a in "$@"; do
  case "$a" in
    --separation) WITH_SEP=1 ;;
    --lyrics) WITH_SEP=1; WITH_LYR=1 ;;
    --no-apt) APT=0 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "[SONATLAS] 모르는 옵션: $a (--help 참고)" >&2; exit 2 ;;
  esac
done

echo "[SONATLAS] CPU: $(uname -m), $(ldd --version 2>/dev/null | head -1)"
if [ "$(uname -m)" != "x86_64" ]; then
  echo "[SONATLAS] 주의: x86_64가 아닙니다. ARM용 PySide6는 glibc 2.39 이상에서만 pip로 설치됩니다." >&2
fi
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
echo "[SONATLAS] 리눅스 환경 메모리: ${MEM_MB}MB"

# 1) 시스템 패키지: Python 가상 환경, Qt 화면(X11·Wayland)·소리(PulseAudio)에 필요한 라이브러리
if [ "$APT" = 1 ]; then
  PKGS="python3 python3-venv python3-pip libegl1 libgl1 libglib2.0-0 libfontconfig1 libdbus-1-3 libxkbcommon0 \
libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
libxcb-shape0 libxcb-xfixes0 libxcb-xkb1 libx11-xcb1 libwayland-client0 libwayland-cursor0 libpulse0"
  MISSING=""
  for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
  if [ -n "$MISSING" ]; then
    echo "[SONATLAS] 시스템 패키지 설치:$MISSING"
    sudo apt-get update
    # shellcheck disable=SC2086
    sudo apt-get install -y $MISSING
  else
    echo "[SONATLAS] 시스템 패키지는 이미 있습니다."
  fi
fi

# 2) 가상 환경과 기본 패키지 (PySide6·numpy·scipy·librosa·soundfile·mido·PyYAML)
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# 3) 선택: 보컬 분리 (CPU)
if [ "$WITH_SEP" = 1 ]; then
  if [ "$MEM_MB" -lt 6000 ]; then
    echo "[SONATLAS] 주의: 보컬 분리는 5분 곡에서 메모리를 2.0~2.3GB 씁니다. 지금 ${MEM_MB}MB라 다른 앱을 닫고 쓰세요." >&2
  fi
  .venv/bin/python scripts/install_separation.py --cpu
fi
if [ "$WITH_LYR" = 1 ]; then
  .venv/bin/python scripts/install_lyrics.py
fi

# 4) 시작 확인 (화면 없이 패키지만)
.venv/bin/python -c "import PySide6, numpy, scipy, librosa, soundfile, mido, yaml; print('[SONATLAS] 패키지 확인 완료')"

# 5) 크롬북 앱 목록(런처)에 바로가기
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/sonatlas.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=SONATLAS
Comment=보컬 채보 편집기
Exec="$ROOT/run.sh" %f
Icon=$ROOT/amp/app/theme/icons/audio-waveform.svg
Terminal=false
Categories=AudioVideo;Audio;
DESKTOP
chmod +x "$ROOT/run.sh"
echo "[SONATLAS] 설치 완료. 런처의 'Linux 앱'에서 SONATLAS를 누르거나 ./run.sh 로 실행하세요."
