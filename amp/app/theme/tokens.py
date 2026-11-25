"""SONATLAS 디자인 토큰 — Obsidian Violet (SONATLAS_UI_SPEC.md 3·4장).

확정 팔레트 8색은 바꾸지 않는다. 상태색·피아노 롤 세부 색은 추가 토큰으로 둔다.
"""

from __future__ import annotations

# --- 확정 팔레트 (명세 3장) ---
BG_PRIMARY = "#121116"  # 창 전체, 편집 영역 바탕
BG_PANEL = "#201E27"  # 사이드바, 도킹 패널
BG_RAISED = "#2B2835"  # 입력 필드, 버튼, 선택 가능 표면
ACCENT = "#AD96F5"  # 주요 버튼, 선택/포커스, 음표 기본색
ACCENT_2 = "#66C8BE"  # 실제 피치 곡선, 보조 분석 표시
TEXT = "#F0ECF6"  # 주요 글자
TEXT_2 = "#A29BAB"  # 보조 레이블
BORDER = "#3B3546"  # 경계선, 격자 기본선

# --- 추가 토큰 (상태색: 다크 배경 대비를 고려해 고름, 색만으로 구분하지 않고 아이콘·문구를 함께 쓴다) ---
SUCCESS = "#7FD3A0"
WARNING = "#E8B04B"
ERROR = "#EF7A86"
DISABLED = "#5E5868"
ACCENT_HOVER = "#C0AEF8"
ACCENT_PRESSED = "#9A80EE"
RAISED_HOVER = "#353141"

# --- 피아노 롤·파형 ---
ROLL_ROW_WHITE = "#17151C"
ROLL_ROW_BLACK = BG_PRIMARY
ROLL_GRID_SUB = "#1E1B25"
ROLL_GRID_BEAT = "#2A2633"
ROLL_GRID_BAR = BORDER
ROLL_OCTAVE = "#4A4358"  # 옥타브 경계 강조
PLAYHEAD = TEXT
WAVE_BG = "#18161E"

# 트랙 표시 색 (명세 6.2: 트랙마다 표시 색상)
TRACK_COLORS = {"original": TEXT_2, "vocals": ACCENT_2, "instrumental": ACCENT,
                # v0.6.6 악기 트랙 (분리한 악기 소리). 상태색과 겹치지 않게 채도를 낮춘 색
                "inst_drums": "#D2A066", "inst_bass": "#7FA6E0", "inst_guitar": "#DC8FA6", "inst_piano": "#B5CB7E",
                "inst_other": "#8F879B"}

# --- 글꼴 (명세 4장) ---
FONT_UI = "Pretendard"
FONT_MONO = "JetBrains Mono"
FONT_UI_FALLBACKS = ("Malgun Gothic", "Segoe UI", "Noto Sans KR", "Apple SD Gothic Neo")
FONT_MONO_FALLBACKS = ("Cascadia Mono", "Consolas", "D2Coding", "DejaVu Sans Mono")
SIZE_TITLE = 18
SIZE_PANEL_TITLE = 13
SIZE_BODY = 13
SIZE_MONO = 12
SIZE_SMALL = 11

# --- 모양 (명세 8장) ---
RADIUS_BUTTON = 6
RADIUS_MODAL = 10
