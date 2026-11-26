"""분석 프로필 (v0.7.0, 명세 13.4·13.3). Qt 비의존.

Lite·Standard·Advanced. 프로필은 '켤 수 있는 기능'의 상한이고, 실제로 켜지는지는 설치된 모델과 장치로 정한다
(모델 등록 정보 ``model_registry.scan`` 결과). 모델이 없거나 실패한 기능만 꺼지고 나머지는 그대로 쓴다.

- Lite: pYIN, 기본 음향 분석(비브라토·호흡·발성 경계 등 신호 처리만), 편집, 이미 있는 분리 파일 재생
- Standard: + 보컬 분리, 가사 인식·정렬
- Advanced: + 외부 피치 모델, 다중 피치, 성부 추적, 모델 비교
가벼운 모드(app/lite.py)는 그대로 둔다. 가벼운 모드에서는 Lite 프로필만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

from amp.engine import model_registry as R

LITE, STANDARD, ADVANCED = "lite", "standard", "advanced"
PROFILES = (LITE, STANDARD, ADVANCED)
PROFILE_LABELS = {LITE: "Lite", STANDARD: "Standard", ADVANCED: "Advanced"}
_LEVEL = {p: i for i, p in enumerate(PROFILES)}


@dataclass(frozen=True)
class Feature:
    feature_id: str
    label: str
    profile: str  # 이 기능이 들어가는 가장 낮은 프로필
    models: tuple[str, ...] = ()  # 이 중 하나라도 쓸 수 있으면 됨 (비우면 신호 처리만: 늘 가능)
    min_models: int = 1
    experimental: bool = False
    depends: tuple[str, ...] = ()  # 먼저 있어야 하는 기능 (분석 선택 창 표시용)


SEPARATION_MODELS = ("htdemucs", "htdemucs_6s", "mel_roformer")
EXTERNAL_PITCH_MODELS = ("crepe_full", "crepe_tiny", "fcpe")

FEATURES: dict[str, Feature] = {f.feature_id: f for f in (
    Feature("mono_pitch", "단성 피치 (pYIN)", LITE, ("pyin",)),
    Feature("vibrato", "비브라토", LITE, depends=("mono_pitch",)),
    Feature("breath", "호흡·프레이즈", LITE),
    Feature("phonation", "발성·음향 지표", LITE, depends=("mono_pitch",)),
    Feature("boundaries", "발성 경계", LITE, depends=("mono_pitch",)),
    Feature("separation", "보컬 분리", STANDARD, SEPARATION_MODELS),
    Feature("lyrics_recognition", "가사 인식", STANDARD, ("whisper",), depends=("separation",)),
    Feature("lyrics_alignment", "가사 정렬", STANDARD, ("mms_fa",), depends=("separation",)),
    Feature("external_pitch", "외부 피치 모델", ADVANCED, EXTERNAL_PITCH_MODELS, experimental=True,
            depends=("separation",)),
    Feature("pitch_compare", "피치 모델 비교", ADVANCED, ("pyin", *EXTERNAL_PITCH_MODELS), min_models=2,
            experimental=True),
    Feature("separation_compare", "분리 모델 비교", ADVANCED, SEPARATION_MODELS, min_models=2),
    Feature("multi_pitch", "다중 피치", ADVANCED, experimental=True, depends=("separation",)),
    Feature("voice_tracking", "성부 추적", ADVANCED, experimental=True, depends=("multi_pitch",)),
    Feature("harmony", "주선율·화음", ADVANCED, experimental=True, depends=("voice_tracking",)),
)}


@dataclass(frozen=True)
class FeatureState:
    feature: Feature
    enabled: bool
    reason: str = ""
    models: tuple[str, ...] = ()  # 쓸 수 있는 모델


def resolve(profile: str, statuses: list[R.ModelStatus], lite_mode: bool = False) -> dict[str, FeatureState]:
    """기능 → 켤 수 있는지와 이유. 가벼운 모드면 프로필을 Lite로 낮춘다."""
    if profile not in PROFILES:
        profile = STANDARD
    if lite_mode:
        profile = LITE
    usable = {s.model_id: s for s in statuses if s.usable}
    out: dict[str, FeatureState] = {}
    for fid, f in FEATURES.items():
        if _LEVEL[f.profile] > _LEVEL[profile]:
            out[fid] = FeatureState(f, False, f"{PROFILE_LABELS[f.profile]} 프로필 기능입니다.")
            continue
        if not f.models:
            out[fid] = FeatureState(f, True)
            continue
        have = tuple(m for m in f.models if m in usable)
        if len(have) >= f.min_models:
            out[fid] = FeatureState(f, True, "", have)
            continue
        why = []
        for m in f.models:
            st = R.status_of(statuses, m)
            if st is not None and not st.usable:
                why.append(f"{st.name}: {st.state_label}" + (f" ({st.reason})" if st.reason else ""))
        need = "모델 2개 이상이 필요합니다. " if f.min_models > 1 else ""
        out[fid] = FeatureState(f, False, need + ("; ".join(why) or "쓸 수 있는 모델이 없습니다."), have)
    return out


def default_profile(lite_mode: bool) -> str:
    return LITE if lite_mode else STANDARD
