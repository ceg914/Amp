"""분석 선택 (v0.7.0 명세 16.4): 어떤 분석을 할지 고르면 필요한 앞 단계를 더하고, 할 수 없는 것은 이유와 함께 뺀 실행 순서를 만든다.

Qt 비의존. 실제 실행은 화면 쪽(main_window.AnalysisChain)이 기존 기능(보컬 분석·다중 피치·음향 분석·가사 인식·정렬)을
차례로 부른다. '전체'를 골라도 실험 기능과 설치되지 않은 외부 모델은 넣지 않는다.

발음·멜리스마는 따로 실행하지 않는다: 가사(·정렬·음표)가 있으면 편집할 때마다 바로 계산된다 ('자동').
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    step: str  # 실행 단계 (같은 단계는 한 번에 돈다). "auto" = 따로 실행하지 않음
    needs: tuple[str, ...] = ()  # 꼭 필요한 기능 (고르면 함께 고른다)
    uses: tuple[str, ...] = ()  # 있으면 더 좋은 기능 (함께 고르지는 않음, 표시만)
    models: tuple[str, ...] = ()  # 이 중 하나는 쓸 수 있어야 함 (model_registry id). 비면 조건 없음
    experimental: bool = False
    note: str = ""


FEATURES: tuple[Feature, ...] = (
    Feature("separation", "보컬 분리", "vocal", models=("htdemucs", "mel_roformer"),
            note="분석 설정의 분리 방식(Demucs·RoFormer)을 씁니다. 끄면 원본 그대로 분석합니다."),
    Feature("pitch", "단성 피치 (pYIN)·음표", "vocal", uses=("separation",), models=("pyin",)),
    Feature("multipitch", "다중 피치·성부 추적", "multipitch", needs=("pitch",), experimental=True,
            note="실제 곡에서 정확도를 확인하지 않은 실험 기능입니다."),
    Feature("harmony", "주선율·화음 후보", "multipitch", needs=("multipitch",), experimental=True,
            note="성부 추적 결과로 계산합니다 (다중 피치와 함께 돕니다)."),
    Feature("vibrato", "비브라토", "acoustics", needs=("pitch",)),
    Feature("breath", "호흡·프레이즈", "acoustics", needs=("pitch",)),
    Feature("boundary", "발성 경계", "acoustics", needs=("pitch",),
            note="결과는 음표에 자동으로 적용하지 않습니다 (교정 제안으로만)."),
    Feature("asr", "가사 인식 (Whisper)", "asr", uses=("separation",), models=("whisper",),
            note="이미 가사가 있으면 바꿀지 묻습니다 (바꾸지 않으면 인식 원본만 보관)."),
    Feature("align", "가사 정렬 (MMS)", "align", needs=("lyrics",), uses=("separation",), models=("mms_fa",)),
    Feature("pronunciation", "발음", "auto", needs=("lyrics",), note="가사가 있으면 바로 만듭니다."),
    Feature("melisma", "멜리스마", "auto", needs=("align", "pitch"), note="정렬·음표가 있으면 바로 판정합니다."),
)
BY_KEY = {f.key: f for f in FEATURES}
STEP_ORDER = ("vocal", "multipitch", "acoustics", "asr", "align")
STEP_LABELS = {"vocal": "보컬 분석 (분리 + 피치 + 음표)", "multipitch": "다중 피치·성부", "acoustics": "음향 분석",
               "asr": "가사 인식", "align": "가사 정렬"}


@dataclass
class Context:
    usable_models: set = field(default_factory=set)  # 지금 쓸 수 있는 모델 id (model_registry)
    has_lyrics: bool = False  # 이미 가사가 있다
    has_vocal_analysis: bool = False  # 이미 보컬 분석 결과가 있다
    lite: bool = False


@dataclass
class Plan:
    chosen: list[str]  # 실제로 하는 기능 (사용자가 고른 것 + 앞 단계)
    added: dict[str, str]  # 앞 단계라 더한 기능 → 어느 기능 때문인지
    skipped: dict[str, str]  # 할 수 없어 뺀 기능 → 이유
    steps: list[str]  # 실행 순서 (STEP_ORDER 안에서)
    separation: bool = True  # 보컬 분석에서 분리할지

    def summary(self) -> str:
        lines = [f"{i + 1}. {STEP_LABELS[s]}" for i, s in enumerate(self.steps)]
        if not lines:
            lines = ["실행할 단계가 없습니다."]
        auto = [BY_KEY[k].label for k in self.chosen if BY_KEY[k].step == "auto"]
        if auto:
            lines.append("자동 (따로 실행하지 않음): " + ", ".join(auto))
        for k, why in self.added.items():
            lines.append(f"더함: {BY_KEY[k].label} ← {why}")
        for k, why in self.skipped.items():
            lines.append(f"뺌: {BY_KEY[k].label} — {why}")
        return "\n".join(lines)


def available(f: Feature, ctx: Context) -> tuple[bool, str]:
    """가벼운 모드의 무거운 모델은 모델 등록 정보가 이미 '사용 불가'로 둔다 (usable_models에 없음)."""
    if ctx.lite and f.experimental:
        return False, "가벼운 모드에서는 실험 기능을 쓰지 않습니다."
    if f.models and not set(f.models) & set(ctx.usable_models):
        return False, "필요한 모델이 설치되지 않았습니다: " + ", ".join(f.models)
    return True, ""


def defaults(ctx: Context) -> list[str]:
    """'전체'로 고를 때: 실험 기능과 쓸 수 없는 기능은 빼고, 가사 인식은 가사가 없을 때만."""
    out = []
    for f in FEATURES:
        if f.experimental or not available(f, ctx)[0]:
            continue
        if f.key == "asr" and ctx.has_lyrics:
            continue
        out.append(f.key)
    return out


def resolve(selected, ctx: Context) -> Plan:
    want = [k for k in dict.fromkeys(selected) if k in BY_KEY]
    chosen: list[str] = []
    added: dict[str, str] = {}
    skipped: dict[str, str] = {}

    def add(key: str, because: str | None) -> bool:
        if key in chosen:
            return True
        if key == "lyrics":  # 가사: 이미 있거나, 가사 인식을 고름
            return ctx.has_lyrics or "asr" in want and add("asr", None)
        f = BY_KEY[key]
        ok, why = available(f, ctx)
        if not ok:
            skipped[key] = why
            return False
        for d in f.needs:
            if not add(d, f.label):
                need = "가사 (직접 입력하거나 가사 인식을 고름)" if d == "lyrics" else BY_KEY[d].label
                skipped[key] = f"먼저 필요한 것이 없습니다: {need}"
                return False
        if because is not None and key not in want:
            added[key] = because
        chosen.append(key)
        return True

    for k in want:
        add(k, None)
    # 피치가 필요한 분석만 골랐고 보컬 분석이 이미 있으면, 보컬 분석을 다시 하지 않는다
    if "pitch" in added and ctx.has_vocal_analysis:
        chosen.remove("pitch")
        added.pop("pitch")
    steps = [s for s in STEP_ORDER if any(BY_KEY[k].step == s for k in chosen)]
    return Plan(chosen, added, skipped, steps, separation="separation" in chosen)
