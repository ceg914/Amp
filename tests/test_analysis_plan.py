"""v0.7.0 명세 16.4 분석 선택: 의존 관계로 앞 단계를 더하고, 할 수 없는 것은 이유와 함께 빼고, '전체'도 실험·미설치 모델은 넣지 않는다."""

from amp.engine import analysis_plan as AP

ALL = {"pyin", "htdemucs", "whisper", "mms_fa"}


def test_dependencies_are_added_and_ordered():
    p = AP.resolve(["melisma"], AP.Context(ALL, has_lyrics=False))
    assert "melisma" in p.skipped and "가사" in p.skipped["melisma"]  # 가사가 없고 인식도 고르지 않음
    p = AP.resolve(["melisma", "asr"], AP.Context(ALL))
    assert set(p.chosen) >= {"asr", "align", "pitch", "melisma"} and p.added["align"] == "멜리스마"
    assert p.steps == ["vocal", "asr", "align"]
    p = AP.resolve(["boundary"], AP.Context(ALL))
    assert p.steps == ["vocal", "acoustics"] and p.added == {"pitch": "발성 경계"} and not p.separation
    p = AP.resolve(["boundary"], AP.Context(ALL, has_vocal_analysis=True))  # 보컬 분석이 이미 있으면 다시 하지 않음
    assert p.steps == ["acoustics"]
    p = AP.resolve(["align"], AP.Context(ALL, has_lyrics=True))
    assert p.steps == ["align"] and not p.skipped


def test_unavailable_and_experimental_features():
    ctx = AP.Context({"pyin"})
    p = AP.resolve(["separation", "asr", "vibrato"], ctx)
    assert "separation" in p.skipped and "whisper" in p.skipped["asr"] and p.steps == ["vocal", "acoustics"]
    full = AP.defaults(AP.Context(ALL))
    assert "multipitch" not in full and "harmony" not in full  # 실험 기능은 '전체'에 넣지 않는다
    assert {"separation", "pitch", "vibrato", "breath", "boundary", "asr", "align"} <= set(full)
    assert "asr" not in AP.defaults(AP.Context(ALL, has_lyrics=True))  # 가사가 있으면 인식은 고르지 않는다
    assert "separation" not in AP.defaults(ctx) and "asr" not in AP.defaults(ctx)
    lite = AP.Context({"pyin"}, lite=True)
    assert "multipitch" in AP.resolve(["multipitch"], lite).skipped
    p = AP.resolve(["harmony"], AP.Context(ALL))
    assert p.steps == ["vocal", "multipitch"] and p.added.get("multipitch") == "주선율·화음 후보"
    text = p.summary()
    assert "1. 보컬 분석" in text and "더함:" in text
