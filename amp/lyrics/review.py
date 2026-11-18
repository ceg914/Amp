"""자동 가사 인식 결과에서 '검토 필요' 구간을 고른다 (순수 함수).

노래는 말과 달라 인식 모델이 없는 가사를 만들어 내기 쉽다(환각). 다음 중 하나라도 걸리면 그 구간을 검토 필요로 표시한다.
어떤 구간도 지우지 않는다. 확정은 사용자가 한다.
"""

from __future__ import annotations

import re

from amp.lyrics.model import AsrSegment, AsrWord

LOW_LOGPROB = -1.0
HIGH_NO_SPEECH = 0.6
HIGH_COMPRESSION = 2.4
LOW_ACTIVITY = 0.25  # 구간 안에서 보컬이 들리는 프레임 비율이 이보다 낮으면
LOW_WORD_PROB = 0.3
MAX_SYL_PER_S = 12.0

# 노래·영상 자막 학습 데이터에서 자주 튀어나오는 문구 (실제로 부르지 않았는데 나오는 경우가 많다)
HALLUCINATION_PHRASES = (
    "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", "구독과 좋아요", "구독 좋아요", "자막 제공", "자막",
    "다음 영상에서", "MBC 뉴스", "KBS 뉴스", "SBS 뉴스", "끝까지 시청",
    "thank you for watching", "thanks for watching", "please subscribe", "subtitles by", "like and subscribe",
    "ご視聴ありがとうございました", "チャンネル登録", "字幕",
)

R_LOW_CONF = "인식 확신이 낮음"
R_NO_SPEECH = "목소리가 아닐 가능성이 높음"
R_REPETITIVE = "같은 말이 반복됨 (환각 의심)"
R_NO_VOCAL = "이 구간에 보컬이 거의 들리지 않음 (환각 의심)"
R_NO_VOCAL_RANGE = "'보컬 없는 구간'으로 지정한 곳과 겹침"
R_PHRASE = "노래와 상관없이 자주 생기는 문구 (환각 의심)"
R_REPEATED_LINE = "같은 문장이 연달아 3번 이상 나옴"
R_RATE = "음절 속도가 비정상적임"


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s.lower())


def _syllable_count(text: str) -> int:
    from amp.lyrics.text import tokenize_line

    return sum(len(w.syllables) for w in tokenize_line(text))


def review_segment(text: str, start: float, end: float, *, avg_logprob=None, no_speech_prob=None,
                   compression_ratio=None, activity: float | None = None, in_no_vocal: float = 0.0,
                   words: tuple[AsrWord, ...] = ()) -> list[str]:
    reasons: list[str] = []
    if avg_logprob is not None and avg_logprob < LOW_LOGPROB:
        reasons.append(R_LOW_CONF)
    # 노래는 말소리 판정(no_speech)이 대부분 높게 나온다. 보컬 음량도 작을 때만 근거로 쓴다
    if no_speech_prob is not None and no_speech_prob > HIGH_NO_SPEECH and activity is not None and activity < 0.5:
        reasons.append(R_NO_SPEECH)
    if compression_ratio is not None and compression_ratio > HIGH_COMPRESSION:
        reasons.append(R_REPETITIVE)
    if activity is not None and activity < LOW_ACTIVITY:
        reasons.append(R_NO_VOCAL)
    if in_no_vocal > 0.5:
        reasons.append(R_NO_VOCAL_RANGE)
    n = _norm(text)
    if n and any(_norm(p) in n for p in HALLUCINATION_PHRASES):
        reasons.append(R_PHRASE)
    dur = max(end - start, 1e-3)
    syl = _syllable_count(text)
    if syl and (syl / dur > MAX_SYL_PER_S):
        reasons.append(R_RATE)
    low = [w.text.strip() for w in words if w.probability is not None and w.probability < LOW_WORD_PROB]
    if low and len(low) >= max(2, len(words) // 2):
        reasons.append("확신이 낮은 단어가 많음: " + ", ".join(low[:5]))
    return reasons


def mark_repeats(segments: list[AsrSegment]) -> list[AsrSegment]:
    """같은 문장이 연달아 3번 이상 나오면 세 번째부터 표시한다 (반복 환각). 노래 후렴은 보통 사이에 다른 줄이 있다."""
    from dataclasses import replace

    out = []
    run = 0
    prev = None
    for s in segments:
        key = _norm(s.text)
        run = run + 1 if key and key == prev else 1
        prev = key
        if run >= 3 and R_REPEATED_LINE not in s.review:
            s = replace(s, review=s.review + (R_REPEATED_LINE,))
        out.append(s)
    return out
