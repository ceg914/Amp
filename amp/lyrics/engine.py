"""가사 인식·정렬 실행 (Qt 비의존). 무거운 계산은 모두 작업 프로세스(``amp.engine.worker``)에서 한다.

캐시 (프로젝트 cache/, 지워도 안전)
- asr        : 보컬 id + 모델·언어·장치·보컬 없는 구간 → asr.json
- mms        : 보컬 id + 정렬 모델 → emissions.npz (프레임별 글자 확률·보컬 크기)
가사를 고친 뒤 다시 정렬할 때는 emissions를 재사용하고 numpy 정렬만 다시 돌린다(수 초).
보컬을 다시 분리하면 보컬 id가 바뀌어 캐시가 자동으로 무효가 된다. 가사 편집본은 캐시와 상관없이 그대로다.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from amp.engine.cache import AnalysisCache
from amp.engine.context import CancelToken
from amp.engine.worker_client import run_worker
from amp.lyrics import ops, review
from amp.lyrics.model import (
    AsrSegment,
    AsrWord,
    LyricLine,
    LyricsState,
    LyricsText,
    Recognition,
    SyllableTiming,
)

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]
ASR_VERSION = 2  # 2: 빔 탐색 + 빈 구간 다시 듣기 (이전 캐시는 쓰지 않음)
MMS_VERSION = 1


@dataclass(frozen=True)
class LyricsRequest:
    vocals_path: str
    input_id: str  # 보컬 식별자 (분리 결과 id). 바뀌면 캐시가 무효
    cache_root: str | None
    whisper_root: str
    mms_root: str
    device: str = "auto"
    model: str = "auto"
    language: str = "auto"
    no_vocal_ranges: tuple = ()
    use_cache: bool = True
    numba_cache_dir: str | None = None


@dataclass
class AlignOutput:
    state: LyricsState  # 요청 시점 가사에 결과를 반영한 상태 (스크립트용)
    aligned: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reused: bool = False
    info: dict = field(default_factory=dict)
    # 앱은 정렬하는 동안 사용자가 가사를 고쳤을 수 있어, 아래 원본 결과를 지금 가사에 다시 반영한다
    results: list = field(default_factory=list)
    targets: set = field(default_factory=set)
    settings: dict = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LyricsEngine:
    def __init__(self, runner=None) -> None:
        self.runner = runner or run_worker

    def _call(self, command: str, args: dict, progress: ProgressFn, token: CancelToken, req: LyricsRequest) -> dict:
        return self.runner(command, args, progress, token, req.numba_cache_dir)

    def _cached(self, req: LyricsRequest, stage: str, params: dict, compute, check, progress, token):
        cache = AnalysisCache(req.cache_root) if req.cache_root else None
        key = (req.input_id or "novocals")[:32]
        if cache is not None and req.use_cache:
            hit = cache.lookup(key, stage, params)
            if hit is not None:
                try:
                    check(hit)
                    progress(1.0, "캐시 사용")
                    return hit, True
                except Exception as e:  # noqa: BLE001
                    log.warning("%s 캐시 손상, 다시 계산: %s", stage, e)
                    cache.remove(key, stage, params)
        if cache is None:
            tmp = Path(tempfile.mkdtemp(prefix="sonatlas-lyrics-"))
            compute(tmp)
            check(tmp)
            return tmp, False
        if not req.use_cache:
            cache.remove(key, stage, params)
        with cache.writer(key, stage, params) as tmp:
            compute(tmp)
            check(tmp)
        return cache.lookup(key, stage, params), False

    # --- 가사 인식 -----------------------------------------------------------

    def recognize(self, req: LyricsRequest, progress: ProgressFn | None = None,
                  token: CancelToken | None = None) -> tuple[Recognition, bool]:
        progress = progress or (lambda f, m: None)
        token = token or CancelToken()
        params = {"model": req.model, "language": req.language, "device": req.device,
                  "ranges": [list(r) for r in req.no_vocal_ranges], "v": ASR_VERSION}

        def compute(out: Path) -> None:
            self._call("asr", {"input": req.vocals_path, "out_dir": str(out), "model": req.model,
                               "language": req.language, "device": req.device, "download_root": req.whisper_root,
                               "no_vocal_ranges": [list(r) for r in req.no_vocal_ranges]}, progress, token, req)

        def check(d: Path) -> None:
            json.loads((d / "asr.json").read_text(encoding="utf-8"))["segments"]

        d, reused = self._cached(req, "asr", params, compute, check, progress, token)
        raw = json.loads((d / "asr.json").read_text(encoding="utf-8"))
        return recognition_from_raw(raw, req, params), reused

    # --- 정렬 ----------------------------------------------------------------

    def emissions(self, req: LyricsRequest, progress: ProgressFn, token: CancelToken) -> tuple[Path, dict, bool]:
        params = {"model": "mms_fa_uroman", "device": req.device, "v": MMS_VERSION}

        def compute(out: Path) -> None:
            self._call("emissions", {"input": req.vocals_path, "out_dir": str(out), "device": req.device,
                                     "model_dir": req.mms_root}, progress, token, req)

        def check(d: Path) -> None:
            import numpy as np

            with np.load(d / "emissions.npz", allow_pickle=False) as z:
                if z["logp"].ndim != 2 or z["logp"].shape[0] != z["rms_db"].shape[0]:
                    raise ValueError("emissions 크기가 맞지 않습니다")

        d, reused = self._cached(req, "mms", params, compute, check, progress, token)
        info_path = d / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        return d / "emissions.npz", info, reused

    def align(self, req: LyricsRequest, state: LyricsState, *, mode: str = "all", ids: Iterable[str] | None = None,
              duration: float | None = None, progress: ProgressFn | None = None,
              token: CancelToken | None = None) -> AlignOutput:
        progress = progress or (lambda f, m: None)
        token = token or CancelToken()
        units, anchors, targets, soft = plan_units(state, mode, ids, with_soft=True)
        if not targets:
            return AlignOutput(state, warnings=["다시 정렬할 음절이 없습니다."])
        em, info, reused = self.emissions(req, lambda f, m: progress(0.9 * f, m), token)
        out = self._call("ctc_align", {"emissions": str(em), "units": units, "anchors": anchors, "soft_anchors": soft,
                                       "no_vocal_ranges": [list(r) for r in req.no_vocal_ranges]},
                         lambda f, m: progress(0.9 + 0.1 * f, m), token, req)
        model = {"name": info.get("model", "MMS_FA"), "license": info.get("license", ""),
                 "device": info.get("device", ""), "note": info.get("note", ""), "created": _now(),
                 "emissions_reused": reused}
        settings = {"mode": mode, "no_vocal_ranges": [list(r) for r in req.no_vocal_ranges]}
        results = out.get("results") or []
        released = set(out.get("released") or ())
        if released:  # 자리가 없어 함께 다시 정렬한 이웃 음절 (사용자가 고친 음절은 풀지 않는다)
            targets = targets | released
            settings["released"] = len(released)
        new, aligned, failed = apply_results(state, results, targets, model, settings, duration)
        warnings = list(out.get("warnings") or [])
        if info.get("note"):
            warnings.append(info["note"])
        return AlignOutput(new, aligned, failed, warnings, reused, model, results, targets, settings)


# --- 순수 함수 (테스트 대상) -----------------------------------------------------

def recognition_from_raw(raw: dict, req: LyricsRequest | None = None, params: dict | None = None) -> Recognition:
    segs = []
    for i, s in enumerate(raw.get("segments") or ()):
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        words = tuple(AsrWord(str(w.get("text", "")), float(w["start"]), float(w["end"]), w.get("probability"))
                      for w in s.get("words") or ())
        reasons = review.review_segment(text, float(s["start"]), float(s["end"]), avg_logprob=s.get("avg_logprob"),
                                        no_speech_prob=s.get("no_speech_prob"),
                                        compression_ratio=s.get("compression_ratio"), activity=s.get("activity"),
                                        in_no_vocal=float(s.get("in_no_vocal", 0.0)), words=words,
                                        level_db=s.get("level_db"))
        if s.get("second_pass"):
            reasons.append(review.R_SECOND_PASS)
        segs.append(AsrSegment(f"seg{i}", float(s["start"]), float(s["end"]), text, s.get("avg_logprob"),
                               s.get("no_speech_prob"), s.get("compression_ratio"), tuple(reasons), words))
    segs = review.mark_repeats(segs)
    warnings = (str(raw["note"]),) if raw.get("note") else ()
    return Recognition(str(raw.get("model", "")), str(raw.get("model_version", "")), str(raw.get("language", "")),
                       raw.get("language_probability"), str(raw.get("device", "")), _now(), dict(params or {}),
                       req.input_id if req else "", tuple(segs), warnings)


def _letters(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s.lower())


def text_from_recognition(state: LyricsState, rec: Recognition, *, replace_text: bool = True,
                          language: str | None = None) -> LyricsState:
    """인식 결과를 상태에 넣는다. replace_text면 편집 가사를 인식 결과로 새로 만든다(이전 가사는 실행 취소로 되돌릴 수 있다).

    음절 시간은 인식 모델의 단어 시각으로 채운다: 단어 첫 음절 시작·마지막 음절 끝은 'asr_word',
    단어 안의 경계는 글자 수로 나눈 'estimated' 다. 실제 음절 검출값이 아니다.
    """
    state = replace(state, recognition=rec)
    if not replace_text:
        return state.with_history("자동 인식 결과 받음", f"{rec.model}, 구간 {len(rec.segments)}개")
    lines: list[LyricLine] = []
    timings: dict[str, SyllableTiming] = {}
    for seg in rec.segments:
        line = ops.make_line(seg.text, "asr", "; ".join(seg.review), seg.id)
        lines.append(line)
        timings.update(_word_timings(line, seg))
    text = LyricsText(language or rec.language or state.text.language, tuple(lines))
    from amp.lyrics.model import Alignment

    al = Alignment(timings, {}, {"name": f"whisper-{rec.model}", "device": rec.device, "created": rec.created},
                   {"source": "asr_words"})
    out = replace(state, text=text, alignment=al)
    flagged = sum(1 for ln in lines if ln.review)
    return out.with_history("자동 인식 결과로 가사 만들기", f"{len(lines)}줄, 검토 필요 {flagged}줄")


def _word_timings(line: LyricLine, seg: AsrSegment) -> dict[str, SyllableTiming]:
    syls = line.syllables
    words = [w for w in seg.words if _letters(w.text)]
    if not syls or not words:
        return {}
    # 음절 글자열과 인식 단어 글자열을 글자 단위로 맞춰 음절 → 단어를 정한다
    syl_chars = []
    for i, s in enumerate(syls):
        syl_chars.extend((c, i) for c in _letters(s.text))
    word_chars = []
    for j, w in enumerate(words):
        word_chars.extend((c, j) for c in _letters(w.text))
    sm = difflib.SequenceMatcher(None, [c for c, _ in syl_chars], [c for c, _ in word_chars], autojunk=False)
    syl_word: dict[int, int] = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            syl_word.setdefault(syl_chars[a + k][1], word_chars[b + k][1])
    groups: dict[int, list[int]] = {}
    for i in range(len(syls)):
        if i in syl_word:
            groups.setdefault(syl_word[i], []).append(i)
    out: dict[str, SyllableTiming] = {}
    for j, idx in groups.items():
        w = words[j]
        if not (w.end > w.start):
            continue
        step = (w.end - w.start) / len(idx)
        conf = w.probability
        for k, i in enumerate(idx):
            a = w.start + k * step
            b = a + step
            out[syls[i].id] = SyllableTiming(a, b, conf, "asr_word" if k == 0 else "estimated",
                                             "asr_word" if k == len(idx) - 1 else "estimated")
    return out


def plan_units(state: LyricsState, mode: str, ids: Iterable[str] | None = None, with_soft: bool = False):
    """정렬 입력: (units, anchors, 결과를 받을 음절 id 집합[, 풀어도 되는 고정점 순번]).

    all      : 사용자가 고친 시간만 고정하고 나머지를 모두 다시 정렬
    changed  : 시간이 없거나(미정렬), 텍스트가 바뀌었거나(stale), 추정값인 음절만 정렬. 나머지는 고정점
    selected : 고른 음절만 정렬. 고르지 않은 정렬된 음절은 고정점
    """
    al = state.alignment
    chosen = set(ids or ())
    units, anchors, targets, soft = [], [], set(), []
    for i, s in enumerate(state.text.syllables):
        t = al.timing(s.id)
        if mode == "all":
            fixed = t.aligned and t.user_edited and not t.stale
        elif mode == "changed":
            fixed = t.aligned and not t.stale and t.start_source not in ("estimated", "asr_word", "")
        elif mode == "selected":
            fixed = t.aligned and s.id not in chosen
        else:
            raise ValueError(mode)
        units.append({"id": s.id, "reading": s.reading, "line": state.text.index[s.id][0]})
        if fixed:
            anchors.append([i, t.start, t.end])
            if mode == "changed" and not t.user_edited:
                soft.append(i)
        elif mode != "selected" or s.id in chosen or not t.aligned:
            if not (t.user_edited and t.aligned and not t.stale):
                targets.add(s.id)
    return (units, anchors, targets, soft) if with_soft else (units, anchors, targets)


def apply_results(state: LyricsState, results: list[dict], targets: set[str], model: dict, settings: dict,
                  duration: float | None = None) -> tuple[LyricsState, list[str], list[str]]:
    al = state.alignment
    updates: dict[str, SyllableTiming] = {}
    for r in results:
        sid = r.get("id")
        if sid not in targets or sid not in state.text.index:  # 그사이 지운 음절은 건너뛴다
            continue
        try:
            start, end = ops.check_time(r["start"], r["end"], duration)
        except Exception:  # noqa: BLE001 - 잘못된 시각은 받지 않는다 (미정렬로 남김)
            continue
        old = al.timing(sid)
        if old.user_edited and old.aligned and not old.stale:
            continue
        updates[sid] = SyllableTiming(start, end, r.get("confidence"), str(r.get("start_source", "model")),
                                      str(r.get("end_source", "inferred")), False, False)
    failed = [sid for sid in targets if sid not in updates and sid in state.text.index]
    new_al = replace(al.with_timings(updates), model=model, settings=settings)
    out = replace(state, alignment=new_al)
    out = out.with_history("가사 시간 정렬", f"정렬 {len(updates)}음절, 실패 {len(failed)}음절 ({settings.get('mode')})")
    return out, list(updates), failed
