"""가사 상태 ↔ JSON dict (lyrics.json · alignment.json).

- lyrics.json  : 텍스트 구조·언어·자동 인식 원본·편집 이력 (가사의 주인)
- alignment.json: 음절 id별 시간과 음표 연결 (텍스트는 담지 않는다 — 한 곳에만 두어 어긋나지 않게)

NaN·무한대는 저장하지 않는다(None). 읽을 때 형식이 이상한 항목은 건너뛰고 경고를 남긴다.
"""

from __future__ import annotations

from typing import Any

from amp.lyrics.model import (
    Alignment,
    AsrSegment,
    AsrWord,
    HistoryEntry,
    LyricLine,
    LyricsState,
    LyricsText,
    LyricSyllable,
    LyricWord,
    NoteLink,
    Recognition,
    SyllableTiming,
    finite_or_none,
)

LYRICS_KIND = "sonatlas.lyrics"
ALIGNMENT_KIND = "sonatlas.alignment"


def _num(x, nd: int = 3):
    v = finite_or_none(x)
    return None if v is None else round(v, nd)


def _clean(d: dict) -> dict:
    """JSON에 쓸 수 없는 값(NaN 등)을 걸러낸 사본. settings·model 같은 자유 형식 dict에 쓴다."""
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, float):
            out[str(k)] = finite_or_none(v)
        elif isinstance(v, dict):
            out[str(k)] = _clean(v)
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [finite_or_none(x) if isinstance(x, float) else x for x in v]
        elif v is None or isinstance(v, (str, int, bool)):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


# --- lyrics.json ---------------------------------------------------------------------

def text_to_dict(t: LyricsText) -> dict:
    return {
        "language": t.language,
        "text": t.text,
        "lines": [
            {
                "id": ln.id, "text": ln.text, "origin": ln.origin, "review": ln.review, "asr_segment": ln.asr_segment,
                "words": [
                    {"id": w.id, "text": w.text,
                     "syllables": [{"id": s.id, "text": s.text, "kind": s.kind, "reading": s.reading, "note": s.note}
                                   for s in w.syllables]}
                    for w in ln.words
                ],
            }
            for ln in t.lines
        ],
    }


def text_from_dict(d: dict, warnings: list[str]) -> LyricsText:
    lines = []
    seen: set[str] = set()
    for ld in d.get("lines") or ():
        try:
            words = []
            for wd in ld.get("words") or ():
                syls = []
                for sd in wd.get("syllables") or ():
                    sid = str(sd["id"])
                    if sid in seen:
                        warnings.append(f"lyrics.json에 겹치는 음절 id가 있어 하나를 건너뛰었습니다: {sid}")
                        continue
                    seen.add(sid)
                    syls.append(LyricSyllable(sid, str(sd.get("text", "")), str(sd.get("kind", "")),
                                              str(sd.get("reading", "")), str(sd.get("note", ""))))
                words.append(LyricWord(str(wd.get("id", "")), str(wd.get("text", "")), tuple(syls)))
            lines.append(LyricLine(str(ld["id"]), str(ld.get("text", "")), tuple(words), str(ld.get("origin", "manual")),
                                   str(ld.get("review", "")), ld.get("asr_segment")))
        except (KeyError, TypeError, AttributeError) as e:
            warnings.append(f"lyrics.json의 줄 하나를 읽지 못해 건너뛰었습니다: {e}")
    return LyricsText(str(d.get("language", "")), tuple(lines))


def recognition_to_dict(r: Recognition) -> dict:
    return {
        "model": r.model, "model_version": r.model_version, "language": r.language,
        "language_probability": _num(r.language_probability), "device": r.device, "created": r.created,
        "settings": _clean(r.settings), "input_id": r.input_id, "warnings": list(r.warnings), "text": r.text,
        "segments": [
            {"id": s.id, "start": _num(s.start), "end": _num(s.end), "text": s.text, "avg_logprob": _num(s.avg_logprob),
             "no_speech_prob": _num(s.no_speech_prob), "compression_ratio": _num(s.compression_ratio),
             "review": list(s.review),
             "words": [{"text": w.text, "start": _num(w.start), "end": _num(w.end), "probability": _num(w.probability)}
                       for w in s.words]}
            for s in r.segments
        ],
    }


def recognition_from_dict(d: dict, warnings: list[str]) -> Recognition | None:
    if not d:
        return None
    segs = []
    for sd in d.get("segments") or ():
        try:
            start, end = finite_or_none(sd.get("start")), finite_or_none(sd.get("end"))
            if start is None or end is None:
                warnings.append("자동 인식 구간 중 시간이 없는 구간을 건너뛰었습니다.")
                continue
            words = tuple(AsrWord(str(w.get("text", "")), float(w["start"]), float(w["end"]),
                                  finite_or_none(w.get("probability")))
                          for w in sd.get("words") or () if finite_or_none(w.get("start")) is not None
                          and finite_or_none(w.get("end")) is not None)
            segs.append(AsrSegment(str(sd.get("id", "")), start, end, str(sd.get("text", "")),
                                   finite_or_none(sd.get("avg_logprob")), finite_or_none(sd.get("no_speech_prob")),
                                   finite_or_none(sd.get("compression_ratio")), tuple(str(x) for x in sd.get("review") or ()),
                                   words))
        except (KeyError, TypeError, ValueError) as e:
            warnings.append(f"자동 인식 구간 하나를 읽지 못했습니다: {e}")
    return Recognition(str(d.get("model", "")), str(d.get("model_version", "")), str(d.get("language", "")),
                       finite_or_none(d.get("language_probability")), str(d.get("device", "")), str(d.get("created", "")),
                       dict(d.get("settings") or {}), str(d.get("input_id", "")), tuple(segs),
                       tuple(str(w) for w in d.get("warnings") or ()))


def lyrics_to_dict(state: LyricsState) -> dict:
    body = text_to_dict(state.text)
    body["recognition"] = recognition_to_dict(state.recognition) if state.recognition else None
    body["history"] = [{"time": h.time, "action": h.action, "detail": h.detail} for h in state.history]
    body["stats"] = {"lines": len(state.text.lines), "syllables": len(state.text.syllables)}
    return body


# --- alignment.json -------------------------------------------------------------------

def alignment_to_dict(state: LyricsState) -> dict:
    al = state.alignment
    rows = []
    for s in state.text.syllables:  # 가사 순서대로. 시간도 연결도 없는 음절은 적지 않는다
        t = al.timing(s.id)
        links = al.links.get(s.id, ())
        if not t.aligned and not links and t == SyllableTiming():
            continue
        rows.append({
            "syllable_id": s.id,
            "start": _num(t.start), "end": _num(t.end), "confidence": _num(t.confidence),
            "start_source": t.start_source, "end_source": t.end_source,
            "user_edited": t.user_edited, "stale": t.stale,
            "linked_note_ids": [lk.note_id for lk in links],
            "links": [{"note_id": lk.note_id, "confidence": _num(lk.confidence), "source": lk.source,
                       "confirmed": lk.confirmed} for lk in links],
        })
    total = len(state.text.syllables)
    aligned = total - len(state.unaligned())
    linked = total - len(state.unlinked())
    return {
        "model": _clean(al.model), "settings": _clean(al.settings),
        "summary": {"syllables": total, "aligned": aligned, "unaligned": total - aligned,
                    "linked": linked, "unlinked": total - linked},
        "syllables": rows,
    }


def alignment_from_dict(d: dict, warnings: list[str]) -> Alignment:
    timings: dict[str, SyllableTiming] = {}
    links: dict[str, tuple[NoteLink, ...]] = {}
    for row in d.get("syllables") or ():
        try:
            sid = str(row["syllable_id"])
            start, end = finite_or_none(row.get("start")), finite_or_none(row.get("end"))
            if (row.get("start") is not None and start is None) or (row.get("end") is not None and end is None):
                warnings.append("alignment.json에 숫자가 아닌 시간이 있어 그 음절을 미정렬로 두었습니다.")
                start = end = None
            t = SyllableTiming(start, end, finite_or_none(row.get("confidence")), str(row.get("start_source", "")),
                               str(row.get("end_source", "")), bool(row.get("user_edited", False)),
                               bool(row.get("stale", False)))
            if t != SyllableTiming():
                timings[sid] = t
            lk = tuple(NoteLink(str(x["note_id"]), finite_or_none(x.get("confidence")), str(x.get("source", "auto")),
                                bool(x.get("confirmed", False))) for x in row.get("links") or ())
            if not lk and row.get("linked_note_ids"):
                lk = tuple(NoteLink(str(n)) for n in row["linked_note_ids"])
            if lk:
                links[sid] = lk
        except (KeyError, TypeError, ValueError) as e:
            warnings.append(f"alignment.json의 음절 하나를 읽지 못했습니다: {e}")
    return Alignment(timings, links, dict(d.get("model") or {}), dict(d.get("settings") or {}))


def state_from_dicts(lyrics: dict | None, alignment: dict | None, warnings: list[str]) -> LyricsState:
    if not lyrics:
        return LyricsState()
    text = text_from_dict(lyrics, warnings)
    rec = recognition_from_dict(lyrics.get("recognition") or {}, warnings)
    hist = tuple(HistoryEntry(str(h.get("time", "")), str(h.get("action", "")), str(h.get("detail", "")))
                 for h in lyrics.get("history") or () if isinstance(h, dict))
    al = alignment_from_dict(alignment or {}, warnings)
    return LyricsState(text, al, rec, hist)


# --- STEP 1 형식(notes.json 안의 lyrics + 음표 syllable_id) 옮기기 ---------------------

def from_legacy(lyrics: dict | None, note_links: dict[str, str]) -> LyricsState:
    """STEP 1 형식: {"language", "syllables": [{id, text, phonetic, line, start, end}]} + 음표 id → 음절 id.

    음절 하나를 단어 하나로 옮긴다(이전 형식에는 단어 정보가 없다). 시간·연결은 'legacy' 출처로 남긴다.
    """
    syls = list((lyrics or {}).get("syllables") or ())
    if not syls:
        return LyricsState()
    by_line: dict[int, list[dict]] = {}
    for s in syls:
        by_line.setdefault(int(s.get("line", 0)), []).append(s)
    lines = []
    timings = {}
    for li in sorted(by_line):
        words = []
        for s in by_line[li]:
            sid = str(s["id"])
            words.append(LyricWord(sid + "-w", str(s.get("text", "")),
                                   (LyricSyllable(sid, str(s.get("text", "")), "legacy"),)))
            start, end = finite_or_none(s.get("start")), finite_or_none(s.get("end"))
            if start is not None and end is not None and end > start:
                timings[sid] = SyllableTiming(start, end, None, "legacy", "legacy")
        text = " ".join(w.text for w in words)
        lines.append(LyricLine(f"legacy-line-{li}", text, tuple(words), "legacy"))
    links: dict[str, list[NoteLink]] = {}
    known = {str(s["id"]) for s in syls}
    for nid, sid in note_links.items():
        if sid in known:
            links.setdefault(sid, []).append(NoteLink(nid, None, "legacy", False))
    al = Alignment(timings, {k: tuple(v) for k, v in links.items()})
    state = LyricsState(LyricsText(str((lyrics or {}).get("language", "ko")), tuple(lines)), al)
    return state.with_history("이전 형식에서 옮김", f"음절 {len(syls)}개, 연결 {sum(len(v) for v in links.values())}개")
