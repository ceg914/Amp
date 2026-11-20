"""분석 오류 자동 검토: 의심 지점 목록. 사용자 데이터는 읽기만 한다(음표 삭제·연결 변경·시간 수정 없음).

규칙마다 무엇을 어떤 기준으로 봤는지 사유에 숫자와 함께 적는다. 근거를 설명할 수 없는 'AI 신뢰도' 숫자는 쓰지 않는다.
보컬 분리 품질을 점수 하나로 판별할 수 있다고 가정하지 않는다: 반주 배음이 보컬보다 큰 음표만 '섞였을 수 있음'으로 알린다.

검토 상태(검토 필요·확인 완료·수정 완료·무시·재검토 필요)는 항목 키로 저장한다. 항목의 근거 값(지문)이 저장 때와 다르면
'재검토 필요'로 바꾼다. 항목이 사라지면(고쳐져서) 상태 기록만 남고 목록에는 나오지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

import numpy as np

SCHEMA = "sonatlas.review"
SCHEMA_VERSION = 1

CATEGORIES = {"note": "음표", "lyrics": "가사", "pron": "발음", "melisma": "멜리스마", "separation": "보컬 분리",
              "model": "모델 비교"}
SEVERITIES = {"high": "높음", "medium": "보통", "low": "낮음"}
STATUSES = {"open": "검토 필요", "checked": "확인 완료", "fixed": "수정 완료", "ignored": "무시", "recheck": "재검토 필요"}
KIND_LABELS = {
    "note_low_conf": "신뢰도 낮은 음표", "note_short": "아주 짧은 음표", "note_leap": "큰 음높이 도약",
    "note_octave": "옥타브 오류 의심", "note_repeat_merged": "반복 발성 병합 의심", "note_oversplit": "긴 음표 과분할 의심",
    "note_portamento": "포르타멘토가 따로 나뉨", "voiced_uncovered": "음표 누락 의심 (유성 구간)",
    "syl_unaligned": "정렬 안 된 음절", "syl_unlinked": "음표 없는 음절", "link_uncertain": "불확실한 연결",
    "onset_gap": "음절·음표 시작 차이", "link_review": "재확인 필요한 연결", "asr_low": "인식 신뢰도 낮은 가사",
    "mel_uncertain": "멜리스마 판정 불확실", "mel_review": "멜리스마 재확인 필요", "pron_unavailable": "발음 변환 불가",
    "pron_stale": "발음 재검토 필요", "sep_novocal_notes": "보컬 없는 구간 안의 음표", "sep_leak": "반주가 섞였을 수 있는 음표",
    # 모델 비교 (amp.compare.model_diff)
    "model_only_a": "A 결과에만 있는 음표", "model_only_b": "B 결과에만 있는 음표", "model_pitch": "모델 간 음높이 차이",
    "model_octave": "모델 간 옥타브 차이 의심", "model_curve": "모델 간 피치 곡선 차이",
    "model_voiced_a": "A에서만 유성인 구간", "model_voiced_b": "B에서만 유성인 구간",
    "model_repeat": "재발음 분할 차이", "model_novocal_a": "보컬 없는 구간의 A 음표", "model_novocal_b": "보컬 없는 구간의 B 음표",
}

# 기준값 (사유에 그대로 적는다)
LOW_CONF = 0.35
VERY_LOW_CONF = 0.2
SHORT_S = 0.08
LEAP = 9
LEAP_GAP_S = 0.15
REPEAT_MIN_S = 0.5
OVERSPLIT_GAP_S = 0.03
PORTA_MAX_S = 0.15
UNCOVERED_S = 0.25
ONSET_GAP_S = 0.15
LINK_CONF = 0.5
LEAK_DB = -6.0


@dataclass(frozen=True)
class ReviewItem:
    key: str
    kind: str
    category: str
    start: float
    end: float
    reason: str
    severity: str
    note_ids: tuple[str, ...] = ()
    syllable_ids: tuple[str, ...] = ()
    fingerprint: str = ""

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)


@dataclass(frozen=True)
class ReviewEntry:
    status: str
    fingerprint: str
    updated: str = ""
    note: str = ""


@dataclass(frozen=True)
class ReviewState:
    entries: Mapping[str, ReviewEntry] = field(default_factory=dict)

    def status_of(self, item: ReviewItem) -> str:
        e = self.entries.get(item.key)
        if e is None:
            return "open"
        if e.status in ("checked", "fixed", "ignored") and e.fingerprint != item.fingerprint:
            return "recheck"  # 근거가 바뀌었다 (재분석·편집 뒤)
        return e.status

    def with_status(self, items: Iterable[ReviewItem], status: str) -> "ReviewState":
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new = dict(self.entries)
        for it in items:
            if status == "open":
                new.pop(it.key, None)
            else:
                new[it.key] = ReviewEntry(status, it.fingerprint, now)
        return ReviewState(new)


def _fp(*vals) -> str:
    return hashlib.sha1(json.dumps(vals, ensure_ascii=False, default=str, sort_keys=True).encode()).hexdigest()[:12]


def _item(kind: str, category: str, start: float, end: float, reason: str, severity: str, note_ids=(), syllable_ids=(),
          basis=()) -> ReviewItem:
    note_ids, syllable_ids = tuple(note_ids), tuple(syllable_ids)
    key = f"{kind}:" + ",".join(sorted(note_ids) + sorted(syllable_ids)) if (note_ids or syllable_ids) \
        else f"{kind}:{round(start, 1)}"
    return ReviewItem(key, kind, category, float(start), float(end), reason, severity, note_ids, syllable_ids,
                      _fp(kind, basis))


def _curve_midi(curve) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.asarray(curve.f0_hz, dtype=float)
    conf = np.asarray(curve.confidence, dtype=float)
    midi = np.full(f0.shape, np.nan)
    ok = np.isfinite(f0) & (f0 > 0)
    midi[ok] = 69 + 12 * np.log2(f0[ok] / 440.0)
    return midi, conf


def note_rules(notes: list, analysis, no_vocal_ranges=(), split_strength: float = 4.0,
               original_ids: frozenset = frozenset()) -> list[ReviewItem]:
    out: list[ReviewItem] = []
    notes = sorted(notes, key=lambda n: n.start)
    curve = analysis.pitch_curve if analysis is not None else None
    artic = analysis.onset_envelope if analysis is not None else None
    midi = conf = None
    if curve is not None:
        midi, conf = _curve_midi(curve)
        fr, t0 = curve.frame_rate, curve.start_time
    for i, n in enumerate(notes):
        d = n.end - n.start
        if n.confidence is not None and n.confidence < LOW_CONF:
            sev = "high" if n.confidence < VERY_LOW_CONF else "medium"
            out.append(_item("note_low_conf", "note", n.start, n.end,
                             f"자동 분석 신뢰도 {n.confidence:.2f} (기준 {LOW_CONF} 미만)", sev, [n.id],
                             basis=(n.pitch, round(n.start, 2), round(n.confidence, 2))))
        if d < SHORT_S and not n.ornament:
            out.append(_item("note_short", "note", n.start, n.end, f"길이 {d * 1000:.0f}ms (기준 {SHORT_S * 1000:.0f}ms 미만)",
                             "low", [n.id], basis=(n.pitch, round(n.start, 2), round(d, 3))))
        if i > 0:
            p = notes[i - 1]
            gap = n.start - p.end
            jump = n.pitch - p.pitch
            if abs(jump) >= LEAP and gap < LEAP_GAP_S:
                # 옥타브 오류 의심: 앞뒤 이웃이 모두 약 12반음 떨어진 외딴 음
                nxt = notes[i + 1] if i + 1 < len(notes) else None
                octave = abs(abs(jump) - 12) <= 1 and nxt is not None and abs(abs(n.pitch - nxt.pitch) - 12) <= 1 \
                    and (n.pitch - p.pitch) * (n.pitch - nxt.pitch) > 0
                if octave:
                    out.append(_item("note_octave", "note", n.start, n.end,
                                     f"앞 음({p.pitch})·뒤 음({nxt.pitch})과 모두 약 한 옥타브 떨어진 외딴 음({n.pitch})",
                                     "high", [n.id], basis=(p.pitch, n.pitch, nxt.pitch)))
                else:
                    out.append(_item("note_leap", "note", p.start, n.end,
                                     f"{abs(jump)}반음 도약 (간격 {gap * 1000:.0f}ms, 기준 {LEAP}반음·{LEAP_GAP_S * 1000:.0f}ms)",
                                     "high" if abs(jump) >= 12 else "medium", [p.id, n.id], basis=(p.pitch, n.pitch, round(gap, 2))))
            if n.pitch == p.pitch and 0 <= gap <= OVERSPLIT_GAP_S and artic is not None and len(artic) and curve is not None:
                k = int(round((n.start - t0) * fr))
                w = artic[max(0, k - 3):k + 4]
                peak = float(np.max(w)) if len(w) else 0.0
                if peak < split_strength * 0.5:
                    out.append(_item("note_oversplit", "note", p.start, n.end,
                                     f"같은 음({n.pitch})이 {gap * 1000:.0f}ms 간격으로 나뉨, 경계의 재발음 지표 {peak:.1f} "
                                     f"(분할 기준 {split_strength:g}의 절반 미만)", "medium", [p.id, n.id],
                                     basis=(n.pitch, round(p.start, 2), round(n.start, 2))))
            if 1 <= abs(jump) <= 4 and 0 <= gap <= OVERSPLIT_GAP_S and (p.end - p.start) <= PORTA_MAX_S and curve is not None:
                i0, i1 = int((p.start - t0) * fr), int((n.start - t0) * fr) + 3
                seg = midi[max(0, i0):i1]
                seg = seg[np.isfinite(seg)]
                if seg.size >= 4 and np.all(np.diff(seg) * np.sign(jump) > -0.15):
                    out.append(_item("note_portamento", "note", p.start, n.end,
                                     f"{(p.end - p.start) * 1000:.0f}ms짜리 음({p.pitch})에서 {n.pitch}로 곡선이 끊김 없이 "
                                     "이어짐 (미끄러지는 소리가 따로 음표가 됐을 수 있음)", "low", [p.id, n.id],
                                     basis=(p.pitch, n.pitch, round(p.start, 2))))
        if d >= REPEAT_MIN_S and artic is not None and len(artic) and curve is not None:
            a, b = int((n.start - t0) * fr) + 5, int((n.end - t0) * fr) - 5
            if b > a:
                w = np.asarray(artic[a:b])
                k = int(np.argmax(w))
                peak = float(w[k])
                if split_strength * 0.5 <= peak < split_strength:
                    t = t0 + (a + k) / fr
                    out.append(_item("note_repeat_merged", "note", n.start, n.end,
                                     f"{t:.2f}초에 재발음 지표 {peak:.1f} (분할 기준 {split_strength:g}에 못 미쳐 나누지 않음) — "
                                     "같은 음을 다시 불렀을 수 있음", "low", [n.id], basis=(n.pitch, round(n.start, 2), round(n.end, 2))))
        for a, b in no_vocal_ranges:
            if n.start < b and n.end > a:
                out.append(_item("sep_novocal_notes", "separation", n.start, n.end,
                                 f"보컬 없는 구간({a:.2f}~{b:.2f}초) 안의 음표"
                                 + (" — 직접 추가·수정한 음표" if n.id not in original_ids else ""), "medium", [n.id],
                                 basis=(n.pitch, round(n.start, 2), a, b)))
                break
    # 음표가 없는 유성 구간
    if curve is not None and midi is not None:
        voiced = np.isfinite(midi) & (conf >= 0.1)  # 곡선은 이미 유성 판정을 거쳤다 (pYIN 확률은 곡마다 낮게 나옴)
        for a, b in no_vocal_ranges:
            voiced[max(0, int((a - t0) * fr)):max(0, int((b - t0) * fr))] = False
        covered = np.zeros_like(voiced)
        for n in notes:
            covered[max(0, int((n.start - t0) * fr)):max(0, int((n.end - t0) * fr))] = True
        gap = voiced & ~covered
        k = 0
        while k < len(gap):
            if gap[k]:
                j = k
                while j < len(gap) and gap[j]:
                    j += 1
                dur = (j - k) / fr
                if dur >= UNCOVERED_S:
                    a, b = t0 + k / fr, t0 + j / fr
                    med = float(np.nanmedian(midi[k:j]))
                    out.append(_item("voiced_uncovered", "note", a, b,
                                     f"{dur:.2f}초 동안 유성 피치(중앙 MIDI {med:.1f})가 있지만 음표가 없음 (기준 {UNCOVERED_S}초)",
                                     "high" if dur >= 0.5 else "medium", basis=(round(a, 1), round(dur, 1))))
                k = j
            else:
                k += 1
    return out


def lyric_rules(state, notes_by_id: Mapping) -> list[ReviewItem]:
    out: list[ReviewItem] = []
    if state.is_empty:
        return out
    al = state.alignment
    for s in state.text.syllables:
        t = al.timing(s.id)
        if not t.aligned:
            out.append(_item("syl_unaligned", "lyrics", 0.0, 0.0, f"'{s.text}' 음절에 시작·끝 시각이 없음", "medium",
                             syllable_ids=[s.id], basis=(s.text,)))
            continue
        links = al.links.get(s.id, ())
        if t.stale:
            out.append(_item("link_review", "lyrics", t.start, t.end, f"'{s.text}' 가사가 바뀌어 다시 정렬해야 함 (이전 시각 사용 중)",
                             "medium", syllable_ids=[s.id], basis=(s.text, round(t.start, 2))))
        if not links:
            out.append(_item("syl_unlinked", "lyrics", t.start, t.end, f"'{s.text}' 음절에 연결된 음표가 없음", "medium",
                             syllable_ids=[s.id], basis=(s.text, round(t.start, 2))))
            continue
        weak = [lk for lk in links if not lk.confirmed and lk.confidence is not None and lk.confidence < LINK_CONF]
        if weak:
            out.append(_item("link_uncertain", "lyrics", t.start, t.end,
                             f"'{s.text}' 연결 신뢰도 {min(lk.confidence for lk in weak):.2f} (기준 {LINK_CONF} 미만, 확정 안 함)",
                             "low", [lk.note_id for lk in weak], [s.id], basis=tuple(sorted(lk.note_id for lk in weak))))
        firsts = [notes_by_id[lk.note_id] for lk in links if lk.note_id in notes_by_id]
        if firsts:
            first = min(firsts, key=lambda n: n.start)
            diff = first.start - t.start
            if abs(diff) > ONSET_GAP_S:
                out.append(_item("onset_gap", "lyrics", min(t.start, first.start), max(t.start, first.start) + 0.05,
                                 f"'{s.text}' 음절 시작과 첫 음표 시작이 {diff * 1000:+.0f}ms 차이 (기준 ±{ONSET_GAP_S * 1000:.0f}ms)",
                                 "low", [first.id], [s.id], basis=(round(t.start, 2), round(first.start, 2))))
    rec = state.recognition
    if rec is not None:
        for seg in rec.segments:
            if seg.review:
                out.append(_item("asr_low", "lyrics", seg.start, seg.end, f"인식 검토 사유: {', '.join(seg.review)} — '{seg.text}'",
                                 "medium", basis=(seg.id, seg.text)))
    return out


def vocal_rules(state, groups: Mapping) -> list[ReviewItem]:
    out: list[ReviewItem] = []
    if state.is_empty:
        return out
    for sid, g in groups.items():
        syl = state.text.syllable(sid) if sid in state.text.index else None
        text = syl.text if syl else sid
        if g.kind == "uncertain" and not g.user_kind and len(g.note_ids) >= 2:
            out.append(_item("mel_uncertain", "melisma", g.start or 0.0, g.end or 0.0,
                             f"'{text}' 음표 {len(g.note_ids)}개: " + ("; ".join(g.reasons[:2]) or "판정 근거 부족"),
                             "low", g.note_ids, [sid], basis=tuple(g.note_ids)))
        if g.needs_review:
            out.append(_item("mel_review", "melisma", g.start or 0.0, g.end or 0.0,
                             f"'{text}' 사용자 멜리스마 판정 뒤 음표가 바뀜 — 다시 확인 필요", "medium", g.note_ids, [sid],
                             basis=tuple(g.note_ids)))
    al = state.alignment
    for sid, e in state.pronunciation.items():
        if sid not in state.text.index:
            continue
        t = al.timing(sid)
        a, b = (t.start, t.end) if t.aligned else (0.0, 0.0)
        text = state.text.syllable(sid).text
        if e.status == "unavailable" and not e.user_edited and text.strip():
            out.append(_item("pron_unavailable", "pron", a, b, f"'{text}' 발음을 만들 수 없음 ({e.language or '언어 모름'})", "low",
                             syllable_ids=[sid], basis=(text, e.language)))
        if e.stale:
            out.append(_item("pron_stale", "pron", a, b, f"'{text}' 원문·문맥이 바뀌어 사용자 발음을 다시 확인해야 함", "medium",
                             syllable_ids=[sid], basis=(text, e.text)))
    return out


def separation_leak_rules(notes: list, vocals: np.ndarray | None, instrumental: np.ndarray | None, sr: int) -> list[ReviewItem]:
    """음표 음높이의 배음 1~8에서 보컬 트랙 에너지 ÷ 반주 트랙 에너지 (dB). 반주가 훨씬 크면 분리 결과에 반주가 섞였을 수 있다.
    분리 품질을 판정하는 점수가 아니라, 직접 들어 볼 곳을 고르는 기준이다."""
    if vocals is None or instrumental is None:
        return []
    out = []
    n_fft = 4096
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    for n in notes:
        a, b = int(n.start * sr), int(n.end * sr)
        if b - a < n_fft // 2:
            continue
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        ev = ei = 0.0
        for s in range(a, max(a + 1, b - n_fft), n_fft // 2):
            v = vocals[s:s + n_fft]
            i = instrumental[s:s + n_fft]
            if len(v) < n_fft:
                break
            fv = np.abs(np.fft.rfft(v * win)) ** 2
            fi = np.abs(np.fft.rfft(i * win)) ** 2
            for h in range(1, 9):
                fh = f0 * h
                if fh > 8000:
                    break
                band = (freqs > fh * 0.97) & (freqs < fh * 1.03)
                ev += fv[band].sum()
                ei += fi[band].sum()
        if ev + ei <= 0:
            continue
        ratio = 10 * np.log10((ev + 1e-12) / (ei + 1e-12))
        if ratio < LEAK_DB:
            out.append(_item("sep_leak", "separation", n.start, n.end,
                             f"이 음의 배음에서 반주 트랙이 보컬 트랙보다 {-ratio:.1f}dB 큼 (기준 {-LEAK_DB:g}dB) — "
                             "반주 소리가 보컬로 잡혔을 수 있음. 들어 보고 판단하세요", "medium", [n.id],
                             basis=(n.pitch, round(n.start, 2), round(float(ratio)))))
    return out


def merge_states(items: list[ReviewItem], state: ReviewState) -> list[tuple[ReviewItem, str]]:
    return [(it, state.status_of(it)) for it in sorted(items, key=lambda x: (x.start, x.kind))]


def state_to_dict(state: ReviewState) -> dict:
    return {"entries": {k: {"status": e.status, "fingerprint": e.fingerprint, "updated": e.updated, "note": e.note}
                        for k, e in sorted(state.entries.items())}}


def state_from_dict(d: dict) -> ReviewState:
    entries = {}
    for k, v in (d.get("entries") or {}).items():
        if isinstance(v, dict) and v.get("status") in STATUSES:
            entries[str(k)] = ReviewEntry(str(v["status"]), str(v.get("fingerprint", "")), str(v.get("updated", "")),
                                          str(v.get("note", "")))
    return ReviewState(entries)


def mark(state: ReviewState, items: Iterable[ReviewItem], status: str) -> ReviewState:
    if status not in STATUSES:
        raise ValueError(status)
    return state.with_status(items, status)


__all__ = ["ReviewItem", "ReviewState", "ReviewEntry", "note_rules", "lyric_rules", "vocal_rules", "separation_leak_rules",
           "merge_states", "state_to_dict", "state_from_dict", "mark", "replace"]
