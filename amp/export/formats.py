"""MIDI·MusicXML·JSON·CSV·LRC 내보내기.

형식마다 담을 수 있는 정보가 다르다. 담지 못한 정보는 ExportReport.omitted에 적고, 모든 정보는 JSON으로 함께 저장할 수 있다.
- MIDI(mido, SMF type 1): 음표·음높이·시각(틱)·템포 변화·박자표·트랙 이름, 가사는 lyrics 메타 이벤트(UTF-8).
  피치 곡선을 넣으면 피치 벤드(범위 ±12반음, RPN 0)로 넣는다. 음표 신뢰도·발음·멜리스마 판정은 담지 못한다.
- MusicXML 3.1 partwise: 16분음표 격자로 맞춰(양자화) 마디·붙임줄·쉼표로 적는다. 가사는 <lyric>, 멜리스마는 extend·이음줄.
  실제 시각·피치 곡선은 담지 못하고, 16분음표보다 짧은 차이는 잃는다. 이명동음은 올림표(♯)로만 적는다.
- JSON: 피치·음표(상태·신뢰도)·가사·발음·멜리스마·분석 정보 전부.
- CSV: 음표별·음절별 표 두 개 (Windows Excel이 한글을 읽도록 UTF-8 BOM).
- LRC: 가사 줄 시각(선택: 음절 시각 확장 표기 <mm:ss.xx>).
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import numpy as np

from amp.core.fileio import atomic_write_bytes
from amp.export.common import TICKS_PER_BEAT, ExportOptions, ExportReport, TickMap, pick_notes, tempo_points

SHARP_NAMES = (("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0))
DIV = 4  # MusicXML divisions: 4분음표 = 4 (16분음표 격자)
NOTE_TYPES = [(16, "whole", 0), (12, "half", 1), (8, "half", 0), (6, "quarter", 1), (4, "quarter", 0), (3, "eighth", 1),
              (2, "eighth", 0), (1, "16th", 0)]
BEND_RANGE = 12


# --- MIDI ----------------------------------------------------------------------------


def export_midi(project, path: str, options: ExportOptions) -> ExportReport:
    import mido

    notes, warnings = pick_notes(project, options)
    rep = ExportReport("MIDI", notes=len(notes), warnings=list(warnings), source=options.source, range=options.range)
    pts, bpb, unit = tempo_points(project)
    tm = TickMap(pts, unit)
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    ev = [(0, mido.MetaMessage("track_name", name="SONATLAS"))]
    ev += [(tk, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(b))) for tk, b in tm.tempo_ticks()]
    ev.append((0, mido.MetaMessage("time_signature", numerator=bpb, denominator=unit)))
    _write_track(meta, ev)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    ev = [(0, mido.MetaMessage("track_name", name="Vocal"))]
    curve = project.analysis.pitch_curve if project.analysis is not None else None
    bend = options.pitch != "none" and curve is not None and options.source != "reference"
    if bend:  # 피치 벤드 범위 ±12반음 (RPN 0)
        for cc, val in ((101, 0), (100, 0), (6, BEND_RANGE), (38, 0), (101, 127), (100, 127)):
            ev.append((0, mido.Message("control_change", control=cc, value=val)))
    for n in notes:
        on, off = tm.tick(n.start), max(tm.tick(n.end), tm.tick(n.start) + 1)
        if options.lyrics and n.lyric:
            # mido는 메타 텍스트를 latin-1로 쓴다: UTF-8 바이트를 그대로 넣는다 (가져오기도 UTF-8로 읽음)
            ev.append((on, mido.MetaMessage("lyrics", text=n.lyric.encode("utf-8").decode("latin-1"))))
            rep.lyrics += 1
        ev.append((on, mido.Message("note_on", note=int(n.pitch), velocity=90)))
        ev.append((off, mido.Message("note_off", note=int(n.pitch), velocity=0)))
        if bend:
            ev += _bends(n, curve, tm)
    _write_track(tr, ev)
    buf = io.BytesIO()
    mid.save(file=buf)
    atomic_write_bytes(path, buf.getvalue())
    rep.path = path
    rep.omitted += ["음표 신뢰도·수정 여부", "발음 표기·IPA", "멜리스마 판정 (같은 음절 음표는 가사 없이 이어짐)"]
    if not bend:
        rep.omitted.append("실제 피치 곡선")
    else:
        rep.warnings.append(f"피치 곡선을 피치 벤드(±{BEND_RANGE}반음, 50ms 간격)로 넣었습니다. 재생기에 따라 벤드 범위 설정(RPN)을 무시할 수 있습니다.")
    return rep


def _bends(n, curve, tm) -> list:
    import mido

    out = []
    fr = curve.frame_rate
    step = max(1, int(round(0.05 * fr)))
    i0 = max(0, int((n.start - curve.start_time) * fr))
    i1 = min(len(curve.f0_hz), int((n.end - curve.start_time) * fr))
    last = None
    for i in range(i0, i1, step):
        f = curve.f0_hz[i]
        if not np.isfinite(f) or f <= 0:
            continue
        semi = 69 + 12 * math.log2(f / 440.0) - n.pitch
        val = int(round(max(-1.0, min(1.0, semi / BEND_RANGE)) * 8191))
        if val != last:
            out.append((tm.tick(curve.start_time + i / fr), mido.Message("pitchwheel", pitch=val)))
            last = val
    out.append((max(tm.tick(n.end), tm.tick(n.start) + 1), mido.Message("pitchwheel", pitch=0)))
    return out


def _write_track(track, events) -> None:
    order = {"note_off": 0, "pitchwheel": 1, "lyrics": 2, "note_on": 3}
    last = 0
    for t, msg in sorted(events, key=lambda e: (e[0], order.get(e[1].type, 1))):
        track.append(msg.copy(time=max(0, t - last)))
        last = max(last, t)


# --- MusicXML ------------------------------------------------------------------------


def _split_value(d: int) -> list[int]:
    out = []
    while d > 0:
        for v, _t, _dot in NOTE_TYPES:
            if v <= d:
                out.append(v)
                d -= v
                break
    return out


def export_musicxml(project, path: str, options: ExportOptions, title: str = "SONATLAS") -> ExportReport:
    notes, warnings = pick_notes(project, options)
    rep = ExportReport("MusicXML", notes=len(notes), warnings=list(warnings), source=options.source, range=options.range)
    tm = project.score.tempo
    bpb, unit = int(tm.beats_per_bar), int(getattr(tm, "beat_unit", 4) or 4)
    bar_div = bpb * DIV * 4 // unit
    q = []  # (시작 div, 끝 div, 음높이, 가사, 이어짐)
    for n in sorted(notes, key=lambda x: (x.start, x.pitch)):
        a = int(round(tm.beat_at(n.start) * DIV))
        b = int(round(tm.beat_at(n.end) * DIV))
        if b <= a:
            b = a + 1
            rep.warnings.append(f"{n.start:.2f}초 음표가 16분음표보다 짧아 16분음표로 늘렸습니다.")
        q.append([a, b, n.pitch, n.lyric if options.lyrics else None, n.extend and options.lyrics])
    # 단성부: 겹치면 앞 음표를 줄이고, 같은 자리에서 시작하면 뒤 음표를 버린다
    mono = []
    for item in q:
        if mono and item[0] <= mono[-1][0]:
            rep.warnings.append("16분음표 격자에서 같은 자리에 시작하는 음표가 있어 하나만 남겼습니다.")
            continue
        if mono and item[0] < mono[-1][1]:
            mono[-1][1] = item[0]
        mono.append(item)
    start_div = min((x[0] for x in mono), default=0)
    first_bar = math.floor(start_div / bar_div)
    end_div = max((x[1] for x in mono), default=bar_div)
    n_bars = max(1, math.ceil(end_div / bar_div) - first_bar)
    melisma_open = False
    bars: list[list[str]] = [[] for _ in range(n_bars)]

    def add_rest(bar: int, dur: int) -> None:
        for v in _split_value(dur):
            t, dot = next((t, d) for vv, t, d in NOTE_TYPES if vv == v)
            bars[bar].append(f"<note><rest/><duration>{v}</duration><voice>1</voice><type>{t}</type>{'<dot/>' * dot}</note>")

    cursor = first_bar * bar_div
    tempo_marks = {int(round(tm.beat_at(p.time) * DIV)): p.bpm for p in tm.points}
    for i, (a, b, pitch, lyric, ext) in enumerate(mono):
        nxt_ext = i + 1 < len(mono) and mono[i + 1][4]
        while cursor < a:  # 쉼표 (마디 경계에서 나눔)
            bar = cursor // bar_div - first_bar
            take = min(a, (cursor // bar_div + 1) * bar_div) - cursor
            add_rest(bar, take)
            cursor += take
        pos = a
        pieces = []
        while pos < b:
            bar_end = (pos // bar_div + 1) * bar_div
            take = min(b, bar_end) - pos
            for v in _split_value(take):
                pieces.append((pos // bar_div - first_bar, v))
                pos += v
        step, alter = SHARP_NAMES[pitch % 12]
        octave = pitch // 12 - 1
        for k, (bar, v) in enumerate(pieces):
            t, dot = next((t, d) for vv, t, d in NOTE_TYPES if vv == v)
            xml = [f"<note><pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "")
                   + f"<octave>{octave}</octave></pitch><duration>{v}</duration>"]
            if len(pieces) > 1:
                if k > 0:
                    xml.append('<tie type="stop"/>')
                if k < len(pieces) - 1:
                    xml.append('<tie type="start"/>')
            xml.append(f"<voice>1</voice><type>{t}</type>{'<dot/>' * dot}")
            notations = []
            if len(pieces) > 1:
                if k > 0:
                    notations.append('<tied type="stop"/>')
                if k < len(pieces) - 1:
                    notations.append('<tied type="start"/>')
            if k == 0 and nxt_ext and not ext and not melisma_open:
                notations.append('<slur type="start" number="1"/>')
                melisma_open = True
            if k == len(pieces) - 1 and ext and melisma_open and not nxt_ext:
                notations.append('<slur type="stop" number="1"/>')
                melisma_open = False
            if notations:
                xml.append("<notations>" + "".join(notations) + "</notations>")
            if k == 0 and lyric:
                xml.append(f'<lyric number="1"><syllabic>single</syllabic><text>{escape(lyric)}</text>'
                           + ('<extend type="start"/>' if nxt_ext else "") + "</lyric>")
                rep.lyrics += 1
            elif k == 0 and ext:
                xml.append('<lyric number="1"><extend type="continue"/></lyric>')
                rep.extended += 1
            xml.append("</note>")
            if k == 0 and a in tempo_marks:
                bars[bar].append(f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                                 f'<per-minute>{tempo_marks[a]:.1f}</per-minute></metronome></direction-type>'
                                 f'<sound tempo="{tempo_marks[a]:.2f}"/></direction>')
            bars[bar].append("".join(xml))
        cursor = b
    while cursor < (first_bar + n_bars) * bar_div:
        bar = cursor // bar_div - first_bar
        take = (cursor // bar_div + 1) * bar_div - cursor
        add_rest(bar, take)
        cursor += take
    first_bpm = tm.points[0].bpm if tm.points else 120.0
    parts = []
    for i, content in enumerate(bars):
        attrs = ""
        if i == 0:
            attrs = (f"<attributes><divisions>{DIV}</divisions><key><fifths>0</fifths></key>"
                     f"<time><beats>{bpb}</beats><beat-type>{unit}</beat-type></time><clef><sign>G</sign><line>2</line></clef>"
                     f"</attributes><direction placement=\"above\"><direction-type><metronome><beat-unit>quarter</beat-unit>"
                     f"<per-minute>{first_bpm:.1f}</per-minute></metronome></direction-type><sound tempo=\"{first_bpm:.2f}\"/>"
                     f"</direction>")
        parts.append(f'<measure number="{i + 1}">{attrs}{"".join(content)}</measure>')
    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">\n'
           f'<score-partwise version="3.1"><work><work-title>{escape(title)}</work-title></work>'
           '<identification><encoding><software>SONATLAS</software>'
           f'<encoding-date>{datetime.now().date().isoformat()}</encoding-date></encoding></identification>'
           '<part-list><score-part id="P1"><part-name>Vocal</part-name>'
           '<score-instrument id="P1-I1"><instrument-name>Voice</instrument-name></score-instrument>'
           '<midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>54</midi-program></midi-instrument>'
           '</score-part></part-list>'
           f'<part id="P1">{"".join(parts)}</part></score-partwise>\n')
    atomic_write_bytes(path, xml.encode("utf-8"))
    rep.path = path
    rep.omitted += ["실제 시각 (16분음표 격자로 맞춤)", "피치 곡선", "음표 신뢰도·수정 여부", "발음 표기·IPA",
                    "이명동음 표기 (모두 ♯로 적음)"]
    if first_bar < 0 or start_div % bar_div:
        rep.warnings.append("첫 음이 첫 박 앞이거나 마디 중간이라 앞을 쉼표로 채웠습니다(못갖춘마디 표기 아님).")
    rep.warnings = list(dict.fromkeys(rep.warnings))
    return rep


# --- JSON·CSV·LRC ---------------------------------------------------------------------


def _timing(ly, sid):
    t = ly.alignment.timing(sid)
    return (t.start, t.end) if t.aligned else (None, None)


def export_json(project, path: str, options: ExportOptions, extra: dict | None = None) -> ExportReport:
    from amp import __version__

    notes, warnings = pick_notes(project, options)
    rep = ExportReport("JSON", notes=len(notes), warnings=list(warnings), source=options.source, range=options.range)
    an = project.analysis
    ly = project.lyrics
    doc = {
        "schema": "sonatlas.export", "schema_version": 1, "app_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "options": {"source": options.source, "range": options.range, "lyrics": options.lyrics, "pitch": options.pitch},
        "audio": {"duration": an.duration if an else None, "fingerprint": an.audio_fingerprint if an else None},
        "tempo": [{"time": p.time, "bpm": p.bpm} for p in project.score.tempo.points],
        "beats_per_bar": project.score.tempo.beats_per_bar,
        "notes": [{"id": n.id, "start": round(n.start, 4), "end": round(n.end, 4), "pitch": n.pitch, "lyric": n.lyric,
                   "extend": n.extend, "syllable_ids": list(n.syllable_ids), "status": n.status, "confidence": n.confidence}
                  for n in notes],
        "analysis": None if an is None else {"created_at": an.created_at, "settings": an.settings, "warnings": list(an.warnings),
                                             "models": {k: {"name": m.name, "version": m.version} for k, m in an.models.items()},
                                             "auto_notes": len(an.notes)},
    }
    if options.pitch != "none" and an is not None and an.pitch_curve is not None:
        c = an.pitch_curve
        doc["pitch_curve"] = {"start_time": c.start_time, "frame_rate": c.frame_rate,
                              "f0_hz": [None if not np.isfinite(v) else round(float(v), 3) for v in c.f0_hz],
                              "confidence": [round(float(v), 3) for v in c.confidence]}
    if not ly.is_empty and options.lyrics:
        sylls = []
        for line in ly.text.lines:
            for w in line.words:
                for s in w.syllables:
                    a, b = _timing(ly, s.id)
                    e = ly.pronunciation.get(s.id)
                    sylls.append({"id": s.id, "text": s.text, "line": line.id, "start": a, "end": b,
                                  "notes": list(ly.alignment.linked_notes(s.id)),
                                  "pronunciation": None if e is None else {"text": e.text, "ipa": e.ipa, "status": e.status,
                                                                           "source": e.source, "user_edited": e.user_edited}})
        doc["lyrics"] = {"text": ly.text.text, "syllables": sylls,
                         "pronunciation_note": "가사 텍스트에서 규칙으로 추정한 발음 (가수의 실제 발음 인식이 아님)"}
    if extra:
        doc.update(extra)
    atomic_write_bytes(path, json.dumps(doc, ensure_ascii=False, indent=1, default=_json_default).encode("utf-8"))
    rep.path = path
    return rep


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def export_csv(project, path: str, options: ExportOptions, durations: dict | None = None, groups: dict | None = None) -> ExportReport:
    """path = 음표 표. 음절 표는 같은 이름에 '-syllables'를 붙인다."""
    notes, warnings = pick_notes(project, options)
    rep = ExportReport("CSV", notes=len(notes), warnings=list(warnings), source=options.source, range=options.range)
    enc = "utf-8-sig" if options.csv_bom else "utf-8"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "start_s", "end_s", "duration_s", "pitch_midi", "lyric", "extend", "status", "confidence"])
    for n in notes:
        w.writerow([n.id, f"{n.start:.4f}", f"{n.end:.4f}", f"{n.end - n.start:.4f}", n.pitch, n.lyric or "",
                    int(n.extend), n.status, "" if n.confidence is None else f"{n.confidence:.3f}"])
    atomic_write_bytes(path, buf.getvalue().encode(enc))
    ly = project.lyrics
    if not ly.is_empty and options.lyrics:
        root, ext = os.path.splitext(path)
        spath = f"{root}-syllables{ext or '.csv'}"
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["syllable_id", "text", "start_s", "end_s", "span_s", "voiced_s", "pronunciation", "ipa",
                    "pron_status", "notes", "melisma"])
        for s in ly.text.syllables:
            a, b = _timing(ly, s.id)
            if options.range is not None and (a is None or not (options.range[0] <= a < options.range[1])):
                continue
            e = ly.pronunciation.get(s.id)
            d = (durations or {}).get(s.id)
            g = (groups or {}).get(s.id)
            w.writerow([s.id, s.text, "" if a is None else f"{a:.4f}", "" if b is None else f"{b:.4f}",
                        "" if d is None or d.span is None else f"{d.span:.4f}",
                        "" if d is None or d.voiced is None else f"{d.voiced:.4f}",
                        "" if e is None else e.text, "" if e is None else e.ipa, "" if e is None else e.status,
                        " ".join(ly.alignment.linked_notes(s.id)),
                        "" if g is None else ("melisma" if g.is_melisma else g.kind)])
            rep.lyrics += 1
        atomic_write_bytes(spath, buf.getvalue().encode(enc))
        rep.warnings.append(f"음절 표: {os.path.basename(spath)}")
    rep.path = path
    return rep


def _lrc_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def export_lrc(project, path: str, options: ExportOptions, title: str = "") -> ExportReport:
    ly = project.lyrics
    rep = ExportReport("LRC", source=options.source, range=options.range)
    if ly.is_empty:
        raise ValueError("가사가 없습니다.")
    lines = []
    if title:
        lines.append(f"[ti:{title}]")
    lines.append("[re:SONATLAS]")
    skipped = 0
    for line in ly.text.lines:
        sylls = [s for w in line.words for s in w.syllables]
        times = [_timing(ly, s.id) for s in sylls]
        aligned = [(s, a) for s, (a, _b) in zip(sylls, times) if a is not None]
        if not aligned:
            skipped += 1
            continue
        start = aligned[0][1]
        if options.range is not None and not (options.range[0] <= start < options.range[1]):
            continue
        if options.lrc_words:
            text = ""
            for w in line.words:
                for s in w.syllables:
                    a, _b = _timing(ly, s.id)
                    text += (f"<{_lrc_time(a)}>" if a is not None else "") + s.text
                text += " "
            lines.append(f"[{_lrc_time(start)}]{text.strip()}")
        else:
            lines.append(f"[{_lrc_time(start)}]{line.text}")
        rep.lyrics += 1
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    rep.path = path
    if skipped:
        rep.warnings.append(f"시각이 없는 가사 줄 {skipped}개는 넣지 않았습니다.")
    rep.omitted += ["음표·음높이", "줄 끝 시각 (LRC는 줄 시작만)"]
    return rep
