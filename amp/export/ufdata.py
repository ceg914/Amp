"""UtaFormatix 데이터(.ufdata) 내보내기. 보컬 합성 프로그램 사이에서 음표·가사·템포를 옮기는 공용 JSON 형식.

형식은 github.com/sdercolin/utaformatix-data (README와 Kotlin·TypeScript 모델 소스)를 읽고 맞췄다.
- 최상위 {formatVersion: 1, project}. project = {name, tracks, timeSignatures, tempos, measurePrefix}.
- track = {name, notes, pitch}. note = {key(반음, C4=60), tickOn, tickOff, lyric, phoneme}. 4분음표 = 480틱.
- tempo = {tickPosition, bpm(소수)}. timeSignature = {measurePosition(마디), numerator, denominator}.
- pitch = {ticks, values, isAbsolute}. values는 반음 단위(센트 아님). 절대값이면 null이 '앞 값 끝'(무성).
  값이 바뀌는 점만 적는다.

JSON(UTF-8)이라 MIDI 가사 메타처럼 인코딩 문제가 없다. 음원 0초 = 0틱이므로 원래 음원을 0에 두면 시각이 맞는다.
같은 음절을 이어 부르는 음표의 가사는 받는 프로그램마다 표기가 다르다(옵션 extend_lyric, 기본 '-').
보이스뱅크·가수 정보는 형식에 없다. 발음 기호(phoneme)는 넣지 않는다(받는 프로그램이 가사에서 만든다).
"""

from __future__ import annotations

import json

import numpy as np

from amp.export.common import (
    ExportOptions,
    ExportReport,
    TickMap,
    first_note_info,
    note_range,
    pick_notes,
    range_text,
    tempo_points,
)

FORMAT_VERSION = 1
MIN_CONF = 0.05
PITCH_STEP_S = 0.02  # 피치 곡선을 이 간격으로 적는다 (값이 같으면 건너뜀)
PITCH_ROUND = 0.01  # 반음


def _pitch(curve, tm: TickMap, offset: float, span: tuple[float, float] | None) -> dict | None:
    if curve is None:
        return None
    fr = curve.frame_rate
    step = max(1, int(round(PITCH_STEP_S * fr)))
    f0 = np.asarray(curve.f0_hz, dtype=float)
    conf = np.asarray(curve.confidence, dtype=float)
    ticks: list[int] = []
    values: list[float | None] = []
    last: float | None | str = "start"
    for i in range(0, len(f0), step):
        t = curve.start_time + i / fr
        if span is not None and not (span[0] <= t < span[1]):
            continue
        ok = np.isfinite(f0[i]) and f0[i] > 0 and conf[i] >= MIN_CONF
        v = round((69 + 12 * np.log2(f0[i] / 440.0) + offset) / PITCH_ROUND) * PITCH_ROUND if ok else None
        v = None if v is None else round(float(v), 2)
        if v == last:
            continue
        tk = tm.tick(t)
        if ticks and tk <= ticks[-1]:
            values[-1] = v
        else:
            ticks.append(tk)
            values.append(v)
        last = v
    return {"ticks": ticks, "values": values, "isAbsolute": True}


def build(project, options: ExportOptions, name: str = "SONATLAS") -> tuple[dict, ExportReport]:
    notes, warnings = pick_notes(project, options)
    rep = ExportReport("UFData", warnings=list(warnings), source=options.source, range=options.range)
    pts, bpb, unit = tempo_points(project)
    tm = TickMap(pts, unit)
    out = []
    prev = None
    for n in sorted(notes, key=lambda x: (x.start, x.pitch)):
        on, off = tm.tick(n.start), tm.tick(n.end)
        if off <= on:
            off = on + 1
        if prev is not None and on < prev["tickOff"]:  # 한 트랙은 단성부: 겹치면 앞 음표를 줄인다
            prev["tickOff"] = max(prev["tickOn"] + 1, on)
        if options.lyrics and n.lyric:
            lyric = n.lyric
            rep.lyrics += 1
        elif options.lyrics and n.extend:
            lyric = options.extend_lyric
            rep.extended += 1
        else:
            lyric = ""
        note = {"key": int(n.pitch), "tickOn": int(on), "tickOff": int(off), "lyric": lyric, "phoneme": None}
        out.append(note)
        prev = note
    curve = project.analysis.pitch_curve if project.analysis is not None else None
    pitch = None
    if options.pitch != "none" and options.source != "reference":
        pitch = _pitch(curve, tm, float(options.transpose), options.range)
    track = {"name": "Vocal", "notes": out, "pitch": pitch}
    doc = {
        "formatVersion": FORMAT_VERSION,
        "project": {
            "name": name,
            "tracks": [track],
            "timeSignatures": [{"measurePosition": 0, "numerator": int(bpb), "denominator": int(unit)}],
            "tempos": [{"tickPosition": int(tk), "bpm": round(float(b), 6)} for tk, b in tm.tempo_ticks()],
            "measurePrefix": 0,
        },
    }
    rep.notes = len(out)
    rep.info += [f"음역: {range_text(note_range((n.pitch, n.end - n.start) for n in notes))}"]
    if (fi := first_note_info(notes, tm, bpb, unit)) is not None:
        rep.info.append(fi)
    if rep.extended:
        shown = repr(options.extend_lyric) if options.extend_lyric else "빈칸"
        rep.info.append(f"같은 음절을 이어 부르는 음표 {rep.extended}개의 가사: {shown}")
    if options.lyrics and not any(n.lyric for n in notes):
        rep.warnings.append("연결된 가사가 없어 가사를 비워 두었습니다(받는 프로그램이 기본 가사를 넣을 수 있음).")
    rep.omitted += ["가수(보이스뱅크) 지정", "음표 신뢰도·수정 여부·발음 기호·멜리스마 판정 근거"]
    if pitch is None:
        rep.omitted.append("실제 피치 곡선 (옵션: 음표와 가사만)")
    else:
        rep.info.append(f"피치 곡선: 절대 음높이(반음) {len(pitch['ticks'])}점, {PITCH_STEP_S * 1000:.0f}ms 간격, 무성 구간은 null")
    if options.source == "reference":
        rep.warnings.append("참조 악보 음표를 음원 시각(정렬 결과)으로 내보냈습니다.")
    return doc, rep


def dumps(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def export(project, path: str, options: ExportOptions, name: str = "SONATLAS") -> ExportReport:
    from amp.core.fileio import atomic_write_bytes

    doc, rep = build(project, options, name)
    atomic_write_bytes(path, dumps(doc).encode("utf-8"))
    rep.path = path
    return rep
