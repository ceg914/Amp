"""MIDI·MusicXML 악보 읽기 → 성부별 음표 (참조 악보 후보).

- MIDI: mido로 읽는다(MIT). 트랙 × 채널을 성부 하나로 본다. 템포 변화·박자표를 반영해 틱을 초로 바꾼다.
  가사는 lyrics/text 메타 이벤트가 음표 시작과 같은 틱에 있을 때만 붙인다(없으면 지어내지 않는다).
  MIDI에는 이명동음 표기(C#/D♭)가 없으므로 음높이 번호만 쓴다.
- MusicXML(.musicxml/.xml)·압축 MusicXML(.mxl): xml.etree로 직접 읽는다(partwise만). divisions·backup·forward·
  chord·tie·rest·grace·lyric·sound tempo·time을 처리한다. 반복 기호는 펼치지 않고 경고만 남긴다.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from amp.compare.reference import PartInfo, ReferenceNote, ReferenceScore, ScoreAlignment
from amp.core.errors import AnalysisError

MIDI_EXT = (".mid", ".midi")
XML_EXT = (".musicxml", ".xml", ".mxl")
DEFAULT_BPM = 120.0


@dataclass
class ParsedScore:
    kind: str
    parts: list[PartInfo] = field(default_factory=list)
    notes: dict[str, list[ReferenceNote]] = field(default_factory=dict)
    tempo: list[tuple[float, float]] = field(default_factory=list)  # (악보 초, BPM)
    time_signatures: list[tuple[float, int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_score(path: str) -> tuple[ParsedScore, bytes]:
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise AnalysisError(f"악보 파일을 읽지 못했습니다: {e}") from e
    if ext in MIDI_EXT:
        return parse_midi(data), data
    if ext in XML_EXT:
        return parse_musicxml(data, compressed=ext == ".mxl" or data[:2] == b"PK"), data
    raise AnalysisError(f"지원하지 않는 악보 형식입니다: {ext} (MIDI .mid/.midi, MusicXML .musicxml/.xml/.mxl)")


def build_reference(parsed: ParsedScore, part_id: str, source_name: str, sha: str, stored_path: str = "",
                    imported: str = "") -> ReferenceScore:
    if part_id not in parsed.notes:
        raise AnalysisError("고른 성부가 악보에 없습니다.")
    notes = sorted(parsed.notes[part_id], key=lambda n: (n.start, -n.pitch))
    first = notes[0].start if notes else 0.0
    return ReferenceScore(
        source_name=source_name, source_sha256=sha, kind=parsed.kind, stored_path=stored_path,
        parts=tuple(parsed.parts), part_id=part_id, notes=tuple(notes), tempo=tuple(parsed.tempo),
        time_signatures=tuple(parsed.time_signatures), warnings=tuple(parsed.warnings),
        alignment=ScoreAlignment(offset=-first if first < 0 else 0.0), imported=imported)


def guess_vocal_part(parsed: ParsedScore) -> str | None:
    """보컬 성부 후보: 가사가 있는 성부 → 이름에 vocal·voice·보컬·노래·melody → 화음 없고 음역이 사람 목소리인 것."""
    parts = [p for p in parsed.parts if p.notes and not p.drums]
    if not parts:
        return None
    for test in (lambda p: p.has_lyrics,
                 lambda p: any(k in p.name.lower() for k in ("vocal", "voice", "vox", "보컬", "노래", "melody", "멜로디", "sing"))):
        hit = [p for p in parts if test(p)]
        if hit:
            return max(hit, key=lambda p: p.notes).id
    mono = [p for p in parts if not p.polyphonic and 45 <= p.low and p.high <= 88]
    return (max(mono, key=lambda p: p.notes) if mono else max(parts, key=lambda p: p.notes)).id


def _part_info(pid: str, name: str, notes: list[ReferenceNote], drums: bool = False) -> PartInfo:
    poly = False
    last_end: dict[str, float] = {}
    for n in sorted(notes, key=lambda x: x.start):
        if n.start < last_end.get(n.voice, -1.0) - 1e-6:
            poly = True
            break
        last_end[n.voice] = max(last_end.get(n.voice, -1.0), n.end)
    return PartInfo(id=pid, name=name or pid, notes=len(notes), low=min((n.pitch for n in notes), default=0),
                    high=max((n.pitch for n in notes), default=0), has_lyrics=any(n.lyric for n in notes),
                    polyphonic=poly, drums=drums)


# --- 템포: 박 위치 → 초 -----------------------------------------------------------------


class _TempoMap:
    """(박 위치(4분음표), BPM) 목록으로 박 → 초."""

    def __init__(self, changes: list[tuple[float, float]]) -> None:
        pts = sorted(changes) or [(0.0, DEFAULT_BPM)]
        if pts[0][0] > 0:
            pts.insert(0, (0.0, pts[0][1]))
        merged: list[tuple[float, float]] = []
        for b, bpm in pts:
            if merged and abs(merged[-1][0] - b) < 1e-9:
                merged[-1] = (b, bpm)
            else:
                merged.append((b, bpm))
        self.pts = merged
        self.secs = [0.0]
        for (b0, bpm0), (b1, _) in zip(merged[:-1], merged[1:]):
            self.secs.append(self.secs[-1] + (b1 - b0) * 60.0 / bpm0)

    def seconds(self, beat: float) -> float:
        i = max(0, sum(1 for b, _ in self.pts if b <= beat + 1e-9) - 1)
        b0, bpm = self.pts[i]
        return self.secs[i] + (beat - b0) * 60.0 / bpm

    def points_seconds(self) -> list[tuple[float, float]]:
        return [(s, bpm) for s, (_b, bpm) in zip(self.secs, self.pts)]


# --- MIDI ----------------------------------------------------------------------------


def parse_midi(data: bytes) -> ParsedScore:
    import mido

    try:
        mid = mido.MidiFile(file=io.BytesIO(data))
    except Exception as e:  # noqa: BLE001 - 손상 파일
        raise AnalysisError(f"MIDI 파일을 읽지 못했습니다: {e}") from e
    if mid.type == 2:
        raise AnalysisError("MIDI type 2(트랙마다 다른 곡)는 지원하지 않습니다.")
    tpb = mid.ticks_per_beat
    out = ParsedScore(kind="midi")
    tempo_ticks: list[tuple[int, float]] = []
    ts_ticks: list[tuple[int, int, int]] = []
    events = []  # (tick, track, msg)
    for ti, track in enumerate(mid.tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_ticks.append((tick, mido.tempo2bpm(msg.tempo)))
            elif msg.type == "time_signature":
                ts_ticks.append((tick, msg.numerator, msg.denominator))
            events.append((tick, ti, msg))
    tmap = _TempoMap([(t / tpb, bpm) for t, bpm in tempo_ticks])
    if not tempo_ticks:
        out.warnings.append(f"템포 정보가 없어 {DEFAULT_BPM:g} BPM으로 계산했습니다.")
    sec = lambda tick: tmap.seconds(tick / tpb)  # noqa: E731
    out.tempo = tmap.points_seconds()
    out.time_signatures = [(sec(t), n, d) for t, n, d in sorted(ts_ticks)] or [(0.0, 4, 4)]
    names: dict[int, str] = {}
    programs: dict[tuple[int, int], int] = {}
    lyrics: dict[tuple[int, int], str] = {}  # (track, tick) → 가사
    for tick, ti, msg in events:
        if msg.type == "track_name" and ti not in names:
            names[ti] = _meta_text(msg.name).strip()
        elif msg.type == "program_change":
            programs.setdefault((ti, msg.channel), msg.program)
        elif msg.type == "lyrics" or (msg.type == "text" and msg.text and not msg.text.startswith("@")):
            text = _meta_text(msg.text).strip().strip("/\\").strip()
            if text:
                lyrics.setdefault((ti, tick), text)
    open_notes: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    notes: dict[tuple[int, int], list[tuple[int, int, int]]] = {}  # (track, ch) → (on tick, off tick, pitch)
    for tick, ti, msg in sorted(events, key=lambda e: (e[0], 0 if e[2].type == "note_off" or
                                                       (e[2].type == "note_on" and e[2].velocity == 0) else 1)):
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault((ti, msg.channel, msg.note), []).append((tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = open_notes.get((ti, msg.channel, msg.note))
            if stack:
                on, _v = stack.pop(0)
                if tick > on:
                    notes.setdefault((ti, msg.channel), []).append((on, tick, msg.note))
    dangling = sum(len(v) for v in open_notes.values())
    if dangling:
        out.warnings.append(f"끝나지 않은 음 {dangling}개는 버렸습니다.")
    for (ti, ch), lst in sorted(notes.items()):
        pid = f"t{ti}c{ch}"
        refs = []
        for k, (on, off, p) in enumerate(sorted(lst)):
            refs.append(ReferenceNote(id=f"{pid}-{k}", start=sec(on), end=sec(off), pitch=int(p),
                                      lyric=lyrics.get((ti, on)), beat=on / tpb, beats=(off - on) / tpb, voice=str(ch)))
        name = names.get(ti) or f"트랙 {ti + 1}"
        prog = programs.get((ti, ch))
        label = f"{name} · 채널 {ch + 1}" + (f" · 프로그램 {prog}" if prog is not None else "")
        out.notes[pid] = refs
        out.parts.append(_part_info(pid, label, refs, drums=ch == 9))
    if not out.parts:
        raise AnalysisError("MIDI 파일에 음표가 없습니다.")
    return out


def _meta_text(text: str) -> str:
    """mido는 메타 텍스트를 latin-1로 읽는다. 한글 가사는 보통 UTF-8이나 CP949로 저장돼 있어 바이트로 되돌려 다시 읽는다."""
    try:
        raw = text.encode("latin-1")
    except UnicodeEncodeError:
        return text
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return text


# --- MusicXML ------------------------------------------------------------------------

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _mxl_main(data: bytes) -> bytes:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise AnalysisError(f".mxl 압축을 풀지 못했습니다: {e}") from e
    names = z.namelist()
    if "META-INF/container.xml" in names:
        c = _strip_ns(ET.fromstring(z.read("META-INF/container.xml")))
        rf = c.find(".//rootfile")
        if rf is not None and rf.get("full-path") in names:
            return z.read(rf.get("full-path"))
    cand = [n for n in names if n.lower().endswith((".musicxml", ".xml")) and not n.startswith("META-INF")]
    if not cand:
        raise AnalysisError(".mxl 안에서 MusicXML 문서를 찾지 못했습니다.")
    return z.read(cand[0])


def parse_musicxml(data: bytes, compressed: bool = False) -> ParsedScore:
    if compressed:
        data = _mxl_main(data)
    try:
        root = _strip_ns(ET.fromstring(data))
    except ET.ParseError as e:
        raise AnalysisError(f"MusicXML을 읽지 못했습니다: {e}") from e
    if root.tag == "score-timewise":
        raise AnalysisError("timewise MusicXML은 아직 지원하지 않습니다 (partwise로 저장해 주세요).")
    if root.tag != "score-partwise":
        raise AnalysisError("MusicXML 악보가 아닙니다.")
    out = ParsedScore(kind="musicxml")
    names = {sp.get("id"): (sp.findtext("part-name") or sp.get("id") or "").strip() for sp in root.iter("score-part")}
    tempo_beats: list[tuple[float, float]] = []
    ts_beats: list[tuple[float, int, int]] = []
    raw: dict[str, list[dict]] = {}
    repeats = False
    for pi, part in enumerate(root.findall("part")):
        pid = part.get("id") or f"P{pi + 1}"
        divisions = 1.0
        measure_start = 0.0  # 4분음표 단위
        notes: list[dict] = []
        open_ties: dict[tuple[str, int], dict] = {}
        for measure in part.findall("measure"):
            pos = measure_start
            longest = 0.0
            last_note_start = pos
            for el in measure:
                tag = el.tag
                if tag == "attributes":
                    d = el.findtext("divisions")
                    if d:
                        divisions = float(d)
                    t = el.find("time")
                    if t is not None and t.findtext("beats") and pi == 0:
                        try:
                            ts_beats.append((pos, int(t.findtext("beats").split("+")[0]), int(t.findtext("beat-type"))))
                        except (TypeError, ValueError):
                            pass
                elif tag in ("direction", "sound"):
                    snd = el if tag == "sound" else el.find(".//sound")
                    if snd is not None and snd.get("tempo") and pi == 0:
                        try:
                            tempo_beats.append((pos, float(snd.get("tempo"))))
                        except ValueError:
                            pass
                    if tag == "direction" and pi == 0 and (snd is None or not snd.get("tempo")):
                        met = el.find(".//metronome")
                        if met is not None and met.findtext("per-minute") and met.findtext("beat-unit") == "quarter":
                            try:
                                tempo_beats.append((pos, float(met.findtext("per-minute"))))
                            except ValueError:
                                pass
                elif tag == "backup":
                    pos -= float(el.findtext("duration") or 0) / divisions
                elif tag == "forward":
                    pos += float(el.findtext("duration") or 0) / divisions
                    longest = max(longest, pos - measure_start)
                elif tag == "barline":
                    if el.find("repeat") is not None or el.find("ending") is not None:
                        repeats = True
                elif tag == "note":
                    if el.find("grace") is not None:
                        continue  # 꾸밈음은 길이가 없다 (참조 음표로 넣지 않음)
                    dur = float(el.findtext("duration") or 0) / divisions
                    chord = el.find("chord") is not None
                    start = last_note_start if chord else pos
                    if not chord:
                        last_note_start = pos
                        pos += dur
                        longest = max(longest, pos - measure_start)
                    if el.find("rest") is not None or dur <= 0:
                        continue
                    p = el.find("pitch")
                    if p is None:
                        continue  # 타악기 표기 등
                    midi = 12 * (int(p.findtext("octave") or 4) + 1) + _STEP.get((p.findtext("step") or "C").upper(), 0) \
                        + round(float(p.findtext("alter") or 0))
                    voice = el.findtext("voice") or "1"
                    ties = {t.get("type") for t in el.findall("tie")}
                    lyric_el = el.find("lyric")
                    text = None
                    extend = False
                    if lyric_el is not None:
                        text = "".join(x.text or "" for x in lyric_el.findall("text")).strip() or None
                        extend = lyric_el.find("extend") is not None and text is None
                    key = (voice, midi)
                    if "stop" in ties and key in open_ties:
                        prev = open_ties[key]
                        prev["end"] = start + dur
                        prev["tied"] = True
                        if "start" not in ties:
                            open_ties.pop(key)
                        continue
                    n = {"start": start, "end": start + dur, "pitch": midi, "lyric": text, "extend": extend,
                         "voice": voice, "tied": False}
                    notes.append(n)
                    if "start" in ties:
                        open_ties[key] = n
            measure_start += max(longest, pos - measure_start, 0.0)
        raw[pid] = notes
    if repeats:
        out.warnings.append("반복 기호가 있지만 펼치지 않았습니다(악보에 적힌 순서대로 한 번씩만 읽음).")
    tmap = _TempoMap(tempo_beats)
    if not tempo_beats:
        out.warnings.append(f"템포 표시가 없어 {DEFAULT_BPM:g} BPM으로 계산했습니다.")
    out.tempo = tmap.points_seconds()
    out.time_signatures = [(tmap.seconds(b), n, d) for b, n, d in ts_beats] or [(0.0, 4, 4)]
    for pid, notes in raw.items():
        refs = [ReferenceNote(id=f"{pid}-{k}", start=tmap.seconds(n["start"]), end=tmap.seconds(n["end"]), pitch=n["pitch"],
                              lyric=n["lyric"], lyric_extend=n["extend"], beat=n["start"], beats=n["end"] - n["start"],
                              voice=n["voice"], tied=n["tied"])
                for k, n in enumerate(sorted(notes, key=lambda x: (x["start"], -x["pitch"])))]
        out.notes[pid] = refs
        out.parts.append(_part_info(pid, names.get(pid, pid), refs))
    if not any(out.notes.values()):
        raise AnalysisError("MusicXML에 음표가 없습니다.")
    return out
