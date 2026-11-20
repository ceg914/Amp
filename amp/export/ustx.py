"""OpenUtau USTX 내보내기.

형식은 OpenUtau 소스(github.com/stakira/OpenUtau, MIT, master)의 OpenUtau.Core/Ustx/*.cs와 Format/USTx.cs를 읽고 맞췄다.
- YAML, 이름 규칙 underscored(ustx_version, time_signatures, voice_parts …), 4분음표 = 480틱.
- ustx_version은 USTX_VERSION("0.6")으로 쓴다. 0.6부터 템포·박자표가 목록이라 템포 변화를 담을 수 있고, OpenUtau는
  열 때 더 낮은 버전 파일을 현재 버전으로 올린다(Ustx.Load). 소프트웨어보다 새 버전 파일은 열지 않으므로 낮게 잡았다.
- 음표 위치는 파트 시작 기준 틱. 파트는 0틱에서 시작(음원 0초 = 0틱)하므로 원본 음원을 0에 두면 시각이 맞는다.
- 가사: 음절이 시작하는 음표에 음절, 같은 음절의 다음 음표는 '+~'(앞 음절 이어 부르기). OpenUtau는 앞 음표 끝과
  이 음표 시작이 같을 때만 이어 부르기로 본다(UPart.cs). 그래서 같은 음절 음표 사이 빈틈은 앞 음표를 늘여 메운다(GAP_CLOSE_S까지).
  더 큰 빈틈은 이어 부르기가 안 되므로 음절을 다시 적고 경고한다.
- 피치: 음표마다 피치점(x = 음표 시작부터 ms, y = 음표 높이 기준 0.1반음). 'none'은 OpenUtau 새 음표 기본값과 같은
  두 점(−40ms·40ms, 0). simple/detailed는 실제 피치 곡선을 간단히 만든 점. 불확실한 프레임은 뺀다. 원본 피치 곡선은 바꾸지 않는다.
- 가수(singer)·phonemizer는 적지 않는다(비우면 OpenUtau 기본값). 보이스뱅크 이름을 지어 넣지 않는다.
"""

from __future__ import annotations

import numpy as np
import yaml

from amp.export.common import EXTEND, ExportNote, ExportOptions, ExportReport, TickMap, pick_notes, tempo_points

USTX_VERSION = "0.6"
GAP_CLOSE_S = 0.5
DEFAULT_LYRIC = "a"  # OpenUtau 새 음표 기본 가사 (NotePresets.DefaultLyric)
PITCH_MODES = {"none": "음표와 가사만", "simple": "음표·가사·단순화한 피치", "detailed": "음표·가사·상세 피치"}
_PITCH_PARAMS = {"simple": (0.25, 8, 0.04), "detailed": (0.06, 48, 0.01)}  # (허용 오차 반음, 음표당 최대 점, 이동 중앙값 창 초)
MIN_CONF = 0.05


def _rdp(xs: np.ndarray, ys: np.ndarray, eps: float) -> list[int]:
    """Ramer–Douglas–Peucker: 남길 점 번호."""
    keep = {0, len(xs) - 1}
    stack = [(0, len(xs) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        x0, y0, x1, y1 = xs[a], ys[a], xs[b], ys[b]
        seg = ys[a + 1:b] - (y0 + (y1 - y0) * (xs[a + 1:b] - x0) / max(x1 - x0, 1e-9))
        k = int(np.argmax(np.abs(seg)))
        if abs(seg[k]) > eps:
            m = a + 1 + k
            keep.add(m)
            stack += [(a, m), (m, b)]
    return sorted(keep)


def note_pitch_points(note: ExportNote, curve, mode: str) -> tuple[list[dict], bool]:
    """(피치점, snap_first). 곡선이 모자라면 OpenUtau 기본 두 점."""
    default = [{"x": -40.0, "y": 0.0, "shape": "io"}, {"x": 40.0, "y": 0.0, "shape": "io"}]
    if mode == "none" or curve is None:
        return default, True
    eps, cap, win = _PITCH_PARAMS[mode]
    fr = curve.frame_rate
    i0 = max(0, int(np.ceil((note.start - curve.start_time) * fr)))
    i1 = min(len(curve.f0_hz), int(np.ceil((note.end - curve.start_time) * fr - 1e-9)))  # 끝 시각 프레임은 다음 음표
    if i1 - i0 < 3:
        return default, True
    f0 = np.asarray(curve.f0_hz[i0:i1], dtype=float)
    conf = np.asarray(curve.confidence[i0:i1], dtype=float)
    ok = np.isfinite(f0) & (f0 > 0) & (conf >= MIN_CONF)
    if ok.sum() < 3:
        return default, True
    t_ms = (curve.start_time + (np.arange(i0, i1) / fr) - note.start) * 1000.0
    semi = 69 + 12 * np.log2(np.where(ok, f0, 440.0) / 440.0) - note.pitch
    t_ms, semi = t_ms[ok], semi[ok]
    w = max(1, int(round(win * fr)))
    if w > 1 and semi.size >= w:
        pad = w // 2
        padded = np.pad(semi, pad, mode="edge")
        semi = np.array([np.median(padded[k:k + w]) for k in range(semi.size)])
    semi = np.clip(semi, -12.0, 12.0)  # 음표에서 한 옥타브 넘게 벗어난 값은 피치 검출 오류로 본다
    idx = _rdp(t_ms, semi, eps)
    while len(idx) > cap:
        eps *= 1.5
        idx = _rdp(t_ms, semi, eps)
    pts = [{"x": round(float(t_ms[k]), 1), "y": round(float(semi[k]) * 10.0, 1), "shape": "l"} for k in idx]
    return pts, False


def build(project, options: ExportOptions, name: str = "SONATLAS") -> tuple[dict, ExportReport]:
    notes, warnings = pick_notes(project, options)
    rep = ExportReport("USTX", notes=len(notes), warnings=list(warnings), source=options.source, range=options.range)
    pts, bpb, unit = tempo_points(project)
    if options.source == "reference" and project.reference is not None and project.reference.tempo:
        # 참조 악보는 정렬된 음원 시각으로 내보내므로 박자 격자는 프로젝트 템포를 쓴다
        rep.warnings.append("참조 악보 음표를 음원 시각(정렬 결과)으로 내보냈습니다. 악보 자체의 템포 변화는 담지 않습니다.")
    tm = TickMap(pts, unit)
    curve = project.analysis.pitch_curve if project.analysis is not None else None
    if options.source == "reference" and options.pitch != "none":
        rep.warnings.append("참조 악보에는 실제 피치 곡선을 붙이지 않았습니다(악보 음높이 그대로).")
    mode = options.pitch if options.source != "reference" else "none"
    out_notes = []
    prev_end_tick = -1
    prev_note: dict | None = None
    for n in sorted(notes, key=lambda x: (x.start, x.pitch)):
        pos, end = tm.tick(n.start), tm.tick(n.end)
        if end <= pos:
            end = pos + 1
        lyric = DEFAULT_LYRIC
        if options.lyrics and n.lyric:
            lyric = n.lyric
            rep.lyrics += 1
        elif options.lyrics and n.extend and prev_note is not None:
            gap_s = n.start - (tm.seconds(prev_end_tick) if prev_end_tick >= 0 else n.start)
            if prev_end_tick < pos and gap_s <= GAP_CLOSE_S:
                prev_note["duration"] = pos - prev_note["position"]  # 빈틈 메우기 (이어 부르기 조건: 앞 끝 = 이 시작)
                rep.gaps_closed += 1
                prev_end_tick = pos
            if prev_end_tick == pos:
                lyric = EXTEND
                rep.extended += 1
            else:
                rep.warnings.append(f"{n.start:.2f}초 음표: 같은 음절 앞 음표와 {gap_s:.2f}초 떨어져 이어 부르기로 쓸 수 없어 "
                                    "기본 가사로 두었습니다.")
        elif options.lyrics and n.extend:
            lyric = EXTEND
        if prev_note is not None and pos < prev_end_tick:  # 겹침: 앞 음표를 줄인다 (OpenUtau 한 파트는 단성부 편집이 기본)
            prev_note["duration"] = max(1, pos - prev_note["position"])
        points, snap = note_pitch_points(n, curve, mode)
        note = {"position": pos, "duration": end - pos, "tone": int(n.pitch), "lyric": lyric,
                "pitch": {"data": points, "snap_first": snap},
                "vibrato": {"length": 0, "period": 175, "depth": 25, "in": 10, "out": 10, "shift": 0, "drift": 0,
                            "vol_link": 0},
                "phoneme_expressions": [], "phoneme_overrides": []}
        out_notes.append(note)
        prev_note, prev_end_tick = note, end
    if options.lyrics and not any(n.lyric for n in notes):
        rep.warnings.append("연결된 가사가 없어 모든 음표에 OpenUtau 기본 가사('a')를 넣었습니다.")
    part_end = max((x["position"] + x["duration"] for x in out_notes), default=0)
    doc = {
        "name": name, "comment": "SONATLAS에서 내보냄. 가수·phonemizer는 OpenUtau에서 고르세요.",
        "output_dir": "Vocal", "cache_dir": "UCache", "ustx_version": USTX_VERSION, "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": int(bpb), "beat_unit": int(unit)}],
        "tempos": [{"position": tk, "bpm": round(b, 4)} for tk, b in tm.tempo_ticks()],
        "tracks": [{"track_name": "SONATLAS 보컬", "track_color": "Blue", "mute": False, "solo": False, "volume": 0, "pan": 0}],
        "voice_parts": [{"name": "보컬", "comment": "", "track_no": 0, "position": 0, "duration": int(part_end),
                         "notes": out_notes, "curves": []}],
        "wave_parts": [],
    }
    rep.notes = len(out_notes)
    rep.omitted += ["가수(보이스뱅크)·phonemizer 지정 (OpenUtau에서 선택)", "음표 신뢰도·수정 여부·발음 기호·멜리스마 판정 근거"]
    if mode == "none":
        rep.omitted.append("실제 피치 곡선 (옵션: 음표와 가사만)")
    return doc, rep


def dumps(doc: dict) -> str:
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096)


def export(project, path: str, options: ExportOptions, name: str = "SONATLAS") -> ExportReport:
    from amp.core.fileio import atomic_write_bytes

    doc, rep = build(project, options, name)
    atomic_write_bytes(path, dumps(doc).encode("utf-8"))
    rep.path = path
    return rep
