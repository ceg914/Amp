"""음향 분석 실행·저장 (v0.7.0 Phase 3). Qt 비의존.

입력: 분석 소스(보컬 등)의 pYIN 프레임(보컬 분석과 같은 캐시) + 프레임 음향 지표(작업 프로세스) + 화면 쪽에서 넘긴
음표·가사 음절 시작·멜리스마 구간(분석 시점의 스냅샷).
결과 파일 (analysis/): vibrato.json · breath.json · phonation.json · boundaries.json · acoustic_features.npz.
모두 공통 머리말(meta)을 두고, 소스가 바뀌면 stale로 표시한다(지우지 않음). 호흡·프레이즈 사용자 수정(overrides)은
다시 분석해도 남긴다. 음표·가사·피치 곡선은 바꾸지 않는다.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from amp.core import result_meta as RM
from amp.engine.acoustics import breath as B
from amp.engine.acoustics import onsets as O
from amp.engine.acoustics import phonation as P
from amp.engine.acoustics import vibrato as VB

FILES = {"vibrato": "vibrato.json", "breath": "breath.json", "phonation": "phonation.json",
         "boundaries": "boundaries.json"}
FEATURES_NPZ = "acoustic_features.npz"
SCHEMAS = {k: f"sonatlas.{k}" for k in FILES}
SCHEMA_VERSION = 1


@dataclass
class Snapshot:
    """분석 시점의 음표·가사 (화면 스레드에서 만들어 넘긴다)."""

    notes: list = field(default_factory=list)
    syllable_starts: list | None = None
    melisma: list = field(default_factory=list)  # (시작, 끝, 음절 id)


@dataclass
class AcousticResult:
    vibrato: list
    vibrato_summary: dict
    frame_classes: list  # [(분류, 시작초, 끝초)] 이어진 구간
    breaths: list
    phrases: list
    breath_summary: dict
    events: list
    note_metrics: list
    phrase_metrics: list
    boundaries: list
    meta: dict = field(default_factory=dict)  # 종류 → ResultMeta
    overrides: dict = field(default_factory=dict)
    syllable_starts: list | None = None


def _class_runs(cls, fr):
    out = []
    if len(cls) == 0:
        return out
    start = 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[start]:
            out.append((str(cls[start]), round(start / fr, 3), round(i / fr, 3)))
            start = i
    return out


def analyze(f0_hz, feat: dict, snap: Snapshot, frame_rate: float = 100.0, overrides: dict | None = None,
            vib_settings: VB.VibratoSettings | None = None, breath_settings: B.BreathSettings | None = None):
    """계산만 (저장하지 않음). 시험·화면 미리 보기에서도 쓴다."""
    f0 = np.asarray(f0_hz, dtype=np.float64)
    voiced = np.isfinite(f0)
    onsets = sorted(n.start for n in snap.notes)
    vib = VB.analyze_curve(f0, frame_rate, 0.0, vib_settings, onsets or None)
    cls, breaths = B.classify(feat, voiced, frame_rate, breath_settings)
    plist = B.phrases(cls, breaths, frame_rate, breath_settings, snap.syllable_starts)
    ov = dict(overrides or {})
    if ov:
        breaths, plist = B.apply_overrides(breaths, plist, ov)
    events = P.detect_events(f0, frame_rate, snap.notes, vib, snap.melisma)
    nm = P.span_metrics(feat, f0, [(n.start, n.end, n.id) for n in sorted(snap.notes, key=lambda n: n.start)], frame_rate)
    pm = P.span_metrics(feat, f0, [(p.start, p.end, p.phrase_id) for p in plist], frame_rate)
    bnd = O.analyze(snap.notes, f0, feat["rms_db"], frame_rate) if snap.notes else []
    return AcousticResult(vib, VB.summary(vib), _class_runs(cls, frame_rate), breaths, plist,
                          B.summary(cls, breaths, plist, frame_rate), events, nm, pm, bnd, overrides=ov,
                          syllable_starts=list(snap.syllable_starts) if snap.syllable_starts is not None else None)


def run(inp, analysis_dir: str | Path, runner, snap: Snapshot, cache_root: str | None = None, progress=None,
        token=None, numba_cache_dir: str | None = None) -> AcousticResult:
    from amp.engine import pitch_runs as PR

    report = progress or (lambda f, m: None)
    d = Path(analysis_dir)
    req = PR.PitchRunRequest(inp, str(d), ("pyin",), cache_root=cache_root, numba_cache_dir=numba_cache_dir)
    with tempfile.TemporaryDirectory(prefix="sonatlas-ac-") as scratch:
        f0, _conf, _v, pinfo = PR._pyin(req, runner, lambda f, m: report(0.4 * f, m), token, Path(scratch))  # noqa: SLF001
        np.savez_compressed(Path(scratch) / "f0.npz", f0_hz=f0)
        out = Path(scratch) / "feat"
        out.mkdir()
        runner("acoustic_features", {"input": inp.path, "f0_npz": str(Path(scratch) / "f0.npz"), "out_dir": str(out)},
               lambda f, m: report(0.4 + 0.4 * f, m), token, numba_cache_dir)
        with np.load(out / "features.npz", allow_pickle=False) as z:
            feat = {k: z[k].astype(np.float64) for k in z.files}
    report(0.85, "비브라토·호흡·발성·경계 분석 중")
    old = load(d)
    res = analyze(f0, feat, snap, 100.0, (old.overrides if old is not None else None))
    metas = {}
    for kind in FILES:
        settings = {"vibrato": VB.VibratoSettings().to_dict(), "breath": B.BreathSettings().to_dict()}.get(kind, {})
        reason = ""
        if kind == "boundaries" and not snap.notes:
            reason = "음표가 없어 경계를 재지 않았습니다."
        metas[kind] = RM.make_meta(SCHEMAS[kind], SCHEMA_VERSION, inp.source_sha256, input_kind=inp.kind,
                                   input_id=inp.input_id, input_hash=inp.input_hash, model_id="pyin+dsp",
                                   analysis_version={"vibrato": VB.ANALYSIS_VERSION, "breath": B.ANALYSIS_VERSION,
                                                     "phonation": P.ANALYSIS_VERSION,
                                                     "boundaries": O.ANALYSIS_VERSION}[kind],
                                   analysis_settings=settings, status=RM.UNCERTAIN if reason else RM.COMPLETED,
                                   status_reason=reason)
    res.meta = metas
    save(d, res, feat)
    report(1.0, "음향 분석 완료")
    return res


def save(analysis_dir: str | Path, res: AcousticResult, feat: dict | None = None) -> None:
    from amp.core.fileio import atomic_write_json

    d = Path(analysis_dir)
    d.mkdir(parents=True, exist_ok=True)
    if feat is not None:
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=d)
        import os

        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **{k: np.asarray(v, dtype=np.float32) for k, v in feat.items()})
            os.replace(tmp, d / FEATURES_NPZ)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    m = lambda k: res.meta[k].to_dict() if k in res.meta else None  # noqa: E731
    head = lambda k: {"schema": SCHEMAS[k], "schema_version": SCHEMA_VERSION, "meta": m(k)}  # noqa: E731
    atomic_write_json(d / FILES["vibrato"], {**head("vibrato"), "summary": res.vibrato_summary,
                                             "segments": [s.to_dict() for s in res.vibrato],
                                             "extent_definition": "extent_cents = 한쪽 진폭, extent_pp_cents = 봉우리 간 폭"})
    atomic_write_json(d / FILES["breath"], {**head("breath"), "summary": res.breath_summary,
                                            "frame_classes": res.frame_classes,
                                            "breaths": [b.to_dict() for b in res.breaths],
                                            "phrases": [p.to_dict() for p in res.phrases], "overrides": res.overrides,
                                            "syllable_starts": res.syllable_starts})
    atomic_write_json(d / FILES["phonation"], {**head("phonation"), "events": [e.to_dict() for e in res.events],
                                               "note_metrics": [x.to_dict() for x in res.note_metrics],
                                               "phrase_metrics": [x.to_dict() for x in res.phrase_metrics],
                                               "note": "분리 과정에서 음색·배음이 바뀔 수 있습니다. 이 지표만으로 흉성·두성·"
                                                       "믹스보이스·성대 상태를 정하지 않습니다."})
    atomic_write_json(d / FILES["boundaries"], {**head("boundaries"), "notes": [b.to_dict() for b in res.boundaries],
                                                "applied": False,
                                                "note": "음표·가사에 자동으로 적용하지 않습니다 (교정 제안으로만)."})


def load(analysis_dir: str | Path) -> AcousticResult | None:
    d = Path(analysis_dir)
    try:
        raw = {k: json.loads((d / name).read_text(encoding="utf-8")) for k, name in FILES.items()}
    except (OSError, ValueError):
        return None
    if any(raw[k].get("schema") != SCHEMAS[k] or int(raw[k].get("schema_version", 0)) > SCHEMA_VERSION for k in FILES):
        return None
    try:
        vib = [VB.VibratoSegment(**{k: v for k, v in s.items() if k != "duration"}) for s in raw["vibrato"]["segments"]]
        br = raw["breath"]
        breaths = [B.Breath(**b) for b in br["breaths"]]
        plist = [B.Phrase(**{k: v for k, v in p.items() if k != "duration"}) for p in br["phrases"]]
        events = [P.PitchEvent(**{k: v for k, v in e.items() if k != "duration"}) for e in raw["phonation"]["events"]]
        nm = [P.SpanMetrics(**x) for x in raw["phonation"]["note_metrics"]]
        pm = [P.SpanMetrics(**x) for x in raw["phonation"]["phrase_metrics"]]
        bnd = [O.NoteBoundary(**b) for b in raw["boundaries"]["notes"]]
    except (TypeError, KeyError):
        return None
    metas = {k: RM.meta_from_dict(raw[k].get("meta")) for k in FILES}
    return AcousticResult(vib, raw["vibrato"].get("summary", {}), [tuple(x) for x in br.get("frame_classes", [])],
                          breaths, plist, br.get("summary", {}), events, nm, pm, bnd,
                          {k: v for k, v in metas.items() if v is not None}, dict(br.get("overrides") or {}),
                          br.get("syllable_starts"))


def classes_array(runs_, frame_rate: float = 100.0) -> np.ndarray:
    n = int(round(max((e for _c, _s, e in runs_), default=0.0) * frame_rate))
    out = np.full(n, "unknown", dtype=object)
    for c, s, e in runs_:
        out[int(round(s * frame_rate)):int(round(e * frame_rate))] = c
    return out


def save_overrides(analysis_dir: str | Path, overrides: dict) -> AcousticResult | None:
    """호흡·프레이즈 사용자 수정만 바꿔 저장 (비브라토·발성·경계 결과는 그대로). 수정을 반영한 결과를 돌려준다.
    자동 호흡 후보는 상태를 되돌린 뒤 수정을 다시 적용하고, 프레이즈는 그 호흡 상태로 다시 나눈다(직접 정한 프레이즈가 있으면 그것)."""
    from dataclasses import asdict

    res = load(analysis_dir)
    if res is None:
        return None
    auto = [B.Breath(**{**asdict(b), "status": "candidate"}) for b in res.breaths if not b.breath_id.startswith("u")]
    cls = classes_array(res.frame_classes)
    breaths, _ = B.apply_overrides(auto, [], {k: v for k, v in overrides.items() if k != "phrases"})
    plist = B.phrases(cls, breaths, 100.0, None, res.syllable_starts)
    if overrides.get("phrases"):
        _b, plist = B.apply_overrides([], plist, {"phrases": overrides["phrases"]})
    res.breaths, res.phrases, res.overrides = breaths, plist, dict(overrides)
    res.breath_summary = B.summary(cls, breaths, plist)
    save(analysis_dir, res)
    return res


def mark_stale(analysis_dir: str | Path, current_source_id: str, source_sha256: str) -> list[str]:
    """분석 소스가 바뀌었으면 각 파일 머리말을 stale로 (결과는 지우지 않음)."""
    from amp.core.fileio import atomic_write_json

    d = Path(analysis_dir)
    changed = []
    for k, name in FILES.items():
        p = d / name
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m = RM.meta_from_dict(j.get("meta"))
        if m is None:
            continue
        m2 = RM.mark_stale_if_changed(m, source_sha256, current_source_id)
        if m2 is not m:
            j["meta"] = m2.to_dict()
            atomic_write_json(p, j)
            changed.append(k)
    return changed
