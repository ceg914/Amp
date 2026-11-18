"""분석 실행 기록과 실행 간 비교 보고서.

분석(전체·음표만 다시 검출·샘플)이 성공할 때마다 기록 하나를 남긴다. 실패·취소된 실행은 남기지 않는다.

    analysis/runs/<run_id>.json        실행 기록 (자동 분석 결과만. 사용자 편집 음표는 개수만 적는다)
    analysis/reports/compare-<A>-vs-<B>.md   두 실행 비교 보고서

기록은 읽기 전용 자료다. 프로젝트의 작업 악보(score)나 notes.json을 바꾸지 않는다.
비교는 음표 시작 시각 ±50ms 안, 같은 음높이면 같은 음표로 본다(amp.devtools.evaluation과 같은 기준).
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from amp.core.errors import ProjectFormatError
from amp.core.fileio import atomic_write_bytes, atomic_write_json, read_json

log = logging.getLogger(__name__)

RUNS_DIR = "runs"
REPORTS_DIR = "reports"
RECORD_VERSION = 1
MATCH_ONSET_TOL = 0.05  # 초
STAGE_LABELS = {"separation": "보컬 분리", "pitch": "피치 추출", "tempo": "BPM 추정", "notes": "음표 검출",
                "total": "전체"}


@dataclass
class RunRecord:
    run_id: str
    kind: str  # "full" | "redetect" | "sample"
    created: str
    app_version: str
    audio_sha256: str
    settings: dict
    models: dict  # 단계 → {name, version, params}
    timings_s: dict  # 단계 → 초 (작업 프로세스 시작 포함)
    reused: dict  # 단계 → 캐시·기존 결과 재사용 여부
    resources: dict  # 단계 → {peak_rss_mb, gpu_peak_mb, ...} (작업 프로세스가 알려 준 값)
    tempo: dict
    notes: list[dict]  # {start, end, pitch, confidence, ornament, vibrato}
    warnings: list[str] = field(default_factory=list)
    score_summary: dict = field(default_factory=dict)  # 실행 시점의 사용자 편집 현황 (개수만)
    seq: int = 0  # 프로젝트 안 저장 순번 (save_run이 매긴다). 시각이 같아도 순서가 정해진다
    version: int = RECORD_VERSION

    @property
    def total_seconds(self) -> float | None:
        return self.timings_s.get("total")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        names = set(cls.__dataclass_fields__)
        missing = {"run_id", "kind", "created", "notes"} - set(d)
        if missing:
            raise ProjectFormatError(f"실행 기록에 필요한 항목이 없습니다: {', '.join(sorted(missing))}")
        base = {"app_version": "", "audio_sha256": "", "settings": {}, "models": {}, "timings_s": {}, "reused": {},
                "resources": {}, "tempo": {}}
        return cls(**{**base, **{k: v for k, v in d.items() if k in names}})


def new_run_id(now: datetime | None = None) -> str:
    """시각(밀리초까지) + 임의 4자리. 이름순이 곧 시간순이다."""
    now = now or datetime.now()
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}-" + uuid.uuid4().hex[:4]


def record_from_result(result, kind: str, *, timings: dict | None = None, reused: dict | None = None,
                       resources: dict | None = None, score_summary: dict | None = None,
                       app_version: str = "") -> RunRecord:
    """AnalysisResult → 실행 기록. result는 바꾸지 않는다."""
    notes = []
    for n in sorted(result.notes, key=lambda n: (n.start, n.pitch)):
        notes.append({
            "start": round(float(n.start), 4), "end": round(float(n.end), 4), "pitch": int(n.pitch),
            "confidence": _finite(n.confidence, 3),
            "ornament": bool(getattr(n, "ornament", False)),
            "vibrato": None if n.vibrato is None else {"rate_hz": _finite(n.vibrato.rate_hz, 2),
                                                       "extent_cents": _finite(n.vibrato.extent_cents, 1)},
        })
    models = {k: {"name": m.name, "version": m.version, "placeholder": m.placeholder, "params": dict(m.params or {})}
              for k, m in (result.models or {}).items()}
    tempo = {}
    if result.tempo is not None and result.tempo.points:
        p0 = result.tempo.points[0]
        tempo = {"bpm": round(float(p0.bpm), 3), "first_beat": round(float(getattr(p0, "time", 0.0)), 4)}
    return RunRecord(
        run_id=new_run_id(), kind=kind, created=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        app_version=app_version, audio_sha256=result.audio_fingerprint or "", settings=dict(result.settings or {}),
        models=models, timings_s={k: round(float(v), 3) for k, v in (timings or {}).items()},
        reused=dict(reused or {}), resources=dict(resources or {}), tempo=tempo, notes=notes,
        warnings=list(result.warnings or ()), score_summary=dict(score_summary or {}),
    )


# --- 저장·읽기 ----------------------------------------------------------------------

def runs_dir(analysis_dir: str | os.PathLike) -> Path:
    return Path(analysis_dir) / RUNS_DIR


def _finite(v, digits: int):
    if v is None:
        return None
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


def _json_safe(o):
    """NaN·무한대는 JSON에 쓸 수 없으므로 None으로 바꾼다 (모델 파라미터 등 어디서 섞여도 기록이 실패하지 않게)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def save_run(analysis_dir: str | os.PathLike, record: RunRecord) -> Path:
    record.seq = max((r.seq for r in list_runs(analysis_dir)), default=0) + 1
    path = runs_dir(analysis_dir) / f"{record.run_id}.json"
    atomic_write_json(path, _json_safe(record.to_dict()))
    return path


def list_runs(analysis_dir: str | os.PathLike) -> list[RunRecord]:
    """오래된 것부터. 읽을 수 없는 기록은 건너뛰고 로그에 남긴다."""
    d = runs_dir(analysis_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(RunRecord.from_dict(read_json(p, what="실행 기록")))
        except (ProjectFormatError, OSError, TypeError, ValueError) as e:
            log.warning("실행 기록을 읽지 못했습니다 (%s): %s", p.name, e)
    out.sort(key=lambda r: (r.seq, r.created, r.run_id))
    return out


def find_run(analysis_dir: str | os.PathLike, run_id: str) -> RunRecord:
    matches = [r for r in list_runs(analysis_dir) if r.run_id == run_id or r.run_id.startswith(run_id)]
    if len(matches) != 1:
        raise KeyError(f"실행 기록을 찾을 수 없습니다: {run_id}" if not matches else f"여러 기록과 일치합니다: {run_id}")
    return matches[0]


# --- 비교 ----------------------------------------------------------------------------

@dataclass
class RunComparison:
    a: RunRecord
    b: RunRecord
    matched: list[tuple[dict, dict]]  # (A 음표, B 음표)
    only_a: list[dict]
    only_b: list[dict]
    pitch_changed: list[tuple[dict, dict]]  # 시작은 같은데(±50ms) 음높이가 다른 쌍

    def _mean_abs(self, key: str) -> float | None:
        if not self.matched:
            return None
        return sum(abs(nb[key] - na[key]) for na, nb in self.matched) / len(self.matched)

    @property
    def mean_onset_shift_ms(self) -> float | None:
        v = self._mean_abs("start")
        return None if v is None else round(v * 1000, 1)

    @property
    def mean_offset_shift_ms(self) -> float | None:
        v = self._mean_abs("end")
        return None if v is None else round(v * 1000, 1)

    def settings_diff(self) -> dict[str, tuple]:
        keys = sorted(set(self.a.settings) | set(self.b.settings))
        return {k: (self.a.settings.get(k), self.b.settings.get(k)) for k in keys
                if self.a.settings.get(k) != self.b.settings.get(k)}

    def models_diff(self) -> dict[str, tuple]:
        out = {}
        for k in sorted(set(self.a.models) | set(self.b.models)):
            ma, mb = self.a.models.get(k, {}), self.b.models.get(k, {})
            va = f"{ma.get('name', '-')} {ma.get('version', '')}".strip()
            vb = f"{mb.get('name', '-')} {mb.get('version', '')}".strip()
            if va != vb:
                out[k] = (va, vb)
        return out


def _pair(a_notes: list[dict], b_notes: list[dict], same_pitch: bool) -> list[tuple[int, int]]:
    cands = []
    for i, na in enumerate(a_notes):
        for j, nb in enumerate(b_notes):
            d = abs(nb["start"] - na["start"])
            if d <= MATCH_ONSET_TOL and ((nb["pitch"] == na["pitch"]) == same_pitch):
                cands.append((d, i, j))
    cands.sort()
    used_a, used_b, out = set(), set(), []
    for _d, i, j in cands:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j))
    return out


def compare_runs(a: RunRecord, b: RunRecord) -> RunComparison:
    pairs = _pair(a.notes, b.notes, same_pitch=True)
    ia, ib = {i for i, _ in pairs}, {j for _, j in pairs}
    rest_a = [n for i, n in enumerate(a.notes) if i not in ia]
    rest_b = [n for j, n in enumerate(b.notes) if j not in ib]
    changed = _pair(rest_a, rest_b, same_pitch=False)
    ca, cb = {i for i, _ in changed}, {j for _, j in changed}
    return RunComparison(
        a, b, [(a.notes[i], b.notes[j]) for i, j in pairs],
        [n for i, n in enumerate(rest_a) if i not in ca], [n for j, n in enumerate(rest_b) if j not in cb],
        [(rest_a[i], rest_b[j]) for i, j in changed],
    )


def _note_text(n: dict) -> str:
    conf = "-" if n.get("confidence") is None else f"{n['confidence']:.2f}"
    orn = " 꾸밈음" if n.get("ornament") else ""
    return f"{n['start']:.3f}–{n['end']:.3f}s  MIDI {n['pitch']}  신뢰도 {conf}{orn}"


def markdown_report(cmp: RunComparison, max_rows: int = 200) -> str:
    a, b = cmp.a, cmp.b
    lines = [f"# 분석 실행 비교: {a.run_id} → {b.run_id}", ""]
    lines += ["| 항목 | A | B |", "|---|---|---|"]
    lines.append(f"| 실행 ID | {a.run_id} | {b.run_id} |")
    lines.append(f"| 종류 | {a.kind} | {b.kind} |")
    lines.append(f"| 시각 (UTC) | {a.created} | {b.created} |")
    lines.append(f"| 앱 버전 | {a.app_version or '-'} | {b.app_version or '-'} |")
    for stage in sorted(set(a.models) | set(b.models)):
        ma, mb = a.models.get(stage, {}), b.models.get(stage, {})
        lines.append(f"| 모델·버전: {stage} | {ma.get('name', '-')} {ma.get('version', '')} | "
                     f"{mb.get('name', '-')} {mb.get('version', '')} |")
    lines.append(f"| 민감도 | {a.settings.get('sensitivity', '-')} | {b.settings.get('sensitivity', '-')} |")
    lines.append(f"| 음표 수 | {len(a.notes)} | {len(b.notes)} |")
    lines.append(f"| BPM | {a.tempo.get('bpm', '-')} | {b.tempo.get('bpm', '-')} |")
    for stage in ("separation", "pitch", "tempo", "notes", "total"):
        if stage in a.timings_s or stage in b.timings_s:
            ra = " (재사용)" if a.reused.get(stage) else ""
            rb = " (재사용)" if b.reused.get(stage) else ""
            lines.append(f"| 처리 시간: {STAGE_LABELS[stage]} | {_sec(a.timings_s.get(stage))}{ra} | "
                         f"{_sec(b.timings_s.get(stage))}{rb} |")
    for stage in sorted(set(a.resources) | set(b.resources)):
        lines.append(f"| 메모리: {STAGE_LABELS.get(stage, stage)} | {_mem(a.resources.get(stage))} | "
                     f"{_mem(b.resources.get(stage))} |")
    if a.score_summary or b.score_summary:
        lines.append(f"| 실행 시점 사용자 편집 (수정/추가) | {_edits(a.score_summary)} | {_edits(b.score_summary)} |")
    lines += ["", "## 음표 비교", "",
              f"시작 시각 ±{int(MATCH_ONSET_TOL * 1000)}ms 안이고 음높이가 같으면 같은 음표로 본다.", "",
              f"- 일치: {len(cmp.matched)}개 (시작 이동 평균 {_ms(cmp.mean_onset_shift_ms)}, "
              f"끝 이동 평균 {_ms(cmp.mean_offset_shift_ms)})",
              f"- 음높이만 바뀜: {len(cmp.pitch_changed)}개",
              f"- A에만 있음: {len(cmp.only_a)}개",
              f"- B에만 있음: {len(cmp.only_b)}개"]
    settings = cmp.settings_diff()
    if settings:
        lines += ["", "## 바뀐 설정", "", "| 설정 | A | B |", "|---|---|---|"]
        lines += [f"| {k} | {va} | {vb} |" for k, (va, vb) in settings.items()]
    for title, rows in (("음높이만 바뀐 음표", [f"{_note_text(x)} → MIDI {y['pitch']}" for x, y in cmp.pitch_changed]),
                        ("A에만 있는 음표", [_note_text(n) for n in cmp.only_a]),
                        ("B에만 있는 음표", [_note_text(n) for n in cmp.only_b])):
        if rows:
            lines += ["", f"## {title}", ""]
            lines += [f"- {r}" for r in rows[:max_rows]]
            if len(rows) > max_rows:
                lines.append(f"- … 외 {len(rows) - max_rows}개")
    warn = [w for w in b.warnings if w not in a.warnings]
    if warn:
        lines += ["", "## B에서 새로 생긴 경고", ""] + [f"- {w}" for w in warn]
    lines += ["", "자동 분석 결과끼리 비교한 것이다. 사용자가 수정·추가한 음표는 비교 대상이 아니며 바뀌지 않는다.", ""]
    return "\n".join(lines)


def write_report(analysis_dir: str | os.PathLike, a: RunRecord, b: RunRecord) -> Path:
    path = Path(analysis_dir) / REPORTS_DIR / f"compare-{a.run_id}-vs-{b.run_id}.md"
    atomic_write_bytes(path, markdown_report(compare_runs(a, b)).encode("utf-8"))
    return path


def _sec(v) -> str:
    return "-" if v is None else f"{v:.2f}s"


def _ms(v) -> str:
    return "-" if v is None else f"{v}ms"


def _mem(r: dict | None) -> str:
    if not r:
        return "-"
    parts = []
    if r.get("peak_rss_mb") is not None:
        parts.append(f"RAM 최대 {r['peak_rss_mb']}MB")
    if r.get("gpu_peak_mb") is not None:
        parts.append(f"GPU 최대 {r['gpu_peak_mb']}MB")
    return ", ".join(parts) or "-"


def _edits(s: dict) -> str:
    if not s:
        return "-"
    return f"{s.get('modified', 0)}/{s.get('user', 0)}"
