"""악기 소리 분리 (v0.6.4): 드럼·베이스·기타·피아노·나머지를 파일로 뽑는다. 악보는 만들지 않는다. Qt 비의존.

입력 (v0.6.7):
- 'instrumental' (기본): 지금 보컬 분석에 쓰는 분리 결과의 반주. 보컬을 이미 뺀 소리라 악기 파일에 보컬이 덜 섞인다.
  결과는 그 분리 결과 폴더에 악기 파일로 더한다(만든 모델·입력·반주 해시를 목록에 적음). 새 '보컬' 결과를 만들지 않는다
- 'original': 원본을 그대로 분리 (v0.6.4 방식). 모델별 보관 결과에 넣는다

- Demucs htdemucs: 드럼·베이스·나머지 (보컬 분리 때 쓰는 모델과 같다)
- Demucs htdemucs_6s: 기타·피아노까지 나눈다. Demucs 설명서에 피아노 분리는 아직 잘 되지 않는다고 적혀 있다
- 분리 결과는 분리 보관소(stems/models/<키>/)에 악기 이름.wav로 둔다. 같은 결과가 있으면 빠진 악기 파일만 더하고,
  보컬·반주·이미 있는 악기 파일은 바꾸지 않는다(같은 곡도 분리할 때마다 조금씩 달라서). 지금 쓰는 분리 결과도 바꾸지 않는다

입력별 구분 (v0.7.0, 명세 15.1·14.3): '이미 있다'는 (악기, 입력, 만든 모델)이 모두 같을 때만이다.
원본에서 만든 드럼이 있어도 반주에서 드럼을 요청하면 분리하고, 그 반대도 같다. 한 폴더에 같은 악기가 여럿이면
두 번째부터는 'drums@instrumental-htdemucs_6s.wav'처럼 입력·모델을 붙인 이름으로 둔다(있던 파일은 덮어쓰지 않음).
반주가 다시 만들어지면 그 반주에서 만든 악기는 지우지 않고 '오래됨'으로 표시한다.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from amp.core.errors import AnalysisError
from amp.engine import stems as stem_store
from amp.engine.context import CancelToken, ProgressFn
from amp.engine.model_store import MODEL_LABELS, ModelStore, file_sha256, identity_for

INPUTS = {"original": "원본", "instrumental": "반주"}
INSTRUMENTS = {"drums": "드럼", "bass": "베이스", "guitar": "기타", "piano": "피아노", "other": "나머지"}
MODEL_STEMS = {"htdemucs": ("drums", "bass", "other"), "htdemucs_6s": ("drums", "bass", "guitar", "piano", "other")}
_PREFERENCE = {"htdemucs": 0, "htdemucs_6s": 1}  # 같은 악기가 여러 결과에 있으면: 지금 쓰는 결과 → htdemucs → 6s
MB_PER_MIN = 15.9  # 악기 하나: 44.1kHz 스테레오 24bit WAV (조각 나비 베이스 4분 36초 = 73MB)


@dataclass(frozen=True)
class StemFile:
    name: str
    key: str  # 분리 보관 결과 키
    model: str  # 이 악기 파일을 만든 모델
    path: Path
    source_model: str = ""  # 반주에서 뽑았으면 그 반주를 만든 분리 결과의 모델
    slot: str = ""  # 목록 안 이름 (v0.7.0: drums, drums@instrumental-htdemucs_6s)
    stale: bool = False  # 이 파일을 만든 반주가 다시 만들어짐 (v0.7.0)
    added: str = ""
    input_sha256: str = ""  # 분리에 넣은 소리의 해시 (v0.7.0에서 적기 시작)

    @property
    def from_instrumental(self) -> bool:
        return bool(self.source_model)

    @property
    def input(self) -> str:
        return "instrumental" if self.source_model else "original"

    @property
    def model_label(self) -> str:
        label = MODEL_LABELS.get(self.model, self.model)
        if self.source_model:
            label += f" · {MODEL_LABELS.get(self.source_model, self.source_model)} 반주에서"
        else:
            label += " · 원본에서"
        if self.stale:
            label += " · 오래됨"
        return label


@dataclass
class ExtractResult:
    key: str
    model: str
    added: list[str] = field(default_factory=list)  # 이번에 새로 저장한 악기
    kept: list[str] = field(default_factory=list)  # 같은 입력·모델로 이미 있어 그대로 둔 악기
    separated: bool = False
    source_model: str = ""  # 반주에서 뽑았으면 그 반주를 만든 분리 결과의 모델
    input: str = "original"


def stem_variants(entry) -> list[StemFile]:
    """보관 결과 하나에 든 악기 파일 전부 (파일이 목록과 맞는 것만)."""
    out = []
    for slot, meta in entry.manifest.files.items():
        name = stem_store.stem_base(slot)
        if name not in INSTRUMENTS or not isinstance(meta, dict):
            continue
        p = stem_store.optional_path(entry.path, entry.manifest, slot)
        if p is None:
            continue
        from_inst = meta.get("input") == "instrumental"
        out.append(StemFile(name, entry.key, str(meta.get("made_by") or entry.model), p,
                            entry.model if from_inst else "", slot, bool(meta.get("stale")),
                            str(meta.get("added") or ""), str(meta.get("input_sha256") or "")))
    return out


def _ranked(stems_dir, source_sha256, prefer_input: str | None):
    entries = ModelStore(stems_dir, source_sha256).entries()
    entries.sort(key=lambda e: e.manifest.created, reverse=True)
    entries.sort(key=lambda e: (not e.active, _PREFERENCE.get(e.model, 2)))
    rank = {e.key: i for i, e in enumerate(entries)}
    items = [v for e in entries for v in stem_variants(e)]
    items.sort(key=lambda v: v.added, reverse=True)
    items.sort(key=lambda v: (v.stale, bool(prefer_input) and v.input != prefer_input, rank[v.key],
                              v.slot != v.name))
    return items


def find_stems(stems_dir: str | Path, source_sha256: str, prefer_input: str | None = None) -> dict[str, StemFile]:
    """악기 이름 → 파일 하나. 오래되지 않은 것 → (prefer_input을 주면) 그 입력 → 지금 쓰는 분리 결과 → htdemucs → 6s
    → 그 밖. 같은 순위면 기본 자리, 최근 것."""
    out: dict[str, StemFile] = {}
    for v in _ranked(stems_dir, source_sha256, prefer_input):
        out.setdefault(v.name, v)
    return {n: out[n] for n in INSTRUMENTS if n in out}


def find_all_stems(stems_dir: str | Path, source_sha256: str) -> dict[str, list[StemFile]]:
    """악기 이름 → 그 악기 파일 전부 (find_stems와 같은 순서). 원본·반주·모델별 결과 비교용."""
    out: dict[str, list[StemFile]] = {}
    for v in _ranked(stems_dir, source_sha256, None):
        out.setdefault(v.name, []).append(v)
    return {n: out[n] for n in INSTRUMENTS if n in out}


def _has(entry, name: str, source: str, model: str, input_sha256: str = "") -> bool:
    for v in stem_variants(entry):
        if v.name != name or v.input != source or v.model != model or v.stale:
            continue
        if input_sha256 and v.input_sha256 and v.input_sha256 != input_sha256:
            continue  # 같은 이름의 반주라도 내용이 다르다
        return True
    return False


def instrumental_source(stems_dir: str | Path, source_sha256: str):
    """반주에서 분리할 때 쓸 분리 결과 (지금 보컬 분석에 쓰는 결과). 없거나 반주 파일이 없으면 None."""
    store = ModelStore(stems_dir, source_sha256)
    key = store.active_key()
    e = store.entry(key) if key else None
    if e is None or "instrumental" not in e.manifest.files:
        return None
    return e


def extract(stems_dir: str | Path, source_sha256: str, source_path: str, model: str, runner,
            progress: ProgressFn | None = None, token: CancelToken | None = None, device: str = "auto",
            names: tuple | None = None, numba_cache_dir: str | None = None, source: str = "original") -> ExtractResult:
    """model로 분리해 악기 소리를 보관 결과에 넣는다. names를 주지 않으면 그 모델이 내는 악기 전부 (시험 모델은 names 필수).
    source='instrumental'이면 지금 보컬 분석에 쓰는 분리 결과의 반주를 분리해 그 결과 폴더에 더한다."""
    report = progress or (lambda f, m: None)
    token = token or CancelToken()
    names = tuple(names or MODEL_STEMS[model])
    if source == "instrumental":
        return _extract_from_instrumental(stems_dir, source_sha256, model, runner, report, token, device, names,
                                          numba_cache_dir)
    store = ModelStore(stems_dir, source_sha256)
    identity = identity_for(model, source_sha256)
    existing = store.find(identity)
    have = [n for n in names if existing is not None and _has(existing, n, "original", model)]
    missing = [n for n in names if n not in have]
    if existing is not None and not missing:
        report(1.0, "이미 분리해 둔 악기 소리가 있습니다")
        return ExtractResult(existing.key, model, [], have, False, "", "original")
    tmp = store.new_tmp()
    try:
        runner("separate", {"input": source_path, "out_dir": str(tmp), "model": model, "device": device,
                            "cpu_fallback": True, "extras": list(missing), "label": "악기 분리"},
               report, token, numba_cache_dir)
        got = [n for n in missing if (tmp / f"{n}.wav").is_file()]
        if not got:
            raise AnalysisError(f"분리 결과에 악기 소리가 없습니다 ({MODEL_LABELS.get(model, model)}).")
        if existing is not None:
            for n in got:
                store.add_stem(existing.key, n, tmp / f"{n}.wav",
                               {"made_by": model, "input": "original", "input_sha256": source_sha256})
            key = existing.key
        else:
            key = store.commit(tmp, identity, {"device": device, "purpose": "instruments"}).key
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            store.root.rmdir()
        except OSError:
            pass
    return ExtractResult(key, model, got, have, True, "", "original")


def _extract_from_instrumental(stems_dir, source_sha256, model, runner, report, token, device, names, numba_cache_dir):
    target = instrumental_source(stems_dir, source_sha256)
    if target is None:
        raise AnalysisError("보컬 분석에 쓰는 분리 결과가 없습니다. 보컬 분석을 먼저 하거나 '원본'에서 분리하세요.")
    store = ModelStore(stems_dir, source_sha256)
    src = target.path / target.manifest.files["instrumental"]["path"]
    inst_sha = file_sha256(str(src))  # 입력 소리를 구분한다 (명세 14.3)
    have = [n for n in names if _has(target, n, "instrumental", model, inst_sha)]
    missing = [n for n in names if n not in have]
    if not missing:
        report(1.0, "이미 분리해 둔 악기 소리가 있습니다")
        return ExtractResult(target.key, model, [], have, False, target.model, "instrumental")
    tmp = store.new_tmp()
    try:
        runner("separate", {"input": str(src), "out_dir": str(tmp), "model": model, "device": device,
                            "cpu_fallback": True, "extras": list(missing), "label": "반주에서 악기 분리"},
               report, token, numba_cache_dir)
        got = [n for n in missing if (tmp / f"{n}.wav").is_file()]
        if not got:
            raise AnalysisError(f"분리 결과에 악기 소리가 없습니다 ({MODEL_LABELS.get(model, model)}).")
        for n in got:  # 반주를 분리하며 나온 보컬·반주(남은 소리)는 버린다
            store.add_stem(target.key, n, tmp / f"{n}.wav",
                           {"made_by": model, "input": "instrumental", "input_sha256": inst_sha,
                            "input_entry": target.key, "input_stems_id": target.manifest.stems_id})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ExtractResult(target.key, model, got, have, True, target.model, "instrumental")
