"""악기 소리 분리 (v0.6.4): 드럼·베이스·기타·피아노·나머지를 파일로 뽑는다. 악보는 만들지 않는다. Qt 비의존.

입력 (v0.6.7):
- 'instrumental' (기본): 지금 보컬 분석에 쓰는 분리 결과의 반주. 보컬을 이미 뺀 소리라 악기 파일에 보컬이 덜 섞인다.
  결과는 그 분리 결과 폴더에 악기 파일로 더한다(만든 모델·입력을 목록에 적음). 새 '보컬' 결과를 만들지 않는다
- 'original': 원본을 그대로 분리 (v0.6.4 방식). 모델별 보관 결과에 넣는다

- Demucs htdemucs: 드럼·베이스·나머지 (보컬 분리 때 쓰는 모델과 같다)
- Demucs htdemucs_6s: 기타·피아노까지 나눈다. Demucs 설명서에 피아노 분리는 아직 잘 되지 않는다고 적혀 있다
- 분리 결과는 분리 보관소(stems/models/<키>/)에 악기 이름.wav로 둔다. 같은 모델 결과가 있으면 빠진 악기 파일만 더하고,
  보컬·반주·이미 있는 악기 파일은 바꾸지 않는다(같은 곡도 분리할 때마다 조금씩 달라서). 지금 쓰는 분리 결과도 바꾸지 않는다
- 이미 다 있으면 분리를 다시 하지 않는다
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from amp.core.errors import AnalysisError
from amp.engine import stems as stem_store
from amp.engine.context import CancelToken, ProgressFn
from amp.engine.model_store import MODEL_LABELS, ModelStore, identity_for

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

    @property
    def from_instrumental(self) -> bool:
        return bool(self.source_model)

    @property
    def model_label(self) -> str:
        label = MODEL_LABELS.get(self.model, self.model)
        if self.source_model:
            label += f" · {MODEL_LABELS.get(self.source_model, self.source_model)} 반주에서"
        return label


@dataclass
class ExtractResult:
    key: str
    model: str
    added: list[str] = field(default_factory=list)  # 이번에 새로 저장한 악기
    kept: list[str] = field(default_factory=list)  # 이미 있어 그대로 둔 악기
    separated: bool = False
    source_model: str = ""  # 반주에서 뽑았으면 그 반주를 만든 분리 결과의 모델


def find_stems(stems_dir: str | Path, source_sha256: str) -> dict[str, StemFile]:
    """악기 이름 → 파일. 지금 쓰는 분리 결과를 먼저, 그다음 htdemucs, 6s, 그 밖. 같은 순위면 최근 결과."""
    entries = ModelStore(stems_dir, source_sha256).entries()
    entries.sort(key=lambda e: e.manifest.created, reverse=True)
    entries.sort(key=lambda e: (not e.active, _PREFERENCE.get(e.model, 2)))
    out: dict[str, StemFile] = {}
    for name in INSTRUMENTS:
        for e in entries:
            p = stem_store.optional_path(e.path, e.manifest, name)
            if p is not None:
                meta = e.manifest.files.get(name, {})
                from_inst = meta.get("input") == "instrumental"
                out[name] = StemFile(name, e.key, str(meta.get("made_by") or e.model), p,
                                     e.model if from_inst else "")
                break
    return out


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
    have = [n for n in names if existing is not None
            and stem_store.optional_path(existing.path, existing.manifest, n) is not None]
    missing = [n for n in names if n not in have]
    if existing is not None and not missing:
        report(1.0, "이미 분리해 둔 악기 소리가 있습니다")
        return ExtractResult(existing.key, model, [], have, False)
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
                store.add_stem(existing.key, n, tmp / f"{n}.wav")
            key = existing.key
        else:
            key = store.commit(tmp, identity, {"device": device, "purpose": "instruments"}).key
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            store.root.rmdir()
        except OSError:
            pass
    return ExtractResult(key, model, got, have, True)


def _extract_from_instrumental(stems_dir, source_sha256, model, runner, report, token, device, names, numba_cache_dir):
    target = instrumental_source(stems_dir, source_sha256)
    if target is None:
        raise AnalysisError("보컬 분석에 쓰는 분리 결과가 없습니다. 보컬 분석을 먼저 하거나 '원본'에서 분리하세요.")
    store = ModelStore(stems_dir, source_sha256)
    have = [n for n in names if stem_store.optional_path(target.path, target.manifest, n) is not None]
    missing = [n for n in names if n not in have]
    if not missing:
        report(1.0, "이미 분리해 둔 악기 소리가 있습니다")
        return ExtractResult(target.key, model, [], have, False, target.model)
    src = target.path / target.manifest.files["instrumental"]["path"]
    tmp = store.new_tmp()
    try:
        runner("separate", {"input": str(src), "out_dir": str(tmp), "model": model, "device": device,
                            "cpu_fallback": True, "extras": list(missing), "label": "반주에서 악기 분리"},
               report, token, numba_cache_dir)
        got = [n for n in missing if (tmp / f"{n}.wav").is_file()]
        if not got:
            raise AnalysisError(f"분리 결과에 악기 소리가 없습니다 ({MODEL_LABELS.get(model, model)}).")
        for n in got:  # 반주를 분리하며 나온 보컬·반주(남은 소리)는 버린다
            store.add_stem(target.key, n, tmp / f"{n}.wav", {"made_by": model, "input": "instrumental"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ExtractResult(target.key, model, got, have, True, target.model)
