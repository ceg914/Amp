"""분리 모델별 결과 보관 (STEP 5 모델 비교). Qt 비의존.

    stems/
        models/
            htdemucs-1a2b3c4d5e/      vocals.wav, instrumental.wav, stems.json (이 결과의 목록 + 식별 정보)
            mel_roformer-6f7e8d9c0b/  …
        stems.json                    지금 분석에 쓰는 결과를 가리키는 목록 (파일 경로가 models/<키>/…)

- 한 프로젝트에 여러 모델의 결과를 함께 둔다. 다른 모델로 분리해도 기존 결과를 덮어쓰지 않는다.
- 식별 키 = 모델 이름 + 해시(원본 해시, 모델 버전, 체크포인트 해시, 설정 파일 해시, 출력에 영향을 주는 옵션).
  같은 키의 결과가 있으면 다시 계산하지 않는다. 체크포인트나 설정이 바뀌면 다른 키가 된다.
- 위쪽 stems.json은 기존 형식 그대로라 재생·파형·가사 인식·내보내기 코드가 바뀌지 않는다. 경로는 모두 상대 경로라
  프로젝트 폴더를 옮기거나 복사해도 된다.
- 예전 프로젝트의 평면 결과(stems/vocals.wav)는 처음 쓸 때 모델 폴더로 옮긴다(덮어쓰지 않음).
- 결과 폴더는 사용자가 고를 때만 지운다. 지금 쓰는 결과는 지우지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from amp.core.fileio import atomic_write_json
from amp.engine import stems as stem_store

MODELS_DIR = "models"
POINTER = "store_key"
ROFORMER_CODE_VERSION = "1"
MODEL_LABELS = {"htdemucs": "Demucs htdemucs", "mel_roformer": "Mel-Band RoFormer", "external": "가져온 결과"}
_SHA_CACHE: dict[tuple, str] = {}


def file_sha256(path: str) -> str:
    """큰 체크포인트도 한 번만 해시한다 (경로·크기·수정 시각이 같으면 캐시)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - 설치 안 됨
        return ""


def identity_for(model: str, source_sha256: str, model_files: dict | None = None) -> dict:
    """결과를 구분하는 값. GUI 프로세스에서도 부를 수 있다(torch를 불러오지 않음)."""
    ident = {"model": model, "source": source_sha256}
    if model == "htdemucs":
        from amp.engine.dsp.separation import SHIFT_SEED

        ident.update(demucs=package_version("demucs"), shifts=1, overlap=0.25, split=True, shift_seed=SHIFT_SEED,
                     normalize="mean-std")
    elif model == "mel_roformer":
        files = model_files or {}
        ck = files.get("checkpoint") or ""
        ident.update(code=ROFORMER_CODE_VERSION, checkpoint_sha256=files.get("checkpoint_sha256") or
                     (file_sha256(ck) if ck and os.path.isfile(ck) else ""),
                     config_sha256=file_sha256(files["config"]) if files.get("config") and os.path.isfile(files["config"]) else "")
    elif model.startswith("_"):
        ident.update(test=True)
    return ident


def store_key(identity: dict) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(identity.get("model", "x"))).strip("_") or "x"
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    return f"{safe}-{digest}"


@dataclass(frozen=True)
class StoreEntry:
    key: str
    path: Path
    manifest: stem_store.StemManifest
    size_bytes: int
    active: bool

    @property
    def model(self) -> str:
        return self.manifest.model

    @property
    def label(self) -> str:
        base = MODEL_LABELS.get(self.model, self.model)
        info = self.manifest.model_info
        if self.model == "mel_roformer" and info.get("checkpoint"):
            base += f" ({info['checkpoint']})"
        if self.model == "external" and info.get("tool"):
            base += f" ({info['tool']})"
        return base

    @property
    def identity(self) -> dict:
        return dict(self.manifest.model_info.get("identity") or {})


class ModelStore:
    def __init__(self, stems_dir: str | os.PathLike, source_sha256: str) -> None:
        self.stems_dir = Path(stems_dir)
        self.root = self.stems_dir / MODELS_DIR
        self.source_sha256 = source_sha256

    # --- 조회 -------------------------------------------------------------------

    def pointer(self) -> stem_store.StemManifest | None:
        return stem_store.read_manifest(self.stems_dir) if self.stems_dir.exists() else None

    def active_key(self) -> str | None:
        m = self.pointer()
        return None if m is None else m.model_info.get(POINTER)

    def entries(self) -> list[StoreEntry]:
        out = []
        if not self.root.is_dir():
            return out
        active = self.active_key()
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            m = stem_store.read_manifest(d)
            ok, _why = stem_store.validate(d, m, self.source_sha256)
            if not ok:
                continue
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            out.append(StoreEntry(d.name, d, m, size, d.name == active))
        return out

    def entry(self, key: str) -> StoreEntry | None:
        return next((e for e in self.entries() if e.key == key), None)

    def find(self, identity: dict) -> StoreEntry | None:
        """같은 결과로 볼 수 있는 보관 결과. 키가 같으면 그것, 아니면 이전 버전에서 옮겨 온 결과 중 같은 원본·모델·
        (Demucs 버전 / RoFormer 체크포인트 내용)인 것. 옮겨 온 결과는 만들 때 옵션을 다 적지 않아 키가 다르다."""
        key = store_key(identity)
        entries = self.entries()
        hit = next((e for e in entries if e.key == key), None)
        if hit is not None:
            return hit
        for e in entries:
            old = e.identity
            if "legacy_created" not in old or old.get("model") != identity.get("model") \
                    or old.get("source") != identity.get("source"):
                continue
            if identity.get("model") == "htdemucs" and old.get("demucs", identity.get("demucs")) == identity.get("demucs"):
                return e
            if identity.get("model") == "mel_roformer" and old.get("checkpoint_sha256") \
                    and old.get("checkpoint_sha256") == identity.get("checkpoint_sha256") and not identity.get("config_sha256"):
                return e
        return None

    # --- 쓰기 -------------------------------------------------------------------

    def new_tmp(self) -> Path:
        import uuid

        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        return tmp

    def commit(self, tmp: Path, identity: dict, settings: dict) -> StoreEntry:
        """작업 프로세스가 tmp에 쓴 결과(info.json 포함)를 키 폴더로 옮긴다. 같은 키가 있으면 그 결과만 바꾼다."""
        info_path = tmp / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["identity"] = identity
        info_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        key = store_key(identity)
        dest = self.root / key
        stem_store.commit(tmp, dest, self.source_sha256, settings)  # 목록을 먼저 지우고 파일을 옮긴 뒤 새 목록
        return self.entry(key)

    def view(self, key: str) -> stem_store.StemManifest | None:
        """stems/ 기준 경로로 바꾼 이 결과의 목록 (파일에 쓰지 않음). 비교 분석이 지금 쓰는 결과를 바꾸지 않고 읽을 때."""
        e = self.entry(key)
        if e is None:
            return None
        files = {n: {**v, "path": f"{MODELS_DIR}/{key}/{v['path']}"} for n, v in e.manifest.files.items()}
        return replace(e.manifest, files=files, model_info={**e.manifest.model_info, POINTER: key})

    def activate(self, key: str) -> stem_store.StemManifest:
        """이 결과를 분석에 쓴다: 위쪽 stems.json이 models/<키>/ 파일을 가리키게 한다. 다른 결과 파일은 그대로."""
        m = self.view(key)
        if m is None:
            raise FileNotFoundError(f"분리 결과 {key}가 없습니다.")
        atomic_write_json(self.stems_dir / stem_store.MANIFEST, m.to_dict())
        return m

    def migrate_legacy(self) -> str | None:
        """예전 평면 결과(stems/vocals.wav + stems.json)를 모델 폴더로 옮겨 보관한다. 옮긴 키 또는 None."""
        m = self.pointer()
        if m is None or m.model_info.get(POINTER):
            return None
        ok, _why = stem_store.validate(self.stems_dir, m, self.source_sha256)
        if not ok:
            return None
        info = m.model_info
        ident = {"model": m.model, "source": m.source_sha256, "legacy_created": m.created}
        for k in ("checkpoint_sha256", "vocals_sha256", "instrumental_sha256", "demucs"):
            if info.get(k):
                ident[k] = info[k]
        key = store_key(ident)
        dest = self.root / key
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, v in m.files.items():
            src = self.stems_dir / v["path"]
            os.replace(src, dest / Path(v["path"]).name)  # 같은 디스크 안 이동 (복사 없이, 덮어쓰지 않음)
            files[name] = {**v, "path": Path(v["path"]).name}
        entry = replace(m, files=files, model_info={**info, "identity": ident, "migrated_from": "stems/"})
        atomic_write_json(dest / stem_store.MANIFEST, entry.to_dict())
        self.activate(key)
        return key

    def delete(self, key: str) -> None:
        if key == self.active_key():
            raise ValueError("지금 분석에 쓰는 분리 결과는 지울 수 없습니다. 다른 결과를 먼저 고르세요.")
        d = self.root / key
        if d.is_dir() and d.parent == self.root:
            shutil.rmtree(d)
