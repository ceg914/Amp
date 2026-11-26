"""모델 등록 정보 (v0.7.0, 명세 13.2·13.3). Qt 비의존, torch를 불러오지 않는다.

프로그램 전체에서 쓰는 분석 모델의 설치 여부·파일 위치·버전·크기·체크포인트 해시·실행 장치·의존성·라이선스·
지원 기능·상태를 한곳에서 본다. 프로젝트 파일과는 따로 앱 데이터 폴더의 ``model_registry.json``에 적는다.

- 패키지 정보는 ``importlib.metadata``로만 읽는다 (패키지를 불러오지 않음). 라이선스는 패키지 메타데이터에 적힌 값이다.
  가중치 라이선스는 패키지 라이선스와 다를 수 있어 따로 적고, 확인하지 않은 것은 '확인하지 않음'으로 둔다
- 사용자가 고른 Mel-Band RoFormer 체크포인트는 경로·크기만 읽는다. 해시는 요청할 때만 계산하고(큰 파일),
  같은 경로·크기·수정 시각이면 저장해 둔 해시를 쓴다. 파일을 바꾸거나 옮기거나 내려받지 않는다
- 상태: available(쓸 수 있음) · download_on_first_use(처음 쓸 때 패키지가 가중치를 받음) · not_installed ·
  file_missing · invalid · unavailable(가벼운 모드 등). 한 모델이 안 돼도 다른 기능은 그대로 쓴다 (13.3)
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sonatlas.model_registry"
SCHEMA_VERSION = 1
FILE_NAME = "model_registry.json"

AVAILABLE = "available"
DOWNLOAD_ON_FIRST_USE = "download_on_first_use"
NOT_INSTALLED = "not_installed"
FILE_MISSING = "file_missing"
INVALID = "invalid"
UNAVAILABLE = "unavailable"
STATES = (AVAILABLE, DOWNLOAD_ON_FIRST_USE, NOT_INSTALLED, FILE_MISSING, INVALID, UNAVAILABLE)
USABLE = (AVAILABLE, DOWNLOAD_ON_FIRST_USE)
STATE_LABELS = {AVAILABLE: "사용 가능", DOWNLOAD_ON_FIRST_USE: "처음 쓸 때 가중치를 받음", NOT_INSTALLED: "설치 안 됨",
                FILE_MISSING: "파일 없음", INVALID: "파일 오류", UNAVAILABLE: "이 모드에서 사용 안 함"}


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    name: str
    model_type: str  # separation · pitch · asr · alignment
    features: tuple[str, ...]
    packages: tuple[str, ...]  # 필요한 배포 패키지 (pip 이름)
    modules: tuple[str, ...]  # 확인할 import 이름 (불러오지 않고 find_spec만)
    devices: tuple[str, ...] = ("cpu",)
    weights: str = ""  # 가중치가 어디서 오는지
    weights_license: str = "확인하지 않음"
    install_hint: str = ""  # 설치 스크립트 키 (core.install_hints)
    frequency_range: tuple[float, float] | None = None
    experimental: bool = False
    lite: bool = False  # 가벼운 모드에서도 쓰는 모델
    code_license: str = ""  # 패키지가 아니라 SONATLAS 안의 구현을 쓰면 그 라이선스 (비우면 패키지 메타데이터)


CATALOG: dict[str, ModelInfo] = {m.model_id: m for m in (
    ModelInfo("pyin", "pYIN (librosa)", "pitch", ("mono_pitch",), ("librosa",), ("librosa",),
              weights="학습 가중치 없음 (신호 처리)", weights_license="해당 없음", frequency_range=(55.0, 1400.0), lite=True),
    ModelInfo("htdemucs", "Demucs htdemucs", "separation", ("vocals", "drums", "bass", "other"),
              ("demucs", "torch"), ("demucs", "torch"), ("cpu", "cuda"),
              weights="Demucs가 처음 쓸 때 내려받는 파일 (torch hub 폴더)", install_hint="separation"),
    ModelInfo("htdemucs_6s", "Demucs htdemucs_6s", "separation", ("vocals", "drums", "bass", "guitar", "piano", "other"),
              ("demucs", "torch"), ("demucs", "torch"), ("cpu", "cuda"),
              weights="Demucs가 처음 쓸 때 내려받는 파일 (torch hub 폴더)", install_hint="separation"),
    ModelInfo("mel_roformer", "Mel-Band RoFormer", "separation", ("vocals",), ("torch",), ("torch",), ("cpu", "cuda"),
              weights="사용자가 고른 체크포인트 파일", weights_license="사용자 파일 (확인하지 않음)", install_hint="separation",
              code_license="MIT (옮긴 구현, amp/engine/dsp/LICENSE-roformer.txt)"),
    ModelInfo("whisper", "OpenAI Whisper", "asr", ("lyrics_recognition",), ("openai-whisper", "torch"),
              ("whisper", "torch"), ("cpu", "cuda"), weights="처음 쓸 때 앱 모델 폴더에 내려받음", install_hint="lyrics"),
    ModelInfo("mms_fa", "MMS 강제 정렬 (wav2vec2)", "alignment", ("lyrics_alignment",), ("torch",), ("torch",),
              ("cpu", "cuda"), weights="처음 쓸 때 앱 모델 폴더에 내려받음 (약 1.2GB)", install_hint="lyrics",
              code_license="SONATLAS 구현 (amp/lyrics/mms.py)"),
    ModelInfo("crepe_full", "CREPE full", "pitch", ("mono_pitch",), ("torchcrepe", "torch"), ("torch",),
              ("cpu", "cuda"), weights="torchcrepe 패키지 안의 full.pth", weights_license="torchcrepe 패키지 라이선스와 같음",
              install_hint="pitch", frequency_range=(32.7, 1975.5), experimental=True),
    ModelInfo("crepe_tiny", "CREPE tiny", "pitch", ("mono_pitch",), ("torchcrepe", "torch"), ("torch",),
              ("cpu", "cuda"), weights="torchcrepe 패키지 안의 tiny.pth", weights_license="torchcrepe 패키지 라이선스와 같음",
              install_hint="pitch", frequency_range=(32.7, 1975.5), experimental=True),
    ModelInfo("fcpe", "FCPE", "pitch", ("mono_pitch",), ("torchfcpe", "torch"), ("torch",), ("cpu", "cuda"),
              weights="torchfcpe 패키지 안의 fcpe_c_v001.pt", weights_license="torchfcpe 패키지 라이선스와 같음",
              install_hint="pitch", experimental=True),
)}


@dataclass
class ModelStatus:
    model_id: str
    name: str
    model_type: str
    state: str
    reason: str = ""
    installed: bool = False
    version: str = ""  # 패키지 버전 (여럿이면 이름=버전)
    path: str = ""
    size_bytes: int | None = None
    checkpoint_sha256: str = ""
    devices: tuple[str, ...] = ("cpu",)
    dependencies: dict = field(default_factory=dict)  # 패키지 → 버전 또는 ""
    license: str = ""  # 패키지 라이선스 (메타데이터)
    weights_license: str = ""
    features: tuple[str, ...] = ()
    experimental: bool = False

    @property
    def usable(self) -> bool:
        return self.state in USABLE

    @property
    def state_label(self) -> str:
        return STATE_LABELS.get(self.state, self.state)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["devices"] = list(self.devices)
        d["features"] = list(self.features)
        return d


@dataclass
class RegistryContext:
    """모델 파일을 찾을 곳. 앱은 session.app_data_dir로 채우고, 시험은 임시 폴더를 준다."""

    models_dir: str = ""  # 앱 모델 폴더 (whisper/, mms/)
    roformer_files: dict = field(default_factory=dict)  # {"checkpoint", "config", "checkpoint_sha256"?}
    torch_hub_dir: str = ""  # 비우면 TORCH_HOME 또는 ~/.cache/torch 아래 hub
    lite: bool = False
    hash_cache: dict = field(default_factory=dict)  # 경로 → {"size", "mtime_ns", "sha256"} (저장해 둔 해시)


def _dist_version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def package_license(dist: str) -> str:
    """패키지 메타데이터에 적힌 라이선스 (License-Expression → License → 분류자). 없으면 ''."""
    try:
        m = metadata.metadata(dist)
    except metadata.PackageNotFoundError:
        return ""
    expr = (m.get("License-Expression") or "").strip()
    if expr:
        return expr
    text = (m.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 60:
        return text
    for c in m.get_all("Classifier") or []:
        if c.startswith("License ::"):
            return c.split("::")[-1].strip()
    return text.splitlines()[0][:60] if text else ""


def _package_dir(module: str, find_spec) -> Path | None:
    try:
        spec = find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def torch_hub_checkpoints(ctx: RegistryContext) -> Path:
    if ctx.torch_hub_dir:
        return Path(ctx.torch_hub_dir) / "checkpoints"
    home = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "torch")
    return Path(home) / "hub" / "checkpoints"


def demucs_weight_files(model: str, find_spec=importlib.util.find_spec) -> list[str]:
    """demucs 패키지의 remote/<모델>.yaml·files.txt에서 가중치 파일 이름을 읽는다 (불러오지 않음)."""
    d = _package_dir("demucs", find_spec)
    if d is None:
        return []
    try:
        yaml_text = (d / "remote" / f"{model}.yaml").read_text(encoding="utf-8")
        files = (d / "remote" / "files.txt").read_text(encoding="utf-8").split()
    except OSError:
        return []
    sigs = []
    for line in yaml_text.splitlines():
        if line.strip().startswith("models:"):
            sigs = [s.strip(" '\"") for s in line.split(":", 1)[1].strip(" []").split(",") if s.strip(" '\"")]
    return [f for s in sigs for f in files if f.startswith(s + "-")]


def cached_sha256(path: str, ctx: RegistryContext, compute: bool) -> str:
    """저장해 둔 해시가 경로·크기·수정 시각과 맞으면 그것. compute=True면 새로 계산해 ctx.hash_cache에 넣는다."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    hit = ctx.hash_cache.get(os.path.abspath(path))
    if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns and hit.get("sha256"):
        return str(hit["sha256"])
    if not compute:
        return ""
    from amp.engine.model_store import file_sha256

    sha = file_sha256(path)
    ctx.hash_cache[os.path.abspath(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
    return sha


def _status(info: ModelInfo, ctx: RegistryContext, find_spec, hash_files: bool, version_of=None) -> ModelStatus:
    version_of = version_of or _dist_version
    deps = {p: version_of(p) for p in info.packages}
    st = ModelStatus(info.model_id, info.name, info.model_type, NOT_INSTALLED, devices=info.devices, dependencies=deps,
                     license=info.code_license
                     or next((package_license(p) for p in info.packages if p != "torch" and deps.get(p)), ""),
                     weights_license=info.weights_license, features=info.features, experimental=info.experimental)
    st.version = " · ".join(f"{p} {v}" for p, v in deps.items() if v)
    missing_mod = [m for m in info.modules if _find(m, find_spec) is None]
    missing_pkg = [p for p, v in deps.items() if not v]
    if ctx.lite and not info.lite:
        st.state, st.reason = UNAVAILABLE, "가벼운 모드에서는 쓰지 않습니다."
        st.installed = not missing_mod and not missing_pkg
        return st
    if missing_mod or missing_pkg:
        st.reason = f"설치되지 않은 패키지: {', '.join(missing_pkg or missing_mod)}"
        return st
    st.installed = True
    files: list[Path] = []
    if info.model_id in ("htdemucs", "htdemucs_6s"):
        names = demucs_weight_files(info.model_id, find_spec)
        hub = torch_hub_checkpoints(ctx)
        files = [hub / n for n in names]
        if names and all(f.is_file() for f in files):
            st.state = AVAILABLE
        else:
            st.state, st.reason = DOWNLOAD_ON_FIRST_USE, f"가중치가 아직 없습니다 ({hub}). 처음 분리할 때 Demucs가 내려받습니다."
    elif info.model_id == "mel_roformer":
        ck = str(ctx.roformer_files.get("checkpoint") or "")
        if not ck:
            st.state, st.reason = FILE_MISSING, "체크포인트 파일을 고르지 않았습니다."
        elif not os.path.isfile(ck):
            st.state, st.reason, st.path = FILE_MISSING, f"체크포인트 파일이 없습니다: {ck}", ck
        else:
            files = [Path(ck)]
            st.state = AVAILABLE
            known = str(ctx.roformer_files.get("checkpoint_sha256") or "")  # 고를 때 계산한 값 (있으면)
            sha = cached_sha256(ck, ctx, hash_files)
            if known and sha and known != sha:
                st.state, st.reason = INVALID, "체크포인트 내용이 처음 고를 때와 다릅니다."
    elif info.model_id == "whisper":
        d = Path(ctx.models_dir) / "whisper" if ctx.models_dir else None
        files = sorted(d.glob("*.pt")) if d is not None and d.is_dir() else []
        st.state = AVAILABLE if files else DOWNLOAD_ON_FIRST_USE
        if not files:
            st.reason = "모델 파일이 아직 없습니다. 처음 인식할 때 내려받습니다."
    elif info.model_id == "mms_fa":
        from amp.lyrics import mms

        d = os.path.join(ctx.models_dir, "mms") if ctx.models_dir else ""
        p = Path(mms.model_path(d)) if d else None
        if p is not None and p.is_file():
            files = [p]
            st.state = AVAILABLE if mms.is_downloaded(d) else INVALID
            if st.state == INVALID:
                st.reason = "모델 파일 크기가 예상과 다릅니다 (내려받다 끊겼을 수 있음)."
        else:
            st.state, st.reason = DOWNLOAD_ON_FIRST_USE, "모델 파일이 아직 없습니다. 처음 정렬할 때 내려받습니다."
    elif info.model_id in ("crepe_full", "crepe_tiny", "fcpe"):
        pkg, fname = {"crepe_full": ("torchcrepe", "full.pth"), "crepe_tiny": ("torchcrepe", "tiny.pth"),
                      "fcpe": ("torchfcpe", "fcpe_c_v001.pt")}[info.model_id]
        d = _package_dir(pkg, find_spec)
        p = d / "assets" / fname if d is not None else None
        if p is not None and p.is_file():
            files = [p]
            st.state = AVAILABLE
        else:
            st.state, st.reason = FILE_MISSING, f"{pkg} 패키지에 가중치 파일({fname})이 없습니다."
    else:
        st.state = AVAILABLE
    if files:
        st.path = str(files[0]) if len(files) == 1 else str(files[0].parent)
        st.size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        if len(files) == 1 and not st.checkpoint_sha256:
            st.checkpoint_sha256 = cached_sha256(str(files[0]), ctx, hash_files)
    return st


def _find(module: str, find_spec):
    try:
        return find_spec(module)
    except (ImportError, ValueError):
        return None


def scan(ctx: RegistryContext | None = None, find_spec=importlib.util.find_spec, hash_files: bool = False,
         ids: tuple[str, ...] | None = None, version_of=None) -> list[ModelStatus]:
    """등록된 모델의 지금 상태. 한 모델을 살피다 오류가 나도 그 모델만 invalid로 적는다 (13.3)."""
    ctx = ctx or RegistryContext()
    out = []
    for mid, info in CATALOG.items():
        if ids is not None and mid not in ids:
            continue
        try:
            out.append(_status(info, ctx, find_spec, hash_files, version_of))
        except Exception as e:  # noqa: BLE001 - 등록 정보는 다른 모델을 막지 않는다
            out.append(ModelStatus(mid, info.name, info.model_type, INVALID, reason=f"상태를 확인하지 못했습니다: {e}",
                                   devices=info.devices, features=info.features, experimental=info.experimental))
    return out


def status_of(statuses: list[ModelStatus], model_id: str) -> ModelStatus | None:
    return next((s for s in statuses if s.model_id == model_id), None)


def to_json(statuses: list[ModelStatus], ctx: RegistryContext | None = None) -> dict:
    return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": [s.to_dict() for s in statuses],
            "hash_cache": dict((ctx.hash_cache if ctx else {}) or {})}


def save(path: str | Path, statuses: list[ModelStatus], ctx: RegistryContext | None = None) -> None:
    from amp.core.fileio import atomic_write_json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(path), to_json(statuses, ctx))


def load_hash_cache(path: str | Path) -> dict:
    """저장한 등록 정보에서 해시 캐시만 읽는다. 파일이 없거나 깨졌으면 빈 값 (다시 계산하면 됨)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        return {}
    cache = d.get("hash_cache")
    return {str(k): dict(v) for k, v in cache.items() if isinstance(v, dict)} if isinstance(cache, dict) else {}
