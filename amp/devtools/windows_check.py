"""v0.7.0 Windows 점검 도구. 사용자 PC(Windows + NVIDIA)에서 실행해 결과 파일을 보내 받는다.

    python scripts/windows_check.py            # 모델 등록 정보·GPU·프로필 확인, 결과 파일 쓰기
    python scripts/windows_check.py --hash     # RoFormer 체크포인트 해시도 계산 (큰 파일이면 수십 초)

하는 일:
- 운영체제·Python·패키지 버전
- 모델 등록 정보(설치 여부·파일 위치·크기·해시·라이선스·상태). 앱이 저장한 RoFormer 경로를 읽기만 한다
- torch·CUDA 확인은 별도 프로세스에서 (이 도구에 CUDA 컨텍스트가 생기지 않게)
- Lite·Standard·Advanced에서 켤 수 있는 기능과 꺼진 이유
- 결과를 sonatlas_windows_check_<시각>.json / .txt로 이 폴더에 쓴다

모델 파일을 내려받거나 바꾸지 않는다. 프로젝트 파일을 읽지 않는다. 결과 파일에는 모델 파일 경로가 들어 있으니
보내기 전에 확인하세요.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from amp.devtools.gpu_check import _PROBE, _cli_app, _no_window, nvidia_smi


def _probe() -> dict:
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="replace", **_no_window())
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"error": r.stderr[-500:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _stored_model_files() -> dict:
    try:
        from PySide6.QtCore import QSettings

        from amp.app.main_window import MODEL_FILES_KEY

        v = QSettings().value(MODEL_FILES_KEY)
        return {k: str(v.get(k) or "") for k in ("checkpoint", "config")} if isinstance(v, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def collect(hash_files: bool = False, probe: bool = True) -> dict:
    from amp import __version__
    from amp.app.lite import lite_mode
    from amp.engine import model_registry as R
    from amp.engine import profiles as P

    try:
        from amp.app.session import app_data_dir

        models_dir, reg_path = app_data_dir("models"), app_data_dir(R.FILE_NAME)
    except Exception:  # noqa: BLE001
        models_dir, reg_path = "", ""
    ctx = R.RegistryContext(models_dir=models_dir, roformer_files=_stored_model_files(), lite=lite_mode(),
                            hash_cache=R.load_hash_cache(reg_path) if reg_path else {})
    statuses = R.scan(ctx, hash_files=hash_files)
    if reg_path:
        try:
            R.save(reg_path, statuses, ctx)
        except OSError:
            pass
    rep = {"tool": "windows_check", "sonatlas": __version__, "created": datetime.now().isoformat(timespec="seconds"),
           "os": platform.platform(), "python": sys.version.split()[0], "lite_mode": ctx.lite,
           "models": [s.to_dict() for s in statuses],
           "profiles": {p: {fid: {"enabled": fs.enabled, "reason": fs.reason, "models": list(fs.models)}
                            for fid, fs in P.resolve(p, statuses, ctx.lite).items()} for p in P.PROFILES}}
    if probe:
        rep["torch_probe"] = _probe()
        rep["nvidia_smi"] = nvidia_smi()
    return rep


def text_report(rep: dict) -> str:
    from amp.engine import model_registry as R

    lines = [f"SONATLAS {rep['sonatlas']} Windows 점검 ({rep['created']})", f"운영체제: {rep['os']}",
             f"Python: {rep['python']} · 가벼운 모드: {'예' if rep['lite_mode'] else '아니요'}", ""]
    pr = rep.get("torch_probe") or {}
    if pr:
        if pr.get("cuda_available"):
            lines.append(f"GPU: {pr.get('gpu_name')} · 남은 메모리 {pr.get('gpu_free_mb')}/{pr.get('gpu_total_mb')}MB · "
                         f"torch {pr.get('torch')} (CUDA {pr.get('torch_cuda')})")
        elif pr.get("torch"):
            lines.append(f"GPU: PyTorch가 CUDA를 쓰지 못함 · torch {pr.get('torch')} (CUDA {pr.get('torch_cuda')})")
        else:
            lines.append(f"GPU: 확인 못 함 ({pr.get('torch_error') or pr.get('error') or 'PyTorch 없음'})")
    lines += ["", "모델"]
    for m in rep["models"]:
        size = f"{m['size_bytes'] / 2**20:.0f}MB" if m.get("size_bytes") else "-"
        lines.append(f"- {m['name']}: {R.STATE_LABELS.get(m['state'], m['state'])} · {m.get('version') or '-'} · {size}"
                     + (f" · {m['reason']}" if m.get("reason") else ""))
        if m.get("path"):
            lines.append(f"    위치: {m['path']}")
        if m.get("checkpoint_sha256"):
            lines.append(f"    sha256: {m['checkpoint_sha256']}")
    for p, feats in rep["profiles"].items():
        on = [f for f, s in feats.items() if s["enabled"]]
        lines += ["", f"{p} 프로필: 켤 수 있음 {len(on)}/{len(feats)}"]
        for f, s in feats.items():
            if not s["enabled"]:
                lines.append(f"  꺼짐 {f}: {s['reason']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SONATLAS v0.7.0 Windows 점검")
    ap.add_argument("--hash", action="store_true", help="RoFormer 체크포인트 해시 계산")
    ap.add_argument("--no-probe", action="store_true", help="torch·CUDA 확인을 건너뜀")
    ap.add_argument("--out", default=".", help="결과 파일을 쓸 폴더")
    ns = ap.parse_args(argv)
    _app = _cli_app()  # noqa: F841 - 앱과 같은 설정·모델 폴더를 읽기 위해
    rep = collect(hash_files=ns.hash, probe=not ns.no_probe)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sonatlas_windows_check_{stamp}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                              encoding="utf-8")
    text = text_report(rep)
    (out / f"sonatlas_windows_check_{stamp}.txt").write_text(text, encoding="utf-8-sig")
    print(text)
    print(f"결과 파일: {out / f'sonatlas_windows_check_{stamp}.txt'}")
    return 0
