"""v0.7.0 Phase 1: 모델 등록 정보·분석 프로필·결과 머리말. torch를 불러오지 않고, 모델 파일을 바꾸거나 내려받지 않는다."""

from __future__ import annotations

import importlib.machinery
import json
import sys

import pytest

from amp.core import result_meta as M
from amp.engine import model_registry as R
from amp.engine import profiles as P


def fake_env(tmp_path, installed: set[str], versions: dict[str, str] | None = None):
    """find_spec·패키지 버전을 흉내 낸다. 패키지 폴더는 tmp_path/pkgs/<이름>."""
    pkgs = tmp_path / "pkgs"

    def find_spec(name):
        if name not in installed:
            return None
        d = pkgs / name
        d.mkdir(parents=True, exist_ok=True)
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(d)]
        return spec

    vers = versions or {}
    dist_of = {"openai-whisper": "whisper"}

    def version_of(dist):
        mod = dist_of.get(dist, dist)
        return vers.get(dist, "1.0") if mod in installed else ""

    return pkgs, find_spec, version_of


def test_scan_without_heavy_packages(tmp_path):
    _pkgs, find_spec, version_of = fake_env(tmp_path, {"librosa"})
    before = "torch" in sys.modules
    st = {s.model_id: s for s in R.scan(R.RegistryContext(models_dir=str(tmp_path / "m")), find_spec,
                                        version_of=version_of)}
    assert set(st) == set(R.CATALOG)
    assert st["pyin"].state == R.AVAILABLE and st["pyin"].usable
    for mid in ("htdemucs", "mel_roformer", "whisper", "crepe_full", "fcpe"):
        assert st[mid].state == R.NOT_INSTALLED and not st[mid].usable and "설치되지 않은" in st[mid].reason
    assert ("torch" in sys.modules) == before  # 등록 정보는 torch를 불러오지 않는다


def test_scan_reports_files_hash_and_user_checkpoint(tmp_path):
    pkgs, find_spec, version_of = fake_env(tmp_path, {"librosa", "torch", "demucs", "whisper", "torchcrepe"})
    remote = pkgs / "demucs" / "remote"
    remote.mkdir(parents=True)
    (remote / "htdemucs.yaml").write_text("models: ['955717e8']\n", encoding="utf-8")
    (remote / "htdemucs_6s.yaml").write_text("models: ['5c90dfd2']\n", encoding="utf-8")
    (remote / "files.txt").write_text("955717e8-8726e21a.th\n5c90dfd2-34c22ccb.th\n", encoding="utf-8")
    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / "955717e8-8726e21a.th").write_bytes(b"x" * 10)
    (pkgs / "torchcrepe" / "assets").mkdir(parents=True)
    (pkgs / "torchcrepe" / "assets" / "tiny.pth").write_bytes(b"t" * 7)
    ck = tmp_path / "사용자 모델" / "MelBandRoformer.ckpt"
    ck.parent.mkdir()
    ck.write_bytes(b"c" * 33)
    before = ck.read_bytes(), ck.stat().st_mtime_ns
    ctx = R.RegistryContext(models_dir=str(tmp_path / "m"), roformer_files={"checkpoint": str(ck)},
                            torch_hub_dir=str(hub))
    st = {s.model_id: s for s in R.scan(ctx, find_spec, version_of=version_of)}
    assert st["htdemucs"].state == R.AVAILABLE and st["htdemucs"].size_bytes == 10
    assert st["htdemucs_6s"].state == R.DOWNLOAD_ON_FIRST_USE and st["htdemucs_6s"].usable
    assert st["whisper"].state == R.DOWNLOAD_ON_FIRST_USE
    assert st["crepe_tiny"].state == R.AVAILABLE and st["crepe_full"].state == R.FILE_MISSING
    rof = st["mel_roformer"]
    assert rof.state == R.AVAILABLE and rof.path == str(ck) and rof.size_bytes == 33 and rof.checkpoint_sha256 == ""
    assert "MIT" in rof.license and "사용자 파일" in rof.weights_license
    hashed = {s.model_id: s for s in R.scan(ctx, find_spec, hash_files=True, version_of=version_of)}
    import hashlib

    assert hashed["mel_roformer"].checkpoint_sha256 == hashlib.sha256(b"c" * 33).hexdigest()
    assert (ck.read_bytes(), ck.stat().st_mtime_ns) == before  # 사용자 체크포인트는 읽기만

    reg = tmp_path / "app" / R.FILE_NAME
    R.save(reg, list(hashed.values()), ctx)
    d = json.loads(reg.read_text(encoding="utf-8"))
    assert d["schema"] == R.SCHEMA and {m["model_id"] for m in d["models"]} == set(R.CATALOG)
    ctx2 = R.RegistryContext(roformer_files={"checkpoint": str(ck)}, hash_cache=R.load_hash_cache(reg))
    again = R.status_of(R.scan(ctx2, find_spec, ids=("mel_roformer",), version_of=version_of), "mel_roformer")
    assert again.checkpoint_sha256 == hashed["mel_roformer"].checkpoint_sha256  # 다시 계산하지 않고 저장값

    ck.write_bytes(b"d" * 33)  # 고를 때와 내용이 다르면 파일 오류로 알린다 (파일은 그대로 둔다)
    ctx3 = R.RegistryContext(roformer_files={"checkpoint": str(ck),
                                             "checkpoint_sha256": hashed["mel_roformer"].checkpoint_sha256})
    bad = R.status_of(R.scan(ctx3, find_spec, hash_files=True, version_of=version_of), "mel_roformer")
    assert bad.state == R.INVALID and not bad.usable
    missing = R.RegistryContext(roformer_files={"checkpoint": str(tmp_path / "없음.ckpt")})
    assert R.status_of(R.scan(missing, find_spec, version_of=version_of), "mel_roformer").state == R.FILE_MISSING


def test_scan_isolates_errors(tmp_path, monkeypatch):
    _pkgs, find_spec, version_of = fake_env(tmp_path, {"librosa", "torch", "demucs"})

    def boom(*a, **k):
        raise RuntimeError("깨진 패키지")

    monkeypatch.setattr(R, "demucs_weight_files", boom)
    st = {s.model_id: s for s in R.scan(R.RegistryContext(), find_spec, version_of=version_of)}
    assert st["htdemucs"].state == R.INVALID and "깨진 패키지" in st["htdemucs"].reason
    assert st["pyin"].usable  # 한 모델의 오류가 다른 모델을 막지 않는다


def test_lite_context_marks_heavy_models_unavailable(tmp_path):
    _pkgs, find_spec, version_of = fake_env(tmp_path, {"librosa", "torch", "demucs", "whisper"})
    st = {s.model_id: s for s in R.scan(R.RegistryContext(lite=True), find_spec, version_of=version_of)}
    assert st["pyin"].usable and st["htdemucs"].state == R.UNAVAILABLE and st["htdemucs"].installed


def test_profiles_follow_installed_models(tmp_path):
    _pkgs, find_spec, version_of = fake_env(tmp_path, {"librosa"})
    statuses = R.scan(R.RegistryContext(), find_spec, version_of=version_of)
    lite = P.resolve(P.LITE, statuses)
    assert lite["mono_pitch"].enabled and lite["vibrato"].enabled and lite["breath"].enabled
    assert not lite["separation"].enabled and "Standard" in lite["separation"].reason
    std = P.resolve(P.STANDARD, statuses)
    assert not std["separation"].enabled and "설치 안 됨" in std["separation"].reason
    assert not std["multi_pitch"].enabled
    adv = P.resolve(P.ADVANCED, statuses)
    assert adv["multi_pitch"].enabled and adv["voice_tracking"].enabled  # 신호 처리만 쓰는 기능
    assert not adv["external_pitch"].enabled and not adv["pitch_compare"].enabled
    assert "2개 이상" in adv["pitch_compare"].reason
    assert P.resolve(P.ADVANCED, statuses, lite_mode=True)["multi_pitch"].enabled is False  # 가벼운 모드 = Lite

    pkgs, find_spec, version_of = fake_env(tmp_path / "b", {"librosa", "torch", "demucs", "torchcrepe"})
    (pkgs / "torchcrepe" / "assets").mkdir(parents=True)
    (pkgs / "torchcrepe" / "assets" / "full.pth").write_bytes(b"f")
    adv = P.resolve(P.ADVANCED, R.scan(R.RegistryContext(torch_hub_dir=str(tmp_path / "h")), find_spec,
                                       version_of=version_of))
    assert adv["external_pitch"].enabled and adv["external_pitch"].models == ("crepe_full",)
    assert adv["pitch_compare"].enabled and adv["separation"].enabled
    assert P.FEATURES["voice_tracking"].depends == ("multi_pitch",)


def test_result_meta_roundtrip_and_stale():
    sha = "a" * 64
    m = M.make_meta("sonatlas.vibrato", 1, sha, input_kind="vocals", input_id="stems:123", input_hash="b" * 64,
                    model_id="pyin", analysis_version="1", analysis_settings={"min_rate": 4.0},
                    analysis_range=(1.23456, 9.0))
    d = json.loads(json.dumps(m.to_dict()))
    back = M.meta_from_dict(d)
    assert back == m and back.analysis_range == (1.2346, 9.0) and back.status == M.COMPLETED
    assert back.analysis_source_id == "vocals:stems:123:" + "b" * 16
    orig = M.make_meta("sonatlas.vibrato", 1, sha)
    assert orig.input_hash == sha and orig.analysis_source_id != m.analysis_source_id  # 같은 모델이라도 입력이 다르면 다름
    assert M.mark_stale_if_changed(m, sha, m.analysis_source_id) is m
    stale = M.mark_stale_if_changed(m, sha, "vocals:stems:999:" + "c" * 16)
    assert stale.status == M.STALE and stale.analysis_settings == m.analysis_settings  # 결과 내용은 그대로
    assert M.mark_stale_if_changed(m, "f" * 64).status == M.STALE
    assert M.meta_from_dict({"schema": "x", "status": "weird"}).status == M.UNCERTAIN
    assert M.meta_from_dict({}) is None and M.meta_from_dict(None) is None
    with pytest.raises(ValueError):
        M.make_meta("x", 1, sha, input_kind="video")
    with pytest.raises(ValueError):
        m.with_status("done")


def test_windows_check_report_without_probe(tmp_path, monkeypatch):
    """점검 도구: 등록 정보·프로필을 글로 적는다 (여기서는 torch 확인 없이)."""
    from amp.devtools import windows_check as W

    monkeypatch.setattr(W, "_stored_model_files", lambda: {"checkpoint": str(tmp_path / "없는 모델.ckpt")})
    monkeypatch.setattr("amp.app.session.app_data_dir", lambda *p: str(tmp_path.joinpath("app", *p)))
    rep = W.collect(probe=False)
    assert {m["model_id"] for m in rep["models"]} == set(R.CATALOG) and set(rep["profiles"]) == set(P.PROFILES)
    rof = next(m for m in rep["models"] if m["model_id"] == "mel_roformer")
    assert rof["state"] == R.FILE_MISSING
    text = W.text_report(rep)
    assert "Mel-Band RoFormer: 파일 없음" in text and "advanced 프로필" in text
    assert (tmp_path / "app" / R.FILE_NAME).is_file()
