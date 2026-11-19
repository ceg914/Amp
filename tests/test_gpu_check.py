"""GPU 진단 도구 자체 확인 (이 환경에는 GPU가 없어 실제 GPU 측정은 사용자 PC에서 한다)."""

from amp.devtools import gpu_check as G


def test_no_cuda_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_run_probe", lambda: {"python": "3.x", "torch": "2.x+cpu", "cuda_available": False})
    monkeypatch.setattr(G, "nvidia_smi", lambda *a: None)
    rep, md = G.check(["large-v3"], None, 5, tmp_path, device="cuda", cancel=False, mms=False)
    assert rep["summary"]["gpu"] == "확인 못 함" and "runs" not in rep
    text = md.read_text(encoding="utf-8")
    assert "확인 못 함" in text and "성공" not in text and (md.parent / "gpu_check.json").exists()


def test_summary_wording_for_outcomes():
    rep = {"runs": [
        {"model_requested": "large-v3", "ok": False, "error_code": "oom_model", "error": "turbo로 바꾸세요", "gpu_released": True},
        {"model_requested": "turbo", "ok": True, "asr": {"device": "cuda", "model": "turbo",
                                                         "stats": {"load_s": 6.1, "transcribe_s": 20.5}}, "gpu_released": False},
    ], "cancel": {"cancelled": True, "gpu_released": True}, "mms": {"ok": True, "wall_s": 9.0, "mms": {"device": "cuda"}}}
    s = G.summarize(rep, "cuda")
    assert s["gpu"] == "GPU 인식 성공"
    assert "조용히 CPU로 넘어가지 않음" in s["large-v3"]
    assert s["turbo"].startswith("성공 · cuda") and "돌아오지 않음" in s["turbo"]
    assert s["cancel"] == "취소됨 · GPU 메모리 반환됨" and s["mms"].startswith("성공 · cuda")
    assert G.summarize({"runs": [rep["runs"][0]]}, "cuda")["gpu"] == "GPU 인식 안 됨"
    md = G.markdown({"created": "t", "env": {}, "summary": s, **rep})
    assert "None" not in md.split("## 요약")[0]
