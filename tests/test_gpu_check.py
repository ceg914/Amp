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


def test_report_shows_slowest_window_and_decode_calls():
    st = {"load_s": 31.5, "transcribe_s": 281.0, "decode_calls": 14, "fallback_segments": 2,
          "windows": [["가사 인식 중", 0.0, 30.0, 6.0, 1], ["가사 인식 중", 30.0, 53.0, 5.0, 1], ["가사 인식 중", 53.0, 60.0, 268.0, 12]]}
    rep = {"created": "t", "env": {}, "summary": {}, "runs": [
        {"model_requested": "large-v3", "ok": True, "asr": {"device": "cuda", "model": "large-v3", "stats": st}}]}
    md = G.markdown(rep)
    assert "디코딩 14번" in md and "53.0~60.0초 268.0초" in md and "디코딩 12번" in md


def test_nearly_full_gpu_is_mentioned():
    rep = {"env": {"gpu_total_mb": 8192}, "runs": [
        {"model_requested": "large-v3", "ok": True, "gpu_used_peak_mb": 7514.0, "gpu_released": True,
         "asr": {"device": "cuda", "stats": {"load_s": 31.5, "transcribe_s": 281.0}}}]}
    assert "거의 참" in G.summarize(rep, "cuda")["large-v3"]
    rep["runs"][0]["gpu_used_peak_mb"] = 5000.0
    assert "거의 참" not in G.summarize(rep, "cuda")["large-v3"]
