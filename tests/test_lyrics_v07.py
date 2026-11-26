"""v0.7.0 Feature H: 선택 구간만 다시 인식 (작업 프로세스의 구간 자르기, 엔진 캐시, 인식 원본 합치기)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from amp.lyrics import asr, engine, review
from amp.lyrics.model import AsrSegment, Recognition


def _seg(i, a, b, text, reasons=()):
    return AsrSegment(f"seg{i}", a, b, text, -0.2, 0.1, 1.0, tuple(reasons))


def test_shift_segments_restores_song_time_and_drops_outside():
    segs = [{"start": 0.2, "end": 0.4, "text": "앞", "words": []},  # 여유 구간(선택 밖)
            {"start": 0.6, "end": 1.2, "text": "가운데", "words": [{"text": "가운데", "start": 0.6, "end": 1.2}]}]
    out = asr.shift_segments(segs, 1.5, 2.0, 3.0)
    assert [s["text"] for s in out] == ["가운데"]
    assert out[0]["start"] == 2.1 and out[0]["words"][0]["end"] == 2.7


def test_worker_crops_region_and_shifts_times(tmp_path, monkeypatch):
    from amp.engine import worker

    p = tmp_path / "v.wav"
    sf.write(p, (0.1 * np.sin(np.linspace(0, 2000, 16000 * 5))).astype(np.float32), 16000)
    seen = {}

    def fake(y16, **kw):
        seen["n"], seen["options"] = len(y16), kw.get("options")
        return {"model": "turbo", "language": "ko", "device": "cpu",
                "segments": [{"start": 0.6, "end": 1.4, "text": "다시", "words": []}]}

    monkeypatch.setattr(asr, "transcribe", fake)
    info = worker.cmd_asr({"input": str(p), "out_dir": str(tmp_path), "download_root": "", "region": [2.0, 3.0]})
    assert seen["n"] == int(2.0 * 16000) and seen["options"] == {"gap_fill": False}  # 1.5~3.5초만
    raw = json.loads((tmp_path / "asr.json").read_text(encoding="utf-8"))
    assert info["segments"] == 1 and raw["region"] == [2.0, 3.0]
    assert raw["segments"][0]["start"] == 2.1 and raw["segments"][0]["activity"] > 0


def test_merge_region_replaces_only_inside_and_keeps_record():
    old = Recognition("turbo", language="ko", segments=(_seg(0, 0.0, 2.0, "하나"), _seg(1, 3.0, 5.0, "둘"),
                                                         _seg(2, 6.0, 8.0, "셋")), settings={"model": "turbo"})
    new = Recognition("turbo", language="ko", segments=(_seg(0, 3.1, 4.0, "둘둘"), _seg(1, 4.2, 5.2, "넷")))
    m = engine.merge_region(old, new, (2.5, 5.5))
    assert [s.text for s in m.segments] == ["하나", "둘둘", "넷", "셋"]
    assert [s.id for s in m.segments] == ["seg0", "seg1", "seg2", "seg3"]
    assert review.R_REGION in m.segments[1].review and review.R_REGION not in m.segments[0].review
    assert m.settings["model"] == "turbo" and m.settings["regions"][0] == {
        "start": 2.5, "end": 5.5, "model": "turbo", "language": "ko", "created": "", "replaced": 1, "added": 2}
    first = engine.merge_region(None, new, (2.5, 5.5))  # 인식 원본이 없을 때
    assert [s.text for s in first.segments] == ["둘둘", "넷"] and first.settings["regions"][0]["replaced"] == 0


class RegionRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, args, progress=None, token=None, numba=None):
        self.calls.append((command, args.get("region")))
        a, b = args["region"]
        Path(args["out_dir"], "asr.json").write_text(json.dumps(
            {"model": "turbo", "language": "ko", "device": "cpu",
             "segments": [{"start": a + 0.1, "end": b - 0.1, "text": "구간", "avg_logprob": -0.1, "no_speech_prob": 0.1,
                           "compression_ratio": 1.0, "activity": 0.9, "words": []}]}), encoding="utf-8")
        return {}


def test_recognize_region_uses_own_cache_per_region(tmp_path):
    req = engine.LyricsRequest(str(tmp_path / "vocals.wav"), "stems-a", str(tmp_path / "cache"), str(tmp_path / "w"),
                               str(tmp_path / "m"))
    r = RegionRunner()
    rec, reused = engine.LyricsEngine(r).recognize_region(req, (1.0, 2.0))
    assert not reused and rec.segments[0].text == "구간" and rec.settings["region"] == [1.0, 2.0]
    _rec, reused = engine.LyricsEngine(r).recognize_region(req, (1.0, 2.0))
    assert reused and len(r.calls) == 1
    engine.LyricsEngine(r).recognize_region(req, (1.0, 2.5))
    assert r.calls[-1] == ("asr", [1.0, 2.5]) and len(r.calls) == 2
    import pytest

    with pytest.raises(ValueError):
        engine.LyricsEngine(r).recognize_region(req, (2.0, 2.0))
