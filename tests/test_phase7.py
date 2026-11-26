"""v0.7.0 Phase 7: 분석 결과 내보내기(JSON·CSV·TXT, 명세 18)와 검증 자료(명세 19). 합성 자료."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from amp.core.model import Note, PitchCurve
from amp.devtools.synth import SR, sung
from amp.engine import validation as V
from amp.engine.acoustics import run as AR
from amp.engine.acoustics.features import frame_features
from amp.engine.dsp.pitch import to_pitch_sr, track_pitch
from amp.export import analysis_export as AE
from amp.lyrics import ops
from amp.lyrics import pronunciation as P
from amp.lyrics.model import LyricsState

MEL = [(0.5, 2.5, 64), (3.0, 5.0, 67), (6.0, 11.0, 69), (12.0, 22.0, 67)]


@pytest.fixture(scope="module")
def analysed():
    y = sung(MEL, 24.0, vibrato_cents=40)
    fr = track_pitch(y, SR)
    feat = frame_features(to_pitch_sr(y, SR), fr.f0_hz)
    notes = [Note(f"n{i}", a, b, p) for i, (a, b, p) in enumerate(MEL)]
    res = AR.analyze(fr.f0_hz, feat, AR.Snapshot(notes, None, []))
    curve = PitchCurve(0.0, fr.frame_rate, np.asarray(fr.f0_hz, np.float32), np.asarray(fr.voiced_prob, np.float32))
    st = P.sync(ops.apply_text(LyricsState(), "가 나 다 라"))
    ids = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {sid: (a, b) for sid, (a, b, _p) in zip(ids, MEL)}, source="model")
    return curve, res, notes, st


def test_export_all_writes_json_csv_txt_with_region(analysed, tmp_path):
    curve, res, notes, st = analysed
    src = AE.Sources("시험 곡", curve, None, res, st, notes)
    written, skipped = AE.export_all(src, tmp_path, "시험")
    names = {p.name for p in written}
    assert {"시험-analysis.json", "시험-pitch.csv", "시험-vibrato.csv", "시험-syllables.csv", "시험-summary.txt",
            "시험-lyrics.txt"} <= names
    assert "csv:voices" in skipped  # 성부 결과가 없으면 파일을 만들지 않고 이유를 남긴다
    raw = (tmp_path / "시험-pitch.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    assert len(rows) == len(curve.f0_hz) and set(rows[0]) == {"time_s", "f0_hz", "midi", "confidence"}
    j = json.loads((tmp_path / "시험-analysis.json").read_text(encoding="utf-8"))
    assert j["schema"] == "sonatlas.analysis_export" and j["acoustics"]["boundaries_applied"] is False
    assert len(j["notes"]) == 4 and j["lyrics"]["syllables"][0]["syllable"] == "가"
    vib = list(csv.DictReader(io.StringIO((tmp_path / "시험-vibrato.csv").read_text(encoding="utf-8-sig"))))
    assert vib and {"extent_one_side_cents", "extent_peak_to_peak_cents"} <= set(vib[0])
    assert "[가]" in (tmp_path / "시험-lyrics.txt").read_text(encoding="utf-8-sig")
    # 구간: 6~11초만
    sub = AE.Sources("시험 곡", curve, None, res, st, notes, region=(6.0, 11.0))
    rows = AE.pitch_rows(sub)
    assert rows and min(r["time_s"] for r in rows) >= 6.0 and max(r["time_s"] for r in rows) <= 11.0
    assert [r["syllable"] for r in AE.syllable_rows(sub)] == ["다"]
    assert all(r["end_s"] >= 6.0 and r["start_s"] <= 11.0 for r in AE.vibrato_rows(sub))
    assert "구간: 6.00~11.00초" in AE.summary_text(sub)


def test_validation_snapshot_save_compare_delete(analysed, tmp_path):
    curve, res, notes, st = analysed
    with pytest.raises(V.ValidationError):
        V.snapshot(0.0, 5.0, "independent", ["pitch"], pitch_curve=curve)  # 10~30초
    with pytest.raises(V.ValidationError):
        V.snapshot(0.0, 20.0, "independent", [], pitch_curve=curve)
    with pytest.raises(V.ValidationError):
        V.snapshot(0.0, 20.0, "independent", ["voices"], pitch_curve=curve)  # 성부 결과 없음
    vs = V.snapshot(2.0, 22.0, "independent", ["pitch", "vibrato", "syllables", "boundaries", "breaths"],
                    pitch_curve=curve, acoustics=res, lyrics=st, note="합성 시험")
    p = V.save(tmp_path, vs)
    assert p.parent.name == "validation" and "학습" in json.loads(p.read_text(encoding="utf-8"))["policy"]
    back = V.load_all(tmp_path)
    assert len(back) == 1 and back[0].set_id == vs.set_id and back[0].items.keys() == vs.items.keys()
    same = V.compare(back[0], pitch_curve=curve, acoustics=res, lyrics=st)
    assert same["independent"] and "warning" not in same
    assert same["pitch"]["raw_pitch_accuracy_50c"] == 1.0 and same["pitch"]["median_cents_error"] == 0.0
    assert same["vibrato"]["rate_error_hz"] == 0.0 and same["syllables"]["median_start_error_ms"] == 0.0
    shifted = PitchCurve(curve.start_time, curve.frame_rate, curve.f0_hz * 2 ** (100 / 1200), curve.confidence)
    off = V.compare(back[0], pitch_curve=shifted)
    assert off["pitch"]["raw_pitch_accuracy_50c"] == 0.0 and 99 <= off["pitch"]["median_cents_error"] <= 101
    dev = V.snapshot(2.0, 22.0, "dev_real", ["pitch"], pitch_curve=curve)
    assert "독립 검증 자료가 아닙니다" in V.compare(dev, pitch_curve=curve)["warning"]
    assert V.delete(tmp_path, vs.set_id) and V.load_all(tmp_path) == []
