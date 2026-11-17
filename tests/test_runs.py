"""STEP 2.5 요구 4: 분석 실행 기록과 실행 간 비교."""

import json

from amp.core.model import AnalysisResult, ModelInfo, Vibrato
from amp.core.tempo import TempoMap
from amp.project import runs
from tests.conftest import note


def result(notes, sens="standard"):
    return AnalysisResult(
        5.0, TempoMap.constant(96.0, 0.25), tuple(notes),
        models={"pitch": ModelInfo("pyin", "librosa 1.0.1", params={"features_version": 2}),
                "notes": ModelInfo("segment", "2")},
        audio_fingerprint="ab" * 32, settings={"sensitivity": sens, "min_note_ms": 100.0},
        warnings=("경고 하나",),
    )


def test_record_contains_required_fields(tmp_path):
    n = note("a", 1.0, 1.5, 64)
    from dataclasses import replace

    n = replace(n, confidence=0.81, vibrato=Vibrato(5.5, 40.0))
    rec = runs.record_from_result(result([n]), "full", timings={"pitch": 1.234567, "total": 2.0},
                                  reused={"pitch": True}, resources={"pitch": {"peak_rss_mb": 300.0}},
                                  score_summary={"original": 1, "modified": 0, "user": 0}, app_version="0.2.5")
    d = rec.to_dict()
    assert d["run_id"] and d["kind"] == "full" and d["app_version"] == "0.2.5"
    assert d["models"]["pitch"] == {"name": "pyin", "version": "librosa 1.0.1", "placeholder": False,
                                    "params": {"features_version": 2}}
    assert d["settings"]["sensitivity"] == "standard" and d["tempo"] == {"bpm": 96.0, "first_beat": 0.25}
    assert d["notes"] == [{"start": 1.0, "end": 1.5, "pitch": 64, "confidence": 0.81, "ornament": False,
                           "vibrato": {"rate_hz": 5.5, "extent_cents": 40.0}}]
    assert d["timings_s"]["pitch"] == 1.235 and d["reused"] == {"pitch": True}
    path = runs.save_run(tmp_path / "analysis", rec)
    assert path.parent.name == "runs" and json.loads(path.read_text(encoding="utf-8"))["run_id"] == rec.run_id


def test_list_skips_corrupt_records_and_sorts(tmp_path):
    a = tmp_path / "analysis"
    r1 = runs.record_from_result(result([note("a", 1.0, 1.5, 64)]), "full")
    r2 = runs.record_from_result(result([note("a", 1.0, 1.5, 64)]), "redetect")
    r1.created, r2.created = "2026-11-17T01:00:00+00:00", "2026-11-17T02:00:00+00:00"
    runs.save_run(a, r2)
    runs.save_run(a, r1)
    (a / "runs" / "broken.json").write_text("{not json", encoding="utf-8")
    (a / "runs" / "partial.json").write_text('{"run_id": "x"}', encoding="utf-8")
    got = runs.list_runs(a)
    assert [r.run_id for r in got] == [r1.run_id, r2.run_id]
    assert runs.find_run(a, r2.run_id[:15] + r2.run_id[15:]).run_id == r2.run_id
    assert runs.list_runs(tmp_path / "없음") == []


def test_compare_runs_and_report(tmp_path):
    a = runs.record_from_result(result([note("a", 1.0, 1.5, 64), note("b", 2.0, 2.5, 67), note("c", 3.0, 3.5, 69)]), "full")
    b = runs.record_from_result(result([note("a", 1.02, 1.5, 64), note("b", 2.01, 2.4, 68), note("d", 4.0, 4.4, 71)],
                                       sens="precise"), "redetect")
    cmp = runs.compare_runs(a, b)
    assert len(cmp.matched) == 1 and cmp.mean_onset_shift_ms == 20.0
    assert [(x["pitch"], y["pitch"]) for x, y in cmp.pitch_changed] == [(67, 68)]
    assert [n["pitch"] for n in cmp.only_a] == [69] and [n["pitch"] for n in cmp.only_b] == [71]
    assert cmp.settings_diff() == {"sensitivity": ("standard", "precise")}
    text = runs.markdown_report(cmp)
    for s in ("일치: 1개", "음표 수 | 3 | 3", "A에만 있는 음표", "MIDI 69", "바뀐 설정", "precise"):
        assert s in text
    path = runs.write_report(tmp_path / "analysis", a, b)
    assert path.name == f"compare-{a.run_id}-vs-{b.run_id}.md" and path.read_text(encoding="utf-8") == text


def test_compare_runs_cli(tmp_path, capsys):
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "compare_runs", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    proj = tmp_path / "프로젝트 #1"
    assert cli.main([str(proj)]) == 1  # 기록 없음
    for k, ns in (("full", [note("a", 1.0, 1.5, 64)]), ("redetect", [note("a", 1.0, 1.5, 64), note("b", 2, 3, 65)])):
        r = runs.record_from_result(result(ns), k)
        r.created = "2026-11-17T0%d:00:00+00:00" % (1 if k == "full" else 2)
        runs.save_run(proj / "analysis", r)
    assert cli.main([str(proj), "--list"]) == 0
    assert "redetect" in capsys.readouterr().out
    assert cli.main([str(proj)]) == 0
    assert list((proj / "analysis" / "reports").glob("compare-*.md"))
    assert cli.main([str(proj), "--stdout"]) == 0
    assert "B에만 있는 음표" in capsys.readouterr().out
