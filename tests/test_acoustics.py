"""v0.7.0 Phase 3: 호흡·프레이즈(E), 발성·음높이 변화(F), 발성 경계(G), 화음 관계(C). 합성 자료 (명세 20.3).
정답이 있는 합성 사례만 수치로 확인한다. 실제 노래의 정확도가 아니다."""

from __future__ import annotations

import numpy as np
import pytest

from amp.core.model import Note
from amp.devtools.synth import SR, breath_noise, sung
from amp.engine.acoustics import breath as B
from amp.engine.acoustics import harmony as H
from amp.engine.acoustics import onsets as O
from amp.engine.acoustics import phonation as P
from amp.engine.acoustics.features import frame_features
from amp.engine.dsp.pitch import to_pitch_sr, track_pitch

MEL = [(0.5, 1.4, 64), (1.5, 2.6, 67), (3.4, 4.4, 69), (4.5, 5.6, 67), (6.4, 7.5, 64)]
BREATHS = [(2.85, 3.3), (5.85, 6.3)]


@pytest.fixture(scope="module")
def song():
    dur = 8.0
    y = sung(MEL, dur)
    rng = np.random.default_rng(5)
    cons = np.zeros(len(y), np.float32)
    for a in (1.44, 4.44):  # 음 시작 바로 앞 60ms 무성 자음
        i = int(a * SR)
        cons[i:i + int(0.06 * SR)] = rng.standard_normal(int(0.06 * SR)).astype(np.float32) * 0.03
    x = y + breath_noise(dur, BREATHS, level_db=-32) + cons + rng.standard_normal(len(y)).astype(np.float32) * 1e-4
    fr = track_pitch(x, SR)
    feat = frame_features(to_pitch_sr(x, SR), fr.f0_hz)
    return x, fr, feat


def test_breath_candidates_and_consonants(song):
    _x, fr, feat = song
    cls, breaths = B.classify(feat, np.isfinite(fr.f0_hz))
    got = [(b.start, b.end) for b in breaths]
    assert len(got) == 2
    for (a, b), (ga, gb) in zip(BREATHS, got):
        assert abs(ga - a) < 0.08 and abs(gb - b) < 0.08
    for a in (1.44, 4.44):  # 자음은 호흡 후보가 아니다
        i = int(round((a + 0.03) * 100))
        assert cls[i] in ("unvoiced", "voiced")
    assert set(np.unique(cls)) <= set(B.CLASSES)
    assert (cls == "silence").any() and (cls == "voiced").sum() > 400


def test_phrases_syllables_and_overrides(song):
    _x, fr, feat = song
    cls, breaths = B.classify(feat, np.isfinite(fr.f0_hz))
    ph = B.phrases(cls, breaths, syllable_starts=[a for a, _b, _m in MEL])
    assert [p.syllables for p in ph] == [2, 2, 1]
    assert ph[1].breath_ids == [breaths[0].breath_id] and ph[0].gap_before_s is None
    assert ph[1].gap_before_s == pytest.approx(0.8, abs=0.1)
    s = B.summary(cls, breaths, ph)
    assert s["breath_candidates"] == 2 and s["phrases"] == 3
    b2, p2 = B.apply_overrides(breaths, ph, {"breath_status": {breaths[0].breath_id: "rejected"},
                                             "breath_added": [[0.1, 0.4]], "phrases": [[0.5, 5.6], [6.4, 7.5]]})
    assert b2[0].status == "rejected" and b2[-1].status == "confirmed" and len(p2) == 2 and p2[0].user_edited
    assert breaths[0].status == "candidate"  # 원래 목록은 그대로


def test_no_breath_in_noise_or_long_silence():
    rng = np.random.default_rng(1)
    dur = 4.0
    y = sung([(0.5, 1.5, 64), (2.5, 3.5, 66)], dur)
    hum = (rng.standard_normal(len(y)) * 0.003).astype(np.float32)  # 반주 유입 같은 고른 잡음 (호흡처럼 오르내리지 않음)
    fr = track_pitch(y + hum, SR)
    feat = frame_features(to_pitch_sr(y + hum, SR), fr.f0_hz)
    _cls, breaths = B.classify(feat, np.isfinite(fr.f0_hz))
    assert all(b.end - b.start <= B.BreathSettings().max_breath_s for b in breaths)
    assert len(breaths) <= 1


def test_harmonic_metrics_reported_without_voice_type_claims(song):
    _x, fr, feat = song
    v = np.isfinite(fr.f0_hz)
    assert np.nanmedian(feat["hnr_db"][v]) > 10  # 합성 노래는 배음이 뚜렷하다
    assert np.isnan(feat["hnr_db"][~v]).all()
    m = P.span_metrics(feat, fr.f0_hz, [(0.5, 1.4, "n1"), (2.7, 3.3, None)])
    assert m[0].voiced_ratio > 0.9 and m[0].hnr_db is not None and m[0].pitch_stability_cents < 20
    assert m[1].voiced_ratio == 0 and m[1].hnr_db is None
    import inspect

    src = inspect.getsource(P)
    assert "정하지 않는다" in src  # 흉성·두성 등을 정하지 않는다는 설명


def _f0(segments, dur=3.0):
    t = np.arange(int(dur * 100)) / 100
    m = np.full(len(t), np.nan)
    for a, b, f in segments:
        sel = (t >= a) & (t < b)
        m[sel] = f(t[sel]) if callable(f) else f
    return np.where(np.isfinite(m), 440 * 2 ** ((m - 69) / 12), np.nan)


def test_pitch_events_multi_label():
    scoop = lambda t: 64 - 1.0 * np.clip((0.6 - t) / 0.1, 0, 1)  # noqa: E731  0.5초에 1반음 아래서 시작해 0.6초에 도달
    glide = lambda t: 64 + 3 * np.clip((t - 1.2) / 0.15, 0, 1)  # noqa: E731
    f0 = _f0([(0.5, 2.0, lambda t: np.where(t < 1.0, scoop(t), glide(t)))])
    ev = P.detect_events(f0)
    kinds = [e.kind for e in ev]
    assert "approach" in kinds and "portamento" in kinds and kinds.count("stable") >= 2
    ap = next(e for e in ev if e.kind == "approach")
    assert ap.delta_cents > 50 and ap.settle_s is not None and ap.start == pytest.approx(0.5, abs=0.05)
    po = next(e for e in ev if e.kind == "portamento")
    assert po.delta_cents == pytest.approx(300, abs=60) and po.rate_cents_per_s > 0
    jump = _f0([(0.5, 2.0, lambda t: np.where(t < 1.2, 64.0, 69.0))])
    assert any(e.kind == "jump" for e in P.detect_events(jump))
    notes = [Note("a", 0.5, 0.95, 64), Note("b", 1.0, 1.5, 64)]
    ev2 = P.detect_events(_f0([(0.5, 1.5, 64.0)]), notes=notes, melisma=[(0.5, 1.5, "syl1")])
    rep = [e for e in ev2 if e.kind == "repeated"]
    assert rep and rep[0].note_id == "b"
    assert any(e.kind == "melisma" and e.syllable_id == "syl1" for e in ev2)
    assert sum(e.kind == "voicing" for e in ev2) == 2


def test_note_boundaries_and_suggestions():
    dur = 3.0
    y = sung([(0.6, 1.6, 64)], dur, attack=0.02)
    rng = np.random.default_rng(2)
    cons = np.zeros(len(y), np.float32)
    i = int(0.5 * SR)
    cons[i:i + int(0.1 * SR)] = rng.standard_normal(int(0.1 * SR)).astype(np.float32) * 0.02  # 앞 자음 100ms
    x = y + cons + rng.standard_normal(len(y)).astype(np.float32) * 1e-5
    fr = track_pitch(x, SR)
    feat = frame_features(to_pitch_sr(x, SR), fr.f0_hz)
    note = Note("n1", 0.45, 1.9, 64)  # 자동 음표가 늦게까지 이어짐
    nb = O.analyze([note], fr.f0_hz, feat["rms_db"])[0]
    assert nb.voiced_onset == pytest.approx(0.6, abs=0.05)
    assert nb.acoustic_onset is not None and nb.acoustic_onset <= nb.voiced_onset and nb.acoustic_onset < 0.58
    assert nb.stable_pitch_onset is not None and nb.stable_pitch_onset >= nb.voiced_onset
    assert nb.voiced_end == pytest.approx(1.6, abs=0.06)
    assert nb.acoustic_tail_end is not None and nb.acoustic_tail_end >= nb.voiced_end - 0.01
    assert nb.measured == 5

    class FakeScore:
        def notes(self):
            return [note]

    class FakeProject:
        score = FakeScore()
        status = "original"

        def note_status(self, n):
            return self.status

    proj = FakeProject()
    sug = O.suggestions(proj, [nb])
    assert len(sug) == 1 and sug[0].suggestion_type == "boundary"
    assert sug[0].proposed_changes["start"] == pytest.approx(nb.voiced_onset, abs=1e-3)
    assert sug[0].proposed_changes["end"] == pytest.approx(nb.voiced_end, abs=1e-3)
    assert any("발성 경계 분석" in e for e in sug[0].evidence)
    proj.status = "modified"  # 사용자가 고친 음표는 제안하지 않는다
    assert O.suggestions(proj, [nb]) == []
    moved = Note("n1", 0.5, 1.9, 64)  # 분석 뒤 음표가 바뀌면 오래된 경계 결과라 제안하지 않는다
    assert O.stale(nb, moved)


def test_silent_note_boundary_is_unmeasured():
    nb = O.analyze([Note("x", 0.5, 1.0, 60)], np.full(200, np.nan), np.full(200, -90.0))[0]
    assert nb.measured == 0 and nb.voiced_onset is None


def test_harmony_intervals_and_roles():
    from amp.core import result_meta as RM
    from amp.engine import voices as V
    from amp.engine.dsp.multipitch import Candidates

    n = 400
    t = np.arange(n)
    main = np.where((t >= 20) & (t < 380), 440.0, np.nan)  # 길게 부르는 성부
    harm = np.where((t >= 100) & (t < 300), 440 * 2 ** (-4 / 12), np.nan)  # 장3도 아래, 짧게
    f0 = np.stack([main, harm], 1)
    c = Candidates(f0, np.where(np.isfinite(f0), [1.0, 0.9], 0.0), np.where(np.isfinite(f0), 0.9, 0.0), np.ones_like(f0),
                   np.zeros(n))
    vs = V.analyze(c, RM.make_meta(V.SCHEMA, 1, "a" * 64), 2)
    ov = H.overlaps(vs)
    assert len(ov) == 1 and ov[0].interval == "장3도" and ov[0].upper == "v1"
    assert ov[0].start == pytest.approx(1.0) and ov[0].end == pytest.approx(3.0)
    assert H.interval_summary(ov) == {"장3도": 2.0}
    roles = {r.voice_track_id: r for r in H.estimate_roles(vs, syllable_starts=[0.2, 1.5])}
    assert roles["v1"].role == "main_candidate" and roles["v2"].role == "harmony_candidate"
    assert any("가사" in e for e in roles["v1"].evidence)
    vs2 = V.set_attr(vs, "v2", role="main_candidate")  # 사용자가 바꾸면 그 값이 우선
    roles2 = {r.voice_track_id: r for r in H.estimate_roles(vs2)}
    assert roles2["v2"].role == "main_candidate" and roles2["v2"].user_set
    assert H.interval_name(-16.0) == ("장3도", 1) and H.interval_name(7.2) == ("5도", 0)


def test_run_save_load_overrides_and_stale(song, tmp_path):
    """음향 분석 전체: 작업 프로세스 명령(이 프로세스에서 실행) → 저장 → 다시 읽기, 호흡 수정 유지, 소스 변경 시 stale."""
    import io

    import soundfile as sf

    from amp.core import result_meta as RM
    from amp.engine import pitch_runs as PR
    from amp.engine import worker
    from amp.engine.acoustics import run as R

    x, _fr, _feat = song
    wav = tmp_path / "보컬.wav"
    sf.write(wav, np.stack([x, x], 1), SR)

    def runner(command, args, progress=None, token=None, numba_cache_dir=None):
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old

    inp = PR.PitchInput(str(wav), "vocals", "stems:x", "1" * 64, "a" * 64, 8.0, "보컬")
    notes = [Note(f"n{i}", a, b, m) for i, (a, b, m) in enumerate(MEL)]
    snap = R.Snapshot(notes, [a for a, _b, _m in MEL], [(3.4, 4.4, "syl3")])
    res = R.run(inp, tmp_path / "analysis", runner, snap, cache_root=str(tmp_path / "cache"))
    assert len(res.breaths) == 2 and len(res.phrases) == 3 and len(res.boundaries) == 5
    assert res.meta["vibrato"].status == RM.COMPLETED and res.meta["breath"].input_id == "stems:x"
    back = R.load(tmp_path / "analysis")
    assert back is not None and [b.breath_id for b in back.breaths] == [b.breath_id for b in res.breaths]
    assert len(back.events) == len(res.events) and back.boundaries[0].note_id == "n0"
    import json

    vj = json.loads((tmp_path / "analysis" / R.FILES["vibrato"]).read_text(encoding="utf-8"))
    assert "한쪽 진폭" in vj["extent_definition"]
    bj = json.loads((tmp_path / "analysis" / R.FILES["boundaries"]).read_text(encoding="utf-8"))
    assert bj["applied"] is False
    upd = R.save_overrides(tmp_path / "analysis", {"breath_status": {"b1": "rejected"}})
    assert upd.breaths[0].status == "rejected" and len(upd.phrases) == 3  # 쉼(0.77초)이 길어 프레이즈는 그대로 나뉜다
    assert upd.phrases[1].breath_ids == []  # 제외한 호흡은 프레이즈에 붙이지 않는다
    assert upd.breath_summary["breath_candidates"] == 1
    again = R.run(inp, tmp_path / "analysis", runner, snap, cache_root=str(tmp_path / "cache"))
    assert again.breaths[0].status == "rejected" and again.overrides == {"breath_status": {"b1": "rejected"}}
    changed = R.mark_stale(tmp_path / "analysis", "vocals:stems:other:" + "2" * 16, "a" * 64)
    assert set(changed) == set(R.FILES) and R.load(tmp_path / "analysis").meta["breath"].status == RM.STALE
