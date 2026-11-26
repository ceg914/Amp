"""v0.7.0 (명세 15.2·15.3): 원본·반주에서 만든 악기 결과 비교(목록·A/B 듣기·스펙트로그램·근사 지표)와 겹쳐 듣기 안내.
분리 모델은 시험용(_half·_six)이다."""

from __future__ import annotations

import numpy as np

from amp.app.main_window import overlap_notice
from tests.ui.test_instruments_ui import six  # noqa: F401 - 픽스처
from tests.ui.test_main_window import create


def test_overlap_notice_only_when_instrument_tracks_join_original_or_instrumental():
    st = {"original": {"gain": 1.0, "mute": False, "solo": False},
          "instrumental": {"gain": 1.0, "mute": True, "solo": False},
          "inst_bass": {"gain": 1.0, "mute": True, "solo": False}}
    rows = list(st)
    assert overlap_notice(rows, st.get) == ""  # 기본 (악기는 음소거)
    st["inst_bass"]["mute"] = False
    assert "Original · Bass" in overlap_notice(rows, st.get) and "믹서 설정은 그대로" in overlap_notice(rows, st.get)
    st["inst_bass"]["solo"] = True  # 솔로면 그것만 들린다
    assert overlap_notice(rows, st.get) == ""
    st["inst_bass"]["solo"] = False
    st["original"]["mute"], st["instrumental"]["mute"] = True, False
    assert "반주에 이미" in overlap_notice(rows, st.get)
    st["inst_bass"]["gain"] = 0.0
    assert overlap_notice(rows, st.get) == ""


def test_compare_lists_both_inputs_switches_track_and_measures(qtbot, window, song, six, monkeypatch):  # noqa: F811
    from amp.engine.transcription import TranscriptionPipeline

    class HalfPipeline(TranscriptionPipeline):
        def __init__(self, *a, **k):
            super().__init__(*a, separation_model="_half", **k)

    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", HalfPipeline)
    w = window
    ic, p = w.instruments, w.instruments.panel
    create(qtbot, w, song)
    qtbot.waitUntil(lambda: not w.session.busy, timeout=20_000)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    qtbot.waitUntil(lambda: p.source.currentData() == "instrumental", timeout=10_000)
    p.basic.click()  # 반주에서
    qtbot.waitUntil(lambda: "drums" in ic.found and not w.session.busy, timeout=60_000)
    p.source.setCurrentIndex(p.source.findData("original"))
    p.basic.click()  # 원본에서 (따로 보관)
    qtbot.waitUntil(lambda: len(ic.variants.get("drums", ())) == 2 and not w.session.busy, timeout=60_000)
    rows = [(p.compare.topLevelItem(i).text(0), p.compare.topLevelItem(i).text(1)) for i in range(p.compare.topLevelItemCount())]
    drums = [r for r in rows if r[0].endswith("드럼")]
    assert len(drums) == 2 and any("반주에서" in r[1] for r in drums) and any("원본에서" in r[1] for r in drums)
    playing = ic.found["drums"]
    assert playing.from_instrumental  # 기본은 지금 보컬 분석 결과(반주에서 뽑은 것)
    other = next(v for v in ic.variants["drums"] if not v.from_instrumental)
    w.tracks.inst_rows["inst_drums"].mute.click()  # 음소거 풀기 → 겹쳐 듣기 안내
    qtbot.waitUntil(lambda: "inst_drums" in w.player.tracks, timeout=20_000)
    assert w.tracks.overlap_note.isVisibleTo(w.tracks) and "Drums" in w.tracks.overlap_note.text()
    before = w.player.track_state("inst_drums")
    first = w.player.tracks["inst_drums"].samples.copy()
    assert ic.choose_variant("drums", f"{other.key}/{other.slot or other.name}")
    qtbot.waitUntil(lambda: "inst_drums" in w.player.tracks, timeout=20_000)
    assert ic.found["drums"].path == other.path and w.player.track_state("inst_drums") == before  # 믹서 그대로
    assert not np.array_equal(first, w.player.tracks["inst_drums"].samples)  # 다른 결과를 듣는다
    assert "▶" in next(p.compare.topLevelItem(i).text(0) for i in range(p.compare.topLevelItemCount())
                       if p.compare.topLevelItem(i).text(1) == other.model_label)
    ic.refresh()  # 다시 읽어도 고른 결과를 지킨다
    assert ic.found["drums"].path == other.path
    ic.measure_shares()
    qtbot.waitUntil(lambda: len(ic.shares) == sum(len(v) for v in ic.variants.values()), timeout=60_000)
    assert all("dB" in p.compare.topLevelItem(i).text(2) or "없음" in p.compare.topLevelItem(i).text(2)
               for i in range(p.compare.topLevelItemCount()))
    assert "근사 지표" in p.message.text()
    p.compare.setCurrentItem(p.compare.topLevelItem(0))
    p.compare_spec.click()
    qtbot.waitUntil(lambda: "스펙트로그램:" in p.message.text(), timeout=20_000)
    assert w.spectrogram.isVisible()
