"""v0.6 품질 탭: 지표·참조 없음 안내, 숫자를 누르면 검토 탭이 그 종류만 보여 줌, 필터 해제."""

from __future__ import annotations

from amp.compare import quality as Q
from tests.ui.test_main_window import create
from tests.ui.test_suggestions_ui import _analysis


def test_quality_tab_counts_filter_review_list(qtbot, window, song):
    w = window
    create(qtbot, w, song)
    w.doc.set_analysis(_analysis(5.0))
    from amp.lyrics import ops

    st = ops.apply_text(w.doc.lyrics, "가 나")
    ids = [x.id for x in st.text.syllables]
    w.doc.push_lyrics(ops.set_times(st, {ids[0]: (0.2, 0.5), ids[1]: (2.4, 2.8)}, source="model"), "가사")  # 음표 없는 음절
    w.review.refresh()
    qtbot.waitUntil(lambda: bool(w.review.items), timeout=10_000)
    w.quality.refresh()
    rep = w.quality.report
    assert rep.indicators["음표 수"] == 2 and Q.NO_REFERENCE_NOTE in w.quality.panel.note.text()
    assert "정확도" not in w.quality.panel.indicators.text()
    key = next(k for k, (_l, n, kinds) in rep.counts.items() if kinds and n > 0)
    w.quality.panel.count_buttons[key].click()
    assert w.side_panel() is w.review.panel
    kinds = rep.counts[key][2]
    rows = w.review.panel._rows
    assert rows and all(it.kind in kinds for it, _s in rows)
    assert w.review.panel.kind_bar.isVisibleTo(w.review.panel)
    w.review.panel.kind_clear.click()
    assert w.review.kind_filter is None
