"""STEP 5 모델 비교 화면: 비교 모드 진입·종료, A/B 청취(위치·반복·속도 유지, 크로스페이드, 음량 맞춤),
스펙트로그램·피아노 롤 비교 표시, 분리 WAV 불변, 분석 소스 전환과 사용자 편집 보호, 저장·다시 열기, GPU 메모리 부족 안내.

분리는 시험용 모델(_identity: 보컬 = 입력, _half: 보컬 = 입력 절반)을 실제 작업 프로세스로 돌린다. 실제 htdemucs·RoFormer
가중치는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import math

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from amp.app import model_compare_controller as MC
from amp.engine.model_compare import ModelSpec
from tests.ui.test_main_window import IdentityPipeline, create


def _hashes(folder) -> dict:
    return {str(p.relative_to(folder.paths.stems)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.paths.stems.rglob("*.wav")}


def _compare(qtbot, w, specs, timeout=180_000):
    h = w.model_compare.run(specs)
    assert h is not None and w.session.busy  # 바쁨 표시는 제출과 함께 켜진다
    qtbot.waitUntil(lambda: not w.session.busy, timeout=timeout)
    return h


@pytest.fixture
def compared(qtbot, window, song):
    folder = create(qtbot, window, song)
    _compare(qtbot, window, [ModelSpec("_identity"), ModelSpec("_half")])
    return window, folder


def test_compare_mode_layout_and_exit_restores_editor(compared):
    w, folder = compared
    mc = w.model_compare
    assert mc.active and w.splitter.widget(0) is mc.panel and not w.tracks.isVisible()
    assert w.right_tabs.indexOf(mc.diff_panel) >= 0
    assert len(mc.outputs) == 2 and mc.keys["a"] and mc.keys["b"] and mc.keys["a"] != mc.keys["b"]
    assert {MC.CMP_A, MC.CMP_B} <= set(w.player.tracks)
    # 지금 분석에 쓰는 결과는 바꾸지 않았다 (분리 전 그대로)
    assert w.session.manifest is None and w.doc.project.analysis is None
    # 모델 차이 표: 지표가 A, B 나란히. '정확도' 같은 말은 없다
    t = mc.diff_panel.table
    assert t.rowCount() >= 5
    text = " ".join(t.item(r, c).text() for r in range(t.rowCount()) for c in range(3))
    assert "정확" not in text and "우수" not in text
    # 끄면 원래대로
    w.model_compare.act_mode.trigger()
    assert not mc.active and w.splitter.widget(0) is w.tracks and w.tracks.isVisible()
    assert w.right_tabs.indexOf(mc.diff_panel) < 0
    assert MC.CMP_A not in w.player.tracks and MC.CMP_B not in w.player.tracks
    assert not any(w.player.track_state(n)["solo"] for n in w.player.tracks)
    assert w.view.roll.model_layers == {} and not w.spectrogram.comparing()
    assert not w.review.extra_items


def test_ab_switch_keeps_position_loop_speed_and_crossfades(qtbot, compared):
    w, folder = compared
    mc = w.model_compare
    before = _hashes(folder)
    w.player.seek(2.0)
    w.listening.set_region(1.5, 3.5, enable=True)
    mc.listen("a")
    st = {n: w.player.track_state(n) for n in w.player.tracks}
    assert st[MC.CMP_A]["solo"] and not st[MC.CMP_B]["solo"]
    pos = w.player.position()
    mc.toggle_ab()
    assert mc.listening == "b"
    assert w.player.track_state(MC.CMP_B)["solo"] and not w.player.track_state(MC.CMP_A)["solo"]
    assert w.player.position() == pytest.approx(pos)
    assert w.player.loop_region == (1.5, 3.5) and w.player.loop_enabled
    if not w.player.output_available():
        pytest.skip("오디오 출력 장치 없음: 재생 중 전환은 확인하지 못함")
    w.player.play()
    # 컨테이너 PulseAudio는 첫 소리까지 1.7~1.9초 걸린다: 위치가 실제로 움직인 뒤 잰다
    qtbot.waitUntil(lambda: abs(w.player.position() - 2.0) > 0.1, timeout=15_000)
    sink, mixer = w.player._sink, w.player._mixer
    t0 = w.player.position()
    mc.toggle_ab()
    assert w.player._sink is sink and w.player._mixer is mixer  # 스트림을 다시 만들지 않는다 → 위치 그대로
    assert mixer.targets()[MC.CMP_A] > 0 and mixer.targets()[MC.CMP_B] == 0
    qtbot.wait(300)
    t1 = w.player.position()
    assert 1.5 <= t1 <= 3.5 and (t1 > t0 or t1 < t0 - 1.0)  # 반복 구간 안에서 계속 진행 (되돌아 감기 포함)
    w.player.pause()
    assert _hashes(folder) == before  # 청취·전환은 분리 WAV를 바꾸지 않는다


def test_listening_level_match_changes_playback_gain_only(compared):
    w, folder = compared
    mc = w.model_compare
    before = _hashes(folder)
    mc.panel.level_mode.setCurrentIndex(mc.panel.level_mode.findData("match"))
    ka, kb = mc.keys["a"], mc.keys["b"]
    la, lb = mc.levels[(ka, "vocals")], mc.levels[(kb, "vocals")]
    assert la - lb == pytest.approx(20 * math.log10(2), abs=0.3)  # _half 보컬은 절반 크기
    g = mc.gains()
    assert g[MC.CMP_A] == pytest.approx(0.5, rel=0.05) and g[MC.CMP_B] == pytest.approx(1.0)
    mc.panel.vol_b.setValue(-6)
    assert w.player.track_state(MC.CMP_B)["gain"] == pytest.approx(10 ** (-6 / 20), rel=1e-3)
    assert "LUFS 아님" in mc.panel.level_info.text()
    mc.panel.level_mode.setCurrentIndex(mc.panel.level_mode.findData("raw"))
    assert mc.gains()[MC.CMP_A] == pytest.approx(1.0)
    assert _hashes(folder) == before


def test_spectrogram_and_piano_roll_compare_views_render(qtbot, compared):
    w, _folder = compared
    mc = w.model_compare
    spec = w.spectrogram
    assert spec.isVisible() and spec.comparing()
    floor, ceil = spec.db_floor, spec.db_ceil
    for mode in ("a", "b", "side", "stack", "diff"):
        mc.panel.spec_mode.setCurrentIndex(mc.panel.spec_mode.findData(mode))
        assert spec.compare_mode == mode
        spec.grab()
        spec.wait_idle()
        img = spec.grab()
        assert not img.isNull()
        assert (spec.db_floor, spec.db_ceil) == (floor, ceil)  # 같은 dB 기준 (자동 범위 없음)
    assert spec.visible_tiles("a") and len(spec.visible_tiles("a")) == len(spec.visible_tiles("b"))
    layers = w.view.roll.model_layers
    assert set(layers) == {"a", "b"} and layers["a"]["notes"] and layers["a"]["curve"] is not None
    mc.panel.layer_boxes["b_notes"].setChecked(False)
    mc.panel.layer_boxes["user_notes"].setChecked(False)
    assert not w.view.roll.model_show["b_notes"]
    assert w.view.viewport().grab().width() > 0


def test_switch_analysis_source_protects_user_edits(qtbot, window, song, monkeypatch):
    from amp.lyrics import ops
    from tests.ui.test_vocal_ui import linking

    w = window
    monkeypatch.setattr("amp.app.session.TranscriptionPipeline", IdentityPipeline)
    folder = create(qtbot, w, song)
    w.run_analysis()
    qtbot.waitUntil(lambda: w.doc.project.analysis is not None and not w.session.busy, timeout=120_000)
    first = w.doc.score.notes()[0].id
    w.doc.move_notes([first], 0.0, 2)  # 사용자 편집 64 → 66
    w.lyrics_panel.applyTextRequested.emit("하나")
    ids = [s.id for s in w.doc.lyrics.text.syllables]
    notes = sorted(w.doc.score.notes(), key=lambda n: n.start)
    assert len(notes) == 3
    st = ops.set_times(w.doc.lyrics, {ids[0]: (notes[0].start, notes[0].end), ids[1]: (notes[1].start, notes[2].end)},
                       source="model")
    w.doc.push_lyrics(linking.auto_link(st, w.doc.score.notes()), "시험")
    w.view.roll.select_ids([notes[1].id, notes[2].id])
    w.lyrics.on_syllable_action("mel_set", [ids[1]])  # 사용자 멜리스마 판정 ('나' = 두 음표)
    lyrics_text = w.doc.lyrics.text
    melisma = dict(w.doc.lyrics.alignment.melisma)
    assert melisma
    _compare(qtbot, w, [ModelSpec("_identity"), ModelSpec("_half")])
    mc = w.model_compare
    keys = {w.session.model_store().entry(k).model: k for k in mc.outputs}
    assert keys["_identity"] == w.session.model_store().active_key()  # 처음 분석한 결과를 다시 쓰고 바꾸지 않음
    before = _hashes(folder)
    mc.keys["b"] = keys["_half"]
    seen = []
    monkeypatch.setattr(MC.SourceSwitchDialog, "exec",
                        lambda self: (seen.append(self.findChildren(MC.QLabel)[1].text()), QDialog.DialogCode.Accepted)[1])
    mc.use_as_source("b")
    assert seen and "사용자가 수정·추가한 음표 1개는 그대로" in seen[0] and "음절 시간 정렬" in seen[0]
    qtbot.waitUntil(lambda: w.session.manifest is not None and w.session.manifest.model == "_half", timeout=20_000)
    assert w.session.model_store().active_key() == keys["_half"]
    assert any(n.pitch == 66 for n in w.doc.score.notes())  # 사용자 편집 유지
    assert w.doc.lyrics.text == lyrics_text  # 가사 그대로
    assert set(w.doc.lyrics.alignment.melisma) == set(melisma)  # 수동 멜리스마 판정 유지 (필요하면 재확인 표시)
    assert w.doc.project.analysis.models["separation"].name == "_half"
    assert _hashes(folder) == before  # 어느 분리 WAV도 지우거나 바꾸지 않음
    w.doc.undo_stack.undo()  # 음표·가사 반영은 실행 취소 가능
    assert w.doc.project.analysis.models["separation"].name == "_identity"


def test_compare_selection_survives_save_and_reopen(qtbot, make_window, song):
    w = make_window("a")
    folder = create(qtbot, w, song)
    _compare(qtbot, w, [ModelSpec("_identity"), ModelSpec("_half")])
    keys = dict(w.model_compare.keys)
    assert (folder.paths.analysis / MC.STATE_FILE).is_file()
    assert w.save()
    root = str(folder.root)
    w.doc.mark_saved(w.doc.file_path or "")
    w.close()
    w2 = make_window("b")
    assert w2.open_project(root + "/project.json")
    qtbot.waitUntil(lambda: w2.session.audio is not None, timeout=20_000)
    w2.model_compare.act_mode.trigger()
    assert w2.model_compare.active and w2.model_compare.keys == keys
    assert w2.model_compare.panel.table.rowCount() == 2
    assert w2.model_compare.transcribe() is not None  # 다시 연 뒤 채보: 분리·피치 캐시를 다시 쓴다
    qtbot.waitUntil(lambda: not w2.session.busy, timeout=120_000)
    assert len(w2.model_compare.outputs) == 2, (w2.model_compare.panel.status.text(), w2.errors)
    assert all(o.reused["separation"] and o.reused["pitch"] for o in w2.model_compare.outputs.values())


def test_gpu_oom_asks_before_cpu_and_keeps_other_result(qtbot, window, song, monkeypatch):
    w = window
    create(qtbot, w, song)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.No)[1])
    _compare(qtbot, w, [ModelSpec("_identity"), ModelSpec("_oom", device="auto")])
    assert asked and "CPU로 바꾸지 않았습니다" in asked[0]
    assert "GPU 메모리 부족" in w.model_compare.panel.status.text()
    assert [e.model for e in w.session.model_store().entries()] == ["_identity"]
    # 사용자가 '예'를 고르면 그 모델만 CPU로 다시
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    _compare(qtbot, w, [ModelSpec("_oom", device="auto")])
    qtbot.waitUntil(lambda: not w.session.busy and len(w.session.model_store().entries()) == 2, timeout=120_000)
    assert sorted(e.model for e in w.session.model_store().entries()) == ["_identity", "_oom"]
