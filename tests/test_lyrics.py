"""STEP 3 가사 시험 (Qt 없음). 번호는 STEP 3 요구사항 14장의 시험 항목이다.

실제 노래에서 얼마나 정확한지는 여기서 재지 않는다 (STEP_3_IMPLEMENTATION_REPORT.md의 실제 곡 평가 참고).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from amp.core.errors import EditError
from amp.core.model import Project, Score
from amp.engine.context import AnalysisCancelled, CancelToken
from amp.lyrics import ctc, engine, io, linking, mms, ops, review, text as tx
from amp.lyrics.model import LyricsState, Recognition, AsrSegment, AsrWord, SyllableTiming
from amp.project import schema
from amp.project.analysis_io import LoadReport, load_analysis_files, load_lyrics_files, save_analysis_files
from amp.project.layout import ProjectPaths
from tests.conftest import note


def texts(state):
    return [s.text for s in state.text.syllables]


def timed(state, spans):
    """음절 순서대로 (시작, 끝)을 준다."""
    ids = [s.id for s in state.text.syllables]
    return ops.set_times(state, {ids[i]: span for i, span in enumerate(spans)}, source="model")


# --- 1. 한국어 음절 분리 ------------------------------------------------------------------

def test_1_korean_syllables():
    st = ops.apply_text(LyricsState(), "사랑해 너를,\n  \n다시  만나!")
    assert texts(st) == ["사", "랑", "해", "너", "를", "다", "시", "만", "나"]
    assert [ln.text for ln in st.text.lines] == ["사랑해 너를,", "", "다시  만나!"]  # 빈 줄(문단)도 남는다
    assert [w.text for w in st.text.lines[0].words] == ["사랑해", "너를,"]  # 문장부호는 단어 원문에 남는다
    assert st.text.language == "ko"
    assert [s.reading for s in st.text.syllables[:3]] == ["sa", "rang", "hae"]
    assert all(s.kind == "hangul" for s in st.text.syllables)
    # 맥에서 복사한 풀어쓴 한글(NFD)도 같은 음절이 된다
    import unicodedata

    nfd = unicodedata.normalize("NFD", "사랑해")
    assert texts(ops.apply_text(LyricsState(), nfd)) == ["사", "랑", "해"]
    # 낱자·숫자
    toks = tx.tokenize_word("ㅋㅋ123번").syllables
    assert [(t.text, t.kind) for t in toks] == [("ㅋ", "jamo"), ("ㅋ", "jamo"), ("123", "number"), ("번", "hangul")]
    assert toks[2].note and not toks[2].reading  # 숫자는 읽는 법을 정하지 않고 검토 안내를 남긴다


def test_1_hangul_reading_table_spot_checks():
    # uroman과 같은 자모 표 (전체 11,172자는 개발 중 대조)
    assert [tx.hangul_reading(c) for c in "꽃잎읽없괜의희닭뚫"] == ["ggoc", "ip", "ilg", "eobs", "gwain", "yi", "hyi",
                                                                 "dalg", "ddulh"]


# --- 2. 영어·일본어·섞인 텍스트 ------------------------------------------------------------

def test_2_english_and_japanese():
    words = {w: [t.text for t in tx.tokenize_word(w).syllables] for w in
             ("little", "table", "happy", "father", "singing", "love", "baby", "don't")}
    assert words == {"little": ["lit", "tle"], "table": ["ta", "ble"], "happy": ["hap", "py"], "father": ["fa", "ther"],
                     "singing": ["sing", "ing"], "love": ["love"], "baby": ["ba", "by"], "don't": ["don't"]}
    assert tx.tokenize_word("happy").syllables[0].note  # 규칙 기반 분리라는 표시
    kana = tx.tokenize_word("きゃっとラーメン").syllables
    assert [(t.text, t.reading) for t in kana] == [("きゃ", "kya"), ("っ", "t"), ("と", "to"), ("ラ", "ra"), ("ー", "a"),
                                                   ("メ", "me"), ("ン", "n")]
    kanji = tx.tokenize_word("東京の空").syllables
    assert [t.kind for t in kanji] == ["kanji", "kanji", "kana", "kanji"] and kanji[0].note and kanji[0].reading == ""
    mixed = ops.apply_text(LyricsState(), "OK사랑 Ｈｅｌｌｏ café")
    assert texts(mixed) == ["OK", "사", "랑", "Hel", "lo", "café"]
    assert mixed.text.syllables[-1].reading == "cafe"
    assert tx.detect_language("君の名は") == "ja" and tx.detect_language("I love you") == "en"


# --- 3. 수동 가사 입력과 저장 -------------------------------------------------------------

def _project_with_lyrics():
    st = ops.apply_text(LyricsState(), "사랑해\n아")
    st = timed(st, [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.5, 3.4)])
    notes = [note("n1", 1.0, 1.6, 60), note("n2", 1.6, 2.0, 62), note("m1", 2.5, 2.8, 64), note("m2", 2.8, 3.1, 65),
             note("m3", 3.1, 3.4, 67)]
    st = linking.auto_link(st, notes)
    return Project(score=Score(None, notes), lyrics=st)


def test_3_manual_lyrics_saved_in_own_files(tmp_path):
    paths = ProjectPaths(tmp_path)
    p = _project_with_lyrics()
    save_analysis_files(paths, p)
    lyr = json.loads(paths.lyrics_json.read_text(encoding="utf-8"))
    ali = json.loads(paths.alignment_json.read_text(encoding="utf-8"))
    assert lyr["schema"] == "sonatlas.lyrics" and ali["schema"] == "sonatlas.alignment"
    assert lyr["text"] == "사랑해\n아" and lyr["stats"]["syllables"] == 4
    row = ali["syllables"][0]
    assert set(row) >= {"syllable_id", "start", "end", "confidence", "start_source", "user_edited", "linked_note_ids"}
    assert "text" not in row  # 가사 글자는 lyrics.json에만 둔다 (두 파일이 어긋나지 않게)
    notes = json.loads(paths.notes_json.read_text(encoding="utf-8"))
    assert notes["schema_version"] == 2 and "lyrics" not in notes["edits"]
    assert all("syllable_id" not in n for n in notes["edits"]["notes"])
    report = LoadReport()
    _a, score = load_analysis_files(paths, report)
    loaded = load_lyrics_files(paths, report, [n.id for n in score.notes()], None)
    assert loaded.text == p.lyrics.text and loaded.alignment.timings == p.lyrics.alignment.timings
    assert loaded.alignment.links == p.lyrics.alignment.links
    assert not report.warnings


def test_3_saved_times_are_not_rounded(tmp_path):
    """다시 열었을 때 시간·신뢰도가 그대로여야 한다 (반올림하면 '저장 뒤 같은 상태'가 깨진다)."""
    paths = ProjectPaths(tmp_path)
    st = ops.apply_text(LyricsState(), "가나")
    a, b = [s.id for s in st.text.syllables]
    st = ops.set_times(st, {a: (1.23456789, 1.5000001)})
    st = LyricsState(st.text, st.alignment.with_timings({b: SyllableTiming(2.000123, 2.4, 0.1234567, "model", "inferred")}))
    save_analysis_files(paths, Project(score=Score(None, []), lyrics=st))
    report = LoadReport()
    loaded = load_lyrics_files(paths, report, [], None)
    assert loaded.alignment.timings == st.alignment.timings


def test_3_project_without_lyrics_writes_no_lyrics_files(tmp_path):
    paths = ProjectPaths(tmp_path)
    save_analysis_files(paths, Project(score=Score(None, [note("a", 0, 1)])))
    assert not paths.lyrics_json.exists() and not paths.alignment_json.exists()


# --- 4. 자동 인식 결과 수정 --------------------------------------------------------------

def _recognition():
    seg1 = AsrSegment("seg0", 1.0, 2.0, "사랑해 너를", -0.3, 0.9, 1.1, (),
                      (AsrWord(" 사랑해", 1.0, 1.6, 0.9), AsrWord(" 너를", 1.6, 2.0, 0.8)))
    seg2 = AsrSegment("seg1", 5.0, 6.0, "시청해 주셔서 감사합니다", -1.2, 0.95, 1.0, (review.R_PHRASE, review.R_LOW_CONF), ())
    return Recognition("turbo", "20250625", "ko", 0.98, "cpu", "t", {}, "stems-x", (seg1, seg2))


def test_4_edit_recognized_lyrics_keeps_raw_and_unchanged_syllables():
    rec = _recognition()
    st = engine.text_from_recognition(LyricsState(), rec)
    assert [ln.origin for ln in st.text.lines] == ["asr", "asr"]
    assert st.text.lines[0].review == "" and "환각" in st.text.lines[1].review  # 검토 필요 표시 (지우지 않음)
    t = [st.alignment.timing(s.id) for s in st.text.lines[0].syllables]
    assert (t[0].start, t[0].start_source) == (1.0, "asr_word")  # 단어 첫 음절은 인식 모델의 단어 시각
    assert t[1].start_source == "estimated"  # 단어 안 경계는 나눈 추정값 (검출값으로 표시하지 않음)
    assert t[2].end == pytest.approx(1.6) and t[2].end_source == "asr_word"
    ids = [s.id for s in st.text.syllables]
    # 오타 고치기 + 환각 줄 지우기
    edited = ops.apply_text(st, "사랑해 너만")
    assert texts(edited) == ["사", "랑", "해", "너", "만"]
    assert [s.id for s in edited.text.syllables] == ids[:5]  # 같은 자리 음절은 id(시간·연결)를 잇는다
    assert edited.alignment.timing(ids[4]).stale  # 글자가 바뀐 음절은 다시 정렬 필요 표시
    assert not edited.alignment.timing(ids[0]).stale
    assert edited.text.lines[0].review == ""
    assert edited.recognition == rec  # 인식 원본은 그대로 남는다
    assert edited.history[-1].action == "가사 수정"


def test_4_partial_replace_and_repeat_lines():
    st = ops.apply_text(LyricsState(), "첫 줄\n후렴 가사\n끝")
    st = timed(st, [(i, i + 0.5) for i in range(len(st.text.syllables))])
    first_ids = [s.id for s in st.text.lines[0].syllables]
    st2 = ops.replace_lines(st, 2, 2, "마지막 줄")
    assert [s.id for s in st2.text.lines[0].syllables] == first_ids
    assert st2.text.lines[2].text == "마지막 줄"
    st3 = ops.repeat_lines(st2, [st2.text.lines[1].id], times=2)
    assert [ln.text for ln in st3.text.lines] == ["첫 줄", "후렴 가사", "후렴 가사", "후렴 가사", "마지막 줄"]
    assert st3.text.lines[2].origin == "repeat"
    rep_ids = {s.id for s in st3.text.lines[2].syllables}
    assert not rep_ids & {s.id for s in st3.text.lines[1].syllables}  # 반복 줄은 따로 정렬되는 새 음절
    assert all(not st3.alignment.timing(i).aligned for i in rep_ids)


def test_4_unchanged_lines_keep_manual_merge():
    st = ops.apply_text(LyricsState(), "東京 love\n두 번째")
    k = [s.id for s in st.text.lines[0].syllables]
    st, merged = ops.merge_syllables(st, k[:2])
    assert texts(st)[:2] == ["東京", "love"]
    st2 = ops.apply_text(st, "東京 love\n두 번째 줄")  # 다른 줄을 고쳐도 손으로 합친 음절은 그대로
    assert texts(st2)[:2] == ["東京", "love"] and st2.text.syllables[0].id == merged


# --- 5. 음절 시간 경계 수정 ---------------------------------------------------------------

def test_5_syllable_time_edits_and_validation():
    st = timed(ops.apply_text(LyricsState(), "가나다"), [(1.0, 1.5), (1.5, 2.0), (2.0, 2.4)])
    a, b, c = [s.id for s in st.text.syllables]
    st2 = ops.set_times(st, {a: (1.0, 1.4), b: (1.4, 2.0)}, duration=10.0)  # 맞닿은 경계를 함께 옮김
    t = st2.alignment.timing(a)
    assert (t.end, t.start_source, t.user_edited, t.confidence) == (1.4, "user", True, None)
    assert st2.alignment.timing(b).start == 1.4
    for bad in ((2.0, 1.0), (math.nan, 1.0), (-0.5, 1.0), (1.0, math.inf), (1.0, 1.005), (12.0, 13.0)):
        with pytest.raises(EditError):
            ops.set_times(st, {a: bad}, duration=10.0)
    st3 = ops.shift_times(st, None, -5.0)  # 전체 오프셋: 0초 아래로는 가지 않는다
    assert st3.alignment.timing(a).start == 0.0
    kanji = ops.apply_text(LyricsState(), "東京")  # 한자는 한 글자씩 → 합쳤다가 다시 나눈다
    kanji, merged = ops.merge_syllables(kanji, [s.id for s in kanji.text.syllables])
    kanji = timed(kanji, [(1.0, 2.0)])
    st4, new = ops.split_syllable(kanji, merged, 1)
    assert texts(st4) == ["東", "京"] and st4.text.syllables[0].id == merged
    assert st4.alignment.timing(merged).end == st4.alignment.timing(new).start == pytest.approx(1.5)
    with pytest.raises(EditError):
        ops.split_syllable(st, a, 1)  # 한 글자 음절은 나누지 않는다 (멜리스마는 음표 연결로)


# --- 6·7·8. 음절과 음표 연결 ---------------------------------------------------------------

def test_6_one_to_one():
    st = timed(ops.apply_text(LyricsState(), "가나"), [(1.0, 1.5), (1.5, 2.0)])
    st = linking.auto_link(st, [note("x", 1.02, 1.48), note("y", 1.52, 1.98)])
    a, b = [s.id for s in st.text.syllables]
    assert st.alignment.linked_notes(a) == ("x",) and st.alignment.linked_notes(b) == ("y",)
    assert linking.relation_kind(st, a) == "1:1"
    lk = st.alignment.links[a][0]
    assert lk.source == "auto" and not lk.confirmed and 0 < lk.confidence <= 1  # 자동 연결은 확정이 아니다


def test_7_one_syllable_many_notes_melisma():
    st = timed(ops.apply_text(LyricsState(), "아"), [(1.0, 2.2)])
    st = linking.auto_link(st, [note("c", 1.0, 1.4, 60), note("d", 1.4, 1.8, 62), note("e", 1.8, 2.2, 64)])
    sid = st.text.syllables[0].id
    assert st.alignment.linked_notes(sid) == ("c", "d", "e")
    assert linking.relation_kind(st, sid) == "1:N"
    assert min(lk.confidence for lk in st.alignment.links[sid]) > 0.5  # 음절 안의 뒤쪽 음표도 낮게 매기지 않는다


def test_8_many_syllables_one_note():
    st = timed(ops.apply_text(LyricsState(), "사랑"), [(1.0, 1.3), (1.3, 1.6)])
    st = linking.auto_link(st, [note("n", 1.0, 1.6)])  # 음표 검출이 같은 높이로 다시 부른 두 음절을 합친 경우
    a, b = [s.id for s in st.text.syllables]
    assert st.alignment.linked_notes(a) == ("n",) == st.alignment.linked_notes(b)
    assert linking.relation_kind(st, a) == "N:1" == linking.relation_kind(st, b)
    assert st.alignment.note_to_syllables["n"] == (a, b)


def test_user_links_are_kept_by_auto_link_and_sequential_linking():
    st = timed(ops.apply_text(LyricsState(), "가나다"), [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5)])
    a, b, c = [s.id for s in st.text.syllables]
    notes = [note("x", 1.0, 1.5), note("y", 1.5, 2.0), note("z", 2.0, 2.5)]
    st = linking.set_links(st, a, ["z"])  # 사용자가 일부러 다르게 이음
    st = linking.auto_link(st, notes)
    assert st.alignment.linked_notes(a) == ("z",) and st.alignment.links[a][0].confirmed
    plain = ops.apply_text(LyricsState(), "하나둘")
    seq = linking.link_sequential(plain, [s.id for s in plain.text.syllables], list(reversed(notes)))
    assert [seq.alignment.linked_notes(s.id) for s in plain.text.syllables] == [("x",), ("y",), ("z",)]
    assert seq.alignment.timing(plain.text.syllables[1].id).start == 1.5  # 시간도 음표에서 가져옴 (출처 notes)
    assert seq.alignment.timing(plain.text.syllables[1].id).start_source == "notes"


# --- 9. 음표 나누기·합치기 뒤 연결 유지 -------------------------------------------------------

def test_9_split_merge_delete_keep_syllables_and_remap_links():
    st = timed(ops.apply_text(LyricsState(), "사랑해"), [(1.0, 1.3), (1.3, 1.6), (1.6, 2.4)])
    a, b, c = [s.id for s in st.text.syllables]
    n = note("n", 1.0, 1.6)
    h = note("h", 1.6, 2.4)
    st = linking.auto_link(st, [n, h])
    st = linking.confirm_links(st, [c])
    before_text = st.text
    # N:1 음표를 음절 경계에서 나누면 각 음절이 자기 조각에 이어진다
    left, right = note("l", 1.0, 1.3), note("r", 1.3, 1.6)
    s1 = linking.reconcile(st, [n], [left, right])
    assert s1.alignment.linked_notes(a) == ("l",) and s1.alignment.linked_notes(b) == ("r",)
    # 1:1 음표를 음절 안에서 나누면 두 조각 모두 그 음절 (멜리스마), 확정 표시는 유지
    h1, h2 = note("h1", 1.6, 2.0), note("h2", 2.0, 2.4)
    s2 = linking.reconcile(s1, [h], [h1, h2])
    assert s2.alignment.linked_notes(c) == ("h1", "h2") and all(lk.confirmed for lk in s2.alignment.links[c])
    # 다시 합치면 합친 음표 하나로
    m = note("m", 1.6, 2.4)
    s3 = linking.reconcile(s2, [h1, h2], [m])
    assert s3.alignment.linked_notes(c) == ("m",)
    # 지우면 연결만 사라지고 음절은 '연결 없음'으로 남는다
    s4 = linking.reconcile(s3, [m], [])
    assert c in s4.unlinked() and s4.text == before_text and s4.alignment.timings == st.alignment.timings
    # 이동(같은 id)은 연결을 바꾸지 않는다
    assert linking.reconcile(st, [n], [n.with_(start=1.1)]) is st


# --- 10. 가사 수정 뒤 재정렬 -------------------------------------------------------------

def _fake_emissions(path: Path, readings: list[str], starts: list[float], seconds: float):
    """글자가 정해진 시각에 튀는 합성 CTC 출력."""
    dt = mms.HOP / mms.SAMPLE_RATE
    n = int(seconds / dt)
    lp = np.full((n, len(mms.LABELS)), -8.0, dtype=np.float32)
    lp[:, mms.BLANK] = -0.01
    lp[:, mms.STAR] = 0.0
    for reading, t in zip(readings, starts):
        f = int(t / dt)
        for k, ch in enumerate(reading):
            lp[f + k, :] = -8.0
            lp[f + k, mms.LABEL_INDEX[ch]] = -0.05
    rms = np.full(n, -20.0, dtype=np.float32)
    np.savez(path, logp=lp.astype(np.float16), rms_db=rms, frame_s=np.float64(dt))


class FakeRunner:
    """작업 프로세스 대신: asr·emissions는 정해진 결과를 쓰고, ctc_align은 실제 numpy 정렬을 돈다."""

    def __init__(self, readings=(), starts=(), seconds=10.0, segments=None, fail=None):
        self.calls = []
        self.readings, self.starts, self.seconds = list(readings), list(starts), seconds
        self.segments = segments or []
        self.fail = fail

    def __call__(self, command, args, progress=None, token=None, numba=None):
        self.calls.append(command)
        if self.fail and command == self.fail[0]:
            raise self.fail[1]
        if command == "asr":
            Path(args["out_dir"], "asr.json").write_text(json.dumps(
                {"model": "turbo", "language": "ko", "device": "cpu", "segments": self.segments}), encoding="utf-8")
            return {}
        if command == "emissions":
            _fake_emissions(Path(args["out_dir"], "emissions.npz"), self.readings, self.starts, self.seconds)
            Path(args["out_dir"], "info.json").write_text(json.dumps({"model": "MMS_FA", "device": "cpu"}), encoding="utf-8")
            return {}
        if command == "ctc_align":
            from amp.engine.worker import cmd_ctc_align

            return cmd_ctc_align(args)
        raise AssertionError(command)


def _req(tmp_path, input_id="stems-a", use_cache=True):
    return engine.LyricsRequest(str(tmp_path / "vocals.wav"), input_id, str(tmp_path / "cache"), str(tmp_path / "w"),
                                str(tmp_path / "m"), use_cache=use_cache)


def test_10_realign_only_changed_part(tmp_path):
    st = ops.apply_text(LyricsState(), "사랑 해요")
    readings = [s.reading for s in st.text.syllables]
    runner = FakeRunner(readings, [1.0, 2.0, 3.0, 4.0], 6.0)
    out = engine.LyricsEngine(runner).align(_req(tmp_path), st, duration=6.0)
    starts = [out.state.alignment.timing(s.id).start for s in st.text.syllables]
    assert starts == pytest.approx([1.0, 2.0, 3.0, 4.0], abs=0.03)
    assert all(out.state.alignment.timing(s.id).start_source == "model" for s in st.text.syllables)
    assert out.state.alignment.timing(st.text.syllables[0].id).end_source == "inferred"
    # 사용자가 첫 음절을 고치고, 가사 끝 단어를 바꾼다
    st2 = ops.set_times(out.state, {st.text.syllables[0].id: (0.9, 1.5)})
    st2 = ops.apply_text(st2, "사랑 해서")
    units, anchors, targets = engine.plan_units(st2, "changed")
    changed = st2.text.syllables[3].id
    assert targets == {changed} and len(anchors) == 3
    out2 = engine.LyricsEngine(runner).align(_req(tmp_path), st2, mode="changed", duration=6.0)
    assert out2.reused  # 정렬 모델 출력은 캐시를 쓰고 numpy 정렬만 다시 돈다
    assert out2.aligned == [changed] and not out2.state.alignment.timing(changed).stale
    assert out2.state.alignment.timing(st.text.syllables[0].id).start == 0.9  # 사용자 값 보호
    # 전체 재정렬도 사용자가 고친 값은 건드리지 않는다
    out3 = engine.LyricsEngine(runner).align(_req(tmp_path), st2, mode="all", duration=6.0)
    assert out3.state.alignment.timing(st.text.syllables[0].id).start == 0.9


# --- 11. 저장·다시 열기 (작업 폴더·복구 파일·이전 형식) ----------------------------------------------

def test_11_notes_v1_with_step1_lyrics_is_migrated(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.analysis.mkdir(parents=True)
    v1 = {"schema": schema.NOTES, "schema_version": 1, "analysis": None,
          "edits": {"notes": [{"id": "a", "start": 0.0, "end": 1.0, "pitch": 60, "syllable_id": "s1"},
                              {"id": "b", "start": 1.0, "end": 2.0, "pitch": 62, "syllable_id": "s1"}],
                    "lyrics": {"language": "ko", "syllables": [{"id": "s1", "text": "가", "line": 0, "start": 0.0, "end": 2.0}]}}}
    paths.notes_json.write_text(json.dumps(v1), encoding="utf-8")
    report = LoadReport()
    _a, score = load_analysis_files(paths, report)
    st = load_lyrics_files(paths, report, [n.id for n in score.notes()], None)
    assert "notes.json" in report.migrated
    assert texts(st) == ["가"] and st.alignment.linked_notes("s1") == ("a", "b")
    assert st.alignment.timing("s1").start_source == "legacy"


def test_11_recovery_file_keeps_lyrics(tmp_path):
    from amp.storage.project_file import load_project, save_project

    p = _project_with_lyrics()
    path = str(tmp_path / "r.ampproj")
    save_project(p, path)
    q = load_project(path)
    assert q.lyrics.text == p.lyrics.text and q.lyrics.alignment.links == p.lyrics.alignment.links


# --- 12. 캐시 재사용 ------------------------------------------------------------------

def _segments():
    return [{"start": 1.0, "end": 2.0, "text": "사랑해", "avg_logprob": -0.2, "no_speech_prob": 0.9,
             "compression_ratio": 1.0, "activity": 0.9, "words": [{"text": "사랑해", "start": 1.0, "end": 2.0,
                                                                    "probability": 0.9}]}]


def test_12_recognition_cache_reuse_and_invalidation(tmp_path):
    runner = FakeRunner(segments=_segments())
    rec, reused = engine.LyricsEngine(runner).recognize(_req(tmp_path))
    assert not reused and rec.segments[0].text == "사랑해"
    rec2, reused2 = engine.LyricsEngine(runner).recognize(_req(tmp_path))
    assert reused2 and runner.calls.count("asr") == 1 and rec2.segments == rec.segments
    engine.LyricsEngine(runner).recognize(_req(tmp_path, input_id="stems-b"))  # 보컬을 다시 분리하면 새로 인식
    engine.LyricsEngine(runner).recognize(_req(tmp_path, use_cache=False))
    assert runner.calls.count("asr") == 3


# --- 13. 취소·실패 뒤 데이터 ------------------------------------------------------------------

def test_13_cancel_and_failure_leave_state_and_cache_clean(tmp_path):
    from amp.engine.cache import AnalysisCache

    st = ops.apply_text(LyricsState(), "가나")
    before = st
    runner = FakeRunner(fail=("emissions", AnalysisCancelled()))
    with pytest.raises(AnalysisCancelled):
        engine.LyricsEngine(runner).align(_req(tmp_path), st, token=CancelToken())
    assert st == before  # 가사 상태는 불변 객체라 그대로
    cache = AnalysisCache(tmp_path / "cache")
    assert cache.lookup("stems-a", "mms", {"model": "mms_fa_uroman", "device": "auto", "v": engine.MMS_VERSION}) is None
    leftovers = [p for p in (tmp_path / "cache").rglob("*") if p.is_dir() and ".tmp" in p.name]
    assert not leftovers  # 반쯤 쓴 캐시가 남지 않는다
    # 손상된 캐시는 지우고 다시 계산한다
    ok = FakeRunner(["ga", "na"], [1.0, 2.0], 4.0)
    engine.LyricsEngine(ok).align(_req(tmp_path), st)
    em = next((tmp_path / "cache").rglob("emissions.npz"))
    em.write_bytes(b"broken")
    out = engine.LyricsEngine(ok).align(_req(tmp_path), st)
    assert not out.reused and ok.calls.count("emissions") == 2 and len(out.aligned) == 2


# --- 14. 잘못된 시간 경계 ---------------------------------------------------------------------

def test_14_invalid_times_are_rejected_or_reset():
    st = ops.apply_text(LyricsState(), "가나다")
    a, b, c = [s.id for s in st.text.syllables]
    raw = {"syllables": [{"syllable_id": a, "start": "NaN", "end": 1.0},
                         {"syllable_id": b, "start": 2.0, "end": 1.0},
                         {"syllable_id": c, "start": -1.0, "end": 0.5},
                         {"syllable_id": "ghost", "start": 1.0, "end": 2.0, "linked_note_ids": ["x"]}]}
    warnings: list[str] = []
    st2 = LyricsState(st.text, io.alignment_from_dict(raw, warnings))
    st3, w2 = ops.sanitize_alignment(st2, duration=10.0)
    assert not any(st3.alignment.timing(x).aligned for x in (a, b, c))
    assert "ghost" not in st3.alignment.links and w2
    # 정렬기가 이상한 값을 주면 받지 않고 미정렬로 남긴다
    results = [{"id": a, "start": 1.0, "end": 1.5}, {"id": b, "start": float("nan"), "end": 2.0},
               {"id": c, "start": 3.0, "end": 2.0}]
    new, ok, failed = engine.apply_results(st, results, {a, b, c}, {}, {"mode": "all"}, 10.0)
    assert ok == [a] and set(failed) == {b, c}
    # 순서가 뒤바뀐 시간은 찾아서 알려 준다
    bad = ops.set_times(st, {a: (2.0, 2.5), b: (1.0, 1.5)})
    assert any("순서" in why for _sid, why in ops.timing_issues(bad))
    # 저장 파일에는 NaN·무한대가 들어가지 않는다
    weird = LyricsState(st.text, st.alignment.with_timings({a: SyllableTiming(1.0, 2.0, float("nan"), "model", "inferred")}))
    json.dumps(io.alignment_to_dict(weird), allow_nan=False)


# --- 정렬기·모델·검토 규칙 -------------------------------------------------------------------

def test_ctc_viterbi_finds_spikes_and_respects_anchors():
    st = ops.apply_text(LyricsState(), "가 나 다")
    readings = [s.reading for s in st.text.syllables]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.npz"
        _fake_emissions(p, readings, [1.0, 2.0, 3.0], 5.0)
        z = np.load(p)
        lp = z["logp"].astype(np.float32)
    units = [ctc.Unit(s.id, tuple(mms.LABEL_INDEX[c] for c in s.reading)) for s in st.text.syllables]
    res, warn, _rel = ctc.align(lp, units, dt=0.02, star=mms.STAR)
    assert [round(r.start, 2) for r in res] == [1.0, 2.0, 3.0] and not warn
    assert res[0].end == pytest.approx(2.0)  # 끝은 다음 음절 시작 (추정)
    res2, _w, _r = ctc.align(lp, units, dt=0.02, star=mms.STAR, anchors={1: (2.5, 2.8)})
    assert [r.syllable_id for r in res2] == [units[0].syllable_id, units[2].syllable_id]
    assert res2[0].end <= 2.5  # 고정한 음절을 넘지 않는다
    with pytest.raises(ctc.AlignmentFailed):
        ctc.viterbi(lp[:2], np.array([1, 2, 3]))


def test_ctc_releases_model_anchors_when_new_syllables_do_not_fit():
    """가사를 고친 뒤 '바뀐 부분만' 정렬: 틀리게 놓인 모델 고정점 사이에 자리가 없으면 그 고정점을 풀어 함께 정렬한다."""
    st = ops.apply_text(LyricsState(), "가 나 다 라")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.npz"
        _fake_emissions(p, [s.reading for s in st.text.syllables], [1.0, 2.0, 2.5, 3.5], 5.0)
        lp = np.load(p)["logp"].astype(np.float32)
    units = [ctc.Unit(s.id, tuple(mms.LABEL_INDEX[c] for c in s.reading)) for s in st.text.syllables]
    anchors = {0: (1.0, 2.9), 3: (2.95, 4.0)}  # 앞뒤 음절이 잘못 넓게 정렬돼 '나·다'가 들어갈 자리가 없다
    res, warn, rel = ctc.align(lp, units, dt=0.02, star=mms.STAR, anchors=anchors)
    assert not res and warn and not rel  # 사용자 고정점(풀 수 없음)이면 실패로 남긴다
    res, warn, rel = ctc.align(lp, units, dt=0.02, star=mms.STAR, anchors=anchors, soft={0, 3})
    assert rel == [0, 3] and not warn
    assert [round(r.start, 2) for r in res] == [1.0, 2.0, 2.5, 3.5]


def test_ctc_unknown_reading_is_estimated_not_detected():
    units = [ctc.Unit("k", ()), ctc.Unit("n", (mms.LABEL_INDEX["n"], mms.LABEL_INDEX["o"]))]
    lp = np.full((100, len(mms.LABELS)), -8.0, dtype=np.float32)
    lp[:, 0] = -0.01
    lp[:, mms.STAR] = 0.0
    lp[60, :] = -8.0
    lp[60, mms.LABEL_INDEX["n"]] = -0.01
    lp[61, :] = -8.0
    lp[61, mms.LABEL_INDEX["o"]] = -0.01
    res, _w, _r = ctc.align(lp, units, dt=0.02, star=mms.STAR)
    by = {r.syllable_id: r for r in res}
    assert by["k"].start_source == "estimated" and by["n"].start_source == "model"


def test_mms_model_matches_public_checkpoint_layout():
    torch = pytest.importorskip("torch")
    model = mms.build_model()
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    lines = [ln.split() for ln in Path(__file__).with_name("data").joinpath("mms_fa_keys.txt").read_text(
        encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    ckpt = {k: tuple(int(x) for x in s.split("x")) for k, s in lines}
    fake = {k: torch.zeros(s) for k, s in ckpt.items()}
    fake["encoder.transformer.pos_conv_embed.conv.weight_g"] = torch.ones(ckpt["encoder.transformer.pos_conv_embed.conv.weight_g"])
    fake["encoder.transformer.pos_conv_embed.conv.weight_v"] = torch.ones(ckpt["encoder.transformer.pos_conv_embed.conv.weight_v"])
    conv = mms.convert_state_dict(fake)
    assert {k: tuple(v.shape) for k, v in conv.items()} == ours  # strict 로딩과 같은 조건
    assert ours["aux.weight"] == (len(mms.LABELS) - 1, 1024)
    # weight_norm(dim=2) 합치기: 커널 위치마다 노름이 g가 된다
    g = torch.full((1, 1, 128), 2.0)
    v = torch.randn(1024, 64, 128)
    w = mms.convert_state_dict({**fake, "encoder.transformer.pos_conv_embed.conv.weight_g": g,
                                "encoder.transformer.pos_conv_embed.conv.weight_v": v})["encoder.transformer.pos_conv_embed.conv.weight"]
    assert torch.allclose(w.norm(dim=(0, 1)), torch.full((128,), 2.0), atol=1e-4)
    model.load_state_dict(conv, strict=True)
    out = model(torch.zeros(1, 16000))
    assert out.shape == (1, 49, len(mms.LABELS)) and torch.all(out[..., -1] == 0)  # 마지막 열 '*'는 log 1


def test_review_flags_likely_hallucinations_without_deleting():
    words = (AsrWord("a", 0, 1, 0.1), AsrWord("b", 1, 2, 0.2))
    assert review.R_PHRASE in review.review_segment("시청해주셔서 감사합니다", 0, 3)
    assert review.R_NO_VOCAL in review.review_segment("노래", 0, 3, activity=0.05)
    assert review.R_NO_VOCAL_RANGE in review.review_segment("노래", 0, 3, in_no_vocal=0.9)
    assert review.R_NO_VOCAL in review.review_segment("아멘", 0, 3, level_db=-30.0)  # 보컬이 거의 없는 곳 (실제 곡 −30dB)
    assert review.review_segment("그대 따라 갈 이 언덕에", 0, 3, level_db=-4.0) == []
    assert review.R_PHRASE in review.review_segment("감사합니다.", 0, 2)  # 구간 전체가 끝인사뿐
    assert review.R_PHRASE not in review.review_segment("정말 감사합니다 그대여", 0, 3)
    assert review.review_segment("노래 가사", 0, 3, avg_logprob=-0.3, no_speech_prob=0.95, activity=0.9) == []  # 노래는 말소리 판정이 높게 나와도 정상
    assert any("확신이 낮은 단어" in r for r in review.review_segment("a b", 0, 2, words=words))
    segs = [AsrSegment(f"s{i}", i, i + 1, "같은 말") for i in range(4)]
    marked = review.mark_repeats(segs)
    assert [bool(s.review) for s in marked] == [False, False, True, True]
    rec = engine.recognition_from_raw({"model": "turbo", "language": "ko", "segments": [
        {"start": 1, "end": 2, "text": "다음 영상에서 만나요", "activity": 0.9}, {"start": 3, "end": 3.5, "text": " "}]})
    assert len(rec.segments) == 1 and rec.segments[0].review  # 빈 구간은 빼고, 의심 구간은 표시만 한다


def test_auto_model_is_turbo():
    from amp.lyrics.asr import resolve_model

    assert resolve_model("auto", "cpu") == resolve_model("auto", "cuda") == "turbo"
    assert resolve_model("small", "cpu") == "small"
