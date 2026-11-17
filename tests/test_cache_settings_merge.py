
import pytest

from amp.core.model import AnalysisResult, Project, Score
from amp.core.tempo import TempoMap
from amp.engine.cache import AnalysisCache, file_content_key, params_key
from amp.engine.merge import KEEP_EDITS, REPLACE_ALL, plan_reanalysis, summarize
from amp.engine.settings import AnalysisSettings, SettingsError
from tests.conftest import note


# --- 캐시 (요구 9, 12) -------------------------------------------------------

def test_cache_roundtrip_and_reuse(tmp_path):
    cache = AnalysisCache(tmp_path / "한글 캐시 (특수문자 #&)")
    params = {"tracker": "pyin", "v": 1}
    assert cache.lookup("abc", "pitch", params) is None
    with cache.writer("abc", "pitch", params) as d:
        (d / "pitch.npz").write_bytes(b"data")
    hit = cache.lookup("abc", "pitch", params)
    assert hit is not None and (hit / "pitch.npz").read_bytes() == b"data"
    assert cache.lookup("abc", "pitch", {"tracker": "pyin", "v": 2}) is None  # 설정·버전이 바뀌면 다른 키
    assert cache.size_bytes() > 0


def test_cache_writer_cleans_up_on_failure(tmp_path):
    cache = AnalysisCache(tmp_path)
    with pytest.raises(RuntimeError):
        with cache.writer("k", "pitch", {}) as d:
            (d / "partial").write_bytes(b"x")
            raise RuntimeError("취소")
    assert cache.lookup("k", "pitch", {}) is None
    assert [p.name for p in (tmp_path / "k").iterdir()] == []  # 임시 폴더도 남지 않는다


def test_cache_overwrite_and_clear(tmp_path):
    cache = AnalysisCache(tmp_path / "c")
    for content in (b"old", b"new"):
        with cache.writer("k", "tempo", {}) as d:
            (d / "tempo.json").write_bytes(content)
    assert (cache.lookup("k", "tempo", {}) / "tempo.json").read_bytes() == b"new"
    cache.clear()
    assert not (tmp_path / "c").exists()


def test_content_key_depends_on_content_not_name(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "노래.wav"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert file_content_key(str(a)) == file_content_key(str(b))
    b.write_bytes(b"diff")
    assert file_content_key(str(a)) != file_content_key(str(b))
    assert params_key({"a": 1, "b": 2}) == params_key({"b": 2, "a": 1})


# --- 설정 --------------------------------------------------------------------

def test_settings_presets_and_validation():
    s = AnalysisSettings()
    assert s.min_note_ms == 100 and s.pitch_change_semitones == 1.0
    simple = AnalysisSettings.for_sensitivity("simple", s)
    precise = AnalysisSettings.for_sensitivity("precise", s)
    assert simple.min_note_ms > s.min_note_ms > precise.min_note_ms
    assert simple.note_params().split_strength is None and precise.note_params().ornament_min_s
    with pytest.raises(SettingsError):
        AnalysisSettings(pitch_low=80, pitch_high=60)
    assert AnalysisSettings.from_dict({"min_note_ms": 120, "unknown": 1}).min_note_ms == 120
    assert AnalysisSettings.from_dict({"min_note_ms": -5}) == AnalysisSettings()


# --- 재분석 병합 (요구 7) -------------------------------------------------------

def _project():
    a = AnalysisResult(5.0, TempoMap.constant(100), (note("a", 0, 1, 60), note("b", 1, 2, 62), note("c", 2, 3, 64)))
    p = Project(analysis=a, score=Score.from_analysis(a))
    p.score.replace_notes(["b"], [p.score.get("b").with_(pitch=63)])  # 수정
    p.score.replace_notes([], [note("u", 3.5, 4.0, 70)])  # 사용자 추가
    return p


def test_keep_edits_protects_user_notes():
    p = _project()
    new = AnalysisResult(5.0, TempoMap.constant(110), (note("x", 0, 1, 61), note("y", 1.05, 1.95, 62), note("z", 3.6, 4.4, 71)))
    s = summarize(p)
    assert (s.original, s.modified, s.user) == (2, 1, 1) and s.has_edits
    plan = plan_reanalysis(p, new, KEEP_EDITS)
    assert {n.id for n in plan.kept} == {"b", "u"}
    assert {n.id for n in plan.removed} == {"a", "c"}  # 손대지 않은 원본만 교체
    assert {n.id for n in plan.added} == {"x"}  # y·z는 보존 음표와 겹쳐서 제외
    assert {n.id for n in plan.dropped} == {"y", "z"}
    assert plan.tempo.initial_bpm == 110  # 템포를 고치지 않았으면 새 분석 템포


def test_keep_edits_keeps_user_tempo_and_replace_all():
    p = _project()
    p.score.tempo = TempoMap.constant(90)
    new = AnalysisResult(5.0, TempoMap.constant(110), (note("x", 0, 1, 61),))
    assert plan_reanalysis(p, new, KEEP_EDITS).tempo.initial_bpm == 90
    plan = plan_reanalysis(p, new, REPLACE_ALL)
    assert len(plan.removed) == 4 and [n.id for n in plan.added] == ["x"] and plan.tempo.initial_bpm == 110
