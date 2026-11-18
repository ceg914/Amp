import json
import os
import zipfile

import numpy as np
import pytest

from amp.audio.data import AudioData
from amp.core.errors import ProjectFormatError
from amp.core.model import AudioRef, Note, Project, Score
from amp.engine.pipeline import AnalysisPipeline
from amp.engine.registry import default_registry
from amp.storage.project_file import load_project, save_project
from tests.conftest import note


@pytest.fixture
def analysed_project(tmp_path):
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"x")
    result = AnalysisPipeline(default_registry()).run(AudioData(np.zeros((22050 * 20, 1), np.float32), 22050))
    from amp.lyrics.io import from_legacy
    from amp.storage.serialization import lyrics_to_dict

    lyrics = from_legacy(lyrics_to_dict(result.lyrics), dict(result.lyric_links))  # 샘플 가사를 프로젝트 가사로
    return Project(AudioRef(str(audio_path), "fp", 20.0, 22050, 1), result, Score.from_analysis(result), lyrics)


def test_roundtrip_keeps_original_and_edits_separate(tmp_path, analysed_project):
    p = analysed_project
    first = p.score.notes()[0]
    p.score.replace_notes([first.id], [first.with_(pitch=first.pitch + 2)])
    p.score.replace_notes([], [note("user1", 0.0, 0.5, 70)])
    path = str(tmp_path / "a.ampproj")
    save_project(p, path)

    loaded = load_project(path)
    assert loaded.analysis.notes == p.analysis.notes
    assert loaded.analysis.pitch_curve == p.analysis.pitch_curve
    assert loaded.lyrics == p.lyrics  # 가사는 분석 원본이 아니라 프로젝트 가사(lyrics.json)에 있다
    assert len(loaded.lyrics.text.syllables) > 0 and loaded.lyrics.alignment.links
    assert loaded.analysis.models == p.analysis.models
    assert loaded.score.notes() == p.score.notes()
    assert loaded.score.tempo == p.score.tempo
    assert loaded.note_status(loaded.score.get(first.id)) == "modified"
    assert loaded.note_status(loaded.score.get("user1")) == "user"
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
    assert {"manifest.json", "project.json", "score.json", "analysis/result.json", "analysis/pitch.npz"} <= names


def test_project_without_analysis(tmp_path):
    path = str(tmp_path / "b.ampproj")
    save_project(Project(score=Score(notes=[note("a", 0, 1)])), path)
    loaded = load_project(path)
    assert loaded.analysis is None and loaded.audio is None
    assert [n.id for n in loaded.score.notes()] == ["a"]


def test_relative_audio_path_is_used_when_moved(tmp_path, analysed_project):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    path = str(proj_dir / "c.ampproj")
    save_project(analysed_project, path)
    # 프로젝트와 오디오를 함께 다른 폴더로 옮긴 상황
    moved = tmp_path / "moved"
    moved.mkdir()
    os.replace(path, moved / "proj.ampproj")
    (tmp_path / "song.wav").rename(tmp_path / "song_old.wav")
    (moved.parent / "song.wav").write_bytes(b"x")
    loaded = load_project(str(moved / "proj.ampproj"))
    assert os.path.exists(loaded.audio.path)


def test_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    path = str(tmp_path / "d.ampproj")
    save_project(Project(score=Score(notes=[note("a", 0, 1)])), path)
    before = open(path, "rb").read()

    import amp.storage.project_file as pf

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pf, "project_to_members", lambda *a, **k: {"x": b"1"})
    monkeypatch.setattr(pf.zipfile.ZipFile, "writestr", boom)
    with pytest.raises(OSError):
        save_project(Project(), path)
    assert open(path, "rb").read() == before
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_invalid_files(tmp_path):
    bad = tmp_path / "bad.ampproj"
    bad.write_bytes(b"garbage")
    with pytest.raises(ProjectFormatError):
        load_project(str(bad))
    with pytest.raises(ProjectFormatError):
        load_project(str(tmp_path / "none.ampproj"))

    newer = tmp_path / "newer.ampproj"
    with zipfile.ZipFile(newer, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "amp-project", "format_version": 99}))
    with pytest.raises(ProjectFormatError, match="새로운 버전"):
        load_project(str(newer))

    broken_note = tmp_path / "note.ampproj"
    with zipfile.ZipFile(broken_note, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "amp-project", "format_version": 1}))
        z.writestr("project.json", json.dumps({"audio": None}))
        z.writestr("score.json", json.dumps({"tempo": {"points": [{"time": 0, "bpm": 120}]},
                                              "notes": [{"id": "a", "start": 1, "end": 0.5, "pitch": 60}]}))
    with pytest.raises(ProjectFormatError):
        load_project(str(broken_note))


def test_non_finite_note_values_do_not_break_saving():
    """NaN 신뢰도·비브라토가 섞여도 JSON으로 저장된다 (값은 비움)."""
    from amp.core.model import Vibrato
    from amp.storage.serialization import note_from_dict, note_to_dict

    d = note_to_dict(Note(id="n", start=0.0, end=1.0, pitch=60, confidence=float("nan"),
                          vibrato=Vibrato(float("nan"), 30.0)))
    text = json.dumps(d, allow_nan=False)
    back = note_from_dict(json.loads(text))
    assert back.confidence is None and back.vibrato is None and back.pitch == 60
