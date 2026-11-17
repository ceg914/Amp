"""프로젝트 폴더 (SONATLAS 요구 테스트 1~12 중 저장 계층)."""

import json
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from amp.audio.data import AudioData
from amp.audio.decoder import load_audio
from amp.core.errors import ProjectFormatError, StorageError
from amp.core.model import AnalysisResult, ModelInfo, PitchCurve, Project, Score
from amp.core.tempo import TempoMap
from amp.engine import stems as stem_store
from amp.engine.worker_client import run_worker
from amp.project import schema
from amp.project.export import ExportOptions, export_audio
from amp.project.layout import sanitize_name, unique_folder
from amp.project.legacy import import_legacy
from amp.project.workspace import COPY, LINK, copy_project, create_project, find_projects_with_source, open_folder
from amp.storage.project_file import save_project
from tests.conftest import note

ODD_NAME = "노래 #1 (라이브) & 'mix' [최종]"


@pytest.fixture
def src_dir(tmp_path):
    d = tmp_path / "원본 폴더 #&"
    d.mkdir()
    return d


@pytest.fixture
def wav_src(src_dir):
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = src_dir / f"{ODD_NAME}.wav"
    sf.write(p, np.stack([x, x], 1), sr)
    return p


@pytest.fixture
def mp3_src(src_dir):
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    p = src_dir / "곡.mp3"
    sf.write(p, np.stack([x, x], 1), sr, format="MP3")
    return p


def analysis(sha="abc"):
    curve = PitchCurve(0.0, 100.0, np.r_[np.full(50, 440.0), np.full(50, np.nan)], np.r_[np.full(50, 0.9), np.zeros(50)])
    return AnalysisResult(
        1.0, TempoMap.constant(117.5, 0.25), (note("a", 0.1, 0.4, 69, confidence=0.9), note("b", 0.5, 0.9, 71)),
        pitch_curve=curve, models={"pitch": ModelInfo("pyin", "librosa 1.0"), "tempo": ModelInfo("librosa-beat")},
        created_at="2026-11-17T00:00:00+00:00", audio_fingerprint=sha, onset_envelope=np.linspace(0, 1, 100),
        tempo_info={"candidates": [{"bpm": 117.5, "score": 0.7}, {"bpm": 58.75, "score": 0.8}], "beats": [0.25, 0.76],
                    "reliable": True, "algorithm": "librosa"},
        warnings=("경고 예시",), settings={"sensitivity": "standard"},
    )


# 1, 2 --------------------------------------------------------------------------

def test_create_copy_project_keeps_original_untouched(wav_src, tmp_path):
    before = wav_src.read_bytes()
    mtime = wav_src.stat().st_mtime
    folder = create_project(wav_src, tmp_path / "SONATLAS Projects", None, COPY,
                            {"duration": 2.0, "sample_rate": 44100, "channels": 2})
    root = folder.root
    assert root.name == sanitize_name(ODD_NAME)
    for sub in ("stems", "analysis", "cache", "exports"):
        assert (root / sub).is_dir()
    assert (root / "original.wav").read_bytes() == before
    assert wav_src.read_bytes() == before and wav_src.stat().st_mtime == mtime  # 원본 그대로
    meta = json.loads((root / "project.json").read_text(encoding="utf-8"))
    assert meta["schema"] == schema.PROJECT and meta["schema_version"] == 1
    assert meta["original"]["mode"] == "copy" and meta["original"]["path"] == "original.wav"
    assert len(meta["original"]["sha256"]) == 64 and meta["id"]
    assert folder.resolve_original().state == "ok"


def test_create_link_project(mp3_src, tmp_path):
    folder = create_project(mp3_src, tmp_path / "P", "링크 곡", LINK)
    assert not any(p.suffix == ".mp3" for p in folder.root.iterdir())  # 사본을 만들지 않는다
    st = folder.resolve_original()
    assert st.state == "ok" and st.path == mp3_src.resolve()
    assert folder.meta.original.relative_path


# 3 ----------------------------------------------------------------------------

def test_duplicate_names_get_new_folders(wav_src, tmp_path):
    parent = tmp_path / "P"
    a = create_project(wav_src, parent, "곡", COPY)
    b = create_project(wav_src, parent, "곡", COPY)
    c = create_project(wav_src, parent, "곡", LINK)
    assert [a.root.name, b.root.name, c.root.name] == ["곡", "곡 (2)", "곡 (3)"]
    assert sorted(p.name for p in find_projects_with_source(parent, a.meta.original.sha256)) == ["곡", "곡 (2)", "곡 (3)"]
    assert a.meta.id != b.meta.id


def test_sanitize_windows_names(tmp_path):
    assert sanitize_name('a<b>c:"d/e\\f|g?h*') == "a_b_c__d_e_f_g_h_"
    assert sanitize_name("CON") == "_CON" and sanitize_name("  ...  ") == "Untitled"
    assert len(sanitize_name("가" * 300)) == 80
    (tmp_path / "x").mkdir()
    assert unique_folder(tmp_path, "x").name == "x (2)"


# 4 ----------------------------------------------------------------------------

def test_stems_saved_in_project(wav_src, tmp_path):
    folder = create_project(wav_src, tmp_path / "P", "s", COPY)
    tmp = folder.paths.stems / ".tmp-x"
    tmp.mkdir()
    run_worker("separate", {"input": str(folder.root / "original.wav"), "out_dir": str(tmp), "model": "_identity"})
    stem_store.commit(tmp, folder.paths.stems, folder.meta.original.sha256, {"device": "cpu"})
    shutil.rmtree(tmp)
    m, ok, reason = folder.stems_status()
    assert ok, reason
    assert sorted(p.name for p in folder.paths.stems.iterdir()) == ["instrumental.wav", "stems.json", "vocals.wav"]
    info = sf.info(str(folder.paths.stems / "vocals.wav"))
    assert info.frames == m.frames == load_audio(str(wav_src)).frames and m.offset_frames == 0


# 5, 6 -------------------------------------------------------------------------

@pytest.mark.parametrize("opts", [ExportOptions("wav", "PCM_16"), ExportOptions("wav", "FLOAT"), ExportOptions("mp3", mp3_bitrate=192)])
def test_export_formats(tmp_path, opts):
    sr = 44100
    x = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype(np.float32)
    audio = AudioData(np.stack([x, x], 1), sr)
    dst = tmp_path / "exports" / f"보컬 내보내기{opts.extension}"
    seen = []
    info = export_audio(audio, dst, opts, lambda f, m: seen.append(f))
    assert dst.is_file() and info["size"] > 0 and seen[-1] == 1.0
    back = load_audio(str(dst))
    assert back.sample_rate == sr and abs(back.frames - audio.frames) <= 1152
    if opts.fmt == "mp3":
        kbps = dst.stat().st_size * 8 / 3 / 1000
        assert 185 < kbps < 200
    else:
        assert sf.info(str(dst)).subtype == opts.wav_subtype
    assert [p.name for p in dst.parent.iterdir()] == [dst.name]  # 임시 파일이 남지 않는다


def test_export_rejects_unsupported_mp3_rate(tmp_path):
    audio = AudioData(np.zeros((22050, 1), np.float32), 22050)
    with pytest.raises(StorageError):
        export_audio(audio, tmp_path / "x.mp3", ExportOptions("mp3"))


def test_batch_export_does_not_rerun_separation(tmp_path):
    sr = 44100
    stems = {n: AudioData(np.random.default_rng(i).uniform(-0.2, 0.2, (sr, 2)).astype(np.float32), sr)
             for i, n in enumerate(("vocals", "instrumental"))}
    for name, a in stems.items():
        for fmt in ("wav", "mp3"):
            export_audio(a, tmp_path / f"{name}.{fmt}", ExportOptions(fmt))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["instrumental.mp3", "instrumental.wav", "vocals.mp3", "vocals.wav"]


# 7, 8 -------------------------------------------------------------------------

def _filled_project(folder):
    a = analysis(folder.meta.original.sha256)
    p = Project(audio=folder.audio_ref(), analysis=a, score=Score.from_analysis(a))
    p.score.replace_notes(["b"], [p.score.get("b").with_(pitch=72)])
    p.score.replace_notes([], [note("u", 0.95, 1.0, 60)])
    p.score.tempo = TempoMap.constant(120.0, 0.25)
    return p


def test_save_and_reopen_keeps_analysis_and_edits_separate(wav_src, tmp_path):
    folder = create_project(wav_src, tmp_path / "P", "곡", COPY)
    project = _filled_project(folder)
    folder.save(project, ui={"px_per_sec": 150}, playback={"position": 1.25, "tracks": {"vocals": {"mute": True}}})
    files = {p.name for p in folder.paths.analysis.iterdir()}
    assert {"pitch.json", "notes.json", "tempo.json"} <= files
    notes = json.loads(folder.paths.notes_json.read_text(encoding="utf-8"))
    assert [n["pitch"] for n in notes["analysis"]["notes"]] == [69, 71]  # 원본은 그대로
    assert {n["status"] for n in notes["edits"]["notes"]} == {"original", "modified", "user"}
    tempo = json.loads(folder.paths.tempo_json.read_text(encoding="utf-8"))
    assert tempo["analysis"]["bpm"] == 117.5 and tempo["user"]["bpm"] == 120 and tempo["user"]["edited"]
    pitch = json.loads(folder.paths.pitch_json.read_text(encoding="utf-8"))
    assert pitch["frames"] == 100 and pitch["voiced"][:50] == [1] * 50 and pitch["f0_hz"][60] is None
    assert pitch["midi"][0] == pytest.approx(69.0)

    reopened, report = open_folder(folder.root)
    p2, rep2 = reopened.load_project()
    assert not report.warnings and not rep2.warnings
    assert p2.analysis.notes == project.analysis.notes
    assert p2.score.notes() == project.score.notes()
    assert p2.score.tempo == project.score.tempo and p2.analysis.tempo == project.analysis.tempo
    assert np.allclose(p2.analysis.pitch_curve.f0_hz, project.analysis.pitch_curve.f0_hz, equal_nan=True, atol=0.01)
    assert p2.analysis.tempo_candidates == (117.5, 58.75) and p2.analysis.warnings == ("경고 예시",)
    assert reopened.meta.playback["position"] == 1.25 and reopened.meta.ui["px_per_sec"] == 150
    assert reopened.meta.edit_state["modified"] == 1 and reopened.meta.edit_state["user_added"] == 1


def test_moved_project_reopens(wav_src, mp3_src, tmp_path):
    a = create_project(wav_src, tmp_path / "P", "복사본", COPY)
    a.save(_filled_project(a))
    create_project(mp3_src, tmp_path / "P", "연결본", LINK)
    # 프로젝트 폴더와 원본 폴더를 함께 다른 곳으로 옮긴다
    moved = tmp_path / "이동한 곳"
    moved.mkdir()
    shutil.move(str(tmp_path / "P"), str(moved / "P"))
    shutil.move(str(mp3_src.parent), str(moved / mp3_src.parent.name))
    a2, _ = open_folder(moved / "P" / "복사본")
    assert a2.resolve_original().state == "ok"
    assert a2.load_project()[0].score.get("u").pitch == 60
    b2, _ = open_folder(moved / "P" / "연결본")
    st = b2.resolve_original()
    assert st.state == "ok" and st.path == (moved / mp3_src.parent.name / "곡.mp3").resolve()


def test_missing_and_changed_link_and_relink(mp3_src, tmp_path):
    folder = create_project(mp3_src, tmp_path / "P", "x", LINK)
    data = mp3_src.read_bytes()
    mp3_src.unlink()
    assert folder.resolve_original().state == "missing"
    elsewhere = tmp_path / "새 위치.mp3"
    elsewhere.write_bytes(data)
    assert folder.relink(str(elsewhere)) is True
    assert folder.resolve_original().state == "ok"
    elsewhere.write_bytes(data + b"\0" * 100)  # 내용이 바뀜
    assert folder.resolve_original().state == "changed"


def test_save_as_copies_without_cache(wav_src, tmp_path):
    folder = create_project(wav_src, tmp_path / "P", "곡", COPY)
    folder.save(_filled_project(folder))
    (folder.paths.cache / "junk.bin").write_bytes(b"x" * 100)
    copy = copy_project(folder, tmp_path / "Q", "곡 사본")
    assert copy.meta.id != folder.meta.id and copy.name == "곡 사본"
    assert list(copy.paths.cache.iterdir()) == []
    p, _ = copy.load_project()
    assert p.score.get("u").pitch == 60
    assert folder.root.exists()  # 원래 폴더는 그대로


# 10, 11 ----------------------------------------------------------------------

def test_clear_cache_keeps_user_data(wav_src, tmp_path):
    folder = create_project(wav_src, tmp_path / "P", "곡", COPY)
    project = _filled_project(folder)
    folder.save(project)
    with folder.cache().writer("k", "pitch", {}) as d:
        (d / "pitch.npz").write_bytes(b"x")
    (folder.paths.exports / "mine.wav").write_bytes(b"y")
    folder.clear_cache()
    assert list(folder.paths.cache.iterdir()) == []
    assert (folder.root / "original.wav").exists() and (folder.paths.exports / "mine.wav").exists()
    assert folder.load_project()[0].score.notes() == project.score.notes()


def test_failed_save_keeps_previous_data(wav_src, tmp_path, monkeypatch):
    folder = create_project(wav_src, tmp_path / "P", "곡", COPY)
    project = _filled_project(folder)
    folder.save(project)
    before = folder.paths.notes_json.read_bytes()
    import amp.core.fileio as fio

    real_replace = fio._replace

    def failing(src, dst):
        if dst.endswith("notes.json"):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(fio, "_replace", failing)
    project.score.replace_notes(["u"], [])
    with pytest.raises(StorageError, match="디스크 공간"):
        folder.save(project)
    assert folder.paths.notes_json.read_bytes() == before
    assert not [p for p in folder.paths.analysis.iterdir() if p.name.endswith(".tmp")]


def test_corrupt_files_fall_back_to_backup(wav_src, tmp_path):
    folder = create_project(wav_src, tmp_path / "P", "곡", COPY)
    project = _filled_project(folder)
    folder.save(project)
    folder.save(project)  # 두 번 저장해 .bak 생성
    folder.paths.notes_json.write_text("{깨진 파일", encoding="utf-8")
    p, report = folder.load_project()
    assert p.score.notes() == project.score.notes()
    assert report.from_backup == ["notes.json"] and report.warnings
    folder.paths.project_json.write_text("garbage", encoding="utf-8")
    _, rep = open_folder(folder.root)
    assert rep.warnings
    (folder.root / "project.json.bak").unlink()
    with pytest.raises(ProjectFormatError):
        open_folder(folder.root)


def test_read_only_folder_error(tmp_path, wav_src):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("관리자(root) 권한에서는 읽기 전용 폴더도 쓸 수 있어 시험할 수 없다")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    with pytest.raises(StorageError):
        create_project(wav_src, ro, "x", COPY)


# 마이그레이션 ---------------------------------------------------------------------

def test_schema_migration_and_newer_version(monkeypatch):
    monkeypatch.setitem(schema.CURRENT, schema.TEMPO, 2)
    monkeypatch.setitem(schema.MIGRATIONS, schema.TEMPO, {1: lambda d: {**d, "added_in_v2": True}})
    data, migrated = schema.migrate(schema.TEMPO, {"schema": schema.TEMPO, "schema_version": 1})
    assert migrated and data["added_in_v2"] and data["schema_version"] == 2
    with pytest.raises(ProjectFormatError, match="새로운 버전"):
        schema.migrate(schema.TEMPO, {"schema": schema.TEMPO, "schema_version": 3})
    with pytest.raises(ProjectFormatError):
        schema.migrate(schema.TEMPO, {"schema": "other"})


def test_import_legacy_ampproj(wav_src, tmp_path):
    a = analysis()
    legacy = Project(audio=None, analysis=a, score=Score.from_analysis(a))
    legacy.score.replace_notes(["a"], [legacy.score.get("a").with_(pitch=50)])  # 사용자 편집
    from amp.core.model import AudioRef

    legacy.audio = AudioRef(str(wav_src), "fp", 2.0, 44100, 2)
    old = tmp_path / "예전 프로젝트.ampproj"
    save_project(legacy, str(old))
    before = old.read_bytes()
    folder, notes = import_legacy(old, tmp_path / "P")
    assert old.read_bytes() == before  # 원래 파일은 그대로
    assert any(p.name.endswith("예전 프로젝트.ampproj") for p in folder.paths.backup.iterdir())
    p, _ = folder.load_project()
    assert p.score.get("a").pitch == 50 and p.note_status(p.score.get("a")) == "modified"
    assert p.analysis.notes == a.notes
    assert folder.meta.original.mode == LINK and folder.resolve_original().state == "ok"
