"""채보 파이프라인: 재사용·강제 재분리·취소 정리·음표만 재검출, 실제 작업 프로세스, (선택) 실제 htdemucs."""

import io
import os
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from amp.audio.decoder import load_audio
from amp.core.errors import AnalysisCancelled
from amp.engine import worker
from amp.engine.cache import file_content_key
from amp.engine.context import CancelToken
from amp.engine.settings import AnalysisSettings
from amp.engine.transcription import TranscriptionPipeline, TranscriptionRequest, redetect_notes
from tests.audio_synth import SR, backing, sung

MELODY = [(1.0, 1.8, 64), (1.9, 2.7, 67), (2.8, 4.0, 69), (4.1, 5.0, 62)]


class InProcessRunner:
    """작업 프로세스 명령을 같은 프로세스에서 실행한다 (빠른 시험용). 분리 모델은 _identity를 쓴다."""

    def __init__(self):
        self.calls = Counter()

    def __call__(self, command, args, progress=None, token=None, numba_cache_dir=None):
        self.calls[command] += 1
        if token is not None:
            token.raise_if_cancelled()
        old = worker._PROTO
        worker._PROTO = io.StringIO()
        try:
            return worker.COMMANDS[command](args)
        finally:
            worker._PROTO = old


@pytest.fixture
def song(tmp_path):
    y = sung(MELODY, 6.0)
    path = tmp_path / "노래.wav"
    sf.write(path, np.stack([y, y], 1), SR)
    return str(path)


def make_request(path, tmp_path, **kw):
    audio = load_audio(path)
    settings = kw.pop("settings", AnalysisSettings())
    return TranscriptionRequest(audio, path, file_content_key(path), settings,
                                str(tmp_path / "proj" / "stems"), str(tmp_path / "proj" / "cache"), **kw)


def test_pipeline_reuses_stages_and_detects_melody(song, tmp_path):
    runner = InProcessRunner()
    pipe = TranscriptionPipeline(runner, separation_model="_identity")
    req = make_request(song, tmp_path)
    out1 = pipe.run(req)
    assert [n.pitch for n in out1.result.notes] == [64, 67, 69, 62]
    assert out1.reused == {"separation": False, "pitch": False, "tempo": False}
    assert set(out1.stems) == {"vocals", "instrumental"}
    assert out1.stems["vocals"].frames == req.audio.frames  # 원본과 같은 시간축
    out2 = pipe.run(req)
    assert out2.reused == {"separation": True, "pitch": True, "tempo": True}
    assert runner.calls == Counter(separate=1, pitch=1, tempo=1)
    # 캐시 사용 끔 → 피치·템포 다시 계산, 분리는 정식 결과(stems/)라 재사용
    pipe.run(replace(req, settings=AnalysisSettings(use_cache=False)))
    assert runner.calls == Counter(separate=1, pitch=2, tempo=2)
    # 명시적 재분리 요청
    pipe.run(replace(req, force_separation=True))
    assert runner.calls["separate"] == 2


def test_original_change_invalidates_stems_and_cache(song, tmp_path):
    runner = InProcessRunner()
    pipe = TranscriptionPipeline(runner, separation_model="_identity")
    pipe.run(make_request(song, tmp_path))
    y = sung(MELODY, 6.0) * 0.9  # 원본 내용이 바뀜
    sf.write(song, np.stack([y, y], 1), SR)
    out = pipe.run(make_request(song, tmp_path))
    assert out.reused == {"separation": False, "pitch": False, "tempo": False}


def test_cancel_cleans_temporary_files(song, tmp_path):
    token = CancelToken()

    def cancelling_runner(command, args, progress=None, token_=None, numba=None):
        if command == "separate":
            with open(os.path.join(args["out_dir"], "vocals.wav"), "wb") as f:
                f.write(b"partial")
            token.cancel()
            raise AnalysisCancelled("취소")
        raise AssertionError("분리 뒤 단계는 실행되면 안 된다")

    with pytest.raises(AnalysisCancelled):
        TranscriptionPipeline(cancelling_runner, separation_model="_identity").run(make_request(song, tmp_path), token=token)
    stems_dir = tmp_path / "proj" / "stems"
    assert list(stems_dir.iterdir()) == []  # 반쯤 쓴 결과·임시 폴더가 남지 않는다


def test_separation_none_and_redetect(song, tmp_path):
    runner = InProcessRunner()
    req = make_request(song, tmp_path, settings=AnalysisSettings(separation="none"))
    out = TranscriptionPipeline(runner, separation_model="_identity").run(req)
    assert out.stems == {} and runner.calls["separate"] == 0
    assert out.result.models["separation"].name == "none"
    simple = redetect_notes(out.result, AnalysisSettings.for_sensitivity("simple"))
    assert simple.pitch_curve is out.result.pitch_curve  # 피치 결과는 재사용
    assert simple.settings["sensitivity"] == "simple"
    assert [n.pitch for n in simple.notes] == [64, 67, 69, 62]


def test_real_worker_processes(tmp_path):
    """실제로 작업 프로세스(pitch·tempo)를 띄운다. 반주 섞인 곡을 '분리 안 함'으로 분석."""
    y = sung(MELODY, 6.0) + 0.0 * backing(6.0)
    path = tmp_path / "곡 (실제 프로세스).wav"
    sf.write(path, y, SR)
    out = TranscriptionPipeline().run(make_request(str(path), tmp_path, settings=AnalysisSettings(separation="none")))
    assert [n.pitch for n in out.result.notes] == [64, 67, 69, 62]
    for n, (a, b, _m) in zip(out.result.notes, MELODY):
        assert abs(n.start - a) < 0.06 and abs(n.end - b) < 0.06


@pytest.mark.skipif(os.environ.get("AMP_TEST_DEMUCS") != "1", reason="실제 htdemucs 시험은 AMP_TEST_DEMUCS=1일 때만 (모델 다운로드·수십 초)")
def test_real_htdemucs(tmp_path):
    pytest.importorskip("demucs")
    # 떨림 없는 합성음은 htdemucs가 반주(other)로 분류한다. 사람 목소리처럼 비브라토를 넣는다.
    y = 0.8 * sung(MELODY, 6.0, vibrato_cents=25) + backing(6.0)
    y = y / np.abs(y).max() * 0.9
    path = tmp_path / "반주 섞인 노래.mp3"
    sf.write(path, np.stack([y, y], 1).astype(np.float32), SR, format="MP3")
    out = TranscriptionPipeline().run(make_request(str(path), tmp_path))
    assert out.result.models["separation"].name == "htdemucs"
    assert out.stems["vocals"].frames == load_audio(str(path)).frames
    assert [n.pitch for n in out.result.notes] == [64, 67, 69, 62]
    assert 115 < out.result.tempo.initial_bpm < 125 or 57 < out.result.tempo.initial_bpm < 63
