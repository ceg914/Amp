# 구조 설계

## 디렉터리와 모듈 책임

```
amp/
├── __main__.py              python -m amp 진입점
├── core/                    데이터 모델과 편집 규칙 (Qt 없음)
│   ├── model.py             Note, PitchCurve, Syllable, LyricsTrack, AnalysisResult, Score, Project
│   ├── tempo.py             TempoMap: 초 ↔ 박 변환, 스냅, 격자
│   ├── editing.py           이동·길이 변경·분할·병합 순수 함수
│   ├── pitch.py             Hz ↔ MIDI, 음 이름
│   ├── errors.py            사용자에게 보여 줄 메시지를 담는 예외 계층
│   └── ids.py
├── audio/                   오디오 처리 (Qt 없음)
│   ├── data.py              AudioData: (프레임, 채널) float32 배열
│   ├── decoder.py           파일 디코딩 (soundfile → miniaudio 순), 진행률·취소, 파일 지문
│   ├── peaks.py             다중 해상도 파형 min/max 요약
│   └── pcm.py               재생 장치 형식으로 변환 (리샘플·채널·양자화)
├── engine/                  분석 엔진 (Qt 없음)
│   ├── interfaces.py        단계별 추상 클래스
│   ├── registry.py          단계별 백엔드 등록소
│   ├── pipeline.py          단계를 순서대로 실행해 AnalysisResult 생성
│   ├── context.py           CancelToken, StageContext(진행률 환산)
│   └── backends/sample.py   샘플(가짜) 백엔드
├── storage/                 저장 (Qt 없음)
│   ├── serialization.py     모델 ↔ dict 변환
│   └── project_file.py      .ampproj 읽기/쓰기, 원자적 저장
└── app/                     PySide6 UI
    ├── main.py              QApplication, 로깅, 전역 예외 대화상자
    ├── main_window.py       메뉴·툴바·작업 흐름(열기·저장·분석·복구)
    ├── document.py          ProjectDocument: 편집 API, QUndoStack, 변경 시그널
    ├── workers.py           TaskRunner: QThreadPool 작업 실행
    ├── player.py            AudioPlayer: QAudioSink 재생
    ├── autosave.py          자동 저장, 비정상 종료 감지
    ├── timeline.py          파형·눈금자·피아노 롤이 공유하는 가로축 상태
    ├── theme.py             색상, 어두운 테마
    ├── pianoroll/
    │   ├── mapping.py       시간·음높이 ↔ 장면 좌표
    │   ├── items.py         NoteItem, PitchCurveItem, PlayheadItem
    │   ├── scene.py         문서 시그널 → 항목 생성·갱신, 격자, 가사 글자
    │   └── view.py          마우스·키보드 편집, 확대/스크롤
    └── widgets/
        ├── waveform.py      파형 (픽스맵 캐시)
        ├── ruler.py         마디·초 눈금자
        └── keyboard.py      왼쪽 건반
tests/                       단위 테스트(코어) + pytest-qt 테스트(tests/ui)
scripts/make_test_audio.py   시험용 오디오 생성
```

### 의존 방향

```
core  ←  audio  ←  engine
  ↑        ↑         ↑
storage    │         │
  ↑        │         │
  └──────  app  ─────┘
```

`app`만 Qt를 씁니다. 나머지 패키지는 Qt 없이 import되고 테스트됩니다.
분석 엔진을 다른 프로세스나 명령줄 도구로 옮길 때 UI 코드를 끌고 가지 않으려는 규칙입니다.

## 데이터 모델

### 원본과 작업본

```
Project
├── audio: AudioRef            경로, 파일 지문, 길이
├── analysis: AnalysisResult   분석 엔진이 만든 원본. 만든 뒤 바꾸지 않음
│   ├── notes, pitch_curve, tempo, lyrics
│   └── models: {단계: ModelInfo(name, version, placeholder)}
└── score: Score               사용자가 편집하는 작업본
    ├── notes (id → Note), tempo, lyrics
```

분석을 실행하면 `Score.from_analysis()`가 원본을 복사해 작업본을 만듭니다. 이후 편집은 작업본에만 반영됩니다.
원본을 남겨 두기 때문에 "분석 원본으로 되돌리기"와 음표별 수정 여부 표시가 가능합니다.

`Note.origin_ids`는 작업본 음표가 어느 원본 음표에서 왔는지 기록합니다.

| 편집 | origin_ids |
|---|---|
| 분석 직후 | `(원본 id,)` |
| 이동·길이 변경 | 그대로 |
| 분할 | 두 조각 모두 부모 값을 물려받음 |
| 병합 | 합친 음표들의 값을 시간 순으로 합침 |
| 새로 그림 | `()` |

`Project.note_status()`는 이 값과 원본을 비교해 `original`, `modified`, `user` 중 하나를 돌려줍니다. 피아노 롤 색이 이 값으로 정해집니다.

### 가사와 멜리스마

음절과 음표의 연결은 음표 쪽 `Note.syllable_id`에 둡니다. 같은 음절 id를 가진 음표가 둘 이상이면 멜리스마입니다.
연결을 음표에 둔 이유는 분할·병합 같은 음표 편집이 음표 목록만 바꾸면 끝나게 하려는 것입니다.
분할하면 두 조각이 같은 음절을 가리켜 자연스럽게 멜리스마가 되고, 병합하면 시간상 첫 음절이 남습니다.

`Syllable`에는 글자, 발음 기호(`phonetic`), 가사 줄 번호, 정렬된 시작·끝 시간이 있습니다.
자음·모음 단위 시간은 범위에서 제외했으므로 모델에도 없습니다.

### 피치 곡선과 템포

`PitchCurve`는 일정 간격(`frame_rate`)으로 샘플링한 f0(Hz)와 신뢰도 배열입니다. 무성 구간은 NaN입니다.
간격이 일정하면 화면에 보이는 시간 범위의 프레임 인덱스를 바로 계산할 수 있어 긴 곡에서도 빠르게 그릴 수 있습니다.

음표 시간은 오디오와 맞추기 위해 초 단위로 저장합니다. `TempoMap`은 격자 표시, 스냅, 이후 MIDI 내보내기에서 초와 박을 변환할 때만 씁니다.
구간별 템포(`TempoPoint` 여러 개)를 지원하지만, 현재 UI에서는 첫 구간 BPM만 바꿀 수 있습니다.

## 편집 흐름

```
PianoRollView (마우스 드래그)
   │  드래그 중: core.editing으로 미리보기 위치만 계산해 항목에 표시
   │  놓을 때: doc.move_notes(ids, dt, dpitch)
   ▼
ProjectDocument
   │  core.editing.move_notes()로 새 Note 계산 (검증 실패 시 EditError)
   │  undo_stack.push(ReplaceNotesCommand(지울 음표, 넣을 음표))
   ▼
ReplaceNotesCommand.redo()
   │  Score.replace_notes()  ← 모든 음표 편집이 지나는 단일 경로
   │  notesReplaced, selectionRequested 시그널
   ▼
PianoRollScene  항목 추가·삭제·갱신, 가사 글자 다시 계산
```

이동, 길이 변경, 분할, 병합, 추가, 삭제는 모두 "음표 몇 개를 지우고 몇 개를 넣는다"로 표현됩니다.
그래서 실행 취소 명령 클래스가 하나(`ReplaceNotesCommand`)뿐입니다. 연속된 BPM 변경은 `SetTempoCommand.mergeWith`로 하나로 합칩니다.

## 분석 파이프라인과 모델 교체

`AnalysisPipeline.run()`은 다음 순서로 단계를 실행합니다.

1. `SourceSeparator.separate(mix)` → 보컬, 반주
2. `TempoEstimator.estimate(mix)` → TempoMap
3. `PitchTracker.track(vocals)` → PitchCurve
4. `NoteTranscriber.transcribe(pitch, vocals, tempo)` → 음표 목록 (비브라토·멜리스마 처리 담당)
5. `LyricsRecognizer.recognize(vocals)` → 음절 목록 (선택)
6. `LyricsAligner.align(lyrics, notes, ...)` → 음절 시간, 음표의 syllable_id (선택)

단계마다 `StageContext`를 받습니다. 단계는 0~1 진행률만 보고하고, 전체 진행률 환산과 취소 확인은 파이프라인이 맡습니다.
백엔드에서 나온 예외는 `AnalysisError`로 감싸서 UI에 전달합니다.

새 모델을 붙이는 방법:

```python
from amp.core.model import ModelInfo, PitchCurve
from amp.engine.interfaces import PitchTracker
from amp.engine.registry import StageKind


class CrepeTracker(PitchTracker):
    @property
    def info(self):
        return ModelInfo(name="crepe", version="0.0.16", params={"step_ms": 10})

    def track(self, vocals, ctx):
        ...  # ctx.progress(0.5, "피치 추적 중") 로 진행률 보고, 내부에서 취소 확인
        return PitchCurve(start_time=0.0, frame_rate=100.0, f0_hz=f0, confidence=conf)


registry.register(StageKind.PITCH, "crepe", CrepeTracker, "CREPE 피치 추적")
pipeline = AnalysisPipeline(registry, PipelineConfig(pitch="crepe"))
```

사용한 모델 이름과 버전은 `AnalysisResult.models`에 기록되어 프로젝트 파일에 함께 저장됩니다.
`placeholder=True`인 모델이 하나라도 있으면 상태 표시줄에 "샘플 데이터"라고 나옵니다.

현재 등록된 백엔드는 `passthrough`(분리 안 함)와 `sample`뿐입니다. 샘플 백엔드는 오디오 길이와 고정 시드로 멜로디 계획을 만들고,
각 단계가 같은 계획을 공유해 서로 맞는 결과를 냅니다. 소리 내용은 보지 않습니다.

## 스레드와 응답성

- **디코딩·파형 요약·분석**은 `TaskRunner`가 QThreadPool에서 실행합니다. 작업 함수는 `fn(progress, token)` 형태로 Qt를 모릅니다.
  결과·실패·취소는 시그널로 UI 스레드에 전달됩니다. 진행률 시그널은 50ms 간격으로 줄여 보냅니다.
- 작업은 `submit()` 다음 이벤트 루프에서 시작합니다. 즉시 실패하는 작업이라도 `submit()` 직후 연결한 시그널을 놓치지 않습니다.
- 오디오를 새로 열면 이전 불러오기 결과는 순번 비교로 버립니다. 분석 도중 오디오가 바뀌면 분석 결과도 버립니다.
- 분석 입력·출력은 피클 가능한 데이터(numpy 배열, frozen dataclass)입니다.
  지금은 스레드에서 돌리지만, PyTorch 모델처럼 무겁거나 메모리를 많이 쓰는 백엔드가 들어오면 같은 `run()`을 별도 프로세스에서 호출하도록 실행기만 바꿀 계획입니다.

화면 쪽에서 긴 오디오를 다루는 방법:

- 파형은 불러올 때 32·128·512… 샘플 단위 min/max 요약을 만들어 두고, 그릴 때 화면 폭만큼만 다시 줄입니다. 결과는 픽스맵으로 캐시하고 재생선만 덧그립니다.
- 피치 곡선 항목은 노출된 영역(`exposedRect`)의 프레임만 골라 그립니다.
- 격자는 `drawBackground`에서 보이는 범위만 그리고, 선 간격이 7px보다 좁으면 생략합니다.

## 재생

`AudioPlayer`는 QMediaPlayer 대신 QAudioSink를 씁니다. 메모리에 있는 numpy 배열을 직접 흘려보내야
나중에 분리된 보컬·반주를 섞거나 음표 안내음을 합성할 수 있기 때문입니다.

- `PcmConverter`(Qt 없음)가 출력 프레임 번호 기준으로 필요한 구간만 리샘플·채널 변환·양자화합니다. 상태가 없어서 탐색이 단순합니다.
- `PcmDevice`(QIODevice)가 변환기를 감쌉니다. 오디오 스레드에서 읽힐 수 있어 위치를 잠금으로 보호합니다.
- 재생 위치는 `시작 위치 + sink.processedUSecs()`로 계산합니다. 장치가 실제로 처리한 양이라 소리와 재생선이 맞습니다.
- PySide6 6.12에서 `QAudioSink.stateChanged` 시그널 인자 변환이 실패해서, 30ms 타이머에서 `sink.state()`를 직접 확인합니다.
- 출력 장치가 없으면 오류 시그널을 보내고 정지 상태로 남습니다. 편집은 계속할 수 있습니다.

## 프로젝트 파일과 복구

`.ampproj`는 ZIP 파일입니다.

| 항목 | 내용 |
|---|---|
| `manifest.json` | 형식 이름, 형식 버전, 앱 버전, 저장 시각 |
| `project.json` | 오디오 경로, 프로젝트 기준 상대 경로, 파일 지문 |
| `analysis/result.json` | 원본 분석 결과 (분석한 경우) |
| `analysis/pitch.npz` | 원본 피치 곡선 배열 |
| `score.json` | 사용자 작업본 |

- 같은 폴더의 임시 파일에 쓴 뒤 `os.replace`로 바꿔치기합니다. 저장 중 실패해도 기존 파일은 그대로입니다.
- 더 높은 형식 버전 파일은 열지 않고 업데이트를 안내합니다. 필드를 추가할 때는 읽기 쪽에 기본값을 둬서 이전 파일을 계속 엽니다.
- 오디오 파일이 원래 경로에 없으면 프로젝트 기준 상대 경로를 시도하고, 그래도 없으면 위치를 묻습니다. 지문이 다르면 경고합니다.

자동 복구는 세션마다 `<세션id>.lock`을 `QLockFile`로 잡고, 변경이 있으면 1분마다 `<세션id>.ampproj`를 씁니다.
정상 종료하면 지웁니다. 시작할 때 잠금을 얻을 수 있는(주인 프로세스가 사라진) 복구 파일이 있으면 비정상 종료로 보고 복구를 제안합니다.
`setStaleLockTime(0)`으로 시간 기준 만료를 꺼서, 오래 켜 둔 다른 창의 파일을 잘못 가져오지 않습니다.

## 오류 처리

- `AmpError` 하위 예외는 사용자에게 그대로 보여 줄 한국어 메시지를 담습니다.
- 작업 스레드의 예외는 `TaskHandle.failed(메시지, traceback)`으로 전달되어 상세 보기가 있는 대화상자로 표시됩니다.
- 편집 검증 실패(`EditError`)는 상태 표시줄에 짧게 띄웁니다.
- 그 밖에 UI 슬롯에서 난 예외는 `sys.excepthook`이 잡아 로그 파일에 남기고 대화상자로 알립니다.

## 테스트

| 파일 | 대상 |
|---|---|
| `tests/test_tempo.py`, `test_model.py`, `test_editing.py` | 템포 변환, 모델 검증, 편집 규칙 |
| `tests/test_audio.py` | WAV·MP3 디코딩, 손상 파일, 파형 요약 정확성, PCM 변환 |
| `tests/test_engine.py` | 파이프라인 실행, 취소, 백엔드 오류 감싸기, 등록소 검사 |
| `tests/test_storage.py` | 저장/열기 왕복, 원본·작업본 분리, 저장 실패 시 기존 파일 보존, 손상 파일 |
| `tests/ui/test_document.py` | 실행 취소/다시 실행, 변경 상태, 되돌리기 |
| `tests/ui/test_pianoroll.py` | 실제 마우스·키 입력으로 이동·길이 변경·추가·선택·확대 |
| `tests/ui/test_workers.py` | 작업 완료·실패·취소, 즉시 끝나는 작업 |
| `tests/ui/test_player.py` | PCM 장치, 장치 없을 때 오류, 실제 장치가 있으면 재생 중 탐색·끝 감지 |
| `tests/ui/test_autosave.py`, `test_main_window.py` | 자동 저장·복구, 불러오기·분석·저장·열기 통합 흐름 |
