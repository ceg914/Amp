# SONATLAS 구조

## 디렉터리와 모듈 책임

```
amp/
├── __main__.py                 python -m amp 진입점 (Python·패키지 검사 후 앱 실행)
├── core/                       데이터 모델·편집 규칙·파일 입출력 기초 (Qt 없음)
│   ├── model.py                Note, PitchCurve, Syllable, AnalysisResult(원본), Score(작업본), Project
│   ├── tempo.py                TempoMap: 초 ↔ 박, 스냅, 격자
│   ├── editing.py              이동·길이·분할·병합 순수 함수
│   ├── fileio.py               원자적 쓰기(+.bak), 긴 경로, 디스크 부족·권한 오류 메시지
│   └── errors.py               사용자 메시지를 담는 예외 계층
├── audio/                      디코딩, 파형 요약, PCM 변환·믹서 (Qt 없음)
├── engine/                     분석 엔진 (Qt 없음)
│   ├── worker.py               분석 작업 프로세스 (separate · pitch · tempo · probe)
│   ├── worker_client.py        작업 프로세스 실행·진행률·취소·오류 코드 → 한국어 메시지
│   ├── dsp/separation.py       Demucs htdemucs 실행, 시간 정렬, GPU→CPU 재시도
│   ├── dsp/pitch.py            librosa pYIN (16kHz, 100fps, 30초 청크) + 온셋(32ms 창)·RMS(64ms·20ms)
│   ├── dsp/tempo.py            BPM 추정, 2배·절반 후보, 박 간격 직선 보정
│   ├── pitch_post.py           에너지·틈 게이트, 옥타브·튐 보정, 재발음 지표
│   ├── baselines/              STEP 2 알고리즘 고정본(v1). 변경 전후 비교에만 쓰고 앱은 쓰지 않는다
│   ├── notes.py                피치 곡선 → 음표
│   ├── transcription.py        분리 → 피치 → BPM → 음표 파이프라인, 음표만 재검출
│   ├── stems.py                stems/ 목록(stems.json) 쓰기·검증·불러오기
│   ├── cache.py                cache/ 키·원자적 기록·정리
│   ├── merge.py                재분석 반영 계획 (사용자 편집 보호)
│   ├── settings.py             분석 설정·민감도 프리셋
│   ├── interfaces.py, pipeline.py, registry.py, backends/sample.py   STEP 1 메모리 기반 단계와 샘플 백엔드
├── project/                    프로젝트 폴더 (Qt 없음)
│   ├── layout.py               폴더 구조, Windows 이름 규칙, 중복 이름 처리
│   ├── schema.py               파일 종류·스키마 버전·마이그레이션
│   ├── workspace.py            생성(복사/연결)·열기·저장·재연결·다른 이름으로 저장
│   ├── analysis_io.py          analysis/pitch·notes·tempo.json
│   ├── lock.py                 운영체제 파일 잠금(.sonatlas.lock) + 같은 프로세스 안 등록부
│   ├── runs.py                 분석 실행 기록(analysis/runs/)·실행 비교 보고서(analysis/reports/)
│   ├── export.py               WAV/MP3 내보내기
│   └── legacy.py               .ampproj → 프로젝트 폴더 변환 (+백업)
├── storage/                    STEP 1 단일 파일(.ampproj) 형식 — 이전 파일 읽기와 자동 복구 스냅샷에 사용
├── devtools/                   검증용 도구 (앱 실행에는 쓰지 않음)
│   ├── synth.py                합성 노래·반주·숨소리 (시험·벤치마크·검증 도구 자가 점검)
│   ├── evaluation.py           음표 F1·시작/끝 오차, 프레임 피치 cent 오차·옥타브 오류·무성 오경보
│   ├── benchmark.py            v1↔v2 비교 (기본 사례 8개 + 변형 사례 8개)
│   └── validate.py             실제 음원 검증 (scripts/validate_audio.py)
└── app/                        PySide6 UI
    ├── main_window.py          레이아웃·메뉴·도구막대, 세션 시그널 연결
    ├── session.py              ProjectSession: 프로젝트 수명 주기·잠금·분석·내보내기 (컨트롤러)
    ├── document.py             편집 API, QUndoStack, ApplyAnalysisCommand
    ├── player.py               QAudioSink 재생, 트랙 믹서(솔로·음소거·음량)
    ├── workers.py              TaskRunner (QThreadPool)
    ├── autosave.py             자동 저장·비정상 종료 복구
    ├── dialogs.py              새 프로젝트, 내보내기, 재분석 반영 대화상자
    ├── theme/                  Obsidian Violet 토큰·QSS·글꼴(OFL)·아이콘(Lucide, ISC)
    ├── pianoroll/              QGraphicsView 피아노 롤
    └── widgets/                트랙 패널, 속성·분석 설정 패널, 파형, 눈금자, 건반, 로그
scripts/install_separation.py   GPU에 맞는 PyTorch + demucs 설치
scripts/validate_audio.py       실제 음원 검증 보고서
scripts/compare_runs.py         프로젝트의 분석 실행 기록 나열·비교
run.bat, install_separation.bat Windows 실행·설치
```

의존 방향: `core ← audio ← engine ← project ← app`. `app`만 Qt를 씁니다. 분석 엔진과 저장 계층은 Qt 없이 테스트됩니다.

## 프로세스·스레드 모델

| 작업 | 실행 위치 | 이유 |
|---|---|---|
| 보컬 분리 (torch) | 작업 프로세스 | Windows에서 PySide6와 torch를 한 프로세스에 올리면 DLL 충돌 사례가 있음. 강제 종료로 즉시 취소, 끝나면 메모리 반환 |
| 피치 추출 (pYIN) | 작업 프로세스 | 측정 결과 30초 처리 중 GIL을 잡아 다른 스레드가 최대 0.78초 멈춤 |
| BPM 추정 | 작업 프로세스 | 같은 이유 (numba) |
| 디코딩·파형·음표 검출·내보내기 | QThreadPool 스레드 | numpy 위주라 UI를 막지 않음 (10분 MP3 불러오기 중 최대 25ms) |
| 재생 | Qt 오디오 스레드 | PcmMixer가 블록마다 트랙을 섞음 |

작업 프로세스는 `python -m amp.engine.worker <명령> --args-json …`으로 띄우고, 표준 출력으로 JSON 줄(progress/result/error)만 보냅니다.
클라이언트는 0.1초마다 취소 토큰을 확인하고, 취소되면 프로세스를 끝내고 임시 결과 폴더를 지웁니다.
오류 코드(missing_dependency, model_download, oom, decode, internal)는 설치 안내 등 한국어 메시지로 바뀝니다.

## 분석 파이프라인

1. **분리**: `stems/stems.json`이 있고 원본 해시·파일 크기·프레임 수가 맞으면 재사용합니다. 아니면 작업 프로세스가 `stems/.tmp-*`에
   쓰고, 기존 목록을 먼저 지운 뒤 파일을 옮기고 새 목록을 씁니다. 중간에 실패하면 목록이 없으므로 '분리 전'으로 보입니다.
2. **시간 정렬**: 작업 프로세스도 GUI와 같은 디코더로 원본을 읽습니다. 44.1kHz로 바꿀 때와 되돌릴 때 `resample_poly`(지연 없는 필터)를
   쓰고, 원본 프레임 수에 맞춰 자르거나 채웁니다. stems.json에 `offset_frames: 0`을 기록합니다.
   시험: 48kHz 입력의 길이 일치·지연 0 샘플, 실제 htdemucs에서 원래 목소리와 상관 0.998·지연 0 샘플.
3. **피치**: 보컬(또는 분리 안 함이면 원본)을 16kHz·100fps로 pYIN 분석하고 `cache/`에 원시 결과를 둡니다.
   캐시 키 = 원본 해시 + 입력(분리 결과 id) + 파라미터·버전.
4. **후처리**: 에너지 게이트(64ms RMS, 곡 최대 −40dB) + 틈 게이트(20ms RMS가 ±250ms 최대보다 30dB 낮음) → 40ms 미만 유성 조각 제거
   → 옥타브(±12) 보정 → 튐 보정 → 3프레임 중앙값. 무성은 NaN으로 남깁니다. 피치 캐시 버전 2(STEP 2.5)부터 20ms RMS를 함께 저장합니다.
5. **BPM**: 반주(또는 원본)에서 librosa로 추정하고, 박 위치에 직선을 맞춰 템포그램 양자화(512 hop에서 117.45/123.0)를 보정합니다.
6. **음표**: 아래 절. 캐시하지 않습니다. `redetect_notes`는 1~5단계 결과를 그대로 쓰고 음표만 다시 만듭니다.
   피치 버전 1(0.2.0)로 만든 결과를 다시 검출하면 전체 재분석을 권하는 경고를 붙입니다.
8. **실행 기록**: 성공한 전체 분석·음표 재검출마다 `analysis/runs/<run_id>.json`(모델·버전·설정·음표·단계별 시간·작업 프로세스 최대 메모리·
   그 시점 사용자 편집 개수)을 남깁니다. 실패·취소한 실행은 남기지 않습니다. 저장 순번(`seq`)으로 순서를 정합니다.
7. **보컬 존재 판정**: 보컬/전체 에너지 비가 -25dB 아래이거나 보컬이 들리는 구간이 3% 미만이면 경고합니다.

## 음표 검출 (`engine/notes.py`)

- 유성 판정은 pYIN의 HMM 결과를 따르고, 확률은 아주 낮은 잡음만 거릅니다(합성 음성은 확률 중앙값이 0.02여도 유성).
- 중심선 = 220ms 중앙값 필터. 4.5Hz 이상 비브라토가 상쇄되고, 음 경계는 날카롭게, 포르타멘토는 경사로 남습니다.
- 변화점: 중심선이 현재 음에서 (기준 − 0.3)반음 이상 벗어나 hold 동안 안정(폭 0.45반음 미만, 흐름 0.3반음 미만)하면 나눕니다.
  경계는 두 음 중간을 지나는 지점이고, 새 음 높이는 도착 지점부터 잽니다(글라이드 중간에서 두 번 나뉘지 않게).
- 재발음 지표 = 온셋 × 음량 하락 비율. 하락 깊이 = min(직전 150ms 최대, 직후 150ms 최대) − 직전 70ms 최저(20ms RMS, dB).
  3dB 미만이면 0, 12dB 이상이면 온셋 그대로. 온셋은 멜 대역 평균 dB 증가량이라 곡 음량과 무관하므로 절대 임계(`split_strength`:
  표준 4.0, 정밀 3.0, 단순 없음)로 판정합니다. 비브라토·트레몰로(±5dB)만 있는 지속음은 이 지표가 임계에 못 미칩니다(합성 확인).
- 재발음 피크가 음높이 경계 ±60ms 안에 있으면 경계를 그리로 옮기고, 같은 높이 구간 안에서는 나눕니다(양 끝 최소 길이 안은 제외).
  짧은 무성 틈 끝에 재발음이 있으면 틈을 잇지 않고, 그 프레이즈 첫 음은 앞 음과 같은 높이여도 합치지 않습니다.
- 미끄러짐 흡수: 두 음 사이에 낀 짧은 조각(하나 250ms 이하, 묶음 400ms 이하)이 높이가 두 음 사이이고 안에서 0.4반음 이상 같은 방향으로
  움직이면 앞뒤 음에 나눠 붙입니다. 첫 조각이 재발음으로 시작하면 따로 부른 음으로 보고 건드리지 않습니다.
- 짧은 조각은 가까운 높이의 이웃에 합칩니다. 표준·정밀에서는 합칠 쪽 경계가 뚜렷한 재발음이고 높이가 이웃과 기준 이상 다르며
  두 이웃 사이 높이가 아닌 조각을 장식음으로 남깁니다. 중심선 창보다 짧은 이탈은 따로 떼어 최소 길이 미만·고신뢰면 장식음,
  이상이면 일반 음표로 만듭니다. 이웃 음 쪽으로 가는 가장자리 조각은 떼어 내지 않습니다.
- 실제 유성 프레임이 60% 미만인 구간(틈 잇기로 메운 곳)은 음표로 만들지 않습니다. 음표별 비브라토(3.5~9Hz, 20cent 이상)를 추정합니다.
- 보컬 없는 구간(`AnalysisSettings.no_vocal_ranges`, 사용자가 지정)의 프레임은 무성으로 보고, 틈 잇기와 같은 음 병합도 이 구간을
  넘지 않습니다. 구간은 정렬·병합해 project.json의 분석 설정에 저장하고, 프로그램 기본 설정(QSettings)에는 넣지 않습니다.

## 데이터와 저장

- `AnalysisResult`는 분석이 만든 원본이고 바꾸지 않습니다. `Score`는 작업본입니다. 작업본 음표의 `origin_ids`로 원본·수정·추가를 가립니다.
- 프로젝트 폴더에서는 `notes.json`의 `analysis`(원본)와 `edits`(작업본), `tempo.json`의 `analysis`와 `user`를 나눠 저장합니다.
  작업본만 바뀌는 저장에서도 원본 부분은 메모리의 불변 객체를 그대로 다시 쓰므로 손상되지 않고, 원자적 교체와 .bak으로 보호합니다.
  `pitch.json`은 분석 결과가 바뀐 경우에만 다시 씁니다.
- 각 JSON은 `schema`·`schema_version`을 가지고, `project/schema.py`의 `MIGRATIONS`로 버전을 올립니다. 더 새로운 버전은 열지 않습니다.
- `project.json`: ID·이름·생성/수정 시각·스키마 버전·원본(방식, 상대/절대 경로, SHA-256, 크기, 길이)·분리 상태·분석 상태·
  편집 상태(원본/수정/추가 개수)·분석 설정·재생 위치·트랙 상태·작업 공간(확대·스크롤·레이어·스냅·패널 크기).
- 자동 복구 스냅샷은 STEP 1 형식(zip)으로 앱 데이터 폴더에 쓰고, 복구할 때 프로젝트 폴더를 연 뒤 내용을 덮어 변경 상태로 둡니다.

## 재분석과 사용자 편집 보호

`engine/merge.plan_reanalysis`가 반영 계획을 만들고 `ApplyAnalysisCommand`가 원본 분석·음표·템포를 함께 바꿉니다(실행 취소 가능).
편집이 있으면 대화상자에서 개수를 비교해 고릅니다.

- **수정 유지하며 적용**: 수정·추가 음표는 그대로 두고, 손대지 않은 원본 음표만 교체. 보존 음표와 20% 넘게 겹치는 새 음표는 넣지 않음.
  사용자가 바꾼 템포도 유지.
- **전부 새 분석으로 교체** / **취소**.

## 재생과 시간축

`TimelineState` 하나를 눈금자·파형·피아노 롤이 공유하고, 재생선은 `시작 위치 + sink.processedUSecs()`로 계산합니다.
트랙(원본·보컬·반주)은 같은 길이·샘플레이트여야 플레이어에 올라가므로, 트랙을 바꾸거나 섞어도 위치가 그대로입니다.

## 테스트

| 파일 | 대상 |
|---|---|
| `test_pitch_tracking.py` | 요구 1: 220Hz 사인·배음 톤(±10cent, 시작 시각 ±40ms), 무음 구간 NaN, 옥타브·튐 보정, 재발음 지표 |
| `test_note_detection.py` | 요구 2·3·4: 연속 음 분리·경계, 포르타멘토, 비브라토 1음, 최소 길이, 장식음 신뢰도, 무음·잡음·호흡, 음역 |
| `test_tempo_estimation.py` | 96·120BPM 클릭, 2배·절반 후보, 첫 박, 짧은/무음 기본값 |
| `test_transcription.py` | 단계 재사용·캐시 끄기·강제 재분리·원본 변경 무효화·취소 정리·음표만 재검출, 실제 작업 프로세스, 비브라토 곡 × 민감도 3단계, 실제 htdemucs(`AMP_TEST_DEMUCS=1`) |
| `test_worker_and_stems.py` | 작업 프로세스 길이·지연 0, 취소, 오류 코드, 무음 입력, stems.json 검증(해시·크기·손상) |
| `test_project_folder.py` | SONATLAS 요구 1~12: 생성(복사·연결), 중복 이름, stems 저장, WAV/MP3·일괄 내보내기, 저장·재열기, 폴더 이동, 재연결, 다른 이름으로 저장, 캐시 삭제 후 보존, 저장 실패 시 기존 파일 보존, 손상 파일 .bak 복구, 한글·특수문자 경로, 마이그레이션, .ampproj 변환 |
| `test_cache_settings_merge.py` | 캐시 원자성·한글 경로, 설정 프리셋, 재분석 병합 규칙(요구 7) |
| `test_accuracy.py` | STEP 2.5: 합성 사례 전 과정 허용 오차(F1·시작/끝 오차·cent·옥타브·무성 오경보), v1↔v2 비교(기본·변형 사례), 온셋 정렬, 틈 게이트 |
| `test_project_lock.py` | STEP 2.5: 같은 프로세스 이중 열기, 다른 프로세스 차단·해제, 강제 종료 뒤 잠금 회수, 열린 채 복사한 폴더 독립 열기, 다른 컴퓨터의 잠금 파일, 이동, 해제, 임시 파일 정리 |
| `test_stability.py` | STEP 2.5: 캐시 손상 감지·재계산, 잘못된 새 결과 미확정, 강제 종료된 작업, 디스크 부족·권한 분류, 잘린·깨진 MP3, GPU 메모리 부족 → CPU, CPU 메모리 부족, 모델 다운로드 실패, torch·demucs 없음, 실패한 분석이 프로젝트 파일을 바꾸지 않음 |
| `test_runs.py`, `test_validate_tool.py` | STEP 2.5: 실행 기록·비교 보고서·CLI, 검증 도구(음높이 표기, 정렬 지연 측정, 정답 CSV, 끝까지 실행) |
| `ui/test_main_window.py` | 프로젝트 생성·샘플 분석, 손상 파일, 저장·재열기(재생 위치·트랙 상태), 이중 열기 잠금, 분할·병합·BPM·첫 박, 비정상 종료 복구, 분석→트랙→솔로·음소거→내보내기→편집 보호, .ampproj 변환, 시간축 동기화(요구 5), 열린 프로젝트 복사본을 두 번째 창에서 열기, 실행 기록·비교 보고서, 실패한 분석 미기록, 저장 충돌 확인 |
| 나머지 | STEP 1 회귀: 모델, 편집, 템포, 오디오·PCM·믹서, 저장 형식, 문서·실행 취소, 피아노 롤 마우스 조작, 작업 실행기, 자동 저장, 재생기, 진입점, 설치 스크립트 |
