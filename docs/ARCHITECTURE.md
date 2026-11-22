# SONATLAS 구조

## 디렉터리와 모듈 책임

```
amp/
├── __main__.py                 python -m amp 진입점 (Python·패키지 검사 후 앱 실행)
├── core/                       데이터 모델·편집 규칙·파일 입출력 기초 (Qt 없음)
│   ├── model.py                Note, PitchCurve, AnalysisResult(원본), Score(작업본), Project(+lyrics). Syllable·LyricsTrack은 STEP 1 형식 읽기용
│   ├── tempo.py                TempoMap: 초 ↔ 박, 스냅, 격자
│   ├── editing.py              이동·길이·분할·병합 순수 함수
│   ├── fileio.py               원자적 쓰기(+.bak), 긴 경로, 디스크 부족·권한 오류 메시지
│   └── errors.py               사용자 메시지를 담는 예외 계층
├── audio/                      디코딩, 파형 요약, PCM 변환·믹서, Windows 기본 출력 조회 (Qt 없음)
│   ├── pcm.py                  PcmMixer: 트랙별 음량·음소거·솔로. 상태가 바뀌면 12ms 음량 램프(A/B 전환 크로스페이드), 여러 트랙 상태를 한 번에 바꾸기
│   ├── stream.py               STEP 5 PlaybackStream: 장치 프레임 ↔ 원본 시간, 구간 반복(이음매 6ms 페이드), 느린 재생
│   ├── stretch.py              STEP 5 위상 보코더(위상 고정) 시간 늘리기: 음높이 유지, 반복 구간·60초 창만 늘림
│   ├── spectrogram.py          STEP 5 타일(256프레임) 계산, 고정 dB 색, 차이 보기(B − A) 색
│   ├── synth.py                STEP 5 음표 미리 듣기 합성음
│   └── loudness.py             STEP 5 비교 청취용 음량(무음 블록 제외 RMS)·맞춤 배율 (파일은 바꾸지 않음)
├── engine/                     분석 엔진 (Qt 없음)
│   ├── worker.py               분석 작업 프로세스 (separate · pitch · tempo · probe · asr · emissions · ctc_align)
│   ├── worker_client.py        작업 프로세스 실행·진행률·취소·오류 코드 → 한국어 메시지
│   ├── dsp/separation.py       Demucs htdemucs·Mel-Band RoFormer 실행, 시간 정렬, GPU→CPU 재시도
│   ├── dsp/roformer.py         Mel-Band RoFormer 추론(MIT 구현을 torch만으로 옮김), 체크포인트 안전 읽기·구조 추론, 조각 추론
│   ├── dsp/pitch.py            librosa pYIN (16kHz, 100fps, 30초 청크) + 온셋(32ms 창)·RMS(64ms·20ms)
│   ├── dsp/tempo.py            BPM 추정, 2배·절반 후보, 박 간격 직선 보정
│   ├── pitch_post.py           에너지·틈 게이트, 옥타브·튐 보정, 재발음 지표
│   ├── baselines/              STEP 2 알고리즘 고정본(v1). 변경 전후 비교에만 쓰고 앱은 쓰지 않는다
│   ├── notes.py                피치 곡선 → 음표 (v3: 접근음 흡수)
│   ├── transcription.py        분리 → 피치 → BPM → 음표 파이프라인, 음표만 재검출
│   ├── stems.py                stems/ 목록(stems.json) 쓰기·검증·불러오기
│   ├── model_store.py          STEP 5 모델별 분리 결과 보관소 stems/models/<키>/ (키 = 원본·모델·버전·체크포인트·설정 해시), 이전 평면 결과 옮기기
│   ├── model_compare.py        STEP 5 모델 비교 실행: 모델별 작업 프로세스 순차 실행, GPU 메모리 부족은 묻기(oom_gpu), 취소해도 끝난 결과 보관, 모델별 채보
│   ├── stem_import.py          외부 분리 결과 가져오기: 리샘플·채널·교차 상관 어긋남 보정·원본 일치 검사, model 'external'
│   ├── cache.py                cache/ 키·원자적 기록·정리
│   ├── merge.py                재분석 반영 계획 (사용자 편집 보호)
│   ├── settings.py             분석 설정·민감도 프리셋
│   ├── interfaces.py, pipeline.py, registry.py, backends/sample.py   STEP 1 메모리 기반 단계와 샘플 백엔드
├── lyrics/                     가사 (STEP 3·4, Qt 없음)
│   ├── model.py                LyricsText(줄·단어·음절, 고유 id) · Alignment(음절 시간·출처·연결·사용자 멜리스마 판정) · Recognition(인식 원본) · PronEntry(음절별 발음) · LyricsState
│   ├── text.py                 음절 분리(한글·영어 규칙·가나 박·한자·숫자), 정규화, 정렬용 로마자(uroman 호환 자모 표)
│   ├── ops.py                  텍스트 반영(바뀐 줄만, id 유지), 줄 교체·반복, 음절 나누기·합치기, 시간 수정·검사
│   ├── linking.py              자동 연결(1:1·1:N·N:1, 신뢰도), 순서대로 연결, 사용자 연결, 음표 변경 뒤 재연결(reconcile)
│   ├── asr.py · review.py      Whisper 인식(작업 프로세스 안), 환각 의심 '검토 필요' 규칙
│   ├── mms.py · ctc.py         MMS_FA wav2vec2(PyTorch로 직접 구현) · numpy CTC 강제 정렬(고정점·보컬 활동·끝 추정)
│   ├── engine.py               인식·정렬 실행, 캐시(asr·mms), 결과 반영(사용자 수정값 보호)
│   ├── io.py                   lyrics.json(v2: 발음) · alignment.json(v3: 멜리스마 판정) ↔ 객체, STEP 1 형식 옮기기
│   ├── pronunciation/          발음 엔진 (STEP 4): 언어별 변환기 선택, 생성·동기화(사용자 값 보호·재검토 표시)·확정·되돌리기
│   │   ├── korean.py           표준 발음법 규칙(제5·9~29항)을 줄 단위 문맥으로 적용 → 발음 한글, 간략 IPA
│   │   └── japanese.py         가나 박 → 가타카나·IPA (한자는 변환 불가)
│   └── vocal.py                음절 구간·유성 가창 길이, 멜리스마 판정(종류 + 근거 속성), 사용자 판정 편집·음표 변경 뒤 옮기기
├── compare/                    STEP 5 비교·검토 (Qt 없음, 사용자 데이터 읽기 전용)
│   ├── review.py               분석 오류 검토 규칙(근거 숫자를 사유에 적음)·검토 상태(지문이 바뀌면 재검토)
│   ├── score_io.py · reference.py   MIDI·MusicXML(MXL) 가져오기, 참조 악보 데이터
│   ├── align.py · intonation.py     악보 ↔ 음원 정렬(상호상관 + DTW + 우연 기준), 음정 비교(센트·도달 시간·비브라토 중심)
│   ├── model_diff.py           두 분리 모델로 만든 채보의 차이(음표 대응·음높이·시각·재발음 분할·유성·옥타브), '모델 비교' 검토 항목. 우열 판정 없음
│   └── source_switch.py        분석 소스 전환 영향 미리 보기 (실제 반영과 같은 plan_reanalysis·linking.reconcile로 계산)
├── export/                     STEP 5 채보·가사 내보내기: USTX(OpenUtau 0.6), MIDI, MusicXML, JSON, CSV, LRC
├── project/                    프로젝트 폴더 (Qt 없음)
│   ├── layout.py               폴더 구조, Windows 이름 규칙, 중복 이름 처리
│   ├── schema.py               파일 종류·스키마 버전·마이그레이션
│   ├── workspace.py            생성(복사/연결)·열기·저장·재연결·다른 이름으로 저장
│   ├── analysis_io.py          analysis/pitch·notes·tempo·lyrics·alignment.json
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
    ├── document.py             편집 API, QUndoStack, ApplyAnalysisCommand, SetLyricsCommand (음표 교체 명령이 가사 연결도 함께 되돌림)
    ├── lyrics_controller.py    가사 탭·가사 레인·피아노 롤 강조·재생 연동·가사 메뉴
    ├── vocal_controller.py     STEP 4: 발음·길이·멜리스마 판정 캐시, 편집 동작(실행 취소 가능), 가사·발음 표시 방식
    ├── player.py               QAudioSink 재생, 트랙 믹서(솔로·음소거·음량), 반복·속도, 여러 트랙 상태 한 번에 바꾸기
    ├── listening_controller.py STEP 5 반복 구간·느린 재생·음표 미리 듣기·레인(파형·스펙트로그램) 표시
    ├── review_controller.py    STEP 5 검토 탭 (다른 기능이 넣는 extra_items 포함)
    ├── compare_controller.py   STEP 5 '악보 비교' 탭 (참조 악보 가져오기·정렬·음정 비교)
    ├── export_controller.py    STEP 5 채보·가사 내보내기 대화상자
    ├── model_compare_controller.py  STEP 5 모델 비교 모드: A/B 패널(왼쪽), '모델 차이' 탭, A/B 청취(cmp_a·cmp_b 솔로 전환), 분석 소스 선택
    ├── workers.py              TaskRunner (QThreadPool)
    ├── autosave.py             자동 저장·비정상 종료 복구
    ├── dialogs.py              새 프로젝트, 내보내기, 재분석 반영 대화상자
    ├── theme/                  Obsidian Violet 토큰·QSS·글꼴(OFL)·아이콘(Lucide, ISC)
    ├── pianoroll/              QGraphicsView 피아노 롤
    └── widgets/                트랙 패널, 속성·분석 설정 패널, 가사 패널, 가사 레인, 파형, 눈금자, 건반, 로그
scripts/install_separation.py   GPU에 맞는 PyTorch + demucs 설치
scripts/install_lyrics.py       openai-whisper 설치 (PyTorch 다음)
scripts/validate_audio.py       실제 음원 검증 보고서
scripts/compare_runs.py         프로젝트의 분석 실행 기록 나열·비교
run.bat, install_separation.bat, install_lyrics.bat   Windows 실행·설치
```

의존 방향: `lyrics.model ← core ← audio ← engine ← lyrics ← project ← app`. `app`만 Qt를 씁니다. 분석 엔진과 저장 계층은 Qt 없이 테스트됩니다.

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
   **외부 분리 결과**(`model: "external"`, `amp/engine/stem_import.py`)도 같은 방식으로 `stems/`에 쓰고, 분석은 이것을 그대로 재사용합니다
   (Demucs로 덮어쓰지 않음, '다시 분리'를 명시할 때만 교체하고 그 전에 `backup/`에 복사). 가져올 때 원본에 맞추는 순서:
   같은 디코더로 읽기 → 샘플레이트가 다르면 `resample_poly` → 채널 맞춤 → 8kHz 교차 상관(±2초)으로 어긋남을 찾고 원래 샘플레이트에서
   ±8샘플로 다듬기 → 옮기고 원본 프레임 수에 맞춤. 검사: 반주가 있으면 (원본 − 보컬 − 반주)가 원본보다 6dB 넘게 작아야 하고(−20dB 위면 경고),
   보컬만 있으면 원본과 상관 0.08 이상. 반주가 없으면 원본 − 보컬. `stems_id`에 가져온 파일 해시를 넣어 다시 가져오면 피치 캐시가 바뀝니다.
   **Mel-Band RoFormer**(`model: "mel_roformer"`): 작업 프로세스가 사용자가 고른 체크포인트를 `torch.load(weights_only=True)`로 읽고,
   텐서 모양에서 dim·층 수·대역 수·머리 수(게이트가 없으면 회전 주파수 개수로)·마스크 깊이를 읽어 모델을 만든 뒤 모든 이름·모양이 맞는지 검사합니다.
   STFT·샘플레이트는 텐서로 알 수 없어 멜 대역별 입력 크기를 다시 계산해 비교하고, 다르면 .yaml을 요구합니다. 정규화 없이 44.1kHz에서
   조각(기본 8초, 겹침 2)을 반사 패딩·선형 페이드 창으로 이어 붙이고(창 합으로 나눠 항등 모델이면 오차 1e-5 미만), 반주 = 원본 − 보컬.
   재사용 조건: stems.json의 `checkpoint_identity`(경로·크기·수정 시각)와 설정 파일 이름이 지금 설정과 같을 때. 모델 파일 경로는 QSettings
   `separation/roformer`에 두고 `TranscriptionRequest.model_files`로 넘깁니다.
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
- 접근음 흡수(STEP 3.5, 검출 v3, `merge_approach`): 프레이즈 첫 음이거나 재발음으로 시작한 0.25초 이하 조각이 바로 뒤의 더 긴
  1~7반음 위 음으로 올라가면 합칩니다. 조건: 뒤 음 시작의 재발음 지표 2.0 미만, 조각 안 기울기 1.5반음/초 이상(끝 1/5 경계 제외),
  0.3반음 폭에 0.12초 이상 머물지 않음, 4반음 이상이면 조각 안에서 1반음 이상 상승, 정렬한 가사가 있으면 뒤 음에서 다른 음절이
  시작하지 않음(`NoteDetectionParams.syllable_starts`, 설정 파일에는 저장하지 않음). 피치 곡선은 바꾸지 않고 높이는 뒤 음 높이입니다.

## 데이터와 저장

- `AnalysisResult`는 분석이 만든 원본이고 바꾸지 않습니다. `Score`는 작업본입니다. 작업본 음표의 `origin_ids`로 원본·수정·추가를 가립니다.
- 프로젝트 폴더에서는 `notes.json`의 `analysis`(원본)와 `edits`(작업본), `tempo.json`의 `analysis`와 `user`를 나눠 저장합니다.
  작업본만 바뀌는 저장에서도 원본 부분은 메모리의 불변 객체를 그대로 다시 쓰므로 손상되지 않고, 원자적 교체와 .bak으로 보호합니다.
  `pitch.json`은 분석 결과가 바뀐 경우에만 다시 씁니다.
- 각 JSON은 `schema`·`schema_version`을 가지고, `project/schema.py`의 `MIGRATIONS`로 버전을 올립니다. 더 새로운 버전은 열지 않습니다.
- `project.json`: ID·이름·생성/수정 시각·스키마 버전·원본(방식, 상대/절대 경로, SHA-256, 크기, 길이)·분리 상태·분석 상태·
  편집 상태(원본/수정/추가 개수)·분석 설정·재생 위치·트랙 상태·작업 공간(확대·스크롤·레이어·스냅·패널 크기).
- 자동 복구 스냅샷은 STEP 1 형식(zip)으로 앱 데이터 폴더에 쓰고, 복구할 때 프로젝트 폴더를 연 뒤 내용을 덮어 변경 상태로 둡니다.
  STEP 3부터 스냅샷에 `lyrics.json`·`alignment.json`도 담습니다.

## 가사 (STEP 3)

- **책임 분리**: 가사 글자의 주인은 `lyrics.json`(줄·단어·음절과 고유 id, 언어, 자동 인식 원본, 편집 이력),
  시간과 연결의 주인은 `alignment.json`(음절 id → 시작·끝·신뢰도·시각 출처·사용자 수정·다시 정렬 필요, 연결 음표 id·연결 신뢰도·출처·확정).
  `notes.json` v2의 음표는 가사를 모릅니다. v1의 `syllable_id`·`edits.lyrics`는 열 때 옮기고 원본은 `backup/`에 남깁니다.
- **불변 상태 + 스냅샷 실행 취소**: `LyricsState`는 불변 객체입니다. 모든 가사 편집은 새 상태를 만들고 `SetLyricsCommand`로 바꿉니다.
  음표 교체 명령(`ReplaceNotesCommand`·`ApplyAnalysisCommand`)은 `linking.reconcile`로 연결을 다시 이은 상태를 함께 보관해,
  실행 취소하면 음표와 연결이 같이 돌아갑니다. 음절 텍스트·시간은 음표 편집으로 바뀌지 않습니다.
- **시각 출처**: model(정렬 모델이 검출한 음절 시작) / asr_word(Whisper 단어 시각) / estimated(단어·줄 안을 나눈 값) /
  inferred(음절 끝: 다음 음절 시작·보컬 끊김) / notes / user / legacy. 화면은 실선·점선·굵은 선으로 구분합니다.
- **정렬**: 보컬 → 16kHz → MMS_FA(작업 프로세스, 결과 `emissions.npz`를 캐시) → numpy CTC 정렬(별도 작업 프로세스).
  사용자가 고친 음절은 고정점이 되고, 고정점 사이만 다시 정렬합니다. 가사를 고친 뒤 '바뀐 부분만'은 미정렬·stale·추정 음절만 정렬합니다.
  정렬하는 동안 사용자가 가사를 고쳤을 수 있어, 결과는 끝난 시점의 가사에 다시 반영합니다(지운 음절·직접 고친 음절은 건너뜀).
- **접근음 연결**(STEP 3.5): 연결에 역할(`NoteLink.role`)이 있습니다. 음절 첫 음표가 음절 시작에서 시작하는 0.25초 이하의
  낮은 음이고 바로 더 긴 위 음으로 이어지면 'approach'로 잇고, 가사 글자는 주 음표(`linking.main_note_id`)에 씁니다.
  음표 편집으로 다시 이은 연결은 역할을 넘기지 않되, 확정하지 않은 자동 연결이고 연결 음표를 모두 알면(음표만 다시 검출 등) 새 음표로 다시 판정합니다.
  확정 연결의 역할은 자동 연결이 바꾸지 않습니다. `alignment.json` v2(v1은 그대로 올리고, 덮어쓰기 전 원본을 `backup/`에 남김).
- **F6 연속 작업**(STEP 3.5, `LyricsChain`): 인식 → (작업 끝 확인 뒤) 정렬 → 음표 연결 → 저장. 단계마다 실제 장치·시간을 기록하고,
  실패·취소한 단계와 다시 시작할 곳(F6/F7/자동 연결)을 가사 탭에 남깁니다. 작업 전부터 저장하지 않은 편집이 있었으면 저장하지 않고
  복구 파일만 씁니다.
- **인식**: Whisper(작업 프로세스). 노래는 '말소리 아님' 판정이 높게 나와 창이 통째로 빠지므로 무음 판정으로 건너뛰지 않고,
  분리 보컬의 구간 음량·활동 비율, '보컬 없는 구간', 흔한 환각 문구, 반복으로 '검토 필요'를 표시합니다. 아무것도 지우지 않습니다.

## 재분석과 사용자 편집 보호

`engine/merge.plan_reanalysis`가 반영 계획을 만들고 `ApplyAnalysisCommand`가 원본 분석·음표·템포를 함께 바꿉니다(실행 취소 가능).
편집이 있으면 대화상자에서 개수를 비교해 고릅니다.

- **수정 유지하며 적용**: 수정·추가 음표는 그대로 두고, 손대지 않은 원본 음표만 교체. 보존 음표와 20% 넘게 겹치는 새 음표는 넣지 않음.
  사용자가 바꾼 템포도 유지.
- **전부 새 분석으로 교체** / **취소**.

## 화면 멈춤 감시

- 메인 창이 100ms 타이머로 이벤트 루프 지연을 재고, 0.5초 넘게 멈추면 그때 진행 중이던 작업과 함께 로그 파일에 경고를 남깁니다(진단용).

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
| `test_roformer.py` | RoFormer 이식: 출력 길이, 마스크 1이면 입력 복원, 회전 인코딩 = 복소수 회전, 멜 대역 덮기, 파라미터 이름 형태(참조 구현과 같음), 체크포인트 감싸기·접두어, 옛 형식(게이트·출력 정규화 없음), 모노 모델, 코드 객체가 든 파일 거부, 손상·다른 모델 파일, STFT 불일치 → .yaml 요구, 조각 추론 항등성, 분리 결과 합 = 원본, 작업 프로세스 확인 명령, 파이프라인 재사용·모델 파일 변경 시 재분리 |
| `test_stem_import.py` | 외부 분리 결과: 같은 파일·앞 무음·앞 잘림 보정(샘플 단위), 48kHz·모노, 원본 − 보컬, 다른 곡·길이 차이 거부, 음량 변경 경고, stems.json 기록, 분석이 분리를 건너뛰고 가져온 보컬 사용 |
| `test_project_folder.py` | SONATLAS 요구 1~12: 생성(복사·연결), 중복 이름, stems 저장, WAV/MP3·일괄 내보내기, 저장·재열기, 폴더 이동, 재연결, 다른 이름으로 저장, 캐시 삭제 후 보존, 저장 실패 시 기존 파일 보존, 손상 파일 .bak 복구, 한글·특수문자 경로, 마이그레이션, .ampproj 변환 |
| `test_cache_settings_merge.py` | 캐시 원자성·한글 경로, 설정 프리셋, 재분석 병합 규칙(요구 7) |
| `test_accuracy.py` | STEP 2.5: 합성 사례 전 과정 허용 오차(F1·시작/끝 오차·cent·옥타브·무성 오경보), v1↔v2 비교(기본·변형 사례), 온셋 정렬, 틈 게이트 |
| `test_project_lock.py` | STEP 2.5: 같은 프로세스 이중 열기, 다른 프로세스 차단·해제, 강제 종료 뒤 잠금 회수, 열린 채 복사한 폴더 독립 열기, 다른 컴퓨터의 잠금 파일, 이동, 해제, 임시 파일 정리 |
| `test_stability.py` | STEP 2.5: 캐시 손상 감지·재계산, 잘못된 새 결과 미확정, 강제 종료된 작업, 디스크 부족·권한 분류, 잘린·깨진 MP3, GPU 메모리 부족 → CPU, CPU 메모리 부족, 모델 다운로드 실패, torch·demucs 없음, 실패한 분석이 프로젝트 파일을 바꾸지 않음 |
| `test_runs.py`, `test_validate_tool.py` | STEP 2.5: 실행 기록·비교 보고서·CLI, 검증 도구(음높이 표기, 정렬 지연 측정, 정답 CSV, 끝까지 실행) |
| `test_lyrics.py` | STEP 3 요구 1~14: 한국어·영어·일본어 음절, 수동 입력 저장(파일 분리·NaN 없음), 인식 결과 수정(id 유지·stale), 줄 교체·반복, 시간 경계 수정·검사, 1:1·1:N·N:1 연결, 음표 나누기·합치기·지우기 뒤 연결, 바뀐 부분 재정렬(사용자 값 보호·캐시), notes.json v1 마이그레이션, 복구 파일, 캐시 재사용·무효화, 취소·실패 뒤 정리, 잘못된 시각, CTC 정렬기, MMS 구조와 공개 체크포인트 대조, 환각 검토 규칙 |
| `ui/test_lyrics_ui.py` | STEP 3 화면: 가사 레인 끌기·경계 조절·실행 취소, 클릭 이동·재생 중 음절, 음표 아래 가사(N:1·멜리스마)·음표 나누기, 선택 연동·재생 강조·표시 끄기, 인식·정렬 결과 반영(한 단계 실행 취소), TXT 인코딩, 시험용 모델로 전체 흐름 → 저장 → 다시 열기 |
| `ui/test_main_window.py` | 프로젝트 생성·샘플 분석, 손상 파일, 저장·재열기(재생 위치·트랙 상태), 이중 열기 잠금, 분할·병합·BPM·첫 박, 비정상 종료 복구, 분석→트랙→솔로·음소거→내보내기→편집 보호, .ampproj 변환, 시간축 동기화(요구 5), 열린 프로젝트 복사본을 두 번째 창에서 열기, 실행 기록·비교 보고서, 실패한 분석 미기록, 저장 충돌 확인 |
| 나머지 | STEP 1 회귀: 모델, 편집, 템포, 오디오·PCM·믹서, 저장 형식, 문서·실행 취소, 피아노 롤 마우스 조작, 작업 실행기, 자동 저장, 재생기, 진입점, 설치 스크립트 |

## 모델별 분리 결과와 모델 비교 (STEP 5)

```text
stems/
├── models/
│   ├── htdemucs-1a2b3c4d5e/        vocals.wav · instrumental.wav · stems.json (이 결과의 목록 + identity)
│   ├── mel_roformer-6f7e8d9c0b/
│   └── external-0123456789/        외부 분리 결과 가져오기
└── stems.json                      지금 분석에 쓰는 결과를 가리킴 (파일 경로가 models/<키>/…, model_info.store_key)
```

- 키 = 모델 이름 + sha1(identity)[:10]. identity: 원본 해시, 모델, Demucs 패키지 버전·shifts·overlap·split·shift seed
  (htdemucs), RoFormer 코드 버전·체크포인트 내용 SHA-256·설정 파일 SHA-256(mel_roformer), 가져온 파일 해시(external).
  장치(CPU/GPU)는 넣지 않는다. 체크포인트를 같은 내용으로 옮기면 같은 키, 내용이 바뀌면 다른 키.
- 위쪽 `stems.json`은 STEP 2 형식 그대로라 재생·가사 인식·내보내기·`workspace.stems_status()`가 바뀌지 않았다. 경로가 모두 상대라
  폴더를 옮기거나 복사해도 열린다. 이전 버전 프로젝트의 평면 결과(`stems/vocals.wav`)는 처음 쓸 때 `os.replace`로 모델 폴더로 옮긴다
  (`created`가 같아 `stems_id`와 피치 캐시가 그대로 이어진다).
- 파이프라인 `_separation`: 지금 결과가 설정 모델과 맞으면 재사용 → 같은 키가 보관소에 있으면 그것을 가리킴 → 없으면 분리해서 새 폴더.
  '다시 분리'는 같은 키 폴더만 바꾼다. 다른 모델·가져온 결과는 남는다.
- 비교 실행(`model_compare.run_compare`)은 모델마다 작업 프로세스를 하나씩 띄운다. 프로세스가 끝나야 다음 모델을 올리므로 GPU에 두 모델이
  함께 있지 않다. `cpu_fallback=False`라 GPU 메모리 부족은 `oom_gpu` 오류로 돌아오고, 화면이 CPU로 다시 할지 묻는다.
  분석 설정의 보통 분석은 예전처럼 GPU 메모리가 부족하면 CPU로 다시 실행한다(이번에 바꾸지 않음).
- 모델별 채보는 `TranscriptionRequest.stems_key`로 보관한 결과를 읽어 같은 피치·음표 설정으로 돌린다. 피치 캐시 키에 분리 결과 id가 들어가
  모델별로 따로 쌓인다. 지금 분석에 쓰는 결과는 바꾸지 않는다.
- A/B 청취: 비교 모드가 재생 트랙 `cmp_a`·`cmp_b`를 올리고 `AudioPlayer.set_track_states`로 솔로를 한 번에 바꾼다. 스트림을 다시 만들지
  않으므로 위치·반복·속도가 그대로고, `PcmMixer`의 음량 램프가 12ms 크로스페이드를 만든다. 장치 버퍼(200ms) 때문에 들리는 시점은 그만큼 늦다.
- 분석 소스 선택: `ModelStore.activate`(위쪽 stems.json만 바꿈) → 재생 트랙 다시 읽기 → `apply_analysis(keep_edits)`. 바뀔 항목은
  `source_switch.preview`가 같은 함수로 미리 계산해 보여 준다. A/B로 고른 키만 `analysis/model_compare.json`에 적는다.

## v0.6 보컬 다듬기

| 모듈 | 책임 |
|---|---|
| `engine/hybrid.py` | Hybrid 구간 계획(샘플 번호 경계, 빈틈·겹침 불가), 편집 연산(나누기·합치기·경계 이동·모델 지정·새 구간·지우기), 소스 상태 검사(결과 없음·다시 만들어짐·체크포인트 변경 → 검증 필요), 결합(직선 교차, 구간 이득, 음량 맞춤 선택, 잘림 방지) → 보관소 항목 `hybrid-<해시>`, 오래된 Hybrid 결과 정리(지금 쓰는 것 + 최근 2개 유지) |
| `engine/dsp/pitch.py` | `track_pitch(reuse=…)`: 30초 청크의 16kHz 입력이 이전 결과와 비트 단위로 같으면 그 청크의 f0·확률을 옮김. 온셋·RMS는 늘 새로 계산 |
| `engine/transcription.py` | 분리 소스가 Hybrid면 그 소스·이전 Hybrid의 피치 캐시를 재사용 후보로 작업 프로세스에 넘김. 활성 Hybrid는 설정의 분리 모델로 덮어쓰지 않음. 음표 검출 v4 |
| `engine/pitch_post.py` | 재발음 지표 v4: 음량 하락 또는 (앞뒤가 유성·같은 높이인 곳의) 유성 확률 하락 |
| `engine/boundaries.py` | 음 끝 잔향 자르기 (실험, 기본 꺼짐) |
| `engine/merge.py` | `carry_note_ids`: 재분석에서 그대로인 자동 음표는 이전 id를 이어받음 (가사 연결·검토 상태 유지) |
| `compare/suggestions.py` | 교정 제안 6종, 근거 문장, 규칙 점수(확률 아님), 지난 제안 판정, 적용 계획(음표만 계산) |
| `compare/quality.py` | 정답 없는 지표, 검토 현황 숫자, 참조 악보 비교(단서), 모델별 지표 |
| `compare/review.py` | 음절 정렬 규칙 추가: 음표 하나에 음절 몰림, 이웃 음절 겹침, 정렬 확신 낮음, 연결 음표가 음절 밖 |
| `app/hybrid_controller.py`, `app/widgets/hybrid_lane.py` | Hybrid 편집 모드(모델 비교 모드의 왼쪽 탭), 레인(끌기 중에는 기록하지 않고 놓을 때 한 번), 만들기·전후 비교·분석 소스 전환 |
| `app/suggestions_controller.py` | 교정 제안 탭, 피아노 롤 미리보기(`scene.set_preview`), 여러 제안 적용을 실행 취소 한 단계로 |
| `app/quality_controller.py` | 품질 탭, 숫자 → 검토 탭 종류 필터 |

저장: `analysis/hybrid.json`(sonatlas.hybrid v1), `analysis/suggestions.json`(sonatlas.suggestions v1). 둘 다 원자적 쓰기 +
`.bak`. 깨진 파일은 프로젝트를 열고 경고만 남기며 지우지 않는다. 더 새로운 버전 파일은 읽지 않는다. 계획을 지우면 파일은
`.removed.bak`으로 돌린다. 문서 필드 `hybrid`·`suggestions`는 `SetFieldCommand`로 바뀌어 실행 취소된다.

### v0.6.1 피치 다시 찾기 구간

`AnalysisSettings.pitch_regions` = [(시작초, 끝초, 최저Hz, 최고Hz)] (프로젝트에만 저장). 파이프라인은 곡 전체 피치(캐시)를 읽은 뒤
구간마다 작업 프로세스 명령 `pitch_region`을 부른다(`dsp/pitch.track_region`: 같은 100fps 격자, 앞뒤 1초 덧붙임, pYIN 문턱 분포 베타(1,4),
좁은 음역). 홀/짝 배음 비가 0.15 미만인 프레임(높은 목소리의 서브하모닉)은 버린다. 다시 찾은 프레임만 바꾸고 나머지는 원래 피치를 둔다.
구간 결과는 `pitch_region` 단계로 따로 캐시한다(구간·음역·입력 결과 id가 키). 음표 재검출은 끼워 넣은 피치 곡선을 그대로 쓴다.

