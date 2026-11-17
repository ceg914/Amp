# Amp 보컬 채보기

MP3에서 보컬 멜로디를 뽑아 음표로 만들고, 피아노 롤에서 고치는 Windows용 데스크톱 프로그램입니다.
Python과 PySide6로 만들고 있습니다.

**현재 버전: 0.1.0 (1단계).** 오디오 불러오기·재생·파형 표시·피아노 롤 편집까지 동작합니다.
AI 분석은 아직 없습니다. 메뉴의 "샘플 분석"은 오디오 길이만 보고 고정된 패턴의 음표와 피치 곡선을 만들어 냅니다.
화면과 저장 구조를 시험하려고 넣은 기능이고, 소리를 분석한 결과가 아닙니다.

## 기능 구현 현황

| # | 최종 목표 기능 | 상태 | 지금 동작하는 범위 |
|---|---|---|---|
| 1 | MP3/WAV 불러오기 및 재생 | 구현 | MP3·WAV·FLAC·OGG 디코딩(백그라운드), 재생·일시정지·정지·탐색. Windows 실기 재생은 미확인 |
| 2 | 보컬과 반주 분리 | 미구현 | 인터페이스(`SourceSeparator`)만 있음. 현재는 믹스를 그대로 넘김 |
| 3 | 음높이·음 길이·BPM 자동 분석 | 미구현 | 샘플 백엔드가 고정 패턴을 만듦. BPM은 직접 입력 가능 |
| 4 | 비브라토·멜리스마를 고려한 채보 | 미구현 | 음표에 비브라토 정보를 담을 필드만 있음 |
| 5 | 음표와 피치 곡선을 겹친 피아노 롤 | 구현 | 곡선 데이터는 샘플 백엔드가 만든 합성 곡선 |
| 6 | 음표 이동·분리·병합·길이 변경 | 구현 | 추가·삭제·전조 포함, 모든 편집 실행 취소 가능 |
| 7 | 가사 자동 인식 및 수동 입력 | 미구현 | 샘플 가사를 음표 위에 표시만 함. 입력·수정 불가 |
| 8 | 음절 단위 정렬과 발음 기호 표시 | 미구현 | 음절에 시간·발음 기호를 담을 필드만 있음. 화면 표시 없음 |
| 9 | 멜리스마 분석, 음절-음표 연결 | 일부 | 한 음절을 여러 음표에 연결하는 데이터 구조와 표시(`–`)만 구현. 분석은 없음 |
| 10 | 저장, MIDI·JSON 내보내기 | 일부 | 프로젝트 저장·열기·자동 복구 구현. MIDI·JSON 내보내기는 미구현 |

아키텍처 요구사항은 이렇게 반영했습니다.

| 요구사항 | 구현 위치 |
|---|---|
| UI와 분석 엔진 분리 | `amp/engine`, `amp/core`, `amp/audio`는 Qt를 import하지 않음 |
| 분석 모델 교체 | 단계별 추상 클래스(`engine/interfaces.py`)와 등록소(`engine/registry.py`) |
| 작업 스레드 실행 | `app/workers.py`의 `TaskRunner`(QThreadPool). 진행률·취소 지원 |
| QGraphicsView 피아노 롤 | `app/pianoroll/` |
| 원본/수정본 분리 저장 | 프로젝트 파일 안에 `analysis/`(원본)와 `score.json`(작업본)을 따로 둠 |
| 저장·자동 복구 | `storage/project_file.py`, `app/autosave.py` |
| 긴 오디오 대응 | 백그라운드 디코딩, 다중 해상도 파형 요약, 보이는 구간만 그리기 |
| 테스트·오류 처리 | `tests/` 테스트 93개, 사용자용 예외 계층(`core/errors.py`) |

구조와 모듈별 책임은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 정리했습니다.

## 설치와 실행 (Windows)

[python.org](https://www.python.org/downloads/windows/)에서 Windows용 Python 3.10 이상 64비트를 설치합니다(개발은 3.13에서 확인).
설치 화면 기본값을 그대로 두면 `py` 런처가 함께 설치됩니다.

MSYS2(`C:\msys64\...`)나 Cygwin에 들어 있는 Python은 쓸 수 없습니다. PyPI의 PySide6 설치 파일이 이 Python들과 맞지 않습니다.
설치된 공식 Python 목록은 아래 명령으로 볼 수 있습니다.

```bash
py -0p
```

### 방법 1: run.bat (권장)

저장소 폴더의 `run.bat`을 더블클릭하거나 명령 프롬프트에서 실행합니다.

```bash
run.bat
```

처음 실행하면 공식 Python을 찾아(`py -3` → `python` 순서, MSYS2·Cygwin Python은 건너뜀) 저장소 폴더에 가상 환경(`.venv`)을 만들고,
필요한 패키지를 설치한 뒤 프로그램을 켭니다. PySide6 내려받기 때문에 몇 분 걸릴 수 있습니다.
두 번째부터는 설치 단계를 건너뛰고 바로 켜집니다.

### 방법 2: 직접 설치

명령 프롬프트에서 저장소 폴더로 이동한 뒤 가상 환경을 만듭니다.
`python` 대신 `py -3`을 쓰는 이유는 PATH에 다른 Python(예: MSYS2)이 먼저 잡혀 있어도 공식 Python을 쓰기 위해서입니다.

```bash
py -3 -m venv .venv
```

가상 환경을 켭니다. PowerShell이라면 `.venv\Scripts\Activate.ps1`입니다.
켜지면 프롬프트 앞에 `(.venv)`가 붙습니다. **새 창을 열 때마다 다시 켜야 합니다.**

```bash
.venv\Scripts\activate.bat
```

패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

실행합니다.

```bash
python -m amp
```

`soundfile` 휠에 MP3를 읽을 수 있는 libsndfile이 들어 있어 FFmpeg를 따로 설치하지 않아도 됩니다.
일부 MP3를 못 읽을 때를 대비한 보조 디코더도 넣을 수 있습니다(선택).

```bash
pip install miniaudio
```

### 실행 옵션

오디오나 프로젝트 파일을 인자로 주면 바로 엽니다.

```bash
python -m amp "C:\음악\노래.mp3"
```

노래 파일이 없으면 시험용 MP3를 만들 수 있습니다. 사인파라 사람 목소리는 아닙니다.

```bash
python scripts/make_test_audio.py test_tone.mp3 30
```

`pip install -e .`로 설치하면 다른 폴더에서도 `amp` 명령으로 실행됩니다.

## 사용법

1. **파일 > 오디오 열기**(Ctrl+O)로 MP3를 엽니다. 파형이 나타납니다.
2. **분석 > 샘플 분석 실행**(F5)을 누르면 테스트용 음표와 피치 곡선이 채워집니다. 음표를 직접 그려도 됩니다.
3. 피아노 롤에서 음표를 고치고 **파일 > 저장**(Ctrl+S)으로 `.ampproj` 파일에 저장합니다.

| 동작 | 방법 |
|---|---|
| 재생 / 일시정지 | Space |
| 정지 (재생을 시작한 위치로) | Esc |
| 처음으로 | Home |
| 재생 위치 이동 | 눈금자나 파형을 클릭·드래그 |
| 음표 선택 | 클릭, Ctrl+클릭(추가/해제), 빈 곳 드래그(범위), Ctrl+A |
| 음표 이동 | 몸통 드래그 (드래그 중 Alt를 누르면 격자 스냅 해제) |
| 길이 변경 | 음표 왼쪽·오른쪽 끝 드래그 |
| 음표 추가 | 빈 곳 더블클릭 (길이 = 격자 한 칸) |
| 분할 | 오른쪽 클릭 > 여기서 분할, 또는 Ctrl+E(재생 위치에서) |
| 병합 | 두 개 이상 선택 후 Ctrl+M |
| 삭제 | Delete |
| 반음 / 옥타브 이동 | ↑↓ / Shift+↑↓ |
| 격자 단위 이동 | ←→ |
| 실행 취소 / 다시 실행 | Ctrl+Z / Ctrl+Y |
| 가로 확대·축소 | Ctrl+휠, Ctrl+= / Ctrl+-, 전체 보기 Ctrl+0 |
| 세로 확대·축소 | Ctrl+Shift+휠 |
| 가로 스크롤 | Shift+휠 |

음표 색은 출처를 나타냅니다. 파랑은 분석 원본 그대로, 주황은 사용자가 고친 음표, 초록은 직접 추가한 음표입니다.
**편집 > 분석 원본으로 되돌리기**를 쓰면 작업본을 원본으로 돌립니다. 이 동작도 실행 취소할 수 있습니다.

자동 저장은 1분마다, 저장하지 않은 변경이 있을 때만 합니다.
프로그램이 비정상 종료되면 다음 실행 때 복구 여부를 묻습니다.
자동 저장 파일과 로그는 `%LOCALAPPDATA%\Amp\Amp\` 아래 `recovery`, `logs` 폴더에 있습니다.

## 문제 해결

**`ModuleNotFoundError: No module named 'PySide6'`**

`python -m amp`를 실행한 Python에 패키지가 설치되지 않은 상태입니다. 흔한 경우는 두 가지입니다.

- `pip install -r requirements.txt`를 실행하지 않았다.
- 가상 환경에 설치했는데 새 창에서 가상 환경을 켜지 않고 실행했다. 프롬프트 앞에 `(.venv)`가 없으면 이 경우입니다.

`run.bat`을 쓰면 두 경우 모두 자동으로 처리됩니다. 지금 버전은 패키지가 없을 때 트레이스백 대신
어떤 Python으로 실행했는지와 설치 명령을 한국어로 보여 줍니다.

**`C:\msys64\ucrt64\bin\python.exe: No module named pip`**

`python` 명령이 MSYS2에 들어 있는 Python을 실행하고 있습니다. 이 Python에는 pip이 없고,
pip을 설치해도 PySide6가 설치되지 않습니다. 공식 Python을 설치한 뒤 `run.bat`을 실행하거나,
위 "방법 2"처럼 `py -3`으로 가상 환경을 만드세요. 가상 환경을 켠 창에서는 `python`이 가상 환경의 Python을 가리킵니다.
지금 버전은 이런 Python으로 `python -m amp`를 실행하면 이 안내를 직접 보여 줍니다.

어떤 `python`이 먼저 잡히는지는 아래 명령으로 확인할 수 있습니다. 맨 위 줄이 실행되는 Python입니다.

```bash
where python
```

**`python`을 입력하면 Microsoft Store가 열린다**

Python 설치 때 PATH 추가를 체크하지 않은 경우입니다. Python 설치 프로그램을 다시 실행해 "Modify"에서 PATH 추가를 켜거나,
`python` 대신 `py`를 쓰세요(예: `py -m venv .venv`).

## 테스트

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

화면 없이도 돌도록 `tests/conftest.py`가 `QT_QPA_PLATFORM=offscreen`을 지정합니다.
오디오 출력 장치가 없으면 실제 재생 테스트 1개는 건너뜁니다.

## 확인한 범위와 한계

- 개발 중 테스트는 Linux 컨테이너(Python 3.13, PySide6 6.12)에서 화면 없는 모드로 돌렸습니다.
  재생은 PulseAudio 가상 출력 장치로 확인했습니다. **Windows 실제 사운드카드에서는 아직 확인하지 못했습니다.**
- 10분 MP3 기준 측정값(같은 환경): 불러오기와 파형 계산 1.1초, 그동안 UI 이벤트 루프가 가장 길게 멈춘 시간 25ms,
  음표 905개 장면에서 확대 시 재배치 40ms.
- 재생은 장치가 원본 샘플레이트를 지원하지 않으면 선형 보간으로 변환합니다. 미리 듣기 용도로는 충분하지만 음질 손실이 조금 있습니다.
- 가사 편집, 발음 기호 표시, 템포 변화 구간 편집(데이터 구조는 지원), MIDI·JSON 내보내기는 아직 없습니다.

## 다음 단계

1. 실제 피치 추적기(예: CREPE, RMVPE 계열)와 음표 분할 알고리즘을 `PitchTracker`, `NoteTranscriber` 백엔드로 연결
2. 음원 분리 모델 연결, 분리된 보컬·반주 트랙 듣기
3. BPM·박자 추정
4. 가사 입력·편집 UI, 음성 인식 백엔드, 음절 정렬과 발음 기호 표시
5. MIDI·JSON 내보내기
6. 무거운 모델을 별도 프로세스에서 실행하는 작업 실행기
