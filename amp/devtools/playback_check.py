"""재생 안정성 점검 도구 (STEP 3.5). 사용자 PC(Windows)에서 실행해 Qt 오디오 경고가 다시 나는지 본다.

    python scripts/playback_check.py                  # 합성 음 20초로 시나리오 3회
    python scripts/playback_check.py --audio 곡.mp3 --rounds 5
    python scripts/playback_check.py --device-change 20   # 20초 동안 직접 출력 장치를 바꿔 본다

앱과 같은 AudioPlayer로 재생 시작·일시정지·정지·재생 중 위치 이동·반복 재생·트랙 전환·프로젝트 닫기·
새 프로젝트 열기·끝까지 재생을 반복한다. 그동안 Qt가 내는 모든 경고를 모아, 'device not open' 같은
재생 장치 경고 수를 센다. 소리가 실제로 들렸는지는 이 도구가 알 수 없다 (귀로 확인).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# 사용자 Windows 로그에 나왔던 경고와 같은 계열 (재생 장치를 읽거나 여는 문제)
DEVICE_WARNINGS = ("device not open", "QIODevice", "pullFromQIODevice", "QAudioSink", "wasapi")
WARNING_TYPES = ("QtWarningMsg", "QtCriticalMsg", "QtFatalMsg")


def _app():
    from PySide6.QtCore import QCoreApplication

    return QCoreApplication.instance() or QCoreApplication([])


class Recorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.step = ""

    def __call__(self, mode, _ctx, msg: str) -> None:
        self.messages.append({"step": self.step, "type": str(getattr(mode, "name", mode)), "text": msg})

    def warnings(self) -> list[dict]:
        return [m for m in self.messages if m["type"] in WARNING_TYPES]

    def device_warnings(self) -> list[dict]:
        return [m for m in self.warnings() if any(k.lower() in m["text"].lower() for k in DEVICE_WARNINGS)]


def _audio(path: str | None, seconds: float):
    import numpy as np

    from amp.audio.data import AudioData

    if path:
        from amp.audio.decoder import load_audio

        return load_audio(path)
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    y = 0.15 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    y = y.astype(np.float32)
    return AudioData(np.stack([y, y], 1), sr)


def run(rounds: int, audio_path: str | None, device_change_s: float, out_root: Path) -> tuple[dict, Path]:
    from PySide6.QtCore import qInstallMessageHandler
    from PySide6.QtMultimedia import QMediaDevices

    app = _app()
    rec = Recorder()
    qInstallMessageHandler(rec)
    from amp.app.player import AudioPlayer, PlaybackState

    audio = _audio(audio_path, 20.0)
    player = AudioPlayer()
    errors: list[dict] = []
    player.errorOccurred.connect(lambda m: errors.append({"step": rec.step, "text": m}))
    steps: list[dict] = []

    def spin(s: float) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < s:
            app.processEvents()
            time.sleep(0.005)

    def step(name: str, fn, wait: float) -> None:
        rec.step = name
        t0 = time.monotonic()
        fn()
        spin(wait)
        steps.append({"step": name, "state": player.state.value, "position": round(player.position(), 2),
                      "s": round(time.monotonic() - t0, 2)})

    out = {"created": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
           "python": sys.version.split()[0], "rounds": rounds, "audio": audio_path or "합성 음 20초",
           "output_available": AudioPlayer.output_available()}
    try:
        import PySide6

        out["pyside6"] = PySide6.__version__
    except Exception:  # noqa: BLE001
        pass
    d = QMediaDevices.defaultAudioOutput()
    out["default_output"] = None if d.isNull() else d.description()
    out["outputs"] = [x.description() for x in QMediaDevices.audioOutputs()]
    if not out["output_available"]:
        out["summary"] = "오디오 출력 장치가 없어 점검하지 못했습니다."
        return _write(out, rec, errors, steps, out_root)

    vocal = type(audio)(audio.samples * 0.5, audio.sample_rate)
    dur = audio.duration
    for r in range(rounds):
        player.set_audio(audio)
        step(f"{r}:재생 시작", player.play, 1.2)
        step(f"{r}:일시정지", player.pause, 0.4)
        step(f"{r}:다시 재생", player.play, 0.8)
        step(f"{r}:정지", player.stop, 0.3)
        step(f"{r}:재생", player.play, 0.5)
        for k in range(10):
            step(f"{r}:재생 중 이동 {k}", lambda k=k: player.seek((k * 1.7) % max(1.0, dur - 2.0)), 0.15)
        for k in range(5):  # 구간 반복: 끝에서 처음으로 돌아가기
            step(f"{r}:반복 {k}", lambda: player.seek(2.0), 0.6)
        step(f"{r}:트랙 추가 (보컬)", lambda: player.set_track("vocals", vocal), 0.6)
        step(f"{r}:솔로", lambda: player.set_track_state("vocals", solo=True), 0.4)
        step(f"{r}:솔로 해제", lambda: player.set_track_state("vocals", solo=False), 0.3)
        step(f"{r}:트랙 빼기", lambda: player.set_track("vocals", None), 0.5)
        step(f"{r}:프로젝트 닫기 (재생 중)", lambda: player.set_audio(None), 0.4)
        step(f"{r}:새 프로젝트 열기", lambda: player.set_audio(audio), 0.2)
        step(f"{r}:재생", player.play, 0.6)
        step(f"{r}:끝 근처로 이동", lambda: player.seek(max(0.0, dur - 0.8)), 1.6)  # 끝까지 재생
        if player.state is not PlaybackState.STOPPED:
            steps[-1]["note"] = "끝까지 재생했는데 정지 상태가 아님"
    if device_change_s > 0:
        player.set_audio(audio)
        player.play()
        print(f"\n지금 {device_change_s:.0f}초 안에 출력 장치를 바꿔 보세요 (Windows 소리 설정에서 "
              "출력 장치를 다른 것으로 바꿨다가 되돌리기. USB 사운드카드의 헤드폰 단자는 꽂아도 장치가 바뀌지 않을 수 있음). "
              "소리가 새 장치로 이어지는지 귀로 확인하세요.", flush=True)
        rec.step = "출력 장치 바꾸기"
        before = player.device_switches
        t0 = time.monotonic()
        while time.monotonic() - t0 < device_change_s:
            spin(0.2)
            if player.state is PlaybackState.STOPPED:
                player.play()
        out["device_change"] = {"seconds": device_change_s, "switches": player.device_switches - before,
                                "state_after": player.state.value,
                                "default_after": QMediaDevices.defaultAudioOutput().description()}
    player.shutdown()
    rec.step = "종료"
    spin(0.5)
    dw = rec.device_warnings()
    out["summary"] = (f"재생 장치 경고 {len(dw)}개, 그 밖의 Qt 경고 {len(rec.warnings()) - len(dw)}개, "
                      f"재생 오류 알림 {len(errors)}개 (시나리오 {rounds}회, 단계 {len(steps)}개)")
    return _write(out, rec, errors, steps, out_root)


def _write(out: dict, rec: Recorder, errors: list, steps: list, out_root: Path) -> tuple[dict, Path]:
    out["device_warnings"] = rec.device_warnings()
    out["other_warnings"] = [m for m in rec.warnings() if m not in out["device_warnings"]][:100]
    out["qt_messages"] = rec.messages[-300:]
    out["player_errors"] = errors
    out["steps"] = steps
    out_dir = (out_root / f"playback_check-{datetime.now().strftime('%Y%m%d-%H%M%S')}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "playback_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# SONATLAS 재생 점검 ({out['created']})", "", f"- {out['platform']} · Python {out['python']} · "
             f"PySide6 {out.get('pyside6', '?')}", f"- 기본 출력: {out.get('default_output')} · 출력 장치: {out.get('outputs')}",
             f"- 입력: {out['audio']}", "", f"## 결과: {out['summary']}", ""]
    if out.get("device_change"):
        dc = out["device_change"]
        lines.append(f"- 출력 장치 바꾸기 {dc['seconds']:.0f}초: 새 장치로 다시 연 횟수 {dc['switches']}, "
                     f"끝난 뒤 상태 {dc['state_after']}, 기본 장치 {dc['default_after']}")
    if out["device_warnings"]:
        lines += ["", "### 재생 장치 경고 (단계별)", ""]
        lines += [f"- [{m['step']}] {m['text']}" for m in out["device_warnings"][:50]]
    if out["other_warnings"]:
        lines += ["", "### 그 밖의 Qt 경고", ""] + [f"- [{m['step']}] {m['text']}" for m in out["other_warnings"][:30]]
    if errors:
        lines += ["", "### 재생 오류 알림", ""] + [f"- [{e['step']}] {e['text']}" for e in errors[:30]]
    odd = [s for s in steps if s.get("note")]
    if odd:
        lines += ["", "### 확인할 단계", ""] + [f"- {s['step']}: {s['note']}" for s in odd]
    lines += ["", "소리가 끊기거나 튀었는지는 이 도구가 알 수 없습니다. 들으면서 이상한 단계가 있었으면 함께 알려 주세요."]
    md = out_dir / "playback_check.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SONATLAS 재생 안정성 점검")
    ap.add_argument("--audio", default=None, help="재생할 파일 (없으면 합성 음 20초)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device-change", type=float, default=0.0, help="이 초 동안 직접 출력 장치를 바꿔 본다")
    ap.add_argument("--out", default="playback_check_reports")
    a = ap.parse_args(argv)
    out, md = run(a.rounds, a.audio, a.device_change, Path(a.out))
    print(md.read_text(encoding="utf-8"))
    print(f"결과 파일: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
