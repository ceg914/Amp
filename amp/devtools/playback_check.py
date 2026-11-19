"""재생 안정성 점검 도구 (STEP 3.5). 사용자 PC(Windows)에서 실행해 Qt 오디오 경고가 다시 나는지 본다.

    python scripts/playback_check.py                  # 합성 음 20초로 시나리오 3회
    python scripts/playback_check.py --audio 곡.mp3 --rounds 5
    python scripts/playback_check.py --device-change 20   # 20초 동안 직접 출력 장치를 바꿔 본다
    python scripts/playback_check.py --rounds 0 --device-change 30   # 장치 바꾸기만 30초

장치 바꾸기 때는 0.5초마다 Windows에 기본 출력 장치를 직접 묻고(Qt를 거치지 않음), Qt 알림·Qt가 '기본'으로 표시한 장치·
앱이 실제로 재생 중인 장치·다시 연 횟수를 시간순으로 적는다. 끝나면 무엇을 어떻게 바꿨고 무엇을 들었는지 묻는다.
그래서 장치를 안 바꿨는지, Qt가 알리지 않았는지, 앱이 알림을 받고도 다시 열지 않았는지 가를 수 있다.

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


def run(rounds: int, audio_path: str | None, device_change_s: float, out_root: Path,
        probe=None, ask=None) -> tuple[dict, Path]:
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
        out["device_change"] = _device_change(player, audio, device_change_s, spin, rec, probe, ask)
    player.shutdown()
    rec.step = "종료"
    spin(0.5)
    dw = rec.device_warnings()
    out["summary"] = (f"재생 장치 경고 {len(dw)}개, 그 밖의 Qt 경고 {len(rec.warnings()) - len(dw)}개, "
                      f"재생 오류 알림 {len(errors)}개 (시나리오 {rounds}회, 단계 {len(steps)}개)")
    return _write(out, rec, errors, steps, out_root)


def windows_default_output_id() -> dict[str, str] | None:
    """Windows에 역할별 기본 출력 장치 id를 직접 묻는다 (amp.audio.windows_output). Windows가 아니면 None, 실패하면 예외."""
    from amp.audio import windows_output

    return windows_output.default_output_ids()


QUESTIONS = (
    ("changed", "점검하는 동안 출력 장치를 바꿨나요? (y/n): ", {"y": "예", "n": "아니요"}),
    ("method", "어떻게 바꿨나요? (1=작업 표시줄 스피커 아이콘, 2=설정 > 소리, 3=제어판 소리, "
               "4=앱 볼륨 및 장치 기본 설정(앱별 출력), 5=기타): ",
     {"1": "작업 표시줄 스피커 아이콘", "2": "설정 > 소리", "3": "제어판 소리", "4": "앱별 출력 설정", "5": "기타"}),
    ("sound", "다른 장치로 바꿨을 때 원래 스피커 소리는? (1=멈춤·새 장치로 옮겨 감, 2=원래 스피커에서 계속 남, 3=모름): ",
     {"1": "새 장치로 옮겨 감", "2": "원래 스피커에서 계속 남", "3": "모름"}),
)


def _ask_user() -> dict:
    """점검이 끝난 뒤 사용자가 한 일과 들은 것을 묻는다 (콘솔에서 실행할 때만)."""
    if sys.stdin is None or not sys.stdin.isatty():
        return {}
    answers: dict[str, str] = {}
    for key, prompt, choices in QUESTIONS:
        if key != "changed" and answers.get("changed") != "예":
            break
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            break
        answers[key] = choices.get(raw, raw or "답 없음")
    return answers


def device_change_verdict(dc: dict) -> str:
    """장치 바꾸기 시간 동안 모은 기록으로 무엇이 일어났는지 한 줄로 정리한다."""
    win = dc.get("windows_defaults")  # Windows 멀티미디어 기본 장치 (차례대로, 바뀐 것만). None = 물어보지 못함
    qt = dc.get("qt_defaults") or []
    signals, switches = dc.get("qt_signals", 0), dc.get("switches", 0)
    user = dc.get("user_report") or {}
    other_roles = dc.get("windows_other_roles_changed", False)
    qt_lag = win is not None and len(win) > 1 and len(qt) <= 1  # Windows는 바뀌었는데 Qt 기본 장치는 한 번도 안 바뀜
    if switches:
        tail = " Qt가 본 기본 장치는 Windows를 따라가지 않았습니다 (Qt 쪽 문제, 앱이 Windows에 직접 물어 우회)." if qt_lag else ""
        return f"새 기본 장치로 재생 출력을 다시 열었습니다 ({switches}회).{tail}"
    if win is not None and len(win) > 1:
        if not signals:
            return "Windows의 기본 장치는 바뀌었는데 Qt가 장치 변경 알림을 보내지 않아 앱이 다시 열지 못했습니다 (Qt 쪽 문제)."
        return "Windows의 기본 장치가 바뀌고 Qt 알림도 왔는데 재생 출력을 다시 열지 않았습니다 (앱 쪽 문제)."
    if win is None and len(qt) > 1:
        return "Qt는 기본 장치가 바뀐 것을 알았는데 재생 출력을 다시 열지 않았습니다 (앱 쪽 문제)."
    if win is not None:
        if other_roles:
            return ("Windows의 '통신 장치'만 바뀌고 멀티미디어 기본 장치는 그대로였습니다. "
                    "앱과 Qt는 멀티미디어 기본 장치를 따릅니다 (제어판에서 '기본 장치로 설정'을 써야 함).")
        if user.get("changed") == "예":
            how = user.get("method", "")
            hint = " 앱별 출력 설정은 Windows 기본 장치를 바꾸지 않습니다." if how == "앱별 출력 설정" else ""
            return (f"장치를 바꿨다고 답했지만({how or '방법 모름'}) 점검 시간 동안 Windows 기본 장치는 그대로였습니다."
                    f"{hint} 점검 시간 밖에서 바꿨을 수도 있습니다.")
        tail = f" Qt 장치 목록 알림은 {signals}회 있었습니다." if signals else ""
        return "이 시간 동안 Windows의 기본 출력 장치가 바뀌지 않았습니다 (장치를 바꾸지 않았거나 시간 밖에서 바꿈)." + tail
    if signals:
        return f"Qt 장치 목록 알림이 {signals}회 왔지만 기본 장치는 그대로였습니다. Windows에 직접 묻지는 못했습니다."
    return "Qt 알림이 없었고 Windows에 직접 묻지도 못해, 장치가 실제로 바뀌었는지 알 수 없습니다."


def _device_change(player, audio, seconds: float, spin, rec: Recorder, probe=None, ask=None) -> dict:
    """재생을 이어 가며 사용자가 출력 장치를 바꾸게 하고, Windows·Qt·재생기가 각각 무엇을 봤는지 시간순으로 적는다.

    probe: 운영체제에 역할별 기본 출력 장치 id를 묻는 함수 ({"multimedia": id, ...}, 지원하지 않으면 None).
    ask: 끝난 뒤 사용자에게 묻는 함수 (기본은 콘솔 질문).
    기록은 QTimer로 0.5초마다 남기므로 어느 대기 중에도 빠지지 않는다.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtMultimedia import QMediaDevices

    from amp.app.player import PlaybackState

    probe = probe or windows_default_output_id
    ask = ask or _ask_user

    def qt_default() -> str | None:
        d = QMediaDevices.defaultAudioOutput()
        return None if d.isNull() else d.description()

    def qt_flagged() -> list[str]:
        return [d.description() for d in QMediaDevices.audioOutputs() if d.isDefault()]

    def describe(dev_id: str | None) -> str | None:
        if not dev_id:
            return dev_id
        for d in QMediaDevices.audioOutputs():
            if bytes(d.id()).decode("utf-8", "replace") == dev_id:
                return d.description()
        return dev_id  # Qt 목록에 없는 장치면 id 그대로

    def playing_on() -> str | None:
        return describe(player._sink_device_id.decode("utf-8", "replace")) if player._sink is not None else None

    probe_state = {"ok": True, "text": "확인함"}

    def os_defaults() -> dict | None:
        if not probe_state["ok"]:
            return None
        try:
            value = probe()
        except Exception as e:  # noqa: BLE001 - 진단 도구: 실패해도 나머지 기록은 남긴다
            probe_state.update(ok=False, text=f"{type(e).__name__}: {e}")
            return None
        if value is None:
            probe_state.update(ok=False, text="Windows 아님")
            return None
        return {k: describe(v) for k, v in value.items()}

    timeline: list[dict] = []
    t0 = time.monotonic()
    phase = {"name": "준비"}

    def note(source: str, **kw) -> None:
        if len(timeline) < 400:
            timeline.append({"t": round(time.monotonic() - t0, 2), "phase": phase["name"], "source": source, **kw})

    seen = {"os": os_defaults(), "qt": qt_default(), "flagged": qt_flagged(), "playing": None, "switches": 0}
    win_seen = [seen["os"]["multimedia"]] if seen["os"] else []
    qt_seen, flagged_seen = [seen["qt"]], [seen["flagged"]]
    other_roles = {"changed": False}
    signals = {"window": 0, "outside": 0}
    before = player.device_switches

    def poll() -> None:
        osd, q, fl, pl = os_defaults(), qt_default(), qt_flagged(), playing_on()
        if osd is not None and osd != seen["os"]:
            if seen["os"] and osd.get("multimedia") != seen["os"].get("multimedia"):
                win_seen.append(osd.get("multimedia"))
            elif seen["os"]:
                other_roles["changed"] = True
            seen["os"] = osd
            note("windows", **osd)
        if q != seen["qt"]:
            seen["qt"] = q
            qt_seen.append(q)
            note("qt_default", default=q)
        if fl != seen["flagged"]:
            seen["flagged"] = fl
            flagged_seen.append(fl)
            note("qt_flagged", flagged=fl)
        if pl != seen["playing"]:
            seen["playing"] = pl
            note("playing_on", device=pl, state=player.state.value)
        if player.device_switches - before != seen["switches"]:
            seen["switches"] = player.device_switches - before
            note("player", reopened=seen["switches"], state=player.state.value)

    def on_signal() -> None:
        signals["window" if phase["name"] == "점검" else "outside"] += 1
        osd = os_defaults()
        note("qt_signal", qt_default=qt_default(), qt_flagged=qt_flagged(),
             windows=(osd or {}).get("multimedia"), outputs=len(QMediaDevices.audioOutputs()))

    watcher = QMediaDevices()
    watcher.audioOutputsChanged.connect(on_signal)
    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(500)
    note("start", windows=seen["os"], qt_default=seen["qt"], qt_flagged=seen["flagged"])
    print(f"\n출력 장치 바꾸기 점검 ({seconds:.0f}초). 소리가 나는 동안 작업 표시줄 오른쪽 아래의 스피커 아이콘을 누르고, "
          "위쪽 장치 이름을 눌러 다른 장치(예: Steam Streaming Speakers)로 바꿨다가 몇 초 뒤 원래 장치로 되돌리세요.", flush=True)
    player.set_audio(audio)
    player.play()
    for k in (3, 2, 1):
        print(f"  {k}초 뒤 시작", flush=True)
        spin(1.0)
    phase["name"] = "점검"
    rec.step = "출력 장치 바꾸기"
    note("window_open")
    print("지금 바꾸세요.", flush=True)
    w0 = time.monotonic()
    last_print = -10.0
    while (elapsed := time.monotonic() - w0) < seconds:
        spin(0.1)
        if player.state is PlaybackState.STOPPED:
            player.play()
        if elapsed - last_print >= 3.0:
            last_print = elapsed
            os_now = (seen["os"] or {}).get("multimedia") if probe_state["ok"] else "(확인 못 함)"
            print(f"  남은 {seconds - elapsed:3.0f}초 · Windows 기본 장치: {os_now} · 앱이 재생 중인 장치: {seen['playing']} · "
                  f"다시 연 횟수 {player.device_switches - before}", flush=True)
    phase["name"] = "끝"
    note("window_closed")
    poll()
    timer.stop()
    watcher.audioOutputsChanged.disconnect(on_signal)
    state_after = player.state.value
    player.stop()
    dc = {"seconds": seconds, "switches": player.device_switches - before, "state_after": state_after,
          "default_after": qt_default(), "qt_signals": signals["window"], "qt_signals_outside": signals["outside"],
          "qt_defaults": qt_seen, "qt_flagged": flagged_seen,
          "windows_defaults": win_seen if probe_state["ok"] else None,
          "windows_other_roles_changed": other_roles["changed"],
          "windows_probe": probe_state["text"], "timeline": timeline}
    print("\n점검 끝. 아래 질문에 답해 주세요 (결과 파일에 함께 적습니다).", flush=True)
    dc["user_report"] = ask()
    dc["verdict"] = device_change_verdict(dc)
    return dc

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
        lines += [f"- 출력 장치 바꾸기 {dc['seconds']:.0f}초: 새 장치로 다시 연 횟수 {dc['switches']}, "
                  f"끝난 뒤 상태 {dc['state_after']}, 기본 장치 {dc['default_after']}",
                  f"- 판단: {dc.get('verdict', '')}",
                  f"- Qt 장치 목록 알림 점검 중 {dc.get('qt_signals', 0)}회(그 밖 {dc.get('qt_signals_outside', 0)}회) · "
                  f"Windows 직접 확인: {dc.get('windows_probe', '')}",
                  f"- Windows 기본 장치 변화: {dc.get('windows_defaults')} · Qt 기본 장치 변화: {dc.get('qt_defaults')} · "
                  f"Qt가 '기본'으로 표시한 장치: {dc.get('qt_flagged')}"]
        if dc.get("user_report"):
            lines.append("- 사용자 답: " + ", ".join(f"{k}={v}" for k, v in dc["user_report"].items()))
        if dc.get("timeline"):
            lines += ["", "### 출력 장치 바꾸기 기록 (초, 단계, 출처)", ""]
            lines += [f"- {e['t']:.1f}초 [{e.get('phase', '')}] {e['source']}: "
                      + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "phase", "source"))
                      for e in dc["timeline"][:80]]
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
