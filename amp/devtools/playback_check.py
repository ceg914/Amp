"""재생 안정성 점검 도구 (STEP 3.5). 사용자 PC(Windows)에서 실행해 Qt 오디오 경고가 다시 나는지 본다.

    python scripts/playback_check.py                  # 합성 음 20초로 시나리오 3회
    python scripts/playback_check.py --audio 곡.mp3 --rounds 5
    python scripts/playback_check.py --device-change 20   # 20초 동안 직접 출력 장치를 바꿔 본다
    python scripts/playback_check.py --rounds 0 --device-change 40   # 장치 바꾸기만 40초

장치 바꾸기 때는 Windows에 기본 출력 장치를 직접 묻고(Qt를 거치지 않음), Qt 알림·재생기의 다시 열기와 함께 시간순으로 적는다.
그래서 장치를 안 바꿨는지, Windows는 바뀌었는데 Qt가 알리지 않았는지, 앱이 알림을 받고도 다시 열지 않았는지 가를 수 있다.

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
        probe=None) -> tuple[dict, Path]:
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
        out["device_change"] = _device_change(player, audio, device_change_s, spin, rec, probe)
    player.shutdown()
    rec.step = "종료"
    spin(0.5)
    dw = rec.device_warnings()
    out["summary"] = (f"재생 장치 경고 {len(dw)}개, 그 밖의 Qt 경고 {len(rec.warnings()) - len(dw)}개, "
                      f"재생 오류 알림 {len(errors)}개 (시나리오 {rounds}회, 단계 {len(steps)}개)")
    return _write(out, rec, errors, steps, out_root)


def windows_default_output_id() -> str | None:
    """Windows가 지금 기본 출력(멀티미디어 역할)으로 잡은 장치 id를 Qt를 거치지 않고 직접 묻는다.

    Qt의 장치 목록은 Windows 알림으로만 갱신되므로, 알림이 오지 않았는지 사용자가 장치를 안 바꿨는지 가르려면 따로 물어야 한다.
    Qt의 QAudioDevice.id()도 같은 끝점 id 문자열이다. Windows가 아니면 None, 실패하면 예외.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

    def guid(text: str) -> GUID:
        g = GUID()
        ctypes.memmove(g.data, uuid.UUID(text).bytes_le, 16)
        return g

    def method(obj: ctypes.c_void_p, index: int, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])

    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ctypes.c_void_p, wintypes.DWORD,
                                       ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    ole32.CoTaskMemFree.restype = None
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoInitializeEx(None, 0x2)  # 이미 초기화돼 있으면 S_FALSE나 RPC_E_CHANGED_MODE인데, 어느 쪽이든 COM은 쓸 수 있다
    enum = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(ctypes.byref(guid("BCDE0395-E52F-467C-8E3D-C4579291692E")), None, 0x17,  # MMDeviceEnumerator
                                ctypes.byref(guid("A95664D2-9614-4F35-A746-DE8DB63617E6")), ctypes.byref(enum))
    if hr < 0 or not enum.value:
        raise OSError(f"MMDeviceEnumerator 생성 실패 (0x{hr & 0xFFFFFFFF:08X})")
    try:
        dev = ctypes.c_void_p()
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender=0, eMultimedia=1) — Qt도 eMultimedia를 쓴다
        hr = method(enum, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(enum, 0, 1, ctypes.byref(dev))
        if hr < 0 or not dev.value:
            return ""  # 기본 출력 없음
        try:
            pid = ctypes.c_wchar_p()
            hr = method(dev, 5, ctypes.POINTER(ctypes.c_wchar_p))(dev, ctypes.byref(pid))  # IMMDevice::GetId
            if hr < 0:
                raise OSError(f"IMMDevice::GetId 실패 (0x{hr & 0xFFFFFFFF:08X})")
            try:
                return pid.value or ""
            finally:
                ole32.CoTaskMemFree(pid)
        finally:
            method(dev, 2)(dev)  # Release
    finally:
        method(enum, 2)(enum)


def device_change_verdict(dc: dict) -> str:
    """장치 바꾸기 시간 동안 모은 기록으로 무엇이 일어났는지 한 줄로 정리한다."""
    win = dc.get("windows_defaults")  # Windows에 직접 물어 본 기본 장치 (차례대로, 바뀐 것만). None = 물어보지 못함
    qt = dc.get("qt_defaults") or []
    signals, switches = dc.get("qt_signals", 0), dc.get("switches", 0)
    if switches:
        return f"새 기본 장치로 재생 출력을 다시 열었습니다 ({switches}회)."
    if len(qt) > 1:
        return "Qt는 기본 장치가 바뀐 것을 알았는데 재생 출력을 다시 열지 않았습니다 (앱 쪽 문제)."
    if win is not None and len(win) > 1:
        if not signals:
            return "Windows의 기본 장치는 바뀌었는데 Qt가 장치 변경 알림을 보내지 않았습니다 (Qt 쪽 문제)."
        return "Windows의 기본 장치는 바뀌었고 Qt 알림도 왔지만, Qt가 본 기본 장치는 그대로였습니다 (Qt 쪽 문제)."
    if win is not None:
        tail = f" Qt 장치 목록 알림은 {signals}회 있었습니다." if signals else ""
        return "이 시간 동안 Windows의 기본 출력 장치가 바뀌지 않았습니다 (장치를 바꾸지 않았거나 시간 밖에서 바꿈)." + tail
    if signals:
        return f"Qt 장치 목록 알림이 {signals}회 왔지만 기본 장치는 그대로였습니다. Windows에 직접 묻지는 못했습니다."
    return "Qt 알림이 없었고 Windows에 직접 묻지도 못해, 장치가 실제로 바뀌었는지 알 수 없습니다."


def _device_change(player, audio, seconds: float, spin, rec: Recorder, probe=None) -> dict:
    """재생을 이어 가며 사용자가 출력 장치를 바꾸게 하고, Windows·Qt·재생기가 각각 무엇을 봤는지 시간순으로 적는다.

    probe: 운영체제에 직접 물어 기본 출력 장치 id를 돌려주는 함수 (기본은 Windows 조회, 지원하지 않으면 None).
    """
    probe = probe or windows_default_output_id
    from PySide6.QtMultimedia import QMediaDevices

    from amp.app.player import PlaybackState

    def qt_default() -> str | None:
        d = QMediaDevices.defaultAudioOutput()
        return None if d.isNull() else d.description()

    def describe(dev_id: str | None) -> str | None:
        if not dev_id:
            return dev_id
        for d in QMediaDevices.audioOutputs():
            if bytes(d.id()).decode("utf-8", "replace") == dev_id:
                return d.description()
        return dev_id  # Qt 목록에 없는 장치면 id 그대로

    probe_state = {"ok": True, "text": "확인함"}

    def win_default() -> str | None:
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
        return describe(value)

    timeline: list[dict] = []
    t0 = [time.monotonic()]

    def note(source: str, **kw) -> None:
        if len(timeline) < 300:
            timeline.append({"t": round(time.monotonic() - t0[0], 2), "source": source, **kw})

    watcher = QMediaDevices()
    state = {"signals": 0}

    def on_signal() -> None:
        state["signals"] += 1
        note("qt_signal", qt_default=qt_default(), outputs=len(QMediaDevices.audioOutputs()))

    watcher.audioOutputsChanged.connect(on_signal)
    print(f"\n출력 장치 바꾸기 점검 ({seconds:.0f}초): Windows 소리 설정을 먼저 열어 두세요. "
          "시작하면 출력 장치를 다른 것으로 바꿨다가 되돌리세요 "
          "(USB 사운드카드의 헤드폰 단자는 꽂아도 Windows 장치가 바뀌지 않을 수 있음).", flush=True)
    if sys.stdin is not None and sys.stdin.isatty():
        input("준비되면 Enter를 누르세요. ")
    player.set_audio(audio)
    player.play()
    t0[0] = time.monotonic()
    rec.step = "출력 장치 바꾸기"
    before = player.device_switches
    win_seen = [win_default()]
    qt_seen = [qt_default()]
    note("start", windows_default=win_seen[0], qt_default=qt_seen[0], state=player.state.value)
    print("지금 바꾸세요.", flush=True)
    last_print = last_poll = 0.0
    switches_seen = 0
    while (elapsed := time.monotonic() - t0[0]) < seconds:
        spin(0.1)
        if player.state is PlaybackState.STOPPED:
            player.play()
        if elapsed - last_poll >= 0.5:
            last_poll = elapsed
            w, q = win_default(), qt_default()
            if probe_state["ok"] and w != win_seen[-1]:
                win_seen.append(w)
                note("windows", default=w)
            if q != qt_seen[-1]:
                qt_seen.append(q)
                note("qt_default", default=q)
            if player.device_switches - before != switches_seen:
                switches_seen = player.device_switches - before
                note("player", reopened=switches_seen, state=player.state.value)
        if elapsed - last_print >= 5.0:
            last_print = elapsed
            shown = win_seen[-1] if probe_state["ok"] else qt_seen[-1]
            print(f"  남은 시간 {seconds - elapsed:4.0f}초 · 기본 장치: {shown} · 다시 연 횟수 {player.device_switches - before}",
                  flush=True)
    watcher.audioOutputsChanged.disconnect(on_signal)
    dc = {"seconds": seconds, "switches": player.device_switches - before, "state_after": player.state.value,
          "default_after": qt_default(), "qt_signals": state["signals"], "qt_defaults": qt_seen,
          "windows_defaults": win_seen if probe_state["ok"] else None,
          "windows_probe": probe_state["text"],
          "timeline": timeline}
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
                  f"- Qt 장치 목록 알림 {dc.get('qt_signals', 0)}회 · Windows 직접 확인: {dc.get('windows_probe', '')}"]
        if dc.get("timeline"):
            lines += ["", "### 출력 장치 바꾸기 기록 (초, 출처)", ""]
            lines += [f"- {e['t']:.1f}초 {e['source']}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "source"))
                      for e in dc["timeline"][:60]]
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
