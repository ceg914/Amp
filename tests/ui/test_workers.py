from amp.app.workers import TaskRunner
from amp.core.errors import AnalysisCancelled, AudioLoadError


def test_finished(qtbot):
    runner = TaskRunner()

    def job(progress, token):
        progress(0.5, "half")
        return 42

    h = runner.submit("ok", job)
    with qtbot.waitSignal(h.finished, timeout=5000) as blocker:
        pass
    assert blocker.args == [42]
    qtbot.waitUntil(lambda: runner.active_count == 0)


def test_failed_with_user_message(qtbot):
    runner = TaskRunner()

    def job(progress, token):
        raise AudioLoadError("못 읽음")

    h = runner.submit("bad", job)
    with qtbot.waitSignal(h.failed, timeout=5000) as blocker:
        pass
    assert blocker.args[0] == "못 읽음"


def test_unexpected_error(qtbot):
    runner = TaskRunner()
    h = runner.submit("crash", lambda p, t: 1 / 0)
    with qtbot.waitSignal(h.failed, timeout=5000) as blocker:
        pass
    assert "예상하지 못한" in blocker.args[0] and "ZeroDivisionError" in blocker.args[1]


def test_missing_package_tells_how_to_install(qtbot):
    """사용자 PC: 'No module named mido'가 '예상하지 못한 오류'로 나왔다. 패키지 이름과 설치 방법을 알린다."""
    runner = TaskRunner()

    def job(progress, token):
        import importlib

        return importlib.import_module("sonatlas_missing_pkg_for_test")

    h = runner.submit("export", job)
    with qtbot.waitSignal(h.failed, timeout=5000) as blocker:
        pass
    msg = blocker.args[0]
    assert "sonatlas_missing_pkg_for_test" in msg and "run.bat" in msg and "예상하지 못한" not in msg


def test_cancel(qtbot):
    runner = TaskRunner()

    def job(progress, token):
        while True:
            if token.cancelled:
                raise AnalysisCancelled()

    h = runner.submit("loop", job)
    with qtbot.waitSignal(h.cancelled, timeout=5000):
        h.cancel()
    assert runner.wait(2000)


def test_fast_failure_is_not_lost(qtbot):
    """작업이 즉시 끝나도 submit 직후 연결한 시그널은 받아야 한다."""
    runner = TaskRunner()
    got = []

    def job(progress, token):
        raise AudioLoadError("즉시 실패")

    h = runner.submit("instant", job)
    import time

    time.sleep(0.05)  # 연결 전에 작업 스레드가 끝날 시간을 준다
    h.failed.connect(lambda msg, tb: got.append(msg))
    qtbot.waitUntil(lambda: got == ["즉시 실패"], timeout=5000)
