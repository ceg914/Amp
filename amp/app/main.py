"""애플리케이션 시작: 로깅, 전역 예외 처리, 메인 창 생성."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback

from PySide6.QtCore import QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from amp import APP_DISPLAY_NAME, APP_NAME, ORGANIZATION, __version__

log = logging.getLogger("amp")


def setup_logging() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    log_dir = os.path.join(base or os.path.expanduser("~/.sonatlas"), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "amp.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)
    return path


def install_excepthook(log_path: str) -> None:
    """UI 슬롯에서 난 예외가 조용히 사라지지 않도록 대화상자로 보여 준다."""

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("처리되지 않은 예외\n%s", details)
        app = QApplication.instance()
        if app is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, "오류", f"예상하지 못한 오류가 발생했습니다.\n{exc}\n\n로그: {log_path}")
            box.setDetailedText(details)
            box.exec()

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    QApplication.setOrganizationName(ORGANIZATION)
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationDisplayName(APP_DISPLAY_NAME)
    QApplication.setApplicationVersion(__version__)
    app = QApplication(argv)

    from amp.app.main_window import MainWindow
    from amp.app.theme import apply_theme

    log_path = setup_logging()
    install_excepthook(log_path)
    apply_theme(app)
    log.info("%s %s 시작", APP_NAME, __version__)

    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.check_recovery)
    if len(argv) > 1:
        path = argv[1]
        if path.lower().endswith((".ampproj", ".json")) or os.path.isdir(path):
            QTimer.singleShot(0, lambda: window.open_project(path))
        else:
            QTimer.singleShot(0, lambda: window.create_project_from_audio(path))
    return app.exec()
