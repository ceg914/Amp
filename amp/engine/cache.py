"""분석 중간 결과 캐시.

같은 오디오 파일을 다시 분석할 때 보컬 분리·피치 추출을 반복하지 않도록 결과를 디스크에 둔다.

    <root>/<오디오 내용 해시>/<단계>-<파라미터 해시>/
        done.json      완료 표시(파라미터·생성 시각). 이 파일이 있어야 유효한 항목이다.
        ...            단계별 결과 파일

쓰기는 같은 폴더 안의 임시 폴더에서 한 뒤 이름을 바꿔 확정한다.
실패하거나 취소되면 임시 폴더를 지우므로 반쯤 쓴 결과가 남지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

DONE_FILE = "done.json"
_TMP_PREFIX = ".tmp-"


def file_content_key(path: str, chunk: int = 1 << 20) -> str:
    """파일 내용 전체의 SHA-256 (앞 32자리). 이름·경로가 달라도 내용이 같으면 같은 키다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:32]


def params_key(params: dict) -> str:
    text = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


class AnalysisCache:
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    def entry_dir(self, audio_key: str, stage: str, params: dict) -> Path:
        return self.root / audio_key / f"{stage}-{params_key(params)}"

    def lookup(self, audio_key: str, stage: str, params: dict) -> Path | None:
        d = self.entry_dir(audio_key, stage, params)
        return d if (d / DONE_FILE).is_file() else None

    @contextmanager
    def writer(self, audio_key: str, stage: str, params: dict, info: dict | None = None) -> Iterator[Path]:
        """임시 폴더를 넘겨주고, 블록이 정상 종료되면 최종 위치로 옮긴다."""
        final = self.entry_dir(audio_key, stage, params)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f"{_TMP_PREFIX}{stage}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            yield tmp
            meta = {
                "stage": stage,
                "params": params,
                "info": info or {},
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            (tmp / DONE_FILE).write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
            if final.exists():
                shutil.rmtree(final)
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def remove(self, audio_key: str, stage: str, params: dict) -> None:
        """손상된 항목을 지운다 (다음에 다시 계산)."""
        d = self.entry_dir(audio_key, stage, params)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    def size_bytes(self) -> int:
        total = 0
        if not self.root.exists():
            return 0
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def remove_stale_temp(self, older_than_s: float = 3600.0) -> int:
        """비정상 종료로 남은 임시 폴더를 지운다. 지운 개수를 돌려준다."""
        removed = 0
        if not self.root.exists():
            return 0
        now = time.time()
        for audio_dir in self.root.iterdir():
            if not audio_dir.is_dir():
                continue
            for d in audio_dir.iterdir():
                if d.name.startswith(_TMP_PREFIX) and now - d.stat().st_mtime > older_than_s:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        return removed
