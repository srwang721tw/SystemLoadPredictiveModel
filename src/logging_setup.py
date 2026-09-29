"""Logging 設定。

同時輸出至 console 與 ``logs/``，含 timestamp 與 log level，
檔案 handler 明確指定 UTF-8 編碼。

讀檔、補值、標籤、回測、訓練、產檔都須落檔記錄，故需要一個共用的初始化入口；
放在 ``main.py`` 會使 notebook 與 pytest 無法重用。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from config import paths, settings


def setup_logging(
    stage: str,
    level: str | None = None,
    to_file: bool = True,
) -> Path | None:
    """初始化 root logger，同時輸出至 console 與檔案。

    重複呼叫時會先清除既有 handler，避免在同一個 process（例如 notebook）中
    累積重複輸出。

    Args:
        stage: 流程名稱，作為 log 檔名的一部分，例如 ``"run_submission"``。
        level: log 等級，None 時採用 ``settings.LOG_LEVEL``。
        to_file: 是否落檔。單元測試中可設為 False。

    Returns:
        Path | None: log 檔路徑；``to_file`` 為 False 時回傳 None。
    """
    root = logging.getLogger()
    root.setLevel(level or settings.LOG_LEVEL)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if not to_file:
        return None

    paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = paths.LOGS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{stage}.log"
    file_handler = logging.FileHandler(log_path, encoding=settings.FILE_ENCODING)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """取得模組層級 logger。

    Args:
        name: 慣例上傳入 ``__name__``。

    Returns:
        logging.Logger: 已掛在 root logger 之下的 logger。
    """
    return logging.getLogger(name)
