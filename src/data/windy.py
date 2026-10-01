"""Windy 太陽光電機組預測發電量的載入。

檔案：``paths.WINDY_FILE``，欄位 ``unit_name``、``forecast_time``、``forecast_power``，
每個機組每 3 小時一筆（少數日子為逐小時）。沒有發布時間欄，每個目標時刻只有一個版本
——與 Accuweather 相同，「只用作業時點前發布的版本」無法逐筆驗證。

這是預報：目標日的值在提交前就存在，所以**不套用資料截止日**（同 Accuweather）；
截止日只限制負載與觀測。
"""

from __future__ import annotations

import polars as pl

from config import paths
from src.data import checks
from src.logging_setup import get_logger

logger = get_logger(__name__)

COLUMNS = {"unit_name": pl.Utf8, "forecast_time": pl.Utf8, "forecast_power": pl.Utf8}

TIME_FORMATS = ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S")
"""``forecast_time`` 可接受的寫法。"""


def load_hourly() -> pl.DataFrame:
    """讀取 Windy 並驗證結構、去除完全重複的列。

    Returns:
        pl.DataFrame: 欄位 ``unit_name``、``forecast_time``（Datetime）、``forecast_power``（Float64）。

    Raises:
        ValueError: 欄位不符，或同一機組同一時刻出現不同數值。
    """
    raw = pl.read_csv(paths.WINDY_FILE, infer_schema_length=0)
    checks.check_columns(raw, COLUMNS, "Windy")
    # 時間有兩種寫法（2024/1/1 02:00 與 2024-01-01 02:00:00），先統一再去重，
    # 同一時刻的兩種寫法才會被視為同一筆。
    parsed = raw.with_columns(
        pl.coalesce(
            pl.col("forecast_time").str.to_datetime(fmt, strict=False)
            for fmt in TIME_FORMATS
        ),
        pl.col("forecast_power").cast(pl.Float64),
    )
    unparsed = raw.filter(parsed["forecast_time"].is_null())
    if unparsed.height:
        raise ValueError(f"Windy 有 {unparsed.height} 列時間無法解析，例如 {unparsed['forecast_time'][0]!r}")
    out = checks.deduplicate(parsed, ["unit_name", "forecast_time"], "Windy").sort("unit_name", "forecast_time")
    logger.info("讀取 Windy：%d 列，%s ~ %s，%d 個機組", out.height,
                out["forecast_time"].min(), out["forecast_time"].max(), out["unit_name"].n_unique())
    return out
