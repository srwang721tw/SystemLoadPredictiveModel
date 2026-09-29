"""主資料讀取。

讀入 ``Date_Time, Load_MW`` 的 10 分鐘序列，並補齊缺漏時間戳為 null。

不可 dropna：缺列會使相鄰差分錯位，直接污染 ramp 標籤。
本模組一律以「補列為 null」處理，缺失值交給 ``imputer`` 補。

原始檔的三項格式細節（皆登記於 ``config/settings.py``）：
    - UTF-8 BOM（polars 會自動處理，欄名不受影響）
    - CRLF 換行
    - 缺失值有**兩種**字面值：空字串與 ``ERROR:``。後者若未列入 null values，
      ``Load_MW`` 整欄會被讀成字串型別。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from config import paths, settings
from src.data import checks
from src.logging_setup import get_logger

logger = get_logger(__name__)


def load_raw_load(path: Path | None = None) -> pl.DataFrame:
    """讀取原始負載 CSV。

    時間格式為 ``2024/1/1 00:00``（月、日不補零）。

    Args:
        path: 資料檔路徑，None 時採用 ``paths.LOAD_DATA_FILE``。

    Returns:
        pl.DataFrame: 欄位 ``ts`` (Datetime)、``Load_MW`` (Float64)，依時間排序。

    Raises:
        FileNotFoundError: 檔案不存在。
    """
    path = path or paths.LOAD_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"主資料檔不存在：{path}")

    raw = pl.read_csv(
        path,
        null_values=list(settings.RAW_NULL_VALUES),
        schema_overrides={"Load_MW": pl.Float64},
    )
    df = (
        raw.with_columns(
            pl.col("Date_Time").str.to_datetime(settings.RAW_DATETIME_FORMAT).alias("ts")
        )
        .select("ts", "Load_MW")
        .sort("ts")
    )
    # 讀檔即驗證結構（欄位、型別、時間頻率、連續性），不符即中止。
    checks.check_columns(df, {"ts": pl.Datetime, "Load_MW": pl.Float64}, "負載")
    df = checks.deduplicate(df, ["ts"], "負載")
    checks.check_step(df, "ts", settings.DATA_FREQ_MIN, "負載")
    checks.check_gaps(df, "ts", dt.timedelta(minutes=settings.DATA_FREQ_MIN), "負載")
    df = checks.apply_cutoff(df, "ts", "負載")

    size_mb = path.stat().st_size / 1024**2
    logger.info(
        "讀取主資料：%s（%.1f MB）→ %d 列，%s ~ %s，Load_MW 缺失 %d 點",
        path.name,
        size_mb,
        df.height,
        df["ts"].min(),
        df["ts"].max(),
        df["Load_MW"].null_count(),
    )
    return df


def reindex_full_grid(df: pl.DataFrame) -> pl.DataFrame:
    """將序列對齊到完整的 10 分鐘格點，缺漏時間戳補為 null。

    格點自資料首日 00:00 到末日 23:50，確保每日恰為
    ``settings.POINTS_PER_DAY`` 筆。

    Args:
        df: :func:`load_raw_load` 的輸出。

    Returns:
        pl.DataFrame: 欄位 ``ts``、``Load_MW``、``is_missing``。
            ``is_missing`` 為真表示該點在原始資料中沒有值
            （時間戳缺漏，或有時間戳但值為空／``ERROR:``）。
    """
    first_day = df["ts"].min().date()  # type: ignore[union-attr]
    last_day = df["ts"].max().date()  # type: ignore[union-attr]

    grid = pl.datetime_range(
        start=pl.datetime(first_day.year, first_day.month, first_day.day),
        end=pl.datetime(last_day.year, last_day.month, last_day.day, 23, 50),
        interval=f"{settings.DATA_FREQ_MIN}m",
        eager=True,
    ).alias("ts")

    full = (
        pl.DataFrame({"ts": grid})
        .join(df, on="ts", how="left")
        .with_columns(pl.col("Load_MW").is_null().alias("is_missing"))
        .sort("ts")
    )

    n_added = full.height - df.height
    n_days = (last_day - first_day).days + 1
    logger.info(
        "對齊格點：%d 天 × %d = %d 列（補入缺漏時間戳 %d 列），缺失合計 %d 點",
        n_days,
        settings.POINTS_PER_DAY,
        full.height,
        n_added,
        full["is_missing"].sum(),
    )
    if full.height != n_days * settings.POINTS_PER_DAY:
        raise ValueError(
            f"格點數不符：預期 {n_days * settings.POINTS_PER_DAY}，實得 {full.height}"
        )
    return full
