"""Accuweather 鄉鎮氣象預報的載入。

本模組把 Accuweather 的逐小時鄉鎮預報，轉成**與 CODiS 小時觀測完全相同的
schema**，好讓 :func:`src.features.weather.daily_features` 原樣重用。

---

## 為什麼要刻意輸出 CODiS 的 schema

honest 模式以 Accuweather 的 ``w_*`` 欄逐欄覆寫目標列
（``pipeline._override_weather``），欄名必須與 :func:`src.data.external.load_weather`
完全一致。與其另寫一套「Accuweather 版的日彙總」再去對欄名（兩份規則
遲早會漂移），不如讓本模組吐出 ``Date`` / ``stn_ID`` / ``AirTemperature`` /
``UVIndex`` 四欄，後面沿用同一條管線——**欄位契約於是在構造上成立，
而不是靠測試去追。**

因此 :data:`config.settings.ACCUWEATHER_STATIONS` 的鍵是 CODiS 測站代號。

---

## 三個實測到的陷阱

1. **約 1 GB**（三個年度檔）。一律 ``pl.scan_csv`` + 欄位投影 + ``location_key``
   過濾；實測只取五個鄉鎮的四欄，掃完三檔約 **2.6 秒**。整檔 ``read_csv`` 會
   把記憶體吃光。
2. **三個年度檔的型別推論不一致**，直接 ``pl.concat`` 會拋
   ``SchemaError: type Float64 is incompatible with expected type Int64``。
   故 ``schema_overrides`` **必須釘死**。
3. **對照表的鄉鎮名會重複**：「北區」在新竹市／臺中市／臺南市各有一個，
   「中正區」在基隆市／臺北市各有一個。**必須用 county + township 兩欄配對**，
   只比對鄉鎮名會靜默對到錯誤縣市。

---

## 與 CODiS 的兩個語意差異（已量測，不可當成等價）

**時間戳語意**：CODiS 的 ``DataTime`` 是**區間終點**（hour 7 代表 06:00–07:00），
Accuweather 的 ``forecast_time`` 是**瞬時值**。兩者的日窗因此差最多一小時。
實測 ``day_tmax`` 在 6–17 與 7–18 兩種慣例下差異 **< 0.01°C**（12 小時窗取
max 對一小時平移不敏感），故沿用 ``settings.WEATHER_DAY_HOURS``。
但 ``day_tmin`` / ``night_*`` 是**邊緣敏感**的，切換來源時必須另外比對。

**UV 的定義不同**：CODiS 取 ``UVIndex.Accumulation``，Accuweather 是
``max_uv_index``（預報的最大 UV 指數）。兩者都叫「UV 指數」但不是同一個量。
``max_uv_index`` 夜間為 null（占 49%），日彙總取 max 會自動略過。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from config import paths, settings
from src.data import checks
from src.logging_setup import get_logger

logger = get_logger(__name__)

CODIS_COLUMNS = ("Date", "stn_ID", "AirTemperature", "UVIndex")
"""輸出欄位，與 CODiS 小時表一致（見模組 docstring）。"""

MAP_COLUMNS = ("county", "township", "location_key")
"""對照表的必要欄位。"""

_SOURCE_COLUMNS = ("location_key", "forecast_time", "temperature", "max_uv_index")
"""從原始檔投影出來的欄位——只取這四欄，其餘不讀進記憶體。"""

_SCHEMA: dict[str, type[pl.DataType]] = {
    "location_key": pl.Utf8,
    "forecast_time": pl.Utf8,
    "temperature": pl.Float64,
    "max_uv_index": pl.Float64,
}
"""釘死型別。三個年度檔的推論結果不一致，不釘會在 concat 時拋 SchemaError。"""


def yearly_files(directory: Path | None = None) -> list[Path]:
    """列出年度預報檔（已排序）。

    Args:
        directory: 資料目錄，None 時採用 ``paths.ACCUWEATHER_DIR``。

    Returns:
        list[Path]: 年度檔路徑。

    Raises:
        FileNotFoundError: 目錄不存在或其中沒有年度檔。
    """
    directory = directory or paths.ACCUWEATHER_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Accuweather 資料目錄不存在：{directory}")
    found = sorted(directory.glob(paths.ACCUWEATHER_FORECAST_PATTERN))
    if not found:
        raise FileNotFoundError(f"{directory} 中找不到「天氣預測詳細表」年度檔")
    return found


def load_station_hourly(
    directory: Path | None = None, stations: dict[str, str] | None = None
) -> pl.DataFrame:
    """載入指定測站對應鄉鎮的逐小時預報，輸出 CODiS 格式。

    Args:
        directory: 年度檔目錄，None 時採用 ``paths.ACCUWEATHER_DIR``。
        stations: ``{CODiS 測站代號: location_key}``，
            None 時採用 ``settings.ACCUWEATHER_STATIONS``。

    Returns:
        pl.DataFrame: 欄位同 :data:`CODIS_COLUMNS`，依 ``stn_ID`` / ``Date`` 排序。

    Raises:
        ValueError: 解析後沒有任何資料列（通常是 ``location_key`` 對錯了）。
    """
    stations = stations or settings.ACCUWEATHER_STATIONS
    key_to_station = {key: station for station, key in stations.items()}

    frame = (
        pl.scan_csv(
            yearly_files(directory),
            encoding="utf8-lossy",
            schema_overrides=_SCHEMA,
            infer_schema_length=0,
        )
        .select(_SOURCE_COLUMNS)
        .filter(pl.col("location_key").is_in(list(key_to_station)))
        .with_columns(
            pl.col("forecast_time")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .alias("Date"),
            pl.col("location_key")
            .replace_strict(key_to_station, return_dtype=pl.Utf8)
            .alias("stn_ID"),
            pl.col("temperature").alias("AirTemperature"),
            pl.col("max_uv_index").alias("UVIndex"),
        )
        .select(CODIS_COLUMNS)
        .collect()
    )

    if not frame.height:
        raise ValueError(
            "Accuweather 解析後沒有任何資料列——請檢查 "
            "settings.ACCUWEATHER_STATIONS 的 location_key 是否正確"
        )

    unparsed = frame["Date"].null_count()
    if unparsed:
        raise ValueError(f"Accuweather 有 {unparsed} 筆 forecast_time 無法解析")
    frame = checks.deduplicate(frame, ["Date", "stn_ID"], "Accuweather").sort("stn_ID", "Date")
    checks.check_step(frame, "Date", 60, "Accuweather")
    checks.check_gaps(frame, "Date", dt.timedelta(hours=settings.EXOGENOUS_MAX_GAP_HOURS),
                      "Accuweather", group="stn_ID")
    logger.info(
        "讀取 Accuweather 預報：%d 列，%s ~ %s，%d 個測站",
        frame.height, frame["Date"].min(), frame["Date"].max(),
        frame["stn_ID"].n_unique(),
    )
    return frame


def reindex_full_hours(hourly: pl.DataFrame) -> pl.DataFrame:
    """把每站補滿到連續的整點格，缺的時點留 ``null``。

    **為什麼需要這一步**：Accuweather 有三段各 2 天的整日缺漏
    （2024-02-24~25、2024-06-09~10、2025-05-03~04）外加 2 個部分缺日。
    整日缺漏時該日**連一列都沒有**，``interpolate_hourly`` 是在既有列之間
    插補的，看不見「不存在的列」，於是 ``add_weather_features`` 會在下游
    以「這些日期缺氣象資料」拋錯。

    補成連續格點之後，缺口變成**顯式的 null**，就能交給既有的
    ``interpolate_hourly`` 處理——而且 48 小時的缺口遠超
    ``settings.WEATHER_LONG_GAP_HOURS``（3 小時），它既有的長缺口警告會
    自動響起。**這不是靜默補值：缺口會被插補，但會被記錄下來。**

    Args:
        hourly: :func:`load_station_hourly` 的輸出。

    Returns:
        pl.DataFrame: 欄位同輸入，每站的時點連續無斷層。
    """
    if not hourly.height:
        return hourly

    grid = pl.datetime_range(
        hourly["Date"].min(), hourly["Date"].max(), interval="1h", eager=True
    ).alias("Date")
    stations = hourly["stn_ID"].unique().sort()
    skeleton = pl.DataFrame({"Date": grid}).join(
        pl.DataFrame({"stn_ID": stations}), how="cross"
    )

    out = (
        skeleton.join(hourly, on=["Date", "stn_ID"], how="left")
        .select(CODIS_COLUMNS)
        .sort("stn_ID", "Date")
    )
    added = out.height - hourly.height
    if added:
        logger.warning(
            "Accuweather 補上 %d 個缺漏時點（%d 站 × 整點格），"
            "其值交由 interpolate_hourly 插補——長缺口會由它再警告一次",
            added, len(stations),
        )
    return out


def coverage_report(hourly: pl.DataFrame) -> pl.DataFrame:
    """逐站回報覆蓋天數與缺值，供 log 與測試比對。

    Args:
        hourly: :func:`load_station_hourly` 的輸出。

    Returns:
        pl.DataFrame: 每站一列，含天數、列數與兩個量的缺值數。
    """
    return (
        hourly.with_columns(pl.col("Date").dt.date().alias("date"))
        .group_by("stn_ID")
        .agg(
            pl.col("date").n_unique().alias("天數"),
            pl.len().alias("列數"),
            pl.col("AirTemperature").null_count().alias("氣溫缺值"),
            pl.col("UVIndex").null_count().alias("UV缺值"),
        )
        .sort("stn_ID")
    )
