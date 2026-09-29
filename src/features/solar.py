"""天文特徵：年內序位、太陽仰角、日出日落時刻。

純天文計算（``astral``），無外部資料依賴。

**資料探索推翻了「`ramp_up` 源自表後光電的日落退場」這個假說**：它主要來自清晨商業啟動（平日 82.8% 落在 07:00–09:00，與日落的相關係數僅 −0.15）。

**但日落對 `t_night` 仍然重要**，且理由不同：日落決定照明負載的啟動時刻，
與 `ramp_up` 的成因無關。10 月初日落約 17:40–17:50，正落在夜尖峰窗口
（17:10–21:00）內；且每日約提前 1 分鐘，三天內移動不足 1 個資料格
——對 10/1–10/3 而言日落條件幾乎相同，這反而讓它成為穩定的錨點。

時刻目標佔總分約 93%，因此與時刻相關的天文特徵特別重要。
"""

from __future__ import annotations

import datetime as dt
import math

import polars as pl
from astral import LocationInfo
from astral.sun import elevation, sun

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def _location() -> LocationInfo:
    """本專案採用的地理位置（台灣地理中心）。"""
    return LocationInfo(
        settings.SITE_NAME,
        "Taiwan",
        settings.TIMEZONE,
        settings.SITE_LAT,
        settings.SITE_LON,
    )


def sun_times(date: dt.date) -> dict[str, int]:
    """計算指定日期的日出、日落、太陽正午時刻（當日分鐘數）。

    Args:
        date: 目標日期（台灣本地時間）。

    Returns:
        dict[str, int]: 鍵為 ``sunrise``、``sunset``、``noon``，值為當日分鐘數。
    """
    events = sun(_location().observer, date=date, tzinfo=settings.TIMEZONE)
    return {
        name: events[key].hour * 60 + events[key].minute
        for name, key in (("sunrise", "sunrise"), ("sunset", "sunset"), ("noon", "noon"))
    }


def noon_elevation(date: dt.date) -> float:
    """正午太陽仰角（度）。反映季節性的日射強度。"""
    location = _location()
    events = sun(location.observer, date=date, tzinfo=settings.TIMEZONE)
    return float(elevation(location.observer, events["noon"]))


def add_solar_features(daily: pl.DataFrame) -> pl.DataFrame:
    """加入天文特徵欄位。

    包含日出／日落／正午時刻（當日分鐘數）、日長、正午太陽仰角，
    以及年內序位的週期性編碼（sin / cos）——後者讓模型知道 12 月 31 日與
    1 月 1 日是相鄰的，而不是相隔 364。

    Args:
        daily: 含 ``date`` 的每日表。

    Returns:
        pl.DataFrame: 加上天文特徵。
    """
    dates = daily["date"].to_list()
    times = [sun_times(d) for d in dates]

    out = daily.with_columns(
        pl.Series("sunrise_min", [t["sunrise"] for t in times], dtype=pl.Int32),
        pl.Series("sunset_min", [t["sunset"] for t in times], dtype=pl.Int32),
        pl.Series("solar_noon_min", [t["noon"] for t in times], dtype=pl.Int32),
        pl.Series("noon_elevation_deg", [noon_elevation(d) for d in dates]),
    ).with_columns(
        (pl.col("sunset_min") - pl.col("sunrise_min")).alias("day_length_min"),
    )

    # 年內序位的週期性編碼。以 365.25 為週期，閏年不需特判。
    angle = 2 * math.pi * pl.col("date").dt.ordinal_day() / 365.25
    out = out.with_columns(
        angle.sin().alias("doy_sin"),
        angle.cos().alias("doy_cos"),
    )

    # 日落相對於夜尖峰窗口的位置：負值代表日落在窗口之前。
    night_start = (
        int(settings.NIGHT_PEAK_START.split(":")[0]) * 60
        + int(settings.NIGHT_PEAK_START.split(":")[1])
    )
    return out.with_columns(
        (pl.col("sunset_min") - night_start).alias("sunset_minus_night_start")
    )


