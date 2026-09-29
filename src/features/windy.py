"""Windy 太陽光電預測的每日特徵。

- ``pv_14``：14:00 的機組平均預測發電量（日尖峰時段中段）
- ``pv_sum``：當日 3 小時格點的機組平均預測發電量加總

機組平均排除 ``settings.WINDY_EXCLUDED_UNITS``；某個時刻缺任一機組就不算平均
（避免組成改變造成跳階），缺值留給 LightGBM 的缺值處理。
"""

from __future__ import annotations

import polars as pl

from config import settings

FEATURES = ("pv_14", "pv_sum")


def daily_features(hourly: pl.DataFrame) -> pl.DataFrame:
    """由逐時段的機組預測彙整成每日兩個特徵。

    Args:
        hourly: :func:`src.data.windy.load_hourly` 的輸出。

    Returns:
        pl.DataFrame: ``date``、``pv_14``、``pv_sum``；一天的格點不齊時 ``pv_sum`` 為空。
    """
    units = sorted(set(hourly["unit_name"].unique()) - set(settings.WINDY_EXCLUDED_UNITS))
    slots = (
        hourly.filter(pl.col("unit_name").is_in(units)
                      & pl.col("forecast_time").dt.hour().cast(pl.Int32).is_in(settings.WINDY_HOURS))
        .group_by("forecast_time")
        .agg(pl.col("forecast_power").mean().alias("pv"), pl.len().alias("n_units"))
        .filter(pl.col("n_units") == len(units))
        .with_columns(pl.col("forecast_time").dt.date().alias("date"),
                      pl.col("forecast_time").dt.hour().cast(pl.Int32).alias("hour"))
    )
    return (
        slots.group_by("date")
        .agg(
            pl.col("pv").filter(pl.col("hour") == 14).first().alias("pv_14"),
            pl.when(pl.len() == len(settings.WINDY_HOURS)).then(pl.col("pv").sum())
            .otherwise(None).alias("pv_sum"),
        )
        .sort("date")
    )


def add_windy_features(daily: pl.DataFrame) -> pl.DataFrame:
    """把 Windy 每日特徵併進每日表（左合併，缺的日子為空值）。"""
    from src.data import windy

    return daily.join(daily_features(windy.load_hourly()), on="date", how="left")
