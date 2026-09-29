"""基線：近 4 週同星期日的中位數。

作為時刻與量值預測的起點：量值模型的基準不可靠時沿用它的量值，
時刻預測則在它的輸出上覆寫 ``t_day``／``t_night``。
中位數可能落在格點之間（例如 17:50 與 18:00 的中位數是 17:55），須對齊到合法格點。
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.features.targets import TARGET_NAMES, day_peak_grid, night_peak_grid


def _snap_to_grid(minutes: float, grid: list[int]) -> int:
    """把任意分鐘數對齊到最近的合法格點，並列時取較早者。"""
    return min(grid, key=lambda g: (abs(g - minutes), g))


def _snap_timing(row: dict[str, float]) -> dict[str, float]:
    """對齊 ``t_day``／``t_night`` 至各自的合法格點。"""
    row["t_day"] = float(_snap_to_grid(row["t_day"], day_peak_grid()))
    row["t_night"] = float(_snap_to_grid(row["t_night"], night_peak_grid()))
    return row


def _recent_median(history: pl.DataFrame, n: int, weekday: int) -> dict[str, float]:
    """取最近 n 個指定星期的各目標中位數。"""
    subset = history.filter(pl.col("date").dt.weekday() == weekday).sort("date").tail(n)
    return {name: float(subset[name].median()) for name in TARGET_NAMES}


def baseline1_same_weekday_median(
    history: pl.DataFrame, target_dates: tuple[date, ...], n_weeks: int = 4
) -> pl.DataFrame:
    """近 ``n_weeks`` 個同星期日的中位數。

    Args:
        history: 截止於預測起點日的每日目標值。
        target_dates: 待預測日期。
        n_weeks: 回溯幾個同星期日。

    Returns:
        pl.DataFrame: 每個目標日一列的 6 項預測值。
    """
    rows = [_snap_timing(_recent_median(history, n_weeks, d.isoweekday())) for d in target_dates]
    return pl.DataFrame(rows).select(TARGET_NAMES)
