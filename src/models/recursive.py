"""遞迴預測的對照組：10 分鐘一步的線性自迴歸，逐步外推 432 步。

比較「直接預測 vs 遞迴預測」。現行 daily-row 架構已是依 horizon 分開建模的
**直接預測**（每個 horizon 各自一個特徵矩陣與模型）；這裡的「遞迴」是
「曲線逐步外推」：先預測下一個 10 分鐘，再把預測值當成已知，一路外推到 D+2 23:50。

模型刻意簡單（最小平方法的線性 AR），它是**對照組**，不是候選：

- 自迴歸項：落後 ``RECURSIVE_LAGS`` 期（1、2、3、6 期與前一天、前一週同時刻）
- 日內時段：144 個時段虛擬變數（取代截距）
- 日別：電價日別的虛擬變數（日曆類，目標日可用未來值）

每窗只用起點（含）以前 ``RECURSIVE_FIT_DAYS`` 天的 10 分鐘資料擬合；
外推時落後項一旦越過起點就改用自己的預測值，絕不讀取起點之後的實際值。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from config import settings
from src.features.targets import TARGET_NAMES, compute_targets

STEPS_PER_DAY = 144


def _design(load: np.ndarray, index: int, slot: int, daytype: str, daytypes: list[str]) -> np.ndarray:
    lags = [load[index - lag] for lag in settings.RECURSIVE_LAGS]
    slots = np.zeros(STEPS_PER_DAY)
    slots[slot] = 1.0
    kinds = np.array([1.0 if daytype == d else 0.0 for d in daytypes[1:]])
    return np.concatenate([lags, slots, kinds])


def forecast_curve(clean: pl.DataFrame, origin: dt.date, target_dates: list[dt.date]) -> np.ndarray:
    """以起點以前的資料擬合 AR，遞迴外推 ``target_dates`` 的 10 分鐘曲線。

    Args:
        clean: 10 分鐘序列，含 ``ts``、``Load_MW``、``date``、``price_daytype``（目標日也要有日別）。
        origin: 預測起點日。
        target_dates: 連續的目標日。

    Returns:
        np.ndarray: 長度 ``144 × len(target_dates)`` 的預測曲線。
    """
    start = origin - dt.timedelta(days=settings.RECURSIVE_FIT_DAYS)
    history = clean.filter((pl.col("date") > start) & (pl.col("date") <= origin)).sort("ts")
    daytype_of = dict(clean.select("date", "price_daytype").unique().iter_rows())
    daytypes = sorted({v for v in daytype_of.values() if v is not None})

    load = history["Load_MW"].to_numpy().astype(float)
    days = history["date"].to_list()
    max_lag = max(settings.RECURSIVE_LAGS)
    rows = [_design(load, i, i % STEPS_PER_DAY, daytype_of[days[i]], daytypes)
            for i in range(max_lag, len(load))]
    coef, *_ = np.linalg.lstsq(np.array(rows), load[max_lag:], rcond=None)

    extended = list(load)
    for day in target_dates:
        for slot in range(STEPS_PER_DAY):
            x = _design(np.asarray(extended), len(extended), slot, daytype_of[day], daytypes)
            extended.append(float(x @ coef))
    return np.asarray(extended[len(load):])


def make_recursive_predictor(clean: pl.DataFrame):
    """包成回測用的預測函式：曲線外推 → ``compute_targets`` 推導 6 個目標。"""

    def predict(history: pl.DataFrame, target_dates: tuple[dt.date, ...]) -> pl.DataFrame:
        origin = history["date"].max()
        days = sorted(target_dates)
        curve = forecast_curve(clean, origin, days)
        ts = [dt.datetime.combine(day, dt.time()) + dt.timedelta(minutes=10 * k)
              for day in days for k in range(STEPS_PER_DAY)]
        frame = pl.DataFrame({"ts": ts, "Load_MW": curve, "is_imputed": [False] * len(ts)})
        return compute_targets(frame).sort("date").select(TARGET_NAMES)

    return predict
