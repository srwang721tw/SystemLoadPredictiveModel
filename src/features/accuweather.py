"""Accuweather 預報的每日特徵與偏差校正。

預測時目標日沒有觀測，氣象由本模組提供：Accuweather 五站的每日特徵，
以**預測起點以前**的重疊期逐站校正到 CODiS 尺度（:func:`make_target_weather_fn`）。
歷史列仍用 CODiS 觀測——這就是 honest 模式，也是上線設定。

---

## 為什麼要校正

取決於 ``settings.WEATHER_SOURCE``（歷史列的來源）：

| 模型來源 | 歷史 | 目標日 | 要不要校正 |
|---|---|---|---|
| ``"codis"`` | CODiS **觀測** | Accuweather **預報** | **要** |
| ``"accuweather"`` | Accuweather 預報 | Accuweather 預報 | **不要** |

``"codis"`` 時歷史與目標日來自兩個不同的量測系統，直接拼接會把系統性偏差
帶進 ``tbin`` 的門檻判定——門檻由 history（CODiS）算，目標日的值卻是 AW 的
尺度。實測 906 天重疊期：

| 項目 | r | bias (AW−CODiS) | MAE |
|---|---|---|---|
| ``day_tmax`` | 0.922 | **+0.97°C** | 1.76 |
| ``day_tmin`` | 0.955 | +0.65 | 1.39 |
| ``night_tmax`` | 0.936 | +0.60 | 1.43 |
| ``night_tmin`` | 0.935 | +0.01 | 1.47 |
| **五站 max（即 `tmax`）** | **0.922** | **+1.17°C** | 1.76 |

未校正時熱箱佔比被推高到 **58.7%**（CODiS 真值 50.4%），校正後回到 49.3%。

## 校正修得掉偏差，修不掉雜訊

逐日分箱不一致率只從 **12.7% → 10.2%**：殘差 sd 1.73°C 相對於門檻鄰域
（30.70°C）是大的，這 10% 不可約。**不要為了再壓低它而去做更複雜的校正**——誤差以隨機為主，
不是系統性偏移，複雜模型只會過擬合重疊期。

## `uv_max` 不是同一個量，不做校正

CODiS 取 ``UVIndex.Accumulation``（連續值），Accuweather 是 ``max_uv_index``
（**整數 1–13**）。清掉哨兵後兩者相關僅 **0.431**、MAE 2.54——這不是尺度差異
而是**定義差異**，線性校正無法對齊。故 :data:`CALIBRATED_METRICS` 只含四個
溫度項；UV 也因此不進模型（``settings.WEATHER_METRICS``）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

CALIBRATED_METRICS = ("day_tmax", "day_tmin", "night_tmax", "night_tmin")
"""要做偏差校正的項目。

刻意**不含** ``uv_max``——它與 CODiS 的 UV 不是同一個量（r=0.431），
線性校正無法對齊定義差異。見模組 docstring。
"""


def _metric_of(column: str) -> str | None:
    """由 ``w_{站名}_{項目}`` 取出項目名。"""
    for metric in CALIBRATED_METRICS:
        if column.endswith(f"_{metric}"):
            return metric
    return None


def fit_bias_correction(
    forecast: pl.DataFrame, observed: pl.DataFrame
) -> dict[str, tuple[float, float]]:
    """在重疊期上逐欄以最小平方法求 ``observed ≈ slope × forecast + intercept``。

    Args:
        forecast: Accuweather 的每日特徵（``date`` + ``w_*`` 氣象欄）。
        observed: CODiS 的每日特徵，欄位相同。

    Returns:
        dict: ``{欄位: (slope, intercept)}``，只含 :data:`CALIBRATED_METRICS`
            對應的欄位。重疊樣本不足 30 天的欄位會被略過並警告。

    Raises:
        ValueError: 兩張表的欄位不一致（代表上游的 欄位契約已破裂）。
    """
    if set(forecast.columns) != set(observed.columns):
        raise ValueError(
            "預報與觀測的欄位不一致，欄位契約已破裂："
            f"{sorted(set(forecast.columns) ^ set(observed.columns))}"
        )

    joined = forecast.join(observed, on="date", how="inner", suffix="_obs")
    coefficients: dict[str, tuple[float, float]] = {}
    for column in forecast.columns:
        if column == "date" or _metric_of(column) is None:
            continue
        pair = joined.select(column, f"{column}_obs").drop_nulls()
        if pair.height < 30:
            logger.warning("%s 的重疊樣本只有 %d 天，不做校正", column, pair.height)
            continue
        x = pair[column].to_numpy().astype(float)
        y = pair[f"{column}_obs"].to_numpy().astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        coefficients[column] = (float(slope), float(intercept))

    logger.info(
        "偏差校正：%d 欄，重疊 %d 天（uv_max 刻意不校正）",
        len(coefficients), joined.height,
    )
    return coefficients


def apply_bias_correction(
    forecast: pl.DataFrame, coefficients: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """套用校正係數。未列於 ``coefficients`` 的欄位原樣保留。

    Args:
        forecast: Accuweather 的每日特徵。
        coefficients: :func:`fit_bias_correction` 的輸出。

    Returns:
        pl.DataFrame: 校正後的每日特徵，欄位與輸入相同。
    """
    return forecast.with_columns(
        [
            (pl.col(column) * slope + intercept).alias(column)
            for column, (slope, intercept) in coefficients.items()
        ]
    )


TARGET_TEMPERATURE_COLUMN = "tmax"
"""目標日氣溫欄名。須與 ``src.models.pipeline.TEMPERATURE_COLUMN`` 一致
（本模組不 import pipeline，以免 features → models 的反向依賴）。"""


def make_target_weather_fn(
    forecast: pl.DataFrame, observed: pl.DataFrame
) -> Callable[[dt.date, Sequence[dt.date]], pl.DataFrame]:
    """建立「目標日氣象」的供應函式，供 honest 模式（與提交時相同的情境）使用。

    提交當天真正會發生的情境是：**歷史**用 CODiS 觀測（分箱門檻、訓練列），
    **目標日**只有預報。CV 若讓目標日也用觀測，量到的就是一個提交時
    達不到的分數——這個函式把目標日換回預報，歷史不動。

    **校正係數逐折擬合，只用 ``date <= origin`` 的重疊期。**
    比照 ``tbin`` 門檻「逐折由 history 算」的規矩：提交時也是用 9/30
    以前的資料擬合、套到 10/1–3。用全期擬合會讓目標日的觀測值經由
    斜率與截距洩漏回來（雖然只佔 1/900，但規矩不因小而破例）。

    ``uv_max`` 不校正（定義不同，見模組 docstring）——提交時目標日的 UV
    也只能來自 Accuweather，所以這是忠實的模擬。

    Args:
        forecast: Accuweather 的每日特徵（``date`` + ``w_*`` 氣象欄），**未校正**。
        observed: CODiS 的每日特徵，欄位相同。

    Returns:
        Callable: ``fn(origin, target_dates)``，回傳目標日的 ``date`` +
            ``w_*`` 氣象欄 + ``tmax``（五站 ``day_tmax`` 的最大值，與
            ``workflow.prepare_context`` 的算法一致）。

    Raises:
        ValueError: 兩張表欄位不一致（由 :func:`fit_bias_correction` 拋出）。
    """
    tmax_columns = [c for c in forecast.columns if c.endswith("_day_tmax")]

    def supply(origin: dt.date, target_dates: Sequence[dt.date]) -> pl.DataFrame:
        coefficients = fit_bias_correction(
            forecast.filter(pl.col("date") <= origin),
            observed.filter(pl.col("date") <= origin),
        ) if settings.WEATHER_CALIBRATE else {}
        rows = apply_bias_correction(
            forecast.filter(pl.col("date").is_in(list(target_dates))), coefficients
        )
        rows = pl.DataFrame({"date": sorted(target_dates)}).join(rows, on="date", how="left")
        rows = _apply_plan_b(_mask_unavailable(rows, origin), observed, origin)
        return rows.with_columns(
            pl.max_horizontal(tmax_columns).alias(TARGET_TEMPERATURE_COLUMN)
        ).sort("date")

    return supply


def _mask_unavailable(rows: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    """把取不到預報的格子清成缺值，交給 :func:`_apply_plan_b`。

    兩個來源：比賽當天實際缺的 ``settings.FORECAST_MISSING_CELLS``（由前置檢查判定，
    某站某天不足 24 小時就算缺），以及回測模擬用的 ``settings.FORECAST_UNAVAILABLE_*``。
    """
    horizons = settings.FORECAST_UNAVAILABLE_HORIZONS
    stations = settings.FORECAST_UNAVAILABLE_STATIONS
    if horizons or stations:
        days = [origin + dt.timedelta(days=h) for h in horizons]
        columns = [c for c in rows.columns if c.startswith("w_")
                   and (not stations or c.split("_")[1] in stations)]
        hit = pl.col("date").is_in(days) if horizons else pl.lit(True)
        rows = rows.with_columns(
            pl.when(hit).then(None).otherwise(pl.col(c)).alias(c) for c in columns)
    for day, missing in settings.FORECAST_MISSING_CELLS.items():
        columns = [c for c in rows.columns if c.startswith("w_") and c.split("_")[1] in missing]
        rows = rows.with_columns(
            pl.when(pl.col("date") == day).then(None).otherwise(pl.col(c)).alias(c)
            for c in columns)
    return rows


def _apply_plan_b(rows: pl.DataFrame, observed: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    """目標日缺預報時，依 ``settings.FORECAST_PLAN_B`` 補值。

    - ``"abort"``：中止。不可靜默退回觀測或留空
    - ``"persistence"``：作業時點前最近一天的觀測（``date <= origin`` 的最後一列）
    - ``"climatology"``：起點以前、與目標日同月份的觀測平均

    兩種補法都只用 ``date <= origin`` 的觀測，逐欄（逐站、逐項）補。

    Raises:
        ValueError: 有缺值且 Plan B 為 ``"abort"``，或 Plan B 名稱不支援。
    """
    columns = [c for c in rows.columns if c.startswith("w_")]
    gaps = rows.filter(pl.any_horizontal(pl.col(c).is_null() for c in columns))
    if not gaps.height:
        return rows
    plan = settings.FORECAST_PLAN_B
    if plan == "abort":
        raise ValueError(f"Accuweather 缺少目標日 {gaps['date'].to_list()} 的預報"
                         "（Plan B 尚未選定，中止）")
    history = observed.filter(pl.col("date") <= origin).sort("date")
    if plan == "persistence":
        latest = history.tail(1)
        fill = {d: latest for d in gaps["date"].to_list()}
    elif plan == "climatology":
        fill = {
            d: history.filter(pl.col("date").dt.month() == d.month).select(
                pl.col(c).mean() for c in columns)
            for d in gaps["date"].to_list()
        }
    else:
        raise ValueError(f"不支援的 FORECAST_PLAN_B：{plan!r}")
    logger.warning("目標日 %s 缺預報，以 Plan B「%s」補值", gaps["date"].to_list(), plan)
    return rows.with_columns(
        pl.col(c).fill_null(pl.col("date").replace_strict(
            {d: frame[c].item() for d, frame in fill.items()}, default=None,
            return_dtype=pl.Float64))
        for c in columns
    )


def build_forecast() -> pl.DataFrame:
    """Accuweather 五站的每日氣象特徵（未校正），格式同 CODiS 的 ``w_*`` 氣象欄。

    校正統一在 :func:`make_target_weather_fn` 內**逐起點**進行——比賽當天也是
    用起點以前的重疊期擬合、套到目標日。

    Returns:
        pl.DataFrame: ``date`` + ``w_*`` 氣象欄，涵蓋 Accuweather 的所有日期。
    """
    from src.data import accuweather
    from src.features import weather as weather_features

    filled, _ = weather_features.interpolate_hourly(accuweather.load_station_hourly())
    return weather_features.daily_features(filled)



def compare_with_observed(forecast: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """預報與觀測在重疊期的逐項誤差（五站合併），供探索分析呈現。

    Args:
        forecast: Accuweather 每日特徵（:func:`build_forecast`）。
        observed: CODiS 每日特徵，欄位相同。

    Returns:
        pl.DataFrame: 每個項目一列：相關係數 ``r``、``bias``（預報 − 觀測）、``MAE``、
            重疊樣本數 ``n``；最後一列為時刻模型用的五站 ``day_tmax`` 最大值。
    """
    joined = forecast.join(observed, on="date", how="inner", suffix="_obs")
    metrics = sorted({c.split("_", 2)[2] for c in forecast.columns if c.startswith("w_")})

    def summarise(label: str, pairs: list[tuple[str, str]]) -> dict:
        frame = pl.concat([
            joined.select(pl.col(f).alias("f"), pl.col(o).alias("o")) for f, o in pairs
        ]).drop_nulls()
        return {
            "項目": label, "n": frame.height,
            "r": frame.select(pl.corr("f", "o")).item(),
            "bias": (frame["f"] - frame["o"]).mean(),
            "MAE": (frame["f"] - frame["o"]).abs().mean(),
        }

    rows = [
        summarise(metric, [(c, f"{c}_obs") for c in forecast.columns
                           if c.startswith("w_") and c.endswith(f"_{metric}")])
        for metric in metrics
    ]
    tmax = [c for c in forecast.columns if c.endswith("_day_tmax")]
    joined = joined.with_columns(
        pl.max_horizontal(tmax).alias("tmax"),
        pl.max_horizontal([f"{c}_obs" for c in tmax]).alias("tmax_obs"),
    )
    rows.append(summarise("五站 day_tmax 最大值", [("tmax", "tmax_obs")]))
    return pl.DataFrame(rows)
