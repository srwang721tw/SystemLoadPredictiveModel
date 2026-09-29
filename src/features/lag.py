"""自迴歸特徵。

關鍵限制：預測 10/1–10/3 時資料只到 9/30，故 **10/2 的預測不能使用 10/1 的
實際值**。因此採**直接多步預測**：h = 1, 2, 3 各訓練一組模型，
每組只使用「預測起點日 T = 目標日 − h」及之前可得的資訊。

**洩漏防線靠 asof join 的結構保證**

所有落後與移動統計量都先在完整序列上算出「截至該日（含）」的值，
再以 ``join_asof(strategy="backward")`` 在 ``T`` 上取值。asof join 只會往回找，
結構上不可能取到 ``T`` 之後的資料。這比逐列判斷可靠，也讓 h = 1/2/3
共用同一份程式碼——只有 ``T`` 的定義不同。
"""

from __future__ import annotations

import polars as pl

from src.features.targets import TARGET_NAMES
from src.logging_setup import get_logger

logger = get_logger(__name__)

LAG_DAYS: tuple[int, ...] = (1, 7, 364)
"""落後日數。364 = 52 週，對齊「去年同星期同日」。

這些 lag 是相對於**目標日**的絕對日曆落後，不是相對於 T。
lag-1 對 h = 3 而言取的是 T−2，仍在可用範圍內；但 lag 小於 h 時就會洩漏，
故 :func:`add_lag_features` 會自動略過 ``lag < horizon`` 的項並記錄。
"""

ROLLING_WINDOWS: tuple[int, ...] = (7, 28)
"""移動平均／移動中位數的視窗長度（日）。"""


def _rolling_lookup(daily: pl.DataFrame, targets: tuple[str, ...]) -> pl.DataFrame:
    """建立「日期 → 截至該日（含）的移動統計量」對照表。

    Args:
        daily: 每日表。
        targets: 要計算統計量的欄位。

    Returns:
        pl.DataFrame: 依 ``date`` 排序，供 asof join 使用。
    """
    expressions = []
    for target in targets:
        for window in ROLLING_WINDOWS:
            expressions.extend(
                [
                    pl.col(target)
                    .rolling_mean(window_size=window, min_periods=2)
                    .alias(f"{target}_ma{window}"),
                    pl.col(target)
                    .rolling_median(window_size=window, min_periods=2)
                    .alias(f"{target}_med{window}"),
                    pl.col(target)
                    .rolling_std(window_size=window, min_periods=3)
                    .alias(f"{target}_std{window}"),
                ]
            )
    return daily.sort("date").with_columns(expressions).select(
        ["date"] + [e.meta.output_name() for e in expressions]
    )


def _same_daytype_lookup(
    daily: pl.DataFrame, targets: tuple[str, ...], daytype_column: str, n: int = 4
) -> pl.DataFrame:
    """建立「日期 → 截至該日（含）的同日別近期均值」對照表。

    Args:
        daily: 每日表。
        targets: 目標欄位。
        daytype_column: 日別欄位。
        n: 取幾個同日別樣本。

    Returns:
        pl.DataFrame: 含 ``date``、日別、各目標的同日別統計量。
    """
    expressions = [
        pl.col(target)
        .rolling_mean(window_size=n, min_periods=2)
        .over(daytype_column)
        .alias(f"{target}_daytype_ma{n}")
        for target in targets
    ]
    return daily.sort("date").with_columns(expressions).select(
        ["date", daytype_column] + [e.meta.output_name() for e in expressions]
    )


def add_lag_features(
    daily: pl.DataFrame,
    horizon: int,
    targets: tuple[str, ...] | None = None,
    daytype_column: str = "price_daytype",
) -> pl.DataFrame:
    """為指定 horizon 建立自迴歸特徵。

    所有特徵一律相對於「預測起點日 T = 目標日 − horizon」計算。

    Args:
        daily: 含目標值的每日表。
        horizon: 預測步長 h ∈ {1, 2, 3}。
        targets: 要取落後值的目標欄位，None 時採全部 6 項。
        daytype_column: 日別欄位。

    Returns:
        pl.DataFrame: 加上 lag、移動統計量、同日別均值等欄位。
    """
    targets = targets or TARGET_NAMES
    out = daily.sort("date").with_columns(
        (pl.col("date") - pl.duration(days=horizon)).alias("_origin")
    )

    # --- 絕對日曆落後：直接以 date 對齊，不需 asof ---
    skipped = [lag for lag in LAG_DAYS if lag < horizon]
    if skipped:
        logger.info("horizon=%d 略過會洩漏的 lag：%s", horizon, skipped)
    for lag in (lag for lag in LAG_DAYS if lag >= horizon):
        source = daily.select(
            (pl.col("date") + pl.duration(days=lag)).alias("date"),
            *[pl.col(t).alias(f"{t}_lag{lag}") for t in targets],
        )
        out = out.join(source, on="date", how="left")

    # --- 移動統計量與同日別均值：以 asof join 在 T 上取值 ---
    rolling = _rolling_lookup(daily, targets).rename({"date": "_roll_date"})
    out = out.sort("_origin").join_asof(
        rolling.sort("_roll_date"),
        left_on="_origin",
        right_on="_roll_date",
        strategy="backward",
    )

    same_daytype = _same_daytype_lookup(daily, targets, daytype_column).rename(
        {"date": "_dt_date"}
    )
    out = out.sort("_origin").join_asof(
        same_daytype.sort("_dt_date"),
        left_on="_origin",
        right_on="_dt_date",
        by=daytype_column,
        strategy="backward",
    )

    return out.drop("_origin", "_roll_date", "_dt_date", strict=False).sort("date")


def assert_no_leakage(
    features: pl.DataFrame, daily: pl.DataFrame, horizon: int, target: str = "p_day"
) -> None:
    """檢查自迴歸特徵是否使用了預測起點日之後的資訊。

    作法是**擾動測試**：把某一天 D 的目標值改成極端值，重算特徵，
    然後確認所有「目標日 ≤ D + horizon − 1」的列都沒有變化。
    若有變化，代表那些列看到了不該看到的 D。

    這比檢查欄位名稱可靠——欄位名稱正確不代表計算正確。

    Args:
        features: 已建好特徵的表。
        daily: 原始每日表。
        horizon: 預測步長。
        target: 用來做擾動的目標欄位。

    Raises:
        ValueError: 偵測到洩漏時拋出，訊息指出洩漏的欄位與日期。
    """
    probe_date = daily["date"][daily.height // 2]
    perturbed = daily.with_columns(
        pl.when(pl.col("date") == probe_date)
        .then(pl.col(target) * 100.0)
        .otherwise(pl.col(target))
        .alias(target)
    )
    after = add_lag_features(perturbed, horizon)

    # 目標日在探針日之前的列，其起點更早，不該受影響。
    before_rows = features.filter(pl.col("date") < probe_date)
    after_rows = after.filter(pl.col("date") < probe_date)

    lag_columns = [c for c in features.columns if target in c and c != target]
    leaked = [
        c
        for c in lag_columns
        if c in after_rows.columns
        and not before_rows[c].equals(after_rows[c])
    ]
    if leaked:
        raise ValueError(
            f"偵測到洩漏（horizon={horizon}）：擾動 {probe_date} 的 {target} 之後，"
            f"更早日期的下列欄位發生變化：{leaked}"
        )
    logger.info("洩漏檢查通過（horizon=%d，探針日 %s，檢查 %d 個欄位）",
                horizon, probe_date, len(lag_columns))
