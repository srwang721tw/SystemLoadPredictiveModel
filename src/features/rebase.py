"""相對基準的比值轉換與還原。

4 個量值目標（p_day / p_night / ramp_up / ramp_down）不直接預測絕對值，
改預測「相對於近期基準的比值」：

    ratio = y_true / base(T)          base 僅由 ≤ T 的資料算出
    y_hat = base(T) × ratio_hat

**為何這樣做**

1. GBDT 無法外插。樹的預測恆被夾在訓練集目標值的範圍內，用絕對值建模時，
   若預測期水準超出訓練集見過的範圍，模型會直接封頂。
2. base 吸收年度成長與水準漂移。實測 2024→2025 年增 +6.5%~+10%、
   2025→2026 卻是 −0.5%，成長並非穩定趨勢，用趨勢項外推會錯得很明顯。

**為何 τ 不受影響**

``base(T)`` 在推論時是已知的正常數，而分位數對正的常數縮放可交換：

    Q_τ(y) = Q_τ(base × ratio) = base × Q_τ(ratio)

因此由評分函數推導出的 τ（peak 0.625、ramp_up 0.70、ramp_down 0.50）
原封不動適用於 ratio，不需要任何修正。

**洩漏防線靠 join_asof 的結構保證，而非靠自律**

對 horizon h 的樣本（目標日 D），基準日一律是 ``T = D − h``。實作上先在
同日別序列上算出「截至該日（含）的最近 K 次中位數」，再以
``join_asof(strategy="backward")`` 在 ``T`` 上取值——asof join 只會往回找，
結構上不可能取到 ``T`` 之後的資料。這比在函式內部逐列判斷可靠得多。
"""

from __future__ import annotations

import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def base_lookup(
    daily: pl.DataFrame,
    target: str,
    daytype_column: str = "price_daytype",
    n_occurrences: int | None = None,
) -> pl.DataFrame:
    """建立「日期 → 截至該日（含）的同日別基準值」對照表。

    基準取最近 ``n_occurrences`` 個同日別樣本的中位數。中位數而非平均，
    以抵抗其中夾雜 1–2 個國定假日造成的向下污染。

    Args:
        daily: 含 ``date``、日別欄位與 ``target`` 的每日表。
        target: 目標欄位名稱。
        daytype_column: 日別欄位。預設用日曆表的 ``price_daytype``，
            因為它把國定假日正確歸為離峰日。
        n_occurrences: 取幾個同日別樣本，None 時採
            ``settings.REBASE_N_OCCURRENCES``。

    Returns:
        pl.DataFrame: 欄位 ``date``、日別、``base``。依 ``date`` 排序，
            供 :func:`add_base_columns` 以 asof join 取用。
    """
    k = n_occurrences or settings.REBASE_N_OCCURRENCES
    return (
        daily.select("date", daytype_column, target)
        .drop_nulls(target)
        .sort("date")
        .with_columns(
            pl.col(target)
            .rolling_median(window_size=k, min_periods=settings.REBASE_MIN_OCCURRENCES)
            .over(daytype_column)
            .alias("base")
        )
        .select("date", daytype_column, "base")
        .sort("date")
    )


def add_base_columns(
    daily: pl.DataFrame,
    horizon: int,
    targets: tuple[str, ...] | None = None,
    daytype_column: str = "price_daytype",
    n_occurrences: int | None = None,
) -> pl.DataFrame:
    """為每個目標日加上其基準值與比值欄位。

    對 horizon h，目標日 D 的基準日為 ``T = D − h``；基準取 T 及之前
    最近 ``n_occurrences`` 個「與 D 同日別」的樣本中位數。

    Args:
        daily: 每日表，須含 ``date``、日別欄位與所有 ``targets`` 欄位。
        horizon: 預測步長 h ∈ {1, 2, 3}。
        targets: 要轉換的目標欄位，None 時採 ``settings.MAGNITUDE_TARGETS``。
        daytype_column: 日別欄位。
        n_occurrences: 同 :func:`base_lookup`。

    Returns:
        pl.DataFrame: 每個目標新增 ``{target}_base`` 與 ``{target}_ratio``，
            並新增 ``base_reliable``（所有基準皆有值時為真）。
    """
    targets = targets or settings.MAGNITUDE_TARGETS
    out = daily.sort("date").with_columns(
        (pl.col("date") - pl.duration(days=horizon)).alias("_origin")
    )

    for target in targets:
        lookup = base_lookup(daily, target, daytype_column, n_occurrences).rename(
            {"base": f"{target}_base", "date": "_base_date"}
        )
        # join_asof 會把右表的鍵欄位一起帶進來，多目標迴圈時會撞名，
        # 故每次 join 後立刻丟掉。
        out = (
            out.sort("_origin")
            .join_asof(
                lookup.sort("_base_date"),
                left_on="_origin",
                right_on="_base_date",
                by=daytype_column,
                strategy="backward",
            )
            .drop("_base_date", strict=False)
        )
        out = out.with_columns(
            pl.when(pl.col(f"{target}_base") > 0)
            .then(pl.col(target) / pl.col(f"{target}_base"))
            .otherwise(None)
            .alias(f"{target}_ratio")
        )

    base_columns = [f"{t}_base" for t in targets]
    return (
        out.with_columns(
            pl.all_horizontal([pl.col(c).is_not_null() & (pl.col(c) > 0) for c in base_columns])
            .alias("base_reliable")
        )
        .drop("_origin", "_base_date", strict=False)
        .sort("date")
    )


def flag_outlier_ratios(
    daily: pl.DataFrame, targets: tuple[str, ...] | None = None
) -> pl.DataFrame:
    """標記比值落在 ``settings.REBASE_CLIP`` 之外的日子。

    這類日子通常是颱風或重大假日（實測颱風日 ramp_up 掉到正常日的 1/3，
    兩群完全不重疊）。比值本身即為一個資料驅動的異常偵測器。

    僅標記與回報，**不自行刪除**任何日子。

    Args:
        daily: 已含比值欄位的每日表。
        targets: 要檢查的目標，None 時採 ``settings.MAGNITUDE_TARGETS``。

    Returns:
        pl.DataFrame: 加上 ``{target}_ratio_outlier`` 與 ``any_ratio_outlier``。
    """
    targets = targets or settings.MAGNITUDE_TARGETS
    low, high = settings.REBASE_CLIP
    flags = [
        (
            pl.col(f"{t}_ratio").is_not_null()
            & ~pl.col(f"{t}_ratio").is_between(low, high)
        ).alias(f"{t}_ratio_outlier")
        for t in targets
    ]
    return daily.with_columns(flags).with_columns(
        pl.any_horizontal([pl.col(f"{t}_ratio_outlier") for t in targets]).alias(
            "any_ratio_outlier"
        )
    )


