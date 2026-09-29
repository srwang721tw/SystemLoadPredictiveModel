"""缺失值補值。

零星缺失（≤ ``IMPUTE_SHORT_GAP_MAX_POINTS`` 點）與連續長段缺失採不同策略，
並保留 ``is_imputed`` 旗標供 ramp 污染分析使用。

**補值必然污染 ramp 標籤，只能量化不能消除**：
線性內插把該段差分壓成常數，人為降低 ramp；前值填補則製造一個 0 差分與一個
雙倍差分。因此旗標必須一路保留到計算標籤時，由
``src/features/targets.compute_targets`` 標記 ramp 極值是否落在補值差分上。
"""

from __future__ import annotations

import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def label_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """為每段連續缺失標上編號與長度。

    Args:
        df: 含 ``ts``、``Load_MW``、``is_missing`` 的完整格點序列。

    Returns:
        pl.DataFrame: 加上 ``gap_id``（非缺失處為 null）與 ``gap_len``。
    """
    # 每當 is_missing 由 False 轉 True 就開啟一個新的缺失段。
    return (
        df.with_columns(
            (pl.col("is_missing") & ~pl.col("is_missing").shift(1, fill_value=False))
            .cum_sum()
            .alias("_run")
        )
        .with_columns(
            pl.when(pl.col("is_missing")).then(pl.col("_run")).otherwise(None).alias("gap_id")
        )
        .with_columns(pl.len().over("gap_id").alias("gap_len"))
        .with_columns(
            pl.when(pl.col("is_missing")).then(pl.col("gap_len")).otherwise(0).alias("gap_len")
        )
        .drop("_run")
    )


def gap_report(df: pl.DataFrame) -> pl.DataFrame:
    """列出所有缺失段的位置與長度。

    Args:
        df: :func:`label_gaps` 的輸出。

    Returns:
        pl.DataFrame: 每段一列，含起訖時刻、長度、分類（零星／長段）。
    """
    gaps = (
        df.filter(pl.col("is_missing"))
        .group_by("gap_id")
        .agg(
            pl.col("ts").min().alias("start"),
            pl.col("ts").max().alias("end"),
            pl.len().alias("n_points"),
        )
        .sort("start")
    )
    if gaps.height == 0:
        return gaps
    return gaps.with_columns(
        pl.when(pl.col("n_points") <= settings.IMPUTE_SHORT_GAP_MAX_POINTS)
        .then(pl.lit("零星"))
        .otherwise(pl.lit("長段"))
        .alias("kind")
    )


def _impute_daytype_profile(df: pl.DataFrame) -> pl.DataFrame:
    """以「鄰近同日別日的同時刻中位數形狀 × 當日水準比例」補長段缺失。

    步驟：
        1. 對每個缺失點，取同一 ``mod``（當日分鐘數）、同一 ``daytype``、
           且日期最接近的 ``LONG_GAP_PROFILE_DAYS`` 天的中位數，作為形狀值。
        2. 以當日**已觀測**點與同一組參考日在相同時刻的比值中位數作為水準係數。
        3. 補值 = 形狀值 × 水準係數。

    形狀與水準分開處理的理由：直接取鄰近日的絕對值會忽略當日整體水準
    （例如當天特別熱），只取當日水準又沒有形狀資訊。

    Args:
        df: 含 ``ts``、``date``、``mod``、``Load_MW``、``is_missing``、
            ``daytype``、``gap_len`` 的序列。

    Returns:
        pl.DataFrame: 加上 ``profile_fill`` 欄位（僅長段缺失點有值）。
    """
    k = settings.LONG_GAP_PROFILE_DAYS
    long_gap_dates = (
        df.filter(pl.col("gap_len") > settings.IMPUTE_SHORT_GAP_MAX_POINTS)["date"]
        .unique()
        .to_list()
    )

    fills: list[pl.DataFrame] = []
    for target_date in long_gap_dates:
        daytype = df.filter(pl.col("date") == target_date)["daytype"][0]
        # 參考日：同日別、非本日、且該日完全沒有缺失，取日期最接近的 2k 天。
        clean_days = (
            df.group_by("date")
            .agg(pl.col("is_missing").any().alias("bad"), pl.col("daytype").first())
            .filter(~pl.col("bad") & (pl.col("daytype") == daytype))
            .with_columns(
                (pl.col("date") - target_date).dt.total_days().abs().alias("dist")
            )
            .sort("dist")
            .head(2 * k)["date"]
            .to_list()
        )
        ref = df.filter(pl.col("date").is_in(clean_days))
        shape = ref.group_by("mod").agg(pl.col("Load_MW").median().alias("shape"))

        today = df.filter(pl.col("date") == target_date).join(shape, on="mod", how="left")
        # 水準係數：當日已觀測點相對於形狀值的比值中位數。
        level = (
            today.filter(~pl.col("is_missing"))
            .select((pl.col("Load_MW") / pl.col("shape")).median())
            .item()
        )
        fills.append(
            today.filter(pl.col("is_missing")).select(
                "ts", (pl.col("shape") * level).alias("profile_fill")
            )
        )

    if not fills:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("profile_fill"))
    return df.join(pl.concat(fills), on="ts", how="left")


def impute_load(df: pl.DataFrame) -> pl.DataFrame:
    """補齊 ``Load_MW`` 缺失值並標記補值點。

    Args:
        df: 完整格點序列，須含 ``ts``、``Load_MW``、``is_missing``、``daytype``。

    Returns:
        pl.DataFrame: 欄位 ``ts``、``Load_MW``（已補值）、``is_imputed``、
            ``impute_method``（``"linear"`` / ``"daytype_profile"`` / null）。
    """
    df = label_gaps(df)
    report = gap_report(df)
    if report.height == 0:
        logger.info("無缺失值，不需補值。")
        return df.select(
            "ts",
            "Load_MW",
            pl.lit(False).alias("is_imputed"),
            pl.lit(None, dtype=pl.String).alias("impute_method"),
        )

    logger.info("缺失段共 %d 段：", report.height)
    for row in report.iter_rows(named=True):
        logger.info(
            "  [%s] %s ~ %s，%d 點",
            row["kind"], row["start"], row["end"], row["n_points"],
        )

    is_long = pl.col("gap_len") > settings.IMPUTE_SHORT_GAP_MAX_POINTS

    if settings.LONG_GAP_STRATEGY == "daytype_profile":
        df = _impute_daytype_profile(df)
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("profile_fill"))

    before = df.filter(~pl.col("is_missing"))["Load_MW"]
    out = (
        df.with_columns(
            # 線性內插只填零星缺失；長段缺失先留 null 再由 profile_fill 覆蓋。
            pl.col("Load_MW").interpolate().alias("_linear")
        )
        .with_columns(
            pl.when(~pl.col("is_missing"))
            .then(pl.col("Load_MW"))
            .when(is_long & pl.col("profile_fill").is_not_null())
            .then(pl.col("profile_fill"))
            .otherwise(pl.col("_linear"))
            .alias("Load_MW"),
            pl.col("is_missing").alias("is_imputed"),
            pl.when(~pl.col("is_missing"))
            .then(None)
            .when(is_long & pl.col("profile_fill").is_not_null())
            .then(pl.lit(settings.LONG_GAP_STRATEGY))
            .otherwise(pl.lit("linear"))
            .alias("impute_method"),
        )
        .select("ts", "Load_MW", "is_imputed", "impute_method")
    )

    after = out.filter(pl.col("is_imputed"))["Load_MW"]
    logger.info(
        "補值完成：%d 點（線性 %d、%s %d）。"
        "補值前非缺失均值 %.1f MW，補入值均值 %.1f MW",
        out["is_imputed"].sum(),
        out.filter(pl.col("impute_method") == "linear").height,
        settings.LONG_GAP_STRATEGY,
        out.filter(pl.col("impute_method") == settings.LONG_GAP_STRATEGY).height,
        before.mean(),
        after.mean() if after.len() else float("nan"),
    )
    if out["Load_MW"].null_count():
        raise ValueError(f"補值後仍有 {out['Load_MW'].null_count()} 個 null")
    return out


def summarize_imputation(df: pl.DataFrame) -> pl.DataFrame:
    """統計補值筆數與方法分布。

    Args:
        df: :func:`impute_load` 的輸出。

    Returns:
        pl.DataFrame: 每種補值方法一列，含筆數與涉及天數。
    """
    return (
        df.filter(pl.col("is_imputed"))
        .with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by("impute_method")
        .agg(pl.len().alias("n_points"), pl.col("date").n_unique().alias("n_days"))
        .sort("impute_method")
    )
