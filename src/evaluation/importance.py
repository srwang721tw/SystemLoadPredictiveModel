"""特徵使用量統計。

在指定起點（預設為資料截止日，也就是提交時真正會用的訓練集）以前的資料上，
依上線設定訓練全部 18 個模型——量值 4 目標 × 3 horizon 與學習式時刻 2 目標 × 3 horizon
——統計每個特徵被用來分裂的次數與增益。

「所有模型的分裂次數都是 0」代表該特徵在訓練集上從未被使用：
刪掉它對模型的擬合沒有影響，只剩下 ``feature_fraction`` 抽樣位置的差異。
這是刪除的**必要條件**，是否刪除仍由回測決定（刪除變數須有數據依據）。
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def split_usage(origin: dt.date | None = None) -> pl.DataFrame:
    """每個特徵在 18 個模型中的分裂次數與增益。

    Args:
        origin: 訓練集截止日，None 時採資料截止日。

    Returns:
        pl.DataFrame: 每個特徵一列：``feature``、``models_used``（在幾個模型中被用到）、
            ``splits``、``gain``，另有每個模型一欄的分裂次數；依 ``splits`` 由小到大排序。
    """
    from src import workflow
    from src.data import checks, external
    from src.features import builder
    from src.models import quantile, timing

    origin = origin or checks.cutoff_date()
    daily = workflow.prepare_context(weather_mode="observed")[0]
    calendar_df, rules = external.load_calendar(), external.load_price_period_rules()
    event_days = external.load_event_days()["date"].to_list()

    columns: dict[str, dict[str, float]] = {}
    gains: dict[str, float] = {}
    for horizon in range(1, settings.PREDICT_HORIZON_DAYS + 1):
        features = builder.build_features(daily, calendar_df, rules, horizon)
        names = builder.feature_names(features)
        history = features.filter(pl.col("date") <= origin)

        boosters = {}
        magnitude = history.filter(pl.col("base_reliable") & ~pl.col("date").is_in(event_days))
        for target in settings.MAGNITUDE_TARGETS:
            fitted = magnitude.drop_nulls(target)
            boosters[f"{target}_h{horizon}"] = quantile.fit(
                fitted, fitted[target], target, names, in_ratio_space=False).booster

        previous = settings.TIMING_MIN_CLASS_COUNT
        settings.TIMING_MIN_CLASS_COUNT = settings.TIMING_LEARNED_MIN_CLASS_COUNT
        try:
            for target in settings.TIMING_TARGETS:
                fitted = history.drop_nulls([target])
                boosters[f"{target}_h{horizon}"] = timing.fit(
                    fitted, fitted[target], target, names,
                    num_rounds=settings.TIMING_LEARNED_ROUNDS).booster
        finally:
            settings.TIMING_MIN_CLASS_COUNT = previous

        for model_name, booster in boosters.items():
            splits = dict(zip(names, booster.feature_importance("split"), strict=True))
            gain = dict(zip(names, booster.feature_importance("gain"), strict=True))
            columns[model_name] = splits
            for name in names:
                gains[name] = gains.get(name, 0.0) + float(gain[name])

    features_all = sorted(gains)
    table = pl.DataFrame({"feature": features_all}).with_columns(
        pl.Series(model, [int(split.get(f, 0)) for f in features_all])
        for model, split in columns.items()
    )
    model_columns = list(columns)
    return table.with_columns(
        pl.sum_horizontal(pl.col(c) > 0 for c in model_columns).alias("models_used"),
        pl.sum_horizontal(model_columns).alias("splits"),
        pl.Series("gain", [gains[f] for f in features_all]),
    ).select("feature", "models_used", "splits", "gain", *model_columns).sort("splits", "gain")
