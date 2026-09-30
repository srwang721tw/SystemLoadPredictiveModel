"""端到端流程：重建中間檔、組裝預測器、預測曲線、回測評估。

``run_submission.py``、``main.py`` 與 notebook 都呼叫這裡的函式，
同一套流程只有一份實作——提交檔與回測分數出自同一段程式。
"""

from __future__ import annotations

import contextlib
import datetime as dt
from pathlib import Path

import polars as pl

from config import paths, settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def build_processed() -> tuple[pl.DataFrame, pl.DataFrame]:
    """由原始檔重建中間檔：10 分鐘補值序列、每日 6 目標、特殊日期每日表。

    流程：讀檔（含結構驗證與截止日過濾）→ 補齊格點 → 完整性檢查 → 補值 →
    每日 6 目標 → 併入電價日曆的日別。換上新的負載檔後必須先跑這一步。

    Returns:
        tuple: ``(imputed, daily)``——補值後的 10 分鐘序列與每日表。
    """
    from src.data import checks, external, imputer, loader, special_days
    from src.features import targets

    raw = loader.load_raw_load()
    full = loader.reindex_full_grid(raw)

    bad_days = checks.check_completeness(full)
    if bad_days.height:
        logger.warning("有缺漏的日子：\n%s", bad_days)

    # 補值需要日別才能取同日別的參考日，故先補上日曆欄位。
    full = targets.add_daytype(targets.add_time_columns(full))
    imputed = imputer.impute_load(full)
    logger.info("補值摘要：\n%s", imputer.summarize_imputation(imputed))
    imputed.write_parquet(paths.CLEAN_LOAD_FILE)

    daily = targets.build_daily(imputed)
    # 日曆表提供權威的日別（國定假日歸「週日及離峰日」），優先於由星期推導者。
    calendar = external.load_calendar()
    logger.info("日曆覆蓋檢查：%s", external.check_coverage(daily, calendar))
    daily = external.merge_on_date(daily, calendar)
    daily.write_parquet(paths.TARGETS_FILE)
    logger.info("中間檔已重建：%d 天（%s ~ %s）", daily.height, daily["date"].min(), daily["date"].max())

    special_days.build()
    return imputed, daily


def extend_for_prediction(daily: pl.DataFrame, calendar_df: pl.DataFrame, days: list) -> pl.DataFrame:
    """把待預測日期以「只有事前已知欄位」的空列附加到每日表。

    目標日沒有標籤，但特徵組裝需要這些列存在，才能算出它們的日曆／天文特徵，
    並對它們取歷史落後值。附加列的標籤一律為 null；日別來自日曆表（事前已知）。

    Args:
        daily: 既有的每日表。
        calendar_df: 電價日曆表。
        days: 待預測日期。

    Returns:
        pl.DataFrame: 附加後的每日表，依日期排序。
    """
    from src.features import targets as target_module

    missing = [d for d in days if daily.filter(pl.col("date") == d).height == 0]
    if not missing:
        return daily

    blank = pl.DataFrame({"date": missing}).with_columns(pl.col("date").cast(daily.schema["date"]))
    blank = target_module.add_daytype(blank.join(calendar_df, on="date", how="left"))
    for column, dtype in daily.schema.items():
        if column not in blank.columns:
            blank = blank.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return pl.concat([daily, blank.select(daily.columns)], how="vertical").sort("date")


WEATHER_MODES = ("observed", "honest", "forecast")
"""評估時的氣象模式：

| 模式 | 歷史 | 目標日 | 意義 |
|---|---|---|---|
| ``honest`` | CODiS 觀測 | 逐起點校正後的 Accuweather 預報 | 與提交時相同（預設） |
| ``observed`` | CODiS 觀測 | CODiS 觀測 | 樂觀：提交時目標日沒有觀測，只作對照 |
| ``forecast`` | Accuweather | Accuweather | 歷史也改用預報 |
"""


def prepare_context(extra_days: list | None = None, weather_mode: str = "honest"):
    """載入資料並組裝預測器（時刻 + 量值）。

    Args:
        extra_days: 需要納入但尚無標籤的待預測日期。
        weather_mode: 見 :data:`WEATHER_MODES`。``forecast`` 模式須由呼叫端先把
            ``settings.WEATHER_SOURCE`` 設為 ``"accuweather"``（特徵矩陣在預測當下才讀氣象）。

    Returns:
        tuple: ``(daily, clean, attributes, target_predictor, calendar_df, rules)``。
    """
    from src.data import external
    from src.features import builder, calendar
    from src.features import targets as target_module
    from src.models import pipeline

    daily = pl.read_parquet(paths.TARGETS_FILE)
    clean = pl.read_parquet(paths.CLEAN_LOAD_FILE)
    calendar_df = external.load_calendar()
    rules = external.load_price_period_rules()
    if extra_days:
        daily = extend_for_prediction(daily, calendar_df, extra_days)

    attributes = calendar.add_summer_flag(
        daily.select("date", "price_daytype", "month"), rules
    ).with_columns(pl.col("date").dt.weekday().alias("weekday"))

    # 白天最高溫是時刻條件變數之一，掛在 attributes 上（時刻的經驗分布不吃特徵矩陣）。
    # 五站取最大值，代表全系統的冷氣負載壓力；目標日的值由預報覆寫。
    if settings.TIMING_TEMPERATURE_BINS > 1:
        weather_daily = external.load_weather()
        station_columns = [c for c in weather_daily.columns if c.endswith("_day_tmax")]
        attributes = attributes.join(
            weather_daily.select(
                "date", pl.max_horizontal(station_columns).alias(pipeline.TEMPERATURE_COLUMN)
            ),
            on="date", how="left",
        )
    # 只併 is_summer：weekday 在每日表裡本來就有，再併一次會產生撞名欄位並悄悄成為特徵。
    daily = daily.join(attributes.select("date", "is_summer"), on="date", how="left")
    # 形狀模板需要日別與夏月旗標，一併掛到 10 分鐘序列上。
    clean = target_module.add_time_columns(clean).join(
        daily.select("date", "price_daytype", "is_summer"), on="date", how="left"
    )

    # 學習式時刻修正需要特徵矩陣。一次建好三個 horizon，逐日訓練時再依起點切：
    # 特徵全部是回望的，改動某天的標籤不會改變它自己或更早日子的特徵。
    learned_features = None
    if settings.TIMING_LEARNED_MIX > 0:
        pending = tuple(extra_days or ()) if weather_mode == "honest" else ()
        learned_features = {
            h: builder.build_features(daily, calendar_df, rules, h, pending_weather_dates=pending)
            for h in range(1, settings.PREDICT_HORIZON_DAYS + 1)
        }
    # honest 模式：歷史用觀測，目標日的氣象換成逐起點校正後的預報。
    target_weather_fn = None
    if weather_mode == "honest":
        from src.features import accuweather as accuweather_features

        target_weather_fn = accuweather_features.make_target_weather_fn(
            accuweather_features.build_forecast(),
            external._load_weather_codis(fill=False),
        )
    timing_fn = pipeline.make_ensemble_timing_predictor(
        attributes, clean, settings.TIMING_ENSEMBLE_WEIGHTS,
        learned_features=learned_features,
        target_weather_fn=target_weather_fn,
    )
    # 事件日（地震等）的負載被動被壓低，標籤不可信，不進量值訓練。
    event_days = tuple(external.load_event_days()["date"].to_list())
    target_fn = pipeline.make_magnitude_predictor(
        daily, calendar_df, rules, timing_fn, in_ratio_space=False,
        excluded_dates=event_days, target_weather_fn=target_weather_fn,
    )
    return daily, clean, attributes, target_fn, calendar_df, rules


def predict_days(origin: dt.date, days: list[dt.date]) -> tuple[dict, dict]:
    """以 ``origin``（含）以前的資料，預測 ``days`` 的 144 點曲線。

    流程：前置檢查 → 預測 6 個尖峰目標 → 依歷史同組日的形狀合成曲線。

    外生資料不齊時的備援（Plan B）：

    - Accuweather 部分城市缺 → 那些城市以 ``settings.FORECAST_PLAN_B``（最近一天的觀測）補
    - Accuweather 所有城市都缺 → 那幾天改用不含氣象的模型
    - Windy 不齊 → 那幾天改用不含 Windy 的模型

    缺的東西不同的日子分成幾組，每組以對應的設定重新訓練、預測一次，只取該組的日子。

    Args:
        origin: 預測起點日，也就是負載資料的最後一天。
        days: 待預測日期。

    Raises:
        ValueError: 前置檢查未通過（負載或 CODiS 涵蓋不足，見 ``checks.precheck``）。

    Returns:
        tuple: ``(curves, intended, requested)``——``{日期: 144 點曲線}``、
            ``{日期: 合成後實際達成的 6 目標}``（供寫檔後讀回驗證），以及
            ``{日期: 模型原始預測的 6 目標}``（合成修補之前，兩者只可能在 ramp 上不同）。
    """
    from src.data import checks

    status = checks.precheck(origin, days)
    missing = status["forecast_missing"]
    if settings.FORECAST_PLAN_B_FULL_DAY != "no_weather":
        raise ValueError(f"不支援的 FORECAST_PLAN_B_FULL_DAY：{settings.FORECAST_PLAN_B_FULL_DAY!r}")

    groups: dict[tuple[bool, bool], list[dt.date]] = {}
    for day in days:
        no_weather = len(missing.get(day, ())) == len(settings.WEATHER_STATIONS)
        no_windy = day in status["windy_missing"]
        groups.setdefault((no_weather, no_windy), []).append(day)

    curves, intended, requested = {}, {}, {}
    for (no_weather, no_windy), group_days in groups.items():
        overrides = {"FORECAST_MISSING_CELLS": dict(missing)}
        if no_weather:
            overrides |= {"ENABLE_TIER1_WEATHER": False, "TIMING_TEMPERATURE_BINS": 1}
        if no_windy:
            overrides |= {"ENABLE_WINDY": False}
        if no_weather or no_windy:
            logger.warning("Plan B：%s 改用不含%s的模型", group_days,
                           "、".join(n for n, f in (("氣象", no_weather), ("Windy", no_windy)) if f))
        with temporary_settings(**overrides):
            group_curves, group_intended, group_requested = _predict_curves(origin, days)
        for day in group_days:
            curves[day], intended[day] = group_curves[day], group_intended[day]
            requested[day] = group_requested[day]
    return curves, intended, requested


def predict_window(origin: dt.date) -> pl.DataFrame:
    """以提交流程預測一個有實際值的歷史窗，回傳與回測紀錄 ``curves.csv`` 相同格式的表。

    Args:
        origin: 起點日；預測 ``origin`` 次日起的 ``settings.PREDICT_HORIZON_DAYS`` 天。

    Returns:
        pl.DataFrame: ``origin, ts, predicted, actual``，每天 144 列。
    """
    days = [origin + dt.timedelta(days=k) for k in range(1, settings.PREDICT_HORIZON_DAYS + 1)]
    curves, _, _ = predict_days(origin, days)
    predicted = pl.concat([
        pl.DataFrame({
            "ts": pl.datetime_range(dt.datetime.combine(day, dt.time(0)),
                                    dt.datetime.combine(day, dt.time(23, 50)), "10m", eager=True),
            "predicted": curves[day],
        })
        for day in days
    ])
    clean = pl.read_parquet(paths.CLEAN_LOAD_FILE).select("ts", pl.col("Load_MW").alias("actual"))
    return (predicted.with_columns(pl.col("ts").cast(clean.schema["ts"]))
            .join(clean, on="ts", how="left")
            .select(pl.lit(origin).alias("origin"), "ts", "predicted", "actual"))


@contextlib.contextmanager
def temporary_settings(**overrides):
    """暫時改動 ``settings``，離開時一律還原（Plan B、回測的設定覆寫、演練共用）。"""
    saved = {name: getattr(settings, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(settings, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(settings, name, value)


def _predict_curves(origin: dt.date, days: list[dt.date]) -> tuple[dict, dict, dict]:
    """:func:`predict_days` 的本體：依目前的 settings 訓練、預測 6 目標、合成曲線。"""
    from src.models import curve

    daily, clean, attributes, target_fn, _, _ = prepare_context(extra_days=days, weather_mode="honest")
    history = daily.filter(pl.col("date") <= origin)
    lookup = {
        r["date"]: (r["price_daytype"], r["is_summer"])
        for r in attributes.select("date", "price_daytype", "is_summer").iter_rows(named=True)
    }
    raw_targets = target_fn(history, tuple(days))

    curves, intended, requested = {}, {}, {}
    for index, day in enumerate(days):
        wanted = curve.DayTargets(
            float(raw_targets["p_day"][index]), int(raw_targets["t_day"][index]),
            float(raw_targets["p_night"][index]), int(raw_targets["t_night"][index]),
            float(raw_targets["ramp_up"][index]), float(raw_targets["ramp_down"][index]),
        )
        shape = curve.shape_template(clean.filter(pl.col("date") <= origin), *lookup[day])
        values, realised, notes = curve.synthesize_from_targets(
            wanted, history, *lookup[day], shape=shape
        )
        curves[day] = values
        intended[day] = realised
        requested[day] = wanted
        if notes:
            logger.warning("%s 的目標經過調整：%s", day, notes)
    return curves, intended, requested


def _recording(predict_fn, store: list, *tagged: list):
    """包住預測函式，記下每折的逐日預測（不改變回傳值）。

    ``tagged`` 是曲線預測器附加曲線、原始目標的串列；本折新增的項目補上起點欄。
    """
    def wrapped(history: pl.DataFrame, target_dates: tuple[dt.date, ...]) -> pl.DataFrame:
        before = [len(items) for items in tagged]
        result = predict_fn(history, target_dates)
        origin = history["date"].max()
        store.append(result.with_columns(
            pl.Series("date", sorted(target_dates)),
            pl.lit(origin).alias("origin"),
        ))
        for items, start in zip(tagged, before):
            for index in range(start, len(items)):
                items[index] = items[index].with_columns(pl.lit(origin).alias("origin"))
        return result
    return wrapped


def evaluate(
    weather_mode: str = "honest",
    fold_indices: list[int] | None = None,
    output_dir: Path | None = None,
    folds: list | None = None,
    observed_lag_days: int = 0,
) -> dict:
    """對現行模型跑 walk-forward 評估（6 目標 → 432 點曲線 → 由曲線重新推導 6 目標後評分）。

    Args:
        weather_mode: 見 :data:`WEATHER_MODES`。各模式用同一組折，可直接配對比較。
        fold_indices: 只跑這些折（依 ``cv.make_folds`` 的順序），供快速回歸使用。
        output_dir: 逐折分數與逐日預測的輸出目錄，None 時採 ``paths.EVALUATION_DIR``。
        folds: 指定的回測窗（``src.evaluation.backtest.windows``），None 時採 60 折。
        observed_lag_days: 模擬 CODiS 觀測只到起點前第幾天（0 = 到起點日）；
            大於 0 時每折重建一次資料，只支援 honest 模式。

    Returns:
        dict: ``table``（逐折分數）、``predictions``（逐日 6 目標，由合成曲線推導）、
            ``curves``（逐窗 432 點預測與實際值）、``raw_targets``（模型的原始 6 目標，
            合成修補之前；後兩者在遞迴對照與觀測延遲模擬時為 None）、``summary``、
            ``specials``（子集）、``stats``（曲線修補次數）、``folds``。

    Raises:
        ValueError: ``weather_mode`` 不支援，或 ``observed_lag_days`` 搭配 honest 以外的模式。
    """
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"未知的 weather_mode：{weather_mode!r}（支援 {WEATHER_MODES}）")
    if observed_lag_days and weather_mode != "honest":
        raise ValueError("observed_lag_days 只支援 honest 模式（歷史用觀測的情境）")
    source = "accuweather" if weather_mode == "forecast" else settings.WEATHER_SOURCE
    with temporary_settings(WEATHER_SOURCE=source):
        return _evaluate(weather_mode, fold_indices, output_dir, folds, observed_lag_days)


def _evaluate(weather_mode: str, fold_indices: list[int] | None, output_dir: Path | None,
              folds: list | None, observed_lag_days: int) -> dict:
    """:func:`evaluate` 的本體。"""
    from src.evaluation import cv, metrics
    from src.features.targets import TARGET_NAMES
    from src.models import pipeline

    daily, clean, attributes, target_fn, _, _ = prepare_context(weather_mode=weather_mode)
    folds = folds or cv.make_folds(daily["date"])
    if fold_indices is not None:
        folds = [folds[i] for i in fold_indices]
    logger.info("評估 %d 折：氣象模式 %s（來源 %s），模型 %s",
                len(folds), weather_mode, settings.WEATHER_SOURCE, settings.MODEL_VARIANT)

    stats: dict = {}
    predictions: list = []
    curves: list = []
    raw: list = []
    if observed_lag_days:
        table = _run_with_observation_lag(folds, observed_lag_days, stats, predictions)
    elif settings.MODEL_VARIANT == "recursive_ar":
        from src.models import recursive

        table = cv.run_cv(daily, folds, _recording(recursive.make_recursive_predictor(clean), predictions))
    else:
        table = cv.run_cv(daily, folds, _recording(
            pipeline.make_curve_predictor(attributes, target_fn, stats, clean, curves, raw), predictions,
            curves, raw))
    summary = cv.summarize_cv(table)

    breakdown = metrics.ScoreBreakdown(*[float(table[c].mean()) for c in (
        "s_peak_mw", "s_peak_time", "s_ramp_up", "s_ramp_down", "s_under_penalty", "total_score")])
    logger.warning(
        "total_score = %.5f（標準誤 %.5f），最壞折 %.5f（%s）［氣象模式 %s］\n%s",
        summary["mean"], summary["stderr"], summary["worst"], summary["worst_origin"],
        weather_mode, metrics.contribution_breakdown(breakdown),
    )

    # 子集直接篩選已算好的逐折表，不重跑。
    specials = {}
    for name, subset in {
        "夏月末期": cv.special_late_summer_folds(folds),
        "含週六": cv.special_saturday_folds(folds),
        "週四五六": cv.special_thu_fri_sat_folds(folds),
    }.items():
        if len(subset) < 2:
            continue   # 標準誤至少要 2 折；只跑部分折時會發生
        special = cv.summarize_cv(table.filter(pl.col("origin").is_in([f.origin for f in subset])))
        specials[name] = special
        logger.info("子集［%s］%d 折：平均 %.5f，最壞 %.5f（%s）", name, special["n_folds"],
                    special["mean"], special["worst"], special["worst_origin"])
    logger.info("曲線修補 %d / %d 天", stats.get("n_repaired", 0), stats.get("n_days", 0))

    predicted = pl.concat(predictions)
    output_dir = output_dir or paths.EVALUATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(output_dir / f"folds_{weather_mode}.csv")
    predicted.write_csv(output_dir / f"predictions_{weather_mode}.csv")
    curve_table = None
    if curves:
        curve_table = pl.concat(curves).join(
            clean.select("ts", pl.col("Load_MW").alias("actual")), on="ts", how="left"
        ).select("origin", "ts", "predicted", "actual")
        curve_table.write_csv(output_dir / f"curves_{weather_mode}.csv")
    raw_table = None
    if raw:
        raw_table = pl.concat(raw).select("origin", "date", *TARGET_NAMES).sort("origin", "date")
        raw_table.write_csv(output_dir / f"raw_targets_{weather_mode}.csv")
    return {"table": table, "predictions": predicted, "curves": curve_table, "raw_targets": raw_table,
            "summary": summary,
            "specials": specials, "stats": stats, "folds": folds}


def _run_with_observation_lag(folds: list, lag_days: int, stats: dict, predictions: list):
    """逐折模擬「CODiS 只到起點前 ``lag_days`` 天」：把那幾天標為取不到觀測後重建資料。

    取不到的日子由 ``external._load_weather_codis`` 以校正後的預報補上；
    時刻的分箱門檻、訓練列與目標日校正都只看得到真正取得到的觀測。
    """
    from src.evaluation import cv
    from src.models import pipeline

    tables = []
    for fold in folds:
        unavailable = tuple(fold.origin - dt.timedelta(days=k) for k in range(lag_days))
        with temporary_settings(CODIS_UNAVAILABLE_DATES=unavailable):
            daily, _, attributes, target_fn, _, _ = prepare_context(weather_mode="honest")
            tables.append(cv.run_cv(daily, [fold], _recording(
                pipeline.make_curve_predictor(attributes, target_fn, stats), predictions)))
    return pl.concat(tables)
