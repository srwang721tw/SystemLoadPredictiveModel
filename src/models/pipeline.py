"""把時刻與量值模型組成回測與提交共用的預測函式。

介面一律是 ``(history, target_dates) -> 每個目標日一列的 6 項預測``（``src/evaluation/cv.py``）。
每一次呼叫都只用 ``history``（起點以前）的資料重新訓練——walk-forward 的每一折
都是一次完整的實戰模擬，沿用前一折的模型會讓後面的折看到未來。

組成：

- **時刻**（``t_day``、``t_night``，佔總分約 93%）：條件經驗分布 + 近期同星期分布
  + 學習式多類別模型在 PMF 層級修正，最後以貝氏決策取期望損失最小的格點
  （:func:`make_ensemble_timing_predictor`）。
- **量值**（``p_day``、``p_night``、``ramp_up``、``ramp_down``）：LightGBM 分位數迴歸，
  τ 由評分公式推導（:func:`make_magnitude_predictor`）。
- **曲線**：6 個目標合成 144 點曲線後，再由曲線重新推導 6 個目標
  （:func:`make_curve_predictor`），回測評的就是提交檔真正呈現的東西。
"""

from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import polars as pl

from config import settings
from src.features import builder, calendar
from src.features.targets import TARGET_NAMES
from src.logging_setup import get_logger
from src.models import baseline, curve, decision, quantile, timing

TEMPERATURE_COLUMN = "tmax"
"""``attributes`` 中承載「白天最高溫（五站取最大）」的欄名。

單獨命名而非寫死字串：它同時出現在 lookup、history 的 join 與分箱三處，
拼錯任何一處都會靜默退回「不使用氣溫條件」而不報錯。
"""

logger = get_logger(__name__)


def ensemble_seeds(n: int) -> list[int]:
    """種子集成用的種子：``RANDOM_SEED + SEED_STRIDE × k``（k = 0 … n−1）。

    以 ``RANDOM_SEED`` 為起點，換 ``RANDOM_SEED`` 做多種子確認時整組種子一起換。
    """
    return [settings.RANDOM_SEED + settings.SEED_STRIDE * k for k in range(n)]


def make_magnitude_predictor(
    daily: pl.DataFrame,
    calendar_df: pl.DataFrame,
    rules: dict,
    timing_predictor,
    in_ratio_space: bool = False,
    seed: int | None = None,
    excluded_dates: tuple[date, ...] | None = None,
    target_weather_fn=None,
):
    """建立量值路徑的預測函式：LightGBM 分位數迴歸，時刻沿用 ``timing_predictor`` 的輸出。

    Args:
        daily: 完整每日表。
        calendar_df: 電價日曆表。
        rules: 電價時段規則。
        timing_predictor: 時刻預測函式，其輸出（含 Baseline 1 的量值）作為起點。
        in_ratio_space: 是否在「相對近期基準的比值」空間建模。現行為絕對值空間：
            只預測 1–3 天，近期水準永遠在訓練範圍內，GBDT 無法外插的顧慮不成立，
            而絕對值空間在回測中較好。
        seed: 隨機種子；None 時用 ``MAGNITUDE_N_SEEDS`` 個種子取平均（預設 1 個）。
        excluded_dates: 不得進入**訓練集**的日期（事件日）。只排除訓練、不排除預測。
        target_weather_fn: ``fn(origin, target_dates)``，回傳目標日要用的氣象
            （honest 模式：逐起點校正後的預報）。None 時目標列沿用特徵矩陣原值。

    Returns:
        PredictFn: 符合 ``src.evaluation.cv.PredictFn`` 的預測函式。
    """
    announcements = calendar.typhoon_announcements()

    def predict(history: pl.DataFrame, target_dates: tuple[date, ...]) -> pl.DataFrame:
        origin = history["date"].max()
        override = target_weather_fn(origin, target_dates) if target_weather_fn else None
        out = timing_predictor(history, target_dates).to_dicts()
        ordered = sorted(target_dates)
        scope = daily.filter(pl.col("date") <= max(target_dates))
        horizon_by_date = {d: (d - origin).days for d in target_dates}

        for horizon in sorted(set(horizon_by_date.values())):
            features = builder.build_features(
                scope, calendar_df, rules, horizon,
                pending_weather_dates=tuple(target_dates) if target_weather_fn else (),
            )
            names = builder.feature_names(features)
            train = features.filter((pl.col("date") <= origin) & pl.col("base_reliable"))
            if excluded_dates:
                # 事件日的標籤本身被污染（例如地震當天清晨爬升被砍半），不拿來訓練。
                train = train.filter(~pl.col("date").is_in(list(excluded_dates)))
            dates_at_h = [d for d, h in horizon_by_date.items() if h == horizon]
            rows = features.filter(pl.col("date").is_in(dates_at_h)).sort("date")
            if override is not None:
                rows = _override_weather(rows, override)
            rows = calendar.mask_unannounced_typhoon(rows, origin, announcements)

            for target in settings.MAGNITUDE_TARGETS:
                base = rows[f"{target}_base"]
                if base.null_count() or (base <= 0).any():
                    # 基準不可靠時保留時刻預測函式給的量值（Baseline 1）。
                    logger.debug("%s 的 base 不可靠，沿用基線量值", target)
                    continue
                label = f"{target}_ratio" if in_ratio_space else target
                fitted = train.drop_nulls(label)
                seeds = [seed] if seed is not None else ensemble_seeds(settings.MAGNITUDE_N_SEEDS)
                values = np.mean([
                    quantile.predict(
                        quantile.fit(fitted, fitted[label], target, names, in_ratio_space, seed=one),
                        rows, base if in_ratio_space else None,
                    )
                    for one in seeds
                ], axis=0)
                for row_date, value in zip(rows["date"].to_list(), values.tolist(), strict=True):
                    out[ordered.index(row_date)][target] = float(value)

        return pl.DataFrame(out).select(TARGET_NAMES)

    return predict


def _override_weather(rows: pl.DataFrame, override: pl.DataFrame) -> pl.DataFrame:
    """把目標列的 ``w_*`` 氣象欄換成 ``override`` 的值，其餘欄位不動。

    提交當天歷史氣象是 CODiS 觀測，目標日卻只有預報；回測若讓目標日也用觀測，
    量到的是提交時達不到的分數。本函式只換目標列，訓練列一律不碰。
    特徵矩陣內沒有由 ``w_*`` 衍生的欄位，換掉這些欄即完整。

    Args:
        rows: 特徵矩陣中目標日那幾列。
        override: 含 ``date`` 與要換上的 ``w_*`` 欄。

    Returns:
        pl.DataFrame: 欄位與順序同 ``rows``（LightGBM 的 ``feature_fraction`` 依欄位
            索引抽樣，換序會改變結果）。

    Raises:
        ValueError: ``override`` 缺少某個目標日（不可靜默保留觀測值）。
    """
    columns = [c for c in override.columns if c.startswith("w_") and c in rows.columns]
    missing = set(rows["date"].to_list()) - set(override["date"].to_list())
    if missing:
        raise ValueError(f"目標日氣象缺少 {sorted(missing)}，無法覆寫")
    return (
        rows.drop(columns)
        .join(override.select("date", *columns), on="date", how="left")
        .select(rows.columns)
    )


def _blend_learned_pmf(
    pmf: np.ndarray,
    classes: list[int],
    target: str,
    learned_features: dict[int, pl.DataFrame],
    origin: date,
    target_date: date,
    mix: float,
    override: pl.DataFrame | None = None,
    typhoon_announcements: dict | None = None,
) -> np.ndarray:
    """用學習式多類別模型的 PMF 修正經驗分布。

    只用 ``origin`` 以前的資料訓練，每個目標日各訓練一次。學習式模型的類別集合
    比完整格點窄（罕見格點被合併），故先攤回完整格點再混合。

    Args:
        pmf: 經驗分布，形狀 ``(1, n_classes)``。
        classes: 完整名目格點。
        target: ``"t_day"`` 或 ``"t_night"``。
        learned_features: ``{horizon: 特徵矩陣}``。
        origin: 預測起點日。
        target_date: 目標日。
        mix: 學習式的權重。
        override: 目標日要換上的氣象，None 時不換。
        typhoon_announcements: 颱風公告時間；目標列只保留作業時點以前已公告的颱風資訊。

    Returns:
        np.ndarray: 混合後的 PMF，形狀與輸入相同。
    """
    horizon = (target_date - origin).days
    features = learned_features.get(horizon)
    if features is None:
        logger.warning("horizon=%d 沒有特徵矩陣，略過學習式修正", horizon)
        return pmf

    names = builder.feature_names(features)
    train = features.filter(pl.col("date") <= origin).drop_nulls([target])
    row = features.filter(pl.col("date") == target_date)
    if not train.height or not row.height:
        return pmf
    if override is not None:
        row = _override_weather(row, override)
    row = calendar.mask_unannounced_typhoon(row, origin, typhoon_announcements)

    previous = settings.TIMING_MIN_CLASS_COUNT
    settings.TIMING_MIN_CLASS_COUNT = settings.TIMING_LEARNED_MIN_CLASS_COUNT
    try:
        # 各種子的類別集合相同（由訓練資料決定），PMF 可直接平均；預設只有 1 個種子。
        models = [
            timing.fit(train, train[target], target, names, seed=seed,
                       num_rounds=settings.TIMING_LEARNED_ROUNDS)
            for seed in ensemble_seeds(settings.TIMING_LEARNED_N_SEEDS)
        ]
        learned = np.mean([timing.predict_pmf(m, row) for m in models], axis=0)
    finally:
        settings.TIMING_MIN_CLASS_COUNT = previous

    spread = np.zeros(len(classes))
    index = {minute: i for i, minute in enumerate(classes)}
    for position, minute in enumerate(models[0].classes):
        slot = index.get(int(minute))
        if slot is not None:
            spread[slot] = learned[0, position]
    if spread.sum() <= 0:
        return pmf

    blended = mix * (spread / spread.sum()) + (1.0 - mix) * pmf.ravel()
    return (blended / blended.sum())[None, :]


def make_curve_predictor(
    attributes: pl.DataFrame,
    target_predictor,
    stats: dict | None = None,
    clean: pl.DataFrame | None = None,
    curves: list | None = None,
    raw: list | None = None,
):
    """把 6 目標預測器包成「經過曲線」的預測器。

    流程：預測 6 目標 → 合成 144 點曲線 → 由曲線重新推導 6 目標 → 回傳。
    回傳的是**從曲線推導出來的**目標，確保回測評的就是主辦單位從提交檔算出來的東西；
    若合成有損，分數會立刻反映出來。

    Args:
        attributes: 每日屬性表，須含 ``date``、``price_daytype``、``is_summer``，且含目標日。
        target_predictor: 6 目標預測函式。
        stats: 選填的可變 dict，累計曲線修補次數。
        clean: 10 分鐘序列（含 ``date``、``mod``、``price_daytype``、``is_summer``）。給定時，
            曲線形狀取自起點以前的同組日（``curve.shape_template``），與提交流程相同。
        curves: 選填的串列，每個目標日附加一張 ``date, ts, predicted`` 的 144 列表。
        raw: 選填的串列，每個目標日附加一列模型的原始 6 目標（合成修補之前）。

    Returns:
        PredictFn: 符合 ``src.evaluation.cv.PredictFn`` 的預測函式。
    """
    lookup = {
        row["date"]: (row["price_daytype"], row["is_summer"])
        for row in attributes.select("date", "price_daytype", "is_summer").iter_rows(named=True)
    }
    if stats is not None:
        stats.setdefault("n_days", 0)
        stats.setdefault("n_repaired", 0)
        stats.setdefault("notes", [])

    def predict(history: pl.DataFrame, target_dates: tuple[date, ...]) -> pl.DataFrame:
        predicted = target_predictor(history, target_dates)
        rows = []
        for index, target_date in enumerate(sorted(target_dates)):
            price_daytype, is_summer = lookup[target_date]
            wanted = curve.DayTargets(
                float(predicted["p_day"][index]), int(predicted["t_day"][index]),
                float(predicted["p_night"][index]), int(predicted["t_night"][index]),
                float(predicted["ramp_up"][index]), float(predicted["ramp_down"][index]),
            )
            if raw is not None:
                raw.append(pl.DataFrame([{"date": target_date,
                                           **{n: getattr(wanted, n) for n in TARGET_NAMES}}]))
            shape = None
            if clean is not None:
                shape = curve.shape_template(
                    clean.filter(pl.col("date") <= history["date"].max()), price_daytype, is_summer)
            values, realised, notes = curve.synthesize_from_targets(
                wanted, history, price_daytype, is_summer, shape=shape
            )
            if curves is not None:
                curves.append(pl.DataFrame({
                    "date": [target_date] * len(values),
                    "ts": pl.datetime_range(
                        datetime.combine(target_date, time(0)), datetime.combine(target_date, time(23, 50)),
                        "10m", eager=True),
                    "predicted": values,
                }))
            if stats is not None:
                stats["n_days"] += 1
                repair = [n for n in notes if n.startswith("ramp")]
                if repair:
                    stats["n_repaired"] += 1
                    stats["notes"].append((target_date, repair))
            rows.append(curve.verify_day(values, realised))
        return pl.DataFrame(rows).select(TARGET_NAMES)

    return predict


def make_ensemble_timing_predictor(
    attributes: pl.DataFrame,
    clean_load: pl.DataFrame,
    weights: dict[str, float],
    n_weeks: int = 4,
    learned_features: dict[int, pl.DataFrame] | None = None,
    target_weather_fn=None,
):
    """時刻預測：條件經驗分布、近期同星期分布、學習式修正，在 PMF 層級組合後做貝氏決策。

    1. **條件經驗分布**：歷史中與目標日同一組（日別 × 夏月 × 星期 × 白天最高溫 2 箱）
       的時刻分布，階層收縮到較粗的分組（``TIMING_SHRINKAGE_ALPHA``）。
       加入星期讓最大的一組從 378 天拆成最多 79 天——關鍵在組別平衡而非組數；
       年內位置、月份等細分都因把組切得太碎而失敗。
    2. **近期同星期分布**：最近 ``n_weeks`` 個同星期日的時刻，只用在 ``t_night``
       （``weights``）：夜尖峰高度集中在 17:10，時近性有價值；日尖峰是三峰分布，近 4 天太雜。
    3. **學習式修正**：LightGBM 多類別模型的 PMF 以 ``TIMING_LEARNED_MIX`` 的權重混入。
       單獨使用多類別模型會慘敗（37 類 × 約 850 列撐不起分類器），混合後則穩定改善。

    所有組合都在 PMF 層級進行，最後才做一次貝氏決策——平均兩個決策時刻是錯的。
    分箱門檻逐折由 history 算；用全期資料算就是洩漏。

    Args:
        attributes: 每日屬性表（日別、夏月、星期，及選填的 ``tmax``）。
        clean_load: 10 分鐘序列（保留介面；時刻取自 ``history`` 的標籤）。
        weights: ``{目標: 條件經驗分布的權重}``，1.0 代表不混近期分布。
        n_weeks: 近期分布回溯幾個同星期日。
        learned_features: ``{horizon: 特徵矩陣}``，None 時不啟用學習式修正。
        target_weather_fn: ``fn(origin, target_dates)``，回傳目標日的氣象與 ``tmax``；
            目標日的分箱與學習式層改用它，分箱門檻與歷史仍用觀測。

    Returns:
        PredictFn: 符合 ``src.evaluation.cv.PredictFn`` 的預測函式。

    Note:
        學習式模型有隨機種子（固定為 ``RANDOM_SEED``），單次執行可完全重現；
        但只換種子，60 折總分的全距約 0.019——比較變體時須以多個種子確認。
    """
    base_condition = tuple(settings.TIMING_CONDITION_COLUMNS)
    n_bins = settings.TIMING_TEMPERATURE_BINS
    use_bins = n_bins > 1 and TEMPERATURE_COLUMN in attributes.columns
    if n_bins > 1 and not use_bins:
        logger.warning("已要求氣溫分箱，但 attributes 沒有 %s 欄——退回不使用氣溫條件",
                       TEMPERATURE_COLUMN)

    mix = settings.TIMING_LEARNED_MIX
    condition = base_condition + (("tbin",) if use_bins else ())
    columns = base_condition + ((TEMPERATURE_COLUMN,) if use_bins else ())
    lookup = {row["date"]: row for row in attributes.select("date", *columns).iter_rows(named=True)}
    # 由粗到細逐層，最細一層即完整的條件變數組合。
    levels = tuple(condition[: i + 1] for i in range(len(condition)))
    announcements = calendar.typhoon_announcements()

    def predict(history: pl.DataFrame, target_dates: tuple[date, ...]) -> pl.DataFrame:
        out = baseline.baseline1_same_weekday_median(history, target_dates).to_dicts()
        enriched = history
        for column in columns:
            if column not in enriched.columns:
                enriched = enriched.join(attributes.select("date", column), on="date", how="left")
        origin = history["date"].max()
        override = target_weather_fn(origin, target_dates) if target_weather_fn else None

        edges: list[float] = []
        if use_bins:
            # 門檻只由 history 算，目標日的氣溫不會透過門檻位置洩漏進來。
            observed = enriched[TEMPERATURE_COLUMN].drop_nulls().to_numpy()
            edges = list(np.quantile(observed, [k / n_bins for k in range(1, n_bins)]))
            enriched = enriched.with_columns(
                pl.col(TEMPERATURE_COLUMN).cut(edges, labels=[str(i) for i in range(n_bins)])
                .cast(pl.Utf8).alias("tbin")
            )

        for index, target_date in enumerate(sorted(target_dates)):
            row = lookup[target_date]
            values = tuple(row[c] for c in base_condition)
            if use_bins:
                temperature = (
                    row[TEMPERATURE_COLUMN] if override is None
                    else override.filter(pl.col("date") == target_date)[TEMPERATURE_COLUMN].item()
                )
                if temperature is None:
                    raise ValueError(f"{target_date} 缺氣溫，無法決定分箱（目標日須有預報值）")
                values = values + (str(int(np.searchsorted(edges, temperature, side="right"))),)
            recent = (
                enriched.filter(
                    (pl.col("date") <= origin)
                    & (pl.col("date").dt.weekday() == target_date.isoweekday())
                )
                .sort("date")
                .tail(n_weeks)
            )
            for target in settings.TIMING_TARGETS:
                weight = weights.get(target, 1.0)
                ours, classes = timing.hierarchical_pmf(
                    enriched, target, levels,
                    tuple(values[: i + 1] for i in range(len(condition))),
                    settings.TIMING_SHRINKAGE_ALPHA[target],
                )
                if weight >= 1.0:
                    pmf = ours
                else:
                    naive, _ = timing.grid_pmf(
                        [int(v) for v in recent[target].drop_nulls().to_list()], target,
                        prior_weight=settings.TIMING_NAIVE_PRIOR_WEIGHT,
                    )
                    pmf = timing.mix_pmf(ours, naive, weight)
                if mix > 0 and learned_features:
                    pmf = _blend_learned_pmf(
                        pmf, classes, target, learned_features,
                        origin, target_date, mix, override, announcements,
                    )
                out[index][target] = float(decision.bayes_decision(pmf, classes, target)[0])

        return pl.DataFrame(out).select(TARGET_NAMES)

    return predict
