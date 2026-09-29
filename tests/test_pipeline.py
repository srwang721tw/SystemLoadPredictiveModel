"""預測管線與量值分位數迴歸。

管線是唯一同時碰到歷史與目標日的地方，核心是證明預測函式看不到目標日的標籤。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from config import paths, settings
from src.features.targets import TARGET_NAMES, to_minutes
from src.models import pipeline, quantile

needs_processed = pytest.mark.skipif(
    not (paths.TARGETS_FILE.exists() and paths.SPECIAL_DAYS_FILE.exists()),
    reason="需要先重建 data/processed/（notebook 01，需 Accuweather 年度檔）")


# =============================================================================
# 預測管線
# =============================================================================


START = dt.date(2025, 1, 1)


N_DAYS = 400


def synthetic_daily() -> pl.DataFrame:
    """構造一張假 daily-row 表，時刻與日別強相關。"""
    rng = np.random.default_rng(0)
    dates = [START + dt.timedelta(days=i) for i in range(N_DAYS)]
    daytypes = [
        "平日" if d.isoweekday() <= 5 else ("週六" if d.isoweekday() == 6 else "週日及離峰日")
        for d in dates
    ]
    # 平日尖峰 14:00、週六 15:00、週日 16:00，各加一點抖動。
    base = {"平日": "14:00", "週六": "15:00", "週日及離峰日": "16:00"}
    t_day = [
        to_minutes(base[t]) + int(rng.integers(-2, 3)) * 10 for t in daytypes
    ]
    return pl.DataFrame(
        {
            "date": dates,
            "price_daytype": daytypes,
            "is_summer": [5 <= d.month <= 10 for d in dates],
            "p_day": rng.normal(30000, 500, N_DAYS),
            "t_day": [float(v) for v in t_day],
            "p_night": rng.normal(28000, 500, N_DAYS),
            "t_night": [float(to_minutes("18:00"))] * N_DAYS,
            "ramp_up": rng.normal(1000, 50, N_DAYS),
            "ramp_down": rng.normal(800, 40, N_DAYS),
        }
    )


def attributes_of(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.select("date", "price_daytype", "is_summer").with_columns(
        pl.col("date").dt.weekday().alias("weekday"))


def timing_predictor(attributes: pl.DataFrame):
    """上線用的時刻預測函式（不含學習式修正與氣溫分箱，只測條件經驗分布的部分）。"""
    return pipeline.make_ensemble_timing_predictor(
        attributes, pl.DataFrame(), {"t_day": 1.0, "t_night": 1.0})


class TestTimingPredictorNoLeakage:
    """預測函式不得看到目標日的標籤。"""

    def test_perturbing_target_day_label_changes_nothing(self) -> None:
        daily = synthetic_daily()
        attributes = attributes_of(daily)
        origin = START + dt.timedelta(days=300)
        targets = tuple(origin + dt.timedelta(days=h) for h in (1, 2, 3))
        history = daily.filter(pl.col("date") <= origin)

        predictor = timing_predictor(attributes)
        clean = predictor(history, targets)

        # 把目標日的標籤改成極端值——預測結果必須完全不變。
        poisoned = daily.with_columns(
            pl.when(pl.col("date").is_in(list(targets)))
            .then(pl.lit(660.0))
            .otherwise(pl.col("t_day"))
            .alias("t_day")
        )
        poisoned_history = poisoned.filter(pl.col("date") <= origin)
        assert clean.equals(predictor(poisoned_history, targets))

    def test_uses_only_rows_up_to_origin(self) -> None:
        daily = synthetic_daily()
        origin = START + dt.timedelta(days=300)
        targets = (origin + dt.timedelta(days=1),)
        predictor = timing_predictor(attributes_of(daily))

        short = predictor(daily.filter(pl.col("date") <= origin), targets)
        # 多給 50 天未來資料，結果應該改變——證明歷史長度真的有作用，
        # 也證明前一個測試不是因為預測函式忽略了 history 才通過。
        long = predictor(
            daily.filter(pl.col("date") <= origin + dt.timedelta(days=50)), targets
        )
        assert not short.equals(long)


class TestTimingPredictorBehaviour:
    """分組與輸出格式。"""

    def test_grouping_is_actually_applied(self) -> None:
        # 平日 14:00、週六 15:00、週日 16:00 —— 三種日別必須給出不同預測。
        daily = synthetic_daily()
        predictor = timing_predictor(attributes_of(daily))
        origin = START + dt.timedelta(days=350)
        history = daily.filter(pl.col("date") <= origin)

        chosen = {}
        for offset in range(1, 8):
            target_date = origin + dt.timedelta(days=offset)
            daytype = daily.filter(pl.col("date") == target_date)["price_daytype"][0]
            chosen[daytype] = predictor(history, (target_date,))["t_day"][0]
        assert len(set(chosen.values())) == 3, chosen

    def test_returns_all_six_targets_in_order(self) -> None:
        daily = synthetic_daily()
        predictor = timing_predictor(attributes_of(daily))
        origin = START + dt.timedelta(days=300)
        targets = tuple(origin + dt.timedelta(days=h) for h in (1, 2, 3))
        out = predictor(daily.filter(pl.col("date") <= origin), targets)
        assert out.columns == list(TARGET_NAMES)
        assert out.height == 3

    def test_unknown_target_date_raises(self) -> None:
        daily = synthetic_daily()
        predictor = timing_predictor(attributes_of(daily))
        far_future = START + dt.timedelta(days=9999)
        with pytest.raises(KeyError):
            predictor(daily, (far_future,))

    def test_magnitude_targets_come_from_baseline(self) -> None:
        # 時刻預測函式只覆寫時刻；量值必須與 Baseline 1 完全相同。
        from src.models import baseline

        daily = synthetic_daily()
        origin = START + dt.timedelta(days=300)
        targets = tuple(origin + dt.timedelta(days=h) for h in (1, 2, 3))
        history = daily.filter(pl.col("date") <= origin)

        predictor = timing_predictor(attributes_of(daily))
        out = predictor(history, targets)
        reference = baseline.baseline1_same_weekday_median(history, targets)
        for column in ("p_day", "p_night", "ramp_up", "ramp_down"):
            assert out[column].to_list() == pytest.approx(reference[column].to_list())


@needs_processed
class TestFeatureLeakage:
    """通則性防線：目標日當天的任何量都不得成為該日的特徵。

    這比逐欄列黑名單可靠：`compute_targets` 新增的欄位（例如曲線合成用的
    `load_start` / `load_min` / `load_end`）若忘了加進 `NON_FEATURE_COLUMNS`，
    就會直接變成特徵，回測分數假性改善且看起來完全合理，這條測試會指出是哪一欄。
    """

    @staticmethod
    def _context():
        from src.data import external

        daily = pl.read_parquet(paths.TARGETS_FILE)
        return daily, external.load_calendar(), external.load_price_period_rules()

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_perturbing_a_days_own_values_changes_none_of_its_features(
        self, horizon: int
    ) -> None:
        from src.features import builder

        daily, calendar_df, rules = self._context()
        probe = daily["date"][daily.height // 2]

        clean = builder.build_features(daily, calendar_df, rules, horizon)
        names = builder.feature_names(clean)

        # 把探針日自己的所有標籤與曲線水準改成極端值。
        perturbed_columns = [
            c for c in ("p_day", "p_night", "ramp_up", "ramp_down",
                        "load_start", "load_min", "load_end")
            if c in daily.columns
        ]
        poisoned = daily.with_columns(
            [
                pl.when(pl.col("date") == probe)
                .then(pl.col(c) * 3.0)
                .otherwise(pl.col(c))
                .alias(c)
                for c in perturbed_columns
            ]
        )
        after = builder.build_features(poisoned, calendar_df, rules, horizon)

        before_row = clean.filter(pl.col("date") == probe).select(names)
        after_row = after.filter(pl.col("date") == probe).select(names)
        leaked = [c for c in names if not before_row[c].equals(after_row[c])]
        assert not leaked, (
            f"horizon={horizon}：擾動 {probe} 自身的值之後，該日的下列特徵改變了"
            f"（＝當日資訊洩漏進特徵）：{leaked}"
        )

    def test_curve_level_columns_are_excluded(self) -> None:
        # 這三欄只供合成曲線的自由水準使用，絕不可當特徵。
        from src.features import builder

        daily, calendar_df, rules = self._context()
        names = builder.feature_names(builder.build_features(daily, calendar_df, rules, 1))
        assert not [n for n in names if n.startswith("load_")]


@needs_processed
class TestProductionFeatureSet:
    """上線路徑的特徵集必須與明列的清單一致。

    洩漏測試（``TestFeatureLeakage``）擋的是「特徵含當日資訊」，
    擋不住「非預期的欄位悄悄成為特徵」。例如把 `weekday` 再 join 進本來就有這一欄的表，
    polars 會產生 `weekday_right`，它不在黑名單裡，就會成為特徵。

    本測試不判斷某欄該不該存在，只要求任何增減都必須是人為的：
    特徵集一變，測試就失敗，改清單的人得說明理由。
    """

    EXPECTED_EXTRA_FROM_PREPARE_CONTEXT: set[str] = set()
    """上線路徑比直接呼叫 ``builder.build_features`` 多出來的欄位。

    應為空集合——兩條路徑必須產生完全相同的特徵集。
    若日後這裡又需要加東西，那就是有欄位再次撞名或悄悄溜進來了，
    必須先查清楚成因，不可為了讓測試通過而直接把新欄位加進來。
    """

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_production_features_match_documented_set(self, horizon: int) -> None:
        from src import workflow
        from src.features import builder

        daily, calendar_df, rules = self._plain_context()
        documented = set(builder.feature_names(
            builder.build_features(daily, calendar_df, rules, horizon)
        ))

        production_daily = workflow.prepare_context()[0]
        produced = set(builder.feature_names(
            builder.build_features(production_daily, calendar_df, rules, horizon)
        ))

        assert produced - documented == self.EXPECTED_EXTRA_FROM_PREPARE_CONTEXT, (
            f"horizon={horizon}：上線路徑的特徵集與文件記載不符。"
            f"多出 {sorted(produced - documented)}，"
            f"預期只多出 {sorted(self.EXPECTED_EXTRA_FROM_PREPARE_CONTEXT)}。"
            "新欄位若是刻意加入，請更新本清單並說明理由。"
        )
        assert not documented - produced, (
            f"horizon={horizon}：上線路徑少了這些特徵：{sorted(documented - produced)}"
        )

    @staticmethod
    def _plain_context():
        from src.data import external

        daily = pl.read_parquet(paths.TARGETS_FILE)
        return daily, external.load_calendar(), external.load_price_period_rules()


class TestLearnedTimingBlend:
    """學習式模型在 PMF 層級修正時刻的經驗分布。

    重點是修正而非取代：單獨使用多類別模型明顯較差，與經驗分布混合後則穩定改善 ``s_peak_time``。
    """

    def test_mix_zero_leaves_pmf_untouched(self) -> None:
        """混合權重 0 必須完全等於只用經驗分布（關掉這個功能就回到純查表）。"""
        import numpy as np

        from src.models import pipeline

        pmf = np.array([[0.2, 0.5, 0.3]])
        out = pipeline._blend_learned_pmf(
            pmf, [660, 670, 680], "t_day", {}, dt.date(2026, 1, 1),
            dt.date(2026, 1, 2), 0.0,
        )
        assert np.allclose(out, pmf)

    def test_missing_horizon_falls_back_to_empirical(self) -> None:
        """沒有對應 horizon 的特徵矩陣時，退回經驗分布而不是拋錯。"""
        import numpy as np

        from src.models import pipeline

        pmf = np.array([[0.2, 0.5, 0.3]])
        out = pipeline._blend_learned_pmf(
            pmf, [660, 670, 680], "t_day", {1: None}, dt.date(2026, 1, 1),
            dt.date(2026, 1, 5), 0.5,
        )
        assert np.allclose(out, pmf)


    def test_seed_is_pinned_for_reproducibility(self) -> None:
        """學習式模型有隨機性，種子必須固定，回測分數才能由指令重現。"""
        import inspect

        from src.models import pipeline

        from config import settings

        source = inspect.getsource(pipeline._blend_learned_pmf)
        assert "ensemble_seeds(settings.TIMING_LEARNED_N_SEEDS)" in source, "學習式模型的種子必須寫死"
        # 預設只有一個種子，且就是 RANDOM_SEED（現行行為）。
        assert pipeline.ensemble_seeds(settings.TIMING_LEARNED_N_SEEDS) == [settings.RANDOM_SEED]
        assert pipeline.ensemble_seeds(3) == [settings.RANDOM_SEED + settings.SEED_STRIDE * k
                                              for k in range(3)]


class TestOverrideWeather:
    """honest 模式：只換目標列的 w_* 欄。"""

    @staticmethod
    def _rows() -> pl.DataFrame:
        return pl.DataFrame({
            "date": [dt.date(2025, 6, 2), dt.date(2025, 6, 3)],
            "w_臺北_day_tmax": [35.5, 30.4],
            "lag_1": [1.0, 2.0],
            "w_臺北_uv_max": [9.1, 8.2],
        })

    def test_replaces_only_weather_columns_and_keeps_order(self) -> None:
        override = pl.DataFrame({
            "date": [dt.date(2025, 6, 3), dt.date(2025, 6, 2)],
            "w_臺北_day_tmax": [24.6, 30.5],
            "w_臺北_uv_max": [3.0, 3.0],
            "tmax": [30.9, 30.9],
        })
        out = pipeline._override_weather(self._rows(), override)
        # 欄序必須不變：feature_fraction 依欄位索引抽樣。
        assert out.columns == self._rows().columns
        assert out["w_臺北_day_tmax"].to_list() == [30.5, 24.6]
        assert out["lag_1"].to_list() == [1.0, 2.0]

    def test_missing_target_day_raises(self) -> None:
        override = pl.DataFrame({
            "date": [dt.date(2025, 6, 2)], "w_臺北_day_tmax": [30.5],
        })
        with pytest.raises(ValueError, match="無法覆寫"):
            pipeline._override_weather(self._rows(), override)


# =============================================================================
# 量值分位數迴歸
# =============================================================================


def synthetic_quantile(n: int = 2000, seed: int = 0) -> tuple[pl.DataFrame, pl.Series, list[str]]:
    """特徵與目標有確定性關係、外加同質雜訊的資料集。

    ``y = 1 + 0.2·x + ε``，``ε ~ N(0, 0.1)``。真實的 τ 分位數是
    ``1 + 0.2·x + 0.1·Φ⁻¹(τ)``，故可用來檢驗模型是否真的估到該分位數。
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1 + 0.2 * x + rng.normal(0, 0.1, n)
    features = pl.DataFrame({"x": x, "noise": rng.normal(size=n)})
    return features, pl.Series("y", y), ["x", "noise"]


class TestTauTable:
    """τ 必須來自評分函數的推導，不得散落在程式碼各處。"""


    def test_values_match_settings(self) -> None:
        assert quantile.TAU_BY_TARGET["p_day"] == settings.TAU_PEAK
        assert quantile.TAU_BY_TARGET["p_night"] == settings.TAU_PEAK
        assert quantile.TAU_BY_TARGET["ramp_up"] == settings.TAU_RAMP_UP
        assert quantile.TAU_BY_TARGET["ramp_down"] == settings.TAU_RAMP_DOWN

    def test_ramp_down_is_symmetric(self) -> None:
        # ramp_down 無低估懲罰，故 τ 必須是 0.5。
        assert quantile.TAU_BY_TARGET["ramp_down"] == 0.5

    def test_peak_and_ramp_up_lean_high(self) -> None:
        # 低估有額外懲罰，故 τ 必須大於 0.5（預測偏高）。
        assert quantile.TAU_BY_TARGET["p_day"] > 0.5
        assert quantile.TAU_BY_TARGET["ramp_up"] > quantile.TAU_BY_TARGET["p_day"]


class TestRatioSpaceRestoration:
    """比值空間的還原。"""

    def test_prediction_scales_with_base(self) -> None:
        features, y, names = synthetic_quantile(n=500)
        model = quantile.fit(features, y, "p_day", names, in_ratio_space=True)
        single = quantile.predict(model, features, pl.Series([1.0] * features.height))
        doubled = quantile.predict(model, features, pl.Series([2.0] * features.height))
        assert np.allclose(doubled, 2 * single)


    def test_absolute_space_ignores_base(self) -> None:
        features, y, names = synthetic_quantile(n=500)
        model = quantile.fit(features, y, "p_day", names, in_ratio_space=False)
        assert np.allclose(
            quantile.predict(model, features),
            quantile.predict(model, features, pl.Series([9.0] * features.height)),
        )
