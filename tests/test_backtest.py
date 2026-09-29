"""回測：折與回測窗的定義、洩漏檢查、選模門檻、配對比較、設定覆寫、離線重組。
"""

from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest

from config import paths, settings
from src.evaluation import backtest, compare, cv, subgroup
from src.evaluation.compare import PairedComparison
from src.features import accuweather as accuweather_features
from src.features.targets import TARGET_NAMES, to_minutes


# =============================================================================
# 回測窗與紀錄
# =============================================================================


needs_special_days = pytest.mark.skipif(
    not paths.SPECIAL_DAYS_FILE.exists(), reason="子集需要本機的特殊日期表")


class TestWindows:
    def test_tuning_is_cv60_plus_daily_season(self) -> None:
        folds = backtest.windows("tuning")
        origins = {f.origin for f in folds}
        cv60 = {dt.date.fromisoformat(d) for d in settings.BACKTEST_CV60_ORIGINS}
        season = {f.origin for f in folds
                  if dt.date(2025, 9, 19) <= f.origin <= dt.date(2025, 10, 9)}
        assert cv60 <= origins and len(season) == 21
        assert len(folds) == len(cv60 | season) == 78

    def test_every_window_predicts_the_next_three_days(self) -> None:
        for group in backtest.GROUPS:
            for fold in backtest.windows(group):
                assert fold.target_dates == tuple(
                    fold.origin + dt.timedelta(days=h) for h in (1, 2, 3))

    def test_holdout_covers_july_to_september(self) -> None:
        folds = backtest.windows("holdout")
        assert len(folds) == 90
        assert folds[0].target_dates[0] == dt.date(2026, 7, 1)
        assert folds[-1].target_dates[-1] == dt.date(2026, 9, 30)

    def test_tuning_and_holdout_never_share_a_target_day(self) -> None:
        """保留組的日子不得出現在任何調參窗中（連訓練都不行：調參窗的歷史只到起點）。"""
        tuning_last = max(d for f in backtest.windows("tuning") for d in f.target_dates)
        holdout_first = min(d for f in backtest.windows("holdout") for d in f.target_dates)
        assert tuning_last <= dt.date.fromisoformat("2026-06-30") < holdout_first

    def test_unknown_group_raises(self) -> None:
        with pytest.raises(ValueError, match="未知的回測組"):
            backtest.windows("everything")

    @pytest.mark.skipif(not paths.TARGETS_FILE.exists(), reason="需要本機資料")
    def test_cv60_is_frozen_copy_of_make_folds(self) -> None:
        """寫死的 60 折必須等於開發期資料下 ``make_folds`` 的輸出，歷次分數才可比。"""
        daily = pl.read_parquet(paths.TARGETS_FILE)
        expected = [str(f.origin) for f in cv.make_folds(daily["date"])]
        assert list(settings.BACKTEST_CV60_ORIGINS) == expected


@needs_special_days
class TestSubsets:
    def test_thu_fri_sat_windows_start_on_wednesday(self) -> None:
        groups = backtest.subsets(backtest.windows("tuning"))
        assert groups["週四五六"] and all(o.isoweekday() == 3 for o in groups["週四五六"])
        assert len(groups["同季節"]) == 21

    def test_holiday_subset_contains_national_day(self) -> None:
        """2025-10-10 國慶日在同季節窗內，起點 10/7~10/9 的窗都含它。"""
        groups = backtest.subsets(backtest.windows("tuning"))
        assert {dt.date(2025, 10, d) for d in (7, 8, 9)} <= set(groups["含連假或停班"])


def _synthetic_daily(start: dt.date, n: int) -> pl.DataFrame:
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    return pl.DataFrame({
        "date": dates,
        "p_day": [30000.0 + i for i in range(n)],
        "t_day": [float(to_minutes("14:00"))] * n,
        "p_night": [28000.0 + i for i in range(n)],
        "t_night": [float(to_minutes("18:00"))] * n,
        "ramp_up": [1000.0] * n,
        "ramp_down": [800.0] * n,
    })


def test_each_window_only_sees_history_up_to_its_origin() -> None:
    """洩漏檢查：每個回測窗交給預測函式的歷史，最後一天恰為起點日（負載到 D-1 23:50）。"""
    daily = _synthetic_daily(dt.date(2024, 1, 1), 915)   # 到 2026-07-03
    seen = []

    def predict(history: pl.DataFrame, target_dates: tuple) -> pl.DataFrame:
        seen.append((history["date"].max(), min(target_dates)))
        return history.tail(len(target_dates)).select(TARGET_NAMES)

    cv.run_cv(daily, backtest.windows("tuning"), predict)
    assert len(seen) == 78
    assert all(last == first - dt.timedelta(days=1) for last, first in seen)


def test_windows_without_data_are_listed_not_dropped_silently(tmp_path, monkeypatch) -> None:
    from src.features import accuweather as accuweather_features

    daily = _synthetic_daily(dt.date(2026, 6, 1), 40)             # 到 2026-07-10
    daily.write_parquet(tmp_path / "targets.parquet")
    monkeypatch.setattr(paths, "TARGETS_FILE", tmp_path / "targets.parquet")
    monkeypatch.setattr(accuweather_features, "build_forecast",
                        lambda: daily.filter(pl.col("date") <= dt.date(2026, 7, 5)).select("date"))
    folds = [backtest._fold(dt.date(2026, 7, d)) for d in (1, 2, 3, 8)]
    usable, skipped = backtest._available(folds, "honest", 0)
    assert [f.origin for f in usable] == [dt.date(2026, 7, 1), dt.date(2026, 7, 2)]
    assert skipped == [
        {"origin": "2026-07-03", "reason": "Accuweather 未涵蓋"},
        {"origin": "2026-07-08", "reason": "負載未涵蓋、Accuweather 未涵蓋"},
    ]


class TestHoldoutGuard:
    GIT = {"commit": "abc123", "dirty": False}

    def test_refuses_without_confirmation(self) -> None:
        with pytest.raises(PermissionError, match="保留確認組"):
            backtest.run("holdout", "偷看")

    def test_usage_is_logged_and_repeat_warns(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(paths, "BACKTEST_DIR", tmp_path)
        warnings = []
        monkeypatch.setattr(backtest.logger, "warning", lambda *a: warnings.append(a))
        backtest._guard_holdout(True, self.GIT, "final")
        assert warnings == []
        backtest._guard_holdout(True, self.GIT, "final_again")
        assert len(warnings) == 1
        log = (tmp_path / "holdout_usage.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["label"] for line in log] == ["final", "final_again"]


def _paired(diff: float, n_stderr: float) -> PairedComparison:
    return PairedComparison("cand", "ref", 78, 1.0 + diff, 1.0, diff, 0.01, n_stderr,
                            n_stderr > 1, "cand")


class TestGate:
    """選模門檻：全體改善 > 1 SE，且同季節沒有變差 > 1 SE。"""

    def test_passes(self) -> None:
        assert backtest.passes_gate(_paired(-0.03, 3.0), _paired(-0.01, 0.5))

    def test_overall_better_but_season_clearly_worse(self) -> None:
        assert not backtest.passes_gate(_paired(-0.03, 3.0), _paired(+0.05, 1.5))

    def test_season_slightly_worse_is_tolerated(self) -> None:
        assert backtest.passes_gate(_paired(-0.03, 3.0), _paired(+0.01, 0.4))

    def test_overall_within_noise(self) -> None:
        assert not backtest.passes_gate(_paired(-0.01, 0.8), _paired(-0.05, 3.0))

    def test_overall_clearly_worse(self) -> None:
        assert not backtest.passes_gate(_paired(+0.03, 3.0), None)


def _fake_run(folder, label: str, origins: list[dt.date], scores: list[float]) -> None:
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps(
        {"label": label, "git": {"commit": "0" * 40}}), encoding="utf-8")
    pl.DataFrame({"origin": origins, "total_score": scores}).write_csv(folder / "folds.csv")
    pl.DataFrame({"origin": origins}).write_csv(folder / "predictions.csv")


def test_compare_refuses_different_windows(tmp_path) -> None:
    _fake_run(tmp_path / "a", "ref", [dt.date(2025, 2, 4), dt.date(2025, 2, 12)], [1.0, 1.1])
    _fake_run(tmp_path / "b", "cand", [dt.date(2025, 2, 4), dt.date(2025, 2, 20)], [1.0, 1.1])
    with pytest.raises(ValueError, match="窗不同"):
        backtest.compare(tmp_path / "a", tmp_path / "b")


@pytest.mark.skipif(not paths.has_accuweather_forecast() or not paths.TARGETS_FILE.exists(),
                    reason="需要 Accuweather 年度檔與重建後的 data/processed/")
def test_recorded_curves_equal_submission_curves(tmp_path) -> None:
    """回測紀錄的 432 點曲線必須與提交流程（``predict_window``）產出的完全相同。"""
    import numpy as np

    from src import workflow

    origin = dt.date(2025, 9, 30)
    folds = [f for f in backtest.windows("tuning") if f.origin in (origin, origin + dt.timedelta(days=1))]
    recorded = workflow.evaluate("honest", output_dir=tmp_path, folds=folds)["curves"]
    assert (tmp_path / "curves_honest.csv").exists()
    recorded = recorded.filter(pl.col("origin") == origin).sort("ts")
    submitted = workflow.predict_window(origin).sort("ts")
    assert recorded.height == settings.PREDICT_HORIZON_DAYS * settings.POINTS_PER_DAY
    assert np.array_equal(recorded["predicted"].to_numpy(), submitted["predicted"].to_numpy())
    assert recorded["actual"].null_count() == 0


@needs_special_days
def test_run_writes_record_with_fingerprint(tmp_path, monkeypatch) -> None:
    """summary.json 必須含 manifest 雜湊、git commit、設定、各子集分數。"""
    from src import workflow

    folds = backtest.windows("tuning")[:3]
    table = pl.DataFrame({
        "origin": [f.origin for f in folds],
        **{name: [0.1, 0.2, 0.3] for name in backtest.SUBSCORES},
        "total_score": [1.0, 1.2, 1.4],
    })

    def fake_evaluate(mode, output_dir, folds, observed_lag_days):
        table.write_csv(output_dir / f"folds_{mode}.csv")
        table.select("origin").write_csv(output_dir / f"predictions_{mode}.csv")
        return {"table": table}

    monkeypatch.setattr(paths, "BACKTEST_DIR", tmp_path)
    monkeypatch.setattr(workflow, "evaluate", fake_evaluate)
    monkeypatch.setattr(backtest, "_available", lambda f, m, lag: (folds, []))
    monkeypatch.setattr(backtest, "_data_fingerprint", lambda: {
        "manifest_sha256": "f" * 64, "data_matches_manifest": True, "files_changed": []})

    folder = backtest.run("tuning", "unit")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest_sha256"] == "f" * 64
    assert len(summary["git"]["commit"]) == 40
    assert summary["group"] == "tuning" and summary["weather_mode"] == "honest"
    assert summary["overall"]["total_score"] == pytest.approx(1.2)
    assert set(summary["subsets"]) == {"同季節", "週四五六", "含週六", "夏月末期", "含連假或停班"}
    assert (folder / "folds.csv").exists() and (folder / "predictions.csv").exists()


def test_experiment_table_collects_records_and_comparisons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "BACKTEST_DIR", tmp_path)
    folder = tmp_path / "20260928_000000_x_tuning"
    folder.mkdir()
    overall = {"total_score": 1.0, **{name: 0.1 for name in backtest.SUBSCORES}}
    (folder / "summary.json").write_text(json.dumps({
        "label": "x", "overrides": {"ENABLE_WINDY": True}, "n_windows": 78,
        "overall": overall, "subsets": {"同季節": {"total_score": 0.8}}, "runtime_seconds": 10,
    }), encoding="utf-8")
    (folder / "comparison_vs_ref.json").write_text(json.dumps({
        "reference": "/a/ref", "overall": {"diff": -0.02, "n_stderr": 2.0},
        "subsets": {"同季節": {"diff": 0.0, "n_stderr": 0.1}}, "passes_gate": True,
    }), encoding="utf-8")
    (tmp_path / "20260928_000001_running_tuning").mkdir()          # 尚未完成的紀錄略過
    table = backtest.experiment_table()
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["label"] == "x" and row["差值"] == -0.02 and row["通過門檻"] is True
    assert row["對照"] == "ref" and row["同季節"] == 0.8


# =============================================================================
# 折與配對比較
# =============================================================================


N_DAYS = 800


START = dt.date(2024, 1, 1)


def synthetic_daily() -> pl.DataFrame:
    """構造一張有足夠長度的假 daily-row 表。"""
    dates = [START + dt.timedelta(days=i) for i in range(N_DAYS)]
    return pl.DataFrame(
        {
            "date": dates,
            "p_day": [30000.0 + i for i in range(N_DAYS)],
            "t_day": [float(to_minutes("14:00"))] * N_DAYS,
            "p_night": [28000.0 + i for i in range(N_DAYS)],
            "t_night": [float(to_minutes("18:00"))] * N_DAYS,
            "ramp_up": [1000.0] * N_DAYS,
            "ramp_down": [800.0] * N_DAYS,
        }
    )


class TestMakeFolds:
    """折的建構。"""

    def test_fold_count_and_horizon(self) -> None:
        folds = cv.make_folds(synthetic_daily()["date"], horizon=3, n_folds=20)
        assert len(folds) == 20
        assert all(len(f.target_dates) == 3 for f in folds)

    def test_target_dates_follow_origin(self) -> None:
        fold = cv.make_folds(synthetic_daily()["date"], horizon=3, n_folds=20)[0]
        assert fold.target_dates == tuple(
            fold.origin + dt.timedelta(days=h) for h in (1, 2, 3)
        )

    def test_origins_respect_min_history(self) -> None:
        folds = cv.make_folds(
            synthetic_daily()["date"], horizon=3, n_folds=20, min_history_days=400
        )
        assert all((f.origin - START).days >= 400 for f in folds)

    def test_origins_spread_across_seasons(self) -> None:
        # 均勻取樣的重點：折不能全擠在同一季節，否則 CV 只測到一種天氣。
        folds = cv.make_folds(synthetic_daily()["date"], horizon=3, n_folds=60)
        assert len({f.origin.month for f in folds}) >= 8


    def test_raises_when_not_enough_history(self) -> None:
        short = synthetic_daily().head(100)
        with pytest.raises(ValueError, match="不足"):
            cv.make_folds(short["date"], horizon=3, n_folds=20, min_history_days=400)


class TestFoldNoLeakage:
    """預測函式拿到的 history 不得含目標日或之後的任何一天。"""

    def test_history_is_truncated_at_origin(self) -> None:
        daily = synthetic_daily()
        folds = cv.make_folds(daily["date"], horizon=3, n_folds=10)
        seen: list[tuple[dt.date, dt.date]] = []

        def spy(history: pl.DataFrame, target_dates: tuple[dt.date, ...]) -> pl.DataFrame:
            seen.append((history["date"].max(), min(target_dates)))  # type: ignore[arg-type]
            return pl.DataFrame(
                {name: [1000.0] * len(target_dates) for name in TARGET_NAMES}
            )

        cv.run_cv(daily, folds, spy)
        assert len(seen) == len(folds)
        for history_max, first_target in seen:
            assert history_max < first_target


class TestRunCv:
    """CV 執行結果。"""

    def test_perfect_predictor_scores_zero(self) -> None:
        daily = synthetic_daily()
        folds = cv.make_folds(daily["date"], horizon=3, n_folds=10)

        def oracle(history: pl.DataFrame, target_dates: tuple[dt.date, ...]) -> pl.DataFrame:
            return (
                daily.filter(pl.col("date").is_in(list(target_dates)))
                .sort("date")
                .select(TARGET_NAMES)
            )

        scores = cv.run_cv(daily, folds, oracle)
        assert scores.height == 10
        assert scores["total_score"].max() == pytest.approx(0.0)

    def test_summarize_reports_worst_fold(self) -> None:
        scores = pl.DataFrame(
            {
                "origin": [dt.date(2025, 1, i) for i in (1, 2, 3)],
                "total_score": [1.0, 5.0, 2.0],
            }
        )
        summary = cv.summarize_cv(scores)
        assert summary["worst"] == 5.0
        assert summary["worst_origin"] == dt.date(2025, 1, 2)
        assert summary["mean"] == pytest.approx(8 / 3)


class TestSpecialFolds:
    """三組專項驗證的篩選條件。"""

    def test_saturday_folds_contain_a_saturday(self) -> None:
        folds = cv.make_folds(synthetic_daily()["date"], horizon=3, n_folds=60)
        subset = cv.special_saturday_folds(folds)
        assert subset
        for fold in subset:
            assert any(d.isoweekday() == 6 for d in fold.target_dates)

    def test_thu_fri_sat_folds_match_target_period_structure(self) -> None:
        # 目標期 2026/10/1(四) 10/2(五) 10/3(六)。
        folds = cv.make_folds(synthetic_daily()["date"], horizon=3, n_folds=60)
        subset = cv.special_thu_fri_sat_folds(folds)
        for fold in subset:
            assert [d.isoweekday() for d in fold.target_dates] == [4, 5, 6]
            assert fold.origin.isoweekday() == 3  # 起點為週三


class TestPairedCompare:
    """配對比較的標準誤門檻。"""

    @staticmethod
    def scores(values: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "origin": [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(len(values))],
                "total_score": values,
            }
        )

    def test_large_consistent_improvement_passes(self) -> None:
        a = self.scores([1.0] * 30)
        b = self.scores([2.0] * 30)
        c = compare.paired_compare(a, b, "A", "B", simpler="B")
        assert c.passes_threshold
        assert c.winner == "A"

    def test_noisy_tiny_improvement_falls_back_to_simpler(self) -> None:
        # A 平均略低但逐折差值雜訊很大 → 不足 1 個標準誤 → 選較簡單的 B。
        a = self.scores([1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
        b = self.scores([3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.1])
        c = compare.paired_compare(a, b, "A", "B", simpler="B")
        assert not c.passes_threshold
        assert c.winner == "B"

    def test_worse_candidate_never_wins(self) -> None:
        a = self.scores([2.0] * 30)
        b = self.scores([1.0] * 30)
        c = compare.paired_compare(a, b, "A", "B", simpler="A")
        assert c.winner == "B"

    def test_single_fold_cannot_pass(self) -> None:
        # 一折算不出標準誤，不得讓任何候選僥倖通過。
        c = compare.paired_compare(
            self.scores([1.0]), self.scores([9.0]), "A", "B", simpler="B"
        )
        assert not c.passes_threshold
        assert c.winner == "B"

    def test_mismatched_folds_raise(self) -> None:
        a = self.scores([1.0, 2.0])
        b = a.with_columns(pl.col("origin") + pl.duration(days=99))
        with pytest.raises(ValueError, match="origin 不一致"):
            compare.paired_compare(a, b, "A", "B", simpler="B")


    def test_zero_variance_improvement_is_certain(self) -> None:
        # 30 折逐折差值完全相同 → 標準誤為 0 → 改善無可置疑。
        # 這是退化案例（實務不會出現），但行為必須明確而非拋錯或回傳 nan。
        c = compare.paired_compare(
            self.scores([1.0] * 30), self.scores([2.0] * 30), "A", "B", simpler="B"
        )
        assert c.stderr == 0.0
        assert c.passes_threshold
        assert c.winner == "A"


# =============================================================================
# Plan B 補值、設定覆寫、離線重組、遞迴對照
# =============================================================================


PLAN_B_ORIGIN = dt.date(2026, 3, 31)


PLAN_B_TARGETS = [PLAN_B_ORIGIN + dt.timedelta(days=h) for h in (1, 2, 3)]


@pytest.fixture
def supply(monkeypatch):
    """合成的預報與觀測：預報每天 30°C；觀測 3 月 20°C、其他月份 10°C，最後一天 25°C。"""
    monkeypatch.setattr(settings, "WEATHER_CALIBRATE", False)
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=k) for k in range(95)]  # 到 4/5
    forecast = pl.DataFrame({"date": days, "w_臺北_day_tmax": [30.0] * 95,
                             "w_臺南_day_tmax": [31.0] * 95})
    observed = pl.DataFrame({
        "date": days,
        "w_臺北_day_tmax": [25.0 if d == PLAN_B_ORIGIN else 20.0 if d.month == 3 else 10.0 for d in days],
        "w_臺南_day_tmax": [20.0] * 95,
    })
    return accuweather_features.make_target_weather_fn(forecast, observed)


class TestPlanB:
    def test_nothing_missing_uses_forecast(self, supply) -> None:
        rows = supply(PLAN_B_ORIGIN, PLAN_B_TARGETS)
        assert rows["w_臺北_day_tmax"].to_list() == [30.0] * 3
        assert rows["tmax"].to_list() == [31.0] * 3

    def test_missing_aborts_when_plan_b_is_abort(self, supply, monkeypatch) -> None:
        monkeypatch.setattr(settings, "FORECAST_UNAVAILABLE_HORIZONS", (3,))
        monkeypatch.setattr(settings, "FORECAST_PLAN_B", "abort")
        with pytest.raises(ValueError, match="Plan B 尚未選定"):
            supply(PLAN_B_ORIGIN, PLAN_B_TARGETS)

    def test_persistence_uses_latest_observation(self, supply, monkeypatch) -> None:
        monkeypatch.setattr(settings, "FORECAST_UNAVAILABLE_HORIZONS", (3,))
        monkeypatch.setattr(settings, "FORECAST_PLAN_B", "persistence")
        rows = supply(PLAN_B_ORIGIN, PLAN_B_TARGETS)
        assert rows["w_臺北_day_tmax"].to_list() == [30.0, 30.0, 25.0]
        assert rows["tmax"].to_list() == [31.0, 31.0, 25.0]

    def test_climatology_uses_same_month_history_only(self, supply, monkeypatch) -> None:
        """3/21 缺預報 → 只用起點（3/20）以前的 3 月觀測，不可用到起點之後。"""
        monkeypatch.setattr(settings, "FORECAST_UNAVAILABLE_HORIZONS", (1,))
        monkeypatch.setattr(settings, "FORECAST_PLAN_B", "climatology")
        rows = supply(dt.date(2026, 3, 20), [dt.date(2026, 3, d) for d in (21, 22, 23)])
        # 3 月 1–20 日的臺北觀測都是 20°C
        assert rows["w_臺北_day_tmax"].to_list() == [20.0, 30.0, 30.0]

    def test_single_station_missing(self, supply, monkeypatch) -> None:
        monkeypatch.setattr(settings, "FORECAST_UNAVAILABLE_STATIONS", ("臺北",))
        monkeypatch.setattr(settings, "FORECAST_PLAN_B", "persistence")
        rows = supply(PLAN_B_ORIGIN, PLAN_B_TARGETS)
        assert rows["w_臺北_day_tmax"].to_list() == [25.0] * 3
        assert rows["w_臺南_day_tmax"].to_list() == [31.0] * 3


    def test_actual_missing_cells_use_plan_b(self, supply, monkeypatch) -> None:
        """比賽當天由前置檢查判定的缺漏（FORECAST_MISSING_CELLS）也走同一個 Plan B。"""
        monkeypatch.setattr(settings, "FORECAST_MISSING_CELLS", {PLAN_B_TARGETS[2]: ("臺北",)})
        rows = supply(PLAN_B_ORIGIN, PLAN_B_TARGETS)
        assert rows["w_臺北_day_tmax"].to_list() == [30.0, 30.0, 25.0]   # 預設為持續性
        assert rows["w_臺南_day_tmax"].to_list() == [31.0] * 3


class TestForecastMissing:
    """某站某天不足 24 小時就算缺（半天的預報會算錯最高溫）。"""

    def test_full_partial_and_hour_short(self, monkeypatch) -> None:
        from src.data import accuweather, checks

        days = [dt.date(2026, 10, d) for d in (1, 2, 3)]
        rows = []
        for day in days:
            for name, code in settings.WEATHER_STATIONS.items():
                for hour in range(24):
                    if day == days[2]:
                        continue                                   # 10/3 全部沒有
                    if day == days[1] and name == "臺北" and hour == 23:
                        continue                                   # 10/2 臺北少一小時
                    rows.append({"Date": dt.datetime.combine(day, dt.time(hour)),
                                 "stn_ID": code, "AirTemperature": 30.0, "UVIndex": None})
        monkeypatch.setattr(accuweather, "load_station_hourly", lambda: pl.DataFrame(rows))
        missing = checks.forecast_missing(days)
        assert days[0] not in missing
        assert missing[days[1]] == ("臺北",)
        assert set(missing[days[2]]) == set(settings.WEATHER_STATIONS)


class TestOverrides:
    def test_parse(self) -> None:
        parsed = backtest.parse_overrides(
            ["TIMING_LEARNED_MIX=0.3", "WEATHER_METRICS=[\"day_tmax\"]", "FORECAST_PLAN_B=persistence"])
        assert parsed == {"TIMING_LEARNED_MIX": 0.3, "WEATHER_METRICS": ("day_tmax",),
                          "FORECAST_PLAN_B": "persistence"}

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="無法解析"):
            backtest.parse_overrides(["TIMING_LEARNED_MIXX=0.3"])

    def test_restored_even_when_run_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(backtest, "_data_fingerprint", lambda: {})
        monkeypatch.setattr(backtest, "_available",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("中途失敗")))
        before = settings.TIMING_LEARNED_MIX
        with pytest.raises(RuntimeError):
            backtest.run("tuning", "x", overrides={"TIMING_LEARNED_MIX": 0.9})
        assert settings.TIMING_LEARNED_MIX == before


REFERENCE_RUNS = sorted(paths.BACKTEST_DIR.glob("*_current_best_tuning")) if paths.BACKTEST_DIR.exists() else []


@pytest.mark.skipif(not REFERENCE_RUNS or not paths.TARGETS_FILE.exists() or not paths.SPECIAL_DAYS_FILE.exists(),
                    reason="需要先重建 data/processed/")
def test_recombine_with_itself_reproduces_scores(tmp_path, monkeypatch) -> None:
    """重組的重新評分必須與原本逐窗評分完全一致，否則離線重組不可信。"""
    monkeypatch.setattr(paths, "BACKTEST_DIR", tmp_path)
    reference = REFERENCE_RUNS[-1]
    folder = backtest.recombine(reference, reference, (3,), "identity")
    before = pl.read_csv(reference / "folds.csv").sort("origin")
    after = pl.read_csv(folder / "folds.csv").sort("origin")
    assert (before["total_score"] - after["total_score"]).abs().max() < 1e-9


def _clean_series(days: int = 70) -> pl.DataFrame:
    """合成的 10 分鐘序列：日週期 + 週末較低，附日別。"""
    import numpy as np

    start = dt.datetime(2026, 1, 1)
    ts = [start + dt.timedelta(minutes=10 * k) for k in range(days * 144)]
    load = [30000 + 3000 * np.sin(2 * np.pi * (k % 144) / 144) - (2000 if t.weekday() >= 5 else 0)
            for k, t in enumerate(ts)]
    frame = pl.DataFrame({"ts": ts, "Load_MW": load}).with_columns(
        pl.col("ts").dt.date().alias("date"))
    return frame.with_columns(
        pl.when(pl.col("ts").dt.weekday() >= 6).then(pl.lit("假日")).otherwise(pl.lit("平日"))
        .alias("price_daytype"))


class TestNoLookAhead:
    """遞迴 AR 對照模型只能看到起點（含）以前的負載。"""

    ORIGIN = dt.date(2026, 3, 1)
    DAYS = (dt.date(2026, 3, 2), dt.date(2026, 3, 3), dt.date(2026, 3, 4))

    def _history(self, clean):
        return clean.filter(pl.col("date") <= self.ORIGIN).select("date").unique()

    def test_recursive_ar_ignores_future_values(self) -> None:
        from src.models import recursive

        clean = _clean_series()
        tampered = clean.with_columns(
            pl.when(pl.col("date") > self.ORIGIN).then(pl.lit(1e9)).otherwise(pl.col("Load_MW"))
            .alias("Load_MW"))
        a = recursive.make_recursive_predictor(clean)(self._history(clean), self.DAYS)
        b = recursive.make_recursive_predictor(tampered)(self._history(clean), self.DAYS)
        assert a.equals(b)


# =============================================================================
# 逐目標拆解
# =============================================================================


def make_target_frame(rows: list[dict]) -> pl.DataFrame:
    """構造 6 目標的每日表。"""
    return pl.DataFrame(rows)


BASE = [
    {"date": dt.date(2025, 1, 1), "t_day": 840, "t_night": 1030,
     "p_day": 30000.0, "p_night": 28000.0, "ramp_up": 1000.0, "ramp_down": 800.0},
    {"date": dt.date(2025, 1, 2), "t_day": 840, "t_night": 1030,
     "p_day": 30000.0, "p_night": 28000.0, "ramp_up": 1000.0, "ramp_down": 800.0},
]


class TestPerTargetLoss:
    """逐日損失。"""

    def test_timing_uses_grid_exponent(self) -> None:
        """時刻損失是 (相差格數) ** 1.2，不是分鐘差。"""
        actual = make_target_frame(BASE)
        pred = make_target_frame([{**BASE[0], "t_day": 860}, BASE[1]])
        loss = subgroup.per_target_loss(actual, pred)
        assert loss["t_day"][0] == pytest.approx(2 ** 1.2)
        assert loss["t_day"][1] == 0.0

    def test_magnitude_uses_relative_error(self) -> None:
        actual = make_target_frame(BASE)
        pred = make_target_frame([{**BASE[0], "p_day": 33000.0}, BASE[1]])
        loss = subgroup.per_target_loss(actual, pred)
        assert loss["p_day"][0] == pytest.approx(0.1)


class TestCompare:
    """拆解比較。"""

    def test_counts_changed_days_not_just_mean(self) -> None:
        """核心行為：平均值再漂亮，也要能看出只有幾天真的改變。"""
        actual = make_target_frame(BASE)
        base_pred = make_target_frame([{**BASE[0], "t_day": 940}, BASE[1]])
        var_pred = make_target_frame(BASE)  # 只有第一天被修好
        table = subgroup.compare(
            subgroup.per_target_loss(actual, base_pred),
            subgroup.per_target_loss(actual, var_pred),
        )
        row = table.filter(pl.col("目標") == "t_day").row(0, named=True)
        assert row["天數"] == 2
        assert row["改變天數"] == 1       # ← 只有 1 天
        assert row["好"] == 1 and row["壞"] == 0
        assert row["平均差值"] < 0

    def test_splits_by_group(self) -> None:
        """能依子群拆開——這正是只看全年總分時缺的那一層。"""
        actual = make_target_frame(BASE)
        pred = make_target_frame(BASE)
        groups = pl.DataFrame({
            "date": [r["date"] for r in BASE],
            "is_holiday": ["連假", "一般"],
        })
        table = subgroup.compare(
            subgroup.per_target_loss(actual, pred),
            subgroup.per_target_loss(actual, pred), groups)
        assert set(table["子群"].to_list()) == {"連假", "一般"}

    def test_report_hides_unchanged_rows(self) -> None:
        actual = make_target_frame(BASE)
        pred = make_target_frame(BASE)
        table = subgroup.compare(subgroup.per_target_loss(actual, pred),
                                 subgroup.per_target_loss(actual, pred))
        assert "沒有任何一天的預測改變" in subgroup.format_report(table)
