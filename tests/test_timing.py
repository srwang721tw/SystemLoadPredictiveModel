"""尖峰時刻：貝氏決策、類別對照表、學習式模型、階層收縮的條件經驗分布。

候選點必須是完整名目格點（37／24 格），與模型的類別集合是兩回事：
期望損失的最小點可以落在歷史上從未出現的時刻。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from config import settings
from src.features.targets import to_minutes
from src.models import decision, timing


# =============================================================================
# 貝氏決策
# =============================================================================


class TestCandidateGrid:
    """候選點必須是完整名目格點。"""

    def test_day_grid_has_nominal_size(self) -> None:
        assert len(decision.candidate_grid("t_day")) == settings.DAY_PEAK_N_POINTS == 37

    def test_night_grid_has_nominal_size(self) -> None:
        assert len(decision.candidate_grid("t_night")) == settings.NIGHT_PEAK_N_POINTS == 24

    def test_grids_span_the_full_window(self) -> None:
        day = decision.candidate_grid("t_day")
        assert day[0] == to_minutes(settings.DAY_PEAK_START)
        assert day[-1] == to_minutes(settings.DAY_PEAK_END)


class TestLossMatrix:
    """長方形損失矩陣與實際分鐘差。"""

    def test_shape_is_rectangular(self) -> None:
        candidates = decision.candidate_grid("t_day")
        classes = [to_minutes("11:00"), to_minutes("14:00")]
        assert decision.loss_matrix(candidates, classes).shape == (37, 2)

    def test_zero_on_exact_match(self) -> None:
        classes = [to_minutes("14:00")]
        matrix = decision.loss_matrix(classes, classes)
        assert matrix[0, 0] == 0.0

    def test_uses_actual_minutes_not_class_index(self) -> None:
        # 類別集合有缺口：11:50 與 13:00 是序號相鄰但實際差 70 分鐘。
        # 若誤用序號差會得到 (1)^1.2 = 1；正確答案是 (70/10)^1.2 = 7^1.2。
        classes = [to_minutes("11:50"), to_minutes("13:00")]
        matrix = decision.loss_matrix([to_minutes("11:50")], classes)
        assert matrix[0, 1] == pytest.approx(7**1.2)


class TestBayesDecisionCore:
    """決策落在候選集合，而非類別集合。"""

    def test_bimodal_optimum_falls_in_the_gap(self) -> None:
        # settings.DECISION_CANDIDATES_FULL_GRID 的核心案例：
        # 11:30 與 13:30 各半，最佳解是 12:30——一個從未出現過的時刻。
        classes = [to_minutes("11:30"), to_minutes("13:30")]
        pmf = np.array([[0.5, 0.5]])
        assert decision.bayes_decision(pmf, classes, "t_day")[0] == to_minutes("12:30")


    def test_expected_loss_of_gap_beats_both_modes(self) -> None:
        # 手算驗證：中間點 8.59 < 兩端點 9.87。
        classes = [to_minutes("11:30"), to_minutes("13:30")]
        pmf = np.array([[0.5, 0.5]])
        candidates = decision.candidate_grid("t_day")
        losses = decision.expected_loss(pmf, candidates, classes)[0]
        at = {m: losses[candidates.index(to_minutes(m))] for m in ("11:30", "12:30", "13:30")}
        assert at["12:30"] == pytest.approx(2 * 0.5 * 6**1.2)
        assert at["11:30"] == pytest.approx(0.5 * 12**1.2)
        assert at["12:30"] < at["11:30"]
        assert at["12:30"] < at["13:30"]

    def test_degenerate_pmf_picks_that_class(self) -> None:
        classes = [to_minutes("11:30"), to_minutes("14:00"), to_minutes("16:00")]
        pmf = np.array([[0.0, 1.0, 0.0]])
        assert decision.bayes_decision(pmf, classes, "t_day")[0] == to_minutes("14:00")

    def test_symmetric_bimodal_picks_midpoint(self) -> None:
        classes = [to_minutes("12:00"), to_minutes("16:00")]
        pmf = np.array([[0.5, 0.5]])
        assert decision.bayes_decision(pmf, classes, "t_day")[0] == to_minutes("14:00")

    def test_asymmetric_bimodal_leans_to_heavier_mode(self) -> None:
        classes = [to_minutes("12:00"), to_minutes("16:00")]
        chosen = decision.bayes_decision(np.array([[0.9, 0.1]]), classes, "t_day")[0]
        assert to_minutes("12:00") <= chosen < to_minutes("14:00")

    def test_decision_never_leaves_the_candidate_grid(self) -> None:
        rng = np.random.default_rng(0)
        classes = decision.candidate_grid("t_night")
        pmf = rng.dirichlet(np.ones(len(classes)), size=200)
        chosen = decision.bayes_decision(pmf, classes, "t_night")
        assert set(chosen.tolist()) <= set(decision.candidate_grid("t_night"))


class TestNotMedianNotMean:
    """指數 1.2 介於 1 與 2 之間，故決策既非中位數也非均值。"""

    @staticmethod
    def _median(pmf: np.ndarray, classes: list[int]) -> int:
        return classes[int(np.searchsorted(np.cumsum(pmf), 0.5))]

    def test_lies_strictly_between_mean_and_median(self) -> None:
        # 左側有一大塊機率、右側兩個相鄰尖峰：
        #   中位數 15:00（被右側尖峰佔住）、均值 13:14（被左側拉走）
        #   貝氏決策 14:00 —— 嚴格落在兩者之間，與兩者都不同。
        classes = [to_minutes(t) for t in ("11:00", "15:00", "15:10")]
        pmf = np.array([[0.45, 0.30, 0.25]])
        chosen = int(decision.bayes_decision(pmf, classes, "t_day")[0])
        median = self._median(pmf[0], classes)
        mean = float(np.dot(pmf[0], classes))

        assert chosen == to_minutes("14:00")
        assert chosen != median
        assert chosen != round(mean)
        assert mean < chosen < median


    def test_matches_brute_force_search(self) -> None:
        # 與逐點暴力搜尋比對，確認向量化實作沒有寫錯。
        rng = np.random.default_rng(7)
        classes = [to_minutes(t) for t in ("11:00", "12:30", "14:00", "15:20", "17:00")]
        candidates = decision.candidate_grid("t_day")
        for _ in range(50):
            pmf = rng.dirichlet(np.ones(len(classes)))
            expected = min(
                candidates,
                key=lambda j: sum(
                    p * (abs(j - k) / 10) ** 1.2
                    for p, k in zip(pmf, classes, strict=True)
                ),
            )
            assert decision.bayes_decision(pmf[None, :], classes, "t_day")[0] == expected


class TestBatchAndValidation:
    """批次處理與輸入檢查。"""

    def test_processes_multiple_rows(self) -> None:
        classes = [to_minutes("12:00"), to_minutes("16:00")]
        pmf = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        chosen = decision.bayes_decision(pmf, classes, "t_day")
        assert chosen.tolist() == [
            to_minutes("12:00"), to_minutes("16:00"), to_minutes("14:00")
        ]

    def test_rejects_unnormalised_pmf(self) -> None:
        with pytest.raises(ValueError, match="列和偏離 1"):
            decision.bayes_decision(np.array([[0.3, 0.3]]), [660, 780], "t_day")


# =============================================================================
# 時刻模型
# =============================================================================


class TestClassMapping:
    """類別對照表：保留、合併、與序號的一致性。"""

    def test_keeps_frequent_grid_points(self) -> None:
        y = pl.Series([to_minutes("14:00")] * 10 + [to_minutes("15:00")] * 10)
        classes, _ = timing.build_class_mapping(y, min_count=3)
        assert classes == [to_minutes("14:00"), to_minutes("15:00")]

    def test_classes_are_sorted_by_time(self) -> None:
        y = pl.Series(
            [to_minutes("16:00")] * 5 + [to_minutes("12:00")] * 5 + [to_minutes("14:00")] * 5
        )
        classes, _ = timing.build_class_mapping(y, min_count=3)
        assert classes == sorted(classes)

    def test_rare_point_merges_into_nearest_kept_class(self) -> None:
        # 13:00 只出現 1 次，應併入時間上最近的保留類別 12:50（差 10 分），
        # 而不是 16:00（差 180 分）。
        y = pl.Series(
            [to_minutes("12:50")] * 10 + [to_minutes("16:00")] * 10 + [to_minutes("13:00")]
        )
        classes, mapping = timing.build_class_mapping(y, min_count=3)
        assert to_minutes("13:00") not in classes
        assert classes[mapping[to_minutes("13:00")]] == to_minutes("12:50")

    def test_rare_samples_are_not_dropped(self) -> None:
        # 罕見格點的樣本必須被併入，不得丟棄——邊界審查就發生在罕見時刻上。
        y = pl.Series([to_minutes("14:00")] * 10 + [to_minutes("17:00")])
        _, mapping = timing.build_class_mapping(y, min_count=3)
        assert to_minutes("17:00") in mapping

    def test_mapping_covers_every_observed_value(self) -> None:
        y = pl.Series(
            [to_minutes("11:00")] * 4
            + [to_minutes("13:20")] * 6
            + [to_minutes("15:40")] * 2
            + [to_minutes("17:00")] * 1
        )
        _, mapping = timing.build_class_mapping(y, min_count=3)
        assert set(mapping) == set(y.to_list())


    def test_all_rare_falls_back_to_single_class(self) -> None:
        # 極端情形不得拋錯——模型退化成常數預測，但流程要能走完。
        y = pl.Series([to_minutes("11:00"), to_minutes("13:00"), to_minutes("15:00")])
        classes, mapping = timing.build_class_mapping(y, min_count=10)
        assert len(classes) == 1
        assert set(mapping.values()) == {0}


def synthetic_training_data(n: int = 300, seed: int = 0) -> tuple[pl.DataFrame, pl.Series, list[str]]:
    """構造一個「特徵可完全決定時刻」的假資料集。

    ``is_weekend`` 為真時尖峰在 16:00，否則在 13:00。模型若學不會這種
    確定性關係，代表訓練流程本身有問題。
    """
    rng = np.random.default_rng(seed)
    is_weekend = rng.integers(0, 2, n)
    features = pl.DataFrame(
        {
            "is_weekend": is_weekend.astype(float),
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        }
    )
    y = pl.Series(
        "t_day",
        [to_minutes("16:00") if w else to_minutes("13:00") for w in is_weekend],
    )
    return features, y, ["is_weekend", "noise_a", "noise_b"]


def decide(model: timing.TimingModel, features: pl.DataFrame) -> np.ndarray:
    """分類器的 PMF 在完整名目格點上做貝氏決策（上線路徑的時刻決策方式）。"""
    return decision.bayes_decision(timing.predict_pmf(model, features), model.classes, model.target)


class TestFitAndPredict:
    """訓練與推論的端到端行為。"""

    def test_learns_a_deterministic_rule(self) -> None:
        features, y, names = synthetic_training_data()
        model = timing.fit(features, y, "t_day", names, num_rounds=150)
        chosen = decide(model, features)
        assert (chosen == y.to_numpy()).mean() > 0.95

    def test_underconfident_pmf_hedges_to_intermediate_grid_points(self) -> None:
        # 貝氏決策在信心不足時會刻意避險到兩個類別之間的格點——
        # 那個時刻在訓練資料中出現 0 次，而且必然「猜錯」，
        # 但它確實使期望損失更低（凸損失下的最適行為）。
        #
        # 實務意涵：模型的機率校準直接決定落點。同一組類別，
        # 訓練 40 輪（最大機率中位數約 0.72）會散到十幾個中間格點，
        # 訓練 150 輪（約 0.96）則精準落在兩個類別上。
        # 因此「提高信心」與「提高準確率」在此是同一件事。
        features, y, names = synthetic_training_data()
        shy = timing.fit(features, y, "t_day", names, num_rounds=40)
        confident = timing.fit(features, y, "t_day", names, num_rounds=150)

        shy_pmf = timing.predict_pmf(shy, features)
        confident_pmf = timing.predict_pmf(confident, features)
        assert shy_pmf.max(axis=1).mean() < confident_pmf.max(axis=1).mean()

        # 信心不足者落在類別集合外；信心足夠者不會。
        classes = set(shy.classes)
        shy_choices = set(decide(shy, features).tolist())
        confident_choices = set(decide(confident, features).tolist())
        assert shy_choices - classes
        assert not (confident_choices - classes)

        # 但兩者的眾數都幾乎全對——散開來自決策層，不是分類器學不會。
        for model, pmf in ((shy, shy_pmf), (confident, confident_pmf)):
            mode = np.array(model.classes)[pmf.argmax(axis=1)]
            assert (mode == y.to_numpy()).mean() > 0.95

    def test_pmf_rows_sum_to_one(self) -> None:
        features, y, names = synthetic_training_data()
        model = timing.fit(features, y, "t_day", names, num_rounds=20)
        pmf = timing.predict_pmf(model, features)
        assert np.allclose(pmf.sum(axis=1), 1.0)


    def test_prediction_lands_on_the_full_nominal_grid(self) -> None:
        # 決策的值域是完整名目格點，不是類別集合。
        features, y, names = synthetic_training_data()
        model = timing.fit(features, y, "t_day", names, num_rounds=20)
        grid = set(decision.candidate_grid("t_day"))
        assert set(decide(model, features).tolist()) <= grid

    def test_same_seed_is_reproducible(self) -> None:
        features, y, names = synthetic_training_data()
        a = decide(timing.fit(features, y, "t_day", names, seed=42, num_rounds=20), features)
        b = decide(timing.fit(features, y, "t_day", names, seed=42, num_rounds=20), features)
        assert np.array_equal(a, b)

    def test_single_class_does_not_crash(self) -> None:
        # 退化情形：訓練折只有一個時刻。必須能走完，不得拋錯。
        features, _, names = synthetic_training_data(n=50)
        y = pl.Series("t_day", [to_minutes("14:00")] * 50)
        model = timing.fit(features, y, "t_day", names, num_rounds=10)
        assert model.booster is None
        chosen = decide(model, features)
        assert set(chosen.tolist()) == {to_minutes("14:00")}


LEVELS_COARSE = (("daytype",),)


LEVELS_FINE = (("daytype",), ("daytype", "month"))


def hierarchical_history() -> pl.DataFrame:
    """兩個日別 × 兩個月份，各有明確不同的尖峰時刻。

    平日 1 月 → 16:00、平日 7 月 → 14:00、週六（不分月）→ 12:00。
    細分組能分開 1 月與 7 月，粗分組不能——這正是階層收縮要處理的情形。
    """
    rows = []
    for month, minute in ((1, "16:00"), (7, "14:00")):
        rows += [{"date": dt.date(2025, month, i + 1), "daytype": "平日",
                  "month": month, "t_day": float(to_minutes(minute))} for i in range(20)]
    rows += [{"date": dt.date(2025, 3, i + 1), "daytype": "週六",
              "month": 3, "t_day": float(to_minutes("12:00"))} for i in range(20)]
    return pl.DataFrame(rows)


class TestHierarchicalPmf:
    """階層收縮：頻率、先驗、與收縮強度的作用。"""

    def test_returns_full_nominal_grid(self) -> None:
        # 收縮後每個名目格點都有定義，不再受限於某組的觀察值域。
        pmf, classes = timing.hierarchical_pmf(
            hierarchical_history(), "t_day", LEVELS_COARSE, (("平日",),), 10.0
        )
        assert classes == decision.candidate_grid("t_day")
        assert pmf.shape == (1, settings.DAY_PEAK_N_POINTS)

    def test_rows_sum_to_one(self) -> None:
        pmf, _ = timing.hierarchical_pmf(
            hierarchical_history(), "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), 10.0
        )
        assert pmf.sum() == pytest.approx(1.0)

    def test_minutes_are_paired_with_their_own_counts(self) -> None:
        # 對 value_counts() 呼叫兩次、分別取值與計數是錯的：
        # polars 不保證兩次的列順序相同，時刻與計數會錯配，
        # 靜默算出一個完全錯誤但形狀正確的分布。
        # 此處用「峰值必須落在實際最常見的時刻上」把它釘住。
        history = hierarchical_history()
        pmf, classes = timing.hierarchical_pmf(
            history, "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), 0.0
        )
        assert classes[int(pmf[0].argmax())] == to_minutes("16:00")

        pmf_july, _ = timing.hierarchical_pmf(
            history, "t_day", LEVELS_FINE, (("平日",), ("平日", 7)), 0.0
        )
        assert classes[int(pmf_july[0].argmax())] == to_minutes("14:00")

    def test_zero_alpha_uses_only_the_finest_level(self) -> None:
        history = hierarchical_history()
        pmf, classes = timing.hierarchical_pmf(
            history, "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), 0.0
        )
        # 1 月平日全部是 16:00，α=0 時該格點機率應為 1。
        assert pmf[0][classes.index(to_minutes("16:00"))] == pytest.approx(1.0)

    def test_large_alpha_falls_back_to_the_prior(self) -> None:
        history = hierarchical_history()
        fine, classes = timing.hierarchical_pmf(
            history, "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), 0.0
        )
        shrunk, _ = timing.hierarchical_pmf(
            history, "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), 10_000.0
        )
        january = classes.index(to_minutes("16:00"))
        july = classes.index(to_minutes("14:00"))
        # 強收縮後 1 月的機率被拉低，而粗分組裡也有的 7 月機率被拉高。
        assert shrunk[0][january] < fine[0][january]
        assert shrunk[0][july] > fine[0][july]

    def test_shrinkage_is_monotonic_in_alpha(self) -> None:
        history = hierarchical_history()
        classes = decision.candidate_grid("t_day")
        january = classes.index(to_minutes("16:00"))
        peaks = [
            timing.hierarchical_pmf(
                history, "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), a
            )[0][0][january]
            for a in (0.0, 5.0, 20.0, 100.0)
        ]
        assert peaks == sorted(peaks, reverse=True)

    def test_empty_group_falls_back_to_coarser_level(self) -> None:
        # 12 月平日沒有任何樣本 → 該層跳過，結果應等於只用粗分組。
        history = hierarchical_history()
        missing, _ = timing.hierarchical_pmf(
            history, "t_day", LEVELS_FINE, (("平日",), ("平日", 12)), 10.0
        )
        coarse, _ = timing.hierarchical_pmf(
            history, "t_day", LEVELS_COARSE, (("平日",),), 10.0
        )
        assert np.allclose(missing, coarse)

    def test_no_grid_point_has_zero_probability(self) -> None:
        # 由均勻分布起步，故任何格點都保有正機率——貝氏決策才不會
        # 因為某個格點在歷史中出現 0 次就完全排除它。
        pmf, _ = timing.hierarchical_pmf(
            hierarchical_history(), "t_day", LEVELS_FINE, (("平日",), ("平日", 1)), 10.0
        )
        assert (pmf > 0).all()

    def test_different_groups_give_different_distributions(self) -> None:
        history = hierarchical_history()
        weekday, classes = timing.hierarchical_pmf(
            history, "t_day", LEVELS_COARSE, (("平日",),), 5.0
        )
        saturday, _ = timing.hierarchical_pmf(
            history, "t_day", LEVELS_COARSE, (("週六",),), 5.0
        )
        assert classes[int(weekday[0].argmax())] != classes[int(saturday[0].argmax())]
        assert classes[int(saturday[0].argmax())] == to_minutes("12:00")
