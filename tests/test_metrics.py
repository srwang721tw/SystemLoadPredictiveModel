"""評分函數 ``total_score``：每一項都以手算案例綁住。
"""

from __future__ import annotations

import polars as pl
import pytest

from config import settings
from src.evaluation import metrics
from src.features.targets import TARGET_NAMES, to_minutes


PERFECT = {
    "p_day": 30000.0,
    "t_day": float(to_minutes("14:00")),
    "p_night": 28000.0,
    "t_night": float(to_minutes("18:00")),
    "ramp_up": 1000.0,
    "ramp_down": 800.0,
}


def frame(**overrides: float) -> pl.DataFrame:
    """以 :data:`PERFECT` 為基礎、覆寫指定欄位，構造單列評分表。"""
    return pl.DataFrame({k: [overrides.get(k, v)] for k, v in PERFECT.items()})


class TestPerfectPrediction:
    """完美預測的總分必須恰為 0。"""

    def test_all_components_zero(self) -> None:
        b = metrics.score_breakdown(frame(), frame())
        assert b.s_peak_mw == 0.0
        assert b.s_peak_time == 0.0
        assert b.s_ramp_up == 0.0
        assert b.s_ramp_down == 0.0
        assert b.s_under_penalty == 0.0
        assert b.total_score == 0.0


class TestPeakMagnitude:
    """尖峰負載量：相對誤差、低估懲罰、以及兩者的不對稱性。"""

    def test_underestimate_p_day_by_5pct(self) -> None:
        # p_day 低估 5%：s_peak_mw = (0.05 + 0) / 2 = 0.025
        #                penalty  = 0.05 × 0.2 = 0.01
        #                total    = 0.6 × 0.025 + 0.01 = 0.025
        pred = frame(p_day=PERFECT["p_day"] * 0.95)
        b = metrics.score_breakdown(frame(), pred)
        assert b.s_peak_mw == pytest.approx(0.025)
        assert b.s_under_penalty == pytest.approx(0.01)
        assert b.total_score == pytest.approx(0.6 * 0.025 + 0.01)

    def test_overestimate_p_day_by_5pct_has_no_penalty(self) -> None:
        # 高估同樣 5%：誤差項相同，但完全沒有懲罰。
        pred = frame(p_day=PERFECT["p_day"] * 1.05)
        b = metrics.score_breakdown(frame(), pred)
        assert b.s_peak_mw == pytest.approx(0.025)
        assert b.s_under_penalty == 0.0
        assert b.total_score == pytest.approx(0.6 * 0.025)

    def test_underestimate_costs_more_than_overestimate(self) -> None:
        # 這個不對稱性正是 τ = 0.625 的來源，方向錯了整套分位數策略就反了。
        under = metrics.total_score(frame(), frame(p_day=PERFECT["p_day"] * 0.95))
        over = metrics.total_score(frame(), frame(p_day=PERFECT["p_day"] * 1.05))
        assert under > over

    def test_marginal_cost_ratio_matches_tau_peak(self) -> None:
        # 低估與高估的邊際成本比，必須與 settings.TAU_PEAK 的推導一致。
        eps = 0.01
        under = metrics.total_score(frame(), frame(p_day=PERFECT["p_day"] * (1 - eps)))
        over = metrics.total_score(frame(), frame(p_day=PERFECT["p_day"] * (1 + eps)))
        c_under, c_over = under / eps, over / eps
        assert c_under / (c_under + c_over) == pytest.approx(settings.TAU_PEAK)

    def test_night_peak_contributes_equally(self) -> None:
        day = metrics.total_score(frame(), frame(p_day=PERFECT["p_day"] * 0.95))
        night = metrics.total_score(frame(), frame(p_night=PERFECT["p_night"] * 0.95))
        assert day == pytest.approx(night)


class TestPeakTime:
    """尖峰時間：以分鐘計、除以 10 換算格數、取 1.2 次方。"""

    def test_one_grid_error(self) -> None:
        # 差 1 格（10 分鐘）：(10/10)^1.2 = 1；只有 t_day 錯 → /2
        pred = frame(t_day=PERFECT["t_day"] + settings.DATA_FREQ_MIN)
        b = metrics.score_breakdown(frame(), pred)
        assert b.s_peak_time == pytest.approx(0.5)

    def test_three_grid_error(self) -> None:
        # 差 3 格（30 分鐘）：3^1.2 = 3.7372；/2
        pred = frame(t_day=PERFECT["t_day"] + 30)
        b = metrics.score_breakdown(frame(), pred)
        assert b.s_peak_time == pytest.approx(3**1.2 / 2)

    def test_direction_does_not_matter(self) -> None:
        # 時間誤差取絕對值，早到與晚到等價（不像負載量有低估懲罰）。
        early = metrics.total_score(frame(), frame(t_day=PERFECT["t_day"] - 30))
        late = metrics.total_score(frame(), frame(t_day=PERFECT["t_day"] + 30))
        assert early == pytest.approx(late)


class TestRamp:
    """ramp_up 有低估懲罰、ramp_down 沒有。"""

    def test_ramp_up_underestimate_has_penalty(self) -> None:
        pred = frame(ramp_up=PERFECT["ramp_up"] * 0.9)
        b = metrics.score_breakdown(frame(), pred)
        assert b.s_ramp_up == pytest.approx(0.1)
        assert b.s_under_penalty == pytest.approx(0.1 * 0.2)

    def test_ramp_down_underestimate_has_no_penalty(self) -> None:
        # ramp_down 不適用低估懲罰。
        pred = frame(ramp_down=PERFECT["ramp_down"] * 0.9)
        b = metrics.score_breakdown(frame(), pred)
        assert b.s_ramp_down == pytest.approx(0.1)
        assert b.s_under_penalty == 0.0

    def test_ramp_down_is_symmetric(self) -> None:
        under = metrics.total_score(frame(), frame(ramp_down=PERFECT["ramp_down"] * 0.9))
        over = metrics.total_score(frame(), frame(ramp_down=PERFECT["ramp_down"] * 1.1))
        assert under == pytest.approx(over)

    def test_ramp_up_marginal_cost_ratio_matches_tau(self) -> None:
        eps = 0.01
        under = metrics.total_score(frame(), frame(ramp_up=PERFECT["ramp_up"] * (1 - eps)))
        over = metrics.total_score(frame(), frame(ramp_up=PERFECT["ramp_up"] * (1 + eps)))
        c_under, c_over = under / eps, over / eps
        assert c_under / (c_under + c_over) == pytest.approx(settings.TAU_RAMP_UP)

    def test_ramp_down_marginal_cost_ratio_is_half(self) -> None:
        eps = 0.01
        under = metrics.total_score(frame(), frame(ramp_down=PERFECT["ramp_down"] * (1 - eps)))
        over = metrics.total_score(frame(), frame(ramp_down=PERFECT["ramp_down"] * (1 + eps)))
        c_under, c_over = under / eps, over / eps
        assert c_under / (c_under + c_over) == pytest.approx(settings.TAU_RAMP_DOWN)


class TestAggregationOverDays:
    """N > 1 時各項確實取平均。"""

    def test_mean_over_three_days(self) -> None:
        truth = pl.concat([frame(), frame(), frame()])
        # 只有第一天的 p_day 低估 6%，三天平均 → 0.02
        pred = pl.concat([frame(p_day=PERFECT["p_day"] * 0.94), frame(), frame()])
        b = metrics.score_breakdown(truth, pred)
        assert b.s_peak_mw == pytest.approx(0.06 / 3 / 2)
        assert b.s_under_penalty == pytest.approx(0.06 * 0.2 / 3)

    def test_total_equals_weighted_sum_of_components(self) -> None:
        truth = pl.concat([frame(), frame()])
        pred = pl.concat(
            [
                frame(p_day=29000.0, t_day=PERFECT["t_day"] + 20, ramp_up=950.0),
                frame(p_night=27000.0, t_night=PERFECT["t_night"] - 30, ramp_down=850.0),
            ]
        )
        b = metrics.score_breakdown(truth, pred)
        expected = (
            settings.W_PEAK_MW * b.s_peak_mw
            + settings.W_PEAK_TIME * b.s_peak_time
            + settings.W_RAMP_UP * b.s_ramp_up
            + settings.W_RAMP_DOWN * b.s_ramp_down
            + b.s_under_penalty
        )
        assert b.total_score == pytest.approx(expected)


class TestScaleAsymmetry:
    """權重不等於重要性：時刻項與負載量項的數量級差約 60 倍。"""

    def test_one_grid_time_error_outweighs_five_pct_mw_error(self) -> None:
        # 時刻只差 1 格（10 分鐘），負載量差 5% —— 前者的加權貢獻仍然更大。
        # 這是本專案調校優先序的關鍵事實，故用測試釘住。
        time_only = metrics.total_score(
            frame(), frame(t_day=PERFECT["t_day"] + settings.DATA_FREQ_MIN)
        )
        mw_only = metrics.total_score(frame(), frame(p_day=PERFECT["p_day"] * 1.05))
        assert time_only > mw_only

    def test_contribution_breakdown_shares_sum_to_one(self) -> None:
        pred = frame(
            p_day=29000.0, t_day=PERFECT["t_day"] + 30, ramp_up=900.0, ramp_down=850.0
        )
        b = metrics.score_breakdown(frame(), pred)
        table = metrics.contribution_breakdown(b)
        assert table["share"].sum() == pytest.approx(1.0)
        assert table["contribution"].sum() == pytest.approx(b.total_score)


class TestInputValidation:
    """壞輸入必須明確拋錯，不得靜默算出一個數字。"""

    def test_row_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="列數不符"):
            metrics.total_score(pl.concat([frame(), frame()]), frame())

    def test_empty_input(self) -> None:
        with pytest.raises(ValueError, match="空的評分輸入"):
            metrics.total_score(frame().head(0), frame().head(0))

    def test_missing_column(self) -> None:
        with pytest.raises(ValueError, match="缺少欄位"):
            metrics.total_score(frame().drop("ramp_up"), frame().drop("ramp_up"))

    def test_null_value(self) -> None:
        bad = frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("p_day"))
        with pytest.raises(ValueError, match="含 null"):
            metrics.total_score(frame(), bad)

    def test_zero_denominator(self) -> None:
        # 相對誤差的分母為實際值；為 0 時無定義，必須拋錯而非回傳 inf。
        with pytest.raises(ValueError, match="含 0"):
            metrics.total_score(frame(ramp_up=0.0), frame())

    def test_all_targets_are_covered(self) -> None:
        # 6 項目標必須全部參與評分，漏掉任何一項都應該讓分數改變。
        for name in TARGET_NAMES:
            delta = 10.0 if name.startswith("t_") else PERFECT[name] * 0.1
            pred = frame(**{name: PERFECT[name] + delta})
            assert metrics.total_score(frame(), pred) > 0, f"{name} 未參與評分"
