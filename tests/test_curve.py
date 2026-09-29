"""曲線合成：合成的曲線以 ``compute_targets`` 重新推導，必須與目標值完全相等。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from config import paths, settings
from src.features import targets as T
from src.features.targets import to_minutes
from src.models import curve


def make_targets(
    p_day: float = 38000.0,
    t_day: str = "14:00",
    p_night: float = 35000.0,
    t_night: str = "18:30",
    ramp_up: float = 1200.0,
    ramp_down: float = 800.0,
) -> curve.DayTargets:
    return curve.DayTargets(
        p_day, to_minutes(t_day), p_night, to_minutes(t_night), ramp_up, ramp_down
    )


def build_and_extract(
    targets_list: list[curve.DayTargets], suggested_ratio: tuple[float, float, float] = (0.68, 0.60, 0.66)
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """合成多天曲線，並以權威函式重新推導目標。

    Returns:
        tuple: ``(推導結果, 意圖值)``，皆依 ``date`` 排序。
    """
    rows, intent = [], []
    for i, targets in enumerate(targets_list):
        day = dt.date(2030, 1, 1) + dt.timedelta(days=i)
        suggested = curve.FreeLevels(
            targets.p_day * suggested_ratio[0],
            targets.p_day * suggested_ratio[1],
            targets.p_day * suggested_ratio[2],
        )
        levels, _ = curve.feasible_levels(targets, suggested)
        values = curve.synthesize_day_checked(
            targets, levels.start, levels.minimum, levels.end
        )
        rows += [
            {
                "ts": dt.datetime.combine(day, dt.time()) + dt.timedelta(minutes=10 * k),
                "Load_MW": float(v),
                "is_imputed": False,
            }
            for k, v in enumerate(values)
        ]
        intent.append({"date": day, **{n: float(getattr(targets, n)) for n in T.TARGET_NAMES}})
    return (
        T.compute_targets(pl.DataFrame(rows)).sort("date"),
        pl.DataFrame(intent).sort("date"),
    )


class TestDayTargetsValidation:
    """目標值本身的合法性。"""

    def test_accepts_legal_targets(self) -> None:
        make_targets().validate()

    def test_rejects_off_grid_day_time(self) -> None:
        with pytest.raises(ValueError, match="不在日尖峰格點上"):
            make_targets(t_day="10:50").validate()


    def test_rejects_nonpositive_ramp(self) -> None:
        with pytest.raises(ValueError, match="必須為正值"):
            make_targets(ramp_up=0.0).validate()


class TestAuthoritativeRoundTrip:
    """合成 → compute_targets 推導 → 必須完全相等。"""

    def test_single_typical_day(self) -> None:
        extracted, intent = build_and_extract([make_targets()])
        for name in T.TARGET_NAMES:
            assert extracted[name][0] == pytest.approx(intent[name][0], abs=1e-3)

    def test_peak_times_match_exactly(self) -> None:
        # 時刻不容許任何誤差——差一格就是 (10/10)^1.2 = 1 分的損失。
        # t_day=17:00 時，下一步（17:10）就進入夜窗口，該點必須已低於
        # p_night，否則會因時間較早而搶走 t_night。故此時必然要求
        # p_night > p_day − ramp_down，見 TestJointFeasibility。
        cases = [
            make_targets(t_day=d, t_night=n, p_night=37500.0 if d == "17:00" else 35000.0)
            for d in ("11:00", "13:20", "17:00")
            for n in ("17:10", "19:00", "21:00")
        ]
        extracted, intent = build_and_extract(cases)
        assert extracted["t_day"].to_list() == intent["t_day"].cast(pl.Int32).to_list()
        assert extracted["t_night"].to_list() == intent["t_night"].cast(pl.Int32).to_list()

    def test_night_peak_above_day_peak(self) -> None:
        # 實測「週日及離峰日」常見：夜尖峰高於日尖峰。
        extracted, intent = build_and_extract(
            [make_targets(p_day=24500.0, p_night=26000.0, t_day="17:00", t_night="20:20",
                          ramp_up=330.0, ramp_down=400.0)]
        )
        for name in T.TARGET_NAMES:
            assert extracted[name][0] == pytest.approx(intent[name][0], abs=1e-3)

    @pytest.mark.skipif(not paths.TARGETS_FILE.exists(), reason="需要先重建 data/processed/")
    def test_every_real_day_round_trips(self) -> None:
        # 以真實資料的 6 項目標為輸入，全部必須可合成且精確還原。
        daily = pl.read_parquet(paths.TARGETS_FILE)
        sample = daily.filter(pl.col("n_points") == 144).sample(60, seed=7)
        cases = [
            curve.DayTargets(r["p_day"], int(r["t_day"]), r["p_night"],
                             int(r["t_night"]), r["ramp_up"], r["ramp_down"])
            for r in sample.iter_rows(named=True)
        ]
        extracted, intent = build_and_extract(cases)
        for name in ("t_day", "t_night"):
            assert extracted[name].to_list() == intent[name].cast(pl.Int32).to_list()
        for name in ("p_day", "p_night", "ramp_up", "ramp_down"):
            assert np.allclose(
                extracted[name].to_numpy(), intent[name].to_numpy(), atol=1e-3
            )


class TestPeakUniqueness:
    """並列時取最早——合成曲線必須讓極大值嚴格唯一。"""

    def test_day_peak_is_strictly_unique(self) -> None:
        targets = make_targets()
        levels, _ = curve.feasible_levels(
            targets, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        values = curve.synthesize_day(targets, levels.start, levels.minimum, levels.end)
        lo, hi = to_minutes(settings.DAY_PEAK_START) // 10, to_minutes(settings.DAY_PEAK_END) // 10
        window = values[lo : hi + 1]
        assert int((window == window.max()).sum()) == 1

    def test_night_peak_is_strictly_unique(self) -> None:
        targets = make_targets()
        levels, _ = curve.feasible_levels(
            targets, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        values = curve.synthesize_day(targets, levels.start, levels.minimum, levels.end)
        lo = to_minutes(settings.NIGHT_PEAK_START) // 10
        hi = to_minutes(settings.NIGHT_PEAK_END) // 10
        window = values[lo : hi + 1]
        assert int((window == window.max()).sum()) == 1


class TestRegressions:
    """四個已修正的缺陷，各釘一條。前三者會靜默失分，第四者會直接崩潰。"""

    def test_zero_rest_step_is_feasible_despite_float_noise(self) -> None:
        # 缺陷 4：當某段「除了指定那一步之外其餘都不動」時，
        # 其餘步的變化量在數學上是 0，但浮點運算常給出 ±1e-14。
        # 而可行性檢查寫的是嚴格的 `rest <= 0`，於是把完全可行的情形判為不可行並拋錯。
        # 實際踩過：接上氣象特徵後量值預測改變，某天恰好落在這條邊界上，
        # 整次 60 折 CV 因此中斷——不是失分，是直接崩潰。
        n_steps, extreme = 40, -811.98
        for wobble in (0.0, 1e-13, -1e-13):
            diffs = curve._monotone_with_one_extreme_step(
                n_steps=n_steps,
                total_change=extreme + wobble,
                extreme_step=extreme,
                extreme_at=3,
            )
            assert len(diffs) == n_steps
            assert diffs[3] == pytest.approx(extreme)
            # 指定步仍必須是全段唯一的最大下降。
            assert diffs.min() == pytest.approx(extreme)
            assert sorted(diffs)[1] == pytest.approx(0.0, abs=1e-9)

    def test_tail_decline_does_not_exceed_ramp_down(self) -> None:
        # 缺陷 1：初版把夜尖峰後壓平到 21:00，使收尾 16 步各降 635 MW，
        # 超過意圖的 ramp_down 627.9。
        targets = make_targets(
            p_day=38807.8, t_day="14:10", p_night=35796.2, t_night="17:10",
            ramp_up=1331.0, ramp_down=627.9,
        )
        extracted, intent = build_and_extract([targets])
        assert extracted["ramp_down"][0] == pytest.approx(intent["ramp_down"][0], abs=1e-3)

    def test_no_cliff_at_night_window_boundary(self) -> None:
        # 缺陷 2：由 p_day 直線降到 t_night 時，進入夜窗口仍高於 p_night，
        # 被壓平後在窗口邊界留下 964 MW 的懸崖（意圖 ramp_down 只有 616）。
        targets = make_targets(
            p_day=32000.0, t_day="15:30", p_night=30675.0, t_night="21:00",
            ramp_up=900.0, ramp_down=616.2,
        )
        extracted, intent = build_and_extract([targets])
        assert extracted["ramp_down"][0] == pytest.approx(intent["ramp_down"][0], abs=1e-3)

    def test_rise_to_higher_night_peak_respects_ramp_up(self) -> None:
        # 缺陷 3：p_night > p_day 時，上升只能發生在日窗口結束之後，
        # 初版誤以為可攤在多步上，算出 2 倍於意圖的 ramp_up。
        targets = make_targets(
            p_day=24530.0, t_day="17:00", p_night=25953.0, t_night="20:20",
            ramp_up=323.8, ramp_down=500.0,
        )
        extracted, intent = build_and_extract([targets])
        assert extracted["ramp_up"][0] == pytest.approx(intent["ramp_up"][0], abs=1e-3)


class TestFeasibleLevels:
    """自由水準的夾取。"""

    def test_leaves_feasible_suggestion_untouched(self) -> None:
        targets = make_targets()
        suggested = curve.FreeLevels(26000.0, 25000.0, 30000.0)
        levels, notes = curve.feasible_levels(targets, suggested)
        assert notes == []
        assert levels == suggested

    def test_clamps_and_reports(self) -> None:
        # 谷底設得比 p_day 還高 → 必須被夾回並回報。
        targets = make_targets()
        levels, notes = curve.feasible_levels(
            targets, curve.FreeLevels(26000.0, 99000.0, 30000.0)
        )
        assert notes
        assert levels.minimum < targets.p_day

    def test_clamped_levels_always_synthesize(self) -> None:
        # 夾取後必須保證可合成——這是 feasible_levels 存在的意義。
        targets = make_targets()
        for suggestion in (
            curve.FreeLevels(1.0, 1.0, 1.0),
            curve.FreeLevels(1e6, 1e6, 1e6),
            curve.FreeLevels(38000.0, 37999.0, 100.0),
        ):
            levels, _ = curve.feasible_levels(targets, suggestion)
            curve.synthesize_day_checked(targets, levels.start, levels.minimum, levels.end)


class TestInfeasibleCombinations:
    """互相衝突的目標值必須明確拋錯，不得靜默產生錯誤曲線。"""

    def test_tiny_ramp_still_synthesizes_by_flattening(self) -> None:
        # ramp_up 極小不是不可行——自由水準會被夾成一條幾乎平坦的高原，
        # 6 項目標仍然精確達成。這正是 feasible_levels 存在的價值，
        # 故這裡驗證「不拋錯且往返一致」，而不是驗證拋錯。
        extracted, intent = build_and_extract([make_targets(ramp_up=1.0)])
        for name in T.TARGET_NAMES:
            assert extracted[name][0] == pytest.approx(intent[name][0], abs=1e-3)

    def test_adjacent_peaks_with_impossible_jump(self) -> None:
        # t_day=17:00、t_night=17:10 只隔一步，卻要求跳 5000 MW
        # 而 ramp 只有 300 → 不可能。
        targets = make_targets(
            p_day=30000.0, t_day="17:00", p_night=35000.0, t_night="17:10",
            ramp_up=300.0, ramp_down=300.0,
        )
        levels, _ = curve.feasible_levels(
            targets, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        with pytest.raises(ValueError, match="需超過"):
            curve.synthesize_day_checked(targets, levels.start, levels.minimum, levels.end)


class TestJointFeasibility:
    """獨立預測的 6 個目標值可能互相衝突，合成必須明確拒絕。

    真實資料不會有這問題（ramp_down 本來就是實際曲線的最大降幅），
    但我們的 6 個目標是各自獨立預測出來的，可能組不成任何一條曲線。
    """

    def test_late_day_peak_forces_night_peak_to_be_close(self) -> None:
        # t_day=17:00 → 下一步 17:10 已在夜窗口內，必須低於 p_night，
        # 故單步降幅需 ≥ p_day − p_night。若超過 ramp_down 就無解。
        infeasible = make_targets(
            p_day=38000.0, t_day="17:00", p_night=35000.0, t_night="19:00",
            ramp_down=800.0,
        )
        levels, _ = curve.feasible_levels(
            infeasible, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        with pytest.raises(ValueError, match="傍晚下凹無可行值"):
            curve.synthesize_day_checked(
                infeasible, levels.start, levels.minimum, levels.end
            )

    def test_same_combination_becomes_feasible_with_larger_ramp_down(self) -> None:
        # 同一組時刻與量值，只要 ramp_down 夠大就可行——
        # 這說明衝突的是「組合」而非任何單一目標值。
        feasible = make_targets(
            p_day=38000.0, t_day="17:00", p_night=35000.0, t_night="19:00",
            ramp_down=3200.0,
        )
        extracted, intent = build_and_extract([feasible])
        for name in T.TARGET_NAMES:
            assert extracted[name][0] == pytest.approx(intent[name][0], abs=1e-3)

    def test_error_message_names_the_conflicting_quantities(self) -> None:
        # 訊息必須指出是哪些量在衝突，否則上游無從修補。
        infeasible = make_targets(
            p_day=38000.0, t_day="17:00", p_night=35000.0, t_night="19:00",
            ramp_down=800.0,
        )
        levels, _ = curve.feasible_levels(
            infeasible, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        with pytest.raises(ValueError) as caught:
            curve.synthesize_day_checked(
                infeasible, levels.start, levels.minimum, levels.end
            )
        message = str(caught.value)
        assert "ramp_down" in message and "p_night" in message


class TestVerifyDay:
    """模組內部的快速驗證函式，須與權威推導一致。"""

    def test_agrees_with_compute_targets(self) -> None:
        targets = make_targets()
        levels, _ = curve.feasible_levels(
            targets, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        values = curve.synthesize_day(targets, levels.start, levels.minimum, levels.end)
        quick = curve.verify_day(values, targets)
        extracted, _ = build_and_extract([targets])
        for name in T.TARGET_NAMES:
            assert quick[name] == pytest.approx(float(extracted[name][0]), abs=1e-3)


    def test_checked_wrapper_reports_every_mismatch(self) -> None:
        # 直接呼叫 synthesize_day 不做檢查；checked 版本必須攔下並指出項目。
        targets = make_targets(
            p_day=30000.0, t_day="17:00", p_night=35000.0, t_night="17:10",
            ramp_up=300.0, ramp_down=300.0,
        )
        levels, _ = curve.feasible_levels(
            targets, curve.FreeLevels(26000.0, 23000.0, 25000.0)
        )
        with pytest.raises(ValueError, match="需超過"):
            curve.synthesize_day_checked(targets, levels.start, levels.minimum, levels.end)
