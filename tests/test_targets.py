"""6 項目標標籤：窗口邊界、不跨日、絕對值、並列取最早、補值污染旗標。
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from config import settings
from src.features import targets


def make_day(
    date: dt.date,
    loads: list[float],
    imputed: list[bool] | None = None,
) -> pl.DataFrame:
    """構造單日 144 點序列。

    Args:
        date: 日期。
        loads: 144 個負載值，索引 i 對應時刻 i × 10 分鐘。
        imputed: 對應的補值旗標，None 時全為 False。

    Returns:
        pl.DataFrame: 欄位 ``ts``、``Load_MW``、``is_imputed``。
    """
    assert len(loads) == settings.POINTS_PER_DAY
    base = dt.datetime(date.year, date.month, date.day)
    return pl.DataFrame(
        {
            "ts": [
                base + dt.timedelta(minutes=settings.DATA_FREQ_MIN * i)
                for i in range(settings.POINTS_PER_DAY)
            ],
            "Load_MW": loads,
            "is_imputed": imputed or [False] * settings.POINTS_PER_DAY,
        }
    )


def slot(hhmm: str) -> int:
    """``"11:00"`` → 該時刻在當日 144 點中的索引。"""
    return targets.to_minutes(hhmm) // settings.DATA_FREQ_MIN


DAY = dt.date(2024, 3, 6)  # 星期三


class TestTimeHelpers:
    """時刻字串與分鐘數的互轉。"""

    @pytest.mark.parametrize(
        ("hhmm", "minutes"), [("00:00", 0), ("09:00", 540), ("11:00", 660), ("23:50", 1430)]
    )
    def test_round_trip(self, hhmm: str, minutes: int) -> None:
        assert targets.to_minutes(hhmm) == minutes
        assert targets.format_hhmm(minutes) == hhmm

    def test_format_pads_to_two_digits(self) -> None:
        # 提交檔要求零填補字串；"9:00" 會被輸出驗證擋下。
        assert targets.format_hhmm(540) == "09:00"

    def test_grid_sizes_and_endpoints(self) -> None:
        day, night = targets.day_peak_grid(), targets.night_peak_grid()
        assert len(day) == settings.DAY_PEAK_N_POINTS == 37
        assert len(night) == settings.NIGHT_PEAK_N_POINTS == 24
        assert targets.format_hhmm(day[0]) == "11:00"
        assert targets.format_hhmm(day[-1]) == "17:00"
        assert targets.format_hhmm(night[0]) == "17:10"
        assert targets.format_hhmm(night[-1]) == "21:00"


class TestPeakWindows:
    """尖峰只能在各自的窗口內取極值。"""

    def test_day_peak_ignores_larger_value_outside_window(self) -> None:
        loads = [1000.0] * settings.POINTS_PER_DAY
        loads[slot("09:00")] = 9999.0  # 窗口外的全日最大值，不得被選中
        loads[slot("14:00")] = 5000.0  # 窗口內的最大值
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["p_day"][0] == 5000.0
        assert out["t_day"][0] == targets.to_minutes("14:00")

    def test_night_peak_ignores_value_outside_window(self) -> None:
        loads = [1000.0] * settings.POINTS_PER_DAY
        loads[slot("22:00")] = 9999.0  # 夜尖峰窗口之後
        loads[slot("18:30")] = 5000.0
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["p_night"][0] == 5000.0
        assert out["t_night"][0] == targets.to_minutes("18:30")

    def test_day_peak_at_window_upper_edge(self) -> None:
        # 17:00 是日尖峰的右端點，必須被視為合法（實測 10.7% 的日子落在此）。
        loads = [float(i) for i in range(settings.POINTS_PER_DAY)]  # 全日單調遞增
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["t_day"][0] == targets.to_minutes("17:00")
        assert out["t_night"][0] == targets.to_minutes("21:00")

    def test_night_peak_at_window_lower_edge(self) -> None:
        # 17:10 是夜尖峰的左端點（實測夏季平日 84–91% 落在此）。
        loads = [float(settings.POINTS_PER_DAY - i) for i in range(settings.POINTS_PER_DAY)]
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["t_night"][0] == targets.to_minutes("17:10")

    def test_ties_take_earliest(self) -> None:
        loads = [1000.0] * settings.POINTS_PER_DAY
        loads[slot("13:00")] = 5000.0
        loads[slot("15:00")] = 5000.0  # 同值並列
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["t_day"][0] == targets.to_minutes("13:00")


class TestRamp:
    """ramp 為 143 個相鄰差分、不跨日、ramp_down 取絕對值。"""

    def test_ramp_values_and_times(self) -> None:
        # 用階梯而非尖刺：單點尖刺會同時製造一次上升與一次等幅的回彈，
        # 使 ramp_up 與 ramp_down 互相干擾，測不出想測的東西。
        loads = (
            [1000.0] * slot("08:10")
            + [1500.0] * (slot("12:10") - slot("08:10"))  # 08:10 上升 500 後維持
            + [700.0] * (settings.POINTS_PER_DAY - slot("12:10"))  # 12:10 下降 800
        )
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["ramp_up"][0] == 500.0
        assert out["t_ramp_up"][0] == targets.to_minutes("08:10")
        assert out["ramp_down"][0] == 800.0  # 絕對值，正值
        assert out["t_ramp_down"][0] == targets.to_minutes("12:10")

    def test_ramp_down_is_positive(self) -> None:
        loads = [float(settings.POINTS_PER_DAY - i) for i in range(settings.POINTS_PER_DAY)]
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["ramp_down"][0] > 0

    def test_ramp_does_not_cross_days(self) -> None:
        # 第一天結尾 1000、第二天開頭 50000。若差分跨日，會產生一個 49000 的
        # 假 ramp_up；不跨日的話兩天的 ramp_up 都應該只有 100。
        day1 = make_day(DAY, [1000.0] * 143 + [1100.0])
        day2 = make_day(DAY + dt.timedelta(days=1), [50000.0] * 143 + [50100.0])
        out = targets.compute_targets(pl.concat([day1, day2]))
        assert out.height == 2
        assert out["ramp_up"].to_list() == [100.0, 100.0]

    def test_diff_count_is_143(self) -> None:
        # 每日 144 點 → 143 個差分。以等差序列驗證：所有差分皆為 1，
        # 故 ramp_up 為 1、ramp_down 也為 1（最小差分同為 1）。
        loads = [float(i) for i in range(settings.POINTS_PER_DAY)]
        out = targets.compute_targets(make_day(DAY, loads))
        assert settings.RAMP_N_DIFFS == settings.POINTS_PER_DAY - 1 == 143
        assert out["ramp_up"][0] == 1.0
        assert out["ramp_down"][0] == 1.0


class TestImputationFlags:
    """補值污染旗標：差分只要碰到補值點就算受污染。"""

    def test_ramp_on_imputed_when_extremum_touches_imputed_point(self) -> None:
        loads = [1000.0] * settings.POINTS_PER_DAY
        loads[slot("08:10")] = 1500.0
        imputed = [False] * settings.POINTS_PER_DAY
        imputed[slot("08:10")] = True  # 極值本身是補值點
        out = targets.compute_targets(make_day(DAY, loads, imputed))
        assert out["ramp_up_on_imputed"][0] is True
        assert out["n_imputed"][0] == 1
        assert out["has_imputed"][0] is True

    def test_ramp_on_imputed_when_left_endpoint_imputed(self) -> None:
        # 差分是兩點之差，左端補值同樣算污染。
        loads = [1000.0] * settings.POINTS_PER_DAY
        loads[slot("08:10")] = 1500.0
        imputed = [False] * settings.POINTS_PER_DAY
        imputed[slot("08:00")] = True
        out = targets.compute_targets(make_day(DAY, loads, imputed))
        assert out["ramp_up_on_imputed"][0] is True

    def test_clean_day_is_not_flagged(self) -> None:
        loads = [1000.0] * settings.POINTS_PER_DAY
        loads[slot("08:10")] = 1500.0
        out = targets.compute_targets(make_day(DAY, loads))
        assert out["ramp_up_on_imputed"][0] is False
        assert out["has_imputed"][0] is False


class TestDaytypeAndFlags:
    """日別、邊界審查、ramp regime 三類結構欄位。"""

    @pytest.mark.parametrize(
        ("date", "expected"),
        [
            (dt.date(2024, 3, 6), "平日"),  # 星期三
            (dt.date(2024, 3, 9), "週六"),
            (dt.date(2024, 3, 10), "週日"),
        ],
    )
    def test_daytype(self, date: dt.date, expected: str) -> None:
        daily = targets.add_daytype(pl.DataFrame({"date": [date]}))
        assert daily["daytype"][0] == expected

    def test_censoring_flags(self) -> None:
        daily = pl.DataFrame(
            {
                "t_day": [
                    targets.to_minutes("11:00"),
                    targets.to_minutes("14:00"),
                    targets.to_minutes("17:00"),
                ],
                "t_night": [
                    targets.to_minutes("17:10"),
                    targets.to_minutes("18:00"),
                    targets.to_minutes("21:00"),
                ],
            }
        )
        out = targets.add_censoring_flags(daily)
        assert out["t_day_censored"].to_list() == ["lower", None, "upper"]
        assert out["t_night_censored"].to_list() == ["lower", None, "upper"]

    def test_ramp_regime_flags(self) -> None:
        daily = pl.DataFrame(
            {
                "t_ramp_up": [targets.to_minutes(t) for t in ("08:10", "12:00", "03:00")],
                "t_ramp_down": [targets.to_minutes(t) for t in ("12:10", "17:00", "03:00")],
            }
        )
        out = targets.add_ramp_regime_flags(daily)
        assert out["ramp_up_regime"].to_list() == ["morning", "midday", "other"]
        assert out["ramp_down_regime"].to_list() == ["midday", "evening", "other"]
