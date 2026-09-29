"""設定常數與評分規則的一致性：時間格點、權重、由評分公式推導的分位數、提交格式。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from config import settings


def _grid_points(start: str, end: str, freq_min: int) -> list[str]:
    """列出 [start, end] 區間內（含端點）的所有時間格點。"""
    t0 = datetime.strptime(start, "%H:%M")
    t1 = datetime.strptime(end, "%H:%M")
    points: list[str] = []
    cur = t0
    while cur <= t1:
        points.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=freq_min)
    return points


class TestTimeStructure:
    """時間結構常數。"""

    def test_points_per_day(self) -> None:
        assert settings.POINTS_PER_DAY == 24 * 60 // settings.DATA_FREQ_MIN

    def test_ramp_diff_count_does_not_cross_days(self) -> None:
        assert settings.RAMP_N_DIFFS == settings.POINTS_PER_DAY - 1

    def test_day_peak_has_37_grid_points(self) -> None:
        grid = _grid_points(
            settings.DAY_PEAK_START, settings.DAY_PEAK_END, settings.DATA_FREQ_MIN
        )
        assert len(grid) == settings.DAY_PEAK_N_POINTS == 37
        assert grid[0] == "11:00"
        assert grid[-1] == "17:00"

    def test_night_peak_has_24_grid_points(self) -> None:
        grid = _grid_points(
            settings.NIGHT_PEAK_START, settings.NIGHT_PEAK_END, settings.DATA_FREQ_MIN
        )
        assert len(grid) == settings.NIGHT_PEAK_N_POINTS == 24
        assert grid[0] == "17:10"
        assert grid[-1] == "21:00"

    def test_peak_windows_do_not_overlap(self) -> None:
        day = set(
            _grid_points(
                settings.DAY_PEAK_START, settings.DAY_PEAK_END, settings.DATA_FREQ_MIN
            )
        )
        night = set(
            _grid_points(
                settings.NIGHT_PEAK_START,
                settings.NIGHT_PEAK_END,
                settings.DATA_FREQ_MIN,
            )
        )
        assert day & night == set()


class TestScoring:
    """評分權重與由其逆推的分位數。"""

    def test_weights_sum_to_one(self) -> None:
        total = (
            settings.W_PEAK_MW
            + settings.W_PEAK_TIME
            + settings.W_RAMP_UP
            + settings.W_RAMP_DOWN
        )
        assert total == 1.0

    def test_tau_peak_derivation(self) -> None:
        # 尖峰負載量共兩項，各佔 W_PEAK_MW 的一半。
        c_under = settings.W_PEAK_MW / 2 + settings.PENALTY_COEF
        c_over = settings.W_PEAK_MW / 2
        assert settings.TAU_PEAK == c_under / (c_under + c_over)

    def test_tau_ramp_up_derivation(self) -> None:
        c_under = settings.W_RAMP_UP + settings.PENALTY_COEF
        c_over = settings.W_RAMP_UP
        assert settings.TAU_RAMP_UP == c_under / (c_under + c_over)

    def test_tau_ramp_down_is_median(self) -> None:
        # ramp_down 無低估懲罰，損失對稱。
        assert settings.TAU_RAMP_DOWN == 0.5

    def test_time_error_exponent_between_linear_and_quadratic(self) -> None:
        # 指數介於 1 與 2 之間，故最佳決策既非中位數也非均值，須用貝氏決策。
        assert 1.0 < settings.TIME_ERROR_EXPONENT < 2.0


class TestPredictionPeriod:
    """預測期間的已知結構。"""

    def test_predict_dates_are_thu_fri_sat(self) -> None:
        start = date.fromisoformat(settings.PREDICT_START)
        weekdays = [
            (start + timedelta(days=i)).isoweekday()
            for i in range(settings.PREDICT_HORIZON_DAYS)
        ]
        assert weekdays == [4, 5, 6]

    def test_predict_dates_fall_in_summer_pricing(self) -> None:
        # 夏月原則為 5/16–10/15（詳細以外部 Excel 為準）。
        start = date.fromisoformat(settings.PREDICT_START)
        end = start + timedelta(days=settings.PREDICT_HORIZON_DAYS - 1)
        assert date(2026, 5, 16) <= start
        assert end <= date(2026, 10, 15)


class TestTargetArchitecture:
    """daily-row + 比值架構的設定自洽性。"""

    def test_targets_partition_into_magnitude_and_timing(self) -> None:
        # 6 個目標必須被完整切成「量值」與「時刻」兩組，不重不漏。
        both = set(settings.MAGNITUDE_TARGETS) | set(settings.TIMING_TARGETS)
        assert len(both) == 6
        assert set(settings.MAGNITUDE_TARGETS) & set(settings.TIMING_TARGETS) == set()


class TestObservedStructure:
    """資料探索寫進設定的結構常數。"""

    def test_daytypes(self) -> None:
        assert settings.DAYTYPES == ("平日", "週六", "週日")

    def test_ramp_windows_are_ordered_and_on_grid(self) -> None:
        windows = (
            settings.MORNING_RAMP_WINDOW,
            settings.MIDDAY_RAMP_DOWN_WINDOW,
            settings.EVENING_RAMP_DOWN_WINDOW,
        )
        for start, end in windows:
            t0 = datetime.strptime(start, "%H:%M")
            t1 = datetime.strptime(end, "%H:%M")
            assert t0 < t1
            for t in (t0, t1):
                assert t.minute % settings.DATA_FREQ_MIN == 0

    def test_midday_ramp_down_window_overlaps_day_peak(self) -> None:
        # 午休下降視窗與日尖峰期間相鄰但不重疊：日尖峰自 11:00 起，
        # 午休下降在 11:30–12:30，兩者確實重疊——這正是 t_day 從未落在
        # 12:00–12:50 的原因，此測試把該關係釘住。
        assert settings.MIDDAY_RAMP_DOWN_WINDOW[0] >= settings.DAY_PEAK_START
        assert settings.MIDDAY_RAMP_DOWN_WINDOW[1] <= settings.DAY_PEAK_END


class TestSubmissionSchema:
    """提交檔格式。

    提交的是10 分鐘負載曲線本身（432 列），不是 6 項尖峰特徵。
    6 項特徵由主辦單位從我們提交的曲線推導後再評分。
    """

    def test_columns_match_the_training_data(self) -> None:
        assert settings.SUBMISSION_COLUMNS == ("Date_Time", "Load_MW")

    def test_row_count_is_three_days_of_ten_minute_points(self) -> None:
        assert settings.SUBMISSION_N_ROWS == 432
        assert (
            settings.SUBMISSION_N_ROWS
            == settings.PREDICT_HORIZON_DAYS * settings.POINTS_PER_DAY
        )

    def test_datetime_format_matches_the_raw_data(self) -> None:
        # 與訓練資料一致：月與日不補零（2026/10/1 00:00）。
        assert settings.SUBMISSION_DATETIME_FORMAT == settings.RAW_DATETIME_FORMAT

    def test_roc_year_offset(self) -> None:
        assert 2026 - settings.ROC_YEAR_OFFSET == 115
