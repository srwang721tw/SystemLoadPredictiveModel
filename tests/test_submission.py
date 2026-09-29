"""提交檔：檔名與流水號、432 列格式、寫檔後讀回重新推導 6 項目標。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from config import settings
from src.features.targets import to_minutes
from src.models import curve
from src.output import submission


def make_day(seed: int = 0) -> tuple[curve.DayTargets, np.ndarray]:
    """造一天合法的目標與其對應曲線。"""
    targets = curve.DayTargets(
        38000.0 + seed * 100, to_minutes("14:00"),
        35000.0 + seed * 100, to_minutes("18:30"),
        1200.0, 800.0,
    )
    levels, _ = curve.feasible_levels(
        targets, curve.FreeLevels(28000.0, 26000.0, 30000.0)
    )
    values = curve.synthesize_day_checked(
        targets, levels.start, levels.minimum, levels.end
    )
    return targets, values


def three_days(start: dt.date = dt.date(2026, 10, 1)):
    """造連續三天的目標與曲線。"""
    curves, intended = {}, {}
    for offset in range(settings.PREDICT_HORIZON_DAYS):
        day = start + dt.timedelta(days=offset)
        targets, values = make_day(offset)
        curves[day], intended[day] = values, targets
    return curves, intended


class TestRocStamp:
    """民國年 7 碼。"""

    def test_converts_year(self) -> None:
        assert submission.roc_stamp(dt.date(2026, 8, 15)) == "1150815"

    def test_pads_month_and_day(self) -> None:
        assert submission.roc_stamp(dt.date(2026, 1, 3)) == "1150103"


class TestVersioning:
    """流水號：取現有最大 + 1，嚴禁覆蓋。"""

    def test_first_version_is_one(self, tmp_path) -> None:
        assert submission.next_version(tmp_path, "1150815") == 1

    def test_increments_past_existing(self, tmp_path) -> None:
        for version in (1, 2, 5):
            (tmp_path / f"1150815_submission_V{version}.csv").touch()
        assert submission.next_version(tmp_path, "1150815") == 6

    def test_other_days_do_not_interfere(self, tmp_path) -> None:
        (tmp_path / "1150814_submission_V9.csv").touch()
        assert submission.next_version(tmp_path, "1150815") == 1

    def test_write_never_overwrites(self, tmp_path) -> None:
        curves, _ = three_days()
        frame = submission.build_submission(curves)
        first = submission.write_submission(frame, tmp_path, dt.date(2026, 8, 15))
        second = submission.write_submission(frame, tmp_path, dt.date(2026, 8, 15))
        assert first != second
        assert first.exists() and second.exists()

    def test_latest_copy_tracks_the_newest_and_does_not_affect_versions(self, tmp_path) -> None:
        curves, _ = three_days()
        frame = submission.build_submission(curves)
        submission.write_submission(frame, tmp_path, dt.date(2026, 10, 1))
        second = submission.write_submission(frame.with_columns(pl.col("Load_MW") + 1),
                                             tmp_path, dt.date(2026, 10, 1))
        latest = tmp_path / settings.SUBMISSION_LATEST_NAME
        assert latest.read_bytes() == second.read_bytes()
        assert submission.next_version(tmp_path, "1151001") == 3


class TestBuildSubmission:
    """432 列與格式。"""

    def test_row_count(self) -> None:
        curves, _ = three_days()
        assert submission.build_submission(curves).height == settings.SUBMISSION_N_ROWS == 432

    def test_columns_and_order(self) -> None:
        curves, _ = three_days()
        frame = submission.build_submission(curves)
        assert tuple(frame.columns) == settings.SUBMISSION_COLUMNS

    def test_datetime_format_matches_training_data(self) -> None:
        curves, _ = three_days()
        frame = submission.build_submission(curves)
        # 月與日不補零，與訓練資料一致。
        assert frame["Date_Time"][0] == "2026/10/1 00:00"
        assert frame["Date_Time"][-1] == "2026/10/3 23:50"

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="曲線長度"):
            submission.build_submission({dt.date(2026, 10, 1): np.zeros(100)})

    def test_rejects_non_finite(self) -> None:
        _, values = make_day()
        values[5] = np.nan
        with pytest.raises(ValueError, match="NaN 或 Inf"):
            submission.build_submission({dt.date(2026, 10, 1): values})

    def test_rejects_discontinuous_dates(self) -> None:
        _, values = make_day()
        curves = {dt.date(2026, 10, 1): values, dt.date(2026, 10, 3): values.copy()}
        with pytest.raises(ValueError, match="日期不連續"):
            submission.build_submission(curves)


class TestRoundTripValidation:
    """寫檔 → 讀回 → 重新推導 6 項目標 → 必須完全相符。"""

    def test_passes_for_a_faithful_submission(self, tmp_path) -> None:
        curves, intended = three_days()
        path = submission.write_submission(
            submission.build_submission(curves), tmp_path, dt.date(2026, 8, 15)
        )
        report = submission.validate_submission(path, intended)
        assert report.height == 3 * 6
        assert report["ok"].all()

    def test_detects_a_tampered_curve(self, tmp_path) -> None:
        # 把日尖峰時段的一個點抬高——格式完全正確，但 t_day 會跑掉。
        # 沒有這道檢查，這種錯誤會靜默失分。
        curves, intended = three_days()
        first = min(curves)
        curves[first] = curves[first].copy()
        curves[first][to_minutes("12:00") // 10] = intended[first].p_day + 500
        path = submission.write_submission(
            submission.build_submission(curves), tmp_path, dt.date(2026, 8, 15)
        )
        with pytest.raises(ValueError, match="讀回驗證失敗"):
            submission.validate_submission(path, intended)

    def test_detects_wrong_row_count(self, tmp_path) -> None:
        curves, intended = three_days()
        frame = submission.build_submission(curves).head(400)
        path = tmp_path / "1150815_submission_V1.csv"
        frame.write_csv(path)
        with pytest.raises(ValueError, match="列數應為"):
            submission.validate_submission(path, intended)

    def test_detects_wrong_columns(self, tmp_path) -> None:
        curves, intended = three_days()
        frame = submission.build_submission(curves).rename({"Load_MW": "Load"})
        path = tmp_path / "1150815_submission_V1.csv"
        frame.write_csv(path)
        with pytest.raises(ValueError, match="欄位不符"):
            submission.validate_submission(path, intended)

    def test_detects_time_gaps(self, tmp_path) -> None:
        # 保持 432 列，但把中間一列的時間戳改掉——列數檢查過得了，
        # 只有連續性檢查能攔下來。
        curves, intended = three_days()
        frame = submission.build_submission(curves)
        stamps = frame["Date_Time"].to_list()
        stamps[200] = stamps[201]  # 製造 0 與 20 分鐘兩種間隔
        broken = frame.with_columns(pl.Series("Date_Time", stamps))
        path = tmp_path / "1150815_submission_V1.csv"
        broken.write_csv(path)
        assert broken.height == settings.SUBMISSION_N_ROWS
        with pytest.raises(ValueError, match="時間不連續"):
            submission.validate_submission(path, intended)

    def test_timing_must_match_exactly(self, tmp_path) -> None:
        # 時刻不容許任何容差——差一格就是 (10/10)^1.2 = 1 分。
        curves, intended = three_days()
        first = min(curves)
        wrong = intended[first]
        intended[first] = curve.DayTargets(
            wrong.p_day, wrong.t_day + 10, wrong.p_night, wrong.t_night,
            wrong.ramp_up, wrong.ramp_down,
        )
        path = submission.write_submission(
            submission.build_submission(curves), tmp_path, dt.date(2026, 8, 15)
        )
        with pytest.raises(ValueError, match="t_day"):
            submission.validate_submission(path, intended)
