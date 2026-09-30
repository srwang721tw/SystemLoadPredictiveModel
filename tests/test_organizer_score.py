"""主辦單位計分程式（``organizer_score.py``）：推導規則、評分公式、讀檔，並與專案實作交叉比對。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

import organizer_score
from config import paths
from src.evaluation import metrics
from src.features.targets import TARGET_NAMES, compute_targets


def slot(hhmm: str) -> int:
    hour, minute = map(int, hhmm.split(":"))
    return (hour * 60 + minute) // 10


def flat_day(level: float = 30000.0) -> list[float]:
    return [level] * 144


class TestDailyTargets:
    def test_tie_takes_the_earliest_day_peak(self) -> None:
        values = flat_day()
        values[slot("13:00")] = values[slot("15:00")] = 35000.0
        assert organizer_score.daily_targets(values)["t_day"] == 13 * 60

    def test_tie_takes_the_earliest_night_peak(self) -> None:
        values = flat_day()
        values[slot("18:30")] = values[slot("20:00")] = 33000.0
        assert organizer_score.daily_targets(values)["t_night"] == 18 * 60 + 30

    def test_flat_window_peaks_at_its_first_slot(self) -> None:
        got = organizer_score.daily_targets(flat_day())
        assert got["t_day"] == 11 * 60 and got["t_night"] == 17 * 60 + 10

    def test_window_endpoints(self) -> None:
        values = flat_day()
        values[slot("17:00")] = 36000.0      # 日窗口的右端點
        values[slot("17:10")] = 35000.0      # 夜窗口的左端點
        got = organizer_score.daily_targets(values)
        assert (got["t_day"], got["p_day"]) == (17 * 60, 36000.0)
        assert (got["t_night"], got["p_night"]) == (17 * 60 + 10, 35000.0)

    def test_values_outside_the_windows_do_not_count(self) -> None:
        values = flat_day()
        values[slot("10:50")] = values[slot("21:10")] = 50000.0
        got = organizer_score.daily_targets(values)
        assert got["p_day"] == 30000.0 and got["p_night"] == 30000.0

    def test_ramps_use_143_differences_and_ramp_down_is_positive(self) -> None:
        values = [30000.0 + 10 * k for k in range(144)]
        values[50] += 500                    # 49→50 上升 510、50→51 下降 490
        got = organizer_score.daily_targets(values)
        assert got["ramp_up"] == pytest.approx(510.0)
        assert got["ramp_down"] == pytest.approx(490.0)

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="144"):
            organizer_score.daily_targets([1.0] * 143)


class TestReadCurves:
    def _write(self, path, stamps, bom: bool = False) -> None:
        text = "Date_Time,Load_MW\n" + "".join(f"{s},{30000 + i}\n" for i, s in enumerate(stamps))
        path.write_text(("﻿" if bom else "") + text, encoding="utf-8")

    def test_reads_submission_format_with_bom(self, tmp_path) -> None:
        stamps = [f"2026/10/1 {k // 6:02d}:{k % 6 * 10:02d}" for k in range(144)]
        self._write(tmp_path / "a.csv", stamps, bom=True)
        days = organizer_score.read_curves(tmp_path / "a.csv")
        assert list(days) == [dt.date(2026, 10, 1)] and len(days[dt.date(2026, 10, 1)]) == 144

    def test_reads_iso_format(self, tmp_path) -> None:
        stamps = [f"2026-10-01 {k // 6:02d}:{k % 6 * 10:02d}:00" for k in range(144)]
        self._write(tmp_path / "b.csv", stamps)
        assert len(organizer_score.read_curves(tmp_path / "b.csv")[dt.date(2026, 10, 1)]) == 144

    def test_incomplete_day_raises(self, tmp_path) -> None:
        stamps = [f"2026/10/1 {k // 6:02d}:{k % 6 * 10:02d}" for k in range(143)]
        self._write(tmp_path / "c.csv", stamps)
        with pytest.raises(ValueError, match="144"):
            organizer_score.read_curves(tmp_path / "c.csv")


class TestAgreesWithProjectImplementation:
    """獨立實作與專案的推導、評分在隨機資料上必須完全相同。"""

    def test_targets_match_compute_targets(self) -> None:
        rng = np.random.default_rng(0)
        days = [dt.date(2026, 10, 1) + dt.timedelta(days=d) for d in range(20)]
        # 以 50 MW 為單位取整，刻意製造大量並列。
        values = (np.round(rng.normal(32000, 3000, (len(days), 144)) / 50) * 50).tolist()
        frame = pl.DataFrame({
            "ts": [dt.datetime.combine(day, dt.time()) + dt.timedelta(minutes=10 * k)
                   for day in days for k in range(144)],
            "Load_MW": [v for row in values for v in row],
            "is_imputed": [False] * (len(days) * 144),
        })
        project = compute_targets(frame).sort("date")
        for index, row in enumerate(project.iter_rows(named=True)):
            ours = organizer_score.daily_targets(values[index])
            for name in TARGET_NAMES:
                assert ours[name] == pytest.approx(row[name], abs=1e-9), (days[index], name)

    def test_score_matches_metrics(self) -> None:
        rng = np.random.default_rng(1)
        n = 30
        actual = pl.DataFrame({
            "p_day": rng.uniform(25000, 40000, n), "t_day": rng.integers(66, 103, n) * 10.0,
            "p_night": rng.uniform(25000, 38000, n), "t_night": rng.integers(103, 127, n) * 10.0,
            "ramp_up": rng.uniform(300, 2000, n), "ramp_down": rng.uniform(300, 2000, n),
        })
        predicted = actual.with_columns(
            pl.col("p_day", "p_night", "ramp_up", "ramp_down") * pl.lit(rng.uniform(0.8, 1.2, n)),
            pl.Series("t_day", rng.integers(66, 103, n) * 10.0),
            pl.Series("t_night", rng.integers(103, 127, n) * 10.0),
        )
        expected = metrics.score_breakdown(actual, predicted).as_dict()
        got = organizer_score.score(actual.to_dicts(), predicted.to_dicts())
        for name, value in expected.items():
            assert got[name] == pytest.approx(value, abs=1e-12), name


@pytest.mark.skipif(not paths.TARGETS_FILE.exists() or not list(paths.BACKTEST_DIR.glob("*_notebook_04_tuning")),
                    reason="需要 notebook 04 的回測紀錄與重建後的 data/processed/")
def test_backtest_record_passes_organizer_check() -> None:
    from src.evaluation import backtest, verify

    result = verify.verify_run(backtest.find_run("notebook_04"))
    assert result["ok"], result["mismatches"]
