"""比賽當天的防線：讀檔結構驗證、資料截止日、前置檢查、資料更新檢查、預報缺漏的備援。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from config import paths, settings
from src.data import checks, loader, manifest


# =============================================================================
# 讀檔檢查
# =============================================================================


class TestCutoff:
    """洩漏檢查：開發期資料不得晚於 config 設定的截止日，且由程式過濾。"""

    @pytest.fixture
    def load_file(self, tmp_path, monkeypatch):
        """負載檔故意多放截止日之後的兩筆。"""
        monkeypatch.setattr(settings, "DATA_AVAILABLE_END", "2026-06-30")
        path = tmp_path / "load.csv"
        path.write_text(
            "Date_Time,Load_MW\n"
            "2026/6/30 23:40,30000\n2026/6/30 23:50,30100\n"
            "2026/7/1 00:00,29900\n2026/7/1 00:10,29800\n",
            encoding="utf-8",
        )
        return path

    def test_loader_drops_rows_after_cutoff(self, load_file) -> None:
        out = loader.load_raw_load(load_file)
        assert out["ts"].max() == dt.datetime(2026, 6, 30, 23, 50)
        assert out.height == 2

    def test_cutoff_follows_config_not_file(self, load_file, monkeypatch) -> None:
        """改設定就改變過濾結果——截止日由 config 決定，不是由檔案決定。"""
        monkeypatch.setattr(settings, "DATA_AVAILABLE_END", "2026-07-01")
        assert loader.load_raw_load(load_file).height == 4

    def test_date_and_datetime_columns(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "DATA_AVAILABLE_END", "2026-06-30")
        days = pl.DataFrame({"date": [dt.date(2026, 6, 30), dt.date(2026, 7, 1)]})
        assert checks.apply_cutoff(days, "date", "測試").height == 1
        stamps = pl.DataFrame({"Date": [dt.datetime(2026, 6, 30, 23, 59),
                                        dt.datetime(2026, 7, 1, 1)]})
        assert checks.apply_cutoff(stamps, "Date", "測試").height == 1


class TestStructure:
    """讀檔結構驗證：不符即中止並說明差異。"""

    FRAME = pl.DataFrame({
        "ts": [dt.datetime(2026, 6, 30, 23, 30), dt.datetime(2026, 6, 30, 23, 40),
               dt.datetime(2026, 6, 30, 23, 50)],
        "Load_MW": [1.0, 2.0, 3.0],
    })

    def test_columns_and_types(self) -> None:
        checks.check_columns(self.FRAME, {"ts": pl.Datetime, "Load_MW": pl.Float64}, "負載")
        with pytest.raises(ValueError, match="缺少欄位 Date_Time"):
            checks.check_columns(self.FRAME, {"Date_Time": pl.Datetime}, "負載")
        with pytest.raises(ValueError, match="Load_MW 應為 String"):
            checks.check_columns(self.FRAME, {"Load_MW": pl.Utf8}, "負載")

    def test_step(self) -> None:
        checks.check_step(self.FRAME, "ts", 10, "負載")
        shifted = self.FRAME.with_columns(pl.col("ts") + dt.timedelta(minutes=5))
        with pytest.raises(ValueError, match="不在 10 分鐘格點上"):
            checks.check_step(shifted, "ts", 10, "負載")

    def test_gaps(self) -> None:
        checks.check_gaps(self.FRAME, "ts", dt.timedelta(minutes=10), "負載")
        holed = self.FRAME.filter(pl.col("Load_MW") != 2.0)
        with pytest.raises(ValueError, match="時間不連續"):
            checks.check_gaps(holed, "ts", dt.timedelta(minutes=10), "負載")

    def test_gaps_are_checked_within_each_group(self) -> None:
        two = pl.DataFrame({
            "stn": ["a", "a", "b", "b"],
            "t": [dt.datetime(2026, 1, 1, h) for h in (0, 1, 5, 6)],
        })
        checks.check_gaps(two, "t", dt.timedelta(hours=1), "測試", group="stn")


class TestDeduplicate:
    """完全重複刪除並記錄；鍵同值異不自行挑選，中止回報。"""

    def test_exact_duplicates_are_dropped(self) -> None:
        frame = pl.DataFrame({"k": [1, 1, 2], "v": [9.0, 9.0, 8.0]})
        assert checks.deduplicate(frame, ["k"], "測試").height == 2

    def test_conflicting_values_raise(self) -> None:
        frame = pl.DataFrame({"k": [1, 1, 2], "v": [9.0, 7.0, 8.0]})
        with pytest.raises(ValueError, match="相同但數值不同"):
            checks.deduplicate(frame, ["k"], "測試")


@pytest.mark.skipif(not paths.has_accuweather_forecast(), reason="需要 Accuweather 年度檔")
class TestPrecheck:
    """預測前的涵蓋檢查（用本機實際資料）。"""

    def test_dev_prediction_window_passes(self) -> None:
        origin = checks.cutoff_date()
        checks.precheck(origin, [origin + dt.timedelta(days=k) for k in (1, 2, 3)])

    def test_dev_window_has_complete_forecast(self) -> None:
        origin = checks.cutoff_date()
        result = checks.precheck(origin, [origin + dt.timedelta(days=k) for k in (1, 2, 3)])
        assert result["forecast_missing"] == {}

    def test_missing_load_still_aborts(self) -> None:
        """負載不足沒有備援，一定中止；預報不足則不在中止理由內（走備援）。"""
        origin = dt.date.fromisoformat(settings.DATA_AVAILABLE_END) + dt.timedelta(days=1)   # 負載還沒到的那天
        with pytest.raises(ValueError) as error:
            checks.precheck(origin, [origin + dt.timedelta(days=h) for h in (1, 2, 3)])
        message = str(error.value)
        assert "負載只到" in message
        assert "Accuweather" not in message


# =============================================================================
# 資料更新檢查
# =============================================================================


SOURCES = (("load.csv", "Date_Time", "raw", ("Date_Time",)),)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """以暫存目錄當專案根目錄，放一份小型負載檔。"""
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    (tmp_path / "load.csv").write_text(
        "Date_Time,Load_MW\n2024/1/1 00:00,100\n2024/1/1 00:10,\n2024/1/1 00:20,102\n",
        encoding="utf-8",
    )
    return tmp_path


def test_same_file_gives_identical_records(root) -> None:
    assert manifest.build_manifest(SOURCES)["files"] == manifest.build_manifest(SOURCES)["files"]


def test_records_rows_nulls_and_time_range(root) -> None:
    (record,) = manifest.build_manifest(SOURCES)["files"]
    assert record["rows"] == 3
    assert record["null_counts"] == {"Date_Time": 0, "Load_MW": 1}
    assert record["time_min"] == "2024-01-01 00:00:00"
    assert record["time_max"] == "2024-01-01 00:20:00"


def test_editing_a_value_changes_hash_and_nulls(root) -> None:
    before = manifest.build_manifest(SOURCES)["files"][0]
    path = root / "load.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",\n", ",101\n"), encoding="utf-8")
    after = manifest.build_manifest(SOURCES)["files"][0]
    assert after["sha256"] != before["sha256"]
    assert after["null_counts"]["Load_MW"] == 0


def test_unparseable_time_raises(root) -> None:
    (root / "load.csv").write_text("Date_Time,Load_MW\n不是時間,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="無法以 MANIFEST_TIME_FORMATS 解析"):
        manifest.build_manifest(SOURCES)


def test_missing_source_raises(root) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        manifest.build_manifest((("nope.csv", None, "raw", None),))


class TestUpdateCheck:
    """資料更新檢查：以「改部分歷史值 + 延長時間 + 刪除一筆」的檔驗證偵測能力。"""

    KEYED = (("load.csv", "Date_Time", "raw", ("Date_Time",)),)

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(paths, "MANIFEST_FILE", tmp_path / "manifest.json")
        monkeypatch.setattr(paths, "BACKUP_DIR", tmp_path / "_backup")
        monkeypatch.setattr(settings, "MANIFEST_SOURCES", self.KEYED)
        (tmp_path / "load.csv").write_text(
            "Date_Time,Load_MW\n2024/1/1 00:00,100\n2024/1/1 00:10,101\n"
            "2024/1/1 00:20,102\n2024/1/1 00:30,103\n", encoding="utf-8")
        manifest.accept(manifest.update_check()["manifest"])   # 建立基準與快照
        return tmp_path

    def test_unchanged_files_raise_nothing(self, env) -> None:
        result = manifest.update_check()
        assert result["files"][0]["status"] == "unchanged"
        assert result["history_changed"] is False

    def test_detects_modified_deleted_and_extended(self, env) -> None:
        (env / "load.csv").write_text(
            "Date_Time,Load_MW\n2024/1/1 00:00,100\n2024/1/1 00:10,999\n"   # 改值
            "2024/1/1 00:30,103\n"                                         # 刪 00:20
            "2024/1/1 00:40,104\n2024/1/1 00:50,105\n", encoding="utf-8")  # 延長兩筆
        result = manifest.update_check()
        diff = result["rows"]["load.csv"]
        assert diff["modified"]["count"] == 1
        assert diff["modified"]["examples"][0]["Load_MW__new"] == "999"
        assert diff["deleted"]["count"] == 1
        assert diff["extended"]["count"] == 2
        assert diff["added_in_history"]["count"] == 0
        assert result["history_changed"] is True

    def test_extension_only_is_not_a_history_change(self, env) -> None:
        with (env / "load.csv").open("a", encoding="utf-8") as handle:
            handle.write("2024/1/1 00:40,104\n")
        result = manifest.update_check()
        assert result["rows"]["load.csv"]["extended"]["count"] == 1
        assert result["history_changed"] is False

    def test_schema_change_is_reported(self, env) -> None:
        (env / "load.csv").write_text(
            "Date_Time,Load_MW,Note\n2024/1/1 00:00,100,x\n", encoding="utf-8")
        result = manifest.update_check()
        entry = result["files"][0]
        assert entry["columns_added"] == ["Note"]
        assert result["rows"]["load.csv"]["only_in_new"] == ["Note"]


def test_prune_keeps_only_latest_snapshots(tmp_path, monkeypatch) -> None:
    """快照只保留最新 ``BACKUP_KEEP`` 份。"""
    monkeypatch.setattr(paths, "BACKUP_DIR", tmp_path)
    names = [f"20260927_0{i}0000" for i in range(7)]
    for name in names:
        (tmp_path / name).mkdir()
    removed = manifest.prune_snapshots(keep=5)
    assert [p.name for p in removed] == names[:2]
    assert sorted(p.name for p in tmp_path.iterdir()) == names[2:]


# =============================================================================
# 預報缺漏的備援
# =============================================================================


ORIGIN = dt.date(2026, 6, 30)


DAYS = [dt.date(2026, 7, d) for d in (1, 2, 3)]


needs_dev_data = pytest.mark.skipif(
    not paths.has_accuweather_forecast() or settings.DATA_AVAILABLE_END != "2026-06-30",
    reason="需要 Accuweather 年度檔與開發期設定")


@pytest.fixture
def degraded(monkeypatch):
    from src.data import accuweather

    original = accuweather.load_station_hourly
    taipei = settings.WEATHER_STATIONS["臺北"]

    def without_some():
        frame = original()
        day = pl.col("Date").dt.date()
        return frame.filter(~(day == DAYS[2]) & ~((day == DAYS[1]) & (pl.col("stn_ID") == taipei)))

    monkeypatch.setattr(accuweather, "load_station_hourly", without_some)


@needs_dev_data
def test_full_day_uses_no_weather_model_and_partial_uses_persistence(degraded) -> None:
    from src import workflow
    from src.data import checks

    missing = checks.precheck(ORIGIN, DAYS)["forecast_missing"]
    assert missing[DAYS[1]] == ("臺北",)
    assert set(missing[DAYS[2]]) == set(settings.WEATHER_STATIONS)

    curves, _, _ = workflow.predict_days(ORIGIN, DAYS)
    with workflow.temporary_settings(ENABLE_TIER1_WEATHER=False, TIMING_TEMPERATURE_BINS=1,
                                     FORECAST_MISSING_CELLS=missing):
        no_weather, _, _ = workflow._predict_curves(ORIGIN, DAYS)
    assert np.array_equal(curves[DAYS[2]], no_weather[DAYS[2]])
    assert not np.array_equal(curves[DAYS[0]], no_weather[DAYS[0]])
    assert settings.FORECAST_MISSING_CELLS == {}          # 暫時設定已還原
    assert settings.ENABLE_TIER1_WEATHER is True


@needs_dev_data
@pytest.mark.skipif(not paths.WINDY_FILE.exists(), reason="需要本機的 Windy 資料")
def test_missing_windy_day_uses_model_without_windy(monkeypatch) -> None:
    """開啟 Windy 時，Windy 不齊的那一天改用不含 Windy 的模型；其餘日子照常含 Windy。"""
    from src import workflow
    from src.data import checks, windy

    monkeypatch.setattr(settings, "ENABLE_WINDY", True)
    original = windy.load_hourly
    monkeypatch.setattr(windy, "load_hourly", lambda: original().filter(
        pl.col("forecast_time").dt.date() != DAYS[2]))

    assert checks.precheck(ORIGIN, DAYS)["windy_missing"] == [DAYS[2]]
    curves, _, _ = workflow.predict_days(ORIGIN, DAYS)
    with workflow.temporary_settings(ENABLE_WINDY=False):
        without, _, _ = workflow._predict_curves(ORIGIN, DAYS)
    assert np.array_equal(curves[DAYS[2]], without[DAYS[2]])
    assert not np.array_equal(curves[DAYS[0]], without[DAYS[0]])


@needs_dev_data
def test_windy_is_not_checked_when_disabled(monkeypatch) -> None:
    from src.data import checks, windy

    monkeypatch.setattr(settings, "ENABLE_WINDY", False)
    monkeypatch.setattr(windy, "load_hourly", lambda: (_ for _ in ()).throw(AssertionError("不該讀 Windy")))
    assert checks.precheck(ORIGIN, DAYS)["windy_missing"] == []
