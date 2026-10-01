"""氣象與太陽光電：CODiS 解析與清理、Accuweather 讀取與校正、觀測延遲的補值、Windy 特徵。

Accuweather 年度檔不附在 repo 內，需要它的測試在檔案缺席時略過。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from config import paths, settings
from src.data import accuweather, checks, external, weather as weather_io
from src.features import accuweather as accuweather_features, weather as weather_features, windy as windy_features


# =============================================================================
# CODiS 觀測
# =============================================================================


# 取自實際 API 回應的結構（2026-08-12 臺北站），只保留需要的欄位與 4 個時刻。
REAL_RESPONSE = {
    "code": 200, "message": "", "metadata": {},
    "data": [{
        "StationID": "466920",
        "dts": [
            {"DataTime": "2026-08-12T01:00:00",
             "AirTemperature": {"Instantaneous": 27.9, "Instantaneousf": None},
             "UVIndex": {"Accumulation": 0, "Accumulationf": None},
             "SoilTemperatureAt0cm": {"Instantaneous": -99.5}},
            {"DataTime": "2026-08-12T12:00:00",
             "AirTemperature": {"Instantaneous": 34.3, "Instantaneousf": None},
             "UVIndex": {"Accumulation": 7.58, "Accumulationf": None}},
            {"DataTime": "2026-08-12T19:00:00",
             "AirTemperature": {"Instantaneous": 30.2, "Instantaneousf": None},
             "UVIndex": {"Accumulation": 0.01, "Accumulationf": None}},
            {"DataTime": "2026-08-12T23:59:00",
             "AirTemperature": {"Instantaneous": 29.6, "Instantaneousf": None},
             "UVIndex": {"Accumulation": 0, "Accumulationf": None}},
        ],
    }],
}


class TestParseResponse:
    """解析保留全部變數與旗標的原始字串，不轉數值、不清特殊值。"""

    def test_keeps_raw_strings(self) -> None:
        rows = weather_io.parse_response(REAL_RESPONSE)
        assert len(rows) == 4
        assert rows[1]["AirTemperature_Instantaneous"] == "34.3"
        assert rows[1]["UVIndex_Accumulation"] == "7.58"
        assert rows[0]["SoilTemperatureAt0cm_Instantaneous"] == "-99.5"

    def test_columns_are_fixed_and_unknown_fields_are_recorded(self) -> None:
        payload = {"data": [{"StationID": "466920", "dts": [
            {"DataTime": "2026-08-12T01:00:00", "NotInList": {"Value": 1}}]}]}
        (row,) = weather_io.parse_response(payload)
        assert tuple(row) == weather_io.CSV_COLUMNS
        assert "NotInList_Value" in weather_io._unknown_fields

    def test_empty_data_is_not_an_error(self) -> None:
        # 舊測站可能停測；回空list而非拋錯，讓爬蟲能繼續跑完其他組合。
        assert weather_io.parse_response({"data": []}) == []


class TestCleanObservations:
    """特殊值依 config 對照表逐值處理並記錄。"""

    @staticmethod
    def _raw(**columns: list[str | None]) -> pl.DataFrame:
        n = len(next(iter(columns.values())))
        base = {c: [None] * n for c in weather_io.CSV_COLUMNS}
        base["Date"] = [f"2026-08-12T{h + 1:02d}:00:00" for h in range(n)]
        base["stn_ID"] = ["466920"] * n
        base.update(columns)
        return pl.DataFrame(base, schema=dict.fromkeys(weather_io.CSV_COLUMNS, pl.Utf8))

    @pytest.mark.parametrize("code", settings.CODIS_MISSING_CODES)
    def test_every_missing_code_becomes_null_and_is_counted(self, code: float) -> None:
        cleaned, report = weather_io.clean_observations(
            self._raw(AirTemperature_Instantaneous=[str(code), "30.1"]))
        assert cleaned["AirTemperature_Instantaneous"].to_list() == [None, 30.1]
        assert report.filter(pl.col("原始值") == str(code))["筆數"].to_list() == [1]

    def test_codes_above_minus_90_are_caught(self) -> None:
        """−9.5、−9.95 在舊的「≤ −90」門檻之外——這正是改用對照表的理由。"""
        cleaned, _ = weather_io.clean_observations(
            self._raw(SunshineDuration_Total=["-9.5", "0.4"],
                      GlobalSolarRadiation_Accumulation=["-9.95", "2.1"]))
        assert cleaned["SunshineDuration_Total"].to_list() == [None, 0.4]
        assert cleaned["GlobalSolarRadiation_Accumulation"].to_list() == [None, 2.1]

    def test_trace_precipitation_is_not_missing(self) -> None:
        cleaned, report = weather_io.clean_observations(
            self._raw(Precipitation_Accumulation=["-9.8", "0", "1.5"]))
        assert cleaned["Precipitation_Accumulation"].to_list() == [
            settings.CODIS_TRACE_VALUE, 0.0, 1.5]
        assert cleaned["Precipitation_Accumulation_trace"].to_list() == [1, 0, 0]
        assert report.filter(pl.col("處理").str.starts_with("雨跡"))["筆數"].item() == 1


    def test_real_negative_values_survive(self) -> None:
        """露點有真實負值，「小於 0 即異常」不成立。"""
        cleaned, report = weather_io.clean_observations(
            self._raw(DewPointTemperature_Instantaneous=["-1.7", "-0.5"]))
        assert cleaned["DewPointTemperature_Instantaneous"].to_list() == [-1.7, -0.5]
        assert report.height == 0

    def test_unlisted_sentinel_and_text_are_reported(self) -> None:
        cleaned, report = weather_io.clean_observations(
            self._raw(AirTemperature_Instantaneous=["-99.6", "X", "25"]))
        assert cleaned["AirTemperature_Instantaneous"].to_list() == [None, None, 25.0]
        assert set(report["處理"]) == {"未登記的缺測碼 → 缺值", "無法轉數值 → 缺值"}


class TestDayNightBoundary:
    """白天 = hour 7–18。差一格就會靜默算錯。"""

    @staticmethod
    def _hourly(pairs: list[tuple[str, float]]) -> pl.DataFrame:
        return pl.DataFrame({
            "Date": [dt.datetime.fromisoformat(t) for t, _ in pairs],
            "stn_ID": ["466920"] * len(pairs),
            "AirTemperature": [v for _, v in pairs],
            "UVIndex": [0.0] * len(pairs),
        })


    @pytest.mark.parametrize(
        ("timestamp", "expected_day"),
        [
            ("2026-08-12T06:00:00", False),  # 05:00–06:00 → 夜
            ("2026-08-12T07:00:00", True),   # 06:00–07:00 → 日（第一格）
            ("2026-08-12T18:00:00", True),   # 17:00–18:00 → 日（最後一格）
            ("2026-08-12T19:00:00", False),  # 18:00–19:00 → 夜
        ],
    )
    def test_four_boundaries(self, timestamp: str, expected_day: bool) -> None:
        # 讓該格的溫度極端，再看它落在 day_* 還是 night_*。
        frame = self._hourly([(timestamp, 99.0), ("2026-08-12T12:00:00", 20.0),
                              ("2026-08-12T02:00:00", 20.0)])
        out = weather_features.daily_features(frame)
        hit = "w_臺北_day_tmax" if expected_day else "w_臺北_night_tmax"
        assert out[hit][0] == 99.0

    def test_day_and_night_each_cover_twelve_hours(self) -> None:
        low, high = settings.WEATHER_DAY_HOURS
        assert high - low + 1 == 12


class TestInterpolation:
    def test_gap_is_filled_between_neighbours_not_with_zero(self) -> None:
        # 補 0 會讓 day_tmin 變成 0 度——這是最容易靜默出錯的地方。
        frame = pl.DataFrame({
            "Date": [dt.datetime(2026, 8, 12, h) for h in (10, 11, 12)],
            "stn_ID": ["466920"] * 3,
            "AirTemperature": [30.0, None, 34.0],
            "UVIndex": [1.0, None, 3.0],
        })
        filled, _ = weather_features.interpolate_hourly(frame)
        assert filled["AirTemperature"][1] == pytest.approx(32.0)
        assert filled["AirTemperature"][1] != 0.0
        assert filled["UVIndex"][1] == pytest.approx(2.0)

    def test_long_gaps_are_reported(self) -> None:
        frame = pl.DataFrame({
            "Date": [dt.datetime(2026, 8, 12, h) for h in range(8)],
            "stn_ID": ["466920"] * 8,
            "AirTemperature": [30.0, None, None, None, None, None, None, 34.0],
            "UVIndex": [0.0] * 8,
        })
        _, diagnostics = weather_features.interpolate_hourly(frame)
        assert diagnostics["long_gaps"], "連續 6 小時缺測必須被列出"
        assert diagnostics["long_gaps"][0]["hours"] == 6

    def test_stations_do_not_bleed_into_each_other(self) -> None:
        # 插補必須 over("stn_ID")，否則臺北的缺值會被高雄的值補上。
        frame = pl.DataFrame({
            "Date": [dt.datetime(2026, 8, 12, 10), dt.datetime(2026, 8, 12, 11),
                     dt.datetime(2026, 8, 12, 10), dt.datetime(2026, 8, 12, 11)],
            "stn_ID": ["466920", "466920", "467441", "467441"],
            "AirTemperature": [30.0, None, 10.0, 10.0],
            "UVIndex": [0.0] * 4,
        })
        filled, _ = weather_features.interpolate_hourly(frame)
        taipei = filled.filter(pl.col("stn_ID") == "466920")["AirTemperature"].to_list()
        assert taipei[1] == 30.0, "應由臺北自己的值前向填補，不得取到高雄的 10 度"


class TestMissingWeatherFailsLoudly:
    """缺氣象時必須拋錯，不得靜默補值。

    靜默補 0 或補平均會產出一份「格式全對、數字全假」的提交檔——
    而我們永遠不會知道。這與第三層產檔驗證的精神一致。
    """

    def test_timing_raises_when_target_day_has_no_temperature(self) -> None:
        from src.models import pipeline

        dates = [dt.date(2026, 6, 1) + dt.timedelta(days=k) for k in range(40)]
        attributes = pl.DataFrame({
            "date": dates,
            "price_daytype": ["平日"] * 40,
            "is_summer": [True] * 40,
            "weekday": [d.isoweekday() for d in dates],
            # 最後一天沒有氣溫——模擬「預測未來但沒填預報值」。
            pipeline.TEMPERATURE_COLUMN: [30.0] * 39 + [None],
        })
        history = pl.DataFrame({
            "date": dates[:-1],
            "t_day": [830] * 39, "t_night": [1030] * 39,
            "p_day": [30000.0] * 39, "p_night": [28000.0] * 39,
            "ramp_up": [1000.0] * 39, "ramp_down": [800.0] * 39,
        })
        predict = pipeline.make_ensemble_timing_predictor(
            attributes, pl.DataFrame(), {"t_day": 1.0, "t_night": 1.0}
        )
        with pytest.raises(ValueError, match="缺氣溫"):
            predict(history, (dates[-1],))

    def test_features_raise_when_a_day_has_no_weather(self) -> None:
        daily = pl.DataFrame({"date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)]})
        weather = pl.DataFrame({
            "date": [dt.date(2026, 6, 1)], "w_臺北_day_tmax": [30.0],
        })
        with pytest.raises(ValueError, match="缺氣象資料"):
            weather_features.add_weather_features(daily, weather)


class TestDailyFeatures:
    def test_produces_configured_metrics_per_station(self) -> None:
        rows = []
        for code in settings.WEATHER_STATIONS.values():
            for hour in range(1, 25):
                rows.append({
                    "Date": dt.datetime(2026, 8, 12) + dt.timedelta(hours=hour - 1),
                    "stn_ID": code, "AirTemperature": float(hour), "UVIndex": float(hour),
                })
        out = weather_features.daily_features(pl.DataFrame(rows))
        expected = len(settings.WEATHER_STATIONS) * len(settings.WEATHER_METRICS)
        assert out.width - 1 == expected
        assert not any(c.endswith("_uv_max") for c in out.columns)   # UV 不進模型
        assert out.height == 1


# =============================================================================
# Accuweather 預報
# =============================================================================


needs_accuweather = pytest.mark.skipif(
    not paths.has_accuweather_forecast(), reason="需要 Accuweather 年度檔")


@pytest.fixture(scope="module")
def hourly() -> pl.DataFrame:
    return accuweather.load_station_hourly()


@pytest.fixture(scope="module")
def forecast_raw() -> pl.DataFrame:
    return accuweather_features.build_forecast()


@needs_accuweather
class TestLoading:
    """讀取層：型別、去重、涵蓋。"""

    def test_yearly_files_concat_without_schema_error(self, hourly) -> None:
        """三個年度檔的型別推論不一致，不釘死 schema 會拋 SchemaError。

        症狀：``type Float64 is incompatible with expected type Int64``。
        這條測試就是那個錯誤的回歸測試——能跑完就代表 schema 有被釘住。
        """
        assert hourly.height > 0
        assert hourly["AirTemperature"].dtype == pl.Float64
        assert hourly["UVIndex"].dtype == pl.Float64

    def test_schema_matches_codis_hourly(self, hourly) -> None:
        """刻意輸出 CODiS 的 schema，好讓日彙總管線能原樣重用。"""
        assert tuple(hourly.columns) == accuweather.CODIS_COLUMNS

    def test_covers_all_five_stations_with_no_temperature_gaps(self, hourly) -> None:
        report = accuweather.coverage_report(hourly)
        assert report.height == len(settings.ACCUWEATHER_STATIONS)
        assert report["氣溫缺值"].sum() == 0

    def test_uv_is_null_at_night_not_zero(self, hourly) -> None:
        """夜間 UV 為 null 是正常語意（約 49%），不可補 0。"""
        null_rate = hourly["UVIndex"].null_count() / hourly.height
        assert 0.4 < null_rate < 0.6

    def test_timestamps_are_unique_per_station(self, hourly) -> None:
        assert hourly.select("Date", "stn_ID").is_duplicated().sum() == 0


@needs_accuweather
class TestColumnContract:
    """預報與觀測的欄位必須一致：honest 模式靠它逐欄覆寫目標日。"""

    def test_forecast_columns_match_observation_exactly(self, forecast_raw) -> None:
        observed = external._load_weather_codis()
        assert set(forecast_raw.columns) == set(observed.columns)
        # date + 5 站 × 進入模型的項目（不含 UV，共 4 項）
        assert forecast_raw.width == 1 + len(settings.WEATHER_STATIONS) * len(settings.WEATHER_METRICS)


@needs_accuweather
class TestBiasCorrection:
    """偏差校正：修得掉偏差、修不掉雜訊。"""

    @staticmethod
    def _tmax(frame: pl.DataFrame) -> pl.DataFrame:
        columns = [f"w_{s}_day_tmax" for s in settings.WEATHER_STATIONS]
        return frame.with_columns(pl.max_horizontal(columns).alias("t")).select("date", "t")

    def test_uv_is_not_calibrated(self) -> None:
        """uv_max 與 CODiS 不是同一個量（r=0.431），線性校正無法對齊定義差異。"""
        assert "uv_max" not in accuweather_features.CALIBRATED_METRICS

    def test_correction_removes_the_warm_bias(self, forecast_raw) -> None:
        """未校正時五站 max 系統性偏高 ~1.2°C，校正後應回到 0 附近。"""
        observed = external._load_weather_codis()
        calibrated = accuweather_features.apply_bias_correction(
            forecast_raw,
            accuweather_features.fit_bias_correction(forecast_raw, observed),
        )
        truth = self._tmax(observed)
        raw = truth.join(self._tmax(forecast_raw), on="date", suffix="_f").drop_nulls()
        cal = truth.join(self._tmax(calibrated), on="date", suffix="_f").drop_nulls()

        raw_bias = (raw["t_f"] - raw["t"]).mean()
        cal_bias = (cal["t_f"] - cal["t"]).mean()
        assert raw_bias > 0.8, "重疊期的暖偏差應明顯為正"
        assert abs(cal_bias) < 0.2, "校正後偏差應接近 0"

    def test_correction_does_not_fix_the_noise(self, forecast_raw) -> None:
        """這是刻意記錄的限制：分箱不一致率仍在 10% 上下。

        誤差以隨機為主，不是系統性偏移——所以不值得做更複雜的校正。
        提交只有 3 天，期望約 0.3 天走錯條件組，這個風險無法消除。
        """
        observed = external._load_weather_codis()
        calibrated = accuweather_features.apply_bias_correction(
            forecast_raw,
            accuweather_features.fit_bias_correction(forecast_raw, observed),
        )
        joined = (self._tmax(observed)
                  .join(self._tmax(calibrated), on="date", suffix="_f").drop_nulls())
        truth = joined["t"].to_numpy()
        threshold = float(np.median(truth))
        disagree = ((joined["t_f"].to_numpy() >= threshold) != (truth >= threshold)).mean()
        assert 0.05 < disagree < 0.20

    def test_rejects_mismatched_columns(self, forecast_raw) -> None:
        with pytest.raises(ValueError, match="欄位契約已破裂"):
            accuweather_features.fit_bias_correction(
                forecast_raw, forecast_raw.drop(forecast_raw.columns[1])
            )


@needs_accuweather
class TestWeatherSourceDispatch:
    """兩種氣象來源必須產生同樣的欄位。"""

    def test_both_sources_produce_identical_columns(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "WEATHER_SOURCE", "codis")
        codis = external.load_weather()
        monkeypatch.setattr(settings, "WEATHER_SOURCE", "accuweather")
        aw = external.load_weather()
        assert set(codis.columns) == set(aw.columns)


@needs_accuweather
class TestHonestTargetWeather:
    """honest 模式的目標日氣象供應。"""

    ORIGIN = dt.date(2025, 6, 1)
    TARGETS = (dt.date(2025, 6, 2), dt.date(2025, 6, 3), dt.date(2025, 6, 4))

    @pytest.fixture(scope="class")
    def observed(self) -> pl.DataFrame:
        return external._load_weather_codis()

    def test_calibration_uses_only_history(self, forecast_raw, observed) -> None:
        """把 origin 之後的觀測放大 3 倍，輸出不得改變。

        若係數用全期擬合，目標日的觀測值會經由斜率與截距洩漏回來。
        """
        weather_columns = [c for c in observed.columns if c != "date"]
        tampered = observed.with_columns([
            pl.when(pl.col("date") > self.ORIGIN).then(pl.col(c) * 3)
            .otherwise(pl.col(c)).alias(c)
            for c in weather_columns
        ])
        clean = accuweather_features.make_target_weather_fn(forecast_raw, observed)
        dirty = accuweather_features.make_target_weather_fn(forecast_raw, tampered)
        assert clean(self.ORIGIN, self.TARGETS).equals(dirty(self.ORIGIN, self.TARGETS))

    def test_tmax_is_max_of_five_stations(self, forecast_raw, observed) -> None:
        """與 workflow.prepare_context 的算法一致：五站 day_tmax 取最大。"""
        out = accuweather_features.make_target_weather_fn(forecast_raw, observed)(
            self.ORIGIN, self.TARGETS
        )
        columns = [c for c in out.columns if c.endswith("_day_tmax")]
        expected = out.select(pl.max_horizontal(columns)).to_series()
        assert out["tmax"].equals(expected, check_names=False)

    def test_missing_target_day_raises_when_plan_b_is_abort(
            self, forecast_raw, observed, monkeypatch) -> None:
        """不可靜默退回目標日的觀測——那會悄悄變成用觀測評分。缺值只能走明確選定的備援。"""
        monkeypatch.setattr(settings, "FORECAST_PLAN_B", "abort")
        fn = accuweather_features.make_target_weather_fn(forecast_raw, observed)
        with pytest.raises(ValueError, match="Plan B 尚未選定"):
            fn(self.ORIGIN, (dt.date(2099, 1, 1),))


# =============================================================================
# 觀測延遲的補值
# =============================================================================


def _observed_hourly(ends: dict[str, str]) -> pl.DataFrame:
    """每站從 2026-06-28 01:00 起逐小時到指定時刻的觀測表（只含時間欄）。"""
    frames = [
        pl.DataFrame({"Date": pl.datetime_range(
            dt.datetime(2026, 6, 28, 1), dt.datetime.fromisoformat(end), "1h", eager=True)})
        .with_columns(pl.lit(station).alias("stn_ID"))
        for station, end in ends.items()
    ]
    return pl.concat(frames)


class TestObservedEnd:
    def test_complete_day_counts(self) -> None:
        hourly = _observed_hourly({"A": "2026-06-30 23:00", "B": "2026-06-30 23:00"})
        assert external.codis_observed_end(hourly) == dt.date(2026, 6, 30)

    def test_half_updated_day_does_not_count(self) -> None:
        """半天的資料會算出錯的當日最高溫，故不算完整日。"""
        hourly = _observed_hourly({"A": "2026-06-30 23:00", "B": "2026-06-30 10:00"})
        assert external.codis_observed_end(hourly) == dt.date(2026, 6, 29)


class TestUnavailableDates:
    @pytest.fixture(autouse=True)
    def cutoff(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "DATA_AVAILABLE_END", "2026-09-30")

    def test_nothing_missing(self) -> None:
        assert external.unavailable_observation_dates(dt.date(2026, 9, 30)) == []

    def test_one_day_behind_is_filled(self) -> None:
        assert external.unavailable_observation_dates(dt.date(2026, 9, 29)) == [
            dt.date(2026, 9, 30)]

    def test_two_days_behind_is_not_filled(self) -> None:
        """超過容許天數不補，由前置檢查中止並說明。"""
        assert external.unavailable_observation_dates(dt.date(2026, 9, 28)) == []

    def test_simulated_dates_are_included(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "CODIS_UNAVAILABLE_DATES", (dt.date(2025, 5, 1),))
        assert external.unavailable_observation_dates(dt.date(2026, 9, 30)) == [
            dt.date(2025, 5, 1)]


def test_fill_uses_only_real_observations_for_calibration(monkeypatch) -> None:
    """觀測 = 2 × 預報 + 1；補上的值要照這個關係，且不受補值日之後的觀測影響。"""
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=k) for k in range(60)]
    forecast = pl.DataFrame({
        "date": days,
        "w_A_day_tmax": [20.0 + (k % 7) for k in range(60)],
        "w_A_uv_max": [5.0] * 60,
    })
    fill_day = days[40]
    observed = forecast.filter(pl.col("date") != fill_day).with_columns(
        pl.when(pl.col("date") < fill_day)
        .then(pl.col("w_A_day_tmax") * 2 + 1)
        .otherwise(-999.0)                       # 補值日之後的觀測不得進入校正
        .alias("w_A_day_tmax"),
        pl.lit(3.0).alias("w_A_uv_max"),
    )
    monkeypatch.setattr(accuweather_features, "build_forecast", lambda: forecast)

    result = external._fill_from_forecast(observed, [fill_day])
    row = result.filter(pl.col("date") == fill_day)
    expected = forecast.filter(pl.col("date") == fill_day)["w_A_day_tmax"].item() * 2 + 1
    assert row["w_A_day_tmax"].item() == pytest.approx(expected)
    assert row["w_A_uv_max"].item() == 5.0       # UV 定義不同，不校正
    assert result.height == 60 and result["date"].is_sorted()


def test_fill_without_forecast_raises(monkeypatch) -> None:
    observed = pl.DataFrame({"date": [dt.date(2026, 1, 1)], "w_A_day_tmax": [20.0]})
    monkeypatch.setattr(accuweather_features, "build_forecast", lambda: observed)
    with pytest.raises(ValueError, match="Accuweather 也沒有"):
        external._fill_from_forecast(observed, [dt.date(2026, 1, 2)])


@needs_accuweather
class TestWithLocalData:
    """用本機實際資料，起點為資料截止日（``settings.DATA_AVAILABLE_END``）。"""

    ORIGIN = dt.date.fromisoformat(settings.DATA_AVAILABLE_END)
    DAYS = [ORIGIN + dt.timedelta(days=1), ORIGIN + dt.timedelta(days=2), ORIGIN + dt.timedelta(days=3)]

    @pytest.fixture
    def truncated(self, tmp_path, monkeypatch):
        """把 CODiS 截短到指定日（含）的函式。"""
        full = pl.read_csv(paths.WEATHER_FILE, infer_schema_length=0)

        def make(last: dt.date) -> None:
            path = tmp_path / "codis.csv"
            full.filter(pl.col("Date").str.slice(0, 10) <= last.isoformat()).write_csv(path)
            monkeypatch.setattr(paths, "WEATHER_FILE", path)
        return make

    def test_simulated_unavailable_day_is_replaced_only_there(self, monkeypatch) -> None:
        normal = external.load_weather()
        monkeypatch.setattr(settings, "CODIS_UNAVAILABLE_DATES", (dt.date(2026, 6, 1),))
        lagged = external.load_weather()
        assert lagged["date"].to_list() == normal["date"].to_list()
        differ = normal.join(lagged, on="date", suffix="_lag").filter(
            pl.col("w_臺北_day_tmax") != pl.col("w_臺北_day_tmax_lag"))["date"].to_list()
        assert dt.date(2026, 6, 1) in differ
        assert all(abs((d - dt.date(2026, 6, 1)).days) <= 1 for d in differ)
        raw = external._load_weather_codis(fill=False)
        assert dt.date(2026, 6, 1) not in raw["date"].to_list()

    def test_precheck_passes_one_day_behind(self, truncated) -> None:
        truncated(self.ORIGIN - dt.timedelta(days=1))
        result = checks.precheck(self.ORIGIN, self.DAYS)
        assert result["codis_filled"] == [self.ORIGIN]
        weather = external.load_weather()
        assert weather["date"].max() == self.ORIGIN

    def test_precheck_aborts_two_days_behind(self, truncated) -> None:
        truncated(self.ORIGIN - dt.timedelta(days=2))
        with pytest.raises(ValueError, match=f"CODiS 觀測只到 {self.ORIGIN - dt.timedelta(days=2)}"):
            checks.precheck(self.ORIGIN, self.DAYS)


# =============================================================================
# Windy 太陽光電
# =============================================================================


UNITS = ("甲光", "乙光", "天篷光")


def _windy_hourly(values: dict[tuple[str, int], float], day: dt.date = dt.date(2026, 6, 1)) -> pl.DataFrame:
    rows = [{"unit_name": u, "forecast_time": dt.datetime.combine(day, dt.time(h)),
             "forecast_power": values.get((u, h), 100.0)}
            for u in UNITS for h in settings.WINDY_HOURS]
    return pl.DataFrame(rows)


def test_average_excludes_listed_units() -> None:
    frame = _windy_hourly({("天篷光", 14): 9999.0, ("甲光", 14): 300.0, ("乙光", 14): 500.0})
    out = windy_features.daily_features(frame)
    assert out["pv_14"].item() == 400.0
    assert out["pv_sum"].item() == 100.0 * 7 + 400.0


def test_missing_unit_leaves_sum_empty() -> None:
    """某時刻缺任一機組就不算平均，當日加總留空（不可用較少機組的平均冒充）。"""
    frame = _windy_hourly({}).filter(~((pl.col("unit_name") == "甲光")
                                 & (pl.col("forecast_time").dt.hour() == 5)))
    out = windy_features.daily_features(frame)
    assert out["pv_sum"].item() is None
    assert out["pv_14"].item() == 100.0


def test_each_day_only_uses_its_own_rows() -> None:
    """擾動測試：改動某一天的預報，只有那一天的特徵會變。"""
    days = [dt.date(2026, 6, d) for d in (1, 2, 3)]
    base = pl.concat([_windy_hourly({}, d) for d in days])
    changed = base.with_columns(
        pl.when(pl.col("forecast_time").dt.date() == days[1]).then(pl.lit(777.0))
        .otherwise(pl.col("forecast_power")).alias("forecast_power"))
    a, b = windy_features.daily_features(base), windy_features.daily_features(changed)
    differ = a.join(b, on="date", suffix="_b").filter(pl.col("pv_sum") != pl.col("pv_sum_b"))
    assert differ["date"].to_list() == [days[1]]


@pytest.mark.skipif(not paths.WINDY_FILE.exists() or not paths.TARGETS_FILE.exists(),
                    reason="需要本機資料")
def test_flag_adds_exactly_the_registered_features(monkeypatch) -> None:
    """開啟 ENABLE_WINDY 只多出事前登記的兩個特徵，不多不少。"""
    from src.data import external
    from src.features import builder

    daily = pl.read_parquet(paths.TARGETS_FILE)
    calendar_df, rules = external.load_calendar(), external.load_price_period_rules()
    monkeypatch.setattr(settings, "ENABLE_WINDY", False)
    without = set(builder.feature_names(builder.build_features(daily, calendar_df, rules, 1)))
    monkeypatch.setattr(settings, "ENABLE_WINDY", True)
    with_windy = set(builder.feature_names(builder.build_features(daily, calendar_df, rules, 1)))
    assert with_windy - without == set(windy_features.FEATURES)
    assert without <= with_windy


def test_windy_missing_flags_incomplete_days(monkeypatch) -> None:
    from src.data import checks, windy

    days = [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)]
    frame = pl.concat([_windy_hourly({}, d) for d in days[:2]]).filter(
        ~((pl.col("forecast_time") == dt.datetime(2026, 6, 2, 14)) & (pl.col("unit_name") == "甲光")))
    monkeypatch.setattr(windy, "load_hourly", lambda: frame)
    # 6/2 缺 14:00 的一個機組 → pv_14、pv_sum 都算不出來；6/3 整天沒有
    assert checks.windy_missing(days) == [days[1], days[2]]
