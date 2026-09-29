"""日曆類特徵與人工清單：颱風公告時間、農曆與節氣、特殊日期、颱風與事件日清單、同日別基準。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from config import settings
from src.data import external, special_days
from src.features import calendar, lunar, rebase


# =============================================================================
# 颱風公告時間
# =============================================================================


class TestTyphoonAnnouncement:
    """目標日只能用作業時點以前已公告的颱風資訊。"""

    ROWS = pl.DataFrame({
        "date": [dt.date(2026, 10, 1), dt.date(2026, 10, 2)],
        "is_typhoon_day": pl.Series([1, 1], dtype=pl.Int8),
        "is_typhoon_impacting": pl.Series([1, 1], dtype=pl.Int8),
        "typhoon_scope": ["全台", "全台"],
        "lag_1": [1.0, 2.0],
    })
    ORIGIN = dt.date(2026, 9, 30)

    def test_unannounced_days_are_reset(self) -> None:
        out = calendar.mask_unannounced_typhoon(self.ROWS, self.ORIGIN, {})
        assert out["is_typhoon_day"].to_list() == [0, 0]
        assert out["typhoon_scope"].to_list() == ["無", "無"]
        assert out["lag_1"].to_list() == [1.0, 2.0]
        assert out.columns == self.ROWS.columns

    def test_only_announcements_before_operation_time_count(self) -> None:
        """9/30 20:00 公告 10/1 停班 → 10/1 08:30 已知；10/1 20:00 才公告 10/2 → 未知。"""
        announced = {
            dt.date(2026, 10, 1): dt.datetime(2026, 9, 30, 20, 0),
            dt.date(2026, 10, 2): dt.datetime(2026, 10, 1, 20, 0),
        }
        out = calendar.mask_unannounced_typhoon(self.ROWS, self.ORIGIN, announced)
        assert out["is_typhoon_day"].to_list() == [1, 0]
        assert out["is_typhoon_impacting"].to_list() == [1, 0]

    def test_announcements_skip_rows_without_time(self) -> None:
        table = pl.DataFrame({"date": [dt.date(2026, 10, 1), dt.date(2026, 10, 2)],
                              "公告時間": ["2026-09-30 20:00", None]})
        assert calendar.typhoon_announcements(table) == {
            dt.date(2026, 10, 1): dt.datetime(2026, 9, 30, 20, 0)}

    def test_shipped_list_has_no_announcement_times(self) -> None:
        """現有清單的公告時間全部空白——不代填，回測中目標日一律視為未公告。"""
        assert calendar.typhoon_announcements() == {}


# =============================================================================
# 農曆與節氣
# =============================================================================


def make_calendar(rows: list[tuple[str, str, str | None]]) -> pl.DataFrame:
    """構造迷你日曆表。

    Args:
        rows: ``(日期, 農曆, 節氣)`` 串列。

    Returns:
        pl.DataFrame: 欄位 ``date`` / ``lunar`` / ``solar_term``。
    """
    return pl.DataFrame({
        "date": [dt.date.fromisoformat(r[0]) for r in rows],
        "lunar": [r[1] for r in rows],
        "solar_term": [r[2] for r in rows],
    })


class TestParseLunar:
    """農曆字串解析——四種日的前綴都要對。"""

    @pytest.mark.parametrize("text,expected", [
        ("正月初一", (1, 1)),      # 春節
        ("八月十五", (8, 15)),     # 中秋
        ("五月初五", (5, 5)),      # 端午
        ("十二月三十", (12, 30)),  # 除夕
        ("十一月初七", (11, 7)),
        ("九月廿六", (9, 26)),
        ("正月二十", (1, 20)),
        ("三月十一", (3, 11)),
        ("六月初十", (6, 10)),
    ])
    def test_parses_all_day_prefixes(self, text: str, expected: tuple[int, int]) -> None:
        assert lunar.parse_lunar(text) == expected

    def test_thirty_is_not_parsed_as_ten(self) -> None:
        """``三十`` 必須先於 ``十`` 比對。

        若比對順序寫反，``三十`` 會被當成前綴 ``十`` 而算成 10——
        錯 20 天，且不會報錯。
        """
        assert lunar.parse_lunar("十二月三十") == (12, 30)
        assert lunar.parse_lunar("十二月初十") == (12, 10)


    @pytest.mark.parametrize("text", [None, "", "沒有月字", "十三月初一"])
    def test_rejects_unparseable(self, text) -> None:
        assert lunar.parse_lunar(text) is None


class TestLunarFeatures:
    """特徵組裝。"""

    def test_mid_autumn_is_flagged(self) -> None:
        """八月十五 = 中秋，且前兩天的 days_to 應為 +2。"""
        out = lunar.add_lunar_features(make_calendar([
            ("2025-10-04", "八月十三", None),
            ("2025-10-05", "八月十四", None),
            ("2025-10-06", "八月十五", None),
            ("2025-10-07", "八月十六", None),
        ]))
        assert out["is_lunar_festival"].to_list() == [0, 0, 1, 0]
        assert out["days_to_lunar_festival"].to_list() == [2, 1, 0, -1]


    def test_new_year_eve_detected_without_hardcoding_the_day(self) -> None:
        """除夕是「該農曆年的最後一天」，可能是三十也可能是廿九。"""
        out = lunar.add_lunar_features(make_calendar([
            ("2025-01-28", "十二月廿九", None),
            ("2025-01-29", "正月初一", None),
        ]))
        assert out["is_lunar_new_year_eve"].to_list() == [1, 0]

    def test_solar_term_is_forward_filled(self) -> None:
        """節氣只標在當天，其餘要前向填補成「目前處於哪個節氣」。"""
        out = lunar.add_lunar_features(make_calendar([
            ("2025-09-23", "八月初二", "秋分"),
            ("2025-09-24", "八月初三", None),
            ("2025-09-25", "八月初四", None),
        ]))
        idx = settings.SOLAR_TERMS.index("秋分")
        assert out["solar_term_index"].to_list() == [idx, idx, idx]
        assert out["days_since_solar_term"].to_list() == [0, 1, 2]

    def test_days_since_resets_on_each_new_term(self) -> None:
        """不可依「節氣的值」分組——那會把不同年的同一個節氣併在一起。

        分組錯了會讓 `days_since_solar_term` 最大值超過 40 天（正常 ≤ 16）。
        """
        out = lunar.add_lunar_features(make_calendar([
            ("2025-09-23", "八月初二", "秋分"),
            ("2025-09-24", "八月初三", None),
            ("2025-10-08", "八月十七", "寒露"),
            ("2025-10-09", "八月十八", None),
        ]))
        assert out["days_since_solar_term"].to_list() == [0, 1, 0, 1]

    def test_gapped_frame_still_correct(self) -> None:
        """推論時 `daily` 是「訓練資料 + 三個預測日」，中間有幾個月的斷裂。

        特徵必須在連續的日曆上算完再併回來。直接在斷裂的 frame 上算，
        2026-10-01 會得到「距節日 −12、節氣 = 夏至」，正確值是「−6、秋分」，
        訓練與推論算出不同的值，而且不會報錯。
        """
        from src.data import external

        cal = external.load_calendar()
        gapped = pl.concat([
            cal.filter(pl.col("date").is_between(dt.date(2026, 6, 1), dt.date(2026, 6, 30))),
            cal.filter(pl.col("date").is_between(dt.date(2026, 10, 1), dt.date(2026, 10, 3))),
        ]).select("date")
        out = lunar.add_lunar_features(gapped)
        row = out.filter(pl.col("date") == dt.date(2026, 10, 1)).row(0, named=True)
        assert row["days_to_lunar_festival"] == -6, "應量到 2026 中秋（9/25）"
        assert row["solar_term_index"] == settings.SOLAR_TERMS.index("秋分")

    def test_requires_source_columns(self) -> None:
        with pytest.raises(ValueError, match="缺少欄位"):
            lunar.add_lunar_features(
                pl.DataFrame({"date": [dt.date(2025, 1, 1)]}),
                source=pl.DataFrame({"date": [dt.date(2025, 1, 1)]}),
            )


class TestRealCalendar:
    """對真實日曆表的檢核。"""

    def test_every_day_parses(self) -> None:
        """2024–2026 每一天都要能解析，來源改格式時由這條擋下。"""
        from src.data import external

        cal = external.load_calendar().filter(
            pl.col("date").dt.year().is_between(2024, 2026)
        )
        bad = [v for v in cal["lunar"].to_list() if lunar.parse_lunar(v) is None]
        assert not bad, f"無法解析的農曆字串：{bad[:5]}"

    def test_solar_term_gap_is_plausible(self) -> None:
        """節氣間隔約 14–16 天，超過就是分組邏輯錯了。"""
        from src.data import external

        out = lunar.add_lunar_features(external.load_calendar().filter(
            pl.col("date").dt.year().is_between(2024, 2026)
        ))
        assert out["days_since_solar_term"].max() <= 20


# =============================================================================
# 特殊日期
# =============================================================================


def make_intervals(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """構造區間表。

    Args:
        rows: ``(類別, 起, 迄)`` 串列。

    Returns:
        pl.DataFrame: 欄位同 :data:`special_days.INTERVAL_COLUMNS`。
    """
    return pl.DataFrame({
        "類別": [r[0] for r in rows],
        "起": [dt.date.fromisoformat(r[1]) for r in rows],
        "迄": [dt.date.fromisoformat(r[2]) for r in rows],
        "說明": ["測試"] * len(rows),
    })


class TestExpand:
    """區間展開。"""

    def test_endpoints_are_inclusive(self) -> None:
        """起日與迄日都算在內。

        差一天的錯誤在這裡最容易發生，而且不會有任何徵兆。
        """
        out = special_days.expand(
            make_intervals([("is_holiday", "2026-02-14", "2026-02-16")]),
            dt.date(2026, 2, 13), dt.date(2026, 2, 17),
        )
        assert out["is_holiday"].to_list() == [0, 1, 1, 1, 0]

    def test_single_day_interval(self) -> None:
        """起 = 迄 的單日區間只標一天。"""
        out = special_days.expand(
            make_intervals([("is_holiday", "2026-01-01", "2026-01-01")]),
            dt.date(2025, 12, 31), dt.date(2026, 1, 2),
        )
        assert out["is_holiday"].to_list() == [0, 1, 0]

    def test_overlapping_intervals_stay_binary(self) -> None:
        """同類別的區間重疊時仍然只是 1——旗標是「有沒有」，不是次數。"""
        out = special_days.expand(
            make_intervals([("is_vacation", "2026-07-01", "2026-07-10"),
                            ("is_vacation", "2026-07-05", "2026-07-15")]),
            dt.date(2026, 7, 1), dt.date(2026, 7, 15),
        )
        assert set(out["is_vacation"].to_list()) == {1}
        assert out["is_vacation"].sum() == 15

    def test_intervals_outside_range_are_ignored(self) -> None:
        """落在展開範圍外的區間不影響結果，也不報錯。"""
        out = special_days.expand(
            make_intervals([("is_holiday", "2023-12-30", "2024-01-01")]),
            dt.date(2024, 6, 1), dt.date(2024, 6, 3),
        )
        assert out["is_holiday"].sum() == 0

    def test_all_categories_present_even_when_empty(self) -> None:
        """沒有任何區間的類別仍要有欄位，整欄為 0。"""
        out = special_days.expand(
            make_intervals([("is_holiday", "2026-01-01", "2026-01-01")]),
            dt.date(2026, 1, 1), dt.date(2026, 1, 2),
        )
        for category in special_days.CATEGORIES:
            assert category in out.columns
        assert out["is_exam"].sum() == 0


class TestLoadIntervals:
    """區間表的檢核。"""

    def test_rejects_unknown_category(self, tmp_path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("類別,起,迄,說明\nis_typo,2026-01-01,2026-01-02,x\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="未知類別"):
            special_days.load_intervals(path)

    def test_rejects_reversed_interval(self, tmp_path) -> None:
        """起日晚於迄日會靜默展開成空區間，故必須擋下來。"""
        path = tmp_path / "x.csv"
        path.write_text("類別,起,迄,說明\nis_holiday,2026-01-05,2026-01-01,x\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="起日晚於迄日"):
            special_days.load_intervals(path)

    def test_rejects_missing_column(self, tmp_path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("類別,起,說明\nis_holiday,2026-01-01,x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="缺少欄位"):
            special_days.load_intervals(path)


class TestSubmissionDates:
    """提交日的旗標值。

    2026-10-01~03 五個旗標全部為 0，所以這批變數只作用在訓練池，不會在提交日觸發。
    改動區間表而讓提交日落進某個旗標時，模型的輸入會無聲改變，由這條測試擋下。
    """

    def test_all_flags_are_zero_on_submission_dates(self) -> None:
        out = special_days.expand(
            special_days.load_intervals(),
            settings.SPECIAL_DAYS_START, settings.SPECIAL_DAYS_END,
        )
        submission = out.filter(
            pl.col("date").is_between(dt.date(2026, 10, 1), dt.date(2026, 10, 3))
        )
        assert submission.height == 3
        for category in special_days.CATEGORIES:
            assert submission[category].sum() == 0, (
                f"{category} 在提交日不再是 0——區間表被改動了。"
                "請先確認這是刻意的，再更新本測試。"
            )

    def test_expansion_covers_the_prediction_period(self) -> None:
        """展開範圍必須涵蓋提交日，否則 predict 會缺值。"""
        # settings.PREDICT_START 是字串（提交檔的時間欄位一律是字串，
        #   見 settings 的時間格式常數），這裡要比較日期得先轉型。
        first = dt.date.fromisoformat(settings.PREDICT_START)
        last = first + dt.timedelta(days=settings.PREDICT_HORIZON_DAYS - 1)
        assert settings.SPECIAL_DAYS_START <= first
        assert settings.SPECIAL_DAYS_END >= last


class TestRealIntervalTable:
    """交付用的那份區間表本身。"""

    def test_every_category_has_at_least_one_interval(self) -> None:
        intervals = special_days.load_intervals()
        present = set(intervals["類別"].to_list())
        assert present == set(special_days.CATEGORIES)


# =============================================================================
# 人工清單
# =============================================================================


def write_csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestTyphoonList:
    """颱風清單：分次補齊欄位時，載入器都不該需要改動。"""

    def test_extra_columns_are_passed_through(self, tmp_path: Path) -> None:
        # 侵臺路徑分類與近臺強度是後來才補的，載入器不該因此報錯。
        path = write_csv(tmp_path, "t.csv",
                         "颱風名稱,日期,影響範圍,侵臺路徑分類,近臺強度\n"
                         "凱米,2024-07-24,全台,2,強烈\n")
        out = external.load_typhoon_days(path)
        assert out["侵臺路徑分類"][0] == 2
        assert out["近臺強度"][0] == "強烈"


    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "t.csv", "颱風名稱,起始日\n凱米,2024-07-24\n")
        with pytest.raises(ValueError, match="必要欄位"):
            external.load_typhoon_days(path)

    def test_duplicate_date_raises(self, tmp_path: Path) -> None:
        # 一天一列是下游 join 的前提；重複會靜默放大列數。
        path = write_csv(tmp_path, "t.csv",
                         "颱風名稱,日期\n凱米,2024-07-24\n山陀兒,2024-07-24\n")
        with pytest.raises(ValueError, match="重複日期"):
            external.load_typhoon_days(path)


    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        out = external.load_typhoon_days(tmp_path / "不存在.csv")
        assert out.height == 0


class TestEventDays:
    """事件日：與颱風分開記錄，因為機制與處置都不同。"""

    def test_reads_required_columns(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "e.csv",
                         "日期,事件,類型\n2024-04-03,花蓮地震 M7.2,地震\n")
        out = external.load_event_days(path)
        assert out["事件"][0] == "花蓮地震 M7.2"

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "e.csv", "日期,說明\n2024-04-03,地震\n")
        with pytest.raises(ValueError, match="必要欄位"):
            external.load_event_days(path)

    def test_shipped_file_contains_the_earthquake(self) -> None:
        out = external.load_event_days()
        assert dt.date(2024, 4, 3) in out["date"].to_list()


class TestTyphoonImpactFlag:
    """路徑分級旗標：路徑有分辨力、強度沒有。"""

    @staticmethod
    def _daily(dates: list[str]) -> pl.DataFrame:
        return pl.DataFrame({"date": [dt.date.fromisoformat(d) for d in dates]})

    def test_benign_paths_are_flagged_as_not_impacting(self) -> None:
        from src.features import calendar

        typhoon = pl.DataFrame({
            "date": [dt.date(2025, 8, 13),
                     dt.date(2024, 7, 24)],
            "影響範圍": ["南部東部", "全台"],
            "侵臺路徑分類": [4, 2],
        })
        out = calendar.add_typhoon_feature(
            self._daily(["2025-08-13", "2024-07-24", "2024-01-01"]), typhoon
        )
        # 兩天都是颱風日，但只有路徑 2 那天算「有影響」。
        assert out["is_typhoon_day"].to_list() == [1, 1, 0]
        assert out["is_typhoon_impacting"].to_list() == [0, 1, 0]

    def test_benign_paths_come_from_settings(self) -> None:
        # 門檻不得硬編碼在特徵程式裡。
        assert settings.TYPHOON_BENIGN_PATHS == (4, 5)

    def test_falls_back_when_path_column_absent(self) -> None:
        # 舊格式（無路徑欄）時，分級旗標應退化為二元旗標而非拋錯。
        from src.features import calendar

        typhoon = pl.DataFrame({
            "date": [dt.date(2024, 7, 24)],
            "影響範圍": ["全台"],
        })
        out = calendar.add_typhoon_feature(self._daily(["2024-07-24"]), typhoon)
        assert out["is_typhoon_impacting"].to_list() == out["is_typhoon_day"].to_list()


# =============================================================================
# 同日別基準
# =============================================================================


BASE_N_DAYS = 200


BASE_START = dt.date(2025, 1, 1)


def synthetic_base_daily(seed: int = 0, spike_date: dt.date | None = None) -> pl.DataFrame:
    """構造含三種日別、有明顯水準漂移的假每日表。"""
    rng = np.random.default_rng(seed)
    dates = [BASE_START + dt.timedelta(days=i) for i in range(BASE_N_DAYS)]
    daytypes = [
        "平日" if d.isoweekday() <= 5 else ("週六" if d.isoweekday() == 6 else "週日及離峰日")
        for d in dates
    ]
    level = {"平日": 1.0, "週六": 0.85, "週日及離峰日": 0.8}
    values = [
        30000 * level[t] * (1 + i / BASE_N_DAYS * 0.1) + rng.normal(0, 200)
        for i, t in enumerate(daytypes)
    ]
    if spike_date is not None:
        idx = (spike_date - BASE_START).days
        values[idx] = 999999.0
    return pl.DataFrame(
        {"date": dates, "price_daytype": daytypes, "p_day": values}
    )


BASE_TARGETS = ("p_day",)


class TestBaseLookup:
    """基準值的計算。"""

    def test_base_is_median_of_recent_same_daytype(self) -> None:
        daily = pl.DataFrame(
            {
                "date": [BASE_START + dt.timedelta(days=i) for i in range(8)],
                "price_daytype": ["平日"] * 8,
                "p_day": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )
        lookup = rebase.base_lookup(daily, "p_day", n_occurrences=4)
        # 第 8 列（80）的基準 = median(50, 60, 70, 80) = 65
        assert lookup["base"][-1] == pytest.approx(65.0)

    def test_base_is_inclusive_of_current_row(self) -> None:
        # base_lookup 給的是「截至該日（含）」；不含當日的保證由
        # add_base_columns 的 asof join 在 T = D − h 上取值來達成。
        daily = pl.DataFrame(
            {
                "date": [BASE_START + dt.timedelta(days=i) for i in range(4)],
                "price_daytype": ["平日"] * 4,
                "p_day": [10.0, 10.0, 10.0, 100.0],
            }
        )
        lookup = rebase.base_lookup(daily, "p_day", n_occurrences=4)
        assert lookup["base"][-1] == pytest.approx(np.median([10, 10, 10, 100]))

    def test_daytypes_do_not_mix(self) -> None:
        daily = synthetic_base_daily()
        lookup = rebase.base_lookup(daily, "p_day", n_occurrences=4)
        merged = daily.join(lookup, on=["date", "price_daytype"], how="left")
        saturdays = merged.filter(pl.col("price_daytype") == "週六").drop_nulls("base")
        weekdays = merged.filter(pl.col("price_daytype") == "平日").drop_nulls("base")
        # 週六水準明顯低於平日，若混用日別，兩者的基準會靠攏。
        assert saturdays["base"].mean() < weekdays["base"].mean() * 0.95


class TestBaseNoLeakage:
    """base 不得使用目標日當天或之後的資料。"""

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_perturbing_a_day_does_not_change_earlier_rows(self, horizon: int) -> None:
        probe = BASE_START + dt.timedelta(days=120)
        clean = rebase.add_base_columns(synthetic_base_daily(), horizon, BASE_TARGETS)
        spiked = rebase.add_base_columns(
            synthetic_base_daily(spike_date=probe), horizon, BASE_TARGETS
        )
        before_clean = clean.filter(pl.col("date") <= probe).select("p_day_base")
        before_spiked = spiked.filter(pl.col("date") <= probe).select("p_day_base")
        assert before_clean.equals(before_spiked), (
            f"horizon={horizon}：擾動 {probe} 之後，該日及更早的 base 發生變化"
        )

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_target_day_value_never_enters_its_own_base(self, horizon: int) -> None:
        # 目標日 D 的 base 取自 T = D − horizon 及之前，故擾動 D 本身
        # 不得改變 D 的 base。
        probe = BASE_START + dt.timedelta(days=150)
        clean = rebase.add_base_columns(synthetic_base_daily(), horizon, BASE_TARGETS)
        spiked = rebase.add_base_columns(
            synthetic_base_daily(spike_date=probe), horizon, BASE_TARGETS
        )
        a = clean.filter(pl.col("date") == probe)["p_day_base"][0]
        b = spiked.filter(pl.col("date") == probe)["p_day_base"][0]
        assert a == pytest.approx(b)

    def test_larger_horizon_uses_older_base(self) -> None:
        # h 越大，基準日越早，故 base 應反映更舊的水準。序列有 +10% 漂移，
        # 因此 h=3 的 base 平均應低於 h=1。
        daily = synthetic_base_daily()
        h1 = rebase.add_base_columns(daily, 1, BASE_TARGETS)["p_day_base"].drop_nulls()
        h3 = rebase.add_base_columns(daily, 3, BASE_TARGETS)["p_day_base"].drop_nulls()
        assert h3.mean() < h1.mean()


class TestReliabilityAndOutliers:
    """基準不可靠與比值離群的標記。"""

    def test_first_rows_of_each_daytype_have_no_reliable_base(self) -> None:
        # 契約：每個日別的前 REBASE_MIN_OCCURRENCES 列湊不出基準，
        # 必須被標為不可靠而非硬算一個數字出來。
        out = rebase.add_base_columns(synthetic_base_daily(), 1, BASE_TARGETS)
        for daytype in ("平日", "週六", "週日及離峰日"):
            head = out.filter(pl.col("price_daytype") == daytype).head(
                settings.REBASE_MIN_OCCURRENCES
            )
            assert not head["base_reliable"].any(), daytype


    def test_outlier_flagged_but_not_removed(self) -> None:
        # 只標記、不刪除。
        probe = BASE_START + dt.timedelta(days=150)
        out = rebase.add_base_columns(
            synthetic_base_daily(spike_date=probe), 1, BASE_TARGETS
        )
        flagged = rebase.flag_outlier_ratios(out, BASE_TARGETS)
        assert flagged.height == out.height  # 一列都沒少
        assert flagged.filter(pl.col("date") == probe)["p_day_ratio_outlier"][0]
