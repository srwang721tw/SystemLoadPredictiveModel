"""日曆與電價制度特徵。

電價時段規則一律由外部檔案讀入，程式中嚴禁硬編碼：
    - 每日的日別（平日 / 週六 / 週日及離峰日）← `data/raw/` 的日曆表
    - 日內時段與夏月起訖 ← `config/price_periods.toml`

**daily-row 架構下該從電價制度取什麼**

日內的時段切分本身不是每日一列的量，必須先摘要。本模組取兩類：

1. **與尖峰窗口的重疊比例**：各費率在日尖峰（11:00–17:00）與夜尖峰
   （17:10–21:00）窗口內各佔幾成。這把「窗口內是什麼費率」變成連續特徵。
2. **窗口內的費率切換次數**：切換點會製造用電行為的轉折，可能就是尖峰時刻
   的成因。實測夏月平日在 16:00 有一個半尖峰→尖峰的跳升，正好落在 `t_day`
   第三個叢集（16:10–17:00）的起點；夏月週六則 09:00–24:00 全段同一費率，
   兩個窗口內都沒有切換點——這與「週六的時刻分布與平日明顯不同」相符。
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from config import settings
from src.features.targets import to_minutes
from src.logging_setup import get_logger

logger = get_logger(__name__)

OFFPEAK = "離峰"
"""未被任何區間涵蓋的時間一律為離峰。"""


def _parse_md(text: str) -> tuple[int, int]:
    """``"05-16"`` → ``(5, 16)``。"""
    month, day = text.split("-")
    return int(month), int(day)


def add_summer_flag(daily: pl.DataFrame, rules: dict) -> pl.DataFrame:
    """加上夏月旗標。

    夏月起訖由 ``config/price_periods.toml`` 的 ``[summer]`` 決定
    （目前為 5/16 – 10/15，含端點）。

    Args:
        daily: 含 ``date`` 的每日表。
        rules: :func:`~src.data.external.load_price_period_rules` 的輸出。

    Returns:
        pl.DataFrame: 加上 ``is_summer``。
    """
    start_month, start_day = _parse_md(rules["summer"]["start"])
    end_month, end_day = _parse_md(rules["summer"]["end"])
    month, day = pl.col("date").dt.month(), pl.col("date").dt.day()
    after_start = (month > start_month) | ((month == start_month) & (day >= start_day))
    before_end = (month < end_month) | ((month == end_month) & (day <= end_day))
    return daily.with_columns((after_start & before_end).alias("is_summer"))


def band_minutes(
    rules: dict, is_summer: bool, price_daytype: str, window: tuple[str, str]
) -> dict[str, int]:
    """計算指定窗口內各費率各佔幾分鐘。

    Args:
        rules: 電價時段規則。
        is_summer: 是否為夏月。
        price_daytype: 日曆表的日別（平日 / 週六 / 週日及離峰日）。
        window: 窗口起訖，例如 ``("11:00", "17:00")``。

    Returns:
        dict[str, int]: ``{費率名稱: 分鐘數}``，含 ``離峰``。
    """
    season = "summer" if is_summer else "non_summer"
    bands = rules["bands"][season].get(price_daytype, {})
    lo, hi = to_minutes(window[0]), to_minutes(window[1])
    span = hi - lo

    out: dict[str, int] = {}
    covered = 0
    for band_name, intervals in bands.items():
        minutes = 0
        for start, end in intervals:
            a, b = to_minutes(start), to_minutes(end)
            minutes += max(0, min(hi, b) - max(lo, a))
        out[band_name] = minutes
        covered += minutes
    out[OFFPEAK] = span - covered
    return out


def band_switches(
    rules: dict, is_summer: bool, price_daytype: str, window: tuple[str, str]
) -> int:
    """計算窗口**內部**的費率切換點數（不含窗口端點本身）。

    Args:
        rules: 電價時段規則。
        is_summer: 是否為夏月。
        price_daytype: 日別。
        window: 窗口起訖。

    Returns:
        int: 切換點數。
    """
    season = "summer" if is_summer else "non_summer"
    bands = rules["bands"][season].get(price_daytype, {})
    lo, hi = to_minutes(window[0]), to_minutes(window[1])
    boundaries = {
        m
        for intervals in bands.values()
        for start, end in intervals
        for m in (to_minutes(start), to_minutes(end))
    }
    return sum(1 for m in boundaries if lo < m < hi)


def add_price_band_features(daily: pl.DataFrame, rules: dict) -> pl.DataFrame:
    """加上尖峰窗口內的費率結構特徵。

    對日尖峰與夜尖峰兩個窗口，各產出：
        - ``{window}_share_{band}``：各費率佔該窗口的比例
        - ``{window}_price_switches``：窗口內的費率切換點數

    Args:
        daily: 含 ``date``、``is_summer``、``price_daytype`` 的每日表。
        rules: 電價時段規則。

    Returns:
        pl.DataFrame: 加上費率結構特徵。
    """
    windows = {
        "day": (settings.DAY_PEAK_START, settings.DAY_PEAK_END),
        "night": (settings.NIGHT_PEAK_START, settings.NIGHT_PEAK_END),
    }
    all_bands = sorted(
        {
            band
            for season in rules["bands"].values()
            for daytype_bands in season.values()
            for band in daytype_bands
        }
        | {OFFPEAK}
    )

    # 組合數很小（2 季 × 3 日別 × 2 窗口），先建查表再 join，避免逐列計算。
    combos = []
    for is_summer in (True, False):
        for price_daytype in rules["bands"]["summer"]:
            row: dict[str, object] = {
                "is_summer": is_summer,
                "price_daytype": price_daytype,
            }
            for window_name, window in windows.items():
                minutes = band_minutes(rules, is_summer, price_daytype, window)
                span = to_minutes(window[1]) - to_minutes(window[0])
                for band in all_bands:
                    row[f"{window_name}_share_{band}"] = minutes.get(band, 0) / span
                row[f"{window_name}_price_switches"] = band_switches(
                    rules, is_summer, price_daytype, window
                )
            combos.append(row)

    return daily.join(pl.DataFrame(combos), on=["is_summer", "price_daytype"], how="left")


def add_holiday_features(daily: pl.DataFrame) -> pl.DataFrame:
    """加上國定假日與連假結構特徵。

    「國定假日」定義為：星期上是平日或週六，但電價日曆歸為「週日及離峰日」。
    這是日曆表提供的權威資訊，比任何推導都可靠。

    連假結構（總長度、當日為第幾天、前後日）對用電的影響不只是「放假」——
    連假第一天與最後一天的行為差異很大（返鄉 vs 收假）。

    Args:
        daily: 含 ``date``、``daytype``（星期推導）、``price_daytype``（日曆）的每日表。

    Returns:
        pl.DataFrame: 加上假日與連假結構欄位。
    """
    offpeak_day = settings.DAYTYPES[2] + "及離峰日"  # "週日及離峰日"
    weekday, saturday = settings.DAYTYPES[0], settings.DAYTYPES[1]

    out = daily.sort("date").with_columns(
        (pl.col("price_daytype") == offpeak_day).alias("is_offpeak_day"),
        (
            pl.col("daytype").is_in([weekday, saturday])
            & (pl.col("price_daytype") == offpeak_day)
        ).alias("is_national_holiday"),
    )

    # 連假 = 連續的離峰日。以「非離峰日」的累計次數當作 run 的識別。
    out = out.with_columns(
        (~pl.col("is_offpeak_day")).cum_sum().alias("_run_id")
    ).with_columns(
        pl.when(pl.col("is_offpeak_day"))
        .then(pl.len().over("_run_id"))
        .otherwise(0)
        .alias("holiday_run_length"),
        pl.when(pl.col("is_offpeak_day"))
        .then(pl.col("date").cum_count().over("_run_id"))
        .otherwise(0)
        .alias("holiday_day_index"),
    )

    return out.with_columns(
        pl.col("is_offpeak_day").shift(-1, fill_value=False).alias("is_day_before_holiday"),
        pl.col("is_offpeak_day").shift(1, fill_value=False).alias("is_day_after_holiday"),
    ).drop("_run_id")


def add_calendar_features(
    daily: pl.DataFrame, calendar: pl.DataFrame, rules: dict
) -> pl.DataFrame:
    """一次完成所有日曆與電價制度特徵。

    Args:
        daily: 每日表（須已含 ``daytype``）。
        calendar: :func:`~src.data.external.load_calendar` 的輸出。
        rules: :func:`~src.data.external.load_price_period_rules` 的輸出。

    Returns:
        pl.DataFrame: 加上日曆、夏月、費率結構、假日與連假結構特徵。
    """
    if "price_daytype" not in daily.columns:
        daily = daily.join(calendar, on="date", how="left")
    out = add_summer_flag(daily, rules)
    out = add_price_band_features(out, rules)
    out = add_holiday_features(out)
    logger.info(
        "日曆特徵：夏月 %d 天、國定假日 %d 天、離峰日合計 %d 天",
        out["is_summer"].sum(),
        out["is_national_holiday"].sum(),
        out["is_offpeak_day"].sum(),
    )
    return out


def add_trend_feature(daily: pl.DataFrame) -> pl.DataFrame:
    """加上年內序位與線性時間索引。

    **不加年度趨勢項。** 實測年增率「跳一年、平一年」
    （2024→2025 +6.5~10%、2025→2026 −0.5%），趨勢外推會嚴重高估；
    水準一律交給 ``rebase`` 的 ``base``。
    此處的時間索引只用於捕捉年內季節，不用於外推水準。

    Args:
        daily: 含 ``date`` 的每日表。

    Returns:
        pl.DataFrame: 加上 ``day_of_year``、``days_since_start``。
    """
    return daily.with_columns(
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
        (pl.col("date") - pl.col("date").min()).dt.total_days().alias("days_since_start"),
    )


def add_typhoon_feature(
    daily: pl.DataFrame, typhoon: pl.DataFrame | None = None
) -> pl.DataFrame:
    """加上颱風停班停課特徵。

    **「是否颱風」與「是否影響全系統負載」不是同一件事。**
    實測（平日 ``ramp_up`` 中位數 1165 MW）：

    | 颱風日 | ``ramp_up`` | vs 中位 |
    |---|---|---|
    | 2024-07-25 凱米 | 383 | 33% |
    | 2024-10-02 山陀兒 | 416 | 36% |
    | 2025-08-13 楊柳 | **1390** | **119%（完全正常）** |
    | 2025-09-23 樺加沙 | **1332** | **114%（完全正常）** |

    原因：楊柳只停台東／高雄／台南／屏東／嘉義，樺加沙主要
    影響花蓮台東——**負載集中在北部，南東部停班對全系統影響有限**。
    因此本函式除了二元旗標外，也把「影響範圍」編碼出來（若清單有該欄）。

    推論時目標日的旗標只採用作業時點以前已公告者（``mask_unannounced_typhoon``），
    這個特徵的價值在於**避免歷史上的颱風日污染條件分布**，而非預測。

    Args:
        daily: 每日表。
        typhoon: 颱風清單；None 時依開關決定是否讀檔。

    Returns:
        pl.DataFrame: 加上 ``is_typhoon_day``，清單有 ``影響範圍`` 時
            另加 ``typhoon_scope``（無颱風日為 ``"無"``）。
    """
    if not settings.ENABLE_TYPHOON_FEATURE:
        logger.warning("颱風特徵未啟用（ENABLE_TYPHOON_FEATURE=False），全部設 0")
        return daily.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_typhoon_day"))

    if typhoon is None:
        from src.data import external

        typhoon = external.load_typhoon_days()

    if typhoon.height == 0:
        logger.warning("颱風清單為空，is_typhoon_day 全部設 0")
        return daily.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_typhoon_day"))

    columns = ["date"] + [c for c in ("影響範圍", "侵臺路徑分類") if c in typhoon.columns]
    out = daily.join(typhoon.select(columns), on="date", how="left")

    marker = "影響範圍" if "影響範圍" in out.columns else (
        "侵臺路徑分類" if "侵臺路徑分類" in out.columns else None
    )
    out = out.with_columns(
        pl.col(marker).is_not_null().cast(pl.Int8).alias("is_typhoon_day")
        if marker
        else pl.lit(0, dtype=pl.Int8).alias("is_typhoon_day")
    )
    if "影響範圍" in out.columns:
        out = out.with_columns(
            pl.col("影響範圍").fill_null("無").alias("typhoon_scope")
        ).drop("影響範圍")

    if "侵臺路徑分類" in out.columns:
        out = out.with_columns(
            (
                pl.col("侵臺路徑分類").is_not_null()
                & ~pl.col("侵臺路徑分類").is_in(settings.TYPHOON_BENIGN_PATHS)
            ).cast(pl.Int8).alias("is_typhoon_impacting")
        ).drop("侵臺路徑分類")
    else:
        out = out.with_columns(pl.col("is_typhoon_day").alias("is_typhoon_impacting"))

    n_hit = int(out["is_typhoon_day"].sum())
    n_impact = int(out["is_typhoon_impacting"].sum())
    logger.info(
        "颱風特徵：清單 %d 天，其中 %d 天落在資料範圍內（%d 天超出，待資料補齊）；"
        "其中 %d 天路徑屬於會影響全系統負載者",
        typhoon.height, n_hit, typhoon.height - n_hit, n_impact,
    )
    return out


TYPHOON_FEATURE_DEFAULTS: dict[str, object] = {
    "is_typhoon_day": 0,
    "is_typhoon_impacting": 0,
    "typhoon_scope": "無",
}
"""颱風特徵欄與「無颱風」時的值（見 :func:`add_typhoon_feature`）。"""


def typhoon_announcements(typhoon: pl.DataFrame | None = None) -> dict[dt.date, dt.datetime]:
    """颱風停班停課清單中，**有公告時間**的日期 → 公告時間。

    Args:
        typhoon: 颱風清單；None 時讀檔。

    Returns:
        dict: 缺公告時間的列不列入（視為未公告）。
    """
    if typhoon is None:
        from src.data import external

        typhoon = external.load_typhoon_days()
    if "公告時間" not in typhoon.columns or typhoon.height == 0:
        return {}
    stamps = typhoon.select(
        "date",
        pl.col("公告時間").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False),
    ).drop_nulls("公告時間")
    return dict(stamps.iter_rows())


def mask_unannounced_typhoon(
    rows: pl.DataFrame, origin: dt.date, announcements: dict[dt.date, dt.datetime]
) -> pl.DataFrame:
    """把目標列中「作業時點以前尚未公告」的颱風資訊改回無颱風的值。

    **洩漏防線**。颱風停班停課清單是事後整理的；作業時點
    （起點次日的 ``settings.OPERATION_TIME``）不可能知道 D+1、D+2 會不會停班。
    只有公告時間早於作業時點的，才能用在目標日。**歷史列（訓練用）不受影響**。

    Args:
        rows: 特徵矩陣中目標日那幾列。
        origin: 預測起點日（負載資料的最後一天）。
        announcements: :func:`typhoon_announcements` 的輸出。

    Returns:
        pl.DataFrame: 欄位與順序同 ``rows``。
    """
    hour, minute = (int(x) for x in settings.OPERATION_TIME.split(":"))
    cutoff = dt.datetime.combine(origin + dt.timedelta(days=1), dt.time(hour, minute))
    known = [d for d, stamp in announcements.items() if stamp <= cutoff]
    unknown = ~pl.col("date").is_in(known)
    return rows.with_columns([
        pl.when(unknown).then(pl.lit(value, dtype=rows.schema[column]))
        .otherwise(pl.col(column)).alias(column)
        for column, value in TYPHOON_FEATURE_DEFAULTS.items() if column in rows.columns
    ])

