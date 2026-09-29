"""由 CODiS 小時觀測算出每日氣象特徵。

每個測站產生 5 項：白天最高／最低溫、夜晚最高／最低溫、全日 UV 最大值。
進入模型的只有 ``settings.WEATHER_METRICS`` 列出的 4 個溫度項（五站共 20 個特徵）。

**白天是 hour 7–18**。CODiS 的 ``DataTime`` 是**區間終點**——
第一筆 ``01:00`` 涵蓋 00:00–01:00，故 06:00–18:00 對應 hour 7 到 18。
定義在 ``settings.WEATHER_DAY_HOURS``，邊界由測試綁住。
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

METRICS = ("day_tmax", "day_tmin", "night_tmax", "night_tmin", "uv_max")
"""每站產生的 5 個日特徵。"""


def _hour_index(column: str = "Date") -> pl.Expr:
    """由時間戳取出 1–24 的小時索引。

    ``23:59`` 那一筆代表 23:00–24:00，其小時部分是 23，
    但它是當日的**第 24 格**，必須映射為 24——否則它會與 ``23:00``
    （代表 22:00–23:00）撞成同一格。兩者都屬夜間，故不影響日／夜切分，
    但會讓「一天必須有 24 筆」的完整性檢查失效。
    """
    hour = pl.col(column).dt.hour().cast(pl.Int32)
    minute = pl.col(column).dt.minute().cast(pl.Int32)
    return pl.when(minute > 0).then(hour + 1).otherwise(hour)


def interpolate_hourly(hourly: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """對各測站的小時序列做線性插補。

    **絕不補 0**：氣溫 0°C 與「沒有資料」是兩件完全不同的事，
    補 0 會讓 ``day_tmin`` 直接變成 0。CODiS 的 −99 系列哨兵已在
    :mod:`src.data.weather` 轉為 null，這裡只處理真正的缺漏。

    Args:
        hourly: 含 ``Date``（Datetime）、``stn_ID``、兩個量測欄位。

    Returns:
        tuple: ``(插補後的表, 診斷)``。診斷含各欄插補格數與長段缺漏清單。
    """
    out = hourly.sort("stn_ID", "Date")
    diagnostics: dict = {"filled": {}, "long_gaps": []}

    for column in ("AirTemperature", "UVIndex"):
        before = out[column].null_count()
        out = out.with_columns(
            pl.col(column).interpolate().over("stn_ID").alias(column)
        )
        # 序列頭尾的缺漏無法內插，用該站最近的有效值補齊。
        out = out.with_columns(
            pl.col(column).forward_fill().backward_fill().over("stn_ID").alias(column)
        )
        diagnostics["filled"][column] = before

    # 長段缺漏要列出而非靜默處理（與負載長段缺失的作法相同）。
    # 巢狀 over() 會被 polars 拒絕（window expression not allowed in
    # aggregation），故分成兩步：先算「與前一列是否不同」，再各自累加。
    flagged = hourly.sort("stn_ID", "Date").with_columns(
        pl.col("AirTemperature").is_null().alias("missing")
    )
    flagged = flagged.with_columns(
        (pl.col("missing") != pl.col("missing").shift(1))
        .fill_null(True)
        .alias("boundary")
    )
    flagged = flagged.with_columns(pl.col("boundary").cum_sum().over("stn_ID").alias("run"))
    runs = (
        flagged.filter(pl.col("missing"))
        .group_by("stn_ID", "run")
        .agg(pl.col("Date").min().alias("start"), pl.len().alias("hours"))
        .filter(pl.col("hours") > settings.WEATHER_LONG_GAP_HOURS)
        .sort("stn_ID", "start")
    )
    diagnostics["long_gaps"] = runs.select("stn_ID", "start", "hours").to_dicts()

    if any(diagnostics["filled"].values()):
        logger.info("氣象插補：%s", diagnostics["filled"])
    if diagnostics["long_gaps"]:
        logger.warning(
            "連續缺測超過 %d 小時的區段共 %d 處，最長 %d 小時——"
            "這些日子的日特徵是插補出來的，請留意",
            settings.WEATHER_LONG_GAP_HOURS, len(diagnostics["long_gaps"]),
            max(g["hours"] for g in diagnostics["long_gaps"]),
        )
    return out, diagnostics


def daily_features(hourly: pl.DataFrame) -> pl.DataFrame:
    """把逐小時觀測彙整成每日、每站的 5 項特徵。

    Args:
        hourly: 已插補的小時表，含 ``Date``（Datetime）、``stn_ID``、
            ``AirTemperature``、``UVIndex``。

    Returns:
        pl.DataFrame: 每日一列，欄位為 ``date`` 與 ``w_{站名}_{項目}``。
    """
    day_lo, day_hi = settings.WEATHER_DAY_HOURS
    code_to_name = {code: name for name, code in settings.WEATHER_STATIONS.items()}

    marked = hourly.with_columns(
        pl.col("Date").dt.date().alias("date"),
        _hour_index().alias("hour"),
    ).with_columns(
        pl.col("hour").is_between(day_lo, day_hi).alias("is_day")
    )

    daily = (
        marked.group_by("date", "stn_ID")
        .agg(
            pl.col("AirTemperature").filter(pl.col("is_day")).max().alias("day_tmax"),
            pl.col("AirTemperature").filter(pl.col("is_day")).min().alias("day_tmin"),
            pl.col("AirTemperature").filter(~pl.col("is_day")).max().alias("night_tmax"),
            pl.col("AirTemperature").filter(~pl.col("is_day")).min().alias("night_tmin"),
            pl.col("UVIndex").max().alias("uv_max"),
        )
        .sort("date", "stn_ID")
    )

    wide = daily.with_columns(
        pl.col("stn_ID").cast(pl.Utf8).replace_strict(
            code_to_name, default="未知"
        ).alias("station")
    ).pivot(on="station", index="date", values=list(METRICS))

    # polars 的 pivot 欄名格式隨版本而異，統一改成 w_{站名}_{項目}。
    renames = {}
    for column in wide.columns:
        if column == "date":
            continue
        for metric in METRICS:
            if column.startswith(f"{metric}_"):
                renames[column] = f"w_{column[len(metric) + 1:]}_{metric}"
                break
    wide = wide.rename(renames).sort("date")
    # 只留 settings.WEATHER_METRICS 指定的項目。
    kept = tuple(f"_{m}" for m in settings.WEATHER_METRICS)
    wide = wide.select("date", *[c for c in wide.columns if c.endswith(kept)])

    logger.info(
        "氣象日特徵：%d 天 × %d 欄（%d 站 × %d 項）",
        wide.height, wide.width - 1, len(settings.WEATHER_STATIONS), len(METRICS),
    )
    return wide


def add_weather_features(
    daily: pl.DataFrame,
    weather: pl.DataFrame | None = None,
    pending_dates: tuple[dt.date, ...] = (),
) -> pl.DataFrame:
    """把氣象日特徵併進每日表。

    **氣象是目標日當天的量**，與其他特徵不同——它之所以合法，是因為
    推論時由**預報**提供。訓練用觀測、推論用預報會有分布落差，
    故回測一律讓目標日改用逐起點校正的預報（honest 模式）來量這個落差。

    Args:
        daily: 每日表，須含 ``date``。
        weather: 氣象日特徵；None 時由檔案載入並計算。
        pending_dates: 允許暫缺氣象的**目標日**。預測時目標日沒有觀測，其氣象由
            逐起點校正的預報覆寫（``pipeline._override_weather``），此處先留空。
            只有明確列出的日期可以暫缺，其餘缺值照樣中止。

    Returns:
        pl.DataFrame: 併入 ``w_*`` 氣象欄後的每日表。

    Raises:
        ValueError: 有日期缺氣象值——**不靜默補值**，因為那會讓模型
            用假資料訓練而我們不會知道。
    """
    if weather is None:
        from src.data import external

        weather = external.load_weather()

    out = daily.join(weather, on="date", how="left")
    feature_columns = [c for c in weather.columns if c != "date"]
    missing = out.filter(
        pl.col(feature_columns[0]).is_null() & ~pl.col("date").is_in(list(pending_dates))
    )
    if missing.height:
        raise ValueError(
            f"這些日期缺氣象資料，無法組裝特徵（共 {missing.height} 天，"
            f"前 5 筆 {missing['date'].to_list()[:5]}）。"
            "請先執行 `python main.py weather` 補齊，或關閉 ENABLE_TIER1_WEATHER。"
        )
    return out
