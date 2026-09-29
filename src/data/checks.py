"""讀檔後的資料檢查：截止日過濾、欄位與型別、時間連續性、去除重複、完整性、預測前的涵蓋檢查。"""

from __future__ import annotations

import datetime as dt

import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def cutoff_date() -> dt.date:
    """資料截止日（含），即 ``settings.DATA_AVAILABLE_END``。"""
    return dt.date.fromisoformat(settings.DATA_AVAILABLE_END)


def apply_cutoff(frame: pl.DataFrame, column: str, label: str) -> pl.DataFrame:
    """只保留時間欄的**日期**不晚於資料截止日的列。

    開發期資料不得晚於 config 設定的截止日，而且由程式過濾，
    不依賴檔案內容本身只到哪一天。以日期比較，所以 CODiS 當日 23:59
    那一筆、負載當日 23:50 那一筆都保留。

    Args:
        frame: 要過濾的表。
        column: 時間欄（``Date`` 或 ``Datetime``）。
        label: 記錄用的資料名稱。

    Returns:
        pl.DataFrame: 過濾後的表。被濾掉的列數會寫入 log。
    """
    day = pl.col(column)
    if frame.schema[column] != pl.Date:
        day = day.dt.date()
    kept = frame.filter(day <= cutoff_date())
    dropped = frame.height - kept.height
    if dropped:
        logger.info("%s：截止日 %s 之後的 %d 列已濾除", label, cutoff_date(), dropped)
    return kept


def check_columns(frame: pl.DataFrame, required: dict[str, type[pl.DataType]], label: str) -> None:
    """必要欄位與型別檢查，不符即中止並列出差異。

    Args:
        frame: 讀入後的表。
        required: ``{欄位: polars 型別}``。
        label: 資料名稱（寫進錯誤訊息）。

    Raises:
        ValueError: 缺欄位或型別不符。
    """
    problems = [f"缺少欄位 {c}" for c in required if c not in frame.columns]
    problems += [
        f"{c} 應為 {dtype}，實為 {frame.schema[c]}"
        for c, dtype in required.items()
        if c in frame.columns and frame.schema[c] != dtype
    ]
    if problems:
        raise ValueError(f"{label} 結構不符：" + "；".join(problems)
                         + f"（現有欄位 {frame.columns}）")


def check_step(frame: pl.DataFrame, column: str, minutes: int, label: str) -> None:
    """時間戳必須落在 ``minutes`` 分鐘的格點上。

    Raises:
        ValueError: 有時間戳不在格點上（列出前 5 筆）。
    """
    stamps = frame[column]
    off = frame.filter(
        (stamps.dt.minute().cast(pl.Int32) % minutes != 0) | (stamps.dt.second() != 0)
    )
    if off.height:
        raise ValueError(
            f"{label} 有 {off.height} 筆時間不在 {minutes} 分鐘格點上，例如 "
            f"{off[column].head(5).to_list()}"
        )


def check_gaps(
    frame: pl.DataFrame, column: str, max_gap: dt.timedelta, label: str,
    group: str | None = None,
) -> None:
    """同一組內相鄰時間戳的間隔不得超過 ``max_gap``（時間連續性）。

    Raises:
        ValueError: 有超過容許值的缺口（列出最大的 5 個）。
    """
    order = ([group] if group else []) + [column]
    gap = pl.col(column).diff()
    gaps = (
        frame.sort(order)
        .with_columns((gap.over(group) if group else gap).alias("_gap"),
                      pl.col(column).shift().over(group).alias("_from") if group
                      else pl.col(column).shift().alias("_from"))
        .filter(pl.col("_gap") > max_gap)
        .sort("_gap", descending=True)
    )
    if gaps.height:
        examples = [
            f"{row.get(group, '')} {row['_from']} → {row[column]}（{row['_gap']}）".strip()
            for row in gaps.head(5).iter_rows(named=True)
        ]
        raise ValueError(
            f"{label} 時間不連續：{gaps.height} 處間隔超過 {max_gap}，例如 {examples}"
        )


def deduplicate(frame: pl.DataFrame, key: list[str], label: str) -> pl.DataFrame:
    """完全重複的列刪除並記錄；鍵相同但數值不同則中止並列出範例。

    不自行挑選要保留哪一筆。預報資料的鍵含目標時刻與地點，
    同一目標時刻不同發布時間是不同版本——本專案的預報檔沒有發布時間欄，
    鍵只到「地點 + 目標時刻」。

    Args:
        frame: 讀入後的表。
        key: 唯一鍵。
        label: 資料名稱。

    Returns:
        pl.DataFrame: 刪除完全重複列後的表（原始順序）。

    Raises:
        ValueError: 有鍵相同但數值不同的列。
    """
    unique = frame.unique(maintain_order=True)
    removed = frame.height - unique.height
    if removed:
        logger.warning("%s：完全重複 %d 列，已刪除", label, removed)
    conflicts = unique.filter(pl.len().over(key) > 1)
    if conflicts.height:
        raise ValueError(
            f"{label} 有 {conflicts.height} 列鍵 {key} 相同但數值不同，不自行挑選，"
            f"請人工確認。範例：\n{conflicts.sort(key).head(6)}"
        )
    return unique


def precheck(origin: dt.date, days: list[dt.date]) -> dict:
    """預測前的涵蓋檢查：資料不足以做這次預測時中止，並把所有問題一次列出。

    - 負載必須涵蓋到 ``origin`` 23:50（預測起點前一刻）
    - CODiS 觀測至少涵蓋到 ``origin`` 前 ``settings.CODIS_MAX_LAG_DAYS`` 天。
      CODiS 中午後才更新到前一日，缺的那幾天由校正後的預報補
    - 以預報補觀測的那一天，Accuweather 五站 00:00–23:00 每小時都要有氣溫預報
    - 目標日的預報**不足不中止**：逐日逐站列出，由 ``workflow.predict_days`` 走備援（Plan B）；
      開啟 ``ENABLE_WINDY`` 時也逐日列出 Windy 不齊的日子


    Args:
        origin: 預測起點日（資料截止日）。
        days: 目標日。

    Returns:
        dict: ``codis_observed_end``（最後一個完整觀測日）、``codis_filled``（以預報補上的
            日期）、``forecast_missing``（``{目標日: 缺預報的站名}``）與 ``windy_missing``
            （Windy 不齊的目標日），供提交摘要呈現。

    Raises:
        ValueError: 任一項涵蓋不足。
    """
    from src.data import external, loader
    from config import paths

    problems: list[str] = []
    load = loader.load_raw_load()
    last = load["ts"].max()
    needed = dt.datetime.combine(origin, dt.time(23, 50))
    if last < needed:
        problems.append(f"負載只到 {last}，需涵蓋到 {needed}")

    observed = apply_cutoff(
        pl.read_csv(paths.WEATHER_FILE, columns=["Date", "stn_ID"], infer_schema_length=0)
        .with_columns(pl.col("Date").str.to_datetime()),
        "Date", "氣象觀測",
    )
    observed_end = external.codis_observed_end(observed)
    earliest = origin - dt.timedelta(days=settings.CODIS_MAX_LAG_DAYS)
    filled = [observed_end + dt.timedelta(days=k)
              for k in range(1, (origin - observed_end).days + 1)]
    if observed_end < earliest:
        problems.append(f"CODiS 觀測只到 {observed_end}，至少需涵蓋到 {earliest}"
                        "（請執行 python main.py weather）")
        filled = []
    elif filled:
        logger.warning("CODiS 觀測只到 %s，%s 的氣象將以校正後的預報補上",
                       observed_end, filled)

    # Plan B：目標日缺預報不再中止，逐日逐站列出，交給 predict_days。
    # 以 CODiS 補觀測的那一天仍必須有預報（它是訓練列，沒有備援）。
    missing = forecast_missing(sorted(set(days) | set(filled)))
    for day in filled:
        if missing.get(day):
            problems.append(f"CODiS 缺 {day} 的觀測，而 Accuweather 在 {day} 缺 {missing[day]}")
    forecast_gaps = {d: missing[d] for d in days if missing.get(d)}
    if forecast_gaps:
        logger.warning("目標日預報不完整，將走 Plan B：%s", forecast_gaps)
    windy_gaps = windy_missing(days) if settings.ENABLE_WINDY else []
    if windy_gaps:
        logger.warning("目標日 Windy 不完整，這幾天將改用不含 Windy 的模型：%s", windy_gaps)

    if problems:
        raise ValueError(
            "前置檢查未通過，無法預測 " + f"{days[0]} ~ {days[-1]}：\n- "
            + "\n- ".join(problems)
        )
    logger.info("前置檢查通過：負載至 %s、CODiS 至 %s", last, observed_end)
    return {"codis_observed_end": observed_end, "codis_filled": filled,
            "forecast_missing": forecast_gaps, "windy_missing": windy_gaps}


def windy_missing(days: list[dt.date]) -> list[dt.date]:
    """Windy 特徵（``pv_14``、``pv_sum``）不齊的目標日：任一個缺值就算缺，那幾天改用不含 Windy 的模型。"""
    from src.data import windy
    from src.features import windy as windy_features

    daily = windy_features.daily_features(windy.load_hourly())
    complete = set(daily.drop_nulls(list(windy_features.FEATURES))["date"].to_list())
    return [d for d in days if d not in complete]


def forecast_missing(days: list[dt.date]) -> dict[dt.date, tuple[str, ...]]:
    """每一天缺 Accuweather 預報的站名：該站當天 00–23 時有任何一小時沒有氣溫就算缺。

    半天的預報會算錯當日最高溫，所以不足 24 小時不拿來用，改走 Plan B。

    Returns:
        dict: ``{日期: (站名, …)}``；完整的日子不列出。
    """
    from src.data import accuweather

    name_of = {code: name for name, code in settings.WEATHER_STATIONS.items()}
    counts = (
        accuweather.load_station_hourly()
        .filter(pl.col("Date").dt.date().is_in(days) & pl.col("AirTemperature").is_not_null())
        .group_by(pl.col("Date").dt.date().alias("day"), "stn_ID").len()
    )
    have = {(r["day"], r["stn_ID"]): r["len"] for r in counts.iter_rows(named=True)}
    out = {}
    for day in days:
        gone = tuple(name for code, name in name_of.items() if have.get((day, code), 0) < 24)
        if gone:
            out[day] = gone
    return out


def check_completeness(df: pl.DataFrame) -> pl.DataFrame:
    """檢查每日是否確為 144 筆，並統計缺失筆數（只回報，不修改資料）。

    Args:
        df: 已對齊完整格點的序列，含 ``is_missing``。

    Returns:
        pl.DataFrame: 只列出筆數不符或含缺失值的日子（日期、實際筆數、缺失筆數）；
            全部正常時為空表。
    """
    daily = (
        df.with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("n_points"), pl.col("is_missing").sum().alias("n_missing"))
        .sort("date")
    )
    bad = daily.filter(
        (pl.col("n_points") != settings.POINTS_PER_DAY) | (pl.col("n_missing") > 0)
    )
    logger.info("完整性檢查：%d 天，其中 %d 天筆數不符或含缺失值", daily.height, bad.height)
    return bad
