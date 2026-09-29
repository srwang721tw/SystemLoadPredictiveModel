"""外部資料統一載入接口。

KISS：以函式 + 設定檔實作，**不建立繼承階層或抽象基底類別**。
路徑集中於 ``config/paths.py``，原始檔置於 ``data/raw/``、衍生檔置於 ``data/processed/``。

讀 xlsx 一律走 ``openpyxl``，不用 ``polars.read_excel``——後者需要
``fastexcel``，多一個相依卻沒有額外好處。
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import polars as pl

from config import paths, settings
from src.data import checks
from src.logging_setup import get_logger

logger = get_logger(__name__)

CALENDAR_DAYTYPE_COLUMN = "平假日"
"""日曆表中的日別欄位。值域：平日 / 週六 / 週日及離峰日。"""


def read_xlsx(path: Path, sheet: str | None = None) -> pl.DataFrame:
    """以 openpyxl 讀取 xlsx 的單一工作表。

    Args:
        path: 檔案路徑。
        sheet: 工作表名稱，None 時取第一張。

    Returns:
        pl.DataFrame: 首列為欄名的資料表。

    Raises:
        FileNotFoundError: 檔案不存在。
    """
    import openpyxl

    if not path.exists():
        raise FileNotFoundError(f"外部資料檔不存在：{path}")

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[sheet] if sheet else workbook.worksheets[0]
    rows = list(worksheet.iter_rows(values_only=True))
    workbook.close()

    header, body = rows[0], rows[1:]
    return pl.DataFrame({name: [row[i] for row in body] for i, name in enumerate(header)})


def load_calendar(path: Path | None = None) -> pl.DataFrame:
    """讀取時間電價日曆表。

    這張表提供**權威的日別分類**，比由星期推導的 ``daytype`` 更可信：
    國定假日在日曆上是平日，但電價上歸類為「週日及離峰日」，用電行為也確實
    像週日。實測 2024-02-12（春節）、06-10（端午）、09-17（中秋）、
    2024/2025-10-10（國慶）、2026-01-01（元旦）皆被正確歸為「週日及離峰日」。

    日曆表**不含颱風停班停課**——那是臨時公告，不在電價制度內。
    實測 2024-10-02/03（山陀兒）與 2025-09-29（樺加沙）在表中仍是「平日」。
    颱風仍需另外的清單。

    Args:
        path: 檔案路徑，None 時採用 ``paths.CALENDAR_FILE``。

    Returns:
        pl.DataFrame: 欄位 ``date`` (Date)、``price_daytype``、``lunar``、``solar_term``。
    """
    path = path or paths.CALENDAR_FILE
    raw = read_xlsx(path)
    checks.check_columns(
        raw, {"YYYYMMDD": pl.Datetime, CALENDAR_DAYTYPE_COLUMN: pl.Utf8,
              "農曆": pl.Utf8, "節氣": pl.Utf8}, "時間電價日曆表")
    checks.check_gaps(raw.sort("YYYYMMDD"), "YYYYMMDD", dt.timedelta(days=1), "時間電價日曆表")
    df = (
        raw.with_columns(pl.col("YYYYMMDD").dt.date().alias("date"))
        .select(
            "date",
            pl.col(CALENDAR_DAYTYPE_COLUMN).alias("price_daytype"),
            pl.col("農曆").alias("lunar"),
            pl.col("節氣").alias("solar_term"),
        )
        .sort("date")
    )
    logger.info(
        "讀取日曆表：%s → %d 列，%s ~ %s",
        path.name, df.height, df["date"].min(), df["date"].max(),
    )
    return df


def load_price_period_rules(path: Path | None = None) -> dict:
    """讀取時間電價的日內時段規則。

    規則置於 ``config/price_periods.toml``，而非程式碼中，
    電價時段不寫死在程式中。

    Args:
        path: 規則檔路徑，None 時採用 ``paths.PRICE_PERIOD_RULES_FILE``。

    Returns:
        dict: 解析後的規則，含 ``summer`` 與 ``bands`` 兩節。
    """
    path = path or paths.PRICE_PERIOD_RULES_FILE
    with path.open("rb") as handle:
        rules = tomllib.load(handle)
    logger.info("讀取電價時段規則：%s", path.name)
    return rules


def merge_on_date(base: pl.DataFrame, other: pl.DataFrame, how: str = "left") -> pl.DataFrame:
    """以 ``date`` 為鍵合併外部資料，並檢查合併後的缺失率。

    Args:
        base: 主表（每日一列）。
        other: 待合併的外部資料。
        how: 合併方式，預設左join 以保留主表所有日期。

    Returns:
        pl.DataFrame: 合併結果。
    """
    merged = base.join(other, on="date", how=how)  # type: ignore[arg-type]
    for column in (c for c in other.columns if c != "date"):
        n_null = merged[column].null_count()
        if n_null:
            logger.warning(
                "合併後 %s 有 %d / %d 列缺值（%.1f%%）",
                column, n_null, merged.height, n_null / merged.height * 100,
            )
    return merged


def check_coverage(base: pl.DataFrame, other: pl.DataFrame) -> dict:
    """檢查外部資料是否完整涵蓋主表的日期範圍。

    Args:
        base: 主表。
        other: 外部資料。

    Returns:
        dict: 覆蓋起訖與未被涵蓋的日數。
    """
    missing = base.join(other.select("date"), on="date", how="anti")
    return {
        "base_range": (base["date"].min(), base["date"].max()),
        "other_range": (other["date"].min(), other["date"].max()),
        "n_uncovered_days": missing.height,
        "uncovered_dates": missing["date"].to_list()[:10],
    }


def load_typhoon_days(path: Path | None = None) -> pl.DataFrame:
    """讀取颱風停班停課清單。

    格式：一天一列，必要欄位 ``颱風名稱`` 與 ``日期``。

    **本載入器刻意只要求那兩個必要欄位，其餘一律原樣帶出。**
    日後在 CSV 加欄位（例如侵臺路徑分類、近臺強度）時載入器不需改動，
    由特徵建構決定要用哪些。

    清單中可能有超出現有負載資料範圍的日期（例如巴威 2026/7/10–11
    在資料補齊至 2026-09-30 之前尚未進入訓練集）。這**不是錯誤**，
    合併時會自然落空，不得因此拋錯。

    Args:
        path: 檔案路徑，None 時採用 ``paths.TYPHOON_FILE``。

    Returns:
        pl.DataFrame: 欄位 ``date``（Date）與檔案中的其餘欄位。
            檔案不存在時回傳空表（欄位齊備），供對應開關關閉時使用。

    Raises:
        ValueError: 缺少必要欄位，或同一天出現多次。
    """
    path = path or paths.TYPHOON_FILE
    if not path.exists():
        logger.warning("颱風清單不存在：%s，回傳空表", paths.relative(path))
        return pl.DataFrame(schema={"date": pl.Date, "颱風名稱": pl.Utf8})

    raw = pl.read_csv(path)
    missing = [c for c in ("颱風名稱", "日期") if c not in raw.columns]
    if missing:
        raise ValueError(f"颱風清單缺少必要欄位：{missing}（現有 {raw.columns}）")

    out = raw.with_columns(
        pl.col("日期").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
    ).drop("日期")

    duplicated = out.group_by("date").len().filter(pl.col("len") > 1)
    if duplicated.height:
        raise ValueError(
            f"颱風清單有重複日期（應一天一列）：{duplicated['date'].to_list()}"
        )

    extra = [c for c in out.columns if c not in ("date", "颱風名稱")]
    logger.info(
        "讀取颱風清單：%d 天、%d 個颱風，額外欄位 %s",
        out.height, out["颱風名稱"].n_unique(), extra or "無",
    )
    return out.sort("date")


def _load_day_list(path: Path, required: tuple[str, ...], label: str) -> pl.DataFrame:
    """讀取「一天一列」的日期清單，共用給放假日與事件日。

    三張日期表（颱風、放假日、事件日）的載入邏輯相同：檢查必要欄位、
    把 ``日期`` 轉成 ``date``、確認無重複日期、其餘欄位原樣帶出。
    抽出來共用，避免三份幾乎一樣的程式各自漂移。

    **刻意只檢查必要欄位，其餘一律原樣帶出**，日後補欄位時不需要改這裡。

    Args:
        path: CSV 路徑。
        required: 必要欄位名稱。
        label: log 用的中文標籤。

    Returns:
        pl.DataFrame: 欄位 ``date``（Date）與檔案中的其餘欄位。
            檔案不存在時回傳空表，供該來源尚未到位時使用。

    Raises:
        ValueError: 缺少必要欄位，或同一天出現多次。
    """
    if not path.exists():
        logger.warning("%s清單不存在：%s，回傳空表", label, paths.relative(path))
        return pl.DataFrame(schema={"date": pl.Date})

    raw = pl.read_csv(path)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{label}清單缺少必要欄位：{missing}（現有 {raw.columns}）")

    out = raw.with_columns(
        pl.col("日期").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
    ).drop("日期")

    duplicated = out.group_by("date").len().filter(pl.col("len") > 1)
    if duplicated.height:
        raise ValueError(
            f"{label}清單有重複日期（應一天一列）：{duplicated['date'].to_list()}"
        )

    logger.info(
        "讀取%s清單：%d 天，欄位 %s",
        label, out.height, [c for c in out.columns if c != "date"],
    )
    return out.sort("date")


def load_weather(path: Path | None = None) -> pl.DataFrame:
    """回傳**每日每站**的氣象特徵，來源由 ``settings.WEATHER_SOURCE`` 決定。

    | 來源 | 內容 | 取得時點 |
    |---|---|---|
    | ``"codis"`` | CODiS 小時**觀測** | 目標日當天才存在 |
    | ``"accuweather"`` | Accuweather 逐小時**預報** | 提交日前即可取得 |

    若歷史用觀測、推論時目標日卻只有預報，回測會系統性樂觀，而且樂觀的幅度
    無法從回測本身看出來。現行設定為 ``"codis"``，目標日由逐起點校正的預報覆寫
    （honest 模式），見 ``settings.WEATHER_SOURCE``。

    兩條路徑都回傳**完全相同的 ``w_*`` 欄**，歷史與目標日才接得起來。

    Args:
        path: 來源檔路徑，None 時依來源採用各自的預設。

    Returns:
        pl.DataFrame: 每日一列，欄位 ``date`` 與 ``w_*`` 氣象欄。

    Raises:
        ValueError: ``settings.WEATHER_SOURCE`` 不是支援的值。
    """
    if settings.WEATHER_SOURCE == "codis":
        return _load_weather_codis(path)
    if settings.WEATHER_SOURCE == "accuweather":
        return _load_weather_accuweather(path)
    raise ValueError(
        f"未知的 WEATHER_SOURCE：{settings.WEATHER_SOURCE!r}"
        "（支援 'codis' 與 'accuweather'）"
    )


def _load_weather_accuweather(directory: Path | None = None) -> pl.DataFrame:
    """Accuweather 預報 → 每日每站特徵。

    刻意複用 CODiS 的那條管線（``interpolate_hourly`` → ``daily_features``），
    因為 :func:`src.data.accuweather.load_station_hourly` 輸出的 schema 與
    CODiS 小時表相同。兩份聚合規則若各寫一套，遲早會漂移。

    Args:
        directory: 年度檔目錄，None 時採用 ``paths.ACCUWEATHER_DIR``。

    Returns:
        pl.DataFrame: 每日一列，欄位 ``date`` 與 ``w_*`` 氣象欄。
    """
    from src.data import accuweather
    from src.features import weather as weather_features

    hourly = accuweather.reindex_full_hours(
        accuweather.load_station_hourly(directory)
    )
    filled, _ = weather_features.interpolate_hourly(hourly)
    return weather_features.daily_features(filled)


def _load_weather_codis(path: Path | None = None, fill: bool = True) -> pl.DataFrame:
    """讀取 CODiS 小時觀測，插補後回傳**每日每站**的氣象特徵。

    小時檔由 ``python main.py weather`` 產生，存的是原始字串；此處以
    :func:`src.data.weather.clean_observations` 依 ``settings.CODIS_*`` 對照表
    轉成數值（特殊值逐值記錄），再取模型使用的氣溫與 UV。缺值做線性插補（見 :func:`src.features.weather.interpolate_hourly`），**絕不補 0**。

    **取不到觀測的日子**（見 :func:`unavailable_observation_dates`）先從觀測中
    移除；``fill=True`` 時再以校正後的 Accuweather 預報補上，讓下游看到一張
    涵蓋到截止日的完整表。

    Args:
        path: 小時檔路徑，None 時採用 ``paths.WEATHER_FILE``。
        fill: 是否以預報補上取不到觀測的日子。預報校正本身要用 ``fill=False``——
            校正只能用真正的觀測。

    Returns:
        pl.DataFrame: 每日一列，欄位 ``date`` 與 ``w_*`` 氣象欄。

    Raises:
        FileNotFoundError: 小時檔不存在——**不回傳空表**，因為靜默的空氣象
            會讓下游以為「沒有氣象特徵」而不是「氣象還沒抓」。
    """
    from src.data import weather as weather_fetch
    from src.features import weather as weather_features

    path = path or paths.WEATHER_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"氣象觀測檔不存在：{path}。請先執行 `python main.py weather`。"
        )

    raw = pl.read_csv(path, infer_schema_length=0)
    checks.check_columns(raw, dict.fromkeys(weather_fetch.CSV_COLUMNS, pl.Utf8), "氣象觀測")
    raw = checks.deduplicate(raw, ["Date", "stn_ID"], "氣象觀測")
    cleaned, report = weather_fetch.clean_observations(raw)
    logger.info("CODiS 特殊值轉換：\n%s", report)
    hourly = cleaned.select(
        "Date", "stn_ID",
        pl.col("AirTemperature_Instantaneous").alias("AirTemperature"),
        pl.col("UVIndex_Accumulation").alias("UVIndex"),
    ).sort("stn_ID", "Date")
    checks.check_gaps(hourly, "Date", dt.timedelta(hours=settings.EXOGENOUS_MAX_GAP_HOURS),
                      "氣象觀測", group="stn_ID")
    hourly = checks.apply_cutoff(hourly, "Date", "氣象觀測")

    logger.info(
        "讀取氣象觀測：%d 列，%s ~ %s，%d 個測站",
        hourly.height, hourly["Date"].min(), hourly["Date"].max(),
        hourly["stn_ID"].n_unique(),
    )
    unavailable = unavailable_observation_dates(codis_observed_end(hourly))
    if not unavailable:
        filled, _ = weather_features.interpolate_hourly(hourly)
        return weather_features.daily_features(filled)

    # 取不到的日子前後分開插補：插補不可跨過那幾天，否則前一天的缺值會用到
    # 「還取不到」的觀測。之後的日子只在回測模擬時存在，本折不會用到。
    day = pl.col("Date").dt.date()
    parts = [
        hourly.filter(day < unavailable[0]),
        hourly.filter(day > unavailable[-1]),
    ]
    daily = pl.concat([
        weather_features.daily_features(weather_features.interpolate_hourly(part)[0])
        for part in parts if part.height
    ])
    return _fill_from_forecast(daily, unavailable) if fill else daily


def codis_observed_end(hourly: pl.DataFrame) -> dt.date:
    """CODiS 最後一個**完整**觀測日：每個測站都已有當日 23 時以後的紀錄。

    CODiS 每天中午後才更新到前一日，更新到一半的日子不算完整，
    否則會用半天的資料算出當日最高溫。

    Args:
        hourly: 含 ``Date``（Datetime）與 ``stn_ID`` 的小時觀測。

    Returns:
        dt.date: 最後一個完整日。
    """
    last = hourly.group_by("stn_ID").agg(pl.col("Date").max())["Date"].min()
    return last.date() if last.hour >= 23 else last.date() - dt.timedelta(days=1)


def unavailable_observation_dates(observed_end: dt.date) -> list[dt.date]:
    """截止日以前取不到觀測、要以預報補上的日期。

    兩個來源：

    - **實際缺**：觀測只到 ``observed_end``、截止日更晚。最多補
      ``settings.CODIS_MAX_LAG_DAYS`` 天；缺更多時不補，由前置檢查中止
    - **回測模擬**：``settings.CODIS_UNAVAILABLE_DATES``

    Args:
        observed_end: 最後一個完整觀測日（:func:`codis_observed_end`）。

    Returns:
        list[dt.date]: 依日期排序。
    """
    gap = (checks.cutoff_date() - observed_end).days
    missing = [observed_end + dt.timedelta(days=k) for k in range(1, gap + 1)]
    if len(missing) > settings.CODIS_MAX_LAG_DAYS:
        logger.error("CODiS 觀測只到 %s，缺 %d 天，超過容許的 %d 天，不以預報補",
                     observed_end, len(missing), settings.CODIS_MAX_LAG_DAYS)
        missing = []
    return sorted(set(missing) | set(settings.CODIS_UNAVAILABLE_DATES))


def _fill_from_forecast(observed: pl.DataFrame, dates: list[dt.date]) -> pl.DataFrame:
    """以逐站校正後的 Accuweather 預報補上 ``dates`` 的氣象。

    校正比照 honest 模式的目標日：只用 ``dates`` 以前實際有觀測的日期擬合。

    Args:
        observed: CODiS 每日特徵（已移除 ``dates``）。
        dates: 要補的日期，依日期排序。

    Returns:
        pl.DataFrame: 欄位與 ``observed`` 相同，依日期排序。

    Raises:
        ValueError: Accuweather 缺少其中某天——不可靜默留空。
    """
    from src.features import accuweather as accuweather_features

    forecast = accuweather_features.build_forecast()
    before = pl.col("date") < dates[0]
    coefficients = accuweather_features.fit_bias_correction(
        forecast.filter(before), observed.filter(before))
    rows = accuweather_features.apply_bias_correction(
        forecast.filter(pl.col("date").is_in(dates)), coefficients)
    missing = sorted(set(dates) - set(rows["date"].to_list()))
    if missing:
        raise ValueError(f"CODiS 取不到 {missing} 的觀測，Accuweather 也沒有這幾天的預報")
    logger.warning("CODiS 取不到 %s 的觀測，已以校正後的 Accuweather 預報補上", dates)
    return pl.concat([observed, rows.select(observed.columns)]).sort("date")

def load_special_days(path: Path | None = None) -> pl.DataFrame:
    """讀取特殊日期的每日 0/1 表。

    由 ``python main.py special-days`` 產生，內容完全由
    ``config/特殊日期區間.csv`` 決定，可隨時重建。

    **這是「行政機關辦公日曆表」的標準，與 ``price_daytype``
    （時間電價日曆）是兩個相似但不相同的標準**——前者決定大家上不上班、
    後者決定計價日別。兩份不一致（實測 20 天）是正常的，不需要校正，
    也**不會**修改 ``price_daytype``。

    Args:
        path: 檔案路徑，None 時採用 ``paths.SPECIAL_DAYS_FILE``。

    Returns:
        pl.DataFrame: 欄位 ``date`` 與五個 0/1 欄。

    Raises:
        FileNotFoundError: 檔案不存在——不回傳空表，避免下游誤以為
            「這些日子都不特殊」而不是「還沒產生」。
        ValueError: 缺少任何一個類別欄。
    """
    from src.data.special_days import CATEGORIES

    path = path or paths.SPECIAL_DAYS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"特殊日期檔不存在：{path}。請先執行 `python main.py special-days`。"
        )

    out = pl.read_csv(path, try_parse_dates=True)
    missing = [c for c in ("date", *CATEGORIES) if c not in out.columns]
    if missing:
        raise ValueError(f"特殊日期檔缺少欄位：{missing}（現有 {out.columns}）")

    logger.info(
        "讀取特殊日期：%d 天，%s ~ %s，各類別天數 %s",
        out.height, out["date"].min(), out["date"].max(),
        {c: int(out[c].sum()) for c in CATEGORIES},
    )
    return out.sort("date")


def load_event_days(path: Path | None = None) -> pl.DataFrame:
    """讀取基礎設施事件日（地震等）。

    與颱風分開記錄，因為機制不同：颱風是**停班停課**（用電行為改變、可預期），
    事件是**供電或生產中斷**（負載被動被壓掉、不可預期）。
    處置方式也因此不同：事件日不進量值訓練，颱風日則保留並以旗標標記。

    Args:
        path: 檔案路徑，None 時採用 ``paths.EVENT_DAY_FILE``。

    Returns:
        pl.DataFrame: 欄位 ``date`` 與 ``事件``、``類型``、``影響``、``來源``。
    """
    return _load_day_list(path or paths.EVENT_DAY_FILE, ("日期", "事件"), "事件日")
