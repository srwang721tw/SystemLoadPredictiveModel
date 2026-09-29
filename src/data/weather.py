"""中央氣象署 CODiS 觀測資料爬取與解析。

資料來源：https://codis.cwa.gov.tw/api/station?（POST，form-urlencoded）
回應結構（實際探測所得，非猜測）：

    {"code":…, "message":…, "metadata":…,
     "data": [{"StationID": "466920",
               "dts": [{"DataTime": "2026-08-12T01:00:00",
                        "AirTemperature": {"Instantaneous": 27.9, …},
                        "UVIndex": {"Accumulation": 0, …}, …}, … 24 筆 …]}]}

**``DataTime`` 是區間終點。** 一天 24 筆從 ``T01:00:00`` 到 ``T23:00:00``，
最後一筆是 ``T23:59:00``。也就是第一筆涵蓋 00:00–01:00，
故「白天 06:00–18:00」對應的是 **hour 7 到 hour 18**（見 :mod:`src.features.weather`）。

**寫檔時保留原始字串，讀檔時才轉數值。** CODiS 以數值編碼特殊狀態
（缺測 −99.5、−9999.5 等；降水 −9.8 為雨跡），``…f`` 旗標欄全期為空。
轉換規則集中在 ``settings.CODIS_*``，由 :func:`clean_observations` 套用並逐值
記錄筆數，特殊值不會一律轉缺值而不留紀錄。
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from config import paths, settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

API_URL = "https://codis.cwa.gov.tw/api/station?"

_write_lock = threading.Lock()


VARIABLES: dict[str, tuple[str, ...]] = {
    "StationPressure": ("Instantaneous",),
    "SeaLevelPressure": ("Instantaneous",),
    "AirTemperature": ("Instantaneous",),
    "DewPointTemperature": ("Instantaneous",),
    "RelativeHumidity": ("Instantaneous",),
    "WindSpeed": ("Mean", "TenMinutelyMaximum"),
    "WindDirection": ("Mean", "TenMinutelyMaximum"),
    "PeakGust": ("Maximum", "Direction"),
    "Precipitation": ("Accumulation",),
    "PrecipitationDuration": ("Total",),
    "SunshineDuration": ("Total",),
    "GlobalSolarRadiation": ("Accumulation",),
    "Visibility": ("Instantaneous", "AutoMean"),
    "UVIndex": ("Accumulation",),
    "TotalCloudAmount": ("Instantaneous", "SatRetrieved"),
    **{f"SoilTemperatureAt{depth}cm": ("Instantaneous",) for depth in (0, 5, 10, 20, 30, 50, 100)},
}
"""CODiS 每小時回傳的全部變數與子欄（實測臺北站 2025-10-02 的回應）。

每個子欄另有一個同名加 ``f`` 的旗標欄（例如 ``Instantaneousf``）；兩者都保留。
"""

CSV_COLUMNS: tuple[str, ...] = ("Date", "stn_ID") + tuple(
    f"{variable}_{field}{suffix}"
    for variable, fields in VARIABLES.items()
    for field in fields
    for suffix in ("", "f")
)
"""輸出欄位：``{變數}_{子欄}`` 與其旗標 ``{變數}_{子欄}f``，全部為原始字串。"""

_unknown_fields: set[str] = set()


def _raw(value: object) -> str | None:
    """保留原始值的字串形式；None 仍為 None。不做任何數值轉換或哨兵判定。"""
    return None if value is None else str(value)


def parse_response(payload: dict) -> list[dict]:
    """把單站單日的 API 回應解析成逐小時的列，保留**全部變數與旗標的原始字串**。

    不挑欄位、不轉數值、不清哨兵：特殊值必須原樣留下，才能在讀檔時逐值
    轉換並記錄（見 :func:`clean_observations`）。

    回應中出現 :data:`VARIABLES` 以外的子欄時，記入 ``_unknown_fields``
    並在 :func:`fetch_range` 結束時警告，不會靜默丟棄而不告知。

    Args:
        payload: ``json.loads`` 後的回應。

    Returns:
        list[dict]: 每小時一列，欄位同 :data:`CSV_COLUMNS`。回應中沒有資料時
            回傳空 list（**不是錯誤**——舊測站可能停測）。
    """
    rows: list[dict] = []
    for block in payload.get("data") or []:
        station = block.get("StationID")
        for record in block.get("dts") or []:
            timestamp = record.get("DataTime")
            if not timestamp:
                continue
            row: dict = dict.fromkeys(CSV_COLUMNS)
            row["Date"], row["stn_ID"] = timestamp, station
            for variable, content in record.items():
                if not isinstance(content, dict):
                    continue
                for field, value in content.items():
                    column = f"{variable}_{field}"
                    if column in row:
                        row[column] = _raw(value)
                    else:
                        _unknown_fields.add(column)
            rows.append(row)
    return rows


def fetch_one(day: dt.date, station: str, timeout: int = 30) -> list[dict]:
    """抓取單站單日，失敗時指數退避重試。

    Args:
        day: 目標日期。
        station: 測站代號。
        timeout: 單次請求逾時秒數。

    Returns:
        list[dict]: 逐小時的列；重試耗盡仍失敗時回傳空list並記錄警告。
    """
    form = urllib.parse.urlencode({
        "date": f"{day:%Y-%m-%d}T00:00:00+08:00",
        "type": "report_date",
        "stn_ID": station,
        "stn_type": "cwb",
        "more": "",
        "start": f"{day:%Y-%m-%d}T00:00:00",
        "end": f"{day:%Y-%m-%d}T23:59:59",
        "item": "",
    }).encode()

    for attempt in range(settings.WEATHER_FETCH_RETRIES):
        request = urllib.request.Request(
            API_URL, data=form,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "User-Agent": settings.WEATHER_FETCH_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return parse_response(json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as error:
            if error.code < 500:
                # 4xx 重試沒有意義，直接放棄並記錄。
                logger.warning("%s %s：HTTP %d，跳過", day, station, error.code)
                return []
            wait = 2 ** attempt
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            wait = 2 ** attempt
            if attempt == settings.WEATHER_FETCH_RETRIES - 1:
                logger.warning("%s %s：重試耗盡（%s）", day, station, error)
                return []
        time.sleep(wait)
    return []


def fetch_range(
    start: dt.date,
    end: dt.date,
    stations: dict[str, str] | None = None,
    path: Path | None = None,
) -> pl.DataFrame:
    """抓取日期區間 × 所有測站，可續跑。

    **可續跑**：已存在於輸出檔的 ``(Date 的日期, stn_ID)`` 一律跳過。
    4560 次請求中途失敗是常態，重跑時不該從頭來。

    併發寫入以 ``_write_lock`` 保護。少了它，多執行緒會把 CSV 寫成交錯的亂碼。

    Args:
        start: 起始日（含）。
        end: 結束日（含）。
        stations: ``{名稱: 代號}``，None 時採用 ``settings.WEATHER_STATIONS``。
        path: 輸出 CSV，None 時採用 ``paths.WEATHER_FILE``。全部欄位以原始字串寫出。

    Returns:
        pl.DataFrame: 本次執行後檔案中的完整內容。
    """
    stations = stations or settings.WEATHER_STATIONS
    path = path or paths.WEATHER_FILE
    schema = dict.fromkeys(CSV_COLUMNS, pl.Utf8)
    path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    if path.exists():
        existing = pl.read_csv(path, infer_schema_length=0)
        if existing.height:
            done = {
                (row["Date"][:10], str(row["stn_ID"]))
                for row in existing.select("Date", "stn_ID").iter_rows(named=True)
            }
        logger.info("續跑：已有 %d 列、%d 個 (日期, 測站) 組合", existing.height, len(done))

    n_days = (end - start).days + 1
    jobs = [
        (start + dt.timedelta(days=k), name, code)
        for k in range(n_days)
        for name, code in stations.items()
        if (f"{start + dt.timedelta(days=k):%Y-%m-%d}", code) not in done
    ]
    if not jobs:
        logger.info("沒有待抓取的組合，直接讀回既有檔案")
        return pl.read_csv(path, infer_schema_length=0)

    logger.info(
        "待抓取 %d 個 (日期, 測站) 組合（%s ~ %s，%d 站），併發 %d",
        len(jobs), start, end, len(stations), settings.WEATHER_FETCH_WORKERS,
    )

    header_needed = not path.exists() or path.stat().st_size == 0
    n_ok = n_empty = 0

    def worker(job: tuple[dt.date, str, str]) -> tuple[str, int]:
        day, name, code = job
        rows = fetch_one(day, code)
        if rows:
            frame = pl.DataFrame(rows, schema=schema)
            with _write_lock:
                nonlocal header_needed
                with path.open("a", encoding="utf-8") as handle:
                    frame.write_csv(handle, include_header=header_needed)
                header_needed = False
        return name, len(rows)

    with ThreadPoolExecutor(max_workers=settings.WEATHER_FETCH_WORKERS) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            _, count = future.result()
            n_ok += count > 0
            n_empty += count == 0
            if index % 500 == 0:
                logger.info("進度 %d / %d（成功 %d、空 %d）", index, len(jobs), n_ok, n_empty)

    out = pl.read_csv(path, infer_schema_length=0)
    logger.info("抓取完成：成功 %d、空回應 %d，檔案共 %d 列", n_ok, n_empty, out.height)
    if _unknown_fields:
        logger.warning("回應中有未登記的子欄，未寫入檔案：%s", sorted(_unknown_fields))
    return out


def clean_observations(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """把原始字串轉成數值，依 ``settings.CODIS_*`` 對照表處理特殊值並逐值記錄。

    | 原始值 | 處理 |
    |---|---|
    | ``CODIS_MISSING_CODES`` 內的碼 | 缺值 |
    | 雨跡 ``CODIS_TRACE_CODE``（僅 ``CODIS_TRACE_COLUMNS``） | 換成 ``CODIS_TRACE_VALUE``，並設 ``{欄位}_trace`` = 1 |
    | ≤ ``CODIS_UNLISTED_THRESHOLD`` 但未登記 | 缺值，**並警告**（請補進對照表） |
    | 無法轉成數值的字串 | 缺值，**並警告** |
    | ``…f`` 旗標欄的非空值 | 記入報告（目前全期為空），旗標欄不帶出 |

    Args:
        raw: :func:`fetch_range` 寫出的原始字串表。

    Returns:
        tuple: ``(cleaned, report)``。``cleaned`` 含 ``Date``（Datetime）、``stn_ID``、
            各變數（Float64）與雨跡旗標；``report`` 每列一種轉換：
            ``欄位``、``原始值``、``處理``、``筆數``。
    """
    value_columns = [c for c in CSV_COLUMNS[2:] if not c.endswith("f") and c in raw.columns]
    flag_columns = [c for c in CSV_COLUMNS[2:] if c.endswith("f") and c in raw.columns]
    missing = [float(v) for v in settings.CODIS_MISSING_CODES]
    records: list[dict] = []

    def note(column: str, value: str, action: str, count: int) -> None:
        if count:
            records.append({"欄位": column, "原始值": value, "處理": action, "筆數": count})

    expressions = []
    for column in value_columns:
        text = pl.col(column)
        number = text.cast(pl.Float64, strict=False)
        values = raw[column].cast(pl.Float64, strict=False)
        not_numeric = raw.filter(text.is_not_null() & number.is_null())[column]
        for bad, count in not_numeric.value_counts().iter_rows():
            note(column, bad, "無法轉數值 → 缺值", count)
        for code in missing:
            note(column, str(code), "缺測碼 → 缺值", int((values == code).sum()))
        unlisted = values.filter((values <= settings.CODIS_UNLISTED_THRESHOLD) & ~values.is_in(missing))
        for code, count in unlisted.value_counts().iter_rows():
            note(column, str(code), "未登記的缺測碼 → 缺值", count)

        is_missing = number.is_in(missing) | (number <= settings.CODIS_UNLISTED_THRESHOLD)
        if column in settings.CODIS_TRACE_COLUMNS:
            is_trace = number == settings.CODIS_TRACE_CODE
            note(column, str(settings.CODIS_TRACE_CODE),
                 f"雨跡 → {settings.CODIS_TRACE_VALUE}", int((values == settings.CODIS_TRACE_CODE).sum()))
            expressions.append(is_trace.cast(pl.Int8).fill_null(0).alias(f"{column}_trace"))
            number = pl.when(is_trace).then(settings.CODIS_TRACE_VALUE).otherwise(number)
        expressions.append(pl.when(is_missing).then(None).otherwise(number).alias(column))

    for column in flag_columns:
        for flag, count in raw[column].drop_nulls().value_counts().iter_rows():
            note(column, flag, "旗標（僅記錄）", count)

    cleaned = raw.select(
        pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"),
        pl.col("stn_ID").cast(pl.Utf8),
        *expressions,
    )
    report = pl.DataFrame(records, schema={"欄位": pl.Utf8, "原始值": pl.Utf8,
                                           "處理": pl.Utf8, "筆數": pl.Int64})
    for row in report.filter(pl.col("處理").str.contains("未登記|無法轉數值")).iter_rows(named=True):
        logger.warning("CODiS %s 出現 %s（%d 筆）：%s", row["欄位"], row["原始值"],
                       row["筆數"], row["處理"])
    return cleaned, report

