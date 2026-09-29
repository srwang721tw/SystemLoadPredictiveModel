"""特殊日期分類：把區間表展開成每日 0/1 旗標。

五個類別依公開資訊人工整理：連假、寒暑假、大型考試、大型運動賽事、購物節。
區間定義存於 ``config/特殊日期區間.csv``（日期不寫死在程式中），
展開後的每日表寫到 ``data/processed/``。

**五個旗標在提交日 2026-10-01~03 全部為 0**（實測）。
這決定了它們唯一可能的作用機制：**不是**把提交日分到一個特殊的組，
而是把訓練池裡的異常日分出去，讓提交日所屬那組的經驗分布變乾淨。
評估變體時要用這個角度看，不能期待它在提交日「觸發」。
:class:`tests.test_special_days.TestSubmissionDates` 把這件事釘住——
日後若有人改動區間而讓提交日的分組無聲改變，測試會擋下來。

**這是「行政機關辦公日曆表」的標準，不是「時間電價日曆表」的標準。**
前者決定大家上不上班，後者決定計價日別，兩者相似但不相同
（實測有 20 天不一致，其中 5 天是日曆上的平日）。
故本模組**不修改** ``price_daytype``——那會把兩個標準混為一談。

已知的兩個缺口，照原始清單收錄、**不自行補**（來源的一致性優先於完整性）：

1. 2026 年的 ``is_holiday`` 沒有國慶（10/10）與光復節（10/25），2024／2025 都有
2. 整份清單沒有**補班日**（辦公日曆表的另一半），例如 2024-02-17
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from config import paths, settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

CATEGORIES: tuple[str, ...] = (
    "is_holiday",
    "is_vacation",
    "is_exam",
    "is_sports_event",
    "is_shopping_fest",
)
"""五個類別，順序即輸出 CSV 的欄位順序。"""

INTERVAL_COLUMNS: tuple[str, ...] = ("類別", "起", "迄", "說明")


def load_intervals(path: Path | None = None) -> pl.DataFrame:
    """讀取區間定義表。

    Args:
        path: CSV 路徑，None 時採用 ``paths.SPECIAL_DAY_INTERVAL_FILE``。

    Returns:
        pl.DataFrame: 欄位 ``類別``、``起``（Date）、``迄``（Date）、``說明``。

    Raises:
        FileNotFoundError: 檔案不存在。
        ValueError: 缺欄位、類別不在 :data:`CATEGORIES`、或起 > 迄。
    """
    path = path or paths.SPECIAL_DAY_INTERVAL_FILE
    if not path.exists():
        raise FileNotFoundError(f"特殊日期區間表不存在：{path}")

    raw = pl.read_csv(path)
    missing = [c for c in INTERVAL_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"特殊日期區間表缺少欄位：{missing}（現有 {raw.columns}）")

    out = raw.with_columns(
        pl.col("起").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("迄").str.strptime(pl.Date, "%Y-%m-%d"),
    )

    unknown = sorted(set(out["類別"].to_list()) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"特殊日期區間表出現未知類別：{unknown}（可用 {CATEGORIES}）")

    reversed_rows = out.filter(pl.col("起") > pl.col("迄"))
    if reversed_rows.height:
        raise ValueError(
            f"特殊日期區間表有起日晚於迄日的列：{reversed_rows.to_dicts()}"
        )

    logger.info("讀取特殊日期區間表：%d 個區間，%d 個類別",
                out.height, out["類別"].n_unique())
    return out


def expand(intervals: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    """把區間展開成 ``start`` ~ ``end`` 的每日 0/1 表。

    **區間含頭尾兩端**（``起 <= date <= 迄``）。
    同一類別的區間若重疊，該日仍然只是 1——旗標是「有沒有」，不是次數。

    Args:
        intervals: :func:`load_intervals` 的輸出。
        start: 起始日（含）。
        end: 結束日（含）。

    Returns:
        pl.DataFrame: 欄位 ``date`` 加上 :data:`CATEGORIES` 五欄（Int8）。
    """
    days = pl.date_range(start, end, "1d", eager=True).alias("date")
    out = pl.DataFrame({"date": days})

    for category in CATEGORIES:
        rows = intervals.filter(pl.col("類別") == category)
        if rows.height == 0:
            logger.warning("類別 %s 沒有任何區間，整欄為 0", category)
            flag = pl.lit(False)
        else:
            # 逐區間做 or。區間數只有數十個，直接展開比 join 清楚（KISS）。
            flag = pl.lit(False)
            for first, last in zip(rows["起"].to_list(), rows["迄"].to_list(),
                                   strict=True):
                flag = flag | (pl.col("date").is_between(first, last))
        out = out.with_columns(flag.cast(pl.Int8).alias(category))

    logger.info(
        "特殊日期展開 %d 天（%s ~ %s）：%s",
        out.height, start, end,
        {c: int(out[c].sum()) for c in CATEGORIES},
    )
    return out


def add_holiday_run_features(daily: pl.DataFrame, intervals: pl.DataFrame | None = None
                             ) -> pl.DataFrame:
    """加上「連假第幾天 / 共幾天」——依**行政機關辦公日曆**的連假區間。

    既有的 ``holiday_run_length`` / ``holiday_day_index``（`calendar.py`）
    是由**電價日曆**的離峰日連段推的，與辦公日曆的連假不是同一個標準
    （實測 20 天不一致）。故本函式**新增而非取代**，兩者都留給模型選。

    假說：連假的頭、中、尾作息不同（出遊、返鄉、收假）。

    Args:
        daily: 含 ``date`` 的每日表。
        intervals: 區間表，None 時讀檔。

    Returns:
        pl.DataFrame: 加上 ``holiday_run_len_admin``（該連假共幾天，非連假為 0）
            與 ``holiday_day_index_admin``（第幾天，1 起算，非連假為 0）。
    """
    if intervals is None:
        intervals = load_intervals()
    runs = intervals.filter(pl.col("類別") == "is_holiday")

    length: dict = {}
    index: dict = {}
    for first, last in zip(runs["起"].to_list(), runs["迄"].to_list(), strict=True):
        span = (last - first).days + 1
        for offset in range(span):
            day = first + dt.timedelta(days=offset)
            length[day] = span
            index[day] = offset + 1

    dates = daily["date"].to_list()
    return daily.with_columns(
        pl.Series("holiday_run_len_admin", [length.get(d, 0) for d in dates], dtype=pl.Int8),
        pl.Series("holiday_day_index_admin", [index.get(d, 0) for d in dates], dtype=pl.Int8),
    )


def build(path: Path | None = None) -> pl.DataFrame:
    """讀區間表、展開、寫出每日 CSV。

    Args:
        path: 輸出路徑，None 時採用 ``paths.SPECIAL_DAYS_FILE``。

    Returns:
        pl.DataFrame: 寫出的每日表。
    """
    path = path or paths.SPECIAL_DAYS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    out = expand(
        load_intervals(),
        settings.SPECIAL_DAYS_START,
        settings.SPECIAL_DAYS_END,
    )
    out.write_csv(path)
    logger.info("特殊日期已寫出：%s（%d 天）", paths.relative(path), out.height)
    return out
