"""農曆與節氣特徵。

**這兩個欄位從專案第一天就在時間電價日曆表裡，但從未被測試過。**
它們原本被列在 :data:`src.features.builder.NON_FEATURE_COLUMNS`——理由是
「字串欄位」，那是**格式問題不是資訊問題**，而黑名單把兩者混為一談了。

為什麼值得測：**農曆節日在國曆上會漂移**。

| 節日 | 2024 | 2025 | 2026 |
|---|---|---|---|
| 中秋 | 9/17 | **10/6** | 9/25 |
| 春節 | 2/10 | 1/29 | 2/17 |

`day_of_year`、`month`、`doy_sin/cos` 這些既有特徵**都抓不到**這種漂移，
只有農曆抓得到。

節氣（24 節氣）在日曆表中**只標在當天**，其餘為空，
故需前向填補成「目前處於哪個節氣」——直接當特徵用會有 97% 是缺值。

**零洩漏**：純日曆量，給定日期即可查得，與當日負載無關。
"""

from __future__ import annotations

import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

LUNAR_COLUMN = "lunar"
SOLAR_TERM_COLUMN = "solar_term"


def parse_lunar(text: str | None) -> tuple[int, int] | None:
    """把農曆字串解析成 ``(月, 日)``。

    格式（實測 912 天 100% 符合）：``八月十五`` = 8 月 15 日。
    日的寫法有四種前綴：``初一``–``初十``、``十一``–``十九``、
    ``二十``、``廿一``–``廿九``、``三十``。

    Args:
        text: 農曆字串，例如 ``"八月十五"``。None 或無法解析時回傳 None。

    Returns:
        tuple[int, int] | None: ``(月, 日)``，月與日皆為 1 起算。
    """
    if not text or "月" not in text:
        return None
    month_text, _, day_text = text.partition("月")
    month_text = month_text.removeprefix("閏")
    if month_text not in settings.LUNAR_MONTHS:
        return None
    month = settings.LUNAR_MONTHS.index(month_text) + 1

    day = _parse_lunar_day(day_text)
    return (month, day) if day else None


def _parse_lunar_day(text: str) -> int | None:
    """把農曆的日解析成 1–30。

    順序有意義：``三十`` 與 ``二十`` 必須在 ``十`` 之前比對，
    否則 ``三十`` 會被當成前綴 ``十`` 而算成 10。

    Args:
        text: 日的部分，例如 ``"十五"``、``"初三"``、``"廿七"``。

    Returns:
        int | None: 1–30，無法解析時 None。
    """
    ones = "一二三四五六七八九十"
    if text == "三十":
        return 30
    if text == "二十":
        return 20
    if text.startswith("初"):
        rest = text[1:]
        return ones.index(rest) + 1 if rest in ones else None
    if text.startswith("廿"):
        rest = text[1:]
        return 20 + ones.index(rest) + 1 if rest in ones else None
    if text.startswith("十"):
        rest = text[1:]
        if not rest:
            return 10
        return 10 + ones.index(rest) + 1 if rest in ones else None
    return None


def add_lunar_features(
    daily: pl.DataFrame, source: pl.DataFrame | None = None
) -> pl.DataFrame:
    """加上農曆與節氣特徵。

    **必須在連續的日期序列上計算，再併回 `daily`。**
    ``days_to_lunar_festival`` 與節氣的前向填補都依賴**相鄰列就是相鄰日**。
    推論時 `daily` 是「訓練資料 + 三個預測日」，中間有好幾個月的斷裂——
    實測 2026-10-01 會算出「距節日 −12、節氣 = 夏至」，
    而正確值是「−6、秋分」（因為 2026 中秋 9/25 不在那個 frame 裡）。
    **訓練時與預測時算出不同的值，比沒有這個特徵更糟，而且完全不會報錯。**
    故 ``source`` 預設為完整的電價日曆表（涵蓋 2011–2060）。

    產生的欄位：

    | 欄位 | 內容 |
    |---|---|
    | ``lunar_month`` | 農曆月 1–12 |
    | ``lunar_day`` | 農曆日 1–30 |
    | ``is_lunar_festival`` | 是否為 :data:`settings.LUNAR_FESTIVALS` 之一 |
    | ``is_lunar_new_year_eve`` | 是否為除夕（該農曆年的最後一天） |
    | ``days_to_lunar_festival`` | 距最近的農曆大節日幾天（**有號**，負為節後） |
    | ``solar_term_index`` | 目前處於第幾個節氣（0 = 立春），前向填補 |
    | ``days_since_solar_term`` | 距上一個節氣起算幾天 |

    Args:
        daily: 含 ``date`` 的每日表（可以有斷裂）。
        source: 連續的日期來源，須含 ``date`` / ``lunar`` / ``solar_term``。
            None 時讀完整的電價日曆表。

    Returns:
        pl.DataFrame: ``daily`` 加上上述七欄。

    Raises:
        ValueError: 來源缺少 ``lunar`` 或 ``solar_term`` 欄。
    """
    if source is None:
        from src.data import external

        source = external.load_calendar()
    missing = [c for c in (LUNAR_COLUMN, SOLAR_TERM_COLUMN) if c not in source.columns]
    if missing:
        raise ValueError(f"缺少欄位 {missing}——農曆特徵需要時間電價日曆表的原始欄位")

    out = source.select("date", LUNAR_COLUMN, SOLAR_TERM_COLUMN).sort("date")
    parsed = [parse_lunar(v) for v in out[LUNAR_COLUMN].to_list()]
    months = [p[0] if p else None for p in parsed]
    days = [p[1] if p else None for p in parsed]
    n_bad = sum(p is None for p in parsed)
    if n_bad:
        logger.warning("有 %d 天的農曆欄位無法解析（將為 null）", n_bad)

    festival = [bool(p) and p in settings.LUNAR_FESTIVALS for p in parsed]
    # 除夕：該農曆年的最後一天，可能是十二月三十或廿九，故不能寫死日期，
    # 改判「今天是十二月、明天是正月初一」。
    eve = [
        bool(p) and p[0] == 12 and (i + 1 < len(parsed))
        and parsed[i + 1] == (1, 1)
        for i, p in enumerate(parsed)
    ]

    out = out.with_columns(
        pl.Series("lunar_month", months, dtype=pl.Int8),
        pl.Series("lunar_day", days, dtype=pl.Int8),
        pl.Series("is_lunar_festival", festival, dtype=pl.Int8),
        pl.Series("is_lunar_new_year_eve", eve, dtype=pl.Int8),
    )

    out = out.with_columns(
        pl.Series("days_to_lunar_festival",
                  _signed_distance([f or e for f, e in zip(festival, eve, strict=True)]),
                  dtype=pl.Int16),
        _solar_term_index(out).alias("solar_term_index"),
    )
    # 不可用 `over("solar_term_index")`——那是**依值分組**，會把 2024 與 2025
    # 的同一個節氣併成一組，計數接續下去（實測最大值變成 47 天，正常應 ≤ 16）。
    # 正確作法是「每遇到一個節氣標記就開一個新的 run」。
    out = out.with_columns(
        pl.col(SOLAR_TERM_COLUMN).is_not_null().cum_sum().alias("_term_run")
    ).with_columns(
        pl.int_range(pl.len()).over("_term_run").cast(pl.Int16)
        .alias("days_since_solar_term")
    ).drop("_term_run")

    produced = ["lunar_month", "lunar_day", "is_lunar_festival",
                "is_lunar_new_year_eve", "days_to_lunar_festival",
                "solar_term_index", "days_since_solar_term"]
    logger.info(
        "農曆特徵在 %d 天的連續來源上計算，併回 %d 天的目標表",
        out.height, daily.height,
    )
    return daily.join(out.select("date", *produced), on="date", how="left")


def _signed_distance(flags: list[bool]) -> list[int | None]:
    """到最近一個 True 的**有號**天數：負為之後、正為之前、0 為當天。

    有號而非絕對值的理由：節前（採買、返鄉）與節後（收假）的用電型態不同，
    取絕對值會把兩者壓成同一個值。

    Args:
        flags: 每天是否為節日。

    Returns:
        list[int | None]: 與輸入等長；沒有任何 True 時全為 None。
    """
    positions = [i for i, f in enumerate(flags) if f]
    if not positions:
        return [None] * len(flags)
    out: list[int | None] = []
    for i in range(len(flags)):
        nearest = min(positions, key=lambda p: abs(p - i))
        out.append(nearest - i)
    return out


def _solar_term_index(daily: pl.DataFrame) -> pl.Expr:
    """節氣序號，前向填補成「目前處於哪個節氣」。

    日曆表只在節氣當天標記，直接當特徵會有 97% 缺值。

    Args:
        daily: 含 ``solar_term`` 的每日表。

    Returns:
        pl.Expr: 前向填補後的節氣序號（0 = 立春）。
    """
    mapping = {name: i for i, name in enumerate(settings.SOLAR_TERMS)}
    return (
        # 必須先 cast 成 Utf8。若整段期間**剛好沒有任何節氣**，polars 會把
        # 該欄推斷成 Null 型別，`replace_strict` 便會因「str → null 轉換失敗」
        # 而拋錯。短視窗（例如三天的預測期）真的可能完全不含節氣。
        pl.col(SOLAR_TERM_COLUMN)
        .cast(pl.Utf8)
        .replace_strict(mapping, default=None, return_dtype=pl.Int8)
        .forward_fill()
    )
