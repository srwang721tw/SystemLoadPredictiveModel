"""daily-row 標籤計算：由 10 分鐘序列產出每日一列的 6 項目標。

本模組是整個 daily-row 架構的入口：把 144 點 × N 天的長表壓成 N 列的寬表，
之後所有建模都在這張表上進行。

關鍵常數（弄錯即全盤皆錯）：
    - 日尖峰 11:00–17:00 含端點，共 **37** 格
    - 夜尖峰 17:10–21:00 含端點，共 **24** 格
    - ramp 為全日 **143** 個相鄰差分，**不跨日**
    - ``ramp_down`` 回報**絕對值**（正值），使評分公式分母恆正
    - 含補值點的差分**仍納入**訓練標籤，但須以旗標統計污染比例

除了 6 項目標，本模組另外產出三類「把資料探索的觀察寫進資料」的欄位，
目的是讓模型不必從頭學已經確知的結構：

    1. ``daytype``：平日 / 週六 / 週日。三者的時刻分布型態差異極大。
    2. ``t_*_censored``：尖峰時刻是否落在窗口端點。落在端點代表真正的尖峰
       在窗口外、被截斷，此時該值不是連續量而是被審查的觀測。
    3. ``ramp_*_regime``：ramp 極值落在哪個結構性時段
       （清晨啟動 / 午休 / 傍晚下班），用於診斷與品質檢查。

時刻欄位一律以「當日分鐘數」(Int32) 儲存，只在輸出提交檔時才轉為
  零填補字串。中途轉成時間型別容易被靜默改型。
"""

from __future__ import annotations

import polars as pl

from config import settings

TARGET_NAMES: tuple[str, ...] = (
    "p_day",
    "t_day",
    "p_night",
    "t_night",
    "ramp_up",
    "ramp_down",
)
"""6 項預測標的的內部名稱。提交檔格式見 ``src/output/submission.py``。"""


def to_minutes(hhmm: str) -> int:
    """``"11:00"`` → 660，即當日分鐘數。

    Args:
        hhmm: 零填補的時刻字串。

    Returns:
        int: 當日分鐘數。
    """
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def format_hhmm(minutes: int) -> str:
    """660 → ``"11:00"``，零填補兩位數字串。

    Args:
        minutes: 當日分鐘數。

    Returns:
        str: ``HH:MM`` 格式，小時與分鐘皆補滿兩位數。
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _grid(start: str, end: str) -> list[int]:
    """列出 [start, end] 區間內（含端點）的所有格點分鐘數。"""
    return list(range(to_minutes(start), to_minutes(end) + 1, settings.DATA_FREQ_MIN))


def day_peak_grid() -> list[int]:
    """日尖峰期間的 37 個合法格點（當日分鐘數）。"""
    return _grid(settings.DAY_PEAK_START, settings.DAY_PEAK_END)


def night_peak_grid() -> list[int]:
    """夜尖峰期間的 24 個合法格點（當日分鐘數）。"""
    return _grid(settings.NIGHT_PEAK_START, settings.NIGHT_PEAK_END)


def add_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    """由 ``ts`` 推導 ``date`` 與 ``mod``（當日分鐘數）。

    ``dt.hour()`` / ``dt.minute()`` 回傳 i8，直接乘 60 會溢位
    （11 × 60 = 660 > 127）而**不會報錯**，必須先 ``cast(pl.Int32)``。
    此錯誤會使所有時間篩選靜默失效。

    Args:
        df: 含 ``ts`` (Datetime) 欄位的長表。

    Returns:
        pl.DataFrame: 加上 ``date`` (Date) 與 ``mod`` (Int32)。
    """
    return df.with_columns(
        pl.col("ts").dt.date().alias("date"),
        (
            pl.col("ts").dt.hour().cast(pl.Int32) * 60
            + pl.col("ts").dt.minute().cast(pl.Int32)
        ).alias("mod"),
    )


def _extremum_per_day(
    df: pl.DataFrame,
    value_col: str,
    out_value: str,
    out_time: str,
    largest: bool,
) -> pl.DataFrame:
    """取每日 ``value_col`` 的極值與其發生時刻。

    並列時取**最早**的時刻：先依值排序、再依時刻升冪，取每組第一列。
    null 值先被排除，故不會被誤選為極值。

    Args:
        df: 含 ``date``、``mod``、``value_col`` 的長表。
        value_col: 要取極值的欄位。
        out_value: 輸出的值欄位名稱。
        out_time: 輸出的時刻欄位名稱。
        largest: True 取最大值，False 取最小值。

    Returns:
        pl.DataFrame: 每日一列，欄位為 ``date``、``out_value``、``out_time``。
    """
    return (
        df.drop_nulls(value_col)
        .sort([value_col, "mod"], descending=[largest, False])
        .group_by("date", maintain_order=True)
        .first()
        .select(
            "date",
            pl.col(value_col).alias(out_value),
            pl.col("mod").cast(pl.Int32).alias(out_time),
        )
    )


def compute_targets(df: pl.DataFrame) -> pl.DataFrame:
    """由 10 分鐘序列計算每日 6 項目標，產出 daily-row 寬表。

    差分以 ``date`` 分組計算，因此**不跨日**，完整的一天恰為 143 個差分。
    極值並列時取最早的時刻。

    Args:
        df: 補值後的完整序列，須含 ``ts``、``Load_MW``、``is_imputed``。
            ``Load_MW`` 為補值後的值；原始缺失位置由 ``is_imputed`` 標記。

    Returns:
        pl.DataFrame: 每日一列，欄位包含
            - 6 項目標 :data:`TARGET_NAMES`
            - ``t_ramp_up`` / ``t_ramp_down``：ramp 極值發生時刻（供診斷，不評分）
            - ``n_points``：當日筆數，應恆為 144
            - ``n_imputed`` / ``has_imputed``：補值統計
            - ``ramp_up_on_imputed`` / ``ramp_down_on_imputed``：
              該 ramp 極值所在的差分是否碰到補值點
    """
    df = add_time_columns(df)

    # ramp 差分：以 date 分組，故不跨日；差分時刻定義為區間的**結束**時刻。
    # 差分只要兩端任一端是補值點就算受污染。
    df = df.sort(["date", "mod"]).with_columns(
        pl.col("Load_MW").diff().over("date").alias("d"),
        (pl.col("is_imputed") | pl.col("is_imputed").shift(1).over("date")).alias(
            "d_on_imputed"
        ),
    )

    day_grid, night_grid = day_peak_grid(), night_peak_grid()

    peak_day = _extremum_per_day(
        df.filter(pl.col("mod").is_between(day_grid[0], day_grid[-1])),
        "Load_MW",
        "p_day",
        "t_day",
        largest=True,
    )
    peak_night = _extremum_per_day(
        df.filter(pl.col("mod").is_between(night_grid[0], night_grid[-1])),
        "Load_MW",
        "p_night",
        "t_night",
        largest=True,
    )

    up = (
        df.drop_nulls("d")
        .sort(["d", "mod"], descending=[True, False])
        .group_by("date", maintain_order=True)
        .first()
        .select(
            "date",
            pl.col("d").alias("ramp_up"),
            pl.col("mod").cast(pl.Int32).alias("t_ramp_up"),
            pl.col("d_on_imputed").alias("ramp_up_on_imputed"),
        )
    )
    down = (
        df.drop_nulls("d")
        .sort(["d", "mod"], descending=[False, False])
        .group_by("date", maintain_order=True)
        .first()
        .select(
            "date",
            pl.col("d").abs().alias("ramp_down"),  # 回報絕對值（正值）
            pl.col("mod").cast(pl.Int32).alias("t_ramp_down"),
            pl.col("d_on_imputed").alias("ramp_down_on_imputed"),
        )
    )

    # 曲線水準：00:00、全日最小、23:50。
    # 這三個量**不參與評分**（不在尖峰窗口的極值上，也不是 ramp 極值），
    # 用途是合成提交曲線時的自由參數（見 curve.FreeLevels）。
    # 把它們算在這裡，是為了讓 daily-row 的歷史自帶合成所需的一切，
    # 合成時不必再回頭讀 10 分鐘序列。
    levels = df.group_by("date").agg(
        pl.col("Load_MW").filter(pl.col("mod") == 0).first().alias("load_start"),
        pl.col("Load_MW").min().alias("load_min"),
        pl.col("Load_MW")
        .filter(pl.col("mod") == (settings.POINTS_PER_DAY - 1) * settings.DATA_FREQ_MIN)
        .first()
        .alias("load_end"),
    )

    quality = df.group_by("date").agg(
        pl.len().alias("n_points"),
        pl.col("is_imputed").sum().cast(pl.Int32).alias("n_imputed"),
        pl.col("is_imputed").any().alias("has_imputed"),
    )

    return (
        quality.join(peak_day, on="date", how="left")
        .join(peak_night, on="date", how="left")
        .join(up, on="date", how="left")
        .join(down, on="date", how="left")
        .join(levels, on="date", how="left")
        .sort("date")
    )


def add_daytype(daily: pl.DataFrame) -> pl.DataFrame:
    """加上日別欄位（平日 / 週六 / 週日）與星期、月份。

    這是最強的單一結構因子。實測 max ramp up 落在清晨 07:00–09:00 的比例：
    平日 82.8%、週六 51.9%、週日 24.6%；而 t_night 晚於 19:10 的比例：
    平日 2.3%、週六 10.1%、週日 72.3%。

    這只是日別，**不等於**「是否為正常工作日」。國定假日與颱風停班日
    在日曆上是平日，但行為像週日。後者需要外部行事曆，見
    ``src/features/calendar.py``。

    Args:
        daily: 含 ``date`` 的每日表。

    Returns:
        pl.DataFrame: 加上 ``weekday``（1 = 週一）、``month``、``daytype``。
    """
    weekday, saturday, sunday = settings.DAYTYPES
    return daily.with_columns(
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.month().alias("month"),
    ).with_columns(
        pl.when(pl.col("weekday") == 6)
        .then(pl.lit(saturday))
        .when(pl.col("weekday") == 7)
        .then(pl.lit(sunday))
        .otherwise(pl.lit(weekday))
        .alias("daytype")
    )


def add_censoring_flags(daily: pl.DataFrame) -> pl.DataFrame:
    """標記尖峰時刻是否落在窗口端點（被審查的觀測）。

    實測 t_day 有 10.7% 落在右端 17:00（冬季週日達 66–92%），
    t_night 有 28.2% 落在左端 17:10（夏季平日達 84–91%）。
    落在端點代表真正的尖峰在窗口外被截斷。

    這是時刻採用「分類 + 貝氏決策」而非回歸的關鍵證據：
    回歸的條件期望值會落在兩個模式之間的空隙——資料裡幾乎不出現的位置；
    分類則能把「就是端點」當成一個高機率類別直接學會。

    Args:
        daily: 含 ``t_day`` / ``t_night`` 的每日表。

    Returns:
        pl.DataFrame: 加上 ``t_day_censored`` / ``t_night_censored``
            （值為 ``"lower"`` / ``"upper"`` / null）。
    """
    for target, grid in (("t_day", day_peak_grid()), ("t_night", night_peak_grid())):
        daily = daily.with_columns(
            pl.when(pl.col(target) == grid[0])
            .then(pl.lit("lower"))
            .when(pl.col(target) == grid[-1])
            .then(pl.lit("upper"))
            .otherwise(None)
            .alias(f"{target}_censored")
        )
    return daily


def add_ramp_regime_flags(daily: pl.DataFrame) -> pl.DataFrame:
    """標記 ramp 極值落在哪個結構性時段。

    依 ``settings`` 中由資料探索訂出的四個視窗：清晨啟動（07:00–09:00）、
    午休下降（11:30–12:30）、午休回工（13:00–13:30）、傍晚下班（16:30–17:30）。

    用途是診斷而非評分特徵：ramp 的**時刻**不被評分，但知道極值落在哪個
    機制上，才能判斷標籤是否被 DR 或補值污染，也才能解釋
    2025 年 ramp_down 從午休模式翻到傍晚模式這類結構變化。

    Args:
        daily: 含 ``t_ramp_up`` / ``t_ramp_down`` 的每日表。

    Returns:
        pl.DataFrame: 加上 ``ramp_up_regime`` / ``ramp_down_regime``。
    """
    regimes = (
        ("morning", settings.MORNING_RAMP_WINDOW),
        ("midday", settings.MIDDAY_RAMP_DOWN_WINDOW),
        ("lunch_return", settings.LUNCH_RETURN_RAMP_WINDOW),
        ("evening", settings.EVENING_RAMP_DOWN_WINDOW),
    )
    for col, out in (
        ("t_ramp_up", "ramp_up_regime"),
        ("t_ramp_down", "ramp_down_regime"),
    ):
        expr = pl.lit("other")
        for name, (start, end) in reversed(regimes):
            expr = (
                pl.when(pl.col(col).is_between(to_minutes(start), to_minutes(end)))
                .then(pl.lit(name))
                .otherwise(expr)
            )
        daily = daily.with_columns(expr.alias(out))
    return daily


def build_daily(df: pl.DataFrame) -> pl.DataFrame:
    """一次完成 daily-row 表的建構：目標 + 日別 + 審查旗標 + regime 旗標。

    Args:
        df: 補值後的完整 10 分鐘序列。

    Returns:
        pl.DataFrame: 每日一列的完整標籤表。
    """
    daily = compute_targets(df)
    daily = add_daytype(daily)
    daily = add_censoring_flags(daily)
    return add_ramp_regime_flags(daily)


def summarize_targets(daily: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """6 項目標的敘述統計、邊界審查比例與補值污染比例。

    Args:
        daily: :func:`build_daily` 的輸出。

    Returns:
        dict[str, pl.DataFrame]: 三張摘要表
            - ``"describe"``：各目標的敘述統計（依日別分組）
            - ``"censoring"``：各日別的邊界審查比例
            - ``"contamination"``：ramp 極值落在補值點的比例
    """
    describe = (
        daily.group_by("daytype")
        .agg(
            pl.len().alias("n"),
            *[
                stat
                for name in TARGET_NAMES
                for stat in (
                    pl.col(name).median().alias(f"{name}_median"),
                    pl.col(name).quantile(0.10).alias(f"{name}_p10"),
                    pl.col(name).quantile(0.90).alias(f"{name}_p90"),
                )
            ],
        )
        .sort("daytype")
    )

    censoring = (
        daily.group_by("daytype")
        .agg(
            pl.len().alias("n"),
            # 未被審查的日子該欄為 null，而 polars 的 mean() 會跳過 null。
            #   不 fill_null(False) 的話算出來的是「在被審查的日子裡佔幾成」，
            #   而不是「佔全部日子的幾成」。
            *[
                (pl.col(f"{target}_censored") == side)
                .fill_null(False)
                .mean()
                .alias(f"{target}_at_{side}")
                for target in ("t_day", "t_night")
                for side in ("lower", "upper")
            ],
        )
        .sort("daytype")
    )

    contamination = daily.select(
        pl.len().alias("n_days"),
        pl.col("has_imputed").sum().alias("n_days_with_imputed"),
        pl.col("ramp_up_on_imputed").sum().alias("n_ramp_up_contaminated"),
        pl.col("ramp_down_on_imputed").sum().alias("n_ramp_down_contaminated"),
        pl.col("ramp_up_on_imputed").mean().alias("ramp_up_contamination_rate"),
        pl.col("ramp_down_on_imputed").mean().alias("ramp_down_contamination_rate"),
    )

    return {"describe": describe, "censoring": censoring, "contamination": contamination}
