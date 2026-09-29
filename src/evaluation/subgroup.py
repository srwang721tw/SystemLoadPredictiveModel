"""變體比較的拆解診斷。

**本模組存在的理由是一個真實的測試盲點。**

曾用「全年 60 折的 `total_score` 平均」測特殊日期旗標，結論是全部無效。
但事後量測發現 `is_holiday` 對 `ramp_up` 的分辨力是全專案最強的單變數訊號
（連假平日中位 623 MW vs 一般平日 1169 MW，**AUC 0.952**）——
之所以測不出來，是因為：

- 連假平日只有 **10 / 912 天**
- `ramp_up` 只佔總分 **2.2%**

**用全年總分去測一個只作用在 1% 天數上的修正，在數學上就看不見。**

本模組強制每個變體都輸出三層拆解：逐目標、逐子群、以及**有幾天真的改變**
（曾兩次差點被少數幾天主導的平均值騙過去）。
"""

from __future__ import annotations

from math import comb

import numpy as np
import polars as pl

from config import settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

MAGNITUDE_TARGETS = ("p_day", "p_night", "ramp_up", "ramp_down")
TIMING_TARGETS = ("t_day", "t_night")


def per_target_loss(actual: pl.DataFrame, predicted: pl.DataFrame) -> pl.DataFrame:
    """逐日、逐目標的損失，供子群比較使用。

    時刻用 ``(相差格數) ** 1.2``、量值用絕對相對誤差——與 `metrics` 的定義一致，
    但**保留逐日明細**而非直接平均，因為子群比較需要逐日的值。

    Args:
        actual: 含 ``date`` 與 6 個目標的實際值。
        predicted: 同樣欄位的預測值，列序須與 ``actual`` 對應。

    Returns:
        pl.DataFrame: 欄位 ``date`` 加上每個目標的損失。

    Raises:
        ValueError: 兩張表的列數不同。
    """
    if actual.height != predicted.height:
        raise ValueError(f"列數不符：實際 {actual.height}、預測 {predicted.height}")

    out = {"date": actual["date"]}
    for target in TIMING_TARGETS:
        grids = (actual[target] - predicted[target]).abs() / settings.DATA_FREQ_MIN
        out[target] = grids ** settings.TIME_ERROR_EXPONENT
    for target in MAGNITUDE_TARGETS:
        out[target] = ((predicted[target] - actual[target]) / actual[target]).abs()
    return pl.DataFrame(out)


def compare(
    baseline: pl.DataFrame, variant: pl.DataFrame, groups: pl.DataFrame | None = None
) -> pl.DataFrame:
    """把兩個變體的逐日損失拆成「逐目標 × 子群」的差值表。

    Args:
        baseline: :func:`per_target_loss` 的輸出（基準）。
        variant: 同上（變體）。
        groups: 可選的分組表，須含 ``date`` 與一個分組欄；
            None 時只做整體比較。

    Returns:
        pl.DataFrame: 每個「子群 × 目標」一列，含天數、**改變的天數**、
            好／壞的天數、平均差值與單尾符號檢定 p 值。
    """
    targets = TIMING_TARGETS + MAGNITUDE_TARGETS
    frame = baseline.select("date", *targets).rename({t: f"b_{t}" for t in targets})
    frame = frame.join(
        variant.select("date", *targets).rename({t: f"v_{t}" for t in targets}),
        on="date", how="inner",
    )
    if groups is not None:
        column = [c for c in groups.columns if c != "date"][0]
        frame = frame.join(groups.select("date", column), on="date", how="left")
    else:
        column = "_all"
        frame = frame.with_columns(pl.lit("整體").alias(column))

    rows = []
    for key, part in frame.group_by(column, maintain_order=True):
        name = key[0] if isinstance(key, tuple) else key
        for target in targets:
            delta = (part[f"v_{target}"] - part[f"b_{target}"]).to_numpy()
            delta = delta[~np.isnan(delta)]
            changed = int((delta != 0).sum())
            better = int((delta < 0).sum())
            rows.append({
                "子群": str(name), "目標": target, "天數": len(delta),
                "改變天數": changed, "好": better, "壞": changed - better,
                "平均差值": float(delta.mean()) if len(delta) else 0.0,
                "符號檢定p": sign_test(better, changed),
            })
    return pl.DataFrame(rows)


def sign_test(better: int, changed: int) -> float:
    """單尾符號檢定：在 ``changed`` 次改變中出現 ``better`` 次改善的機率。

    曾兩次出現「平均值很漂亮、
    bootstrap CI 甚至不含 0，但拆開只有 1–6 天真的改變」的情形。
    離散指標一律先看改變的天數與符號分布，再看平均值。

    Args:
        better: 變好的天數。
        changed: 有變化的總天數。

    Returns:
        float: 單尾 p 值；``changed`` 為 0 時回傳 1.0。
    """
    if changed <= 0:
        return 1.0
    tail = sum(comb(changed, i) for i in range(better, changed + 1))
    return tail / 2 ** changed


def format_report(table: pl.DataFrame, min_changed: int = 1) -> str:
    """把 :func:`compare` 的輸出排成可讀的文字報告。

    只列出**真的有改變**的列——沒改變的目標列出來只是雜訊。

    Args:
        table: :func:`compare` 的輸出。
        min_changed: 至少改變幾天才列出。

    Returns:
        str: 報告文字。
    """
    shown = table.filter(pl.col("改變天數") >= min_changed).sort("平均差值")
    if not shown.height:
        return "（沒有任何一天的預測改變）"
    lines = [f"{'子群':<14}{'目標':<10}{'天數':>5}{'改變':>5}{'好/壞':>8}"
             f"{'平均差值':>11}{'符號p':>8}"]
    for r in shown.iter_rows(named=True):
        lines.append(
            f"{r['子群']:<14}{r['目標']:<10}{r['天數']:>5}{r['改變天數']:>5}"
            f"{f'{r['好']}/{r['壞']}':>8}{r['平均差值']:>+11.4f}{r['符號檢定p']:>8.3f}"
        )
    return "\n".join(lines)
