"""模型配對比較與標準誤門檻。

**所有模型選擇一律呼叫本模組**，不得僅憑 CV 平均分較低即採用某模型。

規則：
    1. 所有候選在**完全相同的折**上評估（配對比較）
    2. 計算逐折差值的平均與標準誤
    3. 改善須大於 ``settings.CV_STDERR_THRESHOLD`` 個標準誤才視為真實；
       否則**選擇較簡單者**

理由：僅一次提交、無 leaderboard 回饋，過度相信雜訊等級的差異等於把賭注押在
無法驗證的假設上。

**為何用逐折差值而不是各自的標準誤**：兩個模型在同一折上的分數高度相關
（難的折對誰都難）。配對後這個共同成分被消掉，差值的標準誤遠小於各自分數的
標準誤，因此配對比較的檢定力高得多。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from config import settings


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """兩個模型在同一組折上的配對比較結果。

    Attributes:
        name_a: 候選 A 名稱。
        name_b: 候選 B 名稱。
        n_folds: 折數。
        mean_a: A 的平均分。
        mean_b: B 的平均分。
        mean_diff: 逐折差值平均 ``mean(score_a − score_b)``，負值代表 A 較佳。
        stderr: 逐折差值的標準誤。
        n_stderr: ``|mean_diff| / stderr``，即差異達幾個標準誤。
        passes_threshold: 是否通過門檻。
        winner: 依規則判定的採用者；未通過門檻時為較簡單者。
    """

    name_a: str
    name_b: str
    n_folds: int
    mean_a: float
    mean_b: float
    mean_diff: float
    stderr: float
    n_stderr: float
    passes_threshold: bool
    winner: str


def paired_compare(
    scores_a: pl.DataFrame,
    scores_b: pl.DataFrame,
    name_a: str,
    name_b: str,
    simpler: str,
    threshold: float | None = None,
) -> PairedComparison:
    """對兩組逐折分數做配對比較。

    Args:
        scores_a: 候選 A 的逐折分數（``run_cv`` 的輸出）。
        scores_b: 候選 B 的逐折分數，折的 ``origin`` 須與 A 完全相同。
        name_a: 候選 A 名稱。
        name_b: 候選 B 名稱。
        simpler: 兩者中較簡單者的名稱，未通過門檻時採用之。
        threshold: 標準誤門檻，None 時採用 ``settings.CV_STDERR_THRESHOLD``。

    Returns:
        PairedComparison: 比較結果。

    Raises:
        ValueError: 兩組折不一致，或 ``simpler`` 不是兩者之一。
    """
    threshold = settings.CV_STDERR_THRESHOLD if threshold is None else threshold
    if simpler not in (name_a, name_b):
        raise ValueError(f"simpler 必須是 {name_a} 或 {name_b}，得到 {simpler}")

    a = scores_a.sort("origin")
    b = scores_b.sort("origin")
    if a["origin"].to_list() != b["origin"].to_list():
        raise ValueError("兩組折的 origin 不一致，配對比較的前提被破壞")

    diff = a["total_score"] - b["total_score"]
    n = diff.len()
    mean_diff = float(diff.mean())
    # n = 1 時標準誤無定義；視為無法判別，一律選較簡單者。
    stderr = float(diff.std() / (n**0.5)) if n > 1 else float("inf")
    n_stderr = abs(mean_diff) / stderr if stderr > 0 else float("inf")

    passes = n_stderr > threshold
    if passes:
        winner = name_a if mean_diff < 0 else name_b
    else:
        winner = simpler

    return PairedComparison(
        name_a=name_a,
        name_b=name_b,
        n_folds=n,
        mean_a=float(a["total_score"].mean()),
        mean_b=float(b["total_score"].mean()),
        mean_diff=mean_diff,
        stderr=stderr,
        n_stderr=n_stderr,
        passes_threshold=passes,
        winner=winner,
    )


