"""尖峰時刻的貝氏決策。

給定機率分布 ``P(k)``，選出使**評分函數的期望值**最小的時刻：

    j* = argmin_j  Σ_k P(k) · (|j − k| / 10) ^ 1.2

這不是 argmax P(k)（眾數），也不是期望值或中位數：指數 1.2 介於 1 與 2 之間，
最佳決策既非中位數（指數 1）也非均值（指數 2），必須真的把期望損失算出來再取最小。

## 兩個必須分清楚的集合

|  | 集合 | 大小 | 性質 |
|---|---|---|---|
| 類別集合 ``classes`` | 訓練資料中**出現過**的格點 | 平日 t_day 約 31 | 模型的限制 |
| 候選集合 ``candidates`` | **完整名目格點** | t_day 37／t_night 24 | 決策的選擇 |

候選集合必須是完整格點：

    P(11:30) = 0.5、P(13:30) = 0.5 時
        j = 11:30 → 0.5 × (120/10)^1.2      = 9.87
        j = 12:30 → 2 × 0.5 × (60/10)^1.2   = 8.59  ← 最佳
        j = 13:30 → 0.5 × (120/10)^1.2      = 9.87

12:30 在 650 個平日裡出現 0 次，卻是最佳解——凸損失會把雙峰分布的最佳點拉進
中間的空隙。因此損失矩陣是長方形的（候選 × 類別），且必須用**實際分鐘差**計算。
"""

from __future__ import annotations

import numpy as np

from config import settings
from src.features.targets import day_peak_grid, night_peak_grid

type FloatArray = np.ndarray


def candidate_grid(target: str) -> list[int]:
    """取得時刻目標的候選點集合（完整名目格點，日尖峰 37 個、夜尖峰 24 個）。

    Raises:
        ValueError: 目標名稱不正確，或設定關閉了完整格點。
    """
    if not settings.DECISION_CANDIDATES_FULL_GRID:
        raise ValueError("DECISION_CANDIDATES_FULL_GRID 必須為 True：縮減候選點會使"
                         "雙峰分布的最佳解無法被選出")
    if target == "t_day":
        return day_peak_grid()
    if target == "t_night":
        return night_peak_grid()
    raise ValueError(f"未知的時刻目標：{target}")


def loss_matrix(candidates: list[int], classes: list[int]) -> FloatArray:
    """候選 × 類別的長方形損失矩陣 ``L[j, k] = (|candidates[j] − classes[k]| / 10) ** 1.2``。

    一律用實際分鐘差，不可用類別序號差：類別集合在值域中有缺口時
    （平日 t_day 從未落在 12:00–12:50），序號差會把 11:50 與 13:00 當成相鄰。
    """
    cand = np.asarray(candidates, dtype=float)[:, None]
    cls = np.asarray(classes, dtype=float)[None, :]
    return (np.abs(cand - cls) / settings.DATA_FREQ_MIN) ** settings.TIME_ERROR_EXPONENT


def expected_loss(pmf: FloatArray, candidates: list[int], classes: list[int]) -> FloatArray:
    """每個候選點的期望損失，形狀 ``(n_samples, n_candidates)``。

    Raises:
        ValueError: ``pmf`` 的欄數與 ``classes`` 長度不符。
    """
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    if pmf.shape[1] != len(classes):
        raise ValueError(f"pmf 有 {pmf.shape[1]} 欄，但 classes 有 {len(classes)} 個")
    return pmf @ loss_matrix(candidates, classes).T


def bayes_decision(pmf: FloatArray, classes: list[int], target: str) -> np.ndarray:
    """對每筆樣本做貝氏決策，回傳最小期望損失的時刻。

    Args:
        pmf: 機率分布，形狀 ``(n_samples, n_classes)``。
        classes: 類別對應的時刻（分鐘），順序須與 ``pmf`` 的欄一致。
        target: ``"t_day"`` 或 ``"t_night"``，決定候選點集合。

    Returns:
        np.ndarray: 形狀 ``(n_samples,)`` 的決策時刻（分鐘），dtype 為 int。

    Raises:
        ValueError: ``pmf`` 含負值或列和明顯偏離 1。
    """
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    if (pmf < 0).any():
        raise ValueError("pmf 含負值")
    row_sums = pmf.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"pmf 的列和偏離 1（最大偏差 {np.abs(row_sums - 1).max():.2e}）")

    candidates = candidate_grid(target)
    losses = expected_loss(pmf, candidates, classes)
    # 並列時取較早的候選點（argmin 的預設行為，候選點已排序）。
    return np.asarray(candidates, dtype=int)[losses.argmin(axis=1)]
