"""尖峰時刻的機率分布：多類別分類模型與經驗分布工具。

``t_day``／``t_night`` 不用回歸，改用「格點上的機率分布 + 貝氏決策」。三個理由：

1. **邊界審查**：``t_day`` 有 10.7% 落在右端 17:00（冬季週日達 66–92%），
   ``t_night`` 有 28.2% 落在左端 17:10（夏季平日達 84–91%）。落在端點代表真正的
   尖峰在窗口外、被截斷；回歸會去擬合不存在的內部極值，分類則直接把邊界學成一個類別。
2. **多峰**：``t_day`` 有三個叢集，回歸會把預測拉到叢集之間機率很低的空隙。
3. **評分需要分布**：貝氏決策要 ``P(k)`` 才能算期望損失。

類別集合 ≠ 候選集合：多類別模型只能對訓練資料出現過的格點輸出機率（模型的限制）；
候選集合則是完整名目格點（37／24），由 :mod:`src.models.decision` 負責——
期望損失的最小點可以落在從未出現過的時刻上。
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from config import settings
from src.logging_setup import get_logger
from src.models import decision

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TimingModel:
    """訓練完成的時刻分類模型。

    Attributes:
        booster: LightGBM 模型。訓練資料只有單一類別時為 ``None``
            （沒有東西可學，:func:`predict_pmf` 直接回傳確定性分布）。
        classes: 類別序號 → 時刻（當日分鐘數），已排序。
        target: 目標名稱（``t_day`` 或 ``t_night``）。
        feature_names: 訓練時的特徵順序，推論時須一致。
    """

    booster: lgb.Booster | None
    classes: list[int]
    target: str
    feature_names: list[str]


def build_class_mapping(
    y_minutes: pl.Series, min_count: int | None = None
) -> tuple[list[int], dict[int, int]]:
    """建立「時刻 → 類別序號」的對照，並合併過於罕見的格點。

    出現次數少於 ``min_count`` 的格點學不到規律，只會膨脹 softmax 的維度。
    這類樣本**併入時間上最接近的保留類別**，而不是丟棄該列——丟棄會使罕見時刻的
    資訊完全消失，而那正是邊界審查發生的地方。

    Args:
        y_minutes: 訓練資料的實際時刻（當日分鐘數）。
        min_count: 保留類別的最少出現次數，None 時採 ``settings.TIMING_MIN_CLASS_COUNT``。

    Returns:
        tuple: ``(classes, mapping)``。``classes`` 為排序後的保留時刻清單；
            ``mapping`` 把**所有**觀察到的時刻映到保留類別的序號。

    Raises:
        ValueError: 沒有任何有效樣本。
    """
    min_count = min_count or settings.TIMING_MIN_CLASS_COUNT
    counts = y_minutes.value_counts().sort("count", descending=True)
    if counts.height == 0:
        raise ValueError("訓練資料沒有任何時刻樣本")

    column = counts.columns[0]
    kept = sorted(counts.filter(pl.col("count") >= min_count)[column].to_list())
    if not kept:
        # 所有格點都很罕見：保留出現最多的那一個，模型退化成常數預測，流程不中斷。
        kept = [int(counts[column][0])]
        logger.warning("所有格點出現次數皆 < %d，退回單一類別 %s", min_count, kept)

    mapping = {
        int(observed): int(np.argmin([abs(observed - k) for k in kept]))
        for observed in counts[column].to_list()
    }
    return kept, mapping


def fit(
    features: pl.DataFrame,
    y_minutes: pl.Series,
    target: str,
    feature_names: list[str],
    seed: int | None = None,
    num_rounds: int | None = None,
) -> TimingModel:
    """訓練時刻分類模型。

    Args:
        features: 特徵矩陣（僅訓練資料）。
        y_minutes: 對應的實際時刻（當日分鐘數）。
        target: ``"t_day"`` 或 ``"t_night"``。
        feature_names: 特徵順序。
        seed: 隨機種子，None 時採 ``settings.RANDOM_SEED``。
        num_rounds: 提升輪數，None 時採 ``settings.TIMING_LEARNED_ROUNDS``。

    Returns:
        TimingModel: 訓練完成的模型。
    """
    classes, mapping = build_class_mapping(y_minutes)
    labels = np.array([mapping[int(m)] for m in y_minutes.to_list()], dtype=int)

    if len(classes) == 1:
        # LightGBM 的 multiclass 不接受 num_class = 1。
        logger.warning("%s 的訓練資料只有 1 個類別（%s），退化為常數預測", target, classes[0])
        return TimingModel(None, classes, target, feature_names)

    params = dict(settings.LGB_TIMING_PARAMS)
    params["num_class"] = len(classes)
    params["seed"] = settings.RANDOM_SEED if seed is None else seed

    dataset = lgb.Dataset(
        features.select(feature_names).to_numpy(),
        label=labels,
        feature_name=feature_names,
        free_raw_data=False,
    )
    booster = lgb.train(params, dataset, num_boost_round=num_rounds or settings.TIMING_LEARNED_ROUNDS)
    return TimingModel(booster, classes, target, feature_names)


def predict_pmf(model: TimingModel, features: pl.DataFrame) -> np.ndarray:
    """輸出各類別的機率分布，形狀 ``(n_samples, n_classes)``，每列加總為 1。"""
    if model.booster is None:
        return np.ones((features.height, 1))
    raw = model.booster.predict(features.select(model.feature_names).to_numpy())
    pmf = np.atleast_2d(np.asarray(raw, dtype=float))
    return pmf / pmf.sum(axis=1, keepdims=True)


def hierarchical_pmf(
    history: pl.DataFrame,
    target: str,
    levels: tuple[tuple[str, ...], ...],
    group_values: tuple[tuple, ...],
    alpha: float,
) -> tuple[np.ndarray, list[int]]:
    """階層式收縮的經驗分布：細分組的頻率，以粗分組為先驗。

    細分組樣本多時信任它，樣本少時退回粗分組。由最粗往最細逐層混合：

        P ← (n_level · P_level + α · P_prior) / (n_level + α)

    ``α`` 是「先驗相當於幾個樣本」：0 = 只用最細分組，→∞ = 忽略細分組。

    Args:
        history: 歷史每日表，須含 ``target`` 與所有分層欄位。
        target: ``"t_day"`` 或 ``"t_night"``。
        levels: 分層欄位，由**粗到細**，例如 ``(("price_daytype",), ("price_daytype", "is_summer"))``。
        group_values: 與 ``levels`` 等長，每層對應目標日的欄位值。
        alpha: 收縮強度（先驗的等效樣本數）。

    Returns:
        tuple: ``(pmf, classes)``。``classes`` 為**完整名目格點**，``pmf`` 形狀 ``(1, n)``。

    Raises:
        ValueError: ``levels`` 與 ``group_values`` 長度不符。
    """
    if len(levels) != len(group_values):
        raise ValueError(f"levels 有 {len(levels)} 層，group_values 有 {len(group_values)} 組")

    grid = decision.candidate_grid(target)
    index = {minute: i for i, minute in enumerate(grid)}
    # 從完整格點上的均勻分布起步，任何格點的機率都不會恰為 0。
    pmf = np.ones(len(grid)) / len(grid)

    for columns, values in zip(levels, group_values, strict=True):
        condition = pl.lit(True)
        for column, value in zip(columns, values, strict=True):
            condition = condition & (pl.col(column) == value)
        rows = history.filter(condition).drop_nulls(target)
        if rows.height == 0:
            continue

        counts = np.zeros(len(grid))
        # value_counts 只能呼叫一次：polars 不保證兩次呼叫的列順序相同，
        # 分開取「時刻」與「計數」會讓兩者錯配。
        for minute, count in rows[target].value_counts().iter_rows():
            position = index.get(int(minute))
            if position is not None:
                counts[position] = count
        observed = counts.sum()
        if observed == 0:
            continue
        pmf = (counts + alpha * pmf) / (observed + alpha)

    return (pmf / pmf.sum())[None, :], grid


def grid_pmf(times: list[int], target: str, prior_weight: float = 0.5) -> tuple[np.ndarray, list[int]]:
    """由一組觀察到的時刻建成**完整名目格點**上的機率分布。

    集成必須讓兩個分布落在同一個支撐集上才能相加。``prior_weight`` 是每個格點上的
    均勻先驗，避免樣本很少時出現硬零、等於直接否決那些時刻。

    Args:
        times: 觀察到的時刻（當日分鐘數），落在格點外者忽略。
        target: ``"t_day"`` 或 ``"t_night"``。
        prior_weight: 均勻先驗的等效樣本數。

    Returns:
        tuple: ``(pmf, classes)``，``pmf`` 形狀為 ``(1, n)``。
    """
    grid = decision.candidate_grid(target)
    index = {minute: i for i, minute in enumerate(grid)}
    counts = np.full(len(grid), prior_weight, dtype=float)
    for minute in times:
        position = index.get(int(minute))
        if position is not None:
            counts[position] += 1.0
    return (counts / counts.sum())[None, :], grid


def mix_pmf(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """凸組合兩個落在相同支撐集上的機率分布。

    集成必須在分布層級做：兩個決策時刻的平均未必是好時刻，而且會破壞貝氏決策的
    最適性。混合分布後重新做一次決策，才會得到新分布下的最適解。

    Args:
        first: 第一個分布，形狀 ``(1, n)``。
        second: 第二個分布，形狀須相同。
        weight: 第一個分布的權重，落在 [0, 1]。

    Returns:
        np.ndarray: 混合後的分布，已正規化。

    Raises:
        ValueError: 形狀不符或權重超出範圍。
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"權重須落在 [0, 1]，得到 {weight}")
    a = np.atleast_2d(np.asarray(first, dtype=float))
    b = np.atleast_2d(np.asarray(second, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f"兩個分布形狀不符：{a.shape} vs {b.shape}")
    mixed = weight * a + (1.0 - weight) * b
    return mixed / mixed.sum(axis=1, keepdims=True)
