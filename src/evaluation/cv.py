"""walk-forward 交叉驗證。

嚴禁隨機 K-Fold。每折完整模擬實戰：
    - 訓練集截止於日 T
    - 預測 T+1、T+2、T+3
    - 特徵計算只能使用 ≤ T 的資訊

至少 ``settings.CV_MIN_FOLDS`` 折，涵蓋不同季節。
另報告三個與提交日相近的子集：夏月末期、含週六、週四五六。

**洩漏防線的實作方式**：``run_cv`` 傳給 ``predict_fn`` 的 ``history`` 已經
先切到 ``<= origin``，預測函式拿不到之後的資料。這比在預測函式內部自律要可靠，
因為切分只在一個地方發生。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from config import settings
from src.evaluation.metrics import score_breakdown
from src.features.targets import TARGET_NAMES
from src.logging_setup import get_logger

logger = get_logger(__name__)

type PredictFn = Callable[[pl.DataFrame, tuple[date, ...]], pl.DataFrame]
"""預測函式簽章：``(history, target_dates) -> 每個目標日一列的 6 項預測值``。

``history`` 保證只含 ``date <= origin`` 的列。
"""


@dataclass(frozen=True, slots=True)
class Fold:
    """單一 walk-forward 折。

    Attributes:
        origin: 預測起點日 T，訓練集與特徵計算僅能使用 ≤ T 的資訊。
        target_dates: 目標日 T+1 ~ T+horizon。
    """

    origin: date
    target_dates: tuple[date, ...]


def make_folds(
    available_dates: pl.Series,
    horizon: int | None = None,
    n_folds: int | None = None,
    min_history_days: int | None = None,
) -> list[Fold]:
    """建立 walk-forward 折序列，起點均勻分布以涵蓋各季節。

    Args:
        available_dates: 有標籤的所有日期。
        horizon: 每折預測天數，None 時採 ``settings.PREDICT_HORIZON_DAYS``。
        n_folds: 折數，None 時採 ``settings.CV_MIN_FOLDS``。
        min_history_days: 起點日之前至少要有幾天歷史，None 時採
            ``settings.CV_MIN_HISTORY_DAYS``。預設值足以讓 lag-364 可用。

    Returns:
        list[Fold]: 依時間排序的折清單。

    Raises:
        ValueError: 可用的起點日不足以產生 ``n_folds`` 折。
    """
    horizon = horizon or settings.PREDICT_HORIZON_DAYS
    n_folds = n_folds or settings.CV_MIN_FOLDS
    min_history_days = min_history_days or settings.CV_MIN_HISTORY_DAYS

    dates = sorted(available_dates.unique().to_list())
    date_set = set(dates)
    first, last = dates[0], dates[-1]

    eligible = [
        d
        for d in dates
        if (d - first).days >= min_history_days
        and all(d + timedelta(days=h) in date_set for h in range(1, horizon + 1))
    ]
    if len(eligible) < n_folds:
        raise ValueError(
            f"可用起點日僅 {len(eligible)} 天，不足 {n_folds} 折"
            f"（資料 {first} ~ {last}，要求至少 {min_history_days} 天歷史）"
        )

    # 均勻取樣而非取最後 n 天：後者會讓所有折擠在同一季節。
    step = len(eligible) / n_folds
    origins = [eligible[int(i * step)] for i in range(n_folds)]

    folds = [
        Fold(
            origin=o,
            target_dates=tuple(o + timedelta(days=h) for h in range(1, horizon + 1)),
        )
        for o in origins
    ]
    logger.info(
        "建立 %d 折：起點 %s ~ %s，horizon = %d，涵蓋月份 %s",
        len(folds), folds[0].origin, folds[-1].origin, horizon,
        sorted({f.origin.month for f in folds}),
    )
    return folds


def run_cv(daily: pl.DataFrame, folds: list[Fold], predict_fn: PredictFn) -> pl.DataFrame:
    """在給定折上執行交叉驗證。

    Args:
        daily: 含目標標籤的完整每日表。
        folds: :func:`make_folds` 的輸出。所有候選模型共用同一組折，
            才能做配對比較。
        predict_fn: 見 :data:`PredictFn`。

    Returns:
        pl.DataFrame: 每折一列，含 ``origin``、各分項指標與 ``total_score``。
    """
    rows: list[dict[str, object]] = []
    for fold in folds:
        history = daily.filter(pl.col("date") <= fold.origin)
        truth = (
            daily.filter(pl.col("date").is_in(list(fold.target_dates)))
            .sort("date")
            .select(TARGET_NAMES)
        )
        pred = predict_fn(history, fold.target_dates).select(TARGET_NAMES)
        breakdown = score_breakdown(truth, pred)
        rows.append({"origin": fold.origin, **breakdown.as_dict()})
    return pl.DataFrame(rows)


def summarize_cv(fold_scores: pl.DataFrame) -> dict[str, float | date]:
    """彙總 CV 結果。

    須報告平均、標準差、**最壞折**，不可只看平均。

    Args:
        fold_scores: :func:`run_cv` 的輸出。

    Returns:
        dict: 折數、平均、標準差、標準誤、中位數、最壞折分數與其起點日。
    """
    scores = fold_scores["total_score"]
    worst = fold_scores.sort("total_score", descending=True).head(1)
    n = scores.len()
    return {
        "n_folds": n,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "stderr": float(scores.std() / (n**0.5)),
        "median": float(scores.median()),
        "worst": float(scores.max()),
        "worst_origin": worst["origin"][0],
    }


def special_late_summer_folds(folds: list[Fold]) -> list[Fold]:
    """專項驗證一：目標日落在夏月末期（接近 10/15 電價切換點）的折。"""
    windows = [
        (date.fromisoformat(a), date.fromisoformat(b))
        for a, b in settings.CV_SPECIAL_LATE_SUMMER
    ]
    return [
        f
        for f in folds
        if any(lo <= d <= hi for d in f.target_dates for lo, hi in windows)
    ]


def special_saturday_folds(folds: list[Fold]) -> list[Fold]:
    """專項驗證二：含週六的折。

    10/3 為週六，佔總分 1/3，但訓練樣本中週六僅佔 1/7。
    若週六誤差顯著較大，須專門處理。
    """
    return [f for f in folds if any(d.isoweekday() == 6 for d in f.target_dates)]


def special_thu_fri_sat_folds(folds: list[Fold]) -> list[Fold]:
    """專項驗證三：目標日為週四五六的折，完全複製目標期間的星期結構。

    目標期 2026/10/1(四)、10/2(五)、10/3(六)，即起點日為**週三**。
    """
    return [
        f
        for f in folds
        if [d.isoweekday() for d in f.target_dates]
        == [settings.CV_SPECIAL_START_WEEKDAY, 5, 6]
    ]


