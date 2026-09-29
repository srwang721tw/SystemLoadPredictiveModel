"""競賽評分函數 ``total_score``。

本模組是全專案的地基：所有模型選擇、調參、特徵篩選一律以 CV 上的
``total_score`` 為準，**不得使用 MAE / RMSE / MAPE**。
評分函數寫錯，後面全部白做，故由 ``tests/test_metrics.py`` 以手算案例逐項綁住。

公式（N 為預測天數，越低越好）：

    s_peak_mw   = (1/2N) · Σ [ |(p_day_true   − p_day_pred)   / p_day_true|
                             + |(p_night_true − p_night_pred) / p_night_true| ]

    s_peak_time = (1/2N) · Σ [ (|t_day_true   − t_day_pred|   / 10) ** 1.2
                             + (|t_night_true − t_night_pred| / 10) ** 1.2 ]

    s_ramp_up   = (1/N) · Σ |(ramp_up_true   − ramp_up_pred)   / ramp_up_true|
    s_ramp_down = (1/N) · Σ |(ramp_down_true − ramp_down_pred) / ramp_down_true|

    penalty_peak = (1/N) · Σ max(0, (p_day_true   − p_day_pred)   / p_day_true   × 0.2)
                 + (1/N) · Σ max(0, (p_night_true − p_night_pred) / p_night_true × 0.2)
    penalty_ramp = (1/N) · Σ max(0, (ramp_up_true − ramp_up_pred) / ramp_up_true × 0.2)
    s_under_penalty = penalty_peak + penalty_ramp

    total_score = 0.6·s_peak_mw + 0.15·s_peak_time
                + 0.15·s_ramp_up + 0.1·s_ramp_down + s_under_penalty

注意事項：
    - 時間誤差 ``|t_true − t_pred|`` 單位為**分鐘**，除以 10 換算為資料格數。
      本專案的時刻欄位一律以「當日分鐘數」儲存，故可直接相減。
    - ``ramp_down`` **無**低估懲罰。
    - ``ramp_down`` 為絕對值（正值），故分母恆正。

**各分項的數量級差異極大，權重不等於重要性。**
   ``s_peak_mw`` 是相對誤差（典型 0.05），``s_peak_time`` 是「差幾格」的
   1.2 次方（典型 3），兩者差約 60 倍。因此 0.15 權重的時刻項在實務上
   反而主導總分。:func:`contribution_breakdown` 專門用來把這件事量化出來。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from config import settings
from src.features.targets import TARGET_NAMES


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """評分明細，供 CV 報告逐項比對使用。

    Attributes:
        s_peak_mw: 尖峰負載量相對誤差項。
        s_peak_time: 尖峰時間誤差項。
        s_ramp_up: 爬升量相對誤差項。
        s_ramp_down: 下降量相對誤差項。
        s_under_penalty: 低估懲罰項。
        total_score: 加權總分，越低越好。
    """

    s_peak_mw: float
    s_peak_time: float
    s_ramp_up: float
    s_ramp_down: float
    s_under_penalty: float
    total_score: float

    def as_dict(self) -> dict[str, float]:
        """轉為 dict，供寫入 DataFrame。"""
        return asdict(self)


def _check(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> None:
    """檢查兩張表的列數與必要欄位。

    Raises:
        ValueError: 列數不符、欄位缺漏，或含 null。
    """
    if y_true.height != y_pred.height:
        raise ValueError(f"列數不符：y_true {y_true.height} 列、y_pred {y_pred.height} 列")
    if y_true.height == 0:
        raise ValueError("空的評分輸入")
    for name, frame in (("y_true", y_true), ("y_pred", y_pred)):
        missing = [c for c in TARGET_NAMES if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} 缺少欄位：{missing}")
        nulls = [c for c in TARGET_NAMES if frame[c].null_count()]
        if nulls:
            raise ValueError(f"{name} 的欄位含 null：{nulls}")


def _relative_error(y_true: pl.DataFrame, y_pred: pl.DataFrame, column: str) -> pl.Series:
    """``(y_true − y_pred) / y_true``。低估時為正。

    Raises:
        ValueError: 分母含 0（相對誤差無定義）。
    """
    denominator = y_true[column]
    if (denominator == 0).any():
        raise ValueError(f"{column} 的實際值含 0，相對誤差無定義")
    return (denominator - y_pred[column]) / denominator


def _time_error_grids(y_true: pl.DataFrame, y_pred: pl.DataFrame, column: str) -> pl.Series:
    """``(|t_true − t_pred| / 10) ** 1.2``，即以資料格數計的時間誤差懲罰。"""
    diff_minutes = (y_true[column] - y_pred[column]).abs()
    return (diff_minutes / settings.DATA_FREQ_MIN) ** settings.TIME_ERROR_EXPONENT


def s_peak_mw(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
    """尖峰負載量的平均絕對相對誤差（日、夜各半）。"""
    day = _relative_error(y_true, y_pred, "p_day").abs()
    night = _relative_error(y_true, y_pred, "p_night").abs()
    return float((day.mean() + night.mean()) / 2)


def s_peak_time(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
    """尖峰時間誤差項。時間以分鐘計，除以 10 換算為格數後取 1.2 次方。"""
    day = _time_error_grids(y_true, y_pred, "t_day")
    night = _time_error_grids(y_true, y_pred, "t_night")
    return float((day.mean() + night.mean()) / 2)


def s_ramp_up(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
    """爬升量的平均絕對相對誤差。"""
    return float(_relative_error(y_true, y_pred, "ramp_up").abs().mean())


def s_ramp_down(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
    """下降量的平均絕對相對誤差（實際值與預測值皆為正）。"""
    return float(_relative_error(y_true, y_pred, "ramp_down").abs().mean())


def s_under_penalty(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
    """低估懲罰。僅 p_day、p_night、ramp_up 三項適用，ramp_down 不適用。"""
    total = 0.0
    for column in ("p_day", "p_night", "ramp_up"):
        error = _relative_error(y_true, y_pred, column)
        total += float(
            (error * settings.PENALTY_COEF).clip(lower_bound=0.0).mean()
        )
    return total


def score_breakdown(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> ScoreBreakdown:
    """計算各分項指標與總分。

    Args:
        y_true: 實際值，須含 6 項目標欄位。
        y_pred: 預測值，欄位與列序須與 ``y_true`` 對齊。

    Returns:
        ScoreBreakdown: 五個分項與總分。
    """
    _check(y_true, y_pred)
    peak_mw = s_peak_mw(y_true, y_pred)
    peak_time = s_peak_time(y_true, y_pred)
    ramp_up = s_ramp_up(y_true, y_pred)
    ramp_down = s_ramp_down(y_true, y_pred)
    penalty = s_under_penalty(y_true, y_pred)
    total = (
        settings.W_PEAK_MW * peak_mw
        + settings.W_PEAK_TIME * peak_time
        + settings.W_RAMP_UP * ramp_up
        + settings.W_RAMP_DOWN * ramp_down
        + penalty
    )
    return ScoreBreakdown(peak_mw, peak_time, ramp_up, ramp_down, penalty, total)


def total_score(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> float:
    """計算加權總分。

    Args:
        y_true: 實際值。
        y_pred: 預測值。

    Returns:
        float: ``total_score``，越低越好。
    """
    return score_breakdown(y_true, y_pred).total_score


def contribution_breakdown(breakdown: ScoreBreakdown) -> pl.DataFrame:
    """把各分項對總分的**實際貢獻**攤開。

    存在的理由：權重（0.6 / 0.15 / 0.15 / 0.1）容易讓人誤以為尖峰負載量最重要，
    但各分項的數量級差異極大——``s_peak_mw`` 是相對誤差（典型 0.05），
    ``s_peak_time`` 是格數的 1.2 次方（典型 3）。加權後時刻項反而主導總分。
    調校投入的優先序應依本表，而非依權重。

    Args:
        breakdown: :func:`score_breakdown` 的輸出。

    Returns:
        pl.DataFrame: 每個分項一列，含原始值、權重、加權貢獻、佔總分比例，
            依貢獻由大到小排序。
    """
    rows = [
        ("s_peak_mw", breakdown.s_peak_mw, settings.W_PEAK_MW),
        ("s_peak_time", breakdown.s_peak_time, settings.W_PEAK_TIME),
        ("s_ramp_up", breakdown.s_ramp_up, settings.W_RAMP_UP),
        ("s_ramp_down", breakdown.s_ramp_down, settings.W_RAMP_DOWN),
        ("s_under_penalty", breakdown.s_under_penalty, 1.0),
    ]
    return (
        pl.DataFrame(
            {
                "component": [r[0] for r in rows],
                "raw": [r[1] for r in rows],
                "weight": [r[2] for r in rows],
                "contribution": [r[1] * r[2] for r in rows],
            }
        )
        .with_columns(
            (pl.col("contribution") / breakdown.total_score).alias("share")
        )
        .sort("contribution", descending=True)
    )
