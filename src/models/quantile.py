"""量值目標的分位數迴歸（LightGBM）。

四個量值目標（``p_day``／``p_night``／``ramp_up``／``ramp_down``）以評分函數逆推的
τ 作為分位數。令低估的邊際成本為 c_under、高估為 c_over，pinball loss 的最佳分位數
是 τ = c_under / (c_under + c_over)：

| 目標 | τ | 由來 |
|---|---|---|
| ``p_day``／``p_night`` | 0.625 | 低估 0.6×½ + 0.2 = 0.50、高估 0.30 |
| ``ramp_up`` | 0.70 | 低估 0.15 + 0.20 = 0.35、高估 0.15 |
| ``ramp_down`` | 0.50 | 無低估懲罰，對稱 |

若在比值空間建模（``y / base``），τ 不需修正：``base`` 在推論時是已知的正常數，
分位數對正的常數縮放可交換，Q_τ(base × ratio) = base × Q_τ(ratio)。
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from config import settings

TAU_BY_TARGET: dict[str, float] = {
    "p_day": settings.TAU_PEAK,
    "p_night": settings.TAU_PEAK,
    "ramp_up": settings.TAU_RAMP_UP,
    "ramp_down": settings.TAU_RAMP_DOWN,
}
"""各量值目標由評分函數逆推的最適分位數。"""


@dataclass(frozen=True, slots=True)
class QuantileModel:
    """訓練完成的分位數模型。

    Attributes:
        booster: LightGBM 模型。
        target: 目標名稱。
        tau: 使用的分位數。
        feature_names: 訓練時的特徵順序，推論時須一致。
        in_ratio_space: 目標是否為比值（True 時預測值須乘回 base）。
    """

    booster: lgb.Booster
    target: str
    tau: float
    feature_names: list[str]
    in_ratio_space: bool


def fit(
    features: pl.DataFrame,
    y: pl.Series,
    target: str,
    feature_names: list[str],
    in_ratio_space: bool = False,
    seed: int | None = None,
    num_rounds: int | None = None,
) -> QuantileModel:
    """訓練單一目標的分位數迴歸模型。

    Args:
        features: 特徵矩陣（僅訓練資料）。
        y: 目標值；``in_ratio_space`` 為真時應為比值。
        target: 目標名稱，用於查表取得 τ。
        feature_names: 特徵順序。
        in_ratio_space: 是否在比值空間建模。
        seed: 隨機種子，None 時採 ``settings.RANDOM_SEED``。
        num_rounds: 提升輪數，None 時採 ``settings.LGB_QUANTILE_NUM_ROUNDS``。

    Returns:
        QuantileModel: 訓練完成的模型。

    Raises:
        ValueError: 目標名稱不在 :data:`TAU_BY_TARGET` 中。
    """
    if target not in TAU_BY_TARGET:
        raise ValueError(f"{target} 不是量值目標，可用：{sorted(TAU_BY_TARGET)}")
    tau = TAU_BY_TARGET[target]

    params = dict(settings.LGB_QUANTILE_PARAMS)
    params["alpha"] = tau
    params["seed"] = settings.RANDOM_SEED if seed is None else seed

    dataset = lgb.Dataset(
        features.select(feature_names).to_numpy(),
        label=y.to_numpy(),
        feature_name=feature_names,
        free_raw_data=False,
    )
    booster = lgb.train(params, dataset, num_boost_round=num_rounds or settings.LGB_QUANTILE_NUM_ROUNDS)
    return QuantileModel(booster, target, tau, feature_names, in_ratio_space)


def predict(model: QuantileModel, features: pl.DataFrame, base: pl.Series | None = None) -> np.ndarray:
    """預測量值；比值空間時乘回 base 還原為絕對值。

    Raises:
        ValueError: 比值空間但未提供 ``base``，或 ``base`` 含非正值。
    """
    raw = np.asarray(model.booster.predict(features.select(model.feature_names).to_numpy()), dtype=float)
    if not model.in_ratio_space:
        return raw
    if base is None:
        raise ValueError(f"{model.target} 在比值空間建模，還原時必須提供 base")
    base_values = base.to_numpy().astype(float)
    if (base_values <= 0).any() or np.isnan(base_values).any():
        raise ValueError(f"{model.target} 的 base 含非正值或 NaN，無法還原")
    return raw * base_values
