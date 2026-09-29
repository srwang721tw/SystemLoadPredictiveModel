"""特徵組裝入口：每日一列、以 ``date`` 為鍵。

特徵一律只能使用「預測起點日 T = 目標日 − horizon」及之前的資訊；目標日當天的
10 分鐘資料不可用（預測 10/1–10/3 時負載只到 9/30）。特徵依洩漏風險分三類：

1. **事前已知**（日曆、電價、天文、特殊日期、農曆）：目標日當天的值可用——
   10/1 是星期四、是夏月平日、日落 17:47，這些在 9/30 就已確定。
2. **自迴歸**（落後值、移動統計量）：只取到 T 為止，由 ``lag`` 模組的 asof join 保證。
3. **外生預報**（氣象、太陽光電）：歷史用觀測或預報，目標日只用預報。
"""

from __future__ import annotations

import polars as pl

from config import settings
from src.features import calendar, lag, rebase, solar
from src.features.targets import TARGET_NAMES
from src.logging_setup import get_logger

logger = get_logger(__name__)

NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "date",
    "month",
    "daytype",
    "price_daytype",
    "lunar",
    "solar_term",
    "n_points",
    "n_imputed",
    "has_imputed",
    "ramp_up_on_imputed",
    "ramp_down_on_imputed",
    "t_day_censored",
    "t_night_censored",
    "ramp_up_regime",
    "ramp_down_regime",
    "t_ramp_up",
    "t_ramp_down",
    "base_reliable",
    "any_ratio_outlier",
    "load_start",
    "load_min",
    "load_end",
    "typhoon_scope",
)
"""不可作為特徵的欄位：識別欄、字串欄、由當日標籤算出的診斷欄。

- ``t_ramp_up``／``t_ramp_down``、各種 ``*_regime``／``*_censored``：由當日標籤算出，
  放進特徵就是直接洩漏答案，只用於品質診斷。
- ``load_start``／``load_min``／``load_end``：目標日當天 00:00、全日最小、23:50 的負載，
  只供曲線合成使用。漏掉它們時 60 折分數會假性「改善」約 0.013，完全來自洩漏。
- ``lunar``／``solar_term``／``typhoon_scope``：字串，數值化後的衍生欄才是特徵。
- ``weekday`` 刻意不在清單內：星期是事前已知的日曆量。
"""


def build_features(
    daily: pl.DataFrame,
    calendar_df: pl.DataFrame,
    rules: dict,
    horizon: int,
    enable_weather: bool | None = None,
    pending_weather_dates: tuple = (),
) -> pl.DataFrame:
    """組裝指定 horizon 的完整特徵矩陣。

    Args:
        daily: 含目標標籤與 ``daytype`` 的每日表。
        calendar_df: 電價日曆表。
        rules: 電價時段規則。
        horizon: 預測步長 h ∈ {1, 2, 3}。
        enable_weather: 是否納入氣象特徵，None 時採 ``settings.ENABLE_TIER1_WEATHER``。
        pending_weather_dates: 允許暫缺氣象的目標日（其氣象稍後由預報覆寫），
            見 :func:`src.features.weather.add_weather_features`。

    Returns:
        pl.DataFrame: 每日一列的特徵矩陣，含 ``date``、所有特徵欄位、
            6 項目標、4 個 ``{target}_base`` 與 ``{target}_ratio``。
    """
    enable_weather = settings.ENABLE_TIER1_WEATHER if enable_weather is None else enable_weather

    out = calendar.add_calendar_features(daily, calendar_df, rules)
    out = calendar.add_trend_feature(out)
    out = calendar.add_typhoon_feature(out)
    out = solar.add_solar_features(out)
    out = lag.add_lag_features(out, horizon)
    out = rebase.add_base_columns(out, horizon)
    out = rebase.flag_outlier_ratios(out)

    if enable_weather:
        from src.features import weather as weather_features

        out = weather_features.add_weather_features(out, pending_dates=pending_weather_dates)

    if settings.ENABLE_WINDY:
        # 太陽光電預測：訓練與預測都用預報（同一來源），缺值交給 LightGBM。
        from src.features import windy as windy_features

        out = windy_features.add_windy_features(out)

    if settings.ENABLE_LUNAR_FEATURES:
        # 農曆與節氣：日曆表裡的字串欄轉成數值衍生欄；原始字串欄仍排除在外。
        from src.features import lunar as lunar_features

        out = lunar_features.add_lunar_features(out)

    if settings.ENABLE_SPECIAL_DAYS:
        from src.data import external as external_data, special_days

        out = out.join(external_data.load_special_days(), on="date", how="left")
        out = special_days.add_holiday_run_features(out)

    logger.info("horizon=%d 特徵矩陣：%d 列 × %d 欄，其中特徵 %d 個",
                horizon, out.height, out.width, len(feature_names(out)))
    return out


def feature_names(features: pl.DataFrame) -> list[str]:
    """取得特徵欄位名稱（排除識別欄、目標、比值、基準欄位與 ``EXCLUDED_FEATURES``）。

    順序即為餵入模型的順序。

    Args:
        features: :func:`build_features` 的輸出。

    Returns:
        list[str]: 特徵名稱，依欄位順序。
    """
    excluded = set(NON_FEATURE_COLUMNS) | set(TARGET_NAMES)
    excluded |= {f"{t}_base" for t in settings.MAGNITUDE_TARGETS}
    excluded |= {f"{t}_ratio" for t in settings.MAGNITUDE_TARGETS}
    excluded |= {f"{t}_ratio_outlier" for t in settings.MAGNITUDE_TARGETS}
    excluded |= set(settings.EXCLUDED_FEATURES)
    return [c for c in features.columns if c not in excluded]
