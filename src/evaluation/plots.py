"""回測結果的圖與表：三天曲線比較、432 點 MAPE、逐窗分數、尖峰時刻誤差、尖峰負載散佈。

輸入都是回測紀錄裡的檔（``curves.csv``、``folds.csv``、``predictions.csv``），
由 ``notebooks/05_模型評估.ipynb`` 呼叫。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 無視窗環境；須在 pyplot 之前設定
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import polars as pl

from config import settings
from src.evaluation.timing_plots import use_cjk_font
from src.features.targets import TARGET_NAMES, format_hhmm, to_minutes
from src.models import curve

use_cjk_font()

ACTUAL_COLOUR = "#111111"
PREDICTED_COLOUR = "#c1272d"


def curve_mape(curves: pl.DataFrame, column: str = "predicted") -> pl.DataFrame:
    """每個回測窗的 432 點 MAPE（%）。

    Args:
        curves: 回測紀錄的 ``curves.csv``，含 ``origin, ts, actual`` 與 ``column``。
        column: 要評估的預測欄。

    Returns:
        pl.DataFrame: ``origin, mape``，一窗一列。
    """
    return (
        curves.group_by("origin")
        .agg(((pl.col(column) - pl.col("actual")).abs() / pl.col("actual")).mean().mul(100).alias("mape"))
        .sort("origin")
    )


def same_weekday_curves(clean: pl.DataFrame, curves: pl.DataFrame, n_weeks: int = 4) -> pl.DataFrame:
    """參考曲線：每個目標日取起點以前最近 ``n_weeks`` 個同星期日的逐點中位數。

    Args:
        clean: 10 分鐘序列，含 ``ts``、``Load_MW``。
        curves: 回測紀錄的 ``curves.csv``（決定要算哪些起點與目標日）。
        n_weeks: 回溯幾個同星期日。

    Returns:
        pl.DataFrame: ``curves`` 加上 ``same_weekday`` 欄。
    """
    series = clean.select(
        "ts", "Load_MW",
        pl.col("ts").dt.date().alias("date"),
        pl.col("ts").dt.weekday().alias("weekday"),
        (pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)).alias("mod"),
    )
    parts = []
    for (origin,), window in curves.group_by("origin", maintain_order=True):
        days = window.select(pl.col("ts").dt.date().unique().sort())["ts"].to_list()
        for day in days:
            history = series.filter((pl.col("date") <= origin) & (pl.col("weekday") == day.isoweekday()))
            keep = history["date"].unique().sort().tail(n_weeks)
            profile = (history.filter(pl.col("date").is_in(keep.to_list()))
                       .group_by("mod").agg(pl.col("Load_MW").median()).sort("mod"))
            parts.append(profile.select(
                pl.lit(origin).alias("origin"),
                (pl.lit(dt.datetime.combine(day, dt.time(0))) + pl.duration(minutes=pl.col("mod"))).alias("ts"),
                pl.col("Load_MW").alias("same_weekday"),
            ))
    reference = pl.concat(parts).with_columns(pl.col("ts").cast(curves.schema["ts"]))
    return curves.join(reference, on=["origin", "ts"], how="left")


def plot_window(curves: pl.DataFrame, origin: dt.date, output_path: Path) -> Path:
    """畫出一個回測窗的三天曲線：實際值與預測值，並標出兩者的日、夜尖峰落點。

    Args:
        curves: 回測紀錄的 ``curves.csv``。
        origin: 回測窗的起點日。
        output_path: 圖檔路徑。

    Returns:
        Path: 圖檔路徑。
    """
    window = curves.filter(pl.col("origin") == origin).sort("ts")
    days = window["ts"].dt.date().unique().sort().to_list()
    hours = np.arange(curve.POINTS_PER_DAY) * curve.STEP / 60
    figure, axes = plt.subplots(1, len(days), figsize=(6 * len(days), 4.6), sharey=True)
    for ax, day in zip(np.atleast_1d(axes), days):
        part = window.filter(pl.col("ts").dt.date() == day)
        actual, predicted = part["actual"].to_numpy(), part["predicted"].to_numpy()
        for start, end in ((settings.DAY_PEAK_START, settings.DAY_PEAK_END),
                           (settings.NIGHT_PEAK_START, settings.NIGHT_PEAK_END)):
            ax.axvspan(to_minutes(start) / 60, to_minutes(end) / 60, color="#f2e6c9", alpha=0.5, lw=0)
        ax.plot(hours, actual, color=ACTUAL_COLOUR, lw=2.0, label="實際值")
        ax.plot(hours, predicted, color=PREDICTED_COLOUR, lw=1.6, label="預測值")
        truth, guess = curve.verify_day(actual), curve.verify_day(predicted)
        lines = []
        for name, label in (("day", "日尖峰"), ("night", "夜尖峰")):
            t_true, t_pred = int(truth[f"t_{name}"]), int(guess[f"t_{name}"])
            ax.plot(t_true / 60, truth[f"p_{name}"], "o", color=ACTUAL_COLOUR, ms=9, mfc="none", mew=2)
            ax.plot(t_pred / 60, guess[f"p_{name}"], "x", color=PREDICTED_COLOUR, ms=10, mew=2.4)
            lines.append(f"{label} 實際 {format_hhmm(t_true)}／預測 {format_hhmm(t_pred)}"
                         f"（差 {abs(t_true - t_pred) // curve.STEP} 格）")
        ax.set_title(f"{day}（{'一二三四五六日'[day.weekday()]}）\n" + "\n".join(lines), fontsize=10)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlim(0, 24)
        ax.set_xlabel("時")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("負載（MW）")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    figure.suptitle(f"起點 {origin}：預測 {days[0]} ～ {days[-1]}（○ 實際尖峰，× 預測尖峰；色帶為尖峰窗口）")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_window_scores(folds: pl.DataFrame, season: list[dt.date], output_path: Path) -> Path:
    """逐窗 total_score，依起點排序，同季節窗另外上色。"""
    ordered = folds.sort("origin")
    colours = ["#c1272d" if o in set(season) else "#8a9bb0" for o in ordered["origin"].to_list()]
    figure, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(range(ordered.height), ordered["total_score"].to_numpy(), color=colours)
    step = max(1, ordered.height // 13)
    ax.set_xticks(range(0, ordered.height, step))
    ax.set_xticklabels([str(o) for o in ordered["origin"].to_list()[::step]], rotation=45, ha="right")
    mean_line = ax.axhline(ordered["total_score"].mean(), color="#333333", lw=1, ls="--", label="全體平均")
    ax.set_ylabel("total_score")
    ax.legend(handles=[mean_line, Patch(color="#c1272d", label="同季節窗（目標日 9/20–10/12）"),
                       Patch(color="#8a9bb0", label="其他窗")], loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


def timing_errors(predictions: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """逐窗逐日的尖峰時刻誤差（預測 − 實際，單位：格）。"""
    joined = predictions.join(truth.select("date", *TARGET_NAMES), on="date", suffix="_true")
    return joined.select(
        "origin", "date",
        ((pl.col("t_day") - pl.col("t_day_true")) / curve.STEP).cast(pl.Int32).alias("t_day"),
        ((pl.col("t_night") - pl.col("t_night_true")) / curve.STEP).cast(pl.Int32).alias("t_night"),
    )


def plot_timing_errors(errors: pl.DataFrame, season: list[dt.date], output_path: Path) -> Path:
    """``t_day``、``t_night`` 誤差格數的直方圖，全體與同季節並列。"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
    in_season = errors.filter(pl.col("origin").is_in(season))
    for ax, name in zip(axes, ("t_day", "t_night")):
        bins = np.arange(-24.5, 25.5, 1)
        ax.hist(errors[name].to_numpy(), bins=bins, color="#8a9bb0", density=True, label=f"全體（{errors.height} 天）")
        ax.hist(in_season[name].to_numpy(), bins=bins, histtype="step", lw=2, color="#c1272d", density=True,
                label=f"同季節（{in_season.height} 天）")
        ax.set_title(f"{name}：預測 − 實際（格，1 格 = 10 分鐘）")
        ax.axvline(0, color="#333333", lw=1)
        ax.legend()
        ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_peak_scatter(predictions: pl.DataFrame, truth: pl.DataFrame, output_path: Path) -> Path:
    """``p_day``、``p_night`` 的預測對實際散佈圖，附 45° 線。"""
    joined = predictions.join(truth.select("date", *TARGET_NAMES), on="date", suffix="_true")
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, name in zip(axes, ("p_day", "p_night")):
        x, y = joined[f"{name}_true"].to_numpy(), joined[name].to_numpy()
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.scatter(x, y, s=10, alpha=0.5, color="#2b6cb0")
        ax.plot([low, high], [low, high], color="#333333", lw=1)
        above = float((y > x).mean() * 100)
        ax.set_title(f"{name}：{above:.0f}% 的預測高於實際")
        ax.set_xlabel("實際（MW）")
        ax.set_ylabel("預測（MW）")
        ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path
