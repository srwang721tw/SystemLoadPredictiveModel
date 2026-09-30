"""以主辦單位計分程式（``organizer_score.py``）複核一份回測紀錄。

逐窗讀出 432 點的預測曲線與實際曲線，用獨立實作重新推導每天 6 個量、重新評分，
再與紀錄中的結果比對：

| 比對 | 不一致代表 |
|---|---|
| 預測曲線推導的 6 個量 vs ``predictions.csv``（合成後實現的值） | 曲線合成或推導有錯 |
| 預測曲線推導的 6 個量 vs ``raw_targets.csv``（模型原始輸出） | 只允許是合成時修補的 ramp |
| 實際曲線推導的 6 個量 vs ``targets.parquet`` | 標籤推導有錯 |
| 逐窗 5 個子項與 total_score vs ``folds.csv`` | 評分函數有錯 |
| 全體平均 vs ``summary.json`` | 同上 |
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

import organizer_score
from config import paths
from src.features.targets import TARGET_NAMES

TOLERANCE_MW = 1e-6
"""量值（MW）的容許誤差；時刻一律要求完全相同。"""

TOLERANCE_SYNTHESIS_MW = 1e-3
"""與模型原始輸出比對時的量值容許誤差，等於曲線合成器自身的容許誤差
（``curve.synthesize_day_checked``）。合成的浮點運算會留下約 1e-6 MW 的差異，
超過此值的 ramp 差異才是合成時刻意放寬的修補。"""

TOLERANCE_SCORE = 1e-9
"""子項與 total_score 的容許誤差。"""

RAMP_TARGETS = ("ramp_up", "ramp_down")
"""合成時可能被修補的量；其餘 4 個量不會被修補。"""

SUBSCORES = ("s_peak_mw", "s_peak_time", "s_ramp_up", "s_ramp_down", "s_under_penalty", "total_score")


def _differs(name: str, a: float, b: float, tolerance: float = TOLERANCE_MW) -> bool:
    if name.startswith("t_"):
        return int(round(a)) != int(round(b))
    return abs(a - b) > tolerance


def _rows(frame: pl.DataFrame) -> dict[tuple[dt.date, dt.date], dict]:
    return {(r["origin"], r["date"]): r for r in frame.iter_rows(named=True)}


def verify_run(folder: Path) -> dict:
    """複核一份回測紀錄。

    Args:
        folder: 回測紀錄目錄，須含 ``curves.csv``、``predictions.csv``、``folds.csv``、
            ``summary.json``；有 ``raw_targets.csv`` 時一併比對模型原始輸出。

    Returns:
        dict: ``ok``（是否全部通過）、``checks``（每項比對的筆數、不一致筆數、最大差異）、
            ``mismatches``（不一致的明細）、``repairs``（被修補的 ramp 與其分數代價）。
    """
    curves = pl.read_csv(folder / "curves.csv", try_parse_dates=True).sort("origin", "ts")
    realised = _rows(pl.read_csv(folder / "predictions.csv", try_parse_dates=True))
    raw_path = folder / "raw_targets.csv"
    raw = _rows(pl.read_csv(raw_path, try_parse_dates=True)) if raw_path.exists() else None
    labels = {r["date"]: r for r in pl.read_parquet(paths.TARGETS_FILE).iter_rows(named=True)}
    folds = {r["origin"]: r for r in pl.read_csv(folder / "folds.csv", try_parse_dates=True).iter_rows(named=True)}
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))

    counts = {name: [0, 0, 0.0] for name in ("合成實現的 6 個量", "模型原始的 6 個量", "實際值的 6 個量",
                                              "逐窗子項與 total_score", "全體平均")}
    mismatches, repairs, window_totals = [], [], []

    def record(check: str, key: str, name: str, expected: float, got: float, failed: bool) -> None:
        counts[check][0] += 1
        counts[check][2] = max(counts[check][2], abs(expected - got))
        if failed:
            counts[check][1] += 1
            mismatches.append({"比對": check, "窗／日": key, "項目": name, "紀錄": expected, "重算": got})

    for (origin,), window in curves.group_by("origin", maintain_order=True):
        window = window.with_columns(pl.col("ts").dt.date().alias("date"))
        days = window["date"].unique().sort().to_list()
        derived, actual, requested = [], [], []
        for day in days:
            part = window.filter(pl.col("date") == day)
            got = organizer_score.daily_targets(part["predicted"].to_list())
            truth = organizer_score.daily_targets(part["actual"].to_list())
            derived.append(got)
            actual.append(truth)
            key = f"{origin}→{day}"
            for name in TARGET_NAMES:
                record("合成實現的 6 個量", key, name, realised[(origin, day)][name], got[name],
                       _differs(name, realised[(origin, day)][name], got[name]))
                record("實際值的 6 個量", key, name, labels[day][name], truth[name],
                       _differs(name, labels[day][name], truth[name]))
            if raw is not None:
                wanted = raw[(origin, day)]
                requested.append({n: wanted[n] for n in TARGET_NAMES})
                for name in TARGET_NAMES:
                    changed = _differs(name, wanted[name], got[name], TOLERANCE_SYNTHESIS_MW)
                    if changed and name in RAMP_TARGETS:   # 合成時刻意放寬的 ramp，不算錯
                        counts["模型原始的 6 個量"][0] += 1
                        repairs.append({"窗／日": key, "項目": name, "模型原始": wanted[name], "合成後": got[name]})
                    else:
                        record("模型原始的 6 個量", key, name, wanted[name], got[name], changed)

        result = organizer_score.score(actual, derived)
        window_totals.append(result)
        for name in SUBSCORES:
            expected = folds[origin][name]
            record("逐窗子項與 total_score", str(origin), name, expected, result[name],
                   abs(expected - result[name]) > TOLERANCE_SCORE)
        if raw is not None and any(r["窗／日"].startswith(f"{origin}→") for r in repairs):
            cost = result["total_score"] - organizer_score.score(actual, requested)["total_score"]
            for item in repairs:
                if item["窗／日"].startswith(f"{origin}→"):
                    item["該窗 total_score 變化"] = cost

    for name in SUBSCORES:
        mean = sum(w[name] for w in window_totals) / len(window_totals)
        expected = summary["overall"][name]
        record("全體平均", "全體", name, expected, mean, abs(expected - mean) > TOLERANCE_SCORE)

    checks = pl.DataFrame([
        {"比對": name, "筆數": n, "不一致": bad, "最大差異": worst}
        for name, (n, bad, worst) in counts.items() if name != "模型原始的 6 個量" or raw is not None
    ])
    return {
        "ok": not mismatches,
        "checks": checks,
        "mismatches": pl.DataFrame(mismatches) if mismatches else None,
        "repairs": pl.DataFrame(repairs) if repairs else None,
        "n_windows": len(window_totals),
    }
