"""以主辦單位的規則，從 10 分鐘負載曲線重算每天 6 個量、5 個子項與 total_score。

用法::

    python organizer_score.py 預測.csv              # 印出每天 6 個量
    python organizer_score.py 預測.csv 實際.csv     # 另外印出 5 個子項與 total_score

這支程式**刻意只用 Python 標準函式庫，也不 import 專案的任何程式**：評分規則在這裡依
比賽說明重新寫一次，與專案內的推導（``src/features/targets.py``）和評分
（``src/evaluation/metrics.py``）各自獨立。兩邊算出相同的結果，才能確定合成曲線、
推導與評分都沒有寫錯。``python main.py verify-curves`` 與 ``run_submission.py``
都用它來複核。

規則：

- 日尖峰窗口 11:00–17:00（37 格）、夜尖峰窗口 17:10–21:00（24 格），皆含端點
- ``p_*`` 是窗口內的最大負載，``t_*`` 是它發生的時刻（當日分鐘數）；
  **同一天窗口內有兩個以上相同的最大值時，取最早發生的時刻**
- ``ramp_up``／``ramp_down`` 是當日 143 個相鄰 10 分鐘差分（不跨日）的最大值，
  以及最小值的絕對值
- 評分公式見 README 第 1 節
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

POINTS_PER_DAY = 144
STEP_MINUTES = 10
DAY_WINDOW = (11 * 60, 17 * 60)            # 11:00–17:00
NIGHT_WINDOW = (17 * 60 + 10, 21 * 60)     # 17:10–21:00
TIME_EXPONENT = 1.2
PENALTY_COEF = 0.2
WEIGHTS = {"s_peak_mw": 0.6, "s_peak_time": 0.15, "s_ramp_up": 0.15, "s_ramp_down": 0.1}
TARGETS = ("p_day", "t_day", "p_night", "t_night", "ramp_up", "ramp_down")
TIME_FORMATS = ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def _parse_time(text: str) -> dt.datetime:
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"無法解析的時間：{text!r}")


def read_curves(path: str | Path) -> dict[dt.date, list[float]]:
    """讀取 ``Date_Time,Load_MW`` 格式的檔案，依日期切成每天 144 點。

    Raises:
        ValueError: 某天不是完整的 00:00–23:50 共 144 筆，或時間重複。
    """
    rows: dict[dt.datetime, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            stamp = _parse_time(record["Date_Time"])
            if stamp in rows:
                raise ValueError(f"時間重複：{stamp}")
            rows[stamp] = float(record["Load_MW"])
    days: dict[dt.date, list[float]] = {}
    for day in sorted({stamp.date() for stamp in rows}):
        expected = [dt.datetime.combine(day, dt.time()) + dt.timedelta(minutes=STEP_MINUTES * k)
                    for k in range(POINTS_PER_DAY)]
        missing = [s for s in expected if s not in rows]
        if missing:
            raise ValueError(f"{day} 缺 {len(missing)} 筆（例如 {missing[0]:%H:%M}），每天必須剛好 144 筆")
        days[day] = [rows[s] for s in expected]
    return days


def _window_peak(values: list[float], window: tuple[int, int]) -> tuple[float, int]:
    """窗口內的最大值與時刻；由早到晚掃描，只有嚴格大於才更新，所以並列時取最早。"""
    start, end = window[0] // STEP_MINUTES, window[1] // STEP_MINUTES
    best_index = start
    for index in range(start + 1, end + 1):
        if values[index] > values[best_index]:
            best_index = index
    return values[best_index], best_index * STEP_MINUTES


def daily_targets(values: list[float]) -> dict[str, float]:
    """由一天 144 點算出 6 個量。時刻以當日分鐘數表示（例如 14:00 = 840）。"""
    if len(values) != POINTS_PER_DAY:
        raise ValueError(f"一天應有 {POINTS_PER_DAY} 點，得到 {len(values)}")
    p_day, t_day = _window_peak(values, DAY_WINDOW)
    p_night, t_night = _window_peak(values, NIGHT_WINDOW)
    diffs = [values[k + 1] - values[k] for k in range(POINTS_PER_DAY - 1)]
    return {"p_day": p_day, "t_day": t_day, "p_night": p_night, "t_night": t_night,
            "ramp_up": max(diffs), "ramp_down": abs(min(diffs))}


def score(actual: list[dict[str, float]], predicted: list[dict[str, float]]) -> dict[str, float]:
    """依主辦單位公式計算 5 個子項與 total_score（N = 天數，越低越好）。

    Args:
        actual: 每天的實際 6 個量，依日期排序。
        predicted: 每天的預測 6 個量，與 ``actual`` 逐日對齊。
    """
    if len(actual) != len(predicted) or not actual:
        raise ValueError(f"實際 {len(actual)} 天、預測 {len(predicted)} 天，必須相同且大於 0")
    n = len(actual)

    def rel(a: dict, p: dict, name: str) -> float:          # 低估時為正
        return (a[name] - p[name]) / a[name]

    s_peak_mw = sum(abs(rel(a, p, "p_day")) + abs(rel(a, p, "p_night"))
                    for a, p in zip(actual, predicted)) / (2 * n)
    s_peak_time = sum((abs(a["t_day"] - p["t_day"]) / STEP_MINUTES) ** TIME_EXPONENT
                      + (abs(a["t_night"] - p["t_night"]) / STEP_MINUTES) ** TIME_EXPONENT
                      for a, p in zip(actual, predicted)) / (2 * n)
    s_ramp_up = sum(abs(rel(a, p, "ramp_up")) for a, p in zip(actual, predicted)) / n
    s_ramp_down = sum(abs(rel(a, p, "ramp_down")) for a, p in zip(actual, predicted)) / n
    s_under_penalty = sum(
        sum(max(0.0, rel(a, p, name) * PENALTY_COEF) for a, p in zip(actual, predicted)) / n
        for name in ("p_day", "p_night", "ramp_up")
    )
    parts = {"s_peak_mw": s_peak_mw, "s_peak_time": s_peak_time,
             "s_ramp_up": s_ramp_up, "s_ramp_down": s_ramp_down}
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) + s_under_penalty
    return parts | {"s_under_penalty": s_under_penalty, "total_score": total}


def _hhmm(minutes: float) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__)
        return 1
    predicted_days = read_curves(argv[0])
    predicted = {day: daily_targets(values) for day, values in predicted_days.items()}
    print(f"{'日期':<12}{'p_day':>11}{'t_day':>7}{'p_night':>11}{'t_night':>9}{'ramp_up':>10}{'ramp_down':>11}")
    for day, t in predicted.items():
        print(f"{day!s:<12}{t['p_day']:>11.2f}{_hhmm(t['t_day']):>7}{t['p_night']:>11.2f}"
              f"{_hhmm(t['t_night']):>9}{t['ramp_up']:>10.2f}{t['ramp_down']:>11.2f}")
    if len(argv) == 2:
        actual_days = read_curves(argv[1])
        missing = sorted(set(predicted) - set(actual_days))
        if missing:
            print(f"實際值缺少這些日期：{missing}")
            return 1
        days = sorted(predicted)
        result = score([daily_targets(actual_days[d]) for d in days], [predicted[d] for d in days])
        print()
        for name, value in result.items():
            print(f"{name:<16}{value:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
