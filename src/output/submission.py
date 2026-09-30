"""提交檔的組裝、寫出與驗證。

**提交的是 10 分鐘瞬時負載曲線本身**（3 天 × 144 點 = 432 列），
不是 6 項尖峰特徵。主辦單位由我們提交的曲線推導 6 項特徵後才評分。

因此本模組最重要的函式是 :func:`validate_submission`——它把寫出的 CSV
**讀回來**，用計算真值標籤的同一段程式（``targets.compute_targets``）
重新推導 6 項目標，再與模型意圖的值比對。

不做這一步就會**靜默失分**：檔案格式完全正確、列數正確、沒有 NaN，
但推導出的目標與模型意圖不同，而我們永遠不會知道。
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from config import paths, settings
from src.features import targets as target_module
from src.logging_setup import get_logger
from src.models.curve import DayTargets

logger = get_logger(__name__)

FILENAME_PATTERN = re.compile(r"^(\d{7})_submission_V(\d+)\.csv$")
"""提交檔名格式：民國年 7 碼 + 當日流水號。"""


def roc_stamp(day: dt.date) -> str:
    """西元日期 → 民國年 7 碼字串。

    Args:
        day: 西元日期。

    Returns:
        str: 例如 2026-08-15 → ``"1150815"``。
    """
    return f"{day.year - settings.ROC_YEAR_OFFSET:03d}{day.month:02d}{day.day:02d}"


def next_version(outputs_dir: Path, stamp: str) -> int:
    """掃描輸出目錄，取得當日的下一個流水號。

    不覆蓋既有檔案，一律取現有最大值 + 1。

    Args:
        outputs_dir: 輸出目錄。
        stamp: 民國年 7 碼。

    Returns:
        int: 下一個版本號，自 1 起。
    """
    versions = [
        int(m.group(2))
        for path in outputs_dir.glob("*_submission_V*.csv")
        if (m := FILENAME_PATTERN.match(path.name)) and m.group(1) == stamp
    ]
    return max(versions, default=0) + 1


def build_submission(curves: dict[dt.date, np.ndarray]) -> pl.DataFrame:
    """把每日曲線組成提交用的長表。

    Args:
        curves: ``{日期: 長度 144 的負載序列}``。

    Returns:
        pl.DataFrame: 欄位 ``Date_Time``（字串）與 ``Load_MW``，依時間排序。

    Raises:
        ValueError: 曲線長度不正確、日期不連續，或含非有限值。
    """
    days = sorted(curves)
    for day in days:
        values = curves[day]
        if values.shape != (settings.POINTS_PER_DAY,):
            raise ValueError(f"{day} 的曲線長度應為 {settings.POINTS_PER_DAY}，得到 {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{day} 的曲線含 NaN 或 Inf")
        if (values <= 0).any():
            raise ValueError(f"{day} 的曲線含非正值")

    for earlier, later in zip(days, days[1:], strict=False):
        if (later - earlier).days != 1:
            raise ValueError(f"日期不連續：{earlier} 之後是 {later}")

    rows = []
    for day in days:
        for index, value in enumerate(curves[day]):
            moment = dt.datetime.combine(day, dt.time()) + dt.timedelta(
                minutes=settings.DATA_FREQ_MIN * index
            )
            rows.append(
                {
                    "Date_Time": moment.strftime(settings.SUBMISSION_DATETIME_FORMAT),
                    "Load_MW": float(value),
                }
            )
    return pl.DataFrame(rows).select(settings.SUBMISSION_COLUMNS)


def write_submission(
    frame: pl.DataFrame, outputs_dir: Path | None = None, today: dt.date | None = None
) -> Path:
    """寫出提交檔，檔名為民國年 7 碼 + 流水號，並另存一份 ``submission_latest.csv``。

    Args:
        frame: :func:`build_submission` 的輸出。
        outputs_dir: 輸出目錄，None 時採 ``paths.SUBMISSION_DIR``。
        today: 產檔日期，None 時採今日。

    Returns:
        Path: 寫出的檔案路徑。

    Raises:
        FileExistsError: 目標檔案已存在（不應發生，流水號已避開）。
    """
    outputs_dir = outputs_dir or paths.SUBMISSION_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stamp = roc_stamp(today or dt.date.today())
    path = outputs_dir / f"{stamp}_submission_V{next_version(outputs_dir, stamp)}.csv"
    if path.exists():
        raise FileExistsError(f"嚴禁覆蓋既有提交檔：{path}")
    frame.write_csv(path)
    shutil.copyfile(path, outputs_dir / settings.SUBMISSION_LATEST_NAME)
    logger.info("提交檔已寫出：%s（%d 列），並另存 %s", paths.relative(path), frame.height,
                settings.SUBMISSION_LATEST_NAME)
    return path


def _read_back(path: Path) -> pl.DataFrame:
    """讀回提交檔並還原為 ``compute_targets`` 可用的格式。"""
    raw = pl.read_csv(path)
    return raw.select(
        pl.col("Date_Time")
        .str.strptime(pl.Datetime, settings.SUBMISSION_DATETIME_FORMAT)
        .alias("ts"),
        pl.col("Load_MW"),
        pl.lit(False).alias("is_imputed"),
    )


def targets_of(path: Path) -> pl.DataFrame:
    """讀回提交檔，以 ``compute_targets`` 推導每日 6 項目標（與主辦單位的評分方式相同）。"""
    return target_module.compute_targets(_read_back(path)).sort("date")


def validate_submission(
    path: Path, intended: dict[dt.date, DayTargets], tolerance: float = 1e-3
) -> pl.DataFrame:
    """讀回提交檔，重新推導 6 項目標，與模型意圖比對。

    這是唯一能保證「合成 → 寫檔 → 讀回」全程無損的檢查。
    刻意使用 ``targets.compute_targets``——計算真值標籤的同一段程式，
    也就是主辦單位會做的事。

    Args:
        path: 提交檔路徑。
        intended: ``{日期: 模型意圖實現的 6 項目標}``。
        tolerance: 量值容許誤差（MW）。時刻要求完全相等。

    Returns:
        pl.DataFrame: 逐日逐項的對照表，供 log 記錄。

    Raises:
        ValueError: 列數、欄位、時間連續性或推導結果有任一不符。
    """
    raw = pl.read_csv(path)
    if list(raw.columns) != list(settings.SUBMISSION_COLUMNS):
        raise ValueError(f"欄位不符：預期 {settings.SUBMISSION_COLUMNS}，得到 {tuple(raw.columns)}")
    if raw.height != settings.SUBMISSION_N_ROWS:
        raise ValueError(f"列數應為 {settings.SUBMISSION_N_ROWS}，得到 {raw.height}")

    frame = _read_back(path)
    gaps = frame["ts"].diff().drop_nulls().unique().to_list()
    expected_gap = dt.timedelta(minutes=settings.DATA_FREQ_MIN)
    if gaps != [expected_gap]:
        raise ValueError(f"時間不連續，出現的間隔：{gaps}")
    if not np.isfinite(frame["Load_MW"].to_numpy()).all():
        raise ValueError("Load_MW 含 NaN 或 Inf")

    extracted = target_module.compute_targets(frame).sort("date")

    rows, problems = [], []
    for day, wanted in sorted(intended.items()):
        got = extracted.filter(pl.col("date") == day)
        if got.height != 1:
            raise ValueError(f"讀回的檔案中找不到 {day}")
        got = got.to_dicts()[0]
        for name in target_module.TARGET_NAMES:
            want = float(getattr(wanted, name))
            have = float(got[name])
            ok = have == want if name.startswith("t_") else abs(have - want) <= tolerance
            rows.append({"date": day, "target": name, "intended": want, "extracted": have, "ok": ok})
            if not ok:
                problems.append(f"{day} {name}: 意圖 {want}、讀回 {have}")

    if problems:
        raise ValueError("提交檔讀回驗證失敗：\n  " + "\n  ".join(problems))
    logger.info("讀回驗證通過：%d 天 × 6 項目標全部相符", len(intended))
    return pl.DataFrame(rows)


def organizer_check(
    path: Path, intended: dict[dt.date, DayTargets], requested: dict[dt.date, DayTargets]
) -> list[str]:
    """用獨立的主辦單位計分程式（``organizer_score.py``）讀提交檔，重算每天 6 個量。

    與 :func:`validate_submission` 互為備援：兩者分別用獨立的程式推導，都相符才算通過。

    Args:
        path: 提交檔路徑。
        intended: ``{日期: 合成後實現的 6 個量}``，必須完全相符。
        requested: ``{日期: 模型原始預測的 6 個量}``。合成時只可能放寬 ramp，差異逐筆回報。

    Returns:
        list[str]: 與模型原始預測不同的 ramp（日期、項目、原值 → 合成後），沒有則為空。

    Raises:
        ValueError: 與合成後實現的值不符，或時刻、尖峰負載與模型原始預測不同。
    """
    import organizer_score

    derived = organizer_score.read_curves(path)
    problems, repairs = [], []
    for day in sorted(intended):
        if day not in derived:
            raise ValueError(f"主辦單位計分程式讀不到 {day}")
        got = organizer_score.daily_targets(derived[day])
        for name in target_module.TARGET_NAMES:
            have = float(got[name])
            for label, source in (("合成後", intended), ("模型原始", requested)):
                want = float(getattr(source[day], name))
                # 合成的浮點運算會留下約 1e-6 MW 的差異；與模型原始值比對時採合成器自身的容許誤差。
                tolerance = 1e-6 if label == "合成後" else 1e-3
                same = have == want if name.startswith("t_") else abs(have - want) <= tolerance
                if same:
                    continue
                if label == "模型原始" and name in ("ramp_up", "ramp_down"):
                    repairs.append(f"{day} {name} {want:.1f} → {have:.1f}")
                else:
                    problems.append(f"{day} {name}：{label} {want}、主辦單位計分程式 {have}")
    if problems:
        raise ValueError("主辦單位計分程式複核失敗：\n  " + "\n  ".join(problems))
    logger.info("主辦單位計分程式複核通過：%d 天 × 6 個量", len(intended))
    return repairs


def check_window_and_range(
    path: Path, days: list[dt.date], history: pl.DataFrame
) -> dict[str, float]:
    """提交檔的時間戳起訖與數值範圍檢查。

    - 第一筆必須是 ``days[0]`` 00:00、最後一筆是 ``days[-1]`` 23:50
    - 數值須落在近 ``SUBMISSION_PLAUSIBLE_DAYS`` 天實際負載的合理範圍內

    Args:
        path: 提交檔。
        days: 目標日。
        history: 10 分鐘負載序列（含 ``ts``、``Load_MW``），只取預測起點以前。

    Returns:
        dict: 提交檔的最小值、最大值與合理範圍上下限。

    Raises:
        ValueError: 起訖時間不對，或數值超出合理範圍。
    """
    frame = _read_back(path)
    first = dt.datetime.combine(days[0], dt.time())
    last = dt.datetime.combine(days[-1], dt.time(23, 50))
    if frame["ts"].min() != first or frame["ts"].max() != last:
        raise ValueError(
            f"時間範圍應為 {first} ~ {last}，實為 {frame['ts'].min()} ~ {frame['ts'].max()}"
        )

    start = first - dt.timedelta(days=settings.SUBMISSION_PLAUSIBLE_DAYS)
    recent = history.filter((pl.col("ts") >= start) & (pl.col("ts") < first))["Load_MW"]
    low = float(recent.min()) * (1 - settings.SUBMISSION_PLAUSIBLE_MARGIN)
    high = float(recent.max()) * (1 + settings.SUBMISSION_PLAUSIBLE_MARGIN)
    values = frame["Load_MW"]
    result = {"min": float(values.min()), "max": float(values.max()), "low": low, "high": high}
    if result["min"] < low or result["max"] > high:
        raise ValueError(
            f"提交檔數值 {result['min']:.0f} ~ {result['max']:.0f} 超出合理範圍 "
            f"{low:.0f} ~ {high:.0f}（近 {settings.SUBMISSION_PLAUSIBLE_DAYS} 天實際負載 ±"
            f"{settings.SUBMISSION_PLAUSIBLE_MARGIN:.0%}）"
        )
    logger.info("時間範圍 %s ~ %s、數值 %.0f ~ %.0f（合理範圍 %.0f ~ %.0f）",
                first, last, result["min"], result["max"], low, high)
    return result

