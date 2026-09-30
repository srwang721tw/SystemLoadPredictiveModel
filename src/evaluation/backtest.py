"""回測框架：所有實驗走同一套回測窗、同一個評分與同一份紀錄格式。

每個回測窗與比賽情境相同：起點日 T 的負載截止於 T 23:50，預測 T+1 ~ T+3
共 432 期，目標日的氣象只用預報（honest 模式，見 ``workflow.WEATHER_MODES``）。
以 ``total_score`` 為主要指標，並報告 5 個子項與各子集。

## 兩組回測窗

| 組 | 內容 | 用途 |
|---|---|---|
| ``tuning`` 調參組 | 現有 60 折 + 2025 同季節每日窗，共 78 窗 | 所有實驗比較、特徵選擇、超參數 |
| ``holdout`` 保留確認組 | 目標日 2026-07-01 ~ 09-30，每天一窗 | 選定最終設定後**只用一次** |

## 選模門檻（:func:`compare`）

候選要勝過參考，必須同時：

1. 全部調參窗的配對改善大於 ``settings.CV_STDERR_THRESHOLD`` 個標準誤
2. ``settings.BACKTEST_SELECTION_SUBSET``（同季節）沒有變差超過同樣的標準誤數

第 2 條防的是：全年平均的改善可能全部來自冬季，對 10 月的提交毫無幫助。

## 紀錄

每次回測寫到 ``paths.BACKTEST_DIR / <時間>_<label>_<group>/``：

- ``folds.csv``：逐窗 total_score 與 5 子項
- ``predictions.csv``：逐窗逐日的 6 目標預測
- ``summary.json``：manifest 雜湊、git commit、設定、各子集分數、執行時間、略過的窗

Accuweather／Windy 沒有發布時間欄，「只用作業時點前發布的版本」無法逐筆驗證；
honest 模式是目前能做到最接近的模擬（每個目標時刻只有一個版本），是下限估計。
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import polars as pl

from config import paths, settings
from src.evaluation import compare as compare_module
from src.evaluation import cv, metrics, subgroup
from src.features.targets import TARGET_NAMES
from src.logging_setup import get_logger

logger = get_logger(__name__)

GROUPS = ("tuning", "holdout")
SUBSCORES = ("s_peak_mw", "s_peak_time", "s_ramp_up", "s_ramp_down", "s_under_penalty")


def _fold(origin: dt.date) -> cv.Fold:
    return cv.Fold(origin=origin, target_dates=tuple(
        origin + dt.timedelta(days=h) for h in range(1, settings.PREDICT_HORIZON_DAYS + 1)))


def _daily_range(start: str, end: str) -> list[dt.date]:
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [first + dt.timedelta(days=k) for k in range((last - first).days + 1)]


def windows(group: str) -> list[cv.Fold]:
    """回傳一組回測窗，依起點日排序。

    Args:
        group: ``"tuning"`` 或 ``"holdout"``。

    Returns:
        list[cv.Fold]: 每個窗的起點與 3 個目標日。

    Raises:
        ValueError: 未知的組名。
    """
    if group == "tuning":
        origins = {dt.date.fromisoformat(d) for d in settings.BACKTEST_CV60_ORIGINS}
        origins |= set(_daily_range(*settings.BACKTEST_SEASON_ORIGINS))
    elif group == "holdout":
        first, last = (dt.date.fromisoformat(d) for d in settings.BACKTEST_HOLDOUT_TARGETS)
        horizon = settings.PREDICT_HORIZON_DAYS
        origins = set(_daily_range(
            (first - dt.timedelta(days=1)).isoformat(),
            (last - dt.timedelta(days=horizon)).isoformat()))
    else:
        raise ValueError(f"未知的回測組：{group!r}（支援 {GROUPS}）")
    return [_fold(o) for o in sorted(origins)]


def _holiday_dates() -> set[dt.date]:
    """連假、逐日判讀的放假日與颱風停班停課日。"""
    from src.data import external

    special = external.load_special_days().filter(pl.col("is_holiday") == 1)["date"]
    days = set(special.to_list())
    days |= set(external._load_day_list(paths.LEAVE_DAY_FILE, ("日期",), "放假日")["date"].to_list())
    days |= set(external.load_typhoon_days()["date"].to_list())
    return days


def subsets(folds: list[cv.Fold]) -> dict[str, list[dt.date]]:
    """各報告子集包含的窗（以起點日表示）。子集只篩選已算好的逐窗表，不重跑。

    - 同季節：起點落在 ``settings.BACKTEST_SEASON_ORIGINS``
    - 週四五六：目標日恰為週四、五、六（與 2026-10-01 ~ 03 相同）
    - 含週六：任一目標日為週六
    - 夏月末期：任一目標日落在 ``settings.CV_SPECIAL_LATE_SUMMER``
    - 含連假或停班：任一目標日為連假、放假日或颱風停班停課日

    Args:
        folds: 回測窗。

    Returns:
        dict: ``{子集名稱: 起點日清單}``。
    """
    season = set(_daily_range(*settings.BACKTEST_SEASON_ORIGINS))
    holidays = _holiday_dates()
    return {
        "同季節": [f.origin for f in folds if f.origin in season],
        "週四五六": [f.origin for f in cv.special_thu_fri_sat_folds(folds)],
        "含週六": [f.origin for f in cv.special_saturday_folds(folds)],
        "夏月末期": [f.origin for f in cv.special_late_summer_folds(folds)],
        "含連假或停班": [f.origin for f in folds if holidays & set(f.target_dates)],
    }


def _describe(table: pl.DataFrame) -> dict:
    """一組窗的分數摘要：平均、標準誤、最壞窗與 5 子項平均。"""
    if table.height == 0:
        return {"n": 0}
    summary = cv.summarize_cv(table) if table.height > 1 else {
        "mean": float(table["total_score"][0]), "stderr": None,
        "worst": float(table["total_score"][0]), "worst_origin": table["origin"][0]}
    return {
        "n": table.height,
        "total_score": summary["mean"],
        "stderr": summary["stderr"],
        "worst": summary["worst"],
        "worst_origin": str(summary["worst_origin"]),
        **{name: float(table[name].mean()) for name in SUBSCORES},
    }


def summarize(table: pl.DataFrame, folds: list[cv.Fold]) -> dict:
    """全體與各子集的分數摘要，外加全體的子項貢獻比例。

    Args:
        table: 逐窗分數（``folds.csv``）。
        folds: 這次實際跑的窗。

    Returns:
        dict: ``overall``、``subsets``、``contribution``。
    """
    breakdown = metrics.ScoreBreakdown(
        *[float(table[c].mean()) for c in (*SUBSCORES, "total_score")])
    return {
        "overall": _describe(table),
        "subsets": {
            name: _describe(table.filter(pl.col("origin").is_in(origins)))
            for name, origins in subsets(folds).items()
        },
        "contribution": metrics.contribution_breakdown(breakdown).to_dicts(),
    }


def _available(folds: list[cv.Fold], weather_mode: str, observed_lag_days: int):
    """把缺資料的窗挑出來：目標日沒有負載標籤，或（honest／forecast）沒有預報。

    Returns:
        tuple: ``(可跑的窗, [{"origin", "reason"}, ...])``。
    """
    from src.features import accuweather as accuweather_features

    labelled = set(pl.read_parquet(paths.TARGETS_FILE)["date"].to_list())
    forecast_days = (
        set(accuweather_features.build_forecast()["date"].to_list())
        if weather_mode in ("honest", "forecast") else None
    )
    usable, skipped = [], []
    for fold in folds:
        reasons = []
        if not set(fold.target_dates) <= labelled or fold.origin not in labelled:
            reasons.append("負載未涵蓋")
        needed = set(fold.target_dates) | {
            fold.origin - dt.timedelta(days=k) for k in range(observed_lag_days)}
        if forecast_days is not None and not needed <= forecast_days:
            reasons.append("Accuweather 未涵蓋")
        if reasons:
            skipped.append({"origin": str(fold.origin), "reason": "、".join(reasons)})
        else:
            usable.append(fold)
    if skipped:
        logger.warning("%d 個窗因資料未涵蓋而略過：%s", len(skipped), skipped)
    return usable, skipped


def _git() -> dict:
    """目前的 git commit 與追蹤中的檔案是否有未提交變更。"""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=paths.PROJECT_ROOT, capture_output=True,
                              text=True, encoding="utf-8", check=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _data_fingerprint() -> dict:
    """``data/manifest.json`` 的雜湊，以及目前的資料是否與它一致。"""
    from src.data import manifest

    recorded = json.loads(paths.MANIFEST_FILE.read_text(encoding="utf-8"))
    changes = [c["path"] for c in manifest.compare_manifests(recorded, manifest.build_manifest())
               if c["status"] != "unchanged"]
    if changes:
        logger.warning("目前的資料與 manifest 不一致（尚未接受）：%s", changes)
    return {"manifest_sha256": manifest.sha256(paths.MANIFEST_FILE),
            "data_matches_manifest": not changes, "files_changed": changes}


def _guard_holdout(confirm: bool, git: dict, label: str) -> None:
    """保留組只在選定最終設定後使用：沒有明確確認就拒跑，並留下使用紀錄。"""
    if not confirm:
        raise PermissionError(
            "保留確認組只能在選定最終設定後使用一次，不得用於任何調整。"
            "確定要跑請加上 confirm_holdout=True（CLI：--confirm-holdout）。")
    log = paths.BACKTEST_DIR / "holdout_usage.jsonl"
    previous = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
                ] if log.exists() else []
    same = [p for p in previous if p["commit"] == git["commit"]]
    if same:
        logger.warning("同一個 commit 已經跑過保留組 %d 次：%s",
                       len(same), [p["label"] for p in same])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"),
                                 "label": label, **git}, ensure_ascii=False) + "\n")


def parse_overrides(pairs: list[str]) -> dict:
    """把 CLI 的 ``名稱=值`` 轉成設定覆寫；值以 JSON 解析（失敗時當字串），串列轉成 tuple。

    Raises:
        ValueError: 格式不對，或 ``settings`` 沒有這個名稱（打錯字不可靜默忽略）。
    """
    overrides = {}
    for pair in pairs:
        name, sep, text = pair.partition("=")
        if not sep or not hasattr(settings, name):
            raise ValueError(f"無法解析的設定覆寫：{pair!r}（須為 settings 內既有的名稱=值）")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        overrides[name] = tuple(value) if isinstance(value, list) else value
    return overrides


def run(
    group: str,
    label: str,
    weather_mode: str = "honest",
    observed_lag_days: int = 0,
    confirm_holdout: bool = False,
    overrides: dict | None = None,
) -> Path:
    """跑一次回測並寫出紀錄。

    Args:
        group: ``"tuning"`` 或 ``"holdout"``。
        label: 這次實驗的名稱，會出現在目錄名與比較報告中。
        weather_mode: 見 ``workflow.WEATHER_MODES``；正式比較一律用 honest。
        observed_lag_days: 模擬 CODiS 觀測只到起點前第幾天（見 ``workflow.evaluate``）。
        confirm_holdout: 跑保留組時必須為 True。
        overrides: 實驗用的設定覆寫 ``{名稱: 值}``，只在這次回測期間生效，
            結束後還原，並記錄在 ``summary.json``。

    Returns:
        Path: 紀錄目錄。

    Raises:
        PermissionError: 跑保留組但未確認。
        ValueError: 沒有任何窗可跑。
    """
    from src import workflow

    started = time.perf_counter()
    git = _git()
    if group == "holdout":
        _guard_holdout(confirm_holdout, git, label)
    fingerprint = _data_fingerprint()

    overrides = overrides or {}
    unknown = [name for name in overrides if not hasattr(settings, name)]
    if unknown:
        raise ValueError(f"settings 沒有這些名稱：{unknown}")
    saved = {name: getattr(settings, name) for name in overrides}
    previous_end = settings.DATA_AVAILABLE_END
    try:
        for name, value in overrides.items():
            setattr(settings, name, value)
        if overrides:
            logger.warning("設定覆寫：%s", overrides)
        if group == "holdout":
            # 保留組的目標日在開發期截止日之後：暫時放寬截止日並重建中間檔，結束後復原。
            settings.DATA_AVAILABLE_END = settings.HOLDOUT_DATA_END
            workflow.build_processed()
        folds, skipped = _available(windows(group), weather_mode, observed_lag_days)
        if not folds:
            raise ValueError(f"{group} 沒有任何窗的資料齊全，無法回測：{skipped}")
        folder = paths.BACKTEST_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{label}_{group}"
        folder.mkdir(parents=True)
        result = workflow.evaluate(weather_mode, output_dir=folder, folds=folds,
                                   observed_lag_days=observed_lag_days)
        data_end = settings.DATA_AVAILABLE_END
    finally:
        for name, value in saved.items():
            setattr(settings, name, value)
        if settings.DATA_AVAILABLE_END != previous_end:
            settings.DATA_AVAILABLE_END = previous_end
            workflow.build_processed()

    # evaluate 的檔名帶模式名；紀錄目錄內只有一種模式，改成固定檔名。
    (folder / f"folds_{weather_mode}.csv").rename(folder / "folds.csv")
    (folder / f"predictions_{weather_mode}.csv").rename(folder / "predictions.csv")
    for name in ("curves", "raw_targets"):
        if (folder / f"{name}_{weather_mode}.csv").exists():
            (folder / f"{name}_{weather_mode}.csv").rename(folder / f"{name}.csv")
    summary = {
        "label": label,
        "group": group,
        "weather_mode": weather_mode,
        "observed_lag_days": observed_lag_days,
        "overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()},
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "git": git,
        **fingerprint,
        "data_available_end": data_end,
        "n_windows": len(folds),
        "skipped": skipped,
        **summarize(result["table"], folds),
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overall = summary["overall"]
    logger.warning("回測 [%s／%s] %d 窗：total_score %.5f（標準誤 %.5f），同季節 %.5f → %s",
                   label, group, len(folds), overall["total_score"], overall["stderr"],
                   summary["subsets"]["同季節"].get("total_score", float("nan")), folder.name)
    return folder


def recombine(base: Path, replacement: Path, horizons: tuple[int, ...], label: str) -> Path:
    """離線重組：``base`` 的預測中，第 ``horizons`` 天換成 ``replacement`` 的預測，再重新評分。

    每個目標日的曲線獨立合成、獨立評分，故「某幾天換成另一個變體」不必重跑。
    用於 Plan B 的情境（例如只有 D+3 缺預報：前兩天照常，第三天用備援）。

    Args:
        base: 基底紀錄（通常是參考）。
        replacement: 提供替換日預測的紀錄；兩者的窗必須相同。
        horizons: 要替換的第幾天（1 = 起點次日）。
        label: 新紀錄的名稱。

    Returns:
        Path: 新紀錄目錄（``summary.json`` 註明由哪兩份紀錄重組而來）。

    Raises:
        ValueError: 兩份紀錄的窗或逐日列不一致。
    """
    base_summary = _read_run(base)[0]
    table, predictions = _recombined(base, replacement, horizons)
    folder = paths.BACKTEST_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{label}_{base_summary['group']}"
    folder.mkdir(parents=True)
    table.write_csv(folder / "folds.csv")
    predictions.write_csv(folder / "predictions.csv")
    folds = [_fold(o) for o in table["origin"].to_list()]
    summary = {
        "label": label, "group": base_summary["group"], "git": _git(),
        "derived_from": {"base": base.name, "replacement": replacement.name,
                         "horizons": list(horizons)},
        "n_windows": table.height,
        **summarize(table, folds),
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return folder


def _recombined(base: Path, replacement: Path, horizons: tuple[int, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """:func:`recombine` 的計算本體（不寫檔）：回傳 ``(逐窗分數, 逐窗逐日預測)``。"""
    _, base_folds, base_pred = _read_run(base)
    _, _, other_pred = _read_run(replacement)
    key = ["origin", "date"]
    if base_pred.select(key).sort(key).to_dicts() != other_pred.select(key).sort(key).to_dicts():
        raise ValueError("兩份紀錄的窗或逐日列不一致，無法重組")
    horizon = (pl.col("date") - pl.col("origin")).dt.total_days()
    predictions = pl.concat([
        base_pred.filter(~horizon.is_in(list(horizons))),
        other_pred.filter(horizon.is_in(list(horizons))).select(base_pred.columns),
    ]).sort(key)

    truth = pl.read_parquet(paths.TARGETS_FILE).select("date", *TARGET_NAMES)
    rows = []
    for origin, part in predictions.group_by("origin", maintain_order=True):
        part = part.sort("date")
        actual = part.select("date").join(truth, on="date", how="left").select(TARGET_NAMES)
        rows.append({"origin": origin[0],
                     **metrics.score_breakdown(actual, part.select(TARGET_NAMES)).as_dict()})
    return pl.DataFrame(rows).select(base_folds.columns), predictions


def find_run(label: str, group: str = "tuning") -> Path:
    """回傳名稱為 ``label`` 的最新一份回測紀錄目錄。

    Raises:
        FileNotFoundError: 找不到這份紀錄（請先以 ``python main.py backtest --label`` 產生）。
    """
    found = sorted(f for f in paths.BACKTEST_DIR.glob(f"*_{label}_{group}")
                   if (f / "summary.json").exists())
    if not found:
        raise FileNotFoundError(f"找不到回測紀錄 {label}（{group}）")
    return found[-1]


def _paired_row(name: str, reference: pl.DataFrame, candidate: pl.DataFrame) -> dict:
    """候選 vs 參考在全體與同季節子集的配對差值（候選為 A，差值為正代表變差）。"""
    season = subsets([_fold(o) for o in reference["origin"].to_list()])[settings.BACKTEST_SELECTION_SUBSET]
    whole = compare_module.paired_compare(candidate, reference, name, "reference", "reference")
    part = compare_module.paired_compare(
        candidate.filter(pl.col("origin").is_in(season)),
        reference.filter(pl.col("origin").is_in(season)), name, "reference", "reference")
    return {"名稱": name, "total_score": whole.mean_a, "差值": whole.mean_diff,
            "標準誤倍數": whole.n_stderr, "同季節差值": part.mean_diff,
            "同季節標準誤倍數": part.n_stderr, "通過門檻": passes_gate(whole, part)}


def plan_b_table() -> pl.DataFrame:
    """預報缺漏時各種備援做法的代價（相對完整預報），全部由回測紀錄計算。

    需要的紀錄（``python main.py backtest`` 產生，設定覆寫見各列）：

    - ``current_best``：完整預報的參考
    - ``4a_N_noweather``：不用氣象（``ENABLE_TIER1_WEATHER=false``、``TIMING_TEMPERATURE_BINS=1``）
    - ``4b_S3_persistence``／``4b_S3_climatology``：三天都沒有預報，以最近觀測／同月氣候值補
    - ``4b_S4_persistence``／``4b_S4_climatology``：臺北三天都沒有預報

    「10/3 缺」「10/2–3 缺」由三天全缺的紀錄離線重組（只替換那幾天的預測）；
    每個目標日的曲線獨立合成、獨立評分，重組與直接執行的分數完全相同。
    """
    reference = find_run("current_best")
    ref_folds = _read_run(reference)[1]
    methods = {"不用氣象的模型": find_run("4a_N_noweather"),
               "最近一天的觀測": find_run("4b_S3_persistence"),
               "同月氣候值": find_run("4b_S3_climatology")}
    rows = []
    for scenario, horizons in (("10/3 缺", (3,)), ("10/2–3 缺", (2, 3)), ("三天全缺", (1, 2, 3))):
        for method, folder in methods.items():
            table = _recombined(reference, folder, horizons)[0]
            rows.append({"情境": scenario, **_paired_row(method, ref_folds, table)})
    for method, label in (("最近一天的觀測", "4b_S4_persistence"), ("同月氣候值", "4b_S4_climatology")):
        rows.append({"情境": "臺北三天缺",
                     **_paired_row(method, ref_folds, _read_run(find_run(label))[1])})
    return pl.DataFrame(rows).drop("通過門檻")


def seed_confirmation_table(pairs: list[tuple[int, str, str]]) -> pl.DataFrame:
    """多種子確認：每個種子各一對（參考, 候選）紀錄的配對比較，最後一列為各窗取種子平均後的比較。

    Args:
        pairs: ``[(種子, 參考紀錄名稱, 候選紀錄名稱), ...]``。

    Returns:
        pl.DataFrame: 每個種子一列加上「平均」一列。
    """
    references, candidates, rows = [], [], []
    for seed, ref_label, cand_label in pairs:
        reference = _read_run(find_run(ref_label))[1].sort("origin")
        candidate = _read_run(find_run(cand_label))[1].sort("origin")
        references.append(reference)
        candidates.append(candidate)
        rows.append({"種子": str(seed), "參考": float(reference["total_score"].mean()),
                     **_paired_row(cand_label, reference, candidate)})

    def mean_of(tables: list[pl.DataFrame]) -> pl.DataFrame:
        return tables[0].select("origin", pl.mean_horizontal([t["total_score"] for t in tables])
                                .alias("total_score"))

    average_ref, average_cand = mean_of(references), mean_of(candidates)
    rows.append({"種子": "平均", "參考": float(average_ref["total_score"].mean()),
                 **_paired_row("平均", average_ref, average_cand)})
    return pl.DataFrame(rows).rename({"total_score": "候選"}).drop("名稱")


def _read_run(folder: Path) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    folds = pl.read_csv(folder / "folds.csv", try_parse_dates=True)
    predictions = pl.read_csv(folder / "predictions.csv", try_parse_dates=True)
    return summary, folds, predictions


def window_losses(predictions: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """逐窗逐日、逐目標的損失；``date`` 欄換成「起點→目標日」鍵，因為各窗的目標日會重疊。"""
    ordered = predictions.sort("origin", "date")
    actual = ordered.select("origin", "date").join(truth, on="date", how="left")
    losses = subgroup.per_target_loss(actual.select("date", *TARGET_NAMES),
                                      ordered.select(TARGET_NAMES))
    key = ordered.select(pl.format("{}→{}", "origin", "date").alias("date"))
    return losses.with_columns(key["date"])


def passes_gate(overall: compare_module.PairedComparison,
                subset: compare_module.PairedComparison | None) -> bool:
    """選模門檻：候選（A）在全體顯著較好，且子集沒有顯著變差。

    Args:
        overall: 候選 vs 參考在全部窗上的配對比較（候選為 A）。
        subset: 同上，只看 ``settings.BACKTEST_SELECTION_SUBSET``；None 表示該子集沒有窗。

    Returns:
        bool: 是否通過。
    """
    threshold = settings.CV_STDERR_THRESHOLD
    better = overall.mean_diff < 0 and overall.n_stderr > threshold
    subset_worse = subset is not None and subset.mean_diff > 0 and subset.n_stderr > threshold
    return better and not subset_worse


def compare(reference: Path, candidate: Path) -> dict:
    """候選 vs 參考的配對比較，套用選模門檻，並寫出報告到候選的紀錄目錄。

    Args:
        reference: 參考（現行模型）的紀錄目錄。
        candidate: 候選的紀錄目錄。

    Returns:
        dict: 全體與子集的配對比較、是否通過門檻、逐目標拆解。

    Raises:
        ValueError: 兩次回測的窗不同——配對比較的前提被破壞。
    """
    ref_summary, ref_folds, ref_pred = _read_run(reference)
    cand_summary, cand_folds, cand_pred = _read_run(candidate)
    if sorted(ref_folds["origin"].to_list()) != sorted(cand_folds["origin"].to_list()):
        raise ValueError("兩次回測的窗不同，無法配對比較（請用同一組、同樣的資料）")
    ref_name, cand_name = ref_summary["label"], cand_summary["label"]

    def paired(origins: list[dt.date] | None):
        a, b = cand_folds, ref_folds
        if origins is not None:
            a = a.filter(pl.col("origin").is_in(origins))
            b = b.filter(pl.col("origin").is_in(origins))
        if a.height < 2:
            return None
        return compare_module.paired_compare(a, b, cand_name, ref_name, simpler=ref_name)

    folds = [_fold(o) for o in sorted(ref_folds["origin"].to_list())]
    groups = subsets(folds)
    overall = paired(None)
    per_subset = {name: paired(origins) for name, origins in groups.items()}
    passed = passes_gate(overall, per_subset[settings.BACKTEST_SELECTION_SUBSET])

    truth = pl.read_parquet(paths.TARGETS_FILE).select("date", *TARGET_NAMES)
    season = set(groups[settings.BACKTEST_SELECTION_SUBSET])
    ref_loss = window_losses(ref_pred, truth)
    cand_loss = window_losses(cand_pred, truth)
    membership = ref_loss.select("date").with_columns(
        pl.when(pl.col("date").str.slice(0, 10).str.to_date().is_in(list(season)))
        .then(pl.lit(settings.BACKTEST_SELECTION_SUBSET)).otherwise(pl.lit("其他"))
        .alias("子群"))
    per_target = subgroup.compare(ref_loss, cand_loss, membership)

    def as_dict(c):
        return None if c is None else {
            "n": c.n_folds, "reference": c.mean_b, "candidate": c.mean_a,
            "diff": c.mean_diff, "stderr": c.stderr, "n_stderr": c.n_stderr}

    result = {
        "reference": reference.name, "candidate": candidate.name,
        "overall": as_dict(overall),
        "subsets": {name: as_dict(c) for name, c in per_subset.items()},
        "passes_gate": passed,
        "per_target": per_target.to_dicts(),
    }
    lines = [
        f"# {cand_name} vs {ref_name}", "",
        f"- 參考：`{reference.name}`（commit {ref_summary['git']['commit'][:8]}）",
        f"- 候選：`{candidate.name}`（commit {cand_summary['git']['commit'][:8]}）",
        f"- 選模門檻：**{'通過' if passed else '未通過'}**（全體改善 > "
        f"{settings.CV_STDERR_THRESHOLD} 個標準誤，且{settings.BACKTEST_SELECTION_SUBSET}"
        "未顯著變差）", "",
        "| 子集 | 窗數 | 參考 | 候選 | 差值 | 標準誤 | 標準誤倍數 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, c in [("全體", overall), *per_subset.items()]:
        if c is not None:
            lines.append(f"| {name} | {c.n_folds} | {c.mean_b:.5f} | {c.mean_a:.5f} | "
                         f"{c.mean_diff:+.5f} | {c.stderr:.5f} | {c.n_stderr:.2f} |")
    lines += ["", "## 逐目標拆解（只列有改變的）", "", "```",
              subgroup.format_report(per_target), "```", ""]
    stem = f"comparison_vs_{reference.name}"
    (candidate / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    (candidate / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    logger.warning("%s", "\n".join(lines[:5 + 3 + len(per_subset) + 1]))
    return result


def experiment_table(pattern: str = "*") -> pl.DataFrame:
    """彙整 ``paths.BACKTEST_DIR`` 下的回測紀錄成一張表。

    Args:
        pattern: 紀錄目錄名稱的 glob（例如 ``"*_4a_*"``）。

    Returns:
        pl.DataFrame: 每份紀錄一列：label、設定覆寫、窗數、全體與同季節 total_score、
            5 子項、耗時；有配對比較時另附對照紀錄、差值、標準誤倍數與是否通過門檻。
    """
    rows = []
    for folder in sorted(paths.BACKTEST_DIR.glob(pattern)):
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        row = {
            "label": summary["label"],
            "覆寫": json.dumps(summary.get("overrides") or summary.get("derived_from") or {},
                              ensure_ascii=False),
            "窗數": summary["n_windows"],
            "total_score": summary["overall"]["total_score"],
            "同季節": summary["subsets"]["同季節"].get("total_score"),
            **{name: summary["overall"][name] for name in SUBSCORES},
            "耗時秒": summary.get("runtime_seconds"),
        }
        comparisons = sorted(folder.glob("comparison_vs_*.json"))
        if comparisons:
            result = json.loads(comparisons[-1].read_text(encoding="utf-8"))
            row |= {
                "對照": Path(result["reference"]).name,
                "差值": result["overall"]["diff"],
                "標準誤倍數": result["overall"]["n_stderr"],
                "同季節差值": result["subsets"]["同季節"]["diff"],
                "同季節標準誤倍數": result["subsets"]["同季節"]["n_stderr"],
                "通過門檻": result["passes_gate"],
            }
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)
