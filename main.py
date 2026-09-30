"""開發與驗證用的指令入口。

比賽當天產生提交檔請用 ``python run_submission.py``；本檔提供準備資料、回測與演練：

    python main.py check              # 環境、設定與資料是否到位
    python main.py weather            # 抓取／補齊 CODiS 逐小時觀測（可續跑）
    python main.py special-days       # 由 config/特殊日期區間.csv 展開每日旗標
    python main.py evaluate           # 60 折評估（--weather-mode honest／observed／forecast）
    python main.py backtest --label X # 回測框架：調參組或保留組，寫出紀錄
    python main.py backtest-compare --reference A --candidate B   # 兩份回測紀錄配對比較
    python main.py rehearse --data-end 2026-06-27                  # 比賽當天流程演練並評分
    python main.py verify-curves [--label notebook_04]             # 以主辦單位計分程式複核回測紀錄

所有參數由 ``config/`` 驅動。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import polars as pl

from config import paths, settings
from src import workflow
from src.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def cmd_check(args: argparse.Namespace) -> int:
    """檢查目錄結構、主資料與外部資料是否到位，只回報不修改。

    Returns:
        int: 0 表示所有必要檔案都在；1 表示有缺。
    """
    paths.ensure_directories()
    logger.info("資料截止日：%s；預測 %s 起 %d 天",
                settings.DATA_AVAILABLE_END, settings.PREDICT_START, settings.PREDICT_HORIZON_DAYS)
    missing = paths.missing_input_files()
    for name, path in paths.INPUT_FILES.items():
        logger.info("[%s] %s：%s", "缺" if name in missing else "有", name, paths.relative(path))
    if "Accuweather 年度檔" in missing:
        logger.info("[缺] Accuweather 年度檔：%s", paths.relative(missing["Accuweather 年度檔"]))
    if missing:
        logger.error("缺少 %d 個必要檔案：%s", len(missing), list(missing))
        return 1
    return 0


def cmd_weather(args: argparse.Namespace) -> int:
    """抓取 CODiS 五站逐小時觀測，範圍為 ``TRAIN_START`` ~ 資料截止日。

    **可續跑**：已抓過的（日期, 測站）會自動跳過，中斷後重跑不會重來；
    比賽當天執行一次即可補上最新的日子。CODiS 每天中午後才更新到前一日，
    抓不到的最後一天由 ``run_submission.py`` 以校正後的預報補上。

    Returns:
        int: 0 表示流程完成。
    """
    from src.data import checks, weather

    start = dt.date.fromisoformat(settings.TRAIN_START)
    end = checks.cutoff_date()
    logger.info("氣象抓取範圍：%s ~ %s", start, end)
    cleaned, report = weather.clean_observations(weather.fetch_range(start, end))
    summary = (
        cleaned.with_columns(pl.col("Date").dt.date().alias("day"))
        .group_by("stn_ID").agg(
            pl.col("day").n_unique().alias("天數"),
            pl.col("day").max().alias("最後一天"),
            pl.col("AirTemperature_Instantaneous").null_count().alias("氣溫缺值"),
        ).sort("stn_ID")
    )
    logger.warning("各測站涵蓋：\n%s\n特殊值轉換：\n%s", summary, report)
    return 0


def cmd_special_days(args: argparse.Namespace) -> int:
    """由 ``config/特殊日期區間.csv`` 展開每日 0/1 旗標，寫到 ``data/processed/特殊日期.csv``。

    ``run_submission.py`` 與 ``workflow.build_processed`` 也會自動重建；
    這個指令供修改區間表後單獨檢查結果。

    Returns:
        int: 0 表示流程完成。
    """
    from src.data import special_days

    out = special_days.build()
    logger.info("各類別天數：%s", {c: int(out[c].sum()) for c in special_days.CATEGORIES})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """對現行模型跑 60 折評估（6 目標 → 432 點曲線 → 由曲線重新推導 6 目標後評分）。

    逐折分數與逐日預測寫到 ``output/evaluation/``，供回歸測試比對。

    Returns:
        int: 0 表示流程完成。
    """
    workflow.build_processed()
    workflow.evaluate(args.weather_mode)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """跑一組回測窗並寫出紀錄，可選擇與參考紀錄配對比較。

    Returns:
        int: 0 表示流程完成；比較時未通過選模門檻仍回傳 0（結果寫在比較報告裡）。
    """
    from src.evaluation import backtest

    if not args.label:
        logger.error("請以 --label 指定這次回測的名稱")
        return 1
    folder = backtest.run(args.group, args.label, args.weather_mode,
                          args.observed_lag_days, args.confirm_holdout,
                          backtest.parse_overrides(args.set or []))
    if args.compare_to:
        backtest.compare(Path(args.compare_to), folder)
    return 0


def cmd_backtest_compare(args: argparse.Namespace) -> int:
    """比較兩份已存在的回測紀錄（``--reference`` 參考、``--candidate`` 候選）。"""
    from src.evaluation import backtest

    if not (args.reference and args.candidate):
        logger.error("請以 --reference 與 --candidate 指定兩份回測紀錄目錄")
        return 1
    backtest.compare(Path(args.reference), Path(args.candidate))
    return 0


def cmd_rehearse(args: argparse.Namespace) -> int:
    """比賽當天流程演練：把資料截止日暫時設為 ``--data-end``，完整執行 ``run_submission``，
    再以實際負載為產出的提交檔評分。

    例如 ``--data-end 2026-06-27`` 會預測 6/28–6/30，而這三天的實際值都在手上。
    演練不寫入 manifest；結束後以原本的截止日重建中間檔。

    Returns:
        int: 0 表示提交檔已產出並完成評分。
    """
    import run_submission
    from src.evaluation import metrics
    from src.features.targets import TARGET_NAMES
    from src.output import submission

    if not args.data_end:
        logger.error("請以 --data-end YYYY-MM-DD 指定演練的資料截止日")
        return 1
    started = time.perf_counter()
    try:
        with workflow.temporary_settings(DATA_AVAILABLE_END=args.data_end):
            code = run_submission.main(accept=False)
    finally:
        workflow.build_processed()
    seconds = time.perf_counter() - started
    if code != 0:
        logger.error("演練中止（%.1f 秒）", seconds)
        return code

    predicted = submission.targets_of(paths.SUBMISSION_DIR / settings.SUBMISSION_LATEST_NAME)
    truth = pl.read_parquet(paths.TARGETS_FILE).filter(
        pl.col("date").is_in(predicted["date"].to_list())).sort("date")
    if truth.height != predicted.height:
        logger.warning("演練的目標日沒有完整的實際值，無法評分（%.1f 秒）", seconds)
        return 0
    breakdown = metrics.score_breakdown(truth.select(TARGET_NAMES), predicted.select(TARGET_NAMES))
    logger.warning("演練 %s ~ %s：total_score %.5f（總耗時 %.1f 秒）\n%s",
                   truth["date"].min(), truth["date"].max(), breakdown.total_score, seconds,
                   metrics.contribution_breakdown(breakdown))

    # 以獨立的主辦單位計分程式，從提交檔的 432 點與實際的 10 分鐘負載再算一次。
    import organizer_score

    submitted = organizer_score.read_curves(paths.SUBMISSION_DIR / settings.SUBMISSION_LATEST_NAME)
    clean = pl.read_parquet(paths.CLEAN_LOAD_FILE).sort("ts")
    actual = [organizer_score.daily_targets(
        clean.filter(pl.col("ts").dt.date() == day)["Load_MW"].to_list()) for day in submitted]
    official = organizer_score.score(actual, [organizer_score.daily_targets(v) for v in submitted.values()])
    same = abs(official["total_score"] - breakdown.total_score) < 1e-9
    logger.warning("主辦單位計分程式重算：total_score %.5f（%s）", official["total_score"],
                   "與上方一致" if same else "與上方不一致")
    return 0 if same else 1


def cmd_verify_curves(args: argparse.Namespace) -> int:
    """以主辦單位計分程式（``organizer_score.py``）複核一份回測紀錄的 432 點曲線與分數。

    預設複核 notebook 04 最新的一份紀錄；``--label`` 可指定其他紀錄。

    Returns:
        int: 0 表示全部一致；1 表示有不一致（明細寫在 log）。
    """
    from src.evaluation import backtest, verify

    folder = backtest.find_run(args.label or "notebook_04", args.group)
    if not (folder / "curves.csv").exists():
        logger.error("%s 沒有 curves.csv，請重跑這份回測", folder.name)
        return 1
    result = verify.verify_run(folder)
    with pl.Config(tbl_rows=50, tbl_cols=10, fmt_str_lengths=40):
        logger.warning("複核 %s（%d 窗）：%s\n%s", folder.name, result["n_windows"],
                       "全部一致" if result["ok"] else "有不一致", result["checks"])
        if result["repairs"] is not None:
            logger.warning("合成時修補的 ramp（%d 筆；其餘 4 個量與模型原始輸出完全相同）：\n%s",
                           result["repairs"].height, result["repairs"])
        if result["mismatches"] is not None:
            logger.error("不一致明細：\n%s", result["mismatches"])
    return 0 if result["ok"] else 1


COMMANDS = {
    "check": cmd_check,
    "weather": cmd_weather,
    "special-days": cmd_special_days,
    "evaluate": cmd_evaluate,
    "backtest": cmd_backtest,
    "backtest-compare": cmd_backtest_compare,
    "rehearse": cmd_rehearse,
    "verify-curves": cmd_verify_curves,
}
"""指令 → 處理函式。新增指令只需在此登記一處（argparse 的選項也由此產生）。"""


def build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器。"""
    parser = argparse.ArgumentParser(prog="main.py", description="台灣系統瞬時負載尖峰預測")
    parser.add_argument("command", choices=list(COMMANDS), help="要執行的指令")
    parser.add_argument("--weather-mode", choices=workflow.WEATHER_MODES, default="honest",
                        help="evaluate／backtest 的氣象模式：honest（預設，與提交時相同）／"
                             "observed（目標日也用觀測，樂觀）／forecast（全部用預報）")
    parser.add_argument("--group", choices=("tuning", "holdout"), default="tuning",
                        help="backtest 的回測組：tuning 調參組（預設）／holdout 保留確認組")
    parser.add_argument("--label", help="backtest 這次實驗的名稱（必填）；verify-curves 要複核的紀錄名稱"
                                         "（預設 notebook_04）")
    parser.add_argument("--observed-lag-days", type=int, default=0,
                        help="backtest 模擬 CODiS 觀測只到起點前第幾天（預設 0）")
    parser.add_argument("--confirm-holdout", action="store_true",
                        help="確認要使用保留確認組（只在選定最終設定後使用一次）")
    parser.add_argument("--compare-to", help="backtest 完成後與這份參考紀錄配對比較")
    parser.add_argument("--set", action="append", metavar="名稱=值",
                        help="backtest 的設定覆寫（可重複），例如 --set TIMING_LEARNED_MIX=0.3")
    parser.add_argument("--reference", help="backtest-compare 的參考紀錄目錄")
    parser.add_argument("--candidate", help="backtest-compare 的候選紀錄目錄")
    parser.add_argument("--data-end", help="rehearse 的資料截止日 YYYY-MM-DD")
    parser.add_argument("--log-level", default=settings.LOG_LEVEL,
                        help=f"log 等級（預設 {settings.LOG_LEVEL}）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """程式進入點。

    Args:
        argv: 命令列參數，None 時取自 ``sys.argv``。

    Returns:
        int: 行程結束碼。
    """
    args = build_parser().parse_args(argv)
    log_path = setup_logging(stage=args.command, level=args.log_level)
    logger.info("=== %s ===（log：%s）", args.command, paths.relative(log_path))
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
