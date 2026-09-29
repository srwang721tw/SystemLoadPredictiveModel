"""比賽當天的執行入口：更新資料後重新訓練，產出提交檔。

用法::

    python run_submission.py

比賽當天只需改兩處設定（不改程式）：

- ``config/paths.py`` 的 ``LOAD_DATA_FILE`` → ``RAW_DIR / "正式數據.csv"``
- ``config/settings.py`` 的 ``DATA_AVAILABLE_END`` → ``"2026-09-30"``

九個步驟全部呼叫 ``src/`` 的函式，與 ``main.py``、notebook 共用同一套實作：

1. 資料更新檢查（對照 manifest，歷史區段有變動時警告不中止）
2. 讀檔與資料結構驗證（欄位、型別、時間頻率、連續性；不符即中止）
3. 前置檢查（負載涵蓋到 D-1 23:50、CODiS 至少到 D-2；不足即中止）。
   CODiS 中午後才更新到前一日，D-1 沒有觀測時以校正後的預報補上；
   目標日預報不齊時不中止，改走備援（Plan B）：部分城市缺以最近一天的觀測補，
   整天缺改用不含氣象的模型，Windy 不齊的日子改用不含 Windy 的模型。都列在摘要中
4. 資料清理與去除重複，重建中間檔
5. 以鎖定設定重新訓練
6. 預測 432 筆
7. 驗證提交檔（列數、時間戳、缺值、數值範圍、格式、讀回推導 6 目標）
8. 輸出提交檔，並接受本次資料（寫入 manifest 與內容快照）
9. 印出摘要（資料版本、各檢查結果、執行時間）

第 5、6 步在同一個函式內完成（``workflow.predict_days`` 在預測當下訓練模型），耗時合併計算。
"""

from __future__ import annotations

import datetime as dt
import sys
import time

from config import paths, settings
from src import workflow
from src.data import accuweather, checks, external, loader, manifest
from src.logging_setup import get_logger, setup_logging
from src.output import submission

logger = get_logger("run_submission")


def main(accept: bool = True) -> int:
    """依序執行九個步驟；任一步失敗即中止並印出已完成的摘要。

    Args:
        accept: 成功後是否接受本次資料（寫入 manifest 與內容快照）。演練時為 False。

    Returns:
        int: 0 表示成功產出提交檔；1 表示中止。
    """
    log_path = setup_logging(stage="run_submission")
    origin = checks.cutoff_date()
    days = [origin + dt.timedelta(days=k) for k in range(1, settings.PREDICT_HORIZON_DAYS + 1)]
    logger.info("=== 比賽當天流程 ===（log：%s）", paths.relative(log_path))
    logger.info("資料截止 %s，預測 %s ~ %s", origin, days[0], days[-1])
    if days[0] != dt.date.fromisoformat(settings.PREDICT_START):
        logger.warning("這是**驗證跑**：比賽要求從 %s 起預測", settings.PREDICT_START)

    summary: dict = {"steps": []}
    state: dict = {}

    def step(name: str, action) -> None:
        start = time.perf_counter()
        logger.info("── %s", name)
        state[name] = action()
        seconds = time.perf_counter() - start
        summary["steps"].append((name, round(seconds, 1)))
        logger.info("   完成（%.1f 秒）", seconds)

    try:
        step("1 資料更新檢查", manifest.update_check)
        step("2 讀檔與結構驗證", lambda: (
            loader.load_raw_load(), external.load_weather(),
            accuweather.load_station_hourly(), external.load_calendar(),
        ))
        step("3 前置檢查", lambda: checks.precheck(origin, days))
        step("4 清理、去重與重建中間檔", workflow.build_processed)
        step("5–6 重新訓練並預測 432 筆", lambda: workflow.predict_days(origin, days))
        curves, intended = state["5–6 重新訓練並預測 432 筆"]

        def validate_and_write():
            frame = submission.build_submission(curves)
            path = submission.write_submission(frame)
            submission.validate_submission(path, intended)
            imputed = state["4 清理、去重與重建中間檔"][0]
            return path, submission.check_window_and_range(path, days, imputed)

        step("7 驗證提交檔", validate_and_write)
        path, value_range = state["7 驗證提交檔"]
        if accept:
            step("8 接受本次資料", lambda: manifest.accept(state["1 資料更新檢查"]["manifest"]))
    except Exception as error:  # noqa: BLE001 —— 任何失敗都要印出已完成的摘要再中止
        logger.error("中止：%s", error)
        _print_summary(summary, None)
        return 1

    check = state["1 資料更新檢查"]
    summary |= {
        "manifest_sha256": manifest.sha256(manifest.paths.MANIFEST_FILE),
        "history_changed": check["history_changed"],
        "files_changed": [f["path"] for f in check["files"] if f["status"] != "unchanged"],
        "value_range": value_range,
        "codis": state["3 前置檢查"],
    }
    _print_summary(summary, path)
    return 0


PLAN_B_NAMES = {"persistence": "以最近一天的觀測補", "climatology": "以同月氣候值補"}
"""部分城市缺預報時的補法，摘要中的顯示名稱。"""


def _print_summary(summary: dict, path) -> None:
    """印出第 9 步的摘要。"""
    lines = ["", "=== 摘要 ==="]
    for name, seconds in summary["steps"]:
        lines.append(f"  {name:<24}{seconds:>8.1f} 秒")
    lines.append(f"  {'合計':<24}{sum(s for _, s in summary['steps']):>8.1f} 秒")
    if "manifest_sha256" in summary:
        lines += [
            f"  資料版本（manifest SHA-256）：{summary['manifest_sha256'][:16]}…",
            f"  本次變動的檔案：{summary['files_changed'] or '無'}",
            f"  歷史區段變動：{'有（詳見 log）' if summary['history_changed'] else '無'}",
            f"  數值範圍：{summary['value_range']['min']:.0f} ~ {summary['value_range']['max']:.0f} MW",
            f"  CODiS 觀測至：{summary['codis']['codis_observed_end']}"
            + (f"（{', '.join(map(str, summary['codis']['codis_filled']))} 以校正後的預報補上）"
               if summary["codis"]["codis_filled"] else ""),
        ]
        n_stations = len(settings.WEATHER_STATIONS)
        for day, gone in summary["codis"]["forecast_missing"].items():
            plan = ("不含氣象的模型" if len(gone) == n_stations
                    else f"{PLAN_B_NAMES.get(settings.FORECAST_PLAN_B, settings.FORECAST_PLAN_B)}（{'、'.join(gone)}）")
            lines.append(f"  Plan B：{day} 缺預報 → {plan}")
        for day in summary["codis"]["windy_missing"]:
            lines.append(f"  Plan B：{day} Windy 不齊 → 不含 Windy 的模型")
    lines.append(f"  提交檔：{paths.relative(path)}" if path else "  提交檔：未產出")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
