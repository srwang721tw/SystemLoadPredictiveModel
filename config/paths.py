"""專案所有檔案與目錄路徑的單一來源。

一律使用 :class:`pathlib.Path`，專案根目錄由本檔案位置推導，不寫死絕對路徑。
比賽當天只需改 ``LOAD_DATA_FILE``（以及 ``settings.DATA_AVAILABLE_END``）。
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# 目錄
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "config"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
"""原始資料（外部取得，唯讀）。"""
PROCESSED_DIR: Path = DATA_DIR / "processed"
"""由原始資料產生、可隨時重建的中間檔。"""
MANIFEST_FILE: Path = DATA_DIR / "manifest.json"
"""原始資料指紋。見 `src/data/manifest.py`。"""
BACKUP_DIR: Path = RAW_DIR / "_backup"
"""每次接受新資料時的原始檔內容快照（parquet，全字串），供下次逐筆比對。"""

OUTPUT_DIR: Path = PROJECT_ROOT / "output"
"""輸出根目錄。"""
SUBMISSION_DIR: Path = OUTPUT_DIR / "submission"
"""提交檔。"""
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
"""圖。"""
EVALUATION_DIR: Path = OUTPUT_DIR / "evaluation"
"""`workflow.evaluate` 的逐折分數與逐日預測。"""
BACKTEST_DIR: Path = OUTPUT_DIR / "backtest"
"""`src.evaluation.backtest` 的每次回測紀錄（每次一個子目錄）與保留組使用紀錄。"""
REPORTS_DIR: Path = OUTPUT_DIR / "reports"
"""notebook 產出的表格。"""
LOGS_DIR: Path = PROJECT_ROOT / "logs"


# =============================================================================
# 主資料
# =============================================================================

LOAD_DATA_FILE: Path = RAW_DIR / "正式競賽資料.csv"
"""系統瞬時負載，欄位 ``Date_Time, Load_MW``，每 10 分鐘一筆。

開發期為主辦單位提供之範例檔（2024-01-01 ~ 2026-06-30）。
比賽當天改為 ``RAW_DIR / "正式競賽資料.csv"``，並把 ``settings.DATA_AVAILABLE_END``
改為 ``"2026-09-30"``——只需改這兩處設定，不需改程式。
"""

# =============================================================================
# 外生資料
# =============================================================================

WEATHER_FILE: Path = RAW_DIR / "氣象觀測_全欄位.csv"
"""CODiS 五站逐小時觀測：各變數與 ``…f`` 旗標的**原始字串**（``python main.py weather`` 產生）。

讀檔時由 `src.data.weather.clean_observations` 依 `settings.CODIS_*` 對照表轉成數值。
"""

ACCUWEATHER_DIR: Path = RAW_DIR / "Accuweather"
"""Accuweather 鄉鎮預報原始檔目錄（一年一檔，逐小時 × 368 鄉鎮，讀取時自動合併）。

約 1 GB，一律以 `pl.scan_csv` + 欄位投影 + `location_key` 過濾讀取，不整檔載入。
"""

ACCUWEATHER_MAP_FILE: Path = (
    ACCUWEATHER_DIR / "Accuweather氣象網站鄉鎮氣象預報資料-區域對照表.csv"
)
"""鄉鎮 → `location_key` 對照表，欄位 `county, township, location_key`。

必須用 county + township 兩欄配對：「北區」與「中正區」在多個縣市重複出現。
"""

ACCUWEATHER_FORECAST_PATTERN: str = "*天氣預測詳細表*.csv"
"""Accuweather 年度預報檔的檔名樣式（各 280–410 MB，不附在 repo 內）。"""

WINDY_FILE: Path = RAW_DIR / "Windy.csv"
"""太陽光電機組預測發電量（6 個機組，3 小時一筆）。"""

CALENDAR_FILE: Path = RAW_DIR / "時間電價日曆表NEW.xlsx"
"""時間電價日曆表（2011–2060）：每日的日別（平日／週六／週日及離峰日）、農曆、節氣。

同時是權威的國定假日清單——國定假日在電價上歸為「週日及離峰日」，
用電行為也確實像週日。不含颱風停班停課（臨時公告），颱風另有清單。
"""

# =============================================================================
# 人工整理的規則與清單
# =============================================================================

PRICE_PERIOD_RULES_FILE: Path = CONFIG_DIR / "price_periods.toml"
"""日內電價時段規則（尖峰／半尖峰／週六半尖峰／離峰）與夏月起訖，程式中不寫死。"""

TYPHOON_FILE: Path = CONFIG_DIR / "颱風停班停課.csv"
"""歷史颱風停班停課清單：日期、影響範圍、侵臺路徑分類、近臺強度、公告時間。"""

EVENT_DAY_FILE: Path = CONFIG_DIR / "事件日.csv"
"""基礎設施事件日（地震等）。負載被動被壓低，標籤不可信，不進量值訓練。"""

LEAVE_DAY_FILE: Path = CONFIG_DIR / "放假日.csv"
"""逐日判讀的放假日。只用於回測的「含連假或停班」子集。"""

SPECIAL_DAY_INTERVAL_FILE: Path = CONFIG_DIR / "特殊日期區間.csv"
"""特殊日期的區間定義（連假／寒暑假／考試／運動賽事／購物節），展開後見 `SPECIAL_DAYS_FILE`。"""

# =============================================================================
# 中間檔（可由 workflow.build_processed 重建）
# =============================================================================

CLEAN_LOAD_FILE: Path = PROCESSED_DIR / "load_clean.parquet"
"""補值與品質標記後的 10 分鐘序列。"""

TARGETS_FILE: Path = PROCESSED_DIR / "targets.parquet"
"""每日 6 項目標標籤。"""

SPECIAL_DAYS_FILE: Path = PROCESSED_DIR / "特殊日期.csv"
"""特殊日期的每日 0/1 表，由 `SPECIAL_DAY_INTERVAL_FILE` 展開而來。"""

# =============================================================================
# 工具函式
# =============================================================================

INPUT_FILES: dict[str, Path] = {
    "負載": LOAD_DATA_FILE,
    "CODiS 觀測": WEATHER_FILE,
    "Accuweather 對照表": ACCUWEATHER_MAP_FILE,
    "Windy": WINDY_FILE,
    "時間電價日曆表": CALENDAR_FILE,
    "電價時段規則": PRICE_PERIOD_RULES_FILE,
    "颱風停班停課": TYPHOON_FILE,
    "事件日": EVENT_DAY_FILE,
    "放假日": LEAVE_DAY_FILE,
    "特殊日期區間": SPECIAL_DAY_INTERVAL_FILE,
}
"""產生提交檔所需的輸入檔，供 ``python main.py check`` 回報。"""


def ensure_directories() -> None:
    """建立所有執行期需要的目錄（若不存在），不觸碰任何既有檔案。"""
    for directory in (RAW_DIR, PROCESSED_DIR, SUBMISSION_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def missing_input_files() -> dict[str, Path]:
    """回報尚未到位的輸入檔 ``{名稱: 路徑}``；全部到位時回傳空字典。"""
    missing = {name: path for name, path in INPUT_FILES.items() if not path.exists()}
    if not has_accuweather_forecast():
        missing["Accuweather 年度檔"] = ACCUWEATHER_DIR / ACCUWEATHER_FORECAST_PATTERN
    return missing


def relative(path: Path | None) -> str:
    """路徑相對於專案根目錄的寫法，供 log 與摘要顯示（不輸出本機絕對路徑）。"""
    if path is None:
        return ""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def has_accuweather_forecast() -> bool:
    """``ACCUWEATHER_DIR`` 中是否至少有一個 Accuweather 年度預報檔。"""
    return ACCUWEATHER_DIR.exists() and any(ACCUWEATHER_DIR.glob(ACCUWEATHER_FORECAST_PATTERN))
