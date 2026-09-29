"""回歸測試：與現行模型的參考輸出（``output/reference/``）逐值比對。

改寫程式但不打算改變模型時，輸出必須與參考一致（差異 < 1e-6）。參考輸出含 60 折的
逐折分數與逐日 6 目標預測，比對逐日預測比只比總分嚴格得多。刻意改變模型之後，
以新的 ``evaluate`` 輸出更新 ``output/reference/``（現行模型 60 折 honest 1.12801）。

兩個測試都需要本機的參考檔與資料，且耗時，故預設略過：

- ``REGRESSION=quick``：重新跑 6 折，honest 與 observed 兩種模式，約 2–4 分鐘
- ``REGRESSION=full``：比對 ``python main.py evaluate --weather-mode …``
  已寫出的 ``output/evaluation/`` 兩個模式（不重新計算）
"""

from __future__ import annotations

import os

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from config import paths


REFERENCE = paths.OUTPUT_DIR / "reference"


MODE = os.environ.get("REGRESSION", "")


QUICK_FOLDS = [0, 10, 20, 30, 40, 50]


TOLERANCE = 1e-6


def _read(path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True).sort("origin", *(
        ["date"] if "date" in pl.read_csv(path, n_rows=1).columns else []))


def _assert_close(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert_frame_equal(actual, expected, check_exact=False, atol=TOLERANCE, rtol=0)


@pytest.mark.skipif(MODE != "quick", reason="設 REGRESSION=quick 才執行")
@pytest.mark.parametrize("mode", ["honest", "observed"])
def test_quick_six_folds_match_reference(tmp_path, mode: str) -> None:
    from src import workflow

    result = workflow.evaluate(mode, QUICK_FOLDS, tmp_path)
    for name in ("folds", "predictions"):
        actual = _read(tmp_path / f"{name}_{mode}.csv")
        expected = _read(REFERENCE / f"{name}_{mode}.csv").filter(
            pl.col("origin").is_in(actual["origin"].unique().to_list()))
        _assert_close(actual, expected)
    assert len(result["folds"]) == len(QUICK_FOLDS)


@pytest.mark.skipif(MODE != "full", reason="設 REGRESSION=full 才執行")
@pytest.mark.parametrize("mode", ["observed", "honest"])
@pytest.mark.parametrize("name", ["folds", "predictions"])
def test_full_outputs_match_reference(mode: str, name: str) -> None:
    actual = paths.EVALUATION_DIR / f"{name}_{mode}.csv"
    assert actual.exists(), f"先執行 python main.py evaluate --weather-mode {mode}"
    _assert_close(_read(actual), _read(REFERENCE / f"{name}_{mode}.csv"))
