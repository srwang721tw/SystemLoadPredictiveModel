"""設定層：所有路徑、參數、常數的單一來源。

用法：
    from config import paths, settings

嚴禁在 src/ 底下硬編碼日期、路徑、閾值或電價時段。
"""

from config import paths, settings

__all__ = ["paths", "settings"]
