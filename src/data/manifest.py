"""原始資料指紋（manifest）。

記錄每個資料檔的 SHA-256、筆數、時間範圍與各欄缺值數，寫成
``data/manifest.json`` 並納入版控。比賽當天覆蓋資料檔後，以新舊 manifest
比對就能知道哪些檔變了、變了多少。

缺值數是以**原始字串**計算（全部欄位以字串讀入，空字串視為缺值），
記錄的是檔案本身的狀態，不受後續清理規則影響。

比賽當天的流程（`run_submission.py` 第 1 步）：

1. :func:`update_check`：以目前檔案重建 manifest，與版控中的舊 manifest 比對；
   有唯一鍵的檔再與上次的內容快照**逐筆**比對
2. 歷史區段有變動時**警告但不中止**
3. :func:`accept`：寫入新 manifest，並為有唯一鍵的檔存新快照

用法::

    python -m src.data.manifest          # 比對並接受（第一次執行時只建立基準）
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import polars as pl

from config import paths, settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

_CHUNK_BYTES = 1 << 20


def sha256(path: Path) -> str:
    """計算檔案的 SHA-256。

    Args:
        path: 檔案路徑。

    Returns:
        str: 十六進位雜湊值。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pl.DataFrame | None:
    """以全字串讀入表格檔；不是表格（如 toml）時回傳 None。"""
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy")
    if path.suffix == ".xlsx":
        from src.data import external

        return external.read_xlsx(path).cast(pl.Utf8)
    return None


def _time_range(table: pl.DataFrame, column: str) -> tuple[str | None, str | None]:
    """依 ``settings.MANIFEST_TIME_FORMATS`` 解析時間欄，回傳最小與最大值。"""
    parsed = pl.coalesce([
        table[column].str.strptime(pl.Datetime, fmt, strict=False)
        for fmt in settings.MANIFEST_TIME_FORMATS
    ])
    values = table.select(parsed.alias("t"))["t"]
    unparsed = int(values.null_count() - table[column].null_count())
    if unparsed:
        raise ValueError(
            f"{column} 有 {unparsed} 筆無法以 MANIFEST_TIME_FORMATS 解析"
        )
    return (
        str(values.min()) if values.len() else None,
        str(values.max()) if values.len() else None,
    )


def describe_file(
    path: Path, time_column: str | None, kind: str, key: tuple[str, ...] | None = None
) -> dict:
    """產生單一檔案的 manifest 紀錄。

    Args:
        path: 檔案路徑。
        time_column: 時間欄名稱，None 時不記錄時間範圍。
        kind: ``raw`` / ``derived`` / ``config``。
        key: 唯一鍵；有值時才會做快照與逐筆比對。

    Returns:
        dict: 檔名、相對路徑、類別、大小、SHA-256、筆數、欄位、時間範圍、各欄缺值數。
    """
    record: dict = {
        "file": path.name,
        "path": path.relative_to(paths.PROJECT_ROOT).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "key": list(key) if key else None,
    }
    table = _read_table(path)
    if table is None:
        return record

    record["rows"] = table.height
    record["columns"] = table.columns
    record["null_counts"] = {c: int(table[c].null_count()) for c in table.columns}
    if time_column:
        record["time_column"] = time_column
        record["time_min"], record["time_max"] = _time_range(table, time_column)
    return record


def build_manifest(
    sources: tuple[tuple[str, str | None, str], ...] | None = None,
) -> dict:
    """依登記清單產生整份 manifest。

    Args:
        sources: ``(glob, 時間欄, 類別, 唯一鍵)``，None 時採 ``settings.MANIFEST_SOURCES``。

    Returns:
        dict: ``{"generated_at": ..., "files": [...]}``，檔案依路徑排序。

    Raises:
        FileNotFoundError: 某個 glob 找不到任何檔案。
    """
    records = []
    for pattern, time_column, kind, key in sources or settings.MANIFEST_SOURCES:
        matched = sorted(paths.PROJECT_ROOT.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"manifest 登記的檔案不存在：{pattern}")
        for path in matched:
            logger.info("manifest：%s", path.name)
            records.append(describe_file(path, time_column, kind, key))
    records.sort(key=lambda r: r["path"])
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "files": records,
    }


def write_manifest(manifest: dict, path: Path | None = None) -> Path:
    """寫出 manifest JSON（UTF-8、縮排、中文不跳脫）。

    Args:
        manifest: :func:`build_manifest` 的輸出。
        path: 輸出路徑，None 時採 ``paths.MANIFEST_FILE``。

    Returns:
        Path: 寫出的路徑。
    """
    path = path or paths.MANIFEST_FILE
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("manifest 已寫出：%s（%d 個檔）", paths.relative(path), len(manifest["files"]))
    return path



def _snapshot_path(directory: Path, record: dict) -> Path:
    return directory / (record["path"] + ".parquet")


def snapshot(manifest: dict, directory: Path) -> int:
    """把有唯一鍵的檔以全字串存成 parquet 快照，供下次逐筆比對。

    Args:
        manifest: :func:`build_manifest` 的輸出。
        directory: 快照目錄（``paths.BACKUP_DIR / <時間>``）。

    Returns:
        int: 存了幾個檔。
    """
    count = 0
    for record in manifest["files"]:
        if not record.get("key"):
            continue
        target = _snapshot_path(directory, record)
        target.parent.mkdir(parents=True, exist_ok=True)
        _read_table(paths.PROJECT_ROOT / record["path"]).write_parquet(target)
        count += 1
    logger.info("快照 %d 個檔 → %s", count, paths.relative(directory))
    return count


def latest_snapshot() -> Path | None:
    """最近一次快照的目錄；尚無快照時回傳 None。"""
    if not paths.BACKUP_DIR.exists():
        return None
    folders = sorted(p for p in paths.BACKUP_DIR.iterdir() if p.is_dir())
    return folders[-1] if folders else None


def compare_manifests(old: dict, new: dict) -> list[dict]:
    """逐檔比對新舊 manifest。

    Returns:
        list[dict]: 每個檔一筆：``path``、``status``（unchanged／changed／added／removed）、
            筆數與時間範圍的前後值、欄位的增減。
    """
    before = {r["path"]: r for r in old["files"]}
    after = {r["path"]: r for r in new["files"]}
    out = []
    for path in sorted(before.keys() | after.keys()):
        a, b = before.get(path), after.get(path)
        if a is None or b is None:
            out.append({"path": path, "status": "added" if a is None else "removed"})
            continue
        entry = {"path": path, "status": "unchanged" if a["sha256"] == b["sha256"] else "changed"}
        if entry["status"] == "changed":
            entry |= {
                "rows": (a.get("rows"), b.get("rows")),
                "time_max": (a.get("time_max"), b.get("time_max")),
                "columns_added": sorted(set(b.get("columns", [])) - set(a.get("columns", []))),
                "columns_removed": sorted(set(a.get("columns", [])) - set(b.get("columns", []))),
            }
        out.append(entry)
    return out


def _parsed_time(frame: pl.DataFrame, column: str) -> pl.Expr:
    return pl.coalesce([
        pl.col(column).str.strptime(pl.Datetime, fmt, strict=False)
        for fmt in settings.MANIFEST_TIME_FORMATS
    ])


def diff_rows(
    old: pl.DataFrame, new: pl.DataFrame, key: list[str],
    time_column: str | None = None, n_examples: int = 5,
) -> dict:
    """以唯一鍵逐筆比對新舊內容（兩者皆為全字串表）。

    完全重複的列先去除（例如 Windy 的 122 筆），否則 join 會把一筆算成多筆。

    Args:
        old: 舊內容。
        new: 新內容。
        key: 唯一鍵。
        time_column: 時間欄；有值時把新增分成「舊檔時間範圍內（歷史區段新增）」與
            「範圍外（延長）」。
        n_examples: 每類保留幾筆範例。

    Returns:
        dict: ``modified``、``deleted``、``added_in_history``、``extended`` 的筆數與範例，
            以及只存在於一邊的欄位。
    """
    only_in_old = sorted(set(old.columns) - set(new.columns))
    only_in_new = sorted(set(new.columns) - set(old.columns))
    common = [c for c in old.columns if c in new.columns]
    old, new = old.select(common).unique(), new.select(common).unique()
    values = [c for c in common if c not in key]
    joined = old.join(new, on=key, how="full", coalesce=True, suffix="__new")

    old_present = old.select(key).with_columns(pl.lit(True).alias("__old"))
    new_present = new.select(key).with_columns(pl.lit(True).alias("__new"))
    joined = joined.join(old_present, on=key, how="left").join(new_present, on=key, how="left")
    both = joined.filter(pl.col("__old") & pl.col("__new"))
    changed = pl.any_horizontal([pl.col(c).ne_missing(pl.col(f"{c}__new")) for c in values]) \
        if values else pl.lit(False)
    modified = both.filter(changed)
    deleted = joined.filter(pl.col("__old") & pl.col("__new").is_null())
    added = joined.filter(pl.col("__old").is_null() & pl.col("__new"))

    added_in_history, extended = added, added.clear()
    if time_column and time_column in key:
        stamps = old.select(_parsed_time(old, time_column).alias("t"))["t"]
        low, high = stamps.min(), stamps.max()
        when = _parsed_time(added, time_column)
        added_in_history = added.filter(when.is_between(low, high))
        extended = added.filter(~when.is_between(low, high))

    def pack(frame: pl.DataFrame, columns: list[str]) -> dict:
        return {"count": frame.height, "examples": frame.select(columns).head(n_examples).to_dicts()}

    pairs = [c for v in values for c in (v, f"{v}__new")]
    return {
        "modified": pack(modified, key + pairs),
        "deleted": pack(deleted, key + values),
        "added_in_history": pack(added_in_history, key + [f"{v}__new" for v in values]),
        "extended": pack(extended, key),
        "only_in_old": only_in_old,
        "only_in_new": only_in_new,
    }


def update_check(old_manifest: Path | None = None) -> dict:
    """資料更新檢查：比對目前檔案與上次接受的狀態。歷史變動只警告，不中止。

    Args:
        old_manifest: 舊 manifest 路徑，None 時採 ``paths.MANIFEST_FILE``。

    Returns:
        dict: ``manifest``（新）、``files``（逐檔比對）、``rows``（逐筆比對，
            以檔案路徑為鍵）、``history_changed``（是否有歷史區段變動）。
    """
    import json

    old_path = old_manifest or paths.MANIFEST_FILE
    new = build_manifest()
    if not old_path.exists():
        logger.warning("尚無舊 manifest，本次只建立基準")
        return {"manifest": new, "files": [], "rows": {}, "history_changed": False}

    old = json.loads(old_path.read_text(encoding="utf-8"))
    files = compare_manifests(old, new)
    backup = latest_snapshot()
    records = {r["path"]: r for r in new["files"]}
    rows: dict = {}
    for entry in files:
        record = records.get(entry["path"])
        if entry["status"] != "changed" or not record or not record.get("key"):
            continue
        if entry.get("columns_added") or entry.get("columns_removed"):
            logger.warning("%s 欄位結構改變：新增 %s、移除 %s", entry["path"],
                           entry["columns_added"], entry["columns_removed"])
        previous = backup and _snapshot_path(backup, record)
        if not previous or not previous.exists():
            logger.warning("%s 已改變，但沒有舊快照可逐筆比對", entry["path"])
            continue
        rows[entry["path"]] = diff_rows(
            pl.read_parquet(previous), _read_table(paths.PROJECT_ROOT / record["path"]),
            record["key"], record.get("time_column"),
        )

    history_changed = False
    for path, diff in rows.items():
        counts = {k: diff[k]["count"] for k in ("modified", "deleted", "added_in_history", "extended")}
        logger.info("%s 逐筆比對：%s", paths.relative(Path(path)), counts)
        if counts["modified"] or counts["deleted"] or counts["added_in_history"]:
            history_changed = True
            logger.warning(
                "%s 的**歷史區段**有變動（修改 %d、刪除 %d、歷史區段新增 %d），"
                "接受新資料並繼續（請確認變動是否合理）。修改範例：%s",
                path, counts["modified"], counts["deleted"], counts["added_in_history"],
                diff["modified"]["examples"][:2],
            )
    for entry in files:
        if entry["status"] != "unchanged":
            logger.info("檔案 %s：%s", entry["path"], entry["status"])
    return {"manifest": new, "files": files, "rows": rows, "history_changed": history_changed}


def accept(manifest: dict) -> Path:
    """接受目前的資料：寫入新 manifest，並存一份內容快照。

    Args:
        manifest: :func:`update_check` 回傳的新 manifest。

    Returns:
        Path: 快照目錄。
    """
    write_manifest(manifest)
    folder = paths.BACKUP_DIR / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot(manifest, folder)
    prune_snapshots()
    return folder


def prune_snapshots(keep: int | None = None) -> list[Path]:
    """只保留最新 ``keep`` 份快照，刪除較舊的。

    Args:
        keep: 保留份數，None 時採 ``settings.BACKUP_KEEP``。

    Returns:
        list[Path]: 被刪除的快照目錄。
    """
    keep = keep or settings.BACKUP_KEEP
    if not paths.BACKUP_DIR.exists():
        return []
    folders = sorted(p for p in paths.BACKUP_DIR.iterdir() if p.is_dir())
    removed = folders[:-keep]
    for folder in removed:
        shutil.rmtree(folder)
        logger.info("刪除舊快照：%s", folder.name)
    return removed


if __name__ == "__main__":
    accept(update_check()["manifest"])
