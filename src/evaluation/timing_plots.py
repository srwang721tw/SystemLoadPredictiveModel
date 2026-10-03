"""尖峰時刻的分布診斷圖：邊際、月份、日夜聯合。

分四組（整體／平日／週六／週日及離峰日）× 三種圖 = **12 張**，
全部寫入 ``output/figures/``（已 gitignored）。

**窗口的端點是這份圖最容易出錯的地方。**
夜尖峰是 **17:10–21:00（24 格）**，不是 17:00 起。17:00 是**日尖峰的右邊界**，
傍晚負載多半自 17:00 起單調下滑——把它放進夜尖峰窗口，argmax 會大量被吸到
左端點，堆出一根 396 天（43%）的假柱，並與日尖峰圖上的 17:00 重複計數同一個時刻。
正確窗口下最高的一格是 17:10 的 258 天（28.3%）。**結論方向完全不同。**
故本檔的格點一律由 :func:`~src.features.targets.night_peak_grid` 產生，
**不得手寫** ``range(17 * 60, 21 * 60 + 1, 10)``。

由 ``notebooks/02_探索分析.ipynb`` 呼叫。
"""

from __future__ import annotations

from typing import Final

import matplotlib

matplotlib.use("Agg")  # 無視窗環境；須在 pyplot 之前設定
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import colors

from config import paths, settings
from src.features.targets import day_peak_grid, format_hhmm, night_peak_grid

CJK_FONTS = ["PingFang HK", "Heiti TC", "Arial Unicode MS", "Songti SC",
             "Microsoft JhengHei", "Noto Sans CJK TC", "WenQuanYi Zen Hei"]
"""中文字型候選。未設定時 matplotlib 會把中文畫成方框。"""


def use_cjk_font() -> None:
    """設定中文字型，並修正負號顯示（notebook 的圖也共用這組設定）。"""
    plt.rcParams["font.sans-serif"] = CJK_FONTS + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


# 圖上有中文；在載入時設定，呼叫端不必記得這件事。
use_cjk_font()

type Group = tuple[str, str, str | None]
"""``(檔名 slug, 顯示名稱, price_daytype 篩選值)``；篩選值 None 代表整體。"""

GROUPS: Final[tuple[Group, ...]] = (
    ("all", "整體", None),
    ("weekday", "平日", "平日"),
    ("saturday", "週六", "週六"),
    ("offpeak", "週日及離峰日", "週日及離峰日"),
)
"""分組一律用 ``price_daytype``（電價日曆的日別），**不是**星期推導的 ``daytype``。

差別在國定假日：日曆上是週一到週五，但用電行為像週日，日曆表已把它們歸入
「週日及離峰日」。用星期推導會把它們混進「平日」那組。
"""

type Panel = tuple[str, str, list[int], str, str]
"""``(欄名, 顯示名稱, 格點, 長條顏色, 熱力圖 colormap)``。"""

PANELS: Final[tuple[Panel, ...]] = (
    (
        "t_day",
        f"日尖峰（{settings.DAY_PEAK_START}–{settings.DAY_PEAK_END}，"
        f"{len(day_peak_grid())} 格）",
        day_peak_grid(),
        "#d95f02",
        "YlOrRd",
    ),
    (
        "t_night",
        f"夜尖峰（{settings.NIGHT_PEAK_START}–{settings.NIGHT_PEAK_END}，"
        f"{len(night_peak_grid())} 格）",
        night_peak_grid(),
        "#1f77b4",
        "Blues",
    ),
)

MONTHS: Final[tuple[int, ...]] = tuple(range(1, 13))

MONTH_CAVEAT: Final[str] = (
    # 圖上一律用「※」而非「」：Heiti TC 沒有 U+26A0，會畫成方框。
    "※ 各月的年數不均（資料為 2024-01 ~ 2026-06，1–6 月有 3 年、7–12 月只有 2 年），"
    "不同列的總天數不可直接相比"
)


# =============================================================================
# 統計表（純函式，由 tests/test_timing_plots.py 綁住）
# =============================================================================


def load_timing() -> pl.DataFrame:
    """讀取每日標籤表，只取畫圖需要的欄位。

    直接讀 ``targets.parquet``，**不重算標籤、不碰模型**。

    Returns:
        pl.DataFrame: 含 ``date`` / ``t_day`` / ``t_night`` / ``month``
            / ``price_daytype``。
    """
    return pl.read_parquet(paths.TARGETS_FILE).select(
        "date", "t_day", "t_night", "month", "price_daytype"
    )


def subset(daily: pl.DataFrame, daytype: str | None) -> pl.DataFrame:
    """取出單一日別的子集；``daytype`` 為 None 時回傳全部。

    Args:
        daily: :func:`load_timing` 的輸出。
        daytype: ``price_daytype`` 的值，或 None（整體）。

    Returns:
        pl.DataFrame: 子集。
    """
    if daytype is None:
        return daily
    return daily.filter(pl.col("price_daytype") == daytype)


def marginal_counts(sub: pl.DataFrame, column: str, grid: list[int]) -> np.ndarray:
    """單一時刻欄位在格點上的出現次數。

    以格點為準做對位，**不是** ``value_counts``——沒出現過的格點必須是 0
    而不是消失，否則長條圖會擠掉空格、x 軸不再等距。

    Args:
        sub: 子集。
        column: ``"t_day"`` 或 ``"t_night"``。
        grid: 該目標的完整格點。

    Returns:
        np.ndarray: 長度為 ``len(grid)`` 的整數計數。
    """
    index = {minute: position for position, minute in enumerate(grid)}
    counts = np.zeros(len(grid), dtype=int)
    for minute in sub[column].to_list():
        counts[index[int(minute)]] += 1
    return counts


def month_counts(sub: pl.DataFrame, column: str, grid: list[int]) -> np.ndarray:
    """月份 × 格點的計數表。

    Args:
        sub: 子集。
        column: ``"t_day"`` 或 ``"t_night"``。
        grid: 該目標的完整格點。

    Returns:
        np.ndarray: 形狀 ``(12, len(grid))``，第 0 列為 1 月。
    """
    table = np.zeros((len(MONTHS), len(grid)), dtype=int)
    for month, minutes in zip(
        sub["month"].to_list(), sub[column].to_list(), strict=True
    ):
        table[int(month) - 1, grid.index(int(minutes))] += 1
    return table


def joint_counts(sub: pl.DataFrame) -> np.ndarray:
    """``t_day`` × ``t_night`` 的聯合列聯表。

    Args:
        sub: 子集。

    Returns:
        np.ndarray: 形狀 ``(37, 24)``，**列是 ``t_day``、行是 ``t_night``**。
    """
    day_grid, night_grid = day_peak_grid(), night_peak_grid()
    table = np.zeros((len(day_grid), len(night_grid)), dtype=int)
    for day, night in zip(sub["t_day"].to_list(), sub["t_night"].to_list(), strict=True):
        table[day_grid.index(int(day)), night_grid.index(int(night))] += 1
    return table


MIN_SUPPORT: Final[int] = 5
"""條件機率的最小樣本數。低於此值的 ``t_day`` 整列留白。

只有 1 天樣本的 ``t_day`` 會得到一個 P = 1.0 的格子——那不是「必然發生」，
是「只有一筆」。不擋掉的話整張圖最亮的幾格全是雜訊，而且亮到會主導判讀。
"""


def row_normalise(table: np.ndarray, min_support: int = MIN_SUPPORT) -> np.ndarray:
    """把列聯表逐列除以列和，得到 ``P(t_night | t_day)``。

    **樣本不足的列填 NaN，不是 0。** 「這個 t_day 沒幾天」與
    「出現過但夜尖峰不會落在這裡」是兩件完全不同的事，畫成同一個顏色
    會讓空白區看起來像有結論。NaN 由 colormap 的 bad color 畫成灰色。

    Args:
        table: :func:`joint_counts` 的輸出。
        min_support: 列和低於此值即整列視為不可判讀。

    Returns:
        np.ndarray: 同形狀的浮點數，樣本不足的列為 NaN。
    """
    totals = table.sum(axis=1, keepdims=True)
    out = np.full(table.shape, np.nan, dtype=float)
    seen = (totals >= max(min_support, 1)).ravel()
    out[seen] = table[seen] / totals[seen]
    return out


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    """列聯表的卡方統計量與 Cramér's V。

    格點稀疏時 V 會被高估（期望次數遠小於 5 的格子很多），故只能用於
    **組間相對比較**，不可當成效果量的絕對值。

    Args:
        table: 列聯表。

    Returns:
        tuple[float, float]: ``(chi2, V)``；樣本數為 0 時回傳 ``(0.0, 0.0)``。
    """
    total = table.sum()
    if total == 0:
        return 0.0, 0.0
    row = table.sum(axis=1, keepdims=True)
    column = table.sum(axis=0, keepdims=True)
    expected = row * column / total
    seen = expected > 0
    chi2 = float((((table - expected) ** 2)[seen] / expected[seen]).sum())
    rank = min(int((row > 0).sum()), int((column > 0).sum()))
    if rank < 2:
        return chi2, 0.0
    return chi2, float(np.sqrt(chi2 / total / (rank - 1)))


N_PERMUTATIONS: Final[int] = 2000
"""重排檢定的次數。4 組共約 10 秒，夠穩定到小數第三位。"""


def permutation_null(
    sub: pl.DataFrame, n_permutations: int = N_PERMUTATIONS
) -> tuple[float, float]:
    """在「t_day 與 t_night 獨立」的虛無假設下，Cramér's V 的分布。

    **這個函式是判讀聯合熱力圖的前提。** 37 × 24 = 888 個格子只有 123 天
    （週六）時，**即使兩者完全獨立，V 也會很高**——因為每個格子期望不到 0.2 天，
    卡方統計量被小期望值撐大。直接比較各組的 V 會得到完全錯誤的結論：
    週六看起來 V = 0.449「最強」，重排後才發現虛無分布的均值就是 0.432。

    作法是把 ``t_night`` 隨機重排（**兩邊的邊際分布都保持不變**，
    只打斷配對關係），重算 V。

    Args:
        sub: 子集。
        n_permutations: 重排次數。

    Returns:
        tuple[float, float]: ``(虛無分布均值, 單尾 p 值)``；
            p 值 = 重排後 V 不低於實際 V 的比例。
    """
    observed = cramers_v(joint_counts(sub))[1]
    rng = np.random.default_rng(settings.RANDOM_SEED)
    night = sub["t_night"].to_numpy()
    values = np.empty(n_permutations)
    for index in range(n_permutations):
        shuffled = sub.with_columns(pl.Series("t_night", rng.permutation(night)))
        values[index] = cramers_v(joint_counts(shuffled))[1]
    return float(values.mean()), float((values >= observed).mean())


# =============================================================================
# 繪圖
# =============================================================================


def _tick_labels(grid: list[int]) -> list[str]:
    """格點（當日分鐘數）轉成 ``HH:MM`` 標籤。"""
    return [format_hhmm(minute) for minute in grid]


def _save(figure: plt.Figure, name: str) -> None:
    """寫出圖檔並關閉，避免大量圖累積在記憶體。"""
    output = paths.FIGURES_DIR / name
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"已輸出：{output.name}")


def plot_marginal(sub: pl.DataFrame, group: Group) -> None:
    """A：日／夜尖峰各自的時刻分布長條圖。"""
    slug, label, _ = group
    figure, axes_list = plt.subplots(2, 1, figsize=(14, 9))

    for axes, (column, title, grid, colour, _) in zip(axes_list, PANELS, strict=True):
        counts = marginal_counts(sub, column, grid)
        positions = np.arange(len(grid))
        axes.bar(positions, counts, color=colour, edgecolor="#333333", linewidth=0.5)
        top = counts.max() if counts.max() else 1
        for position, count in zip(positions, counts, strict=True):
            axes.text(
                position, count + top * 0.015, str(count),
                ha="center", va="bottom", fontsize=7,
            )
        share = counts.max() / sub.height * 100 if sub.height else 0.0
        peak = _tick_labels(grid)[int(counts.argmax())]
        axes.set_title(
            f"{title}　最高：{peak}　{counts.max()} 天（{share:.1f}%）",
            fontsize=12,
        )
        axes.set_xticks(positions)
        axes.set_xticklabels(_tick_labels(grid), rotation=45, ha="right", fontsize=8)
        axes.set_xlabel("時間點")
        axes.set_ylabel("出現天數")
        axes.set_ylim(0, top * 1.12)
        axes.grid(axis="y", alpha=0.25, ls="--")

    figure.suptitle(
        f"最大負載發生時刻分布　【{label}】　n = {sub.height} 天", fontsize=14
    )
    figure.tight_layout()
    _save(figure, f"timing_marginal_{slug}.png")


def plot_month(sub: pl.DataFrame, group: Group) -> None:
    """B：月份 × 時刻的計數熱力圖。"""
    slug, label, _ = group
    figure, axes_list = plt.subplots(2, 1, figsize=(16, 12))

    for axes, (column, title, grid, _, cmap) in zip(axes_list, PANELS, strict=True):
        table = month_counts(sub, column, grid)
        image = axes.imshow(table, aspect="auto", cmap=cmap)
        threshold = table.max() * 0.55 if table.max() else 1
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                value = table[row, col]
                axes.text(
                    col, row, str(value), ha="center", va="center", fontsize=6.5,
                    color="white" if value > threshold else "#333333",
                )
        axes.set_title(f"各月份【{title}】最大負載發生時刻", fontsize=12)
        axes.set_xticks(np.arange(len(grid)))
        axes.set_xticklabels(_tick_labels(grid), rotation=45, ha="right", fontsize=8)
        axes.set_yticks(np.arange(len(MONTHS)))
        axes.set_yticklabels([f"{month}月" for month in MONTHS], fontsize=9)
        axes.set_xlabel("時間點")
        figure.colorbar(image, ax=axes, fraction=0.02, pad=0.01)

    figure.suptitle(
        f"最大負載發生時刻 × 月份　【{label}】　n = {sub.height} 天\n{MONTH_CAVEAT}",
        fontsize=13,
    )
    figure.tight_layout()
    _save(figure, f"timing_month_{slug}.png")


def plot_joint(sub: pl.DataFrame, group: Group) -> None:
    """C：``t_day`` × ``t_night`` 的聯合熱力圖（計數 + 條件機率）。

    兩個面板不是重複：左邊看**質量集中在哪**，右邊看**知道 t_day 之後
    t_night 的分布有沒有變**——後者才是「有沒有關聯」的直接證據，
    在左圖上會被「某些 t_day 本來就罕見」蓋掉。
    """
    slug, label, _ = group
    day_grid, night_grid = day_peak_grid(), night_peak_grid()
    table = joint_counts(sub)
    _, v = cramers_v(table)
    null_mean, p_value = permutation_null(sub)

    figure, axes_list = plt.subplots(1, 2, figsize=(21, 8))

    # imshow 的第一軸是 y。要讓 x = t_day、y = t_night，須轉置。
    #
    # 兩個面板都用**淺 → 深**的序列色階（YlOrRd／Blues），與月份熱力圖
    # 同一套視覺語言：**數字越大顏色越深**。viridis／magma 是反過來的
    # （大 = 亮），並排時會讓人把兩張圖讀反。
    panels = (
        (table.T.astype(float), "計數（天）", "YlOrRd", None),
        (
            row_normalise(table).T,
            f"P（夜尖峰時刻｜日尖峰時刻）　樣本 < {MIN_SUPPORT} 天的 t_day 留白",
            "Blues",
            (0.0, 1.0),
        ),
    )
    for axes, (matrix, title, cmap_name, limits) in zip(axes_list, panels, strict=True):
        cmap = plt.get_cmap(cmap_name).copy()
        # 「樣本不足」用中灰。淺色階的最低色本身就很淡，用淺灰會分不出
        # 「沒樣本」與「機率接近 0」——那正是這張圖最不能混淆的兩件事。
        cmap.set_bad("#9e9e9e")
        if limits:
            kwargs = {"vmin": limits[0], "vmax": limits[1]}
        else:
            # 計數是重尾的（最大格 43、多數格 0–2），線性色階會把結構壓成一片白。
            kwargs = {"norm": colors.PowerNorm(gamma=0.45, vmin=0, vmax=matrix.max())}
        image = axes.imshow(
            np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, **kwargs
        )
        axes.set_title(title, fontsize=12)
        axes.set_xticks(np.arange(len(day_grid)))
        axes.set_xticklabels(_tick_labels(day_grid), rotation=90, fontsize=7)
        axes.set_yticks(np.arange(len(night_grid)))
        axes.set_yticklabels(_tick_labels(night_grid), fontsize=7)
        axes.set_xlabel(f"日尖峰發生時刻 t_day（{len(day_grid)} 格）")
        axes.set_ylabel(f"夜尖峰發生時刻 t_night（{len(night_grid)} 格）")
        figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)

    # 計數面板標出較大的格子，讓團塊的位置可以直接讀出來。
    # 字色依**色階上的深淺**決定（不是依原始計數）——PowerNorm 之後，
    # 5 天那格已經是中等色，深淺與數值大小不成正比。
    count_norm = colors.PowerNorm(gamma=0.45, vmin=0, vmax=table.max())
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table[row, col]
            if value >= 5:
                axes_list[0].text(
                    row, col, str(value), ha="center", va="center", fontsize=6,
                    color="white" if count_norm(value) > 0.65 else "#333333",
                )

    verdict = "有關聯" if p_value < 0.05 else "與隨機重排無異"
    figure.suptitle(
        f"日尖峰 × 夜尖峰 發生時刻的聯合分布　【{label}】　n = {sub.height} 天\n"
        f"Cramér's V = {v:.3f}　隨機重排的虛無均值 = {null_mean:.3f}　"
        f"p = {p_value:.3f}　→ {verdict}\n"
        "灰色 = 該 t_day 樣本不足（不是機率 0）　"
        f"※ V 本身不可跨組比較——格子只有 {sub.height} 天時，"
        "獨立也會給出很高的 V，故一律看重排後的 p",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, f"timing_joint_{slug}.png")
