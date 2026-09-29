"""由 6 項目標合成 10 分鐘負載曲線。

**提交的是曲線，但計分只看能從曲線推導出的 6 個量。**

因此最佳策略是「先用決策理論選出最適的 6 個目標值，再合成一條**恰好實現**
它們的曲線」，而不是「直接預測曲線再取 argmax」（後者回測差 1.95 個標準誤）。
本模組負責第二步。

---

## 合成必須滿足的四個約束

由 ``src/features/targets.py::compute_targets`` 的定義反推：

1. ``max(L[m] for m in 日窗口) == p_day``，且**最早**達到該值的 m 為 ``t_day``
2. ``max(L[m] for m in 夜窗口) == p_night``，且最早達到者為 ``t_night``
3. ``max(L[i+1] - L[i]) == ramp_up``（全日 143 個差分，不跨日）
4. ``max(L[i] - L[i+1]) == ramp_down``

約束 1 與 2 的「最早」是並列時的規則。實測真實資料從未發生並列
（負載是浮點數），但**合成曲線很容易製造並列**——例如把窗口內壓平。
因此本模組一律讓極大值**嚴格唯一**。

## 作法：錨點 + 單調段內插

以錨點界定骨架，錨點之間線性內插。錨點包含兩個**恰好一步**的 ramp 段，
其餘每段的每步差分皆被設計為嚴格小於 ramp 上限。

ramp 的位置刻意放在實測的自然位置（`ramp_up` 07:00–09:00、
`ramp_down` 午休或傍晚，見 ``settings`` 中的 regime 常數），
這既符合資料的觀察，也讓曲線看起來合理——而且不花任何分數代價。

## 6 個目標可能互相衝突

真實曲線推導出的 6 個值必然自洽，但**我們的 6 個目標是各自獨立預測的**，
可能組不成任何一條曲線。最容易踩到的一條：

    t_day = 17:00 時，下一步 17:10 已在夜窗口內，該點必須低於 p_night
    （否則因時間較早而搶走 t_night），故需 p_night > p_day − ramp_down。

實測 `t_day` 有 10.7% 落在 17:00，故這不是罕見情形。本模組一律**明確拋錯**
並在訊息中指出衝突的量，由上游決定如何修補（`ramp_down` 權重僅 0.1
且無低估懲罰，放寬它的代價最小）。**絕不靜默產生一條實現不了目標的曲線。**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from config import settings
from src.features.targets import (
    day_peak_grid,
    format_hhmm,
    night_peak_grid,
    to_minutes,
)
from src.logging_setup import get_logger

logger = get_logger(__name__)

POINTS_PER_DAY = settings.POINTS_PER_DAY
STEP = settings.DATA_FREQ_MIN

EPSILON = 1e-6
"""用來製造嚴格不等式的極小量（MW）。

只需大於浮點誤差、遠小於量測精度即可。它的唯一用途是避免並列，
不會對評分造成可觀測的影響。
"""


@dataclass(frozen=True, slots=True)
class DayTargets:
    """單日的 6 項目標值。

    Attributes:
        p_day: 日尖峰負載量。
        t_day: 日尖峰時刻（當日分鐘數），須為合法格點。
        p_night: 夜尖峰負載量。
        t_night: 夜尖峰時刻（當日分鐘數），須為合法格點。
        ramp_up: 全日最大爬升量（正值）。
        ramp_down: 全日最大下降量（正值，絕對值）。
    """

    p_day: float
    t_day: int
    p_night: float
    t_night: int
    ramp_up: float
    ramp_down: float

    def validate(self) -> None:
        """檢查目標值本身是否合法。

        Raises:
            ValueError: 時刻不在合法格點上，或量值非正。
        """
        if self.t_day not in day_peak_grid():
            raise ValueError(f"t_day={format_hhmm(self.t_day)} 不在日尖峰格點上")
        if self.t_night not in night_peak_grid():
            raise ValueError(f"t_night={format_hhmm(self.t_night)} 不在夜尖峰格點上")
        for name in ("p_day", "p_night", "ramp_up", "ramp_down"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必須為正值，得到 {getattr(self, name)}")


def _index(minutes: int) -> int:
    """當日分鐘數 → 序列索引。"""
    return minutes // STEP


def _fill_linear(curve: np.ndarray, anchors: list[tuple[int, float]]) -> None:
    """依錨點以線性內插填滿曲線（就地修改）。

    Args:
        curve: 長度 144 的陣列。
        anchors: ``(索引, 值)`` 清單，須依索引升冪且涵蓋 0 與 143。
    """
    for (i0, v0), (i1, v1) in zip(anchors, anchors[1:], strict=False):
        span = i1 - i0
        curve[i0 : i1 + 1] = v0 + (v1 - v0) * np.arange(span + 1) / span


def _ramp_up_index(targets: DayTargets) -> int:
    """選擇最大爬升段的起始索引。

    放在實測的清晨啟動時段（平日 82.8% 落在此），且必須留下足夠步數
    讓其餘爬升攤平在 ``ramp_up`` 以下。
    """
    lo, hi = (to_minutes(m) for m in settings.MORNING_RAMP_WINDOW)
    return _index(min(max(lo, hi - STEP), targets.t_day - STEP * 2))


def _ramp_down_index(targets: DayTargets) -> int:
    """選擇最大下降段的起始索引。

    放在夜尖峰之後——那裡在兩個尖峰窗口之外，調整它不可能影響任何極大值。
    """
    return _index(max(targets.t_night + STEP, to_minutes(settings.NIGHT_PEAK_END)))


def shape_template(
    clean_load: pl.DataFrame, price_daytype: str, is_summer: bool, lookback: int = 120
) -> np.ndarray | None:
    """由歷史同組日取出「形狀模板」——逐點中位數的正規化曲線。

    各日先做 min-max 正規化再取中位數，使形狀與水準脫鉤：模板只描述
    「一天之內負載怎麼起伏」，不帶任何水準資訊。

    只用 ``clean_load`` 中 ≤ 預測起點日的列（呼叫端負責篩選）。
    模板**不影響任何評分量**——6 項目標由錨點固定，模板只決定錨點之間的形狀。

    Args:
        clean_load: 已切到 ≤ 預測起點日的 10 分鐘序列，須含
            ``date`` / ``mod`` / ``Load_MW`` / ``price_daytype`` / ``is_summer``。
        price_daytype: 目標日的日別。
        is_summer: 目標日是否為夏月。
        lookback: 最多取最近幾天。

    Returns:
        np.ndarray | None: 長度 144、值域 [0, 1] 的模板；同組樣本不足時為 None
            （呼叫端退回線性內插）。
    """
    group = clean_load.filter(
        (pl.col("price_daytype") == price_daytype) & (pl.col("is_summer") == is_summer)
    )
    days = sorted(group["date"].unique().to_list())[-lookback:]
    group = group.filter(pl.col("date").is_in(days))
    if len(days) < 4:
        return None

    normalised = group.with_columns(
        (
            (pl.col("Load_MW") - pl.col("Load_MW").min().over("date"))
            / (
                pl.col("Load_MW").max().over("date")
                - pl.col("Load_MW").min().over("date")
            )
        ).alias("_norm")
    )
    profile = (
        normalised.group_by("mod").agg(pl.col("_norm").median()).sort("mod")["_norm"]
    )
    if profile.len() != POINTS_PER_DAY:
        return None
    return profile.to_numpy().astype(float)


def _max_blend(
    linear_diffs: np.ndarray,
    template_diffs: np.ndarray,
    max_up: float,
    max_down: float,
) -> float:
    """解出使所有差分都合法的最大混合係數 β。

    混合後的差分是 β 的線性函數：

        d(β) = β · 模板差分 + (1 − β) · 線性差分

    因此每一步的約束 ``−max_down ≤ d(β) ≤ max_up`` 都能直接解出 β 的上界，
    取全段最小值即可——**不需要搜尋或試誤**。β = 0 時退化為線性內插，
    而線性內插在此之前已被驗證合法，故**必定有解**。

    Args:
        linear_diffs: 線性骨架的每步差分。
        template_diffs: 模板形狀的每步差分（已縮放到相同總變化量）。
        max_up: 允許的最大正差分。
        max_down: 允許的最大負差分幅度（正值）。

    Returns:
        float: 落在 [0, 1] 的最大可行 β。
    """
    beta = 1.0
    delta = template_diffs - linear_diffs
    for base, step in zip(linear_diffs, delta, strict=True):
        if step > 0:  # 往上走，受 max_up 限制
            room = max_up - base
            if room <= 0:
                return 0.0
            beta = min(beta, room / step)
        elif step < 0:  # 往下走，受 max_down 限制
            room = base + max_down
            if room <= 0:
                return 0.0
            beta = min(beta, room / (-step))
    return float(max(0.0, min(1.0, beta)))


def _shape_between(
    v0: float,
    v1: float,
    n_points: int,
    template: np.ndarray | None,
    max_up: float,
    max_down: float,
    ceiling: float | None = None,
) -> np.ndarray:
    """在兩個錨點之間鋪出一段曲線，形狀盡量貼近模板。

    端點值 ``v0`` / ``v1`` **永遠精確**——它們是評分量的載體。
    模板只影響中間的起伏，且會被自動收斂到不違反 ramp 上限。

    Args:
        v0: 起點值。
        v1: 終點值。
        n_points: 輸出長度（含首尾）。**明確傳入而非由模板推導**——
            模板為 None 時無從推導，初版漏了這點而產生空陣列。
        template: 模板片段（長度須為 ``n_points``）；None 時回傳線性內插。
        max_up: 允許的最大正差分。
        max_down: 允許的最大負差分幅度（正值）。
        ceiling: 若不為 None，段內任何點都不得超過此值
            （用於尖峰窗口內，避免搶走尖峰時刻）。

    Returns:
        np.ndarray: 長度 ``n_points`` 的曲線段，首尾恰為 ``v0`` / ``v1``。
    """
    linear = np.linspace(v0, v1, n_points)
    if template is None or len(template) != n_points or n_points < 3:
        return linear

    span = template[-1] - template[0]
    if abs(span) < 1e-12:
        return linear
    # 仿射變換：把模板的首尾對齊到 v0 / v1。
    shaped = v0 + (template - template[0]) * ((v1 - v0) / span)

    beta = _max_blend(np.diff(linear), np.diff(shaped), max_up, max_down)
    if ceiling is not None:
        # 天花板同樣是 β 的線性約束，可一併解出。
        excess = shaped - linear
        for index, (base, step) in enumerate(zip(linear, excess, strict=True)):
            if step > 0 and base + step > ceiling:
                room = ceiling - base
                beta = min(beta, max(0.0, room / step))
    if beta <= 0:
        return linear

    blended = linear + beta * (shaped - linear)
    blended[0], blended[-1] = v0, v1
    return blended


def _trough_index(shape: np.ndarray | None, up_start: int) -> int:
    """決定清晨谷底的索引。

    初版把谷底錨在 ``up_start``（爬升段起點，約 08:50），但真實的日最低點
    在 05:00–06:30。這使合成曲線在半夜**往上爬**才下降，與事實相反；
    套上形狀模板後更被放大——模板被迫在錯的位置對齊，起伏整個翻轉。

    有模板時取模板自身的最低點，兩者自然一致；無模板時退回 ``up_start``
    （行為與初版相同，仍是合法曲線，只是形狀較粗糙）。

    Args:
        shape: 形狀模板；None 時回傳 ``up_start``。
        up_start: 爬升段起始索引，谷底必須早於它。

    Returns:
        int: 谷底索引，保證落在 ``[1, up_start]``。
    """
    if shape is None:
        return up_start
    return int(np.clip(int(np.argmin(shape[: up_start + 1])), 1, up_start))


def _monotone_with_one_extreme_step(
    n_steps: int, total_change: float, extreme_step: float, extreme_at: int
) -> np.ndarray:
    """產生 ``n_steps`` 個同向差分，其中恰有一步為 ``extreme_step``。

    其餘步數平均分攤剩下的變化量。這個設計讓「指定的那一步是全段最大」
    自動成立，不需要事後修補——而事後修補正是初版失敗的原因：
    當時把夜尖峰後的曲線壓平到 21:00，使收尾的 16 步必須各降 635 MW，
    反而超過了意圖的 ramp_down 627.9。

    Args:
        n_steps: 差分個數。
        total_change: 全段的總變化量（與 ``extreme_step`` 同號）。
        extreme_step: 指定那一步的變化量。
        extreme_at: 指定步在本段中的序號（0-based）。

    Returns:
        np.ndarray: 長度 ``n_steps`` 的差分序列。

    Raises:
        ValueError: 其餘步數需要的變化量與 ``extreme_step`` 反向，
            或大於等於它——兩者都會使指定步不再是全段極值。
    """
    if n_steps < 1:
        raise ValueError(f"段內步數不足：{n_steps}")
    if n_steps == 1:
        if abs(total_change - extreme_step) > 1e-9:
            raise ValueError(
                f"單步段的總變化 {total_change:.2f} 與指定步 {extreme_step:.2f} 不符"
            )
        return np.array([extreme_step])

    rest = (total_change - extreme_step) / (n_steps - 1)
    # 浮點邊界：其餘步「剛好為 0」在數值上常是 ±1e-14，而下面的
    # `0 <= rest` / `rest <= 0` 是嚴格比較，會把完全可行的情形誤判為不可行。
    # 實際踩過：接上氣象特徵後量值預測改變，某天的 level_end 恰好落在
    # 「總降幅 == ramp_down」這條邊界上，合成因此拋錯而中斷整次 CV。
    # 這一步之外的其餘步本來就該是 0（全段只有那一步在動），故夾回 0。
    if abs(rest) < 1e-9:
        rest = 0.0
    if extreme_step > 0 and not 0 <= rest < extreme_step:
        raise ValueError(
            f"其餘 {n_steps - 1} 步每步需 {rest:.2f} MW，"
            f"無法讓指定步 {extreme_step:.2f} 成為全段最大爬升"
        )
    if extreme_step < 0 and not extreme_step < rest <= 0:
        raise ValueError(
            f"其餘 {n_steps - 1} 步每步需 {rest:.2f} MW，"
            f"無法讓指定步 {extreme_step:.2f} 成為全段最大下降"
        )

    diffs = np.full(n_steps, rest)
    diffs[extreme_at] = extreme_step
    return diffs


@dataclass(frozen=True, slots=True)
class FreeLevels:
    """曲線上三個**不參與評分**的水準。

    00:00、清晨谷底、23:50 都不在任何尖峰窗口內，也不是 ramp 的極值，
    因此 ``total_score`` 完全看不到它們。它們的唯一作用是讓曲線可行且合理。

    利用這個自由度是正確的作法，不是取巧：把它們固定成任意值反而會
    製造假的不可行——初版就是把 23:50 硬設為 ``p_day × 0.66``，
    使 912 天中有 115 天的收尾被迫比 ``ramp_down`` 更陡而合成失敗。

    Attributes:
        start: 00:00 的負載。
        minimum: 清晨谷底負載。
        end: 23:50 的負載。
    """

    start: float
    minimum: float
    end: float


def feasible_levels(
    targets: DayTargets, suggested: FreeLevels, shape: np.ndarray | None = None
) -> tuple[FreeLevels, list[str]]:
    """把建議水準夾到可行區間內。

    可行區間由「指定一步為極值、其餘平均分攤」的設計反推：

        谷底 ∈ [p_day − ramp_up × n_B,  p_day − ramp_up]
        23:50 ∈ (p_night − ramp_down × n_D,  p_night − ramp_down]
        00:00 與谷底的差距 ≤ ramp_down × n_A

    Args:
        targets: 6 項目標值。
        suggested: 由歷史推得的建議水準。
        shape: 形狀模板；用來決定谷底位置（步數會影響可行區間）。

    Returns:
        tuple: ``(夾過的水準, 調整說明清單)``。清單為空代表建議值本來就可行。

    Raises:
        ValueError: 可行區間為空——代表 6 項目標本身互相衝突，
            單靠自由水準救不回來。
    """
    up_start = _ramp_up_index(targets)
    trough = _trough_index(shape, up_start)
    day_index = _index(targets.t_day)
    night_index = _index(targets.t_night)
    last = POINTS_PER_DAY - 1

    n_b = day_index - trough
    n_d = last - night_index
    notes: list[str] = []

    min_lo = targets.p_day - targets.ramp_up * n_b
    min_hi = targets.p_day - targets.ramp_up
    if min_lo > min_hi:
        raise ValueError(
            f"谷底可行區間為空：ramp_up={targets.ramp_up:.1f} 在 {n_b} 步內"
            f"無法從任何水準爬到 p_day={targets.p_day:.1f}"
        )
    # 下界為開區間（在邊界上 rest 會恰等於 ramp_up，不滿足嚴格小於），往內縮。
    margin = max(1e-3, (min_hi - min_lo) * 1e-6)
    minimum = float(np.clip(suggested.minimum, min_lo + margin, min_hi))
    if minimum != suggested.minimum:
        notes.append(f"谷底 {suggested.minimum:.1f} → {minimum:.1f}")

    end_lo = targets.p_night - targets.ramp_down * n_d
    end_hi = targets.p_night - targets.ramp_down
    if end_lo >= end_hi:
        raise ValueError(
            f"收尾可行區間為空：ramp_down={targets.ramp_down:.1f} 在 {n_d} 步內"
            f"無法自 p_night={targets.p_night:.1f} 下降至任何水準"
        )
    # 下界為開區間，故往內縮一個 EPSILON。
    end = float(np.clip(suggested.end, end_lo + EPSILON, end_hi))
    if end != suggested.end:
        notes.append(f"23:50 {suggested.end:.1f} → {end:.1f}")

    # 00:00 → 谷底：下降受 ramp_down 限制，**上升受 ramp_up 限制**。
    # 兩側用同一個界限是錯的——當谷底被夾到很高時（ramp_up 極小的日子），
    # A 段會變成上升，卻仍以 ramp_down 放行，於是製造出超過 ramp_up 的爬升。
    start = float(
        np.clip(
            suggested.start,
            minimum - targets.ramp_up * trough,
            minimum + targets.ramp_down * trough,
        )
    )
    if start != suggested.start:
        notes.append(f"00:00 {suggested.start:.1f} → {start:.1f}")

    return FreeLevels(start, minimum, end), notes


def _feasible_dip(
    targets: DayTargets, day_index: int, dip_index: int, night_index: int
) -> float:
    """決定傍晚下凹的負載值。

    三個約束同時成立才可行：

    1. **下降段**：由 ``p_day`` 降到下凹點，每步不得超過 ``ramp_down``
    2. **上升段**：由下凹點升到 ``p_night``，每步不得超過 ``ramp_up``
    3. **唯一性**：下凹點在夜窗口起點上，必須**嚴格低於** ``p_night``，
       否則它會因為時間較早而搶走 ``t_night``（並列時取最早）

    Args:
        targets: 6 項目標值。
        day_index: ``t_day`` 的索引。
        dip_index: 下凹點索引（夜窗口起點）。
        night_index: ``t_night`` 的索引。

    Returns:
        float: 下凹點的負載值。

    Raises:
        ValueError: 三個約束的可行區間為空，代表這組目標值無法由任何曲線實現。
    """
    n_fall = dip_index - day_index
    n_rise = night_index - dip_index

    # 由 t_day 到下凹點的兩種情形，界限完全不同：
    #
    # 下降（dip ≤ p_day）：可平順攤在 n_fall 步上，受 ramp_down 限制。
    # 上升（dip > p_day）：**只能發生在日窗口結束之後**，因為窗口內任何
    #   高於 p_day 的點都會搶走 t_day。故整個上升被壓縮成 17:00→17:10
    #   這一步，受 ramp_up 限制。
    #
    # 初版誤以為上升也能攤在 n_fall 步上，於 p_night > p_day 的日子
    #   （「週日及離峰日」常見）算出 2 倍於意圖的 ramp_up。
    lower = targets.p_day - targets.ramp_down * n_fall
    upper = targets.p_day - EPSILON + targets.ramp_up

    if n_rise > 0:
        lower = max(lower, targets.p_night - targets.ramp_up * n_rise)
        # 唯一性：下凹點時間較早，必須嚴格低於 p_night 才不會搶走 t_night。
        upper = min(upper, targets.p_night - EPSILON)
    else:
        # t_night 就在夜窗口起點，沒有上升段——下凹點即 t_night 本身。
        if not lower <= targets.p_night <= upper:
            raise ValueError(
                f"t_day={format_hhmm(targets.t_day)} 到 "
                f"t_night={format_hhmm(targets.t_night)}：由 {targets.p_day:.1f} "
                f"變化到 {targets.p_night:.1f} 需超過 "
                f"ramp_up={targets.ramp_up:.1f} / ramp_down={targets.ramp_down:.1f}"
            )
        return targets.p_night

    if lower > upper:
        raise ValueError(
            f"傍晚下凹無可行值（需 ≥{lower:.1f} 且 ≤{upper:.1f}）："
            f"p_day={targets.p_day:.1f}@{format_hhmm(targets.t_day)} 降至 "
            f"p_night={targets.p_night:.1f}@{format_hhmm(targets.t_night)} 的過程中，"
            f"ramp_down={targets.ramp_down:.1f} 或 ramp_up={targets.ramp_up:.1f} 太小"
        )
    # 在可行範圍內取一個自然的下凹深度（低於夜尖峰約 2%），再夾回範圍。
    natural = targets.p_night * (1 - settings.EVENING_DIP_RATIO)
    return float(np.clip(natural, lower, upper))


def suggest_levels(
    history: pl.DataFrame, price_daytype: str, is_summer: bool
) -> FreeLevels:
    """由歷史同組日估計三個自由水準。

    取同一組（日別 × 夏月）歷史日的 ``load_start`` / ``load_min`` /
    ``load_end`` 中位數。這三個量不參與評分，故估得粗略也無妨——
    它們的作用只是讓曲線落在合理的水準上。

    ``history`` 必須已切到 ≤ 預測起點日。同組樣本不足時退回全體中位數，
    仍為空時回傳全 0（由 :func:`feasible_levels` 夾到可行區間）。

    Args:
        history: 每日表，須含 ``load_start`` / ``load_min`` / ``load_end``
            與 ``price_daytype`` / ``is_summer``。
        price_daytype: 目標日的日別（事前已知）。
        is_summer: 目標日是否為夏月（事前已知）。

    Returns:
        FreeLevels: 建議水準。
    """
    columns = ("load_start", "load_min", "load_end")
    group = history
    if all(c in history.columns for c in ("price_daytype", "is_summer")):
        group = history.filter(
            (pl.col("price_daytype") == price_daytype)
            & (pl.col("is_summer") == is_summer)
        )
    if group.height == 0:
        group = history
    if group.height == 0:
        return FreeLevels(0.0, 0.0, 0.0)

    values = [float(group[c].median() or 0.0) for c in columns]
    return FreeLevels(*values)


def _dip_bounds(
    targets: DayTargets, day_index: int, dip_index: int, night_index: int
) -> tuple[float, float]:
    """回傳傍晚下凹的可行區間 ``(lower, upper)``，供修補計算使用。

    與 :func:`_feasible_dip` 使用完全相同的公式；抽出來是為了讓
    :func:`repair_targets` 能直接解出「要放寬多少」，而不必反覆試誤。
    """
    n_fall = dip_index - day_index
    n_rise = night_index - dip_index
    lower = targets.p_day - targets.ramp_down * n_fall
    upper = targets.p_day - EPSILON + targets.ramp_up
    if n_rise > 0:
        lower = max(lower, targets.p_night - targets.ramp_up * n_rise)
        upper = min(upper, targets.p_night - EPSILON)
    return lower, upper


def repair_targets(targets: DayTargets) -> tuple[DayTargets, list[str]]:
    """把互相衝突的 6 項目標修成可合成，並回報改動了什麼。

    獨立預測出來的 6 個目標可能組不成任何曲線（真實資料不會有這問題，
    因為 ramp_down 本來就是實際曲線的最大降幅）。最常見的一條：

        t_day = 17:00 時，下一步 17:10 已在夜窗口內且必須低於 p_night，
        故需 p_night > p_day − ramp_down。實測 t_day 有 10.7% 落在 17:00。

    **放寬順序依分數代價由低到高**：

    1. ``ramp_down``——權重 0.1 且**無低估懲罰**，最便宜
    2. ``ramp_up``——權重 0.15；放寬方向是高估，同樣不觸發低估懲罰
    3. **絕不更動** ``p_day`` / ``p_night`` / ``t_day`` / ``t_night``
       ——時刻佔總分約 92%、尖峰量值另佔約 2%，動它們代價高得多

    Args:
        targets: 可能不可行的目標值。

    Returns:
        tuple: ``(修補後的目標, 改動說明清單)``。清單為空代表原本就可行。

    Raises:
        ValueError: 連放寬兩個 ramp 都無法修好（理論上不應發生，
            因為放大 ramp 一定能撐開可行區間）。
    """
    targets.validate()
    day_index = _index(targets.t_day)
    dip_index = _index(to_minutes(settings.NIGHT_PEAK_START))
    night_index = _index(targets.t_night)
    n_fall = dip_index - day_index
    n_rise = night_index - dip_index

    notes: list[str] = []
    lower, upper = _dip_bounds(targets, day_index, dip_index, night_index)
    if lower <= upper:
        return targets, notes

    # 步驟 1：放寬 ramp_down。它只影響 lower 的第一項 p_day − ramp_down·n_fall，
    # 要讓該項 ≤ upper，需 ramp_down ≥ (p_day − upper) / n_fall。
    ramp_down = targets.ramp_down
    needed = (targets.p_day - upper) / n_fall
    if needed > ramp_down:
        ramp_down = needed * (1 + 1e-9)
        notes.append(f"ramp_down {targets.ramp_down:.1f} → {ramp_down:.1f}")
        targets = DayTargets(
            targets.p_day, targets.t_day, targets.p_night, targets.t_night,
            targets.ramp_up, ramp_down,
        )
        lower, upper = _dip_bounds(targets, day_index, dip_index, night_index)
        if lower <= upper:
            return targets, notes

    # 步驟 2：放寬 ramp_up。它同時抬高 upper 並降低 lower 的第二項。
    if n_rise > 0:
        needed_up = (targets.p_night - upper) / n_rise
    else:
        needed_up = targets.p_night - targets.p_day + EPSILON
    if needed_up > targets.ramp_up:
        ramp_up = needed_up * (1 + 1e-9)
        notes.append(f"ramp_up {targets.ramp_up:.1f} → {ramp_up:.1f}")
        targets = DayTargets(
            targets.p_day, targets.t_day, targets.p_night, targets.t_night,
            ramp_up, targets.ramp_down,
        )
        lower, upper = _dip_bounds(targets, day_index, dip_index, night_index)

    if lower > upper:
        raise ValueError(
            f"放寬兩個 ramp 後仍不可行（需 ≥{lower:.1f} 且 ≤{upper:.1f}）："
            f"p_day={targets.p_day:.1f}@{format_hhmm(targets.t_day)}、"
            f"p_night={targets.p_night:.1f}@{format_hhmm(targets.t_night)}"
        )
    return targets, notes


def synthesize_from_targets(
    targets: DayTargets,
    history: pl.DataFrame,
    price_daytype: str,
    is_summer: bool,
    shape: np.ndarray | None = None,
) -> tuple[np.ndarray, DayTargets, list[str]]:
    """由 6 項目標與歷史，一步產出可提交的當日曲線。

    這是本模組對外的主要入口，串起
    ``suggest_levels → repair_targets → feasible_levels → synthesize_day_checked``。

    Args:
        targets: 模型預測的 6 項目標。
        history: 已切到 ≤ 預測起點日的每日表。
        price_daytype: 目標日的日別。
        is_summer: 目標日是否為夏月。
        shape: 形狀模板；None 時各段線性內插。

    Returns:
        tuple: ``(144 點曲線, 實際實現的目標, 所有調整說明)``。
            第二項可能與輸入不同（修補過），評分**必須用它**而非輸入值。
    """
    repaired, notes = repair_targets(targets)
    suggested = suggest_levels(history, price_daytype, is_summer)
    levels, level_notes = feasible_levels(repaired, suggested, shape)
    values = synthesize_day_checked(
        repaired, levels.start, levels.minimum, levels.end, shape=shape
    )
    return values, repaired, notes + level_notes


def _shaped_monotone_with_one_extreme_step(
    n_steps: int,
    total_change: float,
    extreme_step: float,
    extreme_at: int,
    template: np.ndarray | None,
) -> np.ndarray:
    """同 :func:`_monotone_with_one_extreme_step`，但其餘步依模板分配。

    均分會讓「谷底 → 日尖峰」畫成一條直線，與真實的 S 型爬升差很多
    （實際是清晨緩、07–09 陡、11 點後趨平）。此處把剩餘變化量按模板的
    差分比例分配，再以 β 收斂到「不超過指定的極值步」。

    **指定步仍然嚴格是全段最大**，故 ``ramp_up`` / ``ramp_down`` 的值不變。
    β = 0 時退化為均分，即原本的行為，故必定有解。

    Args:
        n_steps: 差分個數。
        total_change: 全段總變化量。
        extreme_step: 指定那一步的變化量。
        extreme_at: 指定步的序號。
        template: 該段的模板值（長度 ``n_steps + 1``）；None 時均分。

    Returns:
        np.ndarray: 長度 ``n_steps`` 的差分序列。
    """
    uniform = _monotone_with_one_extreme_step(
        n_steps, total_change, extreme_step, extreme_at
    )
    if template is None or len(template) != n_steps + 1 or n_steps < 3:
        return uniform

    others = np.delete(np.arange(n_steps), extreme_at)
    remaining = total_change - extreme_step
    weights = np.diff(np.asarray(template, dtype=float))[others]
    # 只有當模板在本段的走向與整體一致時才用它；否則退回均分。
    if np.sign(remaining) * weights.sum() <= 0:
        return uniform
    weights = np.clip(weights * np.sign(remaining), 0.0, None)
    if weights.sum() <= 0:
        return uniform

    shaped_rest = remaining * weights / weights.sum()
    uniform_rest = uniform[others]

    # β 使 |最大的其餘步| 嚴格小於 |extreme_step|。
    limit = abs(extreme_step)
    beta = 1.0
    for base, target_value in zip(uniform_rest, shaped_rest, strict=True):
        step = target_value - base
        if abs(base + step) <= limit:
            continue
        room = limit - abs(base)
        if room <= 0 or abs(step) <= 0:
            return uniform
        beta = min(beta, room / abs(step))
    beta = float(max(0.0, min(1.0, beta))) * 0.999  # 留一點餘裕，維持嚴格不等式

    diffs = uniform.copy()
    diffs[others] = uniform_rest + beta * (shaped_rest - uniform_rest)
    diffs[extreme_at] = extreme_step
    return diffs


def synthesize_day(
    targets: DayTargets,
    day_start_level: float,
    day_min_level: float,
    day_end_level: float,
    shape: np.ndarray | None = None,
) -> np.ndarray:
    """合成單日 144 點負載曲線，恰好實現給定的 6 項目標。

    曲線分為四段建構，每段的差分都被設計成不超過 ramp 上限：

    | 段 | 範圍 | 作法 |
    |---|---|---|
    | A | 00:00 → 清晨谷底 | 線性下降 |
    | B | 谷底 → ``t_day`` | 指定一步 = ``ramp_up``，其餘平均分攤 |
    | C | ``t_day`` → ``t_night`` | 線性；``p_night > p_day`` 時先在日窗口內壓平 |
    | D | ``t_night`` → 23:50 | 指定一步 = ``ramp_down``，其餘平均分攤 |

    Args:
        targets: 6 項目標值。
        day_start_level: 00:00 的負載水準。
        day_min_level: 清晨谷底負載。
        day_end_level: 23:50 的負載水準。
        shape: 長度 144 的形狀模板（:func:`shape_template` 的輸出）。
            None 時各段以線性內插填滿。**模板不影響任何評分量**——
            6 項目標由錨點固定，模板只決定錨點之間的起伏。

    Returns:
        np.ndarray: 長度 144 的負載序列。

    Raises:
        ValueError: 目標值不合法，或水準設定使約束無法滿足。
    """
    targets.validate()

    up_start = _ramp_up_index(targets)
    down_start = _ramp_down_index(targets)
    day_index = _index(targets.t_day)
    night_index = _index(targets.t_night)
    last = POINTS_PER_DAY - 1

    if not _trough_index(shape, up_start) < day_index:
        raise ValueError(
            f"爬升段（{format_hhmm(up_start * STEP)}）須早於 "
            f"t_day（{format_hhmm(targets.t_day)}）"
        )
    if not day_index < night_index < down_start < last:
        raise ValueError(
            f"時刻順序不合法：t_day={format_hhmm(targets.t_day)}、"
            f"t_night={format_hhmm(targets.t_night)}、"
            f"下降段={format_hhmm(down_start * STEP)}"
        )
    if day_min_level >= targets.p_day:
        raise ValueError(
            f"清晨谷底 {day_min_level:.1f} 不得高於 p_day {targets.p_day:.1f}"
        )

    curve = np.empty(POINTS_PER_DAY)

    # --- A：00:00 → 清晨谷底（真實的日最低點，非爬升段起點）---
    trough = _trough_index(shape, up_start)
    curve[: trough + 1] = _shape_between(
        day_start_level, day_min_level, trough + 1,
        None if shape is None else shape[: trough + 1],
        targets.ramp_up, targets.ramp_down,
    )

    # --- B：谷底 → t_day，其中一步恰為 ramp_up ---
    diffs_b = _shaped_monotone_with_one_extreme_step(
        n_steps=day_index - trough,
        total_change=targets.p_day - day_min_level,
        extreme_step=targets.ramp_up,
        extreme_at=up_start - trough,
        template=None if shape is None else shape[trough : day_index + 1],
    )
    curve[trough + 1 : day_index + 1] = day_min_level + np.cumsum(diffs_b)

    # --- C：t_day → 傍晚下凹 → t_night ---
    # 直接由 p_day 線性降到 t_night 是錯的：曲線在**進入夜窗口時**仍可能高於
    # p_night，被 _enforce_unique_peak 壓平後會在窗口邊界留下一個懸崖，
    # 實測可達 964 MW 而超過意圖的 ramp_down 616。
    # 改為在夜窗口起點下凹到 p_night 之下，再升到 t_night——
    # 這同時是台灣負載的真實形狀（午後尖峰、傍晚下凹、晚間尖峰）。
    dip_index = _index(to_minutes(settings.NIGHT_PEAK_START))
    day_end_index = _index(to_minutes(settings.DAY_PEAK_END))
    dip_value = _feasible_dip(targets, day_index, dip_index, night_index)

    if dip_value <= targets.p_day:
        curve[day_index : dip_index + 1] = _shape_between(
            targets.p_day, dip_value, dip_index - day_index + 1,
            None if shape is None else shape[day_index : dip_index + 1],
            targets.ramp_up, targets.ramp_down,
            ceiling=targets.p_day - EPSILON,
        )
    else:
        # 夜尖峰較高：日窗口內全程壓在 p_day 之下，上升只能在窗口結束後發生。
        curve[day_index : day_end_index + 1] = targets.p_day - EPSILON
        curve[day_index] = targets.p_day
        curve[dip_index] = dip_value

    if night_index > dip_index:
        curve[dip_index : night_index + 1] = _shape_between(
            dip_value, targets.p_night, night_index - dip_index + 1,
            None if shape is None else shape[dip_index : night_index + 1],
            targets.ramp_up, targets.ramp_down,
            ceiling=targets.p_night - EPSILON,
        )
    else:
        curve[night_index] = targets.p_night

    # --- D：t_night → 23:50，其中一步恰為 −ramp_down ---
    diffs_d = _shaped_monotone_with_one_extreme_step(
        n_steps=last - night_index,
        total_change=day_end_level - targets.p_night,
        extreme_step=-targets.ramp_down,
        extreme_at=down_start - night_index,
        template=None if shape is None else shape[night_index:],
    )
    curve[night_index + 1 :] = targets.p_night + np.cumsum(diffs_d)

    _enforce_unique_peak(curve, day_index, targets.p_day, day_peak_grid())
    _enforce_unique_peak(curve, night_index, targets.p_night, night_peak_grid())
    return curve


def _enforce_unique_peak(
    curve: np.ndarray, peak_index: int, peak_value: float, grid: list[int]
) -> None:
    """確保窗口內的極大值嚴格唯一且落在 ``peak_index``（就地修改）。

    線性內插可能在尖峰兩側製造與尖峰等值的點（例如尖峰恰在窗口端點時），
    造成並列。並列時規則取**最早**者——若並列點在意圖時刻之前，
    我們就會得到一個錯的 ``t_day``，而且分數會靜默變差。

    作法：把窗口內其他點壓到 ``peak_value − EPSILON`` 以下。
    EPSILON 遠小於量測精度，對評分無可觀測影響。

    Args:
        curve: 長度 144 的陣列。
        peak_index: 尖峰所在索引。
        peak_value: 尖峰值。
        grid: 該窗口的合法格點（分鐘）。
    """
    lo, hi = _index(grid[0]), _index(grid[-1])
    window = curve[lo : hi + 1]
    offset = peak_index - lo
    limit = peak_value - EPSILON
    np.minimum(window, limit, out=window)
    window[offset] = peak_value


def verify_day(curve: np.ndarray, targets: DayTargets | None = None) -> dict[str, float]:
    """由合成曲線反推 6 項目標，供往返檢查。

    這裡刻意**重新實作**推導邏輯（而非呼叫 ``targets.compute_targets``），
    以便在不建 DataFrame 的情況下快速檢查。真正的權威往返驗證由
    ``tests/test_curve.py`` 以 ``compute_targets`` 執行——
    兩者必須一致，任一方寫錯都會被測出來。

    Args:
        curve: 長度 144 的負載序列。
        targets: 保留參數，未使用（推導完全由曲線決定）。天真對照組
            沒有「意圖目標」可傳，故為選填。

    Returns:
        dict: 由曲線推導出的 6 項目標值。

    Raises:
        ValueError: 曲線長度不正確。
    """
    if curve.shape != (POINTS_PER_DAY,):
        raise ValueError(f"曲線長度應為 {POINTS_PER_DAY}，得到 {curve.shape}")

    def extremum(grid: list[int]) -> tuple[float, int]:
        lo, hi = _index(grid[0]), _index(grid[-1])
        window = curve[lo : hi + 1]
        best = int(np.argmax(window))  # argmax 取**最早**的最大值，與並列規則一致
        return float(window[best]), (lo + best) * STEP

    p_day, t_day = extremum(day_peak_grid())
    p_night, t_night = extremum(night_peak_grid())
    diff = np.diff(curve)
    return {
        "p_day": p_day,
        "t_day": t_day,
        "p_night": p_night,
        "t_night": t_night,
        "ramp_up": float(diff.max()),
        "ramp_down": float(-diff.min()),
    }


def synthesize_day_checked(
    targets: DayTargets,
    day_start_level: float,
    day_min_level: float,
    day_end_level: float,
    tolerance: float = 1e-3,
    shape: np.ndarray | None = None,
) -> np.ndarray:
    """合成曲線並立即驗證往返一致，不一致即拋錯。

    這是對外的主要介面。合成若不精確就會**靜默失分**——曲線看起來正常、
    格式檢查也會過，但推導出的目標與意圖不同。因此驗證不是可選項。

    Args:
        targets: 6 項目標值。
        day_start_level: 00:00 的負載水準。
        day_min_level: 日最低負載。
        day_end_level: 23:50 的負載水準。
        tolerance: 量值的容許誤差（MW）。時刻要求完全相等。

    Returns:
        np.ndarray: 長度 144、已驗證的負載序列。

    Raises:
        ValueError: 往返驗證失敗，訊息列出所有不符的項目。
    """
    curve = synthesize_day(
        targets, day_start_level, day_min_level, day_end_level, shape
    )
    realised = verify_day(curve, targets)

    problems = []
    for name in ("t_day", "t_night"):
        if realised[name] != getattr(targets, name):
            problems.append(
                f"{name}: 意圖 {format_hhmm(getattr(targets, name))}、"
                f"實得 {format_hhmm(int(realised[name]))}"
            )
    for name in ("p_day", "p_night", "ramp_up", "ramp_down"):
        gap = abs(realised[name] - getattr(targets, name))
        if gap > tolerance:
            problems.append(
                f"{name}: 意圖 {getattr(targets, name):.4f}、"
                f"實得 {realised[name]:.4f}（差 {gap:.2e}）"
            )
    if problems:
        raise ValueError("曲線合成未能實現目標：\n  " + "\n  ".join(problems))
    return curve
