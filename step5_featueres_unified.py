#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step5 Unified: 特征工程引擎（单Pass架构）

职责：
  1. 接收清洗后的DataFrame
  2. 执行9大特征分组处理
  3. 返回 observations + states + metadata

不负责：
  - 数据读取/保存
  - 真滑窗处理
  - NPZ导出

特征分组（9大类）：
  1. market_state    (SuperTrend → -1/1)
  2. momentum        (滚动Z-score + robust归一化)
  3. band_width      (rank归一化)
  4. volume          (rank归一化)
  5. ATR/RV          (波动率)
  6. RSI             (连续值 + 事件标记)
  7. time_encoding   (sin/cos周期编码)
  8. price_base      (OHLC基准价格)
  9. return          (对数收益率)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


# ============================================
# 辅助函数
# ============================================


def _period_to_minutes(p: str) -> int:
    """转换周期字符串为分钟数"""
    val = int(re.findall(r"\d+", p)[0])
    unit = p[-1]
    if unit == "m":
        return val
    elif unit == "h":
        return val * 60
    elif unit == "d":
        return val * 1440
    return val


def _get_period_multipliers(base_period: str, all_periods: List[str]) -> Dict[str, float]:
    """根据基准周期动态计算各周期相对倍数"""
    base_minutes = _period_to_minutes(base_period)
    multipliers = {}
    for period in all_periods:
        period_minutes = _period_to_minutes(period)
        multipliers[period] = period_minutes / base_minutes
    return multipliers


def _auto_detect_periods(df: pd.DataFrame) -> List[str]:
    """从DataFrame列名中自动识别周期前缀"""
    period_pattern = re.compile(r"^(\d+[mhd])_")
    detected = set()
    for col in df.columns:
        match = period_pattern.match(str(col))
        if match:
            detected.add(match.group(1))
    return sorted(detected, key=_period_to_minutes)


def _detect_rsi_column(df: pd.DataFrame, period: str) -> Optional[str]:
    """查找给定周期的 RSI 列名，兼容 rsi 与 rsi14 等多种命名。"""
    exact = f"{period}_rsi"
    if exact in df.columns:
        return exact
    pattern = re.compile(rf"^{re.escape(period)}_rsi\d+$")
    for col in df.columns:
        if isinstance(col, str) and pattern.match(col):
            return col
    return None


# ============================================
# Numba加速（可选）
# ============================================

_numba_available = False
try:
    from numba import njit  # type: ignore

    _numba_available = True
except Exception:
    pass


if _numba_available:

    @njit(cache=True, fastmath=True)
    def _calc_percentile_rank_nb(series: np.ndarray, window: int) -> np.ndarray:
        """numba加速的滚动秩分位计算"""
        n = series.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            x = series[i]
            less = 0
            equal = 0
            count = 0
            for j in range(start, i + 1):
                v = series[j]
                if v < x:
                    less += 1
                elif v == x:
                    equal += 1
                count += 1
            rank = (less + 0.5 * equal) / max(1, count)
            if rank < 0.01:
                rank = 0.01
            elif rank > 0.99:
                rank = 0.99
            out[i] = rank
        return out


def calc_percentile_rank(series: np.ndarray, window: int) -> np.ndarray:
    """计算滚动平均秩分位（范围约[0.01, 0.99]）"""
    x = np.asarray(series, dtype=np.float64)
    if window <= 1 or x.size == 0:
        return np.full_like(x, 0.5, dtype=float)
    if _numba_available:
        try:
            return _calc_percentile_rank_nb(x, int(window)).astype(float)
        except Exception:
            pass
    # 纯Python后备实现
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        window_vals = x[start : i + 1]
        if window_vals.size == 0:
            out[i] = 0.5
            continue
        xv = x[i]
        less = np.sum(window_vals < xv)
        equal = np.sum(window_vals == xv)
        rank = (less + 0.5 * equal) / window_vals.size
        rank = np.clip(rank, 0.01, 0.99)
        out[i] = rank
    return out


def calc_rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    """
    滚动Z-score标准化（向量化版本，O(N)复杂度）

    🔥 关键修复: 处理NaN值，避免cumsum传播导致全局污染
    """
    v = np.asarray(series, dtype=np.float64)
    n = v.size
    if n == 0:
        return v

    # 记录原始NaN位置
    nan_mask = ~np.isfinite(v)

    # 用0填充NaN用于cumsum（避免NaN传播）
    v_filled = np.where(np.isfinite(v), v, 0.0)

    w = max(1, int(window))
    idx = np.arange(n)
    starts = np.maximum(0, idx - w + 1)
    prev = starts - 1

    csum = np.cumsum(v_filled)
    csum2 = np.cumsum(v_filled * v_filled)

    sum_win = csum - np.where(prev >= 0, csum[prev], 0.0)
    sum2_win = csum2 - np.where(prev >= 0, csum2[prev], 0.0)
    lengths = (idx - starts + 1).astype(np.float64)
    mean = sum_win / np.maximum(1.0, lengths)
    var = sum2_win / np.maximum(1.0, lengths) - mean * mean
    var = np.where(var < 1e-12, 1e-12, var)
    std = np.sqrt(var)

    z = (v_filled - mean) / std

    # 还原原始NaN位置
    z[nan_mask] = np.nan
    z[~np.isfinite(z)] = np.nan

    return z.astype(float)


def _calibrate_one_sided_power(
    x: np.ndarray, p: float = 0.95, target: float = 0.99, epsilon: float = 0.0
) -> np.ndarray:
    """单边幂变换校准到[0,1]：
    - 令 q = quantile(x, p) (0<q<1)，求 γ 使 q^γ = target ⇒ γ = ln(target)/ln(q)
    - y = clip(x,0,1)^γ，再线性夹紧到[ε, 1-ε]
    保持单调且避免大量点饱和在1。
    """
    v = np.asarray(x, dtype=float)
    finite_mask = np.isfinite(v)
    y = np.zeros_like(v, dtype=float)
    if finite_mask.any():
        q = float(np.quantile(v[finite_mask], p))
        q = float(np.clip(q, 1e-6, 1 - 1e-6))
        # 若q≈1，避免除零：退化为γ=1
        if abs(1.0 - q) < 1e-6:
            gamma = 1.0
        else:
            gamma = float(np.log(max(target, 1e-6)) / np.log(q))
        y[finite_mask] = np.power(np.clip(v[finite_mask], 0.0, 1.0), gamma)
    y = np.clip(y, 0.0 + max(0.0, float(epsilon)), 1.0 - max(0.0, float(epsilon)))
    return y.astype(float)


def _apply_fixed_power_calibration(x: np.ndarray, gamma: float, epsilon: float = 0.0) -> np.ndarray:
    """使用持久化的幂系数进行单边压缩，保持训练/实盘一致。"""
    v = np.asarray(x, dtype=float)
    gamma = float(gamma) if np.isfinite(gamma) else 1.0
    clipped = np.clip(v, 0.0, 1.0)
    y = np.power(clipped, max(1e-6, gamma))
    return np.clip(y, 0.0 + max(0.0, float(epsilon)), 1.0 - max(0.0, float(epsilon)))


def _get_series(df: pd.DataFrame, target_period: str, suffix: str, roll_mode: bool, base_period: str) -> np.ndarray:
    """
    获取指定周期和后缀的序列数据

    优先级：
    - roll_mode=True: {period}_{suffix}_roll → {period}_{suffix}
    - roll_mode=False: {period}_{suffix}_fixed → {period}_{suffix}
    - 兜底: {base_period}_{suffix}
    """
    if roll_mode:
        cand = [f"{target_period}_{suffix}_roll", f"{target_period}_{suffix}"]
    else:
        cand = [f"{target_period}_{suffix}_fixed", f"{target_period}_{suffix}"]

    for col in cand:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(float)
            return np.where(~np.isfinite(series), 0.0, series)

    # 兜底：roll模式下尝试基准周期
    if roll_mode and target_period != base_period:
        fallback = f"{base_period}_{suffix}"
        if fallback in df.columns:
            series = pd.to_numeric(df[fallback], errors="coerce").fillna(0.0).values.astype(float)
            return np.where(~np.isfinite(series), 0.0, series)

    return np.zeros(len(df), dtype=float)


# ============================================
# 特征组1: Market State
# ============================================


def calc_market_state_label(df: pd.DataFrame, period: str) -> np.ndarray:
    """
    计算市场状态标签（二分类：-1/1）

    基于SuperTrend方向：
    - 优先级: {period}_supertrend_direction_roll > _fixed > 无后缀
    - direction >= 0 → 1 (涨势)
    - direction < 0  → -1 (跌势)
    """
    st_candidates = [
        f"{period}_supertrend_direction_roll",
        f"{period}_supertrend_direction_fixed",
        f"{period}_supertrend_direction",
    ]
    for col in st_candidates:
        if col in df.columns:
            try:
                arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(float)
                return np.where(arr >= 0.0, 1, -1).astype(int)
            except Exception:
                continue

    print(f"⚠️ {period}: 未找到SuperTrend direction列，返回全1占位")
    return np.ones(len(df), dtype=int)


def analyze_state_distribution(states: np.ndarray, period: str) -> None:
    """分析并打印市场状态分布"""
    unique, counts = np.unique(states, return_counts=True)
    total = len(states)

    state_names = {-1: "跌势", 0: "震荡", 1: "涨势"}

    print(f"\n📊 {period} 市场状态分布:")
    for state, count in zip(unique, counts):
        percentage = count / total * 100
        name = state_names.get(state, f"未知({state})")
        print(f"  {name}: {count:,} 样本 ({percentage:.1f}%)")

    if len(unique) == 2:
        min_ratio = min(counts) / max(counts)
        if min_ratio < 0.1:
            print("  ⚠️ 极度不平衡")
        elif min_ratio < 0.3:
            print("  ⚠️ 不够平衡")
        else:
            print("  ✅ 分布尚可")


# ============================================
# 特征组2: Momentum (真滑窗版本)
# ============================================


def calc_rolling_window_momentum(
    df: pd.DataFrame,
    period: str,
    window_minutes: int,
    mom_lookback: int,
    ref_method: str = "boundary",
    anchor_offset: int = 0,
) -> np.ndarray:
    """
    真滑窗计算momentum（无未来信息泄露）

    定义（以30m为例，L=14）:
    - point法（默认）: mom_t = close[t] / close[t - L*30m] - 1
    - boundary法（对齐到30m边界，但不看未来）:
        cur_idx = floor(t/30m)*30m
        mom_t = close[cur_idx] / close[cur_idx - L*30m] - 1
    - boundary_intra_avg（对齐边界+区间均值填充）:
        分子: 当前30m区间内的“基准步长”收盘均值（边界起点→t）
        分母: close[t - (L*30m - 30m)]  # 例：L=14 → 390分钟

    Args:
        df: 包含至少一个 "<周期>_close" 列的DataFrame（索引是时间戳）
        period: 周期字符串 (如 '30m', '2h')
        window_minutes: 周期对应的分钟数（30m->30, 2h->120）
        mom_lookback: momentum回溯窗口（默认14）
        ref_method: 参考法（point|boundary）

    Returns:
        momentum数组（原始百分比收益率，如0.05表示5%）
    """
    # 选择可用的基准收盘序列：自动选择列中存在的“最小周期”的 *_close
    import re as _re

    base_close_col = None
    base_period_detected = None
    candidates = [str(c) for c in df.columns if _re.match(r"^\d+[mhd]_close$", str(c))]
    if candidates:

        def _mins(name: str) -> int:
            return _period_to_minutes(name.split("_")[0])

        candidates.sort(key=_mins)
        base_close_col = candidates[0]
        base_period_detected = base_close_col.split("_")[0]
    if base_close_col is None:
        return np.zeros(len(df), dtype=float)

    close_series = pd.to_numeric(df[base_close_col], errors="coerce").ffill().fillna(0.0).values
    n = len(close_series)

    # 以基准步长(分钟)换算目标周期步数与回溯步数
    base_step_minutes = _period_to_minutes(base_period_detected)
    target_minutes = max(1, int(window_minutes))
    steps_per_target = max(1, int(round(float(target_minutes) / float(base_step_minutes))))
    lookback_steps = int(mom_lookback) * steps_per_target

    out = np.full(n, np.nan, dtype=float)

    method = str(ref_method or "boundary").lower().strip()
    anchor_steps = int(round(max(0, int(anchor_offset)) / max(1, base_step_minutes))) % max(1, steps_per_target)
    min_start = lookback_steps + anchor_steps

    if method == "boundary":
        for i in range(min_start, n):
            # 对齐到目标周期边界（按步长），支持锚点
            cur_idx = ((i - anchor_steps) // steps_per_target) * steps_per_target + anchor_steps
            past_idx = cur_idx - lookback_steps
            if cur_idx < n and past_idx >= 0:
                cur = close_series[cur_idx]
                prev = close_series[past_idx]
                if np.isfinite(cur) and np.isfinite(prev) and prev != 0.0:
                    out[i] = (cur / prev) - 1.0
    elif method == "boundary_intra_avg":
        # 需要至少 (L*P - P) 的参考位移（步）
        ref_shift_minutes = lookback_steps * base_step_minutes - target_minutes
        ref_shift_steps = max(0, int(round(float(ref_shift_minutes) / float(base_step_minutes))))
        min_start2 = max(min_start, ref_shift_steps + anchor_steps)
        # 预计算累计和用于快速均值
        cs = np.cumsum(np.nan_to_num(close_series, nan=0.0))
        for i in range(min_start2, n):
            # 当前周期边界起点（步）
            start_idx = ((i - anchor_steps) // steps_per_target) * steps_per_target + anchor_steps
            if start_idx > i:
                continue
            # 区间均值（start_idx..i）
            total = cs[i] - (cs[start_idx - 1] if start_idx > 0 else 0.0)
            length = float(i - start_idx + 1)
            cur_avg = total / max(1.0, length)
            # 参考价：t - (L*P - P)（步）
            past_idx = i - ref_shift_steps
            if past_idx >= 0:
                prev = close_series[past_idx]
                if np.isfinite(cur_avg) and np.isfinite(prev) and prev != 0.0:
                    out[i] = (cur_avg / prev) - 1.0
    else:
        for i in range(min_start, n):
            past_idx = i - lookback_steps
            if past_idx >= 0:
                cur = close_series[i]
                prev = close_series[past_idx]
                if np.isfinite(cur) and np.isfinite(prev) and prev != 0.0:
                    out[i] = (cur / prev) - 1.0

    out = pd.Series(out).ffill().fillna(0.0).values.astype(float)
    return out


def calc_momentum_feature(
    df: pd.DataFrame,
    period: str,
    cfg: Dict,
    roll_mode: bool,
    base_period: str,
    period_multipliers: Dict[str, float],
) -> Tuple[np.ndarray, str]:
    """
    计算动量特征（真滑窗 + 滚动Z-score + robust归一化）

    🔥 关键改进: 不再依赖merged.parquet中的阶跃mom列
               而是基于最小可用步长的 close 序列重新滚动计算，保证信号平稳连续

    返回: (归一化后的数据, 实际使用的列名)
    """
    # 读取配置
    norm_cfg = cfg.get("normalization", {})
    mcfg = cfg.get("momentum", norm_cfg.get("momentum", {}))

    # Momentum回溯窗口（计算当前价格相对N期前的涨跌幅）
    mom_lookback = int(mcfg.get("default_mom_window", 14))

    # Z-score窗口
    mom_base = int(mcfg.get("zscore_window", 50))
    use_mult_for_z = bool(mcfg.get("use_period_multipliers_for_zscore", False))
    window = int(mom_base * period_multipliers.get(period, 1)) if use_mult_for_z else mom_base

    # 归一化/校准参数
    norm_method = str(mcfg.get("norm_method", "robust_zscore")).lower()
    robust_k = float(mcfg.get("robust_k", 1.4826))
    # 分位校准（确保极端≈5%贴近1）
    calib_cfg = mcfg.get("calibration") or {}
    calib_method = str(calib_cfg.get("method", "quantile_clip")).lower().strip()
    calib_p = float(calib_cfg.get("p", 0.95))
    calib_target = float(calib_cfg.get("target", 0.99))

    # 🔥 首选 merged 源：尽量贴合 Step3/Step4 的 {period}_mom 值
    src_mode = str(mcfg.get("source", "merged")).lower().strip()
    out: np.ndarray
    source_used: str
    merged_col: str | None = None
    if src_mode in ("merged", "auto"):
        exact = f"{period}_mom"
        if exact in df.columns:
            merged_col = exact
        else:
            import re as _re

            cands = [c for c in df.columns if _re.match(rf"^{_re.escape(period)}_mom\d+$", str(c))]
            if cands:
                merged_col = str(cands[0])
            else:
                for c in df.columns:
                    if _re.match(rf"^{_re.escape(period)}_.*mom.*$", str(c)):
                        merged_col = str(c)
                        break
    if merged_col is not None:
        try:
            out = pd.to_numeric(df[merged_col], errors="coerce").fillna(0.0).values.astype(float)
            source_used = merged_col
        except Exception:
            out = None  # type: ignore
    else:
        out = None  # type: ignore

    if out is None:
        # 回退：真滑窗计算
        window_minutes = _period_to_minutes(period)
        ref_method = str(mcfg.get("momentum_ref_method", "boundary")).lower().strip()
        # 支持全局数值或按周期覆写：
        # boundary_anchor_offset: 0 | { '30m': 0, '2h': 0 }
        anchor_cfg = mcfg.get("boundary_anchor_offset", 0)
        try:
            if isinstance(anchor_cfg, dict):
                anchor_offset = int(anchor_cfg.get(period, 0))
            else:
                anchor_offset = int(anchor_cfg)
        except Exception:
            anchor_offset = 0
        raw_momentum = calc_rolling_window_momentum(
            df, period, window_minutes, mom_lookback, ref_method=ref_method, anchor_offset=anchor_offset
        )
        out = raw_momentum
        source_used = f"{period}_mom_sliding"

    # 🔥 新算法：Tanh压缩（以0轴为中心，保留正负方向）
    finite_mask = np.isfinite(out)
    normalized = np.zeros_like(out, dtype=float)

    if finite_mask.any():
        # 固定scale：典型动量±5%映射到tanh(1.5) ≈ ±0.905
        # ±10%映射到tanh(3) ≈ ±0.995
        scale = 30.0  # 调节灵敏度

        # Tanh压缩：保留正负方向，压制极端
        normalized[finite_mask] = np.tanh(out[finite_mask] * scale)

    # NaN填充
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)

    # 统计极端占比（|x|≥target）
    try:
        extreme_ratio = float(np.mean(np.abs(normalized) >= calib_target))
        print(
            f"   └─ 均值={np.mean(normalized):.4f}, 标准差={np.std(normalized):.4f}, 范围=[{np.min(normalized):.4f}, {np.max(normalized):.4f}], 极端(|x|>={calib_target:.2f})={extreme_ratio*100:.2f}%"
        )
    except Exception:
        print(
            f"   └─ 均值={np.mean(normalized):.4f}, 标准差={np.std(normalized):.4f}, 范围=[{np.min(normalized):.4f}, {np.max(normalized):.4f}]"
        )

    return normalized, source_used


# ============================================
# 特征组3: Band Width
# ============================================


def calc_band_width_feature(
    df: pd.DataFrame,
    period: str,
    cfg: Dict,
    roll_mode: bool,
    base_period: str,
    period_multipliers: Dict[str, float],
) -> Tuple[np.ndarray, str]:
    """计算布林带宽度特征（rank归一化）"""
    bw_cfg = cfg.get("band_width", {})
    fast_base = int(bw_cfg.get("fast_base", 25))
    slow_base = int(bw_cfg.get("slow_base", 100))
    fuse_w_fast = float(bw_cfg.get("fuse_w_fast", 0.6))
    fuse_w_slow = float(bw_cfg.get("fuse_w_slow", 0.4))
    epsilon = float(bw_cfg.get("shrink_epsilon", 0.03))

    fast_window = int(fast_base * period_multipliers.get(period, 1))
    slow_window = int(slow_base * period_multipliers.get(period, 1))

    suffix = "bb_width"
    bandwidth = _get_series(df, period, suffix, roll_mode, base_period)

    if roll_mode and period != base_period and f"{base_period}_{suffix}" in df.columns:
        source_used = f"{base_period}_{suffix} (roll)"
    elif f"{period}_{suffix}" in df.columns:
        source_used = f"{period}_{suffix}"
    else:
        # 🔥 修复：使用带周期前缀的列名，避免重复的"zeros"列名
        source_used = f"{period}_bb_width"

    bw = np.log1p(np.maximum(bandwidth, 0.0))

    rank_fast = calc_percentile_rank(bw, fast_window)
    rank_slow = calc_percentile_rank(bw, slow_window)
    rank_fused = fuse_w_fast * rank_fast + fuse_w_slow * rank_slow

    out = rank_fused * (1.0 - 2.0 * epsilon) + epsilon

    print(f"✅ {period}: band_width (fast={fast_window}, slow={slow_window})")

    return out, source_used


# ============================================
# 特征组4: Volume
# ============================================


def _aggregate_volume_from_base(df: pd.DataFrame, target_period: str, base_period: str) -> Tuple[np.ndarray, str]:
    """
    使用基础周期的成交量滚动聚合出较大周期的volume

    逻辑：
        target_period = 15m，base_period = 3m
        steps = 15 / 3 = 5
        15m_volume[t] = sum(最近5个3m_volume)
    """
    base_col = f"{base_period}_volume"
    if base_col not in df.columns:
        print(f"      ⚠️ 未找到基础成交量列 {base_col}，无法聚合 {target_period}_volume")
        return np.zeros(len(df), dtype=float), f"{base_period}_volume_missing"

    base_minutes = _period_to_minutes(base_period)
    target_minutes = _period_to_minutes(target_period)
    steps = max(1, int(round(target_minutes / max(1, base_minutes))))

    base_series = pd.to_numeric(df[base_col], errors="coerce").fillna(0.0)
    aggregated = base_series.rolling(window=steps, min_periods=1).sum().values.astype(float)

    if target_period == base_period:
        column_name = f"{target_period}_volume"
    else:
        column_name = f"{base_period}_volume→{target_period}"
    return aggregated, column_name


def calc_volume_feature(
    df: pd.DataFrame,
    period: str,
    cfg: Dict,
    roll_mode: bool,
    base_period: str,
    period_multipliers: Dict[str, float],
) -> Tuple[np.ndarray, str]:
    """计算成交量特征（rank归一化）"""
    vol_base = int(cfg.get("volume", {}).get("rank_window_base", 100))
    window = int(vol_base * period_multipliers.get(period, 1))

    suffix = "volume"
    vol = _get_series(df, period, suffix, roll_mode, base_period)

    if roll_mode and period != base_period and f"{base_period}_{suffix}" in df.columns:
        source_used = f"{base_period}_{suffix} (roll)"
    elif f"{period}_{suffix}" in df.columns:
        source_used = f"{period}_{suffix}"
    else:
        # 🔥 修复：使用带周期前缀的列名，避免重复的"zeros"列名
        source_used = f"{period}_volume"

    # 当目标周期缺失volume列时，_get_series会返回全0数组
    # 此时用基础周期volume滚动聚合，避免出现恒定值
    if (np.nanmax(vol) - np.nanmin(vol)) == 0.0:
        vol, source_used = _aggregate_volume_from_base(df, period, base_period)

    vol[np.isnan(vol)] = 0.0
    vol_ln = np.log1p(np.maximum(vol, 0.0))

    out = calc_percentile_rank(vol_ln, window)

    print(f"✅ {period}: volume (rank_window={window})")

    return out, source_used


# ============================================
# 主特征引擎类
# ============================================


class UnifiedFeatureEngine:
    """统一特征工程引擎（单Pass架构）"""

    def __init__(self, cfg: Dict, base_period: str = "1m"):
        self.cfg = cfg
        self.base_period = base_period
        self.periods: List[str] = []
        self.period_multipliers: Dict[str, float] = {}
        self.roll_mode: bool = False

        # 🔥 归一化配置（从cfg中提取，用于RSI等特征）
        self.norm_config = cfg.get("normalization", {})

        # 结果容器
        self.observations_list: List[np.ndarray] = []
        self.observations_names: List[str] = []
        self.states_list: List[np.ndarray] = []
        self.states_names: List[str] = []
        self.states_types: List[str] = []
        self.num_classes: List[int] = []
        self.timestamps: Optional[np.ndarray] = None
        self.prices: Optional[np.ndarray] = None

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        单Pass处理流程

        Args:
            df: 输入DataFrame（已清洗、已滑窗）

        Returns:
            Dict包含observations, states, metadata等
        """
        print("\n" + "=" * 70)
        print("🚀 特征工程处理")
        print("=" * 70)

        # 1. 自动识别周期
        self.periods = _auto_detect_periods(df)
        cfg_periods = self.cfg.get("periods")
        if cfg_periods:
            self.periods = [p for p in self.periods if p in cfg_periods]

        if not self.periods:
            raise RuntimeError("未识别到任何周期")

        print(f"✅ 识别周期: {self.periods}")

        # 2. 计算周期倍数
        self.period_multipliers = _get_period_multipliers(self.base_period, self.periods)
        print(f"✅ 周期倍数: {self.period_multipliers}")

        # 3. 确定roll模式
        rl_build_cfg = self.cfg.get("rl_build", {})
        source_mode = rl_build_cfg.get("source_mode", "fixed")
        self.roll_mode = source_mode == "sliding"
        print(f"✅ 模式: {'滑窗(roll)' if self.roll_mode else '固定(close)'}")

        # 4. 提取时间戳和价格
        self._extract_timestamps_and_prices(df)

        # 5. 循环处理每个周期
        for period in self.periods:
            print(f"\n🔄 处理 {period} 周期...")
            self._process_period(df, period)

        # 6. 处理RSI
        self._process_rsi_features(df)

        # 7. 处理时间编码
        self._process_time_encoding()

        # 8. 处理基准价格
        self._process_base_prices(df)

        # 9. 处理收益率
        self._process_returns(df)

        # 10. 组装结果
        return self._assemble_results()

    def _extract_timestamps_and_prices(self, df: pd.DataFrame) -> None:
        """提取时间戳和价格"""
        if isinstance(df.index, pd.DatetimeIndex):
            self.timestamps = df.index.values
        elif "timestamp" in df.columns:
            self.timestamps = pd.to_datetime(df["timestamp"]).values
        else:
            self.timestamps = np.arange(len(df))

        price_col = f"{self.base_period}_close"
        if price_col in df.columns:
            self.prices = df[price_col].values.astype(float)
        elif "close" in df.columns:
            self.prices = df["close"].values.astype(float)
        else:
            self.prices = np.zeros(len(df))

        print(f"✅ 时间戳: {len(self.timestamps)} 个, 价格基准: {price_col}")

    def _process_period(self, df: pd.DataFrame, period: str) -> None:
        """处理单个周期的所有特征"""

        # 1️⃣ Market State
        states = calc_market_state_label(df, period)
        self.states_list.append(states.reshape(-1, 1))
        self.states_names.append(f"{period}_market_state")  # 🔧 简化命名：移除冗余的 now_ 和 _win100
        self.states_types.append("classification")
        self.num_classes.append(2)
        analyze_state_distribution(states, period)

        # 2️⃣ Momentum
        momentum, mom_col = calc_momentum_feature(
            df, period, self.cfg, self.roll_mode, self.base_period, self.period_multipliers
        )
        self.states_list.append(momentum.reshape(-1, 1))
        self.states_names.append(mom_col)
        self.states_types.append("regression")
        self.num_classes.append(1)

        # 3️⃣ Band Width
        band_width, bw_col = calc_band_width_feature(
            df, period, self.cfg, self.roll_mode, self.base_period, self.period_multipliers
        )
        self.states_list.append(band_width.reshape(-1, 1))
        self.states_names.append(bw_col)
        self.states_types.append("regression")
        self.num_classes.append(1)

        # 4️⃣ Volume
        volume, vol_col = calc_volume_feature(
            df, period, self.cfg, self.roll_mode, self.base_period, self.period_multipliers
        )
        self.states_list.append(volume.reshape(-1, 1))
        self.states_names.append(vol_col)
        self.states_types.append("regression")
        self.num_classes.append(1)

        # 5️⃣ ATR
        self._process_atr(df, period)

        # 6️⃣ RV
        self._process_rv(df, period)

    def _process_atr(self, df: pd.DataFrame, period: str) -> None:
        """处理ATR特征"""
        atr_pct_col = f"{period}_atr_pct"
        atr_col = f"{period}_atr"
        close_col = f"{period}_close"

        if atr_pct_col in df.columns:
            base = pd.to_numeric(df[atr_pct_col], errors="coerce").fillna(0.0).values.astype(float)
        elif atr_col in df.columns and close_col in df.columns:
            atr = pd.to_numeric(df[atr_col], errors="coerce").fillna(0.0).values.astype(float)
            close = pd.to_numeric(df[close_col], errors="coerce").fillna(1e-8).values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                base = atr / np.where(close == 0.0, np.nan, close)
            base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return

        atr_cfg = self.cfg.get("atr", {})
        fast_base = int(atr_cfg.get("fast_base", 25))
        slow_base = int(atr_cfg.get("slow_base", 100))
        fuse_w_fast = float(atr_cfg.get("fuse_w_fast", 0.6))
        fuse_w_slow = float(atr_cfg.get("fuse_w_slow", 0.4))
        epsilon = float(atr_cfg.get("shrink_epsilon", 0.03))

        x = np.log1p(np.maximum(base, 0.0))
        finite_x = x[np.isfinite(x)]
        if finite_x.size > 0:
            try:
                q_low, q_high = np.percentile(finite_x, [2.0, 98.0])
                if q_high > q_low:
                    x = np.clip(x, q_low, q_high)
            except Exception:
                pass

        fast_window = int(fast_base * self.period_multipliers.get(period, 1))
        slow_window = int(slow_base * self.period_multipliers.get(period, 1))
        rank_fast = calc_percentile_rank(x, fast_window)
        rank_slow = calc_percentile_rank(x, slow_window)
        fused = fuse_w_fast * rank_fast + fuse_w_slow * rank_slow

        # 单边幂校准（确保上5%接近1）
        calib_cfg = atr_cfg.get("calibration") or {}
        calib_p = float(calib_cfg.get("p", 0.95))
        calib_target = float(calib_cfg.get("target", 0.99))
        per_period_cfg = (calib_cfg.get("per_period") or {}).get(period) or {}
        fixed_gamma = per_period_cfg.get("fixed_gamma", calib_cfg.get("fixed_gamma"))
        fixed_quantile = per_period_cfg.get("fixed_quantile", calib_cfg.get("fixed_quantile"))

        if fixed_gamma is not None:
            gamma_val = float(fixed_gamma)
            out = _apply_fixed_power_calibration(fused, gamma_val, epsilon)
            print(f"   ℹ️  {period}: atr_pct 使用固定 gamma={gamma_val:.4f}")
        elif fixed_quantile is not None:
            q = float(fixed_quantile)
            q = float(np.clip(q, 1e-6, 1.0 - 1e-6))
            gamma_val = float(np.log(max(calib_target, 1e-6)) / np.log(q))
            out = _apply_fixed_power_calibration(fused, gamma_val, epsilon)
            print(f"   ℹ️  {period}: atr_pct 使用固定 quantile={q:.4f} 计算 gamma={gamma_val:.4f}")
        else:
            finite_mask = np.isfinite(fused)
            if finite_mask.any():
                q = float(np.quantile(fused[finite_mask], calib_p))
                q = float(np.clip(q, 1e-6, 1.0 - 1e-6))
                gamma_val = float(np.log(max(calib_target, 1e-6)) / np.log(q))
                out = _apply_fixed_power_calibration(fused, gamma_val, epsilon)
                print(f"   ℹ️  {period}: atr_pct 动态 gamma={gamma_val:.4f} (quantile={q:.4f})")
            else:
                out = _calibrate_one_sided_power(fused, p=calib_p, target=calib_target, epsilon=epsilon)

        self.states_list.append(out.reshape(-1, 1))
        self.states_names.append(f"{period}_atr_pct")
        self.states_types.append("regression")
        self.num_classes.append(1)
        print(f"✅ {period}: atr_pct")

    def _process_rv(self, df: pd.DataFrame, period: str) -> None:
        """处理Realized Volatility特征"""
        from tools.columns import find_columns_by_pattern

        rv_cols = find_columns_by_pattern(df, r"(?:rv(?:_?win)?\d+|rv\b)", period=period)

        if not rv_cols:
            return

        for rv_col in rv_cols:
            canonical_name = re.sub(rf"^({period})_rv(?:_?win)?\d+(?:_(?:fixed|roll))?$", r"\1_rv", str(rv_col))
            if canonical_name in self.states_names:
                continue

            rv_vals = pd.to_numeric(df[rv_col], errors="coerce").fillna(0.0).values.astype(float)
            self.states_list.append(rv_vals.reshape(-1, 1))
            self.states_names.append(canonical_name)
            self.states_types.append("regression")
            self.num_classes.append(1)
            print(f"✅ {period}: {canonical_name}")
            break

    def _process_rsi_features(self, df: pd.DataFrame) -> None:
        """
        处理RSI特征

        🔥 重要：RSI超买超卖事件已在Step3计算，这里直接读取
           - Step3输出: rsi14, rsi_overbought, rsi_oversold, rsi_event
           - Step5任务: 归一化RSI值，直接读取事件列
        """
        print("\n✅ 处理RSI特征...")

        # 🔥 处理所有周期的 RSI（包括基础周期）
        # 原因：训练和实盘都需要完整的 RSI 特征集
        rsi_periods = self.periods

        for rsi_period in rsi_periods:
            rsi_col = _detect_rsi_column(df, rsi_period)
            base_rsi_col = _detect_rsi_column(df, self.base_period)
            rsi_vals = None

            # 🔥 滚动模式兼容（已弃用，保留兼容性）
            if self.roll_mode and rsi_period != self.base_period and base_rsi_col:
                win = int(self.period_multipliers.get(rsi_period, 1))
                base_rsi = pd.to_numeric(df[base_rsi_col], errors="coerce").fillna(50.0).values.astype(float)
                rsi_vals = self._sma(base_rsi, max(1, win))
            elif rsi_col:
                rsi_vals = pd.to_numeric(df[rsi_col], errors="coerce").fillna(50.0).values.astype(float)

            if rsi_vals is not None:
                # 🔥 1. RSI归一化（-1到1范围，用于RL模型输入）
                # 注意：RSI已在Step3改造为[-100, +100]范围，0为中性点
                # 🚀 SAC优化：使用可配置的缩放系数（默认80），放大信号分辨率
                # 原理：实际RSI很少超过±80，用80作为分母可以让常用区间[-60,+60]
                #      充分利用[-1,1]空间，提升SAC的梯度敏感度
                # 效果：RSI=+80 → 1.0 (满格多头), RSI=+60 → 0.75 (强多头)
                #      RSI=-80 → -1.0 (满格空头), RSI=-60 → -0.75 (强空头)
                rsi_cfg = self.norm_config.get("rsi", {})
                divisor = float(rsi_cfg.get("normalization_divisor", 80.0))  # 默认80

                rsi_norm = rsi_vals / divisor
                rsi_norm = np.clip(rsi_norm, -1.0, 1.0)  # 极值截断（抗噪）

                self.states_list.append(rsi_norm.reshape(-1, 1))
                self.states_names.append(f"{rsi_period}_rsi")
                self.states_types.append("regression")
                self.num_classes.append(1)

                # 🔥 2. 直接读取Step3生成的事件列（不再重新计算）
                # 优先读取新格式 rsi_event
                event_col = f"{rsi_period}_rsi_event"
                ob_col = f"{rsi_period}_rsi_overbought"
                os_col = f"{rsi_period}_rsi_oversold"

                has_event = event_col in df.columns
                has_ob_os = (ob_col in df.columns) and (os_col in df.columns)

                if has_event:
                    # 新格式：直接读取 rsi_event (-1/0/+1)
                    rsi_event = pd.to_numeric(df[event_col], errors="coerce").fillna(0.0).values.astype(float)
                    self.states_list.append(rsi_event.reshape(-1, 1))
                    self.states_names.append(f"{rsi_period}_rsi_event")
                    self.states_types.append("classification")
                    self.num_classes.append(3)  # -1, 0, +1
                    print(f"   └─ {rsi_period}_rsi + rsi_event (从Step3读取)")

                if has_ob_os:
                    # 旧格式兼容：读取 overbought/oversold (0/1)
                    overbought = pd.to_numeric(df[ob_col], errors="coerce").fillna(0.0).values.astype(float)
                    oversold = pd.to_numeric(df[os_col], errors="coerce").fillna(0.0).values.astype(float)
                    self.states_list.extend([overbought.reshape(-1, 1), oversold.reshape(-1, 1)])
                    self.states_names.extend([f"{rsi_period}_rsi_overbought", f"{rsi_period}_rsi_oversold"])
                    self.states_types.extend(["classification", "classification"])
                    self.num_classes.extend([2, 2])
                    if not has_event:  # 只在没有新格式时打印
                        print(f"   └─ {rsi_period}_rsi + events (从Step3读取，旧格式)")

                if not has_event and not has_ob_os:
                    # 🚨 兼容旧数据：如果Step3未生成事件列，回退到本地计算
                    print(f"   ⚠️  {rsi_period}: Step3未生成事件列，回退到本地计算")

                    # 🔥 从配置读取阈值（兼容新版 RSI: -100 to +100, 阈值 ±40）
                    rsi_cfg = self.cfg.get("rsi", {})
                    min_persist = int(rsi_cfg.get("min_persist", 2))
                    upper_threshold = float(rsi_cfg.get("upper_threshold", 40.0))  # 新版默认 40
                    lower_threshold = float(rsi_cfg.get("lower_threshold", -40.0))  # 新版默认 -40

                    # 计算新格式
                    rsi_event = self._compute_rsi_event(rsi_vals, upper_threshold, lower_threshold, min_persist)
                    self.states_list.append(rsi_event.reshape(-1, 1))
                    self.states_names.append(f"{rsi_period}_rsi_event")
                    self.states_types.append("classification")
                    self.num_classes.append(3)

                    # 计算旧格式
                    overbought, oversold = self._compute_rsi_signal(
                        rsi_vals, upper_threshold, lower_threshold, min_persist
                    )
                    self.states_list.extend([overbought.reshape(-1, 1), oversold.reshape(-1, 1)])
                    self.states_names.extend([f"{rsi_period}_rsi_overbought", f"{rsi_period}_rsi_oversold"])
                    self.states_types.extend(["classification", "classification"])
                    self.num_classes.extend([2, 2])

                    print(f"   └─ {rsi_period}_rsi + events (本地计算，阈值: {lower_threshold}/{upper_threshold})")

    def _sma(self, values: np.ndarray, window: int) -> np.ndarray:
        """简易SMA"""
        if window <= 1:
            v = np.asarray(values, dtype=float)
            v[~np.isfinite(v)] = 0.0
            return v
        v = np.asarray(values, dtype=float)
        v[~np.isfinite(v)] = 0.0
        out = np.zeros_like(v, dtype=float)
        csum = np.cumsum(v)
        for i in range(len(v)):
            start = max(0, i - window + 1)
            total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
            out[i] = total / (i - start + 1)
        out[~np.isfinite(out)] = 0.0
        return out

    def _compute_rsi_event(self, rsi_vals: np.ndarray, upper: float, lower: float, min_persist: int) -> np.ndarray:
        """
        🔥 RSI事件（单列三值输出：-1/0/+1）

        返回格式：
            +1 = 超买触发（看空信号，应减多仓或开空仓）
             0 = 中性（未触发任何事件）
            -1 = 超卖触发（看多信号，应减空仓或开多仓）

        优势：
            - 单列表示，更直观
            - -1/+1 对称，符合"对立事件"语义
            - 避免0值被误认为"无信号"
        """
        rsi = np.asarray(rsi_vals, dtype=float)
        ob = (rsi >= upper).astype(int)
        os = (rsi <= lower).astype(int)

        def _persist(mask: np.ndarray) -> np.ndarray:
            """持续性过滤：必须连续min_persist个周期才触发"""
            if min_persist <= 1:
                return mask
            out = np.zeros_like(mask)
            run = 0
            for i, v in enumerate(mask):
                run = run + 1 if v else 0
                if run >= min_persist:
                    out[i] = 1
            return out

        ob_filtered = _persist(ob)
        os_filtered = _persist(os)

        # 🔥 合并为单列三值
        event = np.zeros_like(rsi, dtype=float)
        event[ob_filtered == 1] = 1.0  # 超买 → +1
        event[os_filtered == 1] = -1.0  # 超卖 → -1

        return event

    def _compute_rsi_signal(
        self, rsi_vals: np.ndarray, upper: float, lower: float, min_persist: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RSI超买/超卖信号（旧格式，保留兼容性）

        返回格式：
            超买列: 1=超买触发, 0=未触发
            超卖列: 1=超卖触发, 0=未触发
        """
        rsi = np.asarray(rsi_vals, dtype=float)
        ob = (rsi >= upper).astype(int)
        os = (rsi <= lower).astype(int)

        def _persist(mask: np.ndarray) -> np.ndarray:
            """持续性过滤：必须连续min_persist个周期才触发"""
            if min_persist <= 1:
                return mask
            out = np.zeros_like(mask)
            run = 0
            for i, v in enumerate(mask):
                run = run + 1 if v else 0
                if run >= min_persist:
                    out[i] = 1
            return out

        return _persist(ob), _persist(os)

    def _process_time_encoding(self) -> None:
        """处理时间编码"""
        print("\n✅ 处理时间编码...")

        try:
            ts = pd.to_datetime(self.timestamps, utc=True)
            day_of_week = ts.dayofweek.values if hasattr(ts, "dayofweek") else np.array([0] * len(ts))
            hour_of_day = ts.hour.values if hasattr(ts, "hour") else np.array([0] * len(ts))

            day_theta = 2.0 * np.pi * (day_of_week.astype(float) % 7.0) / 7.0
            day_sin = np.sin(day_theta)
            day_cos = np.cos(day_theta)
            self.states_list.extend([day_sin.reshape(-1, 1), day_cos.reshape(-1, 1)])
            self.states_names.extend(["time_day_sin", "time_day_cos"])
            self.states_types.extend(["regression", "regression"])
            self.num_classes.extend([1, 1])

            hour_theta = 2.0 * np.pi * (hour_of_day.astype(float) % 24.0) / 24.0
            hour_sin = np.sin(hour_theta)
            hour_cos = np.cos(hour_theta)
            self.states_list.extend([hour_sin.reshape(-1, 1), hour_cos.reshape(-1, 1)])
            self.states_names.extend(["time_hour_sin", "time_hour_cos"])
            self.states_types.extend(["regression", "regression"])
            self.num_classes.extend([1, 1])

            print("   └─ sin/cos编码完成")
        except Exception as e:
            print(f"⚠️ 时间编码失败: {e}")

    def _process_base_prices(self, df: pd.DataFrame) -> None:
        """处理基准价格"""
        print(f"\n✅ 处理基准价格（{self.base_period}）...")

        price_cols = [
            f"{self.base_period}_open",
            f"{self.base_period}_high",
            f"{self.base_period}_low",
            f"{self.base_period}_close",
        ]

        for price_col in price_cols:
            if price_col in df.columns:
                price_data = df[price_col].values.astype(float)
                self.states_list.append(price_data.reshape(-1, 1))
                self.states_names.append(price_col)
                self.states_types.append("regression")
                self.num_classes.append(1)
                print(f"   └─ {price_col}")

    def _process_returns(self, df: pd.DataFrame) -> None:
        """
        处理收益率特征

        🔥 关键改进：在特征生成阶段就完成 winsorize+tanh 治理
        - 避免训练/实盘流程不一致
        - 确保所有使用该特征的地方都获得治理后的数据
        """
        print(f"\n✅ 处理收益率...")

        base_price_col = f"{self.base_period}_close"
        if base_price_col not in df.columns:
            return

        # 1. 计算对数收益率
        p = df[base_price_col].astype(float).values
        p_safe_prev = np.where(p[:-1] == 0.0, 1e-8, p[:-1])
        log_ret = np.zeros_like(p, dtype=float)
        if len(p) > 1:
            log_ret[1:] = np.log(p[1:] / p_safe_prev)

        # 2. 应用收益率治理（winsorize + tanh）
        ret_cfg = self.cfg.get("return_feature", {})
        enable_governance = bool(ret_cfg.get("enable_governance", True))

        if enable_governance:
            log_ret = self._apply_return_governance(log_ret, ret_cfg)

        # 3. 添加到结果
        self.states_list.append(log_ret.reshape(-1, 1))
        self.states_names.append(f"ret_{self.base_period}_log")
        self.states_types.append("regression")
        self.num_classes.append(1)

        gov_status = "已治理" if enable_governance else "原始值"
        print(f"   └─ ret_{self.base_period}_log ({gov_status})")

    def _apply_return_governance(self, ret: np.ndarray, cfg: Dict) -> np.ndarray:
        """
        对收益率特征应用治理（winsorize + tanh）

        治理流程：
        1. Winsorize: 裁剪极端值到 [p_lo, p_hi] 分位数
        2. Tanh: 压缩到 [-1, 1] 范围，避免极端值影响模型

        Args:
            ret: 原始对数收益率
            cfg: 治理配置

        Returns:
            治理后的收益率
        """
        ret_arr = np.asarray(ret, dtype=float)
        finite_mask = np.isfinite(ret_arr)

        if not finite_mask.any():
            return ret_arr

        # 读取配置
        p_lo = float(cfg.get("winsorize_p_lo", 0.1))  # 0.1%分位数
        p_hi = float(cfg.get("winsorize_p_hi", 99.9))  # 99.9%分位数
        tanh_scale_factor = float(cfg.get("tanh_scale_factor", 3.0))  # tanh缩放因子

        # 1️⃣ Winsorize: 裁剪极端值
        lo_bound = float(np.percentile(ret_arr[finite_mask], p_lo))
        hi_bound = float(np.percentile(ret_arr[finite_mask], p_hi))
        ret_clipped = np.clip(ret_arr, lo_bound, hi_bound)

        # 2️⃣ Tanh压缩: 计算缩放因子
        std_ref = float(np.std(ret_clipped[finite_mask]))
        eps = 1e-12
        tanh_scale = tanh_scale_factor * max(std_ref, eps)

        # 应用tanh压缩
        ret_governed = np.tanh(ret_clipped / tanh_scale)

        # 还原NaN位置
        ret_governed[~finite_mask] = 0.0

        # 打印治理统计
        try:
            print(f"      🧪 收益率治理统计:")
            print(f"         Winsorize: [{lo_bound:.6f}, {hi_bound:.6f}] (p=[{p_lo:.1f}%, {p_hi:.1f}%])")
            print(f"         Tanh scale: {tanh_scale:.6f} (factor={tanh_scale_factor})")
            print(f"         原始范围: [{np.min(ret_arr[finite_mask]):.6f}, {np.max(ret_arr[finite_mask]):.6f}]")
            print(
                f"         治理后范围: [{np.min(ret_governed[finite_mask]):.6f}, {np.max(ret_governed[finite_mask]):.6f}]"
            )
        except Exception:
            pass

        return ret_governed

    def _assemble_results(self) -> Dict[str, Any]:
        """组装最终结果"""
        print("\n" + "=" * 70)
        print("📦 数据打包")
        print("=" * 70)

        all_states = np.concatenate(self.states_list, axis=1)
        observations = all_states.copy()

        print(f"✅ States形状: {all_states.shape}")
        print(f"✅ Observations形状: {observations.shape}")
        print(f"   - 分类特征: {sum(1 for t in self.states_types if t == 'classification')}")
        print(f"   - 回归特征: {sum(1 for t in self.states_types if t == 'regression')}")

        return {
            "observations": observations.astype(np.float32),
            "observation_names": [str(n) for n in self.states_names],
            "states": all_states.astype(np.float32),
            "state_names": [str(n) for n in self.states_names],
            "state_types": [str(t) for t in self.states_types],
            "num_classes": np.array(self.num_classes, dtype=int),
            "timestamps": self.timestamps,
            "prices": self.prices,
            "periods": self.periods,
        }
