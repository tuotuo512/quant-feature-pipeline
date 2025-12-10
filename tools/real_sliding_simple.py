#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真滑窗计算（简化版）- 专用于Step4
用3m close重算大周期动量，实现平滑过渡

核心逻辑：
- 15m_mom: 每根3m K线，用最新3m close重算15m动量
- 30m_mom: 每根3m K线，用最新3m close重算30m动量
- 效果：动量不再阶跃跳变，而是平滑过渡
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict


def _tf_to_minutes(tf: str) -> int:
    """将时间周期转换为分钟数"""
    m = re.fullmatch(r"(\d+)([mhdw])", str(tf).strip().lower())
    if not m:
        return 1
    n = int(m.group(1))
    unit = m.group(2)
    factor = {"m": 1, "h": 60, "d": 1440, "w": 10080}.get(unit, 1)
    return n * factor


def apply_real_sliding_momentum(
    df: pd.DataFrame,
    mom_col: str,
    base_close_col: str,
    period_minutes: int,
    base_minutes: int,
    window: int = 20,
) -> pd.DataFrame:
    """
    真滑窗计算动量：用基础周期close重算大周期动量

    公式: mom = (当前close / window周期前close) - 1

    例如：15m_mom10 在 12:10时刻（3m基础）
    - 回溯: 10 * 15 = 150分钟
    - 步数: 150 / 3 = 50根3m K线
    - 计算: (12:10的3m_close / 50根前的3m_close) - 1

    Args:
        df: 包含基础周期close的DataFrame
        mom_col: 动量列名（如 "15m_mom"）
        base_close_col: 基础周期close列名（如 "3m_close"）
        period_minutes: 大周期的分钟数（如 15）
        base_minutes: 基础周期的分钟数（如 3）
        window: 动量窗口（默认20）

    Returns:
        更新后的DataFrame
    """
    if base_close_col not in df.columns:
        print(f"      ⚠️ 未找到基础价格列 {base_close_col}, 跳过 {mom_col}")
        return df

    # 计算真实回溯长度（分钟数 → 基础周期步数）
    lookback_minutes = window * period_minutes
    lookback_steps = lookback_minutes // base_minutes  # 🔥 关键修复：转换为步数

    # 向量化计算
    close_series = pd.to_numeric(df[base_close_col], errors="coerce").astype(float)
    ref_series = close_series.shift(lookback_steps)  # 🔥 用步数shift

    with np.errstate(divide="ignore", invalid="ignore"):
        mom = (close_series / ref_series) - 1.0

    # 前期数据不足的填充
    mom = mom.ffill().fillna(0.0)

    df[mom_col] = mom.values
    return df


def apply_real_sliding_window(
    df: pd.DataFrame, timeframes: List[str], base_tf: str, config: Dict
) -> pd.DataFrame:
    """
    为所有大周期动量列应用真滑窗计算

    Args:
        df: merged DataFrame
        timeframes: 所有周期列表（如 ["3m", "15m", "30m", "2h"]）
        base_tf: 基础周期（如 "3m"）
        config: 配置字典

    Returns:
        更新后的DataFrame
    """
    # 基础周期的close列名
    base_close_col = f"{base_tf}_close"
    if base_close_col not in df.columns:
        print(f"   ⚠️ 未找到基础价格列 {base_close_col}, 跳过真滑窗计算")
        return df

    base_minutes = _tf_to_minutes(base_tf)

    # 配置：哪些指标需要真滑窗
    enabled_indicators = config.get("real_sliding_indicators", ["mom"])
    default_window = int(config.get("real_sliding_window", 20))

    # 统计
    processed_count = 0

    # 遍历所有周期（排除基础周期）
    for tf in timeframes:
        tf_minutes = _tf_to_minutes(tf)
        if tf_minutes <= base_minutes:
            continue  # 跳过基础周期及更小周期

        # 遍历所有指标
        for indicator in enabled_indicators:
            # 构造列名候选
            col_candidates = [
                f"{tf}_{indicator}",
                f"{tf}_{indicator}_fixed",
            ]

            # 找到存在的列
            target_col = None
            for col in col_candidates:
                if col in df.columns:
                    target_col = col
                    break

            if not target_col:
                continue

            # 提取窗口大小（如 "15m_mom20" -> 20）
            match = re.search(r"mom(\d+)", target_col)
            if match:
                window = int(match.group(1))
            else:
                window = default_window

            # 应用真滑窗计算
            lookback_steps = (window * tf_minutes) // base_minutes
            print(
                f"   • {target_col}: 用 {base_tf}_close 重算（窗口={window}, 回溯={window*tf_minutes}分钟 = {lookback_steps}步）"
            )
            df = apply_real_sliding_momentum(df, target_col, base_close_col, tf_minutes, base_minutes, window)
            processed_count += 1

    if processed_count > 0:
        print(f"   ✅ 共处理 {processed_count} 个动量指标")
    else:
        print(f"   ℹ️ 未找到需要处理的动量指标")

    return df


def apply_real_sliding_bb_width(
    df: pd.DataFrame,
    bb_col: str,
    base_close_col: str,
    period_minutes: int,
    window: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    真滑窗计算布林带宽度：用基础周期close重算

    公式: bb_width = (bb_upper - bb_lower) / ma
         其中 bb_upper = ma + std_dev * std
              bb_lower = ma - std_dev * std

    Args:
        df: 包含基础周期close的DataFrame
        bb_col: 布林带宽度列名（如 "15m_bb_width"）
        base_close_col: 基础周期close列名（如 "3m_close"）
        period_minutes: 大周期的分钟数（如 15）
        window: 窗口大小（默认20）
        std_dev: 标准差倍数（默认2.0）

    Returns:
        更新后的DataFrame
    """
    if base_close_col not in df.columns:
        return df

    # 计算滚动窗口大小（分钟数）
    lookback_minutes = window * period_minutes

    # 向量化计算
    close_series = pd.to_numeric(df[base_close_col], errors="coerce").astype(float)

    # 滚动均值和标准差
    ma = close_series.rolling(window=lookback_minutes, min_periods=max(10, lookback_minutes // 2)).mean()
    std = close_series.rolling(window=lookback_minutes, min_periods=max(10, lookback_minutes // 2)).std()

    # 布林带上下轨
    bb_upper = ma + std_dev * std
    bb_lower = ma - std_dev * std

    # 布林带宽度
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = (bb_upper - bb_lower) / ma

    # 填充
    bb_width = bb_width.ffill().fillna(0.0)

    df[bb_col] = bb_width.values
    return df


# 向后兼容：保留旧函数名
apply_real_sliding_to_merged_data = apply_real_sliding_window
