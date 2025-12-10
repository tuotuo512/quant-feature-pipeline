#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI实时验证脚本
用于验证我们计算的RSI是否与TradingView一致
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from common.indicators.calculator import IndicatorCalculator


def verify_rsi_from_csv(csv_path: str, timeframe: str = "30m", window: int = 14):
    """
    从CSV文件验证RSI计算

    Args:
        csv_path: K线CSV文件路径（需包含OHLCV列）
        timeframe: 时间周期（用于显示）
        window: RSI窗口（默认14）
    """
    print("=" * 80)
    print(f"🔍 RSI验证：{timeframe} K线")
    print("=" * 80)

    # 读取数据
    df = pd.read_csv(csv_path)

    # 解析时间列
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df = df.set_index("timestamp")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")

    df = df.sort_index()

    print(f"\n📊 数据信息:")
    print(f"   K线数量: {len(df)}")
    print(f"   时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"   价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    # 计算RSI
    calc = IndicatorCalculator(verbose=False)
    result = calc._calculate_rsi(df, window=window)

    # 提取最后20个值
    tail = result[["rsi14", "rsi_event", "rsi_overbought", "rsi_oversold"]].tail(20)
    tail["close"] = df["close"]
    tail = tail[["close", "rsi14", "rsi_event", "rsi_overbought", "rsi_oversold"]]

    print(f"\n📊 最后20根K线RSI:")
    print(tail.to_string())

    # 最新值
    last_rsi = result["rsi14"].iloc[-1]
    last_event = result["rsi_event"].iloc[-1]

    print(f"\n" + "=" * 80)
    print(f"📊 最新RSI（请对比TradingView）")
    print("=" * 80)
    print(f"\n   时间: {df.index[-1]}")
    print(f"   价格: {df['close'].iloc[-1]:.2f}")
    print(f"   RSI({window}): {last_rsi:.2f}")

    if last_event == 1:
        print(f"   状态: 🔴 超买（RSI >= 70）")
    elif last_event == -1:
        print(f"   状态: 🟢 超卖（RSI <= 30）")
    else:
        print(f"   状态: ⚪ 中性（30 < RSI < 70）")

    print(f"\n🎯 验证步骤:")
    print(f"   1. 打开 TradingView.com")
    print(f"   2. 选择相同交易对和{timeframe}周期")
    print(f"   3. 添加 RSI({window}) 指标")
    print(f"   4. 对比最新RSI值")
    print(f"   5. 如果差异<1点，说明计算正确 ✅")

    return result


def verify_rsi_from_indicator_file(ind_path: str, timeframe: str = "30m"):
    """
    从Step3生成的指标文件验证RSI

    Args:
        ind_path: 指标文件路径（Step3输出）
        timeframe: 时间周期
    """
    print("=" * 80)
    print(f"🔍 验证Step3生成的RSI指标：{timeframe}")
    print("=" * 80)

    # 读取指标文件
    if ind_path.endswith(".parquet"):
        df = pd.read_parquet(ind_path)
    else:
        df = pd.read_csv(ind_path)

    # 解析时间索引
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")

    df = df.sort_index()

    print(f"\n📊 指标文件信息:")
    print(f"   数据行数: {len(df)}")
    print(f"   时间范围: {df.index[0]} ~ {df.index[-1]}")

    # 检查RSI列
    rsi_cols = [col for col in df.columns if "rsi" in col.lower()]
    print(f"\n📊 RSI相关列:")
    for col in rsi_cols:
        print(f"   - {col}")

    # 检查是否有RSI事件列
    has_event = "rsi_event" in df.columns
    has_ob = "rsi_overbought" in df.columns
    has_os = "rsi_oversold" in df.columns

    if has_event:
        print(f"\n✅ 包含 rsi_event 列（新格式）")
    if has_ob and has_os:
        print(f"✅ 包含 rsi_overbought/rsi_oversold 列（旧格式）")

    if not (has_event or (has_ob and has_os)):
        print(f"\n⚠️ 未找到RSI事件列，可能是旧数据")
        print(f"   建议重新运行 Step3")
        return None

    # 显示最后20个值
    cols_to_show = ["rsi14"]
    if has_event:
        cols_to_show.append("rsi_event")
    if has_ob:
        cols_to_show.append("rsi_overbought")
    if has_os:
        cols_to_show.append("rsi_oversold")

    tail = df[cols_to_show].tail(20)
    print(f"\n📊 最后20个RSI值:")
    print(tail.to_string())

    # 最新值
    last_rsi = df["rsi14"].iloc[-1]
    print(f"\n" + "=" * 80)
    print(f"📊 最新RSI值")
    print("=" * 80)
    print(f"\n   时间: {df.index[-1]}")
    print(f"   RSI(14): {last_rsi:.2f}")

    if has_event:
        last_event = df["rsi_event"].iloc[-1]
        if last_event == 1:
            print(f"   状态: 🔴 超买")
        elif last_event == -1:
            print(f"   状态: 🟢 超卖")
        else:
            print(f"   状态: ⚪ 中性")

    print(f"\n🎯 请在TradingView验证此值")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RSI实时验证工具")
    parser.add_argument("--csv", type=str, help="K线CSV文件路径")
    parser.add_argument("--ind", type=str, help="指标文件路径（Step3输出）")
    parser.add_argument("--tf", type=str, default="30m", help="时间周期（用于显示）")
    parser.add_argument("--window", type=int, default=14, help="RSI窗口")

    args = parser.parse_args()

    if args.csv:
        verify_rsi_from_csv(args.csv, args.tf, args.window)
    elif args.ind:
        verify_rsi_from_indicator_file(args.ind, args.tf)
    else:
        print("请指定 --csv 或 --ind 参数")
        print("\n示例:")
        print("  # 从K线CSV验证")
        print("  python verify_rsi_realtime.py --csv data/kline/ETH_USDT_30m.csv --tf 30m")
        print("\n  # 从指标文件验证")
        print(
            "  python verify_rsi_realtime.py --ind data/rl_live/ind/ETH_USDT_30m_indicators.parquet --tf 30m"
        )
