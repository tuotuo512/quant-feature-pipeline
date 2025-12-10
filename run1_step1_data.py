#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run1 装配器：仅负责调度 Step1（默认下载最近200日并补齐基础CSV）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from tools.filling import fill_base_ohlcv_grid
from tools.time_index import timeframe_to_minutes

MAX_GAP_FACTOR = 5
MAX_MISSING_RATIO = 0.02


def _calc_continuity_metrics(ts_data, freq_minutes: int) -> Tuple[float, float, int]:
    if pd is None or freq_minutes <= 0:
        return 0.0, 0.0, 0
    if isinstance(ts_data, (pd.DatetimeIndex, pd.Index)):
        ts_series = ts_data.to_series()
    else:
        ts_series = pd.Series(ts_data)
    ts_series = ts_series.dropna().sort_values()
    if ts_series.empty:
        return 0.0, 0.0, 0
    diffs = ts_series.diff().dropna().dt.total_seconds() / 60.0
    max_gap = float(diffs.max()) if not diffs.empty else 0.0
    total_minutes = max((ts_series.iloc[-1] - ts_series.iloc[0]).total_seconds() / 60.0, 0.0)
    expected_rows = int(total_minutes / freq_minutes) + 1 if freq_minutes > 0 else len(ts_series)
    expected_rows = max(expected_rows, len(ts_series))
    missing_ratio = 0.0 if expected_rows <= 0 else max(0.0, 1.0 - (len(ts_series) / expected_rows))
    return max_gap, missing_ratio, expected_rows


def ensure_base_csv_continuity(
    csv_path: str | None,
    base_tf: str,
    gap_factor: int = MAX_GAP_FACTOR,
    missing_ratio_limit: float = MAX_MISSING_RATIO,
) -> None:
    """
    简易连续性检查：若基础CSV存在大缺口或明显缺失，则自动补齐时间网格。
    """
    if not csv_path:
        return
    if pd is None:
        print("⚠️ pandas 未安装，跳过连续性检查")
        return
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        print(f"⚠️ 未找到基础CSV，跳过连续性检查: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ 无法读取基础CSV({csv_path})，跳过连续性检查: {exc}")
        return

    if df.empty:
        print("⚠️ 基础CSV为空，无法执行连续性检查")
        return

    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    if df.empty:
        print("⚠️ 基础CSV时间列无有效数据，跳过连续性检查")
        return

    freq_minutes = max(timeframe_to_minutes(base_tf), 1)
    max_gap, missing_ratio, expected_rows = _calc_continuity_metrics(df[ts_col], freq_minutes)
    allowed_gap = freq_minutes * gap_factor

    if max_gap <= allowed_gap and missing_ratio <= missing_ratio_limit:
        print(
            f"✅ 基础CSV连续性正常: 最大缺口 {max_gap:.2f} 分钟 | 缺失率 {missing_ratio*100:.2f}%"
        )
        return

    print(
        f"⚠️ 基础CSV连续性不佳: max_gap={max_gap:.2f} 分钟(允许≤{allowed_gap:.2f}) | "
        f"缺失率≈{missing_ratio*100:.2f}% (预期行数≈{expected_rows})，尝试补齐..."
    )

    idx_df = df.set_index(ts_col)
    filled_df = fill_base_ohlcv_grid(idx_df, base_tf)
    if filled_df is None or filled_df.empty:
        print("⚠️ 补齐失败（结果为空），请手动检查原始CSV")
        return

    filled_df.to_csv(csv_path, index=True)
    new_gap, new_missing_ratio, _ = _calc_continuity_metrics(filled_df.index.to_series(), freq_minutes)
    print(
        f"✅ 已补齐 {os.path.basename(csv_path)}: 新最大缺口 {new_gap:.2f} 分钟 | "
        f"缺失率 {new_missing_ratio*100:.4f}%"
    )


def main():
    from step1_data import run_step1_default

    expected_csv = None
    base_tf = "1m"

    # 统一 IO 摘要（不改变 Step1 行为）
    try:
        from features_engineering.congfigs.config_loader import ConfigLoader
        from tools.io_paths import IOManager

        loader = ConfigLoader()
        main_cfg = loader.load_main_config()
        io_mgr = IOManager(main_cfg)
        base_tf = main_cfg.get("timeframes", {}).get("base_download", base_tf)
        expected_csv = io_mgr.path_for("download", timeframe=base_tf)
        print(f"📂 Step1 目标输出(预计CSV): {expected_csv}")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run1 调度 Step1 数据下载")
    parser.add_argument("--days", type=int, default=280, help="最近天数，默认280")
    args = parser.parse_args()

    run_step1_default(days=args.days)
    ensure_base_csv_continuity(expected_csv, base_tf)


if __name__ == "__main__":
    main()
