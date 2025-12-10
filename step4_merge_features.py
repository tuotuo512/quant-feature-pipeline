#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4: 特征融合器（完全配置驱动版）

✨ 特性：
  - 完全由 main_config.yaml + step4_merge.yaml 驱动
  - 以基础周期为时间轴，对齐多周期指标
  - 使用 merge_asof backward 对齐策略
  - 支持时间范围过滤

📋 用法：
  python step4_merge_features.py
  python step4_merge_features.py --start 2024-01-01 --end 2024-12-31

🔧 配置：
  - 全局配置: main_config.yaml
  - 融合策略: step4_merge.yaml
"""

from __future__ import annotations

import os
import sys
import argparse
import re

try:
    import pandas as pd
except ImportError:
    print("❌ 导入 pandas 失败，请运行: pip install pandas")
    sys.exit(1)

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from features_engineering.congfigs.config_loader import ConfigLoader
except Exception as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

# 工具（读写/增量）
from features_engineering.tools.io_paths import (
    read_df_auto,
    IOManager,
    print_latest_timestamp_from_df,
)
from features_engineering.tools.incremental import safe_concat_dedup


def ensure_dir(dir_path: str) -> str:
    """确保目录存在并返回其绝对路径"""
    if not dir_path:
        return dir_path
    abs_dir = os.path.abspath(dir_path)
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Step4: 特征融合器（完全配置驱动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python step4_merge_features.py

  # 指定时间范围
  python step4_merge_features.py --start 2024-01-01 --end 2024-12-31

  # 覆盖输出格式
  python step4_merge_features.py --output_format parquet
        """,
    )
    parser.add_argument("--start", type=str, default=None, help="起始时间(可选)，如 2024-01-01")
    parser.add_argument("--end", type=str, default=None, help="结束时间(可选)，如 2024-12-31")
    parser.add_argument(
        "--output_format",
        type=str,
        default=None,
        choices=["csv", "parquet", "both"],
        help="覆盖输出格式（默认从 main_config.yaml 读取）",
    )
    return parser.parse_args()


def read_base_1m(base_file: str, start: str | None, end: str | None) -> pd.DataFrame:
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"1m 基础CSV不存在: {base_file}")

    # 尝试以第一列为时间索引读取
    try:
        df = pd.read_csv(base_file, parse_dates=[0], index_col=0)
        if df.index.name is None:
            df.index.name = "timestamp"
    except Exception:
        df = pd.read_csv(base_file)
        ts_col = None
        for cand in ["timestamp", "time", "datetime", "ts"]:
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col is None:
            raise ValueError("CSV 中未找到时间列（timestamp/time/datetime/ts）")
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

    # 只保留标准列
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 切片
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    if df.empty:
        raise ValueError("筛选后的1m数据为空，请检查时间范围或输入文件内容")
    return df


def _tf_to_minutes(tf: str) -> int:
    """将时间周期字符串（如 '1m','3m','2h','1d','1w'）转换为分钟数，用于比较大小。"""
    try:
        m = re.fullmatch(r"(\d+)([mhdw])", str(tf).strip().lower())
        if not m:
            return 1_000_000  # 未识别的放到极大，避免被选为最小
        n = int(m.group(1))
        unit = m.group(2)
        factor = {"m": 1, "h": 60, "d": 1440, "w": 10080}.get(unit, 1)
        return n * factor
    except Exception:
        return 1_000_000


def read_kline_for_tf(
    kline_root: str, symbol: str, tf: str, start: str | None, end: str | None
) -> pd.DataFrame:
    """读取已重采样的K线作为基准时间轴（来自 rl_live/kline）。优先Parquet，回退CSV。"""
    base = os.path.abspath(kline_root)
    p_parquet = os.path.join(base, f"{symbol}_{tf}.parquet")
    p_csv = os.path.join(base, f"{symbol}_{tf}.csv")

    df = None
    if os.path.exists(p_parquet):
        df = pd.read_parquet(p_parquet)
    elif os.path.exists(p_csv):
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"未找到K线文件: {p_parquet} 或 {p_csv}")

    # 解析时间索引
    if isinstance(df, pd.DataFrame):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
                df = df.set_index(df.columns[0])
            except Exception:
                raise ValueError("K线文件无法识别时间列")
        df.index.name = "timestamp"
        # 仅保留标准OHLCV列（存在则保留）
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if keep_cols:
            df = df[keep_cols]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # 切片
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        if df.empty:
            raise ValueError("筛选后的基准K线数据为空")
        return df
    else:
        raise ValueError("读取K线失败：数据格式异常")


def read_ind_for_tf(
    ind_root: str,
    symbol: str,
    tf: str,
    start: str | None,
    end: str | None,
    preferred_fmt: str | None = None,
    indicator_pattern: str | None = None,
) -> pd.DataFrame:
    # 目录采用根目录，不进入子目录（遵循 main_config.yaml 模板）
    base = os.path.abspath(ind_root)

    # 优先使用 main_config.yaml 中的模板：io.filename_patterns.indicator
    p_from_pattern_parquet = None
    p_from_pattern_csv = None
    if indicator_pattern:
        # 占位替换：{symbol.trading_pair_std}, $timeframe$
        fname_pat = indicator_pattern
        try:
            fname_pat = fname_pat.replace("{symbol.trading_pair_std}", str(symbol))
            fname_pat = fname_pat.replace("$timeframe$", str(tf))
        except Exception:
            pass
        p_from_pattern_parquet = os.path.join(base, fname_pat)
        # 若模板未指明扩展名，尝试两种
        if not os.path.splitext(fname_pat)[1]:
            p_from_pattern_parquet = os.path.join(base, fname_pat + ".parquet")
            p_from_pattern_csv = os.path.join(base, fname_pat + ".csv")
        else:
            # 反向推导另一种扩展以作回退
            ext = os.path.splitext(fname_pat)[1].lower()
            if ext == ".parquet":
                p_from_pattern_csv = os.path.join(base, os.path.splitext(fname_pat)[0] + ".csv")
            elif ext == ".csv":
                p_from_pattern_parquet = os.path.join(base, os.path.splitext(fname_pat)[0] + ".parquet")

    # 默认命名（向后兼容）
    fname_ind = f"{symbol}_{tf}_indicators"
    fname_old = f"{symbol}_{tf}_ind"
    p_parquet = os.path.join(base, f"{fname_ind}.parquet")
    p_csv = os.path.join(base, f"{fname_ind}.csv")

    # 如果新文件不存在，尝试旧文件名
    if not os.path.exists(p_parquet) and not os.path.exists(p_csv):
        # 兼容旧名
        candidates = [
            os.path.join(base, f"{fname_old}.parquet"),
            os.path.join(base, f"{fname_old}.csv"),
            os.path.join(base, f"{symbol}_{tf}_fixed.parquet"),
            os.path.join(base, f"{symbol}_{tf}_fixed.csv"),
            os.path.join(base, f"{symbol}_{tf}_roll.parquet"),
            os.path.join(base, f"{symbol}_{tf}_roll.csv"),
        ]
        # 选择第一个存在的
        p_parquet = next((c for c in candidates if c.endswith(".parquet") and os.path.exists(c)), p_parquet)
        p_csv = next((c for c in candidates if c.endswith(".csv") and os.path.exists(c)), p_csv)

    # 按首选格式读取；若不存在则回退（优先模板路径）
    fmt = (preferred_fmt or "csv").lower()
    fmt = "csv" if fmt not in ("csv", "parquet", "both") else fmt
    paths_try: list[tuple[str, str]]
    if fmt == "parquet":
        paths_try = [
            ("parquet", p_from_pattern_parquet or ""),
            ("csv", p_from_pattern_csv or ""),
            ("parquet", p_parquet),
            ("csv", p_csv),
        ]
    elif fmt == "csv":
        paths_try = [
            ("csv", p_from_pattern_csv or ""),
            ("parquet", p_from_pattern_parquet or ""),
            ("csv", p_csv),
            ("parquet", p_parquet),
        ]
    else:  # both
        paths_try = [
            ("csv", p_from_pattern_csv or ""),
            ("parquet", p_from_pattern_parquet or ""),
            ("csv", p_csv),
            ("parquet", p_parquet),
        ]

    df = None
    for kind, path in paths_try:
        if path and os.path.exists(path):
            if kind == "csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            break
    if df is None:
        raise FileNotFoundError(f"未找到指标文件: {p_parquet} 或 {p_csv}")

    # 解析时间索引
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index(df.columns[0])
        except Exception:
            raise ValueError("无法识别时间列，请检查指标文件")

    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    if df.empty:
        raise ValueError(f"指标{tf}为空，请检查时间范围或输入文件")

    # 🔧 排除OHLCV列，避免与基础K线重复（重要！）
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    cols_to_drop = [c for c in ohlcv_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = _standardize_indicator_columns(df)
    return df


def _standardize_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一指标列命名，将 rsi14 → rsi、atr14_pct → atr_pct 等。"""
    renamed = df.copy()
    rename_map: dict[str, str] = {}
    used_targets = set(renamed.columns)
    for col in list(renamed.columns):
        if not isinstance(col, str):
            continue
        name = col.strip()
        target = None
        if re.fullmatch(r"rsi\d+", name):
            target = "rsi"
        elif re.fullmatch(r"atr\d+_pct", name):
            target = "atr_pct"
        elif re.fullmatch(r"atr\d+", name):
            target = "atr"
        elif re.fullmatch(r"macd_hist(?:ogram)?", name):
            target = "macd_histogram"
        if target and target not in used_targets and target not in rename_map.values():
            rename_map[col] = target
            used_targets.add(target)
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def prefix_columns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """为DataFrame的所有列添加周期前缀"""
    renamed = df.copy()
    renamed.columns = [f"{tf}_" + str(c) for c in renamed.columns]
    return renamed


def asof_merge_on_1m(base_1m: pd.DataFrame, tf_to_df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = base_1m.reset_index().sort_values("timestamp").copy()
    for tf, df in tf_to_df.items():
        df_pref = prefix_columns(df, tf).reset_index().sort_values("timestamp")
        merged = pd.merge_asof(
            merged,
            df_pref,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
    merged = merged.set_index("timestamp").sort_index()
    return merged


def save_output(df: pd.DataFrame, output_file: str, fmt: str):
    ensure_dir(os.path.dirname(os.path.abspath(output_file)))
    # 基础QC：去重+排序+NaN轻度处理（更深入的QC在Step5）
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()
    # 写CSV优先，便于检查
    if fmt in ("csv", "both"):
        if output_file.endswith(".csv"):
            c = output_file
        else:
            c = os.path.splitext(output_file)[0] + ".csv"
        df.reset_index().to_csv(c, index=False)
        print(f"✅ 写入CSV: {c}")
    if fmt in ("parquet", "both"):
        if output_file.endswith(".parquet"):
            p = output_file
        else:
            p = os.path.splitext(output_file)[0] + ".parquet"
        try:
            df.to_parquet(p, index=True)
            print(f"✅ 写入Parquet: {p}")
        except Exception as e:
            print(f"⚠️ Parquet写入失败(已忽略): {e}")


def execute_step4(
    cfg: dict,
    start: str | None = None,
    end: str | None = None,
    output_format: str | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """执行 Step4 特征融合逻辑，供脚本与统一流水线复用。"""
    if not cfg:
        raise ValueError("配置不能为空")

    log = print if verbose else (lambda *args, **kwargs: None)

    log("🚀 Step4 特征融合启动（完全配置驱动）\n")

    symbol = cfg.get("symbol", {}).get("trading_pair_std", "ETH_USDT")
    market_type = cfg.get("symbol", {}).get("market_type", "swap")

    timeframes_cfg = cfg.get("timeframes", {})
    base_download = timeframes_cfg.get("base_download", "1m")
    timeframes = timeframes_cfg.get("resample_targets", ["3m", "15m", "30m", "2h"])
    variant = str(timeframes_cfg.get("variant", "")).strip().lower()
    source_mode = cfg.get("rl_build", {}).get("source_mode", "fixed")

    io_cfg = cfg.get("io", {})
    io = IOManager(cfg)
    base_dir = io.base_dir
    downloads_dir = io.downloads_dir
    kline_dir = io.kline_dir
    ind_dir = io.indicators_dir
    merged_dir = io.merged_dir

    output_fmt = output_format or io_cfg.get("output_format", "csv")
    io_overwrite = bool(io_cfg.get("overwrite", False))

    merge_cfg = cfg.get("merge", {})
    include_base_ohlcv = merge_cfg.get("include_base_ohlcv", True)
    align_direction = merge_cfg.get("align_direction", "backward")
    allow_exact_match = merge_cfg.get("allow_exact_match", True)
    add_prefix = merge_cfg.get("add_timeframe_prefix", True)

    try:
        base_axis_tf = sorted(timeframes, key=_tf_to_minutes)[0] if timeframes else base_download
    except Exception:
        base_axis_tf = base_download

    if str(base_axis_tf).lower() == str(base_download).lower():
        base_kline_path = io.path_for("download", timeframe=base_download)
    else:
        base_kline_path = io.path_for("kline", timeframe=base_axis_tf)
    base_filename = os.path.basename(base_kline_path)

    output_filename = f"{symbol}_{base_axis_tf}_merged.{output_fmt}"
    output_file = os.path.join(merged_dir, output_filename)

    log("📋 配置摘要:")
    log(f"   交易对: {symbol} ({market_type.upper()})")
    log(f"   基础周期(base_download): {base_download}")
    log(f"   主轴周期(base_axis): {base_axis_tf}")
    log(f"   融合周期: {', '.join(timeframes)}")
    log(f"   数据模式: {source_mode.upper()}")
    log(f"   输出格式: {output_fmt.upper()}")
    if variant in ("fixed", "roll", "both"):
        log(f"   输出变体(merge源): {variant.upper()}")
    log(f"\n📂 路径配置:")
    log(f"   基础K线: {base_filename}")
    log(f"   指标目录: {ind_dir}")
    log(f"   输出文件: {output_filename}")
    log(f"\n🔧 融合选项:")
    log(f"   对齐方式: {align_direction}")
    log(f"   包含OHLCV: {include_base_ohlcv}")
    log(f"   周期前缀: {add_prefix}")
    if start or end:
        log(f"\n⏰ 时间范围: {start or '-∞'} ~ {end or '+∞'}")

    if include_base_ohlcv:
        log(f"\n✓ 读取基础K线...")
        try:
            if str(base_axis_tf).lower() == str(base_download).lower():
                df0 = io.read_table("download", timeframe=base_download)
            else:
                df0 = io.read_table("kline", timeframe=base_axis_tf)
            if "timestamp" in df0.columns:
                df0["timestamp"] = pd.to_datetime(df0["timestamp"], errors="coerce")
                df0 = df0.set_index("timestamp")
            df0.index.name = "timestamp"
            keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df0.columns]
            base_df = df0[keep_cols].sort_index()
        except Exception as e:
            raise RuntimeError(f"未找到主轴K线({base_axis_tf})，请先运行Step2生成。详情: {e}") from e
        log(f"  基础数据: {len(base_df):,} 行")
        log(f"  时间范围: {base_df.index.min()} ~ {base_df.index.max()}")
        log(f"  OHLCV列: {list(base_df.columns)}")
    else:
        base_df = None
        log(f"\n⏭️ 跳过基础K线（仅融合指标）")

    log(f"\n✓ 读取多周期指标...")
    tf_to_df: dict[str, pd.DataFrame] = {}
    for i, tf in enumerate(timeframes, 1):
        log(f"  [{i}/{len(timeframes)}] 周期 {tf}...", end=" ")
        try:
            df0 = io.read_table("indicator", timeframe=tf)
            if "timestamp" in df0.columns:
                df0["timestamp"] = pd.to_datetime(df0["timestamp"], errors="coerce")
                df0 = df0.set_index("timestamp")
            df0.index.name = "timestamp"
            df0 = df0[~df0.index.duplicated(keep="last")].sort_index()
            if start:
                df0 = df0[df0.index >= pd.to_datetime(start)]
            if end:
                df0 = df0[df0.index <= pd.to_datetime(end)]
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            df0 = df0.drop(columns=[c for c in ohlcv_cols if c in df0.columns], errors="ignore")
            df0 = _standardize_indicator_columns(df0)
            tf_to_df[tf] = df0
            log(f"✓ {len(df0):,} 行, {df0.shape[1]} 列特征")
        except FileNotFoundError as e:
            log(f"❌ 文件不存在，跳过")
            if verbose:
                log(f"      {e}")
        except Exception as e:
            log(f"❌ 读取失败: {e}")

    if not tf_to_df:
        raise RuntimeError("没有成功读取任何周期的指标数据")

    log(f"\n✓ 开始融合...")
    if base_df is None:
        raise RuntimeError("当前配置未包含基础K线，离线融合建议启用 include_base_ohlcv=True")

    merged = base_df.reset_index().sort_values("timestamp").copy()
    log(f"  主轴: {len(merged):,} 行")

    for tf, df in tf_to_df.items():
        if add_prefix:
            df_prefixed = prefix_columns(df, tf)
        else:
            df_prefixed = df.copy()
        df_prefixed = df_prefixed.reset_index().sort_values("timestamp")
        merged = pd.merge_asof(
            merged,
            df_prefixed,
            on="timestamp",
            direction=align_direction,
            allow_exact_matches=allow_exact_match,
        )
        log(f"  + {tf}: {df_prefixed.shape[1]-1} 列特征 → 累计 {merged.shape[1]-1} 列")

    merged = merged.set_index("timestamp").sort_index()

    if add_prefix and include_base_ohlcv:
        base_tf = str(base_axis_tf)
        log(f"\n✓ 为基础K线添加周期前缀: {base_tf}_*")
        base_prefix_map = {
            "open": f"{base_tf}_open",
            "high": f"{base_tf}_high",
            "low": f"{base_tf}_low",
            "close": f"{base_tf}_close",
            "volume": f"{base_tf}_volume",
        }
        cols_to_rename = {c: base_prefix_map[c] for c in base_prefix_map.keys() if c in merged.columns}
        if cols_to_rename:
            merged = merged.rename(columns=cols_to_rename)

    if merge_cfg.get("enable_real_sliding", True):
        log(f"\n✓ 应用真滑窗计算...")
        try:
            from features_engineering.tools.real_sliding_simple import apply_real_sliding_window

            merged = apply_real_sliding_window(merged, timeframes, base_axis_tf, merge_cfg)
        except Exception as e:
            log(f"   ⚠️ 真滑窗计算失败（已忽略）: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    log(f"\n{'='*70}")
    log("融合完成统计")
    log(f"{'='*70}")
    log(f"最终数据: {len(merged):,} 行 × {merged.shape[1]} 列")
    log(f"时间范围: {merged.index.min()} ~ {merged.index.max()}")
    log(f"特征列数: {merged.shape[1]}")

    if add_prefix:
        log(f"\n列分布:")
        for tf in timeframes:
            tf_cols = [c for c in merged.columns if c.startswith(f"{tf}_")]
            if tf_cols:
                log(f"  {tf}: {len(tf_cols)} 列")
        if include_base_ohlcv:
            base_tf = str(base_axis_tf)
            base_cols = [c for c in merged.columns if c.startswith(f"{base_tf}_")]
            if base_cols:
                log(f"  {base_tf} (基础): {len(base_cols)} 列")

    log(f"\n✓ 保存融合结果...")
    if not io_overwrite:
        try:
            base, ext = os.path.splitext(output_file)
            for cand in (base + ".parquet", base + ".csv"):
                if os.path.exists(cand):
                    old = read_df_auto(cand)
                    if "timestamp" in old.columns:
                        old["timestamp"] = pd.to_datetime(old["timestamp"], errors="coerce")
                        old = old.set_index("timestamp")
                    old.index.name = "timestamp"
                    merged = safe_concat_dedup(old, merged)
                    log(f"   🔁 合并历史 merged: {len(merged):,} 行")
                    break
        except Exception as _e:
            log(f"   ⚠️ 合并历史失败(merged): {_e}")

    save_output(merged, output_file, output_fmt)

    log(f"\n{'='*70}")
    log("🎉 特征融合完成！")
    log(f"{'='*70}\n")

    if verbose:
        print_latest_timestamp_from_df(merged)

    return {
        "symbol": symbol,
        "market_type": market_type,
        "base_download": base_download,
        "base_axis": base_axis_tf,
        "timeframes": timeframes,
        "kline_dir": kline_dir,
        "indicators_dir": ind_dir,
        "merged_dir": merged_dir,
        "output_file": output_file,
        "output_format": output_fmt,
        "include_base_ohlcv": include_base_ohlcv,
        "align_direction": align_direction,
        "add_prefix": add_prefix,
        "start": start,
        "end": end,
    }


def main():
    """主流程：完全配置驱动的特征融合"""
    args = parse_args()

    try:
        loader = ConfigLoader()
        cfg = loader.load_step4_config()
        if not cfg:
            print("❌ 未找到或无法加载配置文件")
            return 1

    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        import traceback

        traceback.print_exc()
        return 1
    try:
        execute_step4(
            cfg,
            start=args.start,
            end=args.end,
            output_format=args.output_format,
            verbose=True,
        )
        return 0
    except Exception as e:
        print(f"❌ Step4 执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
