#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step2: K线重采样器（完全配置驱动版）

✨ 特性：
  - 完全由 main_config.yaml + step2_resample.yaml 驱动
  - 自动处理 fixed/sliding 模式
  - 支持时间范围过滤
  - 支持多种输出格式

📋 用法：
  python step2_resample.py
  python step2_resample.py --start 2024-01-01 --end 2024-12-31

🔧 配置：
  - 全局配置: main_config.yaml
  - 重采样策略: step2_resample.yaml
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

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

# 工具集（增量/读写/时间）
from features_engineering.tools.io_paths import (
    read_df_auto,
    get_last_timestamp,
    print_latest_timestamp_from_df,
)
from features_engineering.tools.incremental import (
    safe_concat_dedup,
)
from features_engineering.tools.time_index import (
    timeframe_to_minutes,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Step2: K线重采样器（完全配置驱动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python step2_resample.py

  # 指定时间范围
  python step2_resample.py --start 2024-01-01 --end 2024-12-31

  # 覆盖输出格式
  python step2_resample.py --output_format parquet
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


def detect_base_interval(df: pd.DataFrame) -> str:
    """检测基础数据的时间间隔"""
    if len(df) < 2:
        return "1m"  # 默认假设1分钟

    # 计算前几个时间间隔
    time_diffs = df.index[1:6] - df.index[0:5]
    avg_diff = time_diffs.mean()

    if abs(avg_diff.total_seconds() - 60) < 30:  # 1分钟 ±30秒
        return "1m"
    elif abs(avg_diff.total_seconds() - 300) < 60:  # 5分钟 ±1分钟
        return "5m"
    elif abs(avg_diff.total_seconds() - 900) < 120:  # 15分钟 ±2分钟
        return "15m"
    else:
        # 默认返回1分钟，但给出警告
        print(f"⚠️ 无法准确检测基础间隔，平均间隔: {avg_diff}，假设为1m")
        return "1m"


def read_base_csv(
    input_file: str, start: str | None = None, end: str | None = None
) -> tuple[pd.DataFrame, str]:
    """读取基础CSV，解析时间索引，自动检测间隔，并按需切片。"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 尝试多种读取方式，兼容 index=timestamp 或列含 timestamp
    try:
        df = pd.read_csv(input_file, parse_dates=[0], index_col=0)
        if df.index.name is None:
            df.index.name = "timestamp"
    except Exception:
        df = pd.read_csv(input_file)
        ts_col = None
        for cand in ["timestamp", "time", "datetime", "ts"]:
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col is None:
            raise ValueError("CSV 中未找到时间列（timestamp/time/datetime/ts）")
        # 智能解析：整数用毫秒，字符串自动推断
        if pd.api.types.is_integer_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

    # 只保留标准列
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]

    # 排序/去重
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 切片（start/end通常是字符串，不需要指定unit）
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("筛选后的数据为空，请检查时间范围或输入文件内容")

    # 检测基础间隔
    base_interval = detect_base_interval(df)

    return df, base_interval


def timeframe_to_rule(tf: str) -> str:
    """
    周期字符串转 pandas resample 规则（智能解析，支持任意周期）

    支持格式:
    - 分钟: 1m, 3m, 5m, 10m, 15m, 30m, 45m 等
    - 小时: 1h, 2h, 4h, 6h, 8h, 12h 等
    - 天: 1d, 2d, 3d 等
    - 周: 1w, 2w 等

    示例:
        "5m" -> "5min"
        "2h" -> "2h"
        "1d" -> "1d"
    """
    tf = tf.strip().lower()

    # 解析数字和单位
    if not tf:
        raise ValueError("周期字符串不能为空")

    # 提取数字部分
    num_str = ""
    unit = ""
    for char in tf:
        if char.isdigit():
            num_str += char
        else:
            unit += char

    if not num_str or not unit:
        raise ValueError(f"无效的周期格式: {tf}，应为数字+单位，如 '5m', '2h', '1d'")

    try:
        num = int(num_str)
    except ValueError:
        raise ValueError(f"无效的周期数字: {num_str}")

    if num <= 0:
        raise ValueError(f"周期数字必须大于0: {num}")

    # 映射单位到 pandas 规则
    unit_mapping = {
        "m": "min",  # 分钟
        "min": "min",
        "h": "h",  # 小时
        "hour": "h",
        "d": "d",  # 天
        "day": "d",
        "w": "w",  # 周
        "week": "w",
    }

    if unit not in unit_mapping:
        raise ValueError(f"不支持的时间单位: {unit}，支持的单位: {', '.join(unit_mapping.keys())}")

    pandas_unit = unit_mapping[unit]
    return f"{num}{pandas_unit}"


def resample_ohlcv(df_base: pd.DataFrame, tf: str, base_interval: str = "1m") -> pd.DataFrame:
    """基础周期 → 指定周期的标准OHLCV重采样。支持1m/5m等基础数据。"""
    # 如果目标周期与基础周期相同，直接返回副本
    if tf == base_interval:
        print(f"  目标周期与基础周期相同 ({tf})，直接复制数据")
        df = df_base.copy()
        df.index.name = "timestamp"
        return df

    rule = timeframe_to_rule(tf)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # TODO: 🔥 未来数据泄露风险 - 需要重新生成训练数据并重新训练模型
    # closed='right': 时间戳T的bar包含 (T-period, T]，包含T时刻的数据（未来数据）
    # 正确配置应该是 closed='left': [T-period, T)，但需要重新训练模型
    df = df_base.resample(rule, label="right", closed="right").agg(agg)
    # 丢弃不完整bar
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df.index.name = "timestamp"
    return df


def _tf_to_minutes(tf: str) -> int:
    """
    将周期字符串转换为分钟数（智能解析）

    示例:
        "5m" -> 5
        "2h" -> 120
        "1d" -> 1440
        "1w" -> 10080
    """
    tf = tf.strip().lower()

    # 提取数字和单位
    num_str = ""
    unit = ""
    for char in tf:
        if char.isdigit():
            num_str += char
        else:
            unit += char

    if not num_str or not unit:
        raise ValueError(f"无效的周期格式: {tf}")

    try:
        num = int(num_str)
    except ValueError:
        raise ValueError(f"无效的周期数字: {num_str}")

    # 单位转换为分钟数
    unit_to_minutes = {
        "m": 1,  # 分钟
        "min": 1,
        "h": 60,  # 小时
        "hour": 60,
        "d": 1440,  # 天 (24 * 60)
        "day": 1440,
        "w": 10080,  # 周 (7 * 24 * 60)
        "week": 10080,
    }

    if unit not in unit_to_minutes:
        raise ValueError(f"不支持的时间单位: {unit}")

    return num * unit_to_minutes[unit]


def _interval_to_minutes(interval: str) -> int:
    return _tf_to_minutes(interval)


def rolling_preview_ohlcv(df_base: pd.DataFrame, tf: str, base_interval: str = "1m") -> pd.DataFrame:
    """
    生成以基础步长滚动的预览K线（用于未收盘rolling）。
    例如：base=5m, tf=15m → 窗口=3，每根5m更新一次：
    open=窗口首open, high=max, low=min, close=窗口末close, volume=sum。
    """
    tf_min = _tf_to_minutes(tf)
    base_min = _interval_to_minutes(base_interval)
    window = max(1, tf_min // base_min)
    if window <= 1:
        # 与基础一致，直接复制
        out = df_base.copy()
        out.index.name = "timestamp"
        return out

    # 使用滚动窗口聚合
    o = df_base["open"].rolling(window, min_periods=window).apply(lambda x: x[0], raw=True)
    h = df_base["high"].rolling(window, min_periods=window).max()
    l = df_base["low"].rolling(window, min_periods=window).min()
    c = df_base["close"].rolling(window, min_periods=window).apply(lambda x: x[-1], raw=True)
    v = df_base["volume"].rolling(window, min_periods=window).sum()
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=df_base.index)
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df.index.name = "timestamp"
    return df


def save_output(df: pd.DataFrame, kline_dir: str, symbol: str, tf: str, fmt: str, rolling: bool = False):
    """保存K线数据（根目录，不创建按周期子目录；文件名 {symbol}_{tf}.parquet/csv）"""
    output_dir = Path(kline_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{symbol}_{tf}"

    if fmt in ("csv", "both"):
        csv_path = output_dir / f"{fname}.csv"
        df.reset_index().to_csv(csv_path, index=False)
        print(f"   ✓ CSV: {csv_path.name}")

    if fmt in ("parquet", "both"):
        parquet_path = output_dir / f"{fname}.parquet"
        try:
            df.to_parquet(parquet_path, index=True)
            print(f"   ✓ Parquet: {parquet_path.name}")
        except Exception as e:
            print(f"   ⚠️ Parquet 写入失败: {e}")


def _find_existing_kline_path(kline_dir: str, symbol: str, tf: str, rolling: bool) -> Path | None:
    """查找已存在的K线文件（优先根目录；兼容旧路径 kline/<tf>/... 与 _roll）。"""
    base_root = Path(kline_dir)
    name_new = f"{symbol}_{tf}"
    # 1) 根目录
    for ext in (".parquet", ".csv"):
        p = base_root / f"{name_new}{ext}"
        if p.exists():
            return p
    # 2) 兼容旧路径：子目录 <tf>
    base_old = Path(kline_dir) / tf
    for ext in (".parquet", ".csv"):
        p = base_old / f"{name_new}{ext}"
        if p.exists():
            return p
    # 3) 兼容旧名 _roll
    name_old = f"{symbol}_{tf}_roll"
    for ext in (".parquet", ".csv"):
        p = base_old / f"{name_old}{ext}"
        if p.exists():
            return p
    return None


def _compute_incremental_start_for_step2(
    kline_dir: str, symbol: str, targets: list[str], include_rolling: bool
) -> tuple[pd.Timestamp | None, bool]:
    """返回(最早回溯起点, 是否检测到可增量)。若不存在历史文件则返回(None, False)。"""
    last_starts: list[pd.Timestamp] = []
    detected = False
    for tf in targets:
        p = _find_existing_kline_path(kline_dir, symbol, tf, rolling=False)
        if p is None:
            continue
        detected = True
        last_ts = get_last_timestamp(str(p))
        if last_ts is None:
            continue
        warm_minutes = timeframe_to_minutes(tf)
        warm_start = pd.to_datetime(last_ts) - pd.Timedelta(minutes=warm_minutes)
        last_starts.append(warm_start)
        # rolling 也一并考虑（若存在）
        if include_rolling:
            pr = _find_existing_kline_path(kline_dir, symbol, tf, rolling=True)
            if pr is not None:
                last_ts_r = get_last_timestamp(str(pr))
                if last_ts_r is not None:
                    warm_start_r = pd.to_datetime(last_ts_r) - pd.Timedelta(minutes=warm_minutes)
                    last_starts.append(warm_start_r)
    if not last_starts:
        return (None, detected)
    return (min(last_starts), True)


def execute_step2(
    cfg: dict,
    start: str | None = None,
    end: str | None = None,
    output_format: str | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """执行 Step2 重采样逻辑，供脚本与统一流水线复用。"""
    if not cfg:
        raise ValueError("配置不能为空")

    log = print if verbose else (lambda *args, **kwargs: None)

    log("🚀 Step2 K线重采样启动（完全配置驱动）\n")

    # ========== 1. 提取配置参数（零硬编码）==========
    symbol = cfg.get("symbol", {}).get("trading_pair_std", "ETH_USDT")
    market_type = cfg.get("symbol", {}).get("market_type", "swap")

    timeframes_cfg = cfg.get("timeframes", {})
    base_download = timeframes_cfg.get("base_download", "1m")
    timeframes = timeframes_cfg.get("resample_targets", ["3m", "15m", "30m", "2h"])
    include_rolling = timeframes_cfg.get("include_rolling", False)

    variant_val = timeframes_cfg.get("variant", None)
    if variant_val is None:
        variant_val = cfg.get("timeframes.variant", "")
    variant = str(variant_val or "").strip().lower()
    if variant not in ("fixed", "roll"):
        variant = "fixed"
    source_mode = cfg.get("rl_build", {}).get("source_mode", "fixed")

    io_cfg = cfg.get("io", {})
    base_dir = io_cfg.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
    downloads_dir = io_cfg.get("downloads_dir") or f"{base_dir}/rl_live/data_downloads"
    kline_dir = io_cfg.get("kline_dir") or f"{base_dir}/rl_live/kline"

    output_fmt = output_format or io_cfg.get("output_format", "csv")
    io_overwrite = bool(io_cfg.get("overwrite", False))

    incr_start, detected_any = _compute_incremental_start_for_step2(
        kline_dir, symbol, timeframes, include_rolling
    )
    incremental_mode = (not io_overwrite) and (incr_start is not None)
    effective_start = start
    if effective_start is None and incremental_mode:
        effective_start = str(incr_start)

    input_filename = f"{symbol}_{market_type.upper()}_{base_download}.csv"
    input_file = os.path.join(downloads_dir, input_filename)

    # ========== 2. 打印配置摘要 ==========
    log("📋 配置摘要:")
    log(f"   交易对: {symbol} ({market_type.upper()})")
    log(f"   基础周期: {base_download}")
    log(f"   目标周期: {', '.join(timeframes)}")
    log(f"   数据模式: {source_mode.upper()} ({'滑窗滚动' if source_mode == 'sliding' else '固定K线'})")
    log(f"   输出格式: {output_fmt.upper()}")
    if variant in ("fixed", "roll"):
        log(f"   输出变体: {variant.upper()}")
    log(f"\n📂 路径配置:")
    log(f"   输入文件: {input_filename}")
    log(f"   输出目录: {kline_dir}")
    if effective_start or end:
        log(f"\n⏰ 时间范围: {effective_start or '-∞'} ~ {end or '+∞'}")

    # ========== 3. 读取输入数据 ==========
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"未找到输入文件: {input_file}\n请先运行 Step1 下载数据: python run1_step1_data.py"
        )

    df_base, base_interval = read_base_csv(input_file, start=effective_start, end=end)
    log(f"\n✅ 读取完成: {len(df_base):,} 行")
    log(f"   时间范围: {df_base.index.min()} ~ {df_base.index.max()}")
    log(f"   基础间隔: {base_interval}")

    # ========== 4. 执行重采样 ==========
    log("\n" + "=" * 80)
    log("🔄 开始重采样")
    log("=" * 80)

    produce_fixed = variant == "fixed"
    produce_roll = variant == "roll"

    for tf in timeframes:
        log(f"\n📍 周期: {tf}")

        try:
            if produce_fixed:
                tf_df = resample_ohlcv(df_base, tf, base_interval)
                log(f"   ✓ Fixed: {len(tf_df):,} 行 ({tf_df.index.min()} ~ {tf_df.index.max()})")
                if incremental_mode:
                    p_exist = _find_existing_kline_path(kline_dir, symbol, tf, rolling=False)
                    if p_exist is not None:
                        try:
                            old_df = read_df_auto(str(p_exist))
                            if "timestamp" in old_df.columns:
                                if pd.api.types.is_integer_dtype(old_df["timestamp"]):
                                    old_df["timestamp"] = pd.to_datetime(
                                        old_df["timestamp"], unit="ms", errors="coerce"
                                    )
                                else:
                                    old_df["timestamp"] = pd.to_datetime(old_df["timestamp"], errors="coerce")
                                old_df = old_df.set_index("timestamp")
                            old_df.index.name = "timestamp"
                        except Exception:
                            old_df = None
                        tf_df = safe_concat_dedup(old_df, tf_df)
                        log(f"   🔁 合并历史: {len(tf_df):,} 行")
                save_output(tf_df, kline_dir, symbol, tf, output_fmt, rolling=False)

            if produce_roll:
                tf_roll = rolling_preview_ohlcv(df_base, tf, base_interval)
                log(f"   ✓ Rolling: {len(tf_roll):,} 行")
                if incremental_mode:
                    pr_exist = _find_existing_kline_path(kline_dir, symbol, tf, rolling=True)
                    if pr_exist is not None:
                        try:
                            old_r = read_df_auto(str(pr_exist))
                            if "timestamp" in old_r.columns:
                                old_r["timestamp"] = pd.to_datetime(old_r["timestamp"], errors="coerce")
                                old_r = old_r.set_index("timestamp")
                            old_r.index.name = "timestamp"
                        except Exception:
                            old_r = None
                        tf_roll = safe_concat_dedup(old_r, tf_roll)
                        log(f"   🔁 合并历史(roll): {len(tf_roll):,} 行")
                save_output(tf_roll, kline_dir, symbol, tf, output_fmt, rolling=True)

            log(f"   ✅ 周期 {tf} 完成")

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Step2 周期 {tf} 处理失败: {e}") from e

    # ========== 5. 输出结果摘要 ==========
    log("\n" + "=" * 80)
    log("✅ Step2 K线重采样完成！")
    log("=" * 80)
    log(f"\n📂 输出目录: {kline_dir}")
    log(f"⏱️  处理周期: {', '.join(timeframes)}")
    log(f"🔧 数据模式: {source_mode.upper()}")

    print_latest_timestamp_from_df(df_base)

    log(f"\n💡 下一步操作:")
    log(f"   运行 Step3 生成指标:")
    log(f"   python step3_generate_indicators.py")

    return {
        "symbol": symbol,
        "market_type": market_type,
        "base_download": base_download,
        "base_interval": base_interval,
        "timeframes": timeframes,
        "kline_dir": kline_dir,
        "downloads_dir": downloads_dir,
        "source_mode": source_mode,
        "incremental_mode": incremental_mode,
        "include_rolling": include_rolling,
        "start": df_base.index.min(),
        "end": df_base.index.max(),
        "detected_history": detected_any,
        "output_format": output_fmt,
    }


def main():
    """主流程：完全配置驱动的K线重采样"""
    args = parse_args()

    try:
        loader = ConfigLoader()
        cfg = loader.load_step2_config()
        if not cfg:
            print("❌ 未找到或无法加载配置文件")
            return 1
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        import traceback

        traceback.print_exc()
        return 1

    try:
        execute_step2(
            cfg,
            start=args.start,
            end=args.end,
            output_format=args.output_format,
            verbose=True,
        )
        return 0
    except Exception as e:
        print(f"❌ Step2 执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
