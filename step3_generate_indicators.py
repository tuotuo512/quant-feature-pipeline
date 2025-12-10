#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step3: 技术指标生成器（完全配置驱动版）

✨ 特性：
  - 完全由 main_config.yaml + base_indicators.yaml 驱动
  - 自动处理 fixed/sliding 模式
  - 支持指标子集选择或全量计算
  - 支持时间范围过滤

📋 用法：
  python step3_generate_indicators.py
  python step3_generate_indicators.py --start 2024-01-01 --end 2024-12-31

🔧 配置：
  - 全局配置: main_config.yaml
  - 指标参数: base_indicators.yaml
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
    from features_engineering.indicators import IndicatorCalculator
    from features_engineering.congfigs.config_loader import ConfigLoader
except Exception as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

# 🔥 动态设置指标配置路径（从当前脚本所在目录推导）
if "INDICATORS_CONFIG" not in os.environ:
    _CONFIG_DIR = os.path.join(os.path.dirname(__file__), "congfigs")
    _INDICATORS_CONFIG = os.path.join(_CONFIG_DIR, "base_indicators.yaml")
    os.environ["INDICATORS_CONFIG"] = os.path.abspath(_INDICATORS_CONFIG)

# 工具（增量/读写）
from features_engineering.tools.io_paths import read_df_auto
from features_engineering.tools.incremental import safe_concat_dedup


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Step3: 技术指标生成器（完全配置驱动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python step3_generate_indicators.py

  # 指定时间范围
  python step3_generate_indicators.py --start 2024-01-01 --end 2024-12-31

  # 覆盖输出格式
  python step3_generate_indicators.py --output_format parquet
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


def read_kline(
    kline_dir: str,
    symbol: str,
    tf: str,
    start: str | None,
    end: str | None,
    rolling: bool = False,
    preferred_fmt: str | None = None,
) -> pd.DataFrame:
    """读取指定周期的K线数据（根目录结构：kline/{symbol}_{tf}.parquet，与 Step2 对齐）。"""
    fname_new = f"{symbol}_{tf}"

    # 🔥 根目录结构（与 Step2 保存路径对齐）
    csv_path = Path(kline_dir) / f"{fname_new}.csv"
    parquet_path = Path(kline_dir) / f"{fname_new}.parquet"

    # 依据首选格式读取，不存在则自动回退
    fmt = (preferred_fmt or "parquet").lower()
    fmt = "parquet" if fmt not in ("csv", "parquet", "both") else fmt
    paths_try: list[tuple[str, Path]]
    if fmt == "parquet":
        paths_try = [("parquet", parquet_path), ("csv", csv_path)]
    elif fmt == "csv":
        paths_try = [("csv", csv_path), ("parquet", parquet_path)]
    else:  # both
        paths_try = [("parquet", parquet_path), ("csv", csv_path)]

    df = None
    used = None
    for kind, path in paths_try:
        if path.exists():
            used = kind
            if kind == "csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            break

    if df is None:
        # 兼容旧路径：子目录结构（向后兼容）
        csv_old_subdir = Path(kline_dir) / tf / f"{fname_new}.csv"
        parquet_old_subdir = Path(kline_dir) / tf / f"{fname_new}.parquet"
        if parquet_old_subdir.exists():
            df = pd.read_parquet(parquet_old_subdir)
        elif csv_old_subdir.exists():
            df = pd.read_csv(csv_old_subdir)
        else:
            raise FileNotFoundError(f"未找到K线文件: {parquet_path} 或 {csv_path}")

    # 解析时间索引（智能检测：整数用毫秒，字符串自动推断）
    if "timestamp" in df.columns:
        if pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        first_col = df.iloc[:, 0]
        if pd.api.types.is_integer_dtype(first_col):
            df.iloc[:, 0] = pd.to_datetime(first_col, unit="ms", errors="coerce")
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.set_index(df.columns[0])

    df.index.name = "timestamp"

    # 只保留标准OHLCV
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]

    # 排序/去重
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 时间切片
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("筛选后的K线为空，请检查时间范围")

    return df


def save_output(df: pd.DataFrame, ind_dir: str, symbol: str, tf: str, fmt: str):
    """保存指标数据（根目录，不创建按周期子目录；文件名 {symbol}_{tf}_indicators）"""
    output_dir = Path(ind_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{symbol}_{tf}_indicators"

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


def execute_step3(
    cfg: dict,
    start: str | None = None,
    end: str | None = None,
    output_format: str | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """执行 Step3 指标生成逻辑，供脚本与统一流水线复用。"""
    if not cfg:
        raise ValueError("配置不能为空")

    log = print if verbose else (lambda *args, **kwargs: None)

    log("🚀 Step3 指标生成启动（完全配置驱动）\n")

    symbol = cfg.get("symbol", {}).get("trading_pair_std", "ETH_USDT")

    timeframes_cfg = cfg.get("timeframes", {})
    timeframes = timeframes_cfg.get("resample_targets", ["3m", "15m", "30m", "2h"])
    include_rolling = timeframes_cfg.get("include_rolling", False)
    variant = str(timeframes_cfg.get("variant", "")).strip().lower()
    source_mode = cfg.get("rl_build", {}).get("source_mode", "fixed")

    io_cfg = cfg.get("io", {})
    base_dir = io_cfg.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
    kline_dir = io_cfg.get("kline_dir") or f"{base_dir}/rl_live/kline"
    ind_dir = io_cfg.get("indicators_dir") or f"{base_dir}/rl_live/ind"

    output_fmt = output_format or io_cfg.get("output_format", "csv")
    io_overwrite = bool(io_cfg.get("overwrite", False))

    params_cfg = cfg.get("params", {})
    default_params = params_cfg.get("default", {})

    log("📋 配置摘要:")
    log(f"   交易对: {symbol}")
    log(f"   目标周期: {', '.join(timeframes)}")
    log(f"   数据模式: {source_mode.upper()} ({'滑窗滚动' if source_mode == 'sliding' else '固定K线'})")
    log(f"   输出格式: {output_fmt.upper()}")
    if variant in ("fixed", "roll"):
        log(f"   输出变体: {variant.upper()}")
    log(f"\n📂 路径配置:")
    log(f"   K线目录: {kline_dir}")
    log(f"   指标目录: {ind_dir}")
    if start or end:
        log(f"\n⏰ 时间范围: {start or '-∞'} ~ {end or '+∞'}")

    calc = IndicatorCalculator(verbose=verbose)

    log("\n" + "=" * 80)
    log("🔄 开始计算指标")
    log("=" * 80)

    for tf in timeframes:
        log(f"\n📍 周期: {tf}")

        try:
            kline_fixed = read_kline(
                kline_dir, symbol, tf, start, end, rolling=False, preferred_fmt=output_fmt
            )
            log(
                f"   ✓ K线[fixed]: {len(kline_fixed):,} 行 ({kline_fixed.index.min()} ~ {kline_fixed.index.max()})"
            )

            params = default_params

            ind_fixed = calc.calculate_all_indicators(kline_fixed, params=params)
            if ind_fixed is None or ind_fixed.empty:
                raise RuntimeError(f"指标计算失败 (周期: {tf})")

            log(f"   ✓ 指标[fixed]: {ind_fixed.shape[1]} 列")

            ind_roll_data = None
            if variant in ("fixed", "roll"):
                produce_roll = variant == "roll"
            else:
                produce_roll = include_rolling
            if produce_roll:
                try:
                    kline_roll = read_kline(
                        kline_dir, symbol, tf, start, end, rolling=True, preferred_fmt=output_fmt
                    )
                    log(f"   ✓ K线[rolling]: {len(kline_roll):,} 行")
                    ind_roll = calc.calculate_all_indicators(kline_roll, params=params)
                    if ind_roll is not None and not ind_roll.empty:
                        log(f"   ✓ 指标[rolling]: {ind_roll.shape[1]} 列")
                        ind_roll_data = ind_roll
                except FileNotFoundError:
                    log("   ℹ️  未找到 rolling K线，跳过")
                except Exception as e:
                    log(f"   ⚠️  rolling 处理失败: {e}")

            produce_fixed = not produce_roll
            if produce_fixed and not io_overwrite:
                try:
                    for ext in (".parquet", ".csv"):
                        p = Path(ind_dir) / f"{symbol}_{tf}_indicators{ext}"
                        if p.exists():
                            old = read_df_auto(str(p))
                            if "timestamp" in old.columns:
                                if pd.api.types.is_integer_dtype(old["timestamp"]):
                                    old["timestamp"] = pd.to_datetime(
                                        old["timestamp"], unit="ms", errors="coerce"
                                    )
                                else:
                                    old["timestamp"] = pd.to_datetime(old["timestamp"], errors="coerce")
                                old = old.set_index("timestamp")
                            old.index.name = "timestamp"
                            ind_fixed = safe_concat_dedup(old, ind_fixed)
                            log(f"   🔁 合并历史: {len(ind_fixed):,} 行")
                            break
                    else:
                        for ext in (".parquet", ".csv"):
                            p = Path(ind_dir) / tf / f"{symbol}_{tf}_indicators{ext}"
                            if p.exists():
                                old = read_df_auto(str(p))
                                if "timestamp" in old.columns:
                                    if pd.api.types.is_integer_dtype(old["timestamp"]):
                                        old["timestamp"] = pd.to_datetime(
                                            old["timestamp"], unit="ms", errors="coerce"
                                        )
                                    else:
                                        old["timestamp"] = pd.to_datetime(old["timestamp"], errors="coerce")
                                    old = old.set_index("timestamp")
                                old.index.name = "timestamp"
                                ind_fixed = safe_concat_dedup(old, ind_fixed)
                                log(f"   🔁 合并历史: {len(ind_fixed):,} 行")
                                break
                except Exception as _e:
                    log(f"   ⚠️ 合并历史失败: {_e}")

            if produce_fixed:
                save_output(ind_fixed, ind_dir, symbol, tf, output_fmt)

            if ind_roll_data is not None:
                if not io_overwrite:
                    try:
                        for ext in (".parquet", ".csv"):
                            p = Path(ind_dir) / f"{symbol}_{tf}_indicators{ext}"
                            if p.exists():
                                oldr = read_df_auto(str(p))
                                if "timestamp" in oldr.columns:
                                    oldr["timestamp"] = pd.to_datetime(oldr["timestamp"], errors="coerce")
                                    oldr = oldr.set_index("timestamp")
                                oldr.index.name = "timestamp"
                                ind_roll_data = safe_concat_dedup(oldr, ind_roll_data)
                                log(f"   🔁 合并历史: {len(ind_roll_data):,} 行")
                                break
                        else:
                            for ext in (".parquet", ".csv"):
                                p = Path(ind_dir) / tf / f"{symbol}_{tf}_indicators{ext}"
                                if p.exists():
                                    oldr = read_df_auto(str(p))
                                    if "timestamp" in oldr.columns:
                                        oldr["timestamp"] = pd.to_datetime(oldr["timestamp"], errors="coerce")
                                        oldr = oldr.set_index("timestamp")
                                    oldr.index.name = "timestamp"
                                    ind_roll_data = safe_concat_dedup(oldr, ind_roll_data)
                                    log(f"   🔁 合并历史: {len(ind_roll_data):,} 行")
                                    break
                    except Exception as _e:
                        log(f"   ⚠️ 合并历史失败: {_e}")
                save_output(ind_roll_data, ind_dir, symbol, tf, output_fmt)

            log(f"   ✅ 周期 {tf} 完成")

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Step3 周期 {tf} 处理失败: {e}") from e

    log("\n" + "=" * 80)
    log("✅ Step3 指标生成完成！")
    log("=" * 80)
    log(f"\n📂 输出目录: {ind_dir}")
    log(f"⏱️  处理周期: {', '.join(timeframes)}")
    log(f"🔧 数据模式: {source_mode.upper()}")

    from features_engineering.tools.io_paths import print_latest_timestamp

    for tf in timeframes:
        kline_path = Path(kline_dir) / f"{symbol}_{tf}.parquet"
        if not kline_path.exists():
            kline_path = Path(kline_dir) / f"{symbol}_{tf}.csv"
        if kline_path.exists():
            print_latest_timestamp(str(kline_path), fast=True)
            break

    log(f"\n💡 下一步操作:")
    log(f"   运行 Step4 融合特征:")
    log(f"   python step4_merge_features.py")

    return {
        "symbol": symbol,
        "timeframes": timeframes,
        "kline_dir": kline_dir,
        "indicators_dir": ind_dir,
        "source_mode": source_mode,
        "include_rolling": include_rolling,
        "variant": variant,
        "output_format": output_fmt,
        "start": start,
        "end": end,
    }


def main():
    """主流程：完全配置驱动的指标生成"""
    args = parse_args()

    try:
        loader = ConfigLoader()
        cfg = loader.load_step3_config()
        if not cfg:
            print("❌ 未找到或无法加载配置文件")
            return 1

    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        import traceback

        traceback.print_exc()
        return 1

    try:
        execute_step3(
            cfg,
            start=args.start,
            end=args.end,
            output_format=args.output_format,
            verbose=True,
        )
        return 0
    except Exception as e:
        print(f"❌ Step3 执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
