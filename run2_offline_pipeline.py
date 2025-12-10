# 3) volume 缺失填 0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run2: 离线流水线统一入口（Step2 → Step5）

当前脚本直接调用 `unified_feature_pipeline.generate_rl_features`
一次性完成 Step2/Step3/Step4/Step5，并保持原有的 CLI 体验。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in (PROJECT_ROOT, CURRENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from features_engineering.congfigs.config_loader import ConfigLoader
from tools.io_paths import IOManager
from unified_feature_pipeline import generate_rl_features, PipelineResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run2: 离线流水线 Step2→Step5（统一入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run2_offline_pipeline.py
  python run2_offline_pipeline.py --start 2024-01-01 --end 2024-12-31
  python run2_offline_pipeline.py --sample_ratio 0.001
        """,
    )
    parser.add_argument("--start", type=str, default=None, help="起始时间(可选)，例如 2024-01-01 00:00:00")
    parser.add_argument("--end", type=str, default=None, help="结束时间(可选)")
    parser.add_argument("--sample_ratio", type=float, default=None, help="采样比例(0-1)，用于快速验算")
    parser.add_argument("--output_dir", type=str, default=None, help="重定向 RL npz 输出目录")
    parser.add_argument("--verbose", action="store_true", help="打印完整流水线日志（默认已开启）")
    parser.add_argument(
        "--legacy-output-format",
        dest="legacy_output_format",
        choices=["csv", "parquet", "both"],
        default=None,
        help="兼容旧参数，实际输出仍由 main_config.yaml 控制",
    )
    return parser.parse_args()


def _export_merged_header_txt(merged_file: str, header_txt_path: str) -> None:
    """复用旧逻辑，输出 merged 文件的列信息与时间范围。"""
    lower = merged_file.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(merged_file)
    else:
        df = pd.read_csv(merged_file)

    if "timestamp" in df.columns:
        if pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = df["timestamp"]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    else:
        first_col = df.iloc[:, 0]
        if pd.api.types.is_integer_dtype(first_col):
            df.iloc[:, 0] = pd.to_datetime(first_col, unit="ms", errors="coerce")
        else:
            df.iloc[:, 0] = pd.to_datetime(first_col, errors="coerce")
        ts = df.iloc[:, 0]

    cols = [c for c in df.columns if c.lower() not in ("timestamp", "time", "datetime")]
    out_lines = [
        "=" * 80,
        "📊 Merged特征文件元数据摘要",
        "=" * 80,
        "",
        f"📂 文件路径: {merged_file}",
        "",
        "=" * 80,
        "⏰ 时间范围",
        "=" * 80,
        f"起始时间: {ts.min()}",
        f"结束时间: {ts.max()}",
        f"数据行数: {len(df):,}",
        "",
        "=" * 80,
        f"📋 特征列表 (共 {len(cols)} 列)",
        "=" * 80,
    ]
    out_lines.extend(f"  • {c}" for c in cols)
    Path(header_txt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(header_txt_path).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"📝 已生成特征摘要: {Path(header_txt_path).name}")


def _print_config_summary(cfg: dict, io_mgr: IOManager) -> None:
    exchange = cfg.get("exchange", {}).get("name", "unknown")
    symbol = cfg.get("symbol", {}).get("trading_pair_std", "UNKNOWN")
    market_type = cfg.get("symbol", {}).get("market_type", "swap")
    tf_cfg = cfg.get("timeframes", {}) or {}
    base = tf_cfg.get("base_download", "1m")
    targets = tf_cfg.get("resample_targets", [])
    source_mode = cfg.get("rl_build", {}).get("source_mode", "fixed")

    print("📋 配置摘要:")
    print(f"   交易所: {exchange.upper()}")
    print(f"   交易对: {symbol} ({market_type.upper()})")
    print(f"   基础周期: {base}")
    print(f"   目标周期: {', '.join(targets)}")
    print(f"   数据模式: {source_mode.upper()}")
    print(f"\n📂 路径配置:")
    print(f"   下载目录: {io_mgr.downloads_dir}")
    print(f"   K线目录: {io_mgr.kline_dir}")
    print(f"   指标目录: {io_mgr.indicators_dir}")
    print(f"   融合目录: {io_mgr.merged_dir}")
    print(f"   RL输出目录: {io_mgr.rl_ready_dir}")


def _summarize_pipeline_result(res: PipelineResult) -> None:
    print("\n" + "=" * 80)
    print("✅ Run2 离线流水线执行完成！")
    print("=" * 80)
    print(f"📦 merged:   {res.merged_path}")
    print(f"📦 features: {res.features_path}")
    print(f"📦 labels:   {res.labels_path}")
    print(f"📊 记录数: {res.records}")

    header_txt = str(Path(res.merged_path).with_suffix("")) + "_header.txt"
    try:
        _export_merged_header_txt(res.merged_path, header_txt)
    except Exception as e:
        print(f"⚠️ 生成 merged 摘要失败: {e}")

    print("\n💡 下一步操作:")
    print("   1) 使用 preflight/run_preflight_seed.py 对比实盘 vs 训练特征")
    print("   2) 使用 run3_featueres_unified.py 或训练脚本继续下游流程")


def main() -> int:
    args = parse_args()

    if args.legacy_output_format:
        print(f"ℹ️ 提示: --legacy-output-format 已废弃，仍将使用 main_config.yaml 中的 io.output_format")

    try:
        loader = ConfigLoader()
        main_cfg = loader.load_main_config()
    except Exception as e:
        print(f"❌ 加载 main_config.yaml 失败: {e}")
        return 1

    if not main_cfg:
        print("❌ main_config.yaml 为空或不存在")
        return 1

    io_mgr = IOManager(main_cfg)
    _print_config_summary(main_cfg, io_mgr)

    print("\n" + "=" * 80)
    print("🔄 开始执行统一流水线")
    print("=" * 80)

    res = generate_rl_features(
        mode="offline",
        start=args.start,
        end=args.end,
        sample_ratio=args.sample_ratio,
        output_dir=args.output_dir,
        verbose=True,
    )

    _summarize_pipeline_result(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
