#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run3 Unified: Step5 快速导出包装器（复用 unified_feature_pipeline） 主要验证用的
"""

import argparse
import os
import shutil
import sys
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in (PROJECT_ROOT, CURRENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from features_engineering.unified_feature_pipeline import generate_rl_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run3 Unified: 快速将 merged 转换为 RL 特征（统一流水线包装器）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="指定 merged 文件（默认按配置自动定位）")
    parser.add_argument("--start", type=str, default=None, help="可选：限制时间起点，仅完整离线流程时生效")
    parser.add_argument("--end", type=str, default=None, help="可选：限制时间终点")
    parser.add_argument("--sample", type=float, default=None, help="尾部采样比例 (0-1)，在 Step5 中截尾")
    parser.add_argument("--output-dir", type=str, default=None, help="自定义输出目录（features/labels 共用）")
    parser.add_argument("--output-features", type=str, default=None, help="另存 FEATURES 至指定路径")
    parser.add_argument("--output-labels", type=str, default=None, help="另存 LABELS 至指定路径")
    parser.add_argument("--no-reuse", dest="reuse_merged", action="store_false", help="强制重跑 Step2/3/4")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    parser.set_defaults(reuse_merged=True)
    return parser.parse_args()


def maybe_copy(src: str, dst: Optional[str], label: str) -> str:
    if not dst:
        return src
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"   📦 {label}: {dst}")
    return dst


def main() -> int:
    args = parse_args()

    print("\n" + "=" * 80)
    print("Run3 Unified: 统一流水线包装器")
    print("=" * 80)

    res = generate_rl_features(
        mode="offline",
        start=args.start,
        end=args.end,
        sample_ratio=args.sample,
        output_dir=args.output_dir,
        verbose=args.verbose,
        reuse_merged=args.reuse_merged,
        merged_path=args.input,
    )

    features_path = maybe_copy(res.features_path, args.output_features, "FEATURES")
    labels_path = (
        maybe_copy(res.labels_path, args.output_labels, "LABELS") if res.labels_path else res.labels_path
    )

    print("\n" + "=" * 80)
    print("🎉 Run3 执行完成")
    print("=" * 80)
    print(f"模式: {res.mode}")
    print(f"基准周期: {res.base_timeframe}")
    print(f"记录数: {res.records}")
    print(f"FEATURES: {features_path}")
    if labels_path:
        print(f"LABELS: {labels_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
