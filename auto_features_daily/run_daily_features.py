#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔌 每日自动更新外挂 - 多币种调度器
独立模块，与主项目解耦，只通过接口调用

✅ 支持右键运行（IDE 直接运行）

启动方式:
    1. 右键运行 / 手动运行:
       直接在 IDE 中右键运行本文件，或：
       cd /root/FinRL_bn && conda activate finrl_ml_env
       python features_engineering/auto_features_daily/run_daily_features.py --force

    2. Cron 定时 (每日 00:10 UTC):
       10 0 * * * cd /root/FinRL_bn && /root/miniconda3/envs/finrl_ml_env/bin/python features_engineering/auto_features_daily/run_daily_features.py >> /root/FinRL_bn/logs/auto_features_daily/cron.log 2>&1

    3. 开机自动 (@reboot):
       @reboot sleep 30 && cd /root/FinRL_bn && /root/miniconda3/envs/finrl_ml_env/bin/python features_engineering/auto_features_daily/run_daily_features.py --boot >> /root/FinRL_bn/logs/auto_features_daily/boot.log 2>&1

    4. Systemd Timer (推荐生产环境):
       参见同目录 finrl-daily-features.service/timer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ========== 路径设置（支持右键运行）==========
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = FEATURES_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "config.yaml"

# 切换工作目录到项目根目录（支持右键运行）
os.chdir(PROJECT_ROOT)

# 确保项目根目录在 sys.path（用于导入 step1_data）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))


def load_config() -> Dict[str, Any]:
    """加载本模块的独立配置"""
    import yaml
    
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    return config


def setup_logging(config: Dict[str, Any]) -> Path:
    """设置日志目录"""
    output_cfg = config.get("output", {})
    log_dir = output_cfg.get("log_dir", "logs/auto_features_daily")
    
    # 转换为绝对路径（相对于项目根目录）
    if not os.path.isabs(log_dir):
        log_dir = PROJECT_ROOT / log_dir
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_enabled_symbols(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """获取启用的币种列表"""
    symbols = config.get("symbols", [])
    enabled = []
    
    for sym in symbols:
        if sym.get("enabled", True):
            enabled.append({
                "name": sym.get("name", ""),
                "exchange_pair": sym.get("exchange_pair", ""),
            })
    
    return enabled


def detect_environment() -> str:
    """检测运行环境: local / cloud"""
    cloud_indicators = [
        "/etc/cloud_env",
        "/etc/aliyun_ecs",
        "/etc/tencent_cloud",
    ]
    
    for indicator in cloud_indicators:
        if os.path.exists(indicator):
            return "cloud"
    
    try:
        import socket
        hostname = socket.gethostname().lower()
        cloud_keywords = ["ali", "tencent", "aws", "azure", "gcp", "ecs", "cvm"]
        for kw in cloud_keywords:
            if kw in hostname:
                return "cloud"
    except Exception:
        pass
    
    return "local"


def run_step1_for_symbol(
    symbol_name: str,
    exchange_pair: str,
    config: Dict[str, Any],
) -> bool:
    """
    为单个币种运行 Step1 数据下载
    
    🔌 解耦设计：只调用 step1_data.run_step1_with_override() 接口
    """
    # 🔌 延迟导入，仅在需要时加载主项目模块
    from step1_data import run_step1_with_override
    
    download_cfg = config.get("download", {})
    exchange_cfg = config.get("exchange", {})
    
    days = download_cfg.get("days", 280)
    base_tf = download_cfg.get("base_tf", "1m")
    exchange_name = exchange_cfg.get("name", "okx")
    market_type = exchange_cfg.get("market_type", "swap")
    
    print(f"\n{'='*60}")
    print(f"📥 开始下载: {symbol_name} ({exchange_pair})")
    print(f"   交易所: {exchange_name} | 市场: {market_type} | 周期: {base_tf}")
    print(f"   天数: {days}")
    print(f"{'='*60}")
    
    try:
        # 🔌 核心调用：通过标准接口与主项目交互
        run_step1_with_override(
            days=days,
            symbol_std=symbol_name,
            symbol_exchange=exchange_pair,
            exchange_name=exchange_name,
            market_type=market_type,
            base_tf=base_tf,
        )
        print(f"✅ {symbol_name} 下载完成")
        return True
    except Exception as e:
        print(f"❌ {symbol_name} 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_daily_update(config: Dict[str, Any] = None, force: bool = False) -> Dict[str, bool]:
    """
    执行每日更新（所有启用的币种）
    
    Args:
        config: 配置字典（可选，默认从 config.yaml 加载）
        force: 是否强制运行
    
    Returns:
        {symbol_name: success} 字典
    """
    if config is None:
        config = load_config()
    
    # 设置日志
    log_dir = setup_logging(config)
    
    # 检测环境
    env = detect_environment()
    print(f"\n🌐 运行环境: {env}")
    print(f"📁 日志目录: {log_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取启用的币种
    symbols = get_enabled_symbols(config)
    if not symbols:
        print("⚠️ 没有启用的币种，跳过")
        return {}
    
    print(f"\n📊 待更新币种: {[s['name'] for s in symbols]}")
    
    # 下载参数
    download_cfg = config.get("download", {})
    symbol_delay = download_cfg.get("symbol_delay_sec", 5)
    max_retries = download_cfg.get("max_retries", 3)
    
    # 逐个币种下载
    results: Dict[str, bool] = {}
    
    for i, sym in enumerate(symbols):
        symbol_name = sym["name"]
        exchange_pair = sym["exchange_pair"]
        
        # 重试逻辑
        success = False
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"   ⏳ 第 {attempt}/{max_retries} 次重试...")
                time.sleep(symbol_delay * 2)
            
            success = run_step1_for_symbol(
                symbol_name=symbol_name,
                exchange_pair=exchange_pair,
                config=config,
            )
            
            if success:
                break
        
        results[symbol_name] = success
        
        # 币种间延迟（避免限流）
        if i < len(symbols) - 1:
            print(f"\n⏳ 等待 {symbol_delay} 秒后继续下一个币种...")
            time.sleep(symbol_delay)
    
    # 汇总
    print("\n" + "=" * 60)
    print("📊 每日更新汇总")
    print("=" * 60)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for name, ok in results.items():
        status = "✅ 成功" if ok else "❌ 失败"
        print(f"   {name}: {status}")
    
    print(f"\n🎯 完成: {success_count}/{total_count}")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return results


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="🔌 每日自动更新外挂 - 多币种调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_daily_features.py                # 使用默认配置
  python run_daily_features.py --force        # 强制立即运行
  python run_daily_features.py --days 100     # 覆盖下载天数
  python run_daily_features.py --symbols ETH_USDT BTC_USDT  # 指定币种
        """,
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制运行（忽略时间检查）",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=None,
        help="覆盖下载天数",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=None,
        help="指定币种列表（覆盖配置）",
    )
    parser.add_argument(
        "--boot",
        action="store_true",
        help="开机模式（等待网络就绪）",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="指定配置文件路径（默认使用同目录 config.yaml）",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔌 FinRL 每日数据更新外挂")
    print("=" * 60)
    
    # 加载配置
    try:
        if args.config:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            print(f"📄 使用配置: {args.config}")
        else:
            config = load_config()
            print(f"📄 使用配置: {CONFIG_FILE}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # 开机模式：等待网络就绪
    if args.boot:
        schedule_cfg = config.get("schedule", {})
        delay = schedule_cfg.get("boot_delay_sec", 30)
        print(f"🔄 开机模式：等待 {delay} 秒...")
        time.sleep(delay)
    
    # CLI 参数覆盖配置
    if args.days is not None:
        config.setdefault("download", {})["days"] = args.days
        print(f"📝 覆盖下载天数: {args.days}")
    
    if args.symbols:
        config["symbols"] = [
            {"name": s, "exchange_pair": s.replace("_", "/"), "enabled": True}
            for s in args.symbols
        ]
        print(f"📝 覆盖币种列表: {args.symbols}")
    
    # 执行更新
    results = run_daily_update(config, force=args.force)
    
    # 退出码
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
