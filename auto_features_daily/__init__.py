"""
auto_features_daily: 每日自动更新外挂模块

🔌 独立外挂，与主项目解耦
   - 只通过 step1_data.run_step1_with_override() 接口调用
   - 配置文件独立管理
   - 可独立部署到任意环境

包含：
- run_daily_features.py: 多币种调度器主脚本
- config.yaml: 币种/调度配置
- setup_daily_cron.sh: Cron 一键安装
- finrl-daily-features.service/timer: Systemd 配置

使用方式：
    # 手动运行
    python -m features_engineering.auto_features_daily.run_daily_features --force
    
    # 或直接运行
    python features_engineering/auto_features_daily/run_daily_features.py --force
"""

__version__ = "1.0.0"
__all__ = ["run_daily_update", "load_config"]

