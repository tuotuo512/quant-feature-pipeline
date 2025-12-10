"""
tools: 通用工具集

包含：
- time_index: 周期与频率映射、分钟转换
- filling: 网格补齐与NaN填充
- io_paths: 通用读写与last_timestamp工具
- incremental: 增量区间计算与安全拼接
- export_utils: NPZ 导出、健康检查、schema 计算

🔌 每日自动更新外挂已独立为 auto_features_daily/ 模块

注意：保持轻依赖，仅使用 pandas/numpy。
"""

from .time_index import timeframe_to_pandas_freq, timeframe_to_minutes  # noqa: F401
from .filling import (
    fill_base_ohlcv_grid,
    fill_kline_grid,
    fill_nan,
)  # noqa: F401
from .io_paths import read_df_auto, write_df_auto, get_last_timestamp  # noqa: F401
from .incremental import compute_increment_range, safe_concat_dedup  # noqa: F401
