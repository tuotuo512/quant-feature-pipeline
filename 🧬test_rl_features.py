#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精简测试脚本：生成7张核心图表
每张图：价格K线主图 + 4个不同周期的同一指标副图

1. 价格 + 多周期动量（1m/3m/15m/2h）
2. 价格 + 多周期市场状态（1m/3m/15m/2h）
3. 价格 + 多周期ATR（1m/3m/15m/2h）
4. 价格 + 多周期RSI（1m/3m/15m/2h）
5. 价格 + 多周期布林带宽度（1m/3m/15m/2h）
6. 特征相关性热力图
7. 30m 动量 × 近5个30m周期成交量均值

python /root/FinRL_bn/features_engineering/test_rl_features.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import argparse
import yaml
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import re

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-darkgrid")

# ============================================
# 🎯 总控参数（可在此直接修改默认值）
# ============================================
DEFAULT_LAST_N_BARS = 1000  # 默认绘制最新300根K线
# 可选预设：
#   - 100: 快速预览（约6-8小时，3m周期）
#   - 300: 标准视图（约15-24小时，3m周期）
#   - 500: 详细分析（约1-2天，3m周期）
#   - 1000: 完整回顾（约2-4天，3m周期）
#   - 2000: 长期趋势（约4-7天，3m周期）

# ============================================
# 辅助函数
# ============================================


def load_main_config() -> dict:
    """加载 main_config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), "congfigs", "main_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"⚠️ 无法加载配置: {e}")
        return {}


def get_npz_path_from_config() -> str:
    """从配置自动定位RL_FEATURES（强制优先最小周期，通常为3m；兼容旧命名）。"""
    cfg = load_main_config()
    io_cfg = cfg.get("io", {}) or {}
    base_dir = io_cfg.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
    rl_ready_dir = io_cfg.get("rl_ready_dir", "{io.base_dir}/data_ready")
    rl_ready_dir = rl_ready_dir.replace("{io.base_dir}", base_dir)
    symbol = cfg.get("symbol", {}).get("trading_pair_std", "ETH_USDT")

    # 辅助函数：周期转分钟数
    def _minutes(p: str) -> int:
        try:
            p = str(p).lower().strip()
            num = int(re.findall(r"\d+", p)[0])
            unit = p[-1]
            return num if unit == "m" else (num * 60 if unit == "h" else num * 1440)
        except Exception:
            return 10**9

    # 1) 扫描目录，匹配 {symbol}_{tf}_rl_features.npz
    try:
        files = []
        if os.path.isdir(rl_ready_dir):
            files = [
                f
                for f in os.listdir(rl_ready_dir)
                if f.startswith(f"{symbol}_") and f.endswith("_rl_features.npz")
            ]
        tf_to_file = {}
        for fname in files:
            m = re.match(rf"^{re.escape(symbol)}_(\d+[mhd])_rl_features\\.npz$", fname, flags=re.IGNORECASE)
            if m:
                tf = m.group(1).lower()
                tf_to_file[tf] = os.path.join(rl_ready_dir, fname)

        # 🔥 策略：强制选择最小周期（通常为3m）
        if tf_to_file:
            tf_sorted = sorted(tf_to_file.keys(), key=_minutes)
            chosen_tf = tf_sorted[0]
            chosen = tf_to_file[chosen_tf]
            logger.info(f"📂 选择最小周期特征包({chosen_tf}): {chosen}")
            return chosen
    except Exception as e:
        logger.warning(f"⚠️ 扫描RL特征目录失败: {e}")

    # 2) 兼容旧命名：{symbol}_rl_features.npz
    fallback = os.path.join(rl_ready_dir, f"{symbol}_rl_features.npz")
    if os.path.exists(fallback):
        logger.info(f"📂 使用兼容旧命名: {fallback}")
        return fallback

    # 3) 最后兜底：glob 再扫一次（防某些环境差异）
    try:
        import glob as _glob

        pats = _glob.glob(os.path.join(rl_ready_dir, f"{symbol}_*_rl_features.npz"))
        if pats:
            # 按文件名排序，选择最小周期
            tf_map = {}
            for pth in pats:
                m = re.search(rf"_(\d+[mhd])_rl_features\\.npz$", pth, flags=re.IGNORECASE)
                if m:
                    tf = m.group(1).lower()
                    tf_map[tf] = pth
            if tf_map:
                sorted_tfs = sorted(tf_map.keys(), key=_minutes)
                chosen = tf_map[sorted_tfs[0]]
                logger.info(f"📂 兜底选择最小周期特征包({sorted_tfs[0]}): {chosen}")
                return chosen
            # 如果无法解析周期，选第一个
            logger.info(f"📂 兜底选择任意特征包: {pats[0]}")
            return pats[0]
    except Exception:
        pass

    logger.info(f"📂 使用兼容旧命名(可能不存在): {fallback}")
    return fallback


def load_reasoning_data(npz_path):
    """加载RL_FEATURES NPZ数据"""
    logger.info(f"📂 加载RL_FEATURES数据: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        # RL_FEATURES包结构：observations/feature_names/feature_groups/timestamps/prices/schema_sha/metadata
        observations = data["observations"].copy()  # 复制出来以便在with块外使用
        feature_names = data["feature_names"].copy()
        raw_ts = data["timestamps"].copy()
        prices = data.get("prices").copy() if "prices" in data else None  # 收盘价/基准价格
        # 可选：feature_groups, schema_sha, metadata
        if "feature_groups" in data:
            logger.info(f"📊 特征分组信息已加载")
        if "schema_sha" in data:
            # schema_sha 是0维numpy数组，需要用.item()获取字符串值
            schema_str = (
                data["schema_sha"].item() if hasattr(data["schema_sha"], "item") else str(data["schema_sha"])
            )
            logger.info(f"📊 Schema SHA: {schema_str[:16]}...")
    # 标准化时间戳（自动判别单位：ns/us/ms/s），仅用前2个样本推断
    timestamps = normalize_timestamps(raw_ts)
    if isinstance(timestamps, pd.DatetimeIndex) and len(timestamps) > 0:
        head_preview = [timestamps[0], timestamps[min(1, len(timestamps) - 1)]]
        logger.info(f"🕒 时间预览: {head_preview[0]} → {head_preview[-1]}")
    logger.info(f"✅ 数据形状: {observations.shape}, 特征数: {len(feature_names)}, 时间点: {len(timestamps)}")
    return observations, feature_names, timestamps, prices


def normalize_timestamps(ts_arr):
    """将NPZ中的timestamps标准化为 pandas.DatetimeIndex（推断单位）。
    规则：
      - 若为数值：用前1-2个样本判断数量级：
          ns≈1e18，us≈1e15，ms≈1e12，s≈1e9
      - 若为字符串/对象：直接 to_datetime(utc=True)
    最终输出为无时区的 DatetimeIndex。
    """
    try:
        arr = np.array(ts_arr)
        # 数值路径
        if np.issubdtype(arr.dtype, np.number):
            # 取前两个非nan样本
            sample = arr[:2].astype(np.float64)
            m = np.nanmax(sample)
            unit = "s"
            if m >= 1e17:
                unit = "ns"
            elif m >= 1e14:
                unit = "us"
            elif m >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(arr, unit=unit, utc=True, errors="coerce")
        else:
            # 字符串/对象路径
            dt = pd.to_datetime(arr, utc=True, errors="coerce")
        # 去时区，转为naive，便于matplotlib
        try:
            return dt.tz_convert(None)
        except Exception:
            try:
                return dt.tz_localize(None)
            except Exception:
                return dt
    except Exception as e:
        logger.warning(f"⚠️ 时间戳标准化失败: {e}，回退到直接to_datetime")
        dt = pd.to_datetime(ts_arr, utc=True, errors="coerce")
        try:
            return dt.tz_convert(None)
        except Exception:
            return dt


def extract_all_periods_from_features(feature_names) -> list:
    """从特征名称提取所有周期"""
    periods = set()
    for name in feature_names:
        name_str = str(name)
        if "_" in name_str:
            prefix = name_str.split("_")[0]
            if any(prefix.endswith(unit) for unit in ["m", "h", "d"]):
                try:
                    num_part = prefix[:-1]
                    if num_part.isdigit():
                        periods.add(prefix)
                except Exception:
                    pass

    def period_to_minutes(p: str) -> int:
        num = int(p[:-1])
        unit = p[-1]
        if unit == "m":
            return num
        elif unit == "h":
            return num * 60
        elif unit == "d":
            return num * 1440
        return 0

    sorted_periods = sorted(periods, key=period_to_minutes)
    logger.info(f"🔍 提取到的周期: {sorted_periods}")
    return sorted_periods


def auto_detect_base_period(feature_names) -> str:
    """自动检测基准周期（最小周期）"""
    all_periods = extract_all_periods_from_features(feature_names)
    if not all_periods:
        cfg = load_main_config()
        return cfg.get("timeframes", {}).get("base_download", "1m")
    return all_periods[0]


def draw_candlestick(ax, o, h, l, c, time_axis=None):
    """绘制K线图

    Args:
        ax: matplotlib axis
        o, h, l, c: OHLC数据
        time_axis: 时间轴（datetime或索引）
    """
    # 统一将时间轴转换为数值坐标，避免Datetime与Timedelta导致的图形异常
    if time_axis is not None and isinstance(time_axis[0], pd.Timestamp):
        x_values = mdates.date2num(time_axis)
        # 宽度取相邻点的中位步长的80%
        if len(x_values) > 1:
            step = np.median(np.diff(x_values))
            width = float(step) * 0.8
        else:
            width = (1.0 / (24 * 60)) * 0.8  # 约1分钟
    else:
        x_values = np.arange(len(o))
        width = 0.8

    for i in range(len(o)):
        x = x_values[i]
        color = "#26a69a" if c[i] >= o[i] else "#ef5350"
        body_height = float(abs(c[i] - o[i]))
        body_bottom = float(min(o[i], c[i]))

        # 蜡烛实体
        ax.add_patch(
            Rectangle(
                (x - width / 2.0, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
        )

        # 上下影线（垂直线）
        ax.vlines(x, l[i], h[i], color=color, linewidth=0.8, alpha=0.6)


# ============================================
# 核心6图
# ============================================


def plot_price_with_indicator(
    states,
    feature_names,
    timestamps,
    prices,
    output_dir,
    last_n,
    indicator_name,
    ylabel,
    title,
    filename,
    plot_zero_line=False,
    y_range=None,
    discrete=False,
):
    """
    通用函数：价格K线 + 多周期指标副图

    Args:
        indicator_name: 指标列名模式，如 'mom', 'market_state', 'rsi14'
        ylabel: Y轴标签
        title: 图表标题
        filename: 输出文件名
        plot_zero_line: 是否绘制零线
        y_range: Y轴范围，如 (-1, 1)
        discrete: 是否为离散值（如市场状态）
    """
    logger.info(f"📊 绘制{title}...")

    base_period = auto_detect_base_period(feature_names)
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}

    # 获取OHLC索引
    ohlc_cols = {}
    for col in ["open", "high", "low", "close"]:
        cand = f"{base_period}_{col}"
        if cand in name_to_idx:
            ohlc_cols[col] = name_to_idx[cand]

    if len(ohlc_cols) < 4:
        logger.warning("⚠️ OHLC数据不完整")
        return

    # 时间窗口
    n = len(timestamps)
    start = max(0, n - int(last_n))
    idxs = np.arange(start, n)

    # 转换时间戳为datetime
    try:
        time_axis = pd.to_datetime(timestamps[idxs])
    except Exception as e:
        logger.warning(f"⚠️ 时间转换失败: {e}，使用索引")
        time_axis = np.arange(len(idxs))

    # 提取OHLC数据
    o = states[idxs, ohlc_cols["open"]]
    h = states[idxs, ohlc_cols["high"]]
    l = states[idxs, ohlc_cols["low"]]

    # 处理收盘价：优先用特征列，否则用prices字段
    if ohlc_cols["close"] == "prices":
        c = prices[idxs]
    else:
        c = states[idxs, ohlc_cols["close"]]

    # 获取多周期指标数据
    all_periods = extract_all_periods_from_features(feature_names)
    target_periods = []
    for p in all_periods[:4]:  # 最多4个周期
        ind_idx = None
        # 首选精确匹配
        exact = f"{p}_{indicator_name}"
        if exact in name_to_idx:
            ind_idx = name_to_idx[exact]
        # 动量兼容：支持 mom_sliding、momXX、包含 _mom 的变体
        if ind_idx is None and indicator_name == "mom":
            alt1 = f"{p}_mom_sliding"
            alt2 = f"{p}_mom20"
            if alt1 in name_to_idx:
                ind_idx = name_to_idx[alt1]
            elif alt2 in name_to_idx:
                ind_idx = name_to_idx[alt2]
            else:
                # 正则匹配：{p}_xxxmomxxx
                pattern = re.compile(rf"^{re.escape(p)}_.*mom.*$", re.IGNORECASE)
                for n, idx in name_to_idx.items():
                    if pattern.match(str(n)):
                        ind_idx = idx
                        break
        if ind_idx is not None:
            target_periods.append((p, ind_idx))

    if not target_periods:
        logger.warning(f"⚠️ 未找到{indicator_name}特征")
        return

    # 创建画布：1主图+N副图
    fig = plt.figure(figsize=(16, 12))
    n_subplots = len(target_periods)
    gs = fig.add_gridspec(n_subplots + 1, 1, height_ratios=[3] + [1] * n_subplots, hspace=0.1)

    ax_main = fig.add_subplot(gs[0])
    axes_sub = [fig.add_subplot(gs[i + 1], sharex=ax_main) for i in range(n_subplots)]

    fig.suptitle(f"{title} (last {last_n})", fontsize=16, fontweight="bold")

    # ====== 主图：K线 ======
    draw_candlestick(ax_main, o, h, l, c, time_axis)
    ax_main.set_ylabel("Price", fontweight="bold", fontsize=11)

    # 设置X轴范围和时间格式
    if isinstance(time_axis[0], pd.Timestamp):
        ax_main.set_xlim(time_axis[0], time_axis[-1])
        # 自动格式化时间轴（4-5个刻度）
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax_main.set_xlim(-0.5, len(idxs) - 0.5)

    ax_main.set_ylim(l.min() * 0.998, h.max() * 1.002)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelbottom=False)
    ax_main.legend([f"{base_period} Candlestick"], loc="upper left", fontsize=9)

    # ====== 副图：各周期指标 ======
    colors = ["#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]
    for idx, (p, ind_idx) in enumerate(target_periods):
        ax = axes_sub[idx]
        ind_data = states[idxs, ind_idx]

        if discrete:
            # 离散值用step plot
            ax.plot(
                time_axis,
                ind_data,
                color=colors[idx % len(colors)],
                linewidth=1.5,
                alpha=0.8,
                marker="o",
                markersize=1,
                drawstyle="steps-post",
            )
        else:
            # 连续值用普通plot
            ax.plot(time_axis, ind_data, color=colors[idx % len(colors)], linewidth=1.2, alpha=0.85)

        if plot_zero_line:
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

        if y_range:
            ax.set_ylim(y_range)
            # 如果是市场状态，添加参考线
            if discrete and y_range == (-1.2, 1.2):
                ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3, linewidth=0.6)
                ax.axhline(y=0, color="orange", linestyle="--", alpha=0.3, linewidth=0.6)
                ax.axhline(y=1, color="green", linestyle="--", alpha=0.3, linewidth=0.6)
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels(["Down", "Range", "Up"], fontsize=8)

        ax.set_ylabel(f"{p} {ylabel}", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 最后一个副图显示时间标签
        if idx < n_subplots - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time", fontweight="bold", fontsize=10)
            if isinstance(time_axis[0], pd.Timestamp):
                # 设置时间格式（4-5个刻度），旋转标签防止重叠
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=9)

        # 添加统计信息
        if not discrete:
            mean_val = np.mean(ind_data)
            ax.text(
                0.02,
                0.95,
                f"mean={mean_val:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    plt.tight_layout()
    path = f"{output_dir}/{filename}"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ {title} saved: {path}")


def plot_core_01_price_momentum(
    states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300
):
    """核心图1：价格 + 多周期动量（新命名规范：mom）"""
    plot_price_with_indicator(
        states,
        feature_names,
        timestamps,
        prices,
        output_dir,
        last_n,
        indicator_name="mom",  # 🔥 新命名规范
        ylabel="Mom",
        title="Core 01: Price & Multi-Period Momentum",
        filename="core_01_price_momentum.png",
        plot_zero_line=True,
    )


def plot_core_01b_price_momentum_overlay(
    states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300
):
    """新增图：价格K线 + 基准周期动量叠加（同一张图，双轴）"""
    logger.info("📊 绘制Core 01b: 价格 + 基准周期动量叠加...")

    base_period = auto_detect_base_period(feature_names)
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}

    # OHLC索引
    ohlc_cols = {}
    for col in ["open", "high", "low"]:
        cand = f"{base_period}_{col}"
        if cand in name_to_idx:
            ohlc_cols[col] = name_to_idx[cand]
    # 收盘价：优先使用 prices 字段
    if prices is not None and len(prices) == len(states):
        ohlc_cols["close"] = "prices"
    elif f"{base_period}_close" in name_to_idx:
        ohlc_cols["close"] = name_to_idx[f"{base_period}_close"]
    else:
        logger.warning("⚠️ OHLC数据不完整：缺少收盘价数据")
        return

    if len(ohlc_cols) < 4:
        logger.warning("⚠️ OHLC数据不完整")
        return

    # 动量列索引（基准周期）
    mom_idx = None
    for cand in [f"{base_period}_mom", f"{base_period}_mom_sliding", f"{base_period}_mom20"]:
        if cand in name_to_idx:
            mom_idx = name_to_idx[cand]
            break
    if mom_idx is None:
        pattern = re.compile(rf"^{re.escape(base_period)}_.*mom.*$", re.IGNORECASE)
        for n, idx in name_to_idx.items():
            if pattern.match(str(n)):
                mom_idx = idx
                break
    if mom_idx is None:
        logger.warning("⚠️ 未找到基准周期动量特征")
        return

    # 时间窗口
    n = len(timestamps)
    start = max(0, n - int(last_n))
    idxs = np.arange(start, n)

    # 时间轴
    try:
        time_axis = pd.to_datetime(timestamps[idxs])
    except Exception as e:
        logger.warning(f"⚠️ 时间转换失败: {e}，使用索引")
        time_axis = np.arange(len(idxs))

    # OHLC数据
    o = states[idxs, ohlc_cols["open"]]
    h = states[idxs, ohlc_cols["high"]]
    l = states[idxs, ohlc_cols["low"]]
    c = prices[idxs] if ohlc_cols["close"] == "prices" else states[idxs, ohlc_cols["close"]]

    # 动量数据
    mom = states[idxs, mom_idx]

    # 画布：主图蜡烛 + 副图动量
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_mom = fig.add_subplot(gs[1], sharex=ax_main)
    fig.suptitle(
        f"Core 01b: Price & Base-Period Momentum Overlay (last {last_n})", fontsize=16, fontweight="bold"
    )

    # 主图：K线
    draw_candlestick(ax_main, o, h, l, c, time_axis)
    ax_main.set_ylabel("Price", fontweight="bold", fontsize=11)
    if isinstance(time_axis[0], pd.Timestamp):
        ax_main.set_xlim(time_axis[0], time_axis[-1])
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax_main.set_xlim(-0.5, len(idxs) - 0.5)
    ax_main.set_ylim(l.min() * 0.998, h.max() * 1.002)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelbottom=False)
    ax_main.legend([f"{base_period} Candlestick"], loc="upper left", fontsize=9)

    # 副图：动量
    ax_mom.plot(time_axis, mom, color="#2E86C1", linewidth=1.2, alpha=0.9)
    ax_mom.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_mom.set_ylim(-1.05, 1.05)
    ax_mom.set_ylabel(f"{base_period} Mom", fontweight="bold", fontsize=9)
    ax_mom.grid(True, alpha=0.3)
    if isinstance(time_axis[0], pd.Timestamp):
        ax_mom.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_mom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax_mom.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=9)
    else:
        ax_mom.set_xlim(-0.5, len(idxs) - 0.5)
        ax_mom.set_xlabel("Index", fontweight="bold", fontsize=10)

    plt.tight_layout()
    path = f"{output_dir}/core_01b_price_momentum_overlay.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Core 01b saved: {path}")


def plot_core_02_price_market_states(
    states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300
):
    """核心图2：价格 + 多周期市场状态"""
    plot_price_with_indicator(
        states,
        feature_names,
        timestamps,
        prices,
        output_dir,
        last_n,
        indicator_name="market_state",  # 🔧 简化命名
        ylabel="State",
        title="Core 02: Price & Multi-Period Market States",
        filename="core_02_price_market_states.png",
        y_range=(-1.2, 1.2),
        discrete=True,
    )


def plot_core_03_price_atr(states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300):
    """核心图3：价格 + 多周期ATR（新命名规范：atr14_pct）"""
    # 尝试atr14_pct（新规范），如果不存在则用旧格式
    base_period = auto_detect_base_period(feature_names)
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}

    # 优先级：atr14_pct（新） > atr_pct（旧） > atr（兼容）
    if f"{base_period}_atr14_pct" in name_to_idx:
        indicator_name = "atr14_pct"
    elif f"{base_period}_atr_pct" in name_to_idx:
        indicator_name = "atr_pct"
    else:
        indicator_name = "atr14"

    plot_price_with_indicator(
        states,
        feature_names,
        timestamps,
        prices,
        output_dir,
        last_n,
        indicator_name=indicator_name,
        ylabel="ATR",
        title="Core 03: Price & Multi-Period ATR",
        filename="core_03_price_atr.png",
        y_range=(0, 1),
    )


def plot_core_04_price_rsi(states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300):
    """核心图4：价格 + 多周期RSI"""
    logger.info("📊 绘制价格+多周期RSI...")

    base_period = auto_detect_base_period(feature_names)
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}

    # 获取OHLC索引
    ohlc_cols = {}
    for col in ["open", "high", "low"]:
        cand = f"{base_period}_{col}"
        if cand in name_to_idx:
            ohlc_cols[col] = name_to_idx[cand]

    # 使用prices字段作为收盘价
    if prices is not None and len(prices) == len(states):
        ohlc_cols["close"] = "prices"
    elif f"{base_period}_close" in name_to_idx:
        ohlc_cols["close"] = name_to_idx[f"{base_period}_close"]
    else:
        logger.warning("⚠️ OHLC数据不完整：缺少收盘价数据")
        return

    if len(ohlc_cols) < 4:
        logger.warning("⚠️ OHLC数据不完整")
        return

    # 时间窗口
    n = len(timestamps)
    start = max(0, n - int(last_n))
    idxs = np.arange(start, n)

    # 转换时间戳为datetime
    try:
        time_axis = pd.to_datetime(timestamps[idxs])
    except Exception as e:
        logger.warning(f"⚠️ 时间转换失败: {e}，使用索引")
        time_axis = np.arange(len(idxs))

    # 提取OHLC数据
    o = states[idxs, ohlc_cols["open"]]
    h = states[idxs, ohlc_cols["high"]]
    l = states[idxs, ohlc_cols["low"]]

    # 处理收盘价：优先用特征列，否则用prices字段
    if ohlc_cols["close"] == "prices":
        c = prices[idxs]
    else:
        c = states[idxs, ohlc_cols["close"]]

    # 获取多周期RSI数据（包含超买超卖标记）
    # 新命名规范：rsi14（带窗口参数）
    all_periods = extract_all_periods_from_features(feature_names)
    target_periods = []
    for p in all_periods[:4]:
        # 优先尝试新命名规范 rsi14
        rsi_col = f"{p}_rsi14" if f"{p}_rsi14" in name_to_idx else f"{p}_rsi"

        # 🔥 优先使用新格式：rsi_event（-1/0/+1三值）
        event_col = f"{p}_rsi_event"
        if event_col in name_to_idx:
            # 使用新格式：单列三值事件
            if rsi_col in name_to_idx:
                target_periods.append(
                    (
                        p,
                        name_to_idx[rsi_col],
                        None,  # 不使用旧格式超买列
                        None,  # 不使用旧格式超卖列
                        name_to_idx[event_col],  # 新格式事件列
                    )
                )
        else:
            # 兼容旧格式：两列分离（overbought/oversold）
            ob_col = f"{p}_rsi_overbought"
            os_col = f"{p}_rsi_oversold"
            if rsi_col in name_to_idx:
                target_periods.append(
                    (
                        p,
                        name_to_idx[rsi_col],
                        name_to_idx.get(ob_col),
                        name_to_idx.get(os_col),
                        None,  # 没有新格式事件列
                    )
                )

    if not target_periods:
        logger.warning("⚠️ 未找到RSI特征")
        return

    # 创建画布
    fig = plt.figure(figsize=(16, 12))
    n_subplots = len(target_periods)
    gs = fig.add_gridspec(n_subplots + 1, 1, height_ratios=[3] + [1] * n_subplots, hspace=0.1)

    ax_main = fig.add_subplot(gs[0])
    axes_sub = [fig.add_subplot(gs[i + 1], sharex=ax_main) for i in range(n_subplots)]

    fig.suptitle(f"Core 04: Price & Multi-Period RSI (last {last_n})", fontsize=16, fontweight="bold")

    # 主图：K线
    draw_candlestick(ax_main, o, h, l, c, time_axis)
    ax_main.set_ylabel("Price", fontweight="bold", fontsize=11)

    # 设置X轴范围和时间格式
    if isinstance(time_axis[0], pd.Timestamp):
        ax_main.set_xlim(time_axis[0], time_axis[-1])
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax_main.set_xlim(-0.5, len(idxs) - 0.5)

    ax_main.set_ylim(l.min() * 0.998, h.max() * 1.002)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelbottom=False)
    ax_main.legend([f"{base_period} Candlestick"], loc="upper left", fontsize=9)

    # 副图：各周期RSI
    colors = ["#C0392B", "#16A085", "#8E44AD", "#D35400"]
    for idx, period_data in enumerate(target_periods):
        # 解包：兼容新旧格式（新格式有5个元素，旧格式有4个元素）
        if len(period_data) == 5:
            p, rsi_idx, ob_idx, os_idx, event_idx = period_data
        else:
            p, rsi_idx, ob_idx, os_idx = period_data
            event_idx = None

        ax = axes_sub[idx]
        rsi_data = states[idxs, rsi_idx]
        ax.plot(time_axis, rsi_data, color=colors[idx % len(colors)], linewidth=1.2, alpha=0.85)

        # 超买超卖阈值线
        ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="OB threshold")
        ax.axhline(y=-0.4, color="green", linestyle="--", alpha=0.5, linewidth=0.8, label="OS threshold")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.6)

        # 🔥 标记超买超卖点：优先使用新格式（rsi_event）
        if event_idx is not None:
            # 新格式：单列三值事件（-1=超卖, 0=中性, +1=超买）
            event_data = states[idxs, event_idx]
            ob_mask = event_data > 0.5  # 超买: +1
            os_mask = event_data < -0.5  # 超卖: -1

            if ob_mask.any():
                if isinstance(time_axis[0], pd.Timestamp):
                    ax.scatter(
                        time_axis[ob_mask],
                        rsi_data[ob_mask],
                        color="red",
                        marker="^",
                        s=20,
                        alpha=0.8,
                        zorder=5,
                        label="Overbought",
                    )
                else:
                    ax.scatter(
                        np.where(ob_mask)[0],
                        rsi_data[ob_mask],
                        color="red",
                        marker="^",
                        s=20,
                        alpha=0.8,
                        zorder=5,
                        label="Overbought",
                    )
            if os_mask.any():
                if isinstance(time_axis[0], pd.Timestamp):
                    ax.scatter(
                        time_axis[os_mask],
                        rsi_data[os_mask],
                        color="green",
                        marker="v",
                        s=20,
                        alpha=0.8,
                        zorder=5,
                        label="Oversold",
                    )
                else:
                    ax.scatter(
                        np.where(os_mask)[0],
                        rsi_data[os_mask],
                        color="green",
                        marker="v",
                        s=20,
                        alpha=0.8,
                        zorder=5,
                        label="Oversold",
                    )
        else:
            # 旧格式：两列分离（兼容）
            if ob_idx is not None:
                ob_mask = states[idxs, ob_idx] > 0.5
                if ob_mask.any():
                    if isinstance(time_axis[0], pd.Timestamp):
                        ax.scatter(
                            time_axis[ob_mask],
                            rsi_data[ob_mask],
                            color="red",
                            marker="^",
                            s=15,
                            alpha=0.7,
                            zorder=5,
                            label="Overbought (old)",
                        )
                    else:
                        ax.scatter(
                            np.where(ob_mask)[0],
                            rsi_data[ob_mask],
                            color="red",
                            marker="^",
                            s=15,
                            alpha=0.7,
                            zorder=5,
                            label="Overbought (old)",
                        )
            if os_idx is not None:
                os_mask = states[idxs, os_idx] > 0.5
                if os_mask.any():
                    if isinstance(time_axis[0], pd.Timestamp):
                        ax.scatter(
                            time_axis[os_mask],
                            rsi_data[os_mask],
                            color="green",
                            marker="v",
                            s=15,
                            alpha=0.7,
                            zorder=5,
                            label="Oversold (old)",
                        )
                    else:
                        ax.scatter(
                            np.where(os_mask)[0],
                            rsi_data[os_mask],
                            color="green",
                            marker="v",
                            s=15,
                            alpha=0.7,
                            zorder=5,
                            label="Oversold (old)",
                        )

        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"{p} RSI", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3)

        if idx < n_subplots - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time", fontweight="bold", fontsize=10)
            if isinstance(time_axis[0], pd.Timestamp):
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=9)

        mean_val = np.mean(rsi_data)
        ax.text(
            0.02,
            0.95,
            f"mean={mean_val:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    path = f"{output_dir}/core_04_price_rsi.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Core 04 saved: {path}")


def plot_core_05_price_bb_width(
    states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300
):
    """核心图5：价格 + 多周期布林带宽度"""
    plot_price_with_indicator(
        states,
        feature_names,
        timestamps,
        prices,
        output_dir,
        last_n,
        indicator_name="bb_width",
        ylabel="BB Width",
        title="Core 05: Price & Multi-Period Bollinger Band Width",
        filename="core_05_price_bb_width.png",
        y_range=(0, 1),
    )


def plot_core_06_correlation_heatmap(states, feature_names, output_dir: str):
    """核心图6：特征相关性热力图"""
    logger.info("📊 Core 06: 特征相关性热力图...")

    sample_size = min(10000, states.shape[0])
    sample_indices = np.random.choice(states.shape[0], sample_size, replace=False)
    sample_states = states[sample_indices, :]

    corr_matrix = np.corrcoef(sample_states.T)

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        fmt=".2f",
        xticklabels=[str(name)[:15] for name in feature_names],
        yticklabels=[str(name)[:15] for name in feature_names],
        cmap="RdYlBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Core 06: Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = f"{output_dir}/core_06_correlation_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Core 06 saved: {path}")


def _parse_period_minutes(period: str) -> int:
    """将周期字符串转换为分钟数，例如 30m -> 30, 2h -> 120, 1d -> 1440"""
    try:
        period = str(period).strip().lower()
        number = int(re.findall(r"\d+", period)[0])
        unit = period[-1]
        if unit == "m":
            return number
        if unit == "h":
            return number * 60
        if unit == "d":
            return number * 1440
    except Exception:
        pass
    return 1


def plot_core_07_price_volume_momentum_combo(
    states, feature_names, timestamps, prices, output_dir: str, last_n: int = 300
):
    """核心图7：价格 + 30m成交量 + 30m动量 + 动量 × 近5个30m成交量均值"""
    logger.info("📊 绘制Core 07: 价格 + 30m成交量 + 30m动量 × 近5个30m成交量均值...")

    target_period = "30m"
    base_period = auto_detect_base_period(feature_names) or "3m"
    base_minutes = _parse_period_minutes(base_period)
    target_minutes = _parse_period_minutes(target_period)

    if target_minutes < base_minutes or target_minutes % max(base_minutes, 1) != 0:
        logger.warning(f"⚠️ 目标周期{target_period}无法由基准周期{base_period}聚合")
        return

    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}

    base_cols = {}
    for col in ["open", "high", "low", "volume"]:
        key = f"{base_period}_{col}"
        if key in name_to_idx:
            base_cols[col] = states[:, name_to_idx[key]]
    if prices is not None and len(prices) == len(states):
        base_cols["close"] = prices
    elif f"{base_period}_close" in name_to_idx:
        base_cols["close"] = states[:, name_to_idx[f"{base_period}_close"]]

    missing_cols = [col for col in ["open", "high", "low", "close", "volume"] if col not in base_cols]
    if missing_cols:
        logger.warning(f"⚠️ 基准周期缺少必要列，无法绘制图表: {missing_cols}")
        return

    try:
        time_index = pd.to_datetime(timestamps)
    except Exception as e:
        logger.warning(f"⚠️ 时间转换失败: {e}")
        time_index = pd.RangeIndex(len(states))

    base_df = pd.DataFrame(base_cols, index=time_index).sort_index()
    rule = f"{target_minutes}min"
    agg_df = (
        base_df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    if agg_df.empty:
        logger.warning("⚠️ 聚合后数据为空，无法绘制")
        return

    momentum_idx = None
    for cand in [f"{target_period}_mom", f"{target_period}_mom_sliding", f"{target_period}_mom20"]:
        if cand in name_to_idx:
            momentum_idx = name_to_idx[cand]
            break
    if momentum_idx is None:
        pattern = re.compile(rf"^{re.escape(target_period)}_.*mom.*$", re.IGNORECASE)
        for n, idx in name_to_idx.items():
            if pattern.match(str(n)):
                momentum_idx = idx
                break
    if momentum_idx is None:
        logger.warning("⚠️ 未找到30m动量特征")
        return

    momentum_series = pd.Series(states[:, momentum_idx], index=time_index).sort_index().resample(rule).last()
    agg_df = agg_df.join(momentum_series.rename("momentum"), how="inner").dropna(subset=["momentum"])
    if agg_df.empty:
        logger.warning("⚠️ 聚合后的动量数据为空")
        return

    volume_window = 5  # 最近5个30m周期（约150分钟）
    agg_df["volume_ma"] = agg_df["volume"].rolling(window=volume_window, min_periods=1).mean()
    agg_df["combo"] = agg_df["momentum"] * agg_df["volume_ma"]

    if last_n is not None and last_n > 0:
        agg_df = agg_df.tail(int(last_n))
    if agg_df.empty:
        logger.warning("⚠️ 选取窗口后数据为空")
        return

    time_axis = agg_df.index
    o = agg_df["open"].to_numpy()
    h = agg_df["high"].to_numpy()
    l = agg_df["low"].to_numpy()
    c = agg_df["close"].to_numpy()
    volume = agg_df["volume"].to_numpy()
    momentum = agg_df["momentum"].to_numpy()
    momentum_volume_combo = agg_df["combo"].to_numpy()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)
    ax_main = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_main)
    ax_mom = fig.add_subplot(gs[2], sharex=ax_main)
    ax_combo = fig.add_subplot(gs[3], sharex=ax_main)

    fig.suptitle(
        f"Core 07: Price & 30m Momentum × VolMA(5×30m) (last {len(agg_df)} bars)",
        fontsize=16,
        fontweight="bold",
    )

    draw_candlestick(ax_main, o, h, l, c, time_axis)
    ax_main.set_ylabel("Price", fontweight="bold", fontsize=11)
    if isinstance(time_axis[0], pd.Timestamp):
        ax_main.set_xlim(time_axis[0], time_axis[-1])
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax_main.set_xlim(-0.5, len(o) - 0.5)
    ax_main.set_ylim(l.min() * 0.998, h.max() * 1.002)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelbottom=False)
    ax_main.legend([f"{target_period} Candlestick"], loc="upper left", fontsize=9)

    ax_vol.plot(time_axis, volume, color="#2980B9", linewidth=1.2, alpha=0.85)
    ax_vol.set_ylabel("30m Vol", fontweight="bold", fontsize=9)
    ax_vol.grid(True, alpha=0.3)
    ax_vol.tick_params(labelbottom=False)

    ax_mom.plot(time_axis, momentum, color="#E67E22", linewidth=1.2, alpha=0.85)
    ax_mom.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_mom.set_ylabel("30m Mom", fontweight="bold", fontsize=9)
    ax_mom.grid(True, alpha=0.3)
    ax_mom.tick_params(labelbottom=False)

    ax_combo.plot(time_axis, momentum_volume_combo, color="#8E44AD", linewidth=1.2, alpha=0.9)
    ax_combo.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_combo.set_ylabel("Mom × VolMA(5)", fontweight="bold", fontsize=9)
    ax_combo.grid(True, alpha=0.3)
    if isinstance(time_axis[0], pd.Timestamp):
        ax_combo.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_combo.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax_combo.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=9)
    else:
        ax_combo.set_xlabel("Index", fontweight="bold", fontsize=10)

    plt.tight_layout()
    path = f"{output_dir}/core_07_price_volume_momentum_combo.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Core 07 saved: {path}")

    # ============================================


# 主函数
# ============================================


def main():
    """主函数"""
    ap = argparse.ArgumentParser(description="Test RL features - 7 core charts")
    ap.add_argument("--npz", type=str, default=None, help="NPZ文件路径（默认自动从配置读取）")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认: data/rl_live/plots）")
    ap.add_argument(
        "--last_n",
        type=int,
        default=DEFAULT_LAST_N_BARS,
        help=f"🎯 总控：绘制最新N根K线（默认{DEFAULT_LAST_N_BARS}）",
    )
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 开始生成核心7图...")
    logger.info(f"🎯 总控参数：绘制最新 {args.last_n} 根K线")
    logger.info("=" * 60)

    if args.npz is None:
        npz_path = get_npz_path_from_config()
    else:
        npz_path = args.npz

    if args.out is None:
        cfg = load_main_config()
        io_cfg = cfg.get("io", {}) or {}
        base_dir = io_cfg.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
        output_dir = os.path.join(base_dir, "rl_live", "plots")
    else:
        output_dir = args.out
    last_n = args.last_n

    os.makedirs(output_dir, exist_ok=True)

    try:
        states, feature_names, timestamps, prices = load_reasoning_data(npz_path)

        logger.info("📊 开始绘制7张核心图（每张：价格 + 多周期同一指标）...")

        # 核心1：价格 + 多周期动量
        plot_core_01_price_momentum(states, feature_names, timestamps, prices, output_dir, last_n)
        # 新增：价格 + 基准周期动量叠加
        plot_core_01b_price_momentum_overlay(states, feature_names, timestamps, prices, output_dir, last_n)

        # 核心2：价格 + 多周期市场状态
        plot_core_02_price_market_states(states, feature_names, timestamps, prices, output_dir, last_n)

        # 核心3：价格 + 多周期ATR
        plot_core_03_price_atr(states, feature_names, timestamps, prices, output_dir, last_n)

        # 核心4：价格 + 多周期RSI（带超买超卖标记）
        plot_core_04_price_rsi(states, feature_names, timestamps, prices, output_dir, last_n)

        # 核心5：价格 + 多周期布林带宽度
        plot_core_05_price_bb_width(states, feature_names, timestamps, prices, output_dir, last_n)

        # 核心6：特征相关性热力图
        plot_core_06_correlation_heatmap(states, feature_names, output_dir)

        # 核心7：30m 动量 × 五日成交量均值
        plot_core_07_price_volume_momentum_combo(
            states, feature_names, timestamps, prices, output_dir, last_n
        )

        logger.info("🎉 所有核心图表生成完成!")
        logger.info(f"📁 输出目录: {output_dir}")

    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        raise


if __name__ == "__main__":
    main()
