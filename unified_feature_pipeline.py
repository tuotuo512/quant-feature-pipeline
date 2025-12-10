"""
统一特征流水线入口

职责：
- 调用 Step2/Step3/Step4 的核心执行函数（execute_stepX），避免脚本化子进程
- 统一执行 Step5，产出 RL_FEATURES/RL_LABELS
- 为实盘/离线/测试提供统一调用接口
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yaml

from features_engineering.congfigs.config_loader import ConfigLoader
from features_engineering.step2_resample import execute_step2
from features_engineering.step3_generate_indicators import execute_step3
from features_engineering.step4_merge_features import execute_step4
from features_engineering.step5_featueres_unified import UnifiedFeatureEngine
from features_engineering.tools.io_paths import IOManager
from features_engineering.tools.export_utils import (
    abspath,
    build_feature_groups,
    clean_data,
    export_npz_summary_txt,
    export_to_npz,
    health_check_features,
    infer_feature_groups,
)


def _resolve_symbol_from_live_cfg(
    live_cfg: Optional[Dict[str, Any]],
    main_cfg: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    🔥 统一从 live_overrides.yaml 的 exchanges 配置读取 symbol
    
    优先级：
    1. live_cfg.exchanges.{okx|bitget}.symbol（启用的交易所）
    2. main_cfg.symbol.trading_pair_std（回退）
    
    Returns:
        (symbol_std, symbol_exchange)
        例如：("BTC_USDT", "BTC/USDT:USDT")
    """
    # 默认值（回退）
    symbol_cfg = (main_cfg or {}).get("symbol", {}) or {}
    default_std = str(symbol_cfg.get("trading_pair_std", "ETH_USDT"))
    default_exchange = str(symbol_cfg.get("trading_pair_exchange", "ETH/USDT"))
    
    if not live_cfg:
        return default_std, default_exchange
    
    # 🔥 从 exchanges 配置读取启用的交易所
    exchanges_cfg = live_cfg.get("exchanges", {}) or {}
    
    active_symbol = None
    for ex_name in ["okx", "bitget"]:
        ex_cfg = exchanges_cfg.get(ex_name, {}) or {}
        if ex_cfg.get("enabled", False):
            active_symbol = ex_cfg.get("symbol")
            if active_symbol:
                break
    
    if not active_symbol:
        return default_std, default_exchange
    
    # 解析 symbol：例如 "BTC/USDT:USDT" → ("BTC_USDT", "BTC/USDT:USDT")
    symbol_exchange = active_symbol
    # 去掉 :USDT 后缀，然后替换 / 为 _
    symbol_std = active_symbol.split(":")[0].replace("/", "_")
    
    return symbol_std, symbol_exchange
from live_trading.features.schema_aligned_builder import build_obs_from_npz_schema_batch
from live_trading.monitoring.core.probes import write_rl_features_npz_summary


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


@dataclass
class LiveCachePrepResult:
    cache_path: str
    source_path: str
    copied_rows: int
    added_rows: int
    source_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
    updated_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]


@dataclass
class PipelineResult:
    """统一流水线的输出摘要"""

    mode: str
    features_path: str
    labels_path: str
    merged_path: str
    base_timeframe: str
    records: int
    config_dir: str
    extra: Dict[str, Any]


def load_live_overrides(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = path or os.path.join(PROJECT_ROOT, "live_trading", "config", "live_overrides.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 校验 live_overrides 与主配置一致性
    try:
        loader = ConfigLoader()
        loader.validate_live_alignment(cfg)
    except Exception as exc:
        raise ValueError(f"live_overrides 配置与主配置不一致: {exc}") from exc

    return cfg


def prepare_live_cache(
    *,
    save_n: Optional[int] = None,
    main_cfg: Optional[Dict[str, Any]] = None,
    live_cfg: Optional[Dict[str, Any]] = None,
    source_csv: Optional[str] = None,
    cache_path: Optional[str] = None,
    update_to_latest: bool = True,
    logger: Callable[..., None] = print,
) -> LiveCachePrepResult:
    """
    复制训练侧 1m 数据到实盘 cache，并按需调用 KlineAppender 更新到最新。
    供预检和实盘流程复用。
    """

    log = logger or (lambda *args, **kwargs: None)

    loader = None
    if main_cfg is None:
        loader = ConfigLoader()
        main_cfg = loader.load_main_config()
    io_mgr = IOManager(main_cfg)

    if live_cfg is None:
        live_cfg = load_live_overrides()

    tf_cfg = main_cfg.get("timeframes", {}) or {}
    base_download = str(tf_cfg.get("base_download", "1m"))
    
    # 🔥 优先从 live_overrides.yaml 的 exchanges 配置读取 symbol
    # 回退到 main_config.yaml 的 symbol.trading_pair_std
    symbol_std, symbol_exchange = _resolve_symbol_from_live_cfg(live_cfg, main_cfg)
    symbol_cfg = main_cfg.get("symbol", {}) or {}
    market_type = str(symbol_cfg.get("market_type", "swap")).lower()

    preheat_cfg = live_cfg.get("preheat", {}) or {}
    if save_n is None:
        save_n = int(preheat_cfg.get("save_n", 300))
    min_lookback = int(preheat_cfg.get("min_lookback_bars", max(save_n * 10, 100000)))
    copy_rows = max(save_n * 10, min_lookback)

    # 🔥 源文件路径：使用动态 symbol（来自 live_overrides.yaml）
    if source_csv is None:
        # 构建源文件路径：data/rl_live/data_downloads/{symbol_std}_SWAP_1m.csv
        market_suffix = "SWAP" if market_type == "swap" else ""
        source_fname = f"{symbol_std}_{market_suffix}_{base_download}.csv" if market_suffix else f"{symbol_std}_{base_download}.csv"
        source_csv = os.path.join(PROJECT_ROOT, "data", "rl_live", "data_downloads", source_fname)
    
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"训练侧基准 CSV 不存在: {source_csv}")

    if cache_path is None:
        cache_fname = f"{symbol_std}_{base_download}_cache.csv"
        cache_path = os.path.join(PROJECT_ROOT, "data_trading", "cache", cache_fname)

    log("\n📦 准备实盘 cache（统一 Hook）")
    log(f"   源: {source_csv}")
    log(f"   目标: {cache_path}")
    log(f"   save_n={save_n}, min_lookback={min_lookback}, 实际复制={copy_rows}")

    df_source = pd.read_csv(source_csv)
    if df_source.empty:
        raise ValueError("训练侧基准 CSV 为空，无法复制")
    df_copy = df_source.tail(copy_rows).copy()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_copy.to_csv(cache_path, index=False)

    source_ts = None
    if "timestamp" in df_copy.columns:
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], errors="coerce")
        source_ts = (df_copy["timestamp"].min(), df_copy["timestamp"].max())
        log(f"   📅 复制时间范围: {source_ts[0]} ~ {source_ts[1]}")

    added_rows = 0
    updated_range = (None, None)

    if update_to_latest:
        init_cfg = live_cfg.get("initialization", {}) or {}
        max_backfill_hours = int(init_cfg.get("max_backfill_hours", 168))
        try:
            from live_trading.data_feed.kline_appender import KlineAppender

            exchange_cfg = live_cfg.get("exchange", {}) or {}
            exchange_id = str(exchange_cfg.get("name", "okx"))
            proxy = exchange_cfg.get("proxy")

            instrument = symbol_exchange
            if market_type == "swap" and ":" not in instrument:
                instrument = f"{instrument}:USDT"

            appender = KlineAppender(cache_path=cache_path, exchange_id=exchange_id, proxy=proxy)
            last_ts_before = appender.get_last_timestamp()
            added_rows = appender.backfill_missing_data(
                symbol=instrument,
                timeframe=base_download,
                max_gap_hours=max_backfill_hours,
            )
            last_ts_after = appender.get_last_timestamp()
            updated_range = (last_ts_before, last_ts_after)
            if added_rows > 0:
                log(f"   ✅ 已追加 {added_rows} 条 K 线")
            else:
                log("   ℹ️  数据已是最新，无需追加")
        except Exception as e:
            log(f"   ⚠️ 更新到最新失败（已忽略）: {e}")

    return LiveCachePrepResult(
        cache_path=cache_path,
        source_path=source_csv,
        copied_rows=len(df_copy),
        added_rows=added_rows,
        source_range=(source_ts[0] if source_ts else None, source_ts[1] if source_ts else None),
        updated_range=updated_range,
    )


def _period_to_pandas_freq(period: str) -> str:
    s = str(period).strip().lower()
    num = "".join(ch for ch in s if ch.isdigit())
    unit = "".join(ch for ch in s if ch.isalpha())
    if not num or not unit:
        raise ValueError(f"无效的周期: {period}")
    if unit == "m":
        return f"{int(num)}min"
    if unit == "h":
        return f"{int(num)}h"
    if unit == "d":
        return f"{int(num)}d"
    if unit == "w":
        return f"{int(num)}w"
    raise ValueError(f"不支持的周期单位: {period}")


def _period_to_timedelta(period: str) -> pd.Timedelta:
    freq = _period_to_pandas_freq(period)
    return pd.to_timedelta(freq)


def _load_npz_header(npz_path: str) -> Tuple[str, str]:
    with np.load(npz_path, allow_pickle=True) as data:
        version_raw = data.get("version")
        metadata_raw = data.get("metadata")

    def _to_str(val: Any) -> str:
        if val is None:
            return ""
        if hasattr(val, "tolist"):
            converted = val.tolist()
            if isinstance(converted, (list, tuple)):
                return "\n".join(str(x) for x in converted)
            return str(converted)
        return str(val)

    return _to_str(version_raw) or "rl_features_v1", _to_str(metadata_raw)


def _compute_online_target_timestamps(cache_csv: str, length: int, base_period: str) -> List[pd.Timestamp]:
    if length <= 0:
        raise ValueError("microbatch 长度必须大于0")
    df = pd.read_csv(cache_csv, usecols=["timestamp"])
    if df.empty:
        raise ValueError("cache CSV 为空，无法生成特征")
    ts_series = pd.to_datetime(df["timestamp"], errors="coerce")
    ts_series = ts_series.dropna()
    if ts_series.empty:
        raise ValueError("cache CSV 无有效时间戳")
    freq = _period_to_pandas_freq(base_period)
    base_delta = _period_to_timedelta(base_period)
    last_aligned = ts_series.iloc[-1].floor(freq)
    start = last_aligned - base_delta * (length - 1)
    earliest = ts_series.iloc[0].floor(freq)
    if start < earliest:
        start = earliest
        periods = int(((last_aligned - start) / base_delta)) + 1
        periods = max(1, periods)
    else:
        periods = length
    rng = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    if len(rng) == 0:
        raise ValueError("无法构造目标时间戳序列")
    return [ts.to_pydatetime() for ts in rng]


def _run_online_pipeline(
    *,
    main_cfg: Dict[str, Any],
    loader: ConfigLoader,
    start: Optional[str],
    end: Optional[str],
    sample_ratio: Optional[float],
    output_dir: Optional[str],
    verbose: bool,
    reuse_cache_path: Optional[str] = None,
    skip_cache_prepare: bool = False,
    target_npz_path: Optional[str] = None,
    microbatch_length: Optional[int] = None,
) -> PipelineResult:
    log = print if verbose else (lambda *args, **kwargs: None)

    live_cfg = load_live_overrides()
    online_cfg = main_cfg.get("online", {}) or {}
    micro_cfg = online_cfg.get("microbatch", {}) or {}
    micro_len = max(1, int(micro_cfg.get("length", 300)))
    if microbatch_length is not None:
        try:
            micro_len = max(1, int(microbatch_length))
        except Exception:
            pass
    io_mgr = IOManager(main_cfg)
    base_axis_tf = io_mgr.get_min_resample_timeframe()
    cache_path_override = os.path.abspath(reuse_cache_path) if reuse_cache_path else None

    if skip_cache_prepare:
        if cache_path_override is None:
            raise ValueError("skip_cache_prepare=True 时必须提供 reuse_cache_path")
        if not os.path.exists(cache_path_override):
            raise FileNotFoundError(f"指定的 cache 文件不存在: {cache_path_override}")
        source_csv = io_mgr.path_for("download", timeframe=base_axis_tf)
        prep_res = LiveCachePrepResult(
            cache_path=cache_path_override,
            source_path=source_csv,
            copied_rows=0,
            added_rows=0,
            source_range=(None, None),
            updated_range=(None, None),
        )
        log("\n📦 复用现有实盘 cache（跳过训练侧复制）")
        log(f"   cache: {prep_res.cache_path}")
    else:
        prep_res = prepare_live_cache(
            save_n=micro_len,
            main_cfg=main_cfg,
            live_cfg=live_cfg,
            source_csv=None,
            cache_path=cache_path_override,
            update_to_latest=True,
            logger=log,
        )

    if start or end:
        freq = _period_to_pandas_freq(base_axis_tf)
        start_ts = pd.Timestamp(start).tz_localize("UTC") if start else None
        end_ts = pd.Timestamp(end).tz_localize("UTC") if end else pd.Timestamp.utcnow().tz_localize("UTC")
        if start_ts is None:
            delta = _period_to_timedelta(base_axis_tf)
            start_ts = end_ts - delta * (micro_len - 1)
        rng = pd.date_range(start=start_ts, end=end_ts, freq=freq, tz="UTC")
        if len(rng) == 0:
            raise ValueError("提供的时间范围不足以构造目标时间戳")
        target_timestamps = [ts.to_pydatetime() for ts in rng]
    else:
        target_timestamps = _compute_online_target_timestamps(prep_res.cache_path, micro_len, base_axis_tf)

    model_cfg = live_cfg.get("model", {}) or {}
    session_dir = str(model_cfg.get("active_session_dir", "")).strip()
    smoothing_cfg: Optional[Dict[str, Any]] = None
    if session_dir:
        train_cfg_path = os.path.join(session_dir, "models", "used_main_config.yaml")
        if os.path.exists(train_cfg_path):
            try:
                with open(train_cfg_path, "r", encoding="utf-8") as f:
                    train_cfg = yaml.safe_load(f) or {}
                smoothing_candidate = (
                    ((train_cfg.get("data", {}) or {}).get("processing", {}) or {})
                    .get("features", {})
                    .get("smoothing", {})
                )
                if smoothing_candidate and bool(smoothing_candidate.get("enable")):
                    smoothing_cfg = smoothing_candidate
            except Exception as e:
                log(f"   ⚠️ 无法加载训练特征平滑配置: {e}")

    fc_cfg = live_cfg.get("features_contract", {}) or {}
    source_cfg = fc_cfg.get("source", {}) or {}
    if target_npz_path:
        training_npz = os.path.abspath(target_npz_path)
    else:
        path_pattern = source_cfg.get("path_pattern")
        if not path_pattern:
            raise ValueError("live_overrides.features_contract.source.path_pattern 未配置")
        training_npz = path_pattern.format(base_period=base_axis_tf)
    if not os.path.exists(training_npz):
        raise FileNotFoundError(f"训练契约 NPZ 不存在: {training_npz}")

    version_str, metadata_str = _load_npz_header(training_npz)

    log("\n" + "=" * 100)
    log("🏭 Unified Feature Pipeline → Online 批量生成")
    log("=" * 100)

    results = build_obs_from_npz_schema_batch(
        training_npz,
        prep_res.cache_path,
        base_axis_tf,
        target_timestamps=target_timestamps,
        smoothing_config=smoothing_cfg,
        verbose=verbose,
    )
    if not results:
        raise RuntimeError("在线批量生成结果为空")

    feature_names = [str(x) for x in results[0][1]]
    observations = np.stack([r[0] for r in results]).astype(np.float32)
    timestamps = [r[2] for r in results]
    prices = np.array([r[3] for r in results], dtype=np.float64)
    feature_groups = infer_feature_groups(feature_names)
    schema_sha = compute_schema_sha(feature_names)

    ts_ms = timestamps_to_ms(timestamps)

    meta_dict: Dict[str, Any] = {}
    if metadata_str:
        try:
            meta_dict = yaml.safe_load(metadata_str) or {}
        except Exception:
            meta_dict = {"training_metadata": metadata_str}
    meta_dict["generated_at_utc"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    meta_dict["mode"] = "online"
    meta_dict["microbatch_length"] = len(observations)
    meta_dict["cache_path"] = prep_res.cache_path
    meta_dict["source_range"] = {
        "start": prep_res.source_range[0].isoformat() if prep_res.source_range[0] else None,
        "end": prep_res.source_range[1].isoformat() if prep_res.source_range[1] else None,
    }
    if prep_res.updated_range[1]:
        meta_dict["updated_to"] = prep_res.updated_range[1].isoformat()
    metadata_dump = yaml.safe_dump(meta_dict, sort_keys=False, allow_unicode=True)

    symbol_std = (main_cfg.get("symbol", {}) or {}).get("trading_pair_std", "ETH_USDT")
    out_dir = output_dir or os.path.join(PROJECT_ROOT, "data_trading", "online_features")
    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(out_dir, f"{symbol_std}_{base_axis_tf}_online_rl_features.npz")

    np.savez_compressed(
        features_path,
        version=np.array(version_str),
        observations=observations,
        feature_names=np.array(feature_names, dtype=object),
        feature_groups=np.array(feature_groups, dtype=object),
        timestamps=ts_ms,
        prices=prices,
        schema_sha=np.array(schema_sha),
        metadata=np.array(metadata_dump),
    )
    log(f"   💾 Online features 保存至: {features_path}")

    try:
        summary_path = write_rl_features_npz_summary(
            features_path,
            base_period=base_axis_tf,
            out_dir=out_dir,
        )
        if summary_path:
            log(f"   📝 Summary: {summary_path}")
    except Exception as e:
        log(f"   ⚠️ Summary 生成失败: {e}")

    try:
        health_check_features(features_path)
    except Exception as e:
        log(f"   ⚠️ 健康检查失败: {e}")

    ts_iso: List[str] = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            ts_iso.append(ts.replace(tzinfo=timezone.utc).isoformat())
        elif hasattr(ts, "isoformat"):
            ts_iso.append(ts.isoformat())
        else:
            try:
                ts_iso.append(str(ts))
            except Exception:
                ts_iso.append("")

    extra = {
        "cache": {
            "path": prep_res.cache_path,
            "copied_rows": prep_res.copied_rows,
            "added_rows": prep_res.added_rows,
        },
        "microbatch": {
            "length": len(observations),
            "base_timeframe": base_axis_tf,
            "feature_names": feature_names,
            "observations": observations.tolist(),
            "timestamps": ts_iso,
            "prices": prices.tolist(),
            "schema_sha": schema_sha,
            "version": version_str,
            "metadata": metadata_dump,
            "training_npz": training_npz,
        },
    }

    return PipelineResult(
        mode="online",
        features_path=features_path,
        labels_path="",
        merged_path="",
        base_timeframe=base_axis_tf,
        records=len(observations),
        config_dir=str(loader.config_dir),
        extra=extra,
    )


def generate_rl_features(
    *,
    mode: str = "offline",
    start: Optional[str] = None,
    end: Optional[str] = None,
    sample_ratio: Optional[float] = None,
    config_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    force: bool = True,
    verbose: bool = True,
    reuse_merged: bool = False,
    merged_path: Optional[str] = None,
    reuse_cache_path: Optional[str] = None,
    skip_cache_prepare: bool = False,
    target_npz_path: Optional[str] = None,
    microbatch_length: Optional[int] = None,
) -> PipelineResult:
    """
    统一生成 RL 特征（offline / online）
    """

    if mode not in ("offline", "online"):
        raise NotImplementedError(f"暂未实现模式: {mode}")

    loader = ConfigLoader(config_dir=config_dir)
    main_cfg = loader.load_main_config()
    log = print if verbose else (lambda *args, **kwargs: None)

    if mode == "online":
        return _run_online_pipeline(
            main_cfg=main_cfg,
            loader=loader,
            start=start,
            end=end,
            sample_ratio=sample_ratio,
            output_dir=output_dir,
            verbose=verbose,
            reuse_cache_path=reuse_cache_path,
            skip_cache_prepare=skip_cache_prepare,
            target_npz_path=target_npz_path,
            microbatch_length=microbatch_length,
        )

    step5_cfg = loader.load_step5_config()

    io_mgr = IOManager(main_cfg)
    symbol_std = (main_cfg.get("symbol", {}) or {}).get("trading_pair_std", "ETH_USDT")

    if reuse_merged:
        base_axis_tf = io_mgr.get_min_resample_timeframe()
        merged_path_expected = merged_path or io_mgr.path_for("merged", timeframe=base_axis_tf)
        merged_path_actual = io_mgr.resolve_existing(merged_path_expected) or merged_path_expected
        if not os.path.exists(merged_path_actual):
            raise FileNotFoundError(f"未找到 merged 文件: {merged_path_actual}")
        step2_info = {"status": "skipped"}
        step3_info = {"status": "skipped"}
        step4_info = {"status": "skipped"}
    else:
        step2_cfg = loader.load_step2_config()
        step3_cfg = loader.load_step3_config()
        step4_cfg = loader.load_step4_config()

        log("\n" + "=" * 100)
        log("🏭 Unified Feature Pipeline → Step2")
        log("=" * 100)
        step2_info = execute_step2(
            step2_cfg,
            start=start,
            end=end,
            output_format=None,
            verbose=verbose,
        )

        log("\n" + "=" * 100)
        log("🏭 Unified Feature Pipeline → Step3")
        log("=" * 100)
        step3_info = execute_step3(
            step3_cfg,
            start=start,
            end=end,
            output_format=None,
            verbose=verbose,
        )

        log("\n" + "=" * 100)
        log("🏭 Unified Feature Pipeline → Step4")
        log("=" * 100)
        step4_info = execute_step4(
            step4_cfg,
            start=start,
            end=end,
            output_format=None,
            verbose=verbose,
        )

        base_axis_tf = step4_info.get("base_axis") or io_mgr.get_min_resample_timeframe()
        merged_path_expected = io_mgr.path_for("merged", timeframe=base_axis_tf)
        merged_path_actual = io_mgr.resolve_existing(merged_path_expected) or merged_path_expected
        if not os.path.exists(merged_path_actual):
            raise FileNotFoundError(f"未找到 Step4 输出: {merged_path_actual}")

    # =========================
    # Step5：统一导出 RL 特征
    # =========================
    log("\n" + "=" * 100)
    log("🏭 Unified Feature Pipeline → Step5 (RL 合约导出)")
    log("=" * 100)

    merged_path_actual = io_mgr.resolve_existing(merged_path_expected) or merged_path_expected
    if not os.path.exists(merged_path_actual):
        raise FileNotFoundError(f"未找到 Step4 输出: {merged_path_actual}")

    log(f"📦 merged 输入: {merged_path_actual}")
    if merged_path_actual.lower().endswith(".parquet"):
        df_merged = pd.read_parquet(merged_path_actual)
    else:
        df_merged = pd.read_csv(merged_path_actual)

    if sample_ratio is not None and 0 < sample_ratio < 1:
        sample_rows = max(100, int(len(df_merged) * sample_ratio))
        df_merged = df_merged.tail(sample_rows)
        log(f"🧪 采样模式: ratio={sample_ratio:.4f} → {len(df_merged):,} 行")

    df_clean = clean_data(df_merged)

    # Step5 需要知道 base_download
    step5_cfg.setdefault("timeframes", {})["base_download"] = base_axis_tf

    engine = UnifiedFeatureEngine(step5_cfg, base_axis_tf)
    step5_result = engine.process(df_clean)

    features_path_default = io_mgr.path_for("rl_features", timeframe=base_axis_tf)
    labels_path_default = io_mgr.path_for("rl_labels", timeframe=base_axis_tf)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        features_path = os.path.join(output_dir, os.path.basename(features_path_default))
        labels_path = os.path.join(output_dir, os.path.basename(labels_path_default))
    else:
        features_path = features_path_default
        labels_path = labels_path_default

    export_to_npz(step5_result, features_path, labels_path, symbol_std, main_cfg)
    try:
        export_npz_summary_txt(features_path)
        export_npz_summary_txt(labels_path)
    except Exception as e:
        log(f"⚠️ 摘要导出失败: {e}")

    try:
        health_check_features(features_path)
    except Exception as e:
        log(f"⚠️ 健康检查失败: {e}")

    return PipelineResult(
        mode=mode,
        features_path=features_path,
        labels_path=labels_path,
        merged_path=merged_path_actual,
        base_timeframe=base_axis_tf,
        records=len(step5_result.get("timestamps", [])),
        config_dir=str(loader.config_dir),
        extra={
            "step2": step2_info,
            "step3": step3_info,
            "step4": step4_info,
        },
    )
