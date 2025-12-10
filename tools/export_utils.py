"""
工具：特征导出与校验辅助函数
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from .io_paths import print_latest_timestamp_from_df


def abspath(path: str) -> str:
    return os.path.abspath(path)


def build_feature_groups(names: List[str]) -> List[str]:
    groups: List[str] = []
    for n in names:
        s = str(n).lower()
        if "market_state" in s:
            groups.append("market_state")
        elif "_mom" in s or s.endswith("_mom"):
            groups.append("momentum")
        elif "_atr_pct" in s:
            groups.append("atr_pct")
        elif "_bb_width" in s:
            groups.append("band_width")
        elif "_volume" in s:
            groups.append("volume")
        elif "_rsi" in s:
            groups.append("rsi_continuous")
        elif s.startswith("ret_") and s.endswith("_log"):
            groups.append("return")
        elif s.startswith("time_"):
            groups.append("time_encoding")
        else:
            groups.append("other")
    return groups


def compute_schema_sha(names: List[str]) -> str:
    import hashlib

    return hashlib.sha1("|".join(map(str, names)).encode("utf-8")).hexdigest()


def infer_feature_groups(names: List[str]) -> List[str]:
    return build_feature_groups(names)


def timestamps_to_ms(timestamps: List[Any]) -> np.ndarray:
    out: List[int] = []
    for ts in timestamps:
        try:
            ts_obj = pd.Timestamp(ts)
            if ts_obj.tzinfo is None:
                ts_obj = ts_obj.tz_localize("UTC")
            else:
                ts_obj = ts_obj.tz_convert("UTC")
            out.append(int(ts_obj.value // 10**6))
        except Exception:
            out.append(int(pd.Timestamp(ts).value // 10**6))
    return np.array(out, dtype=np.int64)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("🧹 数据清洗")
    print("=" * 70)

    if "timestamp" in df.columns:
        if pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].ffill().fillna(0.0)

    print(f"✅ 清洗后形状: {df.shape}")
    print_latest_timestamp_from_df(df)
    return df


def export_npz_summary_txt(npz_path: str) -> str:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    base_stem = os.path.splitext(os.path.abspath(npz_path))[0]
    out_txt = base_stem + "_summary.txt"

    with np.load(npz_path, allow_pickle=True) as data:
        ts = data.get("timestamps")
        n_rows = 0
        start_iso = end_iso = ""
        if ts is not None:
            ts_pd = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
            n_rows = len(ts_pd)
            if n_rows > 0:
                start_iso = pd.to_datetime(ts_pd.min(), utc=True).isoformat()
                end_iso = pd.to_datetime(ts_pd.max(), utc=True).isoformat()

        names = data.get("state_names") if "state_names" in data else data.get("feature_names")
        types = data.get("state_types") if "state_types" in data else None
        stored_schema = data.get("schema_sha")
        if names is None:
            names = np.array([], dtype=object)
        names = [str(x) for x in names]
        if types is None:
            types = np.array(["regression"] * len(names), dtype=object)
        else:
            types = [str(x) for x in types]
        schema_sha_calc = compute_schema_sha(names) if len(names) > 0 else ""
        try:
            schema_sha_stored = (
                stored_schema
                if isinstance(stored_schema, str)
                else (stored_schema.item() if stored_schema is not None else "")
            )
        except Exception:
            schema_sha_stored = ""

        cls_names = [n for n, t in zip(names, types) if t == "classification"]
        reg_names = [n for n, t in zip(names, types) if t == "regression"]

    lines: List[str] = [
        f"file: {npz_path}",
        f"generated_at_utc: {pd.Timestamp.utcnow().isoformat()}Z",
        f"records: {n_rows}",
    ]
    if schema_sha_calc:
        lines.append(f"schema_sha_calc: {schema_sha_calc}")
        lines.append(f"schema_sha_stored: {schema_sha_stored}")
    if start_iso and end_iso:
        lines.append(f"time_start_utc: {start_iso}")
        lines.append(f"time_end_utc: {end_iso}")
    lines.append(f"names_count: {len(names)}")

    if cls_names:
        lines.append("")
        lines.append(f"classification ({len(cls_names)}):")
        lines.extend(f"  - {n}" for n in cls_names)

    if reg_names:
        lines.append("")
        lines.append(f"regression ({len(reg_names)}):")
        lines.extend(f"  - {n}" for n in reg_names)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_txt


def _ratio_out_of_range(values: np.ndarray, lo: float, hi: float) -> float:
    if values.size == 0:
        return 0.0
    total = values.size
    count = int(((values < lo) | (values > hi)).sum())
    return count / max(total, 1)


def health_check_features(npz_path: str) -> None:
    print("\n" + "=" * 70)
    print("🩺 特征健康检查")
    print("=" * 70)

    with np.load(npz_path, allow_pickle=True) as data:
        observations = data.get("observations")
        feature_names = [str(x) for x in data.get("feature_names", [])]
        timestamps = data.get("timestamps")

    if observations is None or observations.size == 0:
        print("⚠️ 无可用特征，跳过健康检查")
        return

    obs = observations.astype(np.float64)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    ts_min = ts_max = None
    if timestamps is not None and len(timestamps):
        ts_pd = pd.to_datetime(timestamps, unit="ms", utc=True, errors="coerce")
        ts_min = pd.to_datetime(ts_pd.min(), utc=True).isoformat()
        ts_max = pd.to_datetime(ts_pd.max(), utc=True).isoformat()

    print(f"📊 基础信息:")
    print(f"   形状: {obs.shape}")
    if ts_min and ts_max:
        print(f"   时间: {ts_min} ~ {ts_max}")

    def report_block(title: str, idx: List[int], lo: float, hi: float) -> None:
        if not idx:
            return
        vals = obs[:, idx]
        ratio = _ratio_out_of_range(vals, lo, hi)
        mean_val = float(np.mean(vals))
        print(f"\n📌 {title}: 列={len(idx)}")
        print(f"   均值={mean_val:.4f}, 越界比例={ratio:.4f}")

    idx_market_state = [i for i, n in enumerate(feature_names) if "market_state" in n]
    idx_momentum = [i for i, n in enumerate(feature_names) if n.endswith("_mom")]
    idx_band = [i for i, n in enumerate(feature_names) if "_bb_width" in n]
    idx_volume = [i for i, n in enumerate(feature_names) if "_volume" in n]
    idx_rsi = [i for i, n in enumerate(feature_names) if n.endswith("_rsi") and "over" not in n]
    idx_price = [
        i for i, n in enumerate(feature_names) if any(k in n for k in ["_open", "_high", "_low", "_close"])
    ]

    report_block("market_state", idx_market_state, -1, 1)
    report_block("momentum", idx_momentum, -1, 1)
    report_block("band_width", idx_band, 0, 1)
    report_block("volume", idx_volume, 0, 1)
    report_block("rsi", idx_rsi, -1, 1)
    report_block("price (OHLC)", idx_price, 0, float("inf"))


def export_to_npz(
    result: Dict[str, Any], features_path: str, labels_path: str, symbol: str, cfg: Dict[str, Any]
) -> None:
    print("\n" + "=" * 70)
    print("💾 契约导出")
    print("=" * 70)

    observations = result["observations"]
    observation_names = result["observation_names"]
    states = result["states"]
    state_names = result["state_names"]
    state_types = result["state_types"]
    num_classes = result["num_classes"]
    timestamps = result["timestamps"]
    prices = result["prices"]

    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        ts_pd = pd.DatetimeIndex(timestamps)
        ts_ms = (ts_pd.astype("int64") // 10**6).astype(np.int64)
    else:
        ts_ms = np.arange(len(timestamps), dtype=np.int64)

    feature_groups = build_feature_groups(observation_names)
    schema_sha = compute_schema_sha(observation_names)

    rl_build_cfg = cfg.get("rl_build", {})
    metadata = {
        "generated_at_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "symbol": symbol,
        "base_period": cfg.get("timeframes", {}).get("base_download", "1m"),
        "source_mode": rl_build_cfg.get("source_mode", "fixed"),
        "periods": result.get("periods", []),
        "notes": "统一流水线生成",
    }
    metadata_str = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)

    os.makedirs(os.path.dirname(abspath(features_path)), exist_ok=True)
    tmp_features = abspath(features_path)[:-4] + ".tmp"

    try:
        np.savez_compressed(
            tmp_features,
            version=np.array("rl_features_v1"),
            observations=observations,
            feature_names=np.array(observation_names, dtype=object),
            feature_groups=np.array(feature_groups, dtype=object),
            timestamps=ts_ms,
            prices=prices.astype(np.float64),
            schema_sha=np.array(schema_sha),
            metadata=np.array(metadata_str),
        )
        tmp_npz = tmp_features + ".npz"
        if not os.path.exists(tmp_npz):
            raise RuntimeError(f"临时文件未生成: {tmp_npz}")
        os.replace(tmp_npz, abspath(features_path))
        print(f"✅ RL_FEATURES: {features_path}")
    finally:
        for cleanup in [tmp_features, tmp_features + ".npz"]:
            if os.path.exists(cleanup):
                try:
                    os.remove(cleanup)
                except Exception:
                    pass

    tmp_labels = abspath(labels_path)[:-4] + ".tmp"
    state_mappings = {
        name: {"type": state_types[i], "num_classes": int(num_classes[i])}
        for i, name in enumerate(state_names)
    }
    state_meta = {
        "generated_at_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "symbol": symbol,
        "state_mappings": state_mappings,
        "notes": "RL_LABELS标准包，仅用于分析/可视化",
    }
    state_meta_str = yaml.safe_dump(state_meta, sort_keys=False, allow_unicode=True)

    try:
        np.savez_compressed(
            tmp_labels,
            version=np.array("rl_labels_v1"),
            states=states,
            state_names=np.array(state_names, dtype=object),
            state_types=np.array(state_types, dtype=object),
            num_classes=num_classes,
            timestamps=ts_ms,
            metadata=np.array(state_meta_str),
        )
        tmp_npz = tmp_labels + ".npz"
        if not os.path.exists(tmp_npz):
            raise RuntimeError(f"临时文件未生成: {tmp_npz}")
        os.replace(tmp_npz, abspath(labels_path))
        print(f"✅ RL_LABELS: {labels_path}")
    finally:
        for cleanup in [tmp_labels, tmp_labels + ".npz"]:
            if os.path.exists(cleanup):
                try:
                    os.remove(cleanup)
                except Exception:
                    pass
