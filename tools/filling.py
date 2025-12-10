from __future__ import annotations

import pandas as pd

from .time_index import timeframe_to_pandas_freq


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.set_index(df.columns[0])
    df.index.name = "timestamp"
    return df


def fill_base_ohlcv_grid(df: pd.DataFrame, base_tf: str) -> pd.DataFrame:
    df = _ensure_dt_index(df)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        return df
    freq = timeframe_to_pandas_freq(base_tf)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    if "close" in df.columns:
        df["close"] = df["close"].ffill()
    for col in ["open", "high", "low"]:
        if col in df.columns:
            if "close" in df.columns:
                df[col] = df[col].fillna(df["close"]).ffill()
            else:
                df[col] = df[col].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    return df


def fill_kline_grid(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    df = _ensure_dt_index(df)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        return df
    freq = timeframe_to_pandas_freq(tf)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    if "close" in df.columns:
        df["close"] = df["close"].ffill()
    for col in ["open", "high", "low"]:
        if col in df.columns:
            if "close" in df.columns:
                df[col] = df[col].fillna(df["close"]).ffill()
            else:
                df[col] = df[col].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    return df


def fill_nan(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    m = str(method).lower().strip()
    if m == "ffill":
        return df.ffill()
    if m == "bfill":
        return df.bfill()
    if m in ("ffill_bfill", "both"):
        return df.ffill().bfill()
    return df
