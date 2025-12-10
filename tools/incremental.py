from __future__ import annotations

import pandas as pd


def compute_increment_range(
    last_ts: pd.Timestamp | None, new_max_ts: pd.Timestamp, warmup_delta: pd.Timedelta | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    计算增量区间 [warmup_start, new_max_ts]
    - last_ts: 现有产物最后时间
    - new_max_ts: 基础数据最新时间
    - warmup_delta: 需要回溯的时间跨度
    """
    if last_ts is None:
        # 首次构建：全量
        return (
            (
                pd.Timestamp.min.tz_localize(None)
                if hasattr(pd.Timestamp.min, "tz_localize")
                else pd.Timestamp.min
            ),
            new_max_ts,
        )
    start = last_ts
    if warmup_delta is not None:
        start = start - warmup_delta
    return (start, new_max_ts)


def safe_concat_dedup(old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return new_df
    df = pd.concat([old_df, new_df], axis=0)
    # 去重+排序（按时间索引或首列）
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = df.drop_duplicates().reset_index(drop=True)
    return df
