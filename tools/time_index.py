from __future__ import annotations


def timeframe_to_pandas_freq(tf: str) -> str:
    t = str(tf).strip().lower()
    if t.endswith("m"):
        return f"{int(t[:-1])}min"
    if t.endswith("h"):
        return f"{int(t[:-1])}h"
    if t.endswith("d"):
        return f"{int(t[:-1])}d"
    if t.endswith("w"):
        return f"{int(t[:-1])}w"
    return "1min"


def timeframe_to_minutes(tf: str) -> int:
    t = str(tf).strip().lower()
    if t.endswith("m"):
        return int(t[:-1])
    if t.endswith("h"):
        return int(t[:-1]) * 60
    if t.endswith("d"):
        return int(t[:-1]) * 1440
    if t.endswith("w"):
        return int(t[:-1]) * 10080
    return 1
