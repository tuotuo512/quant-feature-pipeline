#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列名处理工具（规范版）
 - 去除 _fixed/_roll 后缀
 - 模糊/正则匹配列名
 - 批量清理列名

与旧版 column_utils.py 功能等价，命名更规范，位于 tools 命名空间。
"""

from __future__ import annotations

import pandas as pd
import re


def normalize_column_name(col: str, remove_suffix: bool = True) -> str:
    if not remove_suffix:
        return col
    if col.endswith("_fixed"):
        return col[:-6]
    if col.endswith("_roll"):
        return col[:-5]
    return col


def find_column_fuzzy(df: pd.DataFrame, pattern: str, period: str | None = None) -> str | None:
    if period:
        candidates = [
            f"{period}_{pattern}",
            f"{period}_{pattern}_fixed",
            f"{period}_{pattern}_roll",
        ]
    else:
        candidates = [pattern, f"{pattern}_fixed", f"{pattern}_roll"]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        col_normalized = normalize_column_name(col)
        pattern_full = f"{period}_{pattern}" if period else pattern
        if pattern_full in col_normalized or col_normalized.startswith(pattern_full):
            return col
    return None


def find_columns_by_pattern(df: pd.DataFrame, pattern: str, period: str | None = None) -> list[str]:
    pattern_full = f"{period}_{pattern}" if period else pattern
    regex = re.compile(pattern_full)
    matched: list[str] = []
    for col in df.columns:
        col_normalized = normalize_column_name(col)
        if regex.search(col_normalized):
            matched.append(col)
    return matched


def clean_dataframe_columns(df: pd.DataFrame, remove_suffix: bool = True) -> pd.DataFrame:
    if not remove_suffix:
        return df
    df_clean = df.copy()
    df_clean.columns = [normalize_column_name(c, remove_suffix=True) for c in df_clean.columns]
    return df_clean
