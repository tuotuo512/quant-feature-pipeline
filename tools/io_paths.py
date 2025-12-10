from __future__ import annotations

import os
import re
from typing import Optional, Dict, Any, List
import pandas as pd


def read_df_auto(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    base, _ = os.path.splitext(path)
    for cand in (base + ".parquet", base + ".csv"):
        if os.path.exists(cand):
            return read_df_auto(cand)
    raise FileNotFoundError(path)


def write_df_auto(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".csv"):
        df.reset_index().to_csv(path, index=False)
        return
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=True)
        return
    # 默认parquet
    df.to_parquet(path + ".parquet", index=True)


def get_last_timestamp(path: str, fast: bool = True) -> pd.Timestamp | None:
    """
    获取文件的最新时间戳

    Args:
        path: 文件路径（支持csv/parquet，自动查找）
        fast: True=仅读取最后几行（快速），False=读取全文件（精确）

    Returns:
        最新时间戳，失败返回None
    """
    try:
        # 解析实际路径
        actual_path = path
        if not os.path.exists(path):
            base, _ = os.path.splitext(path)
            for cand in (base + ".parquet", base + ".csv"):
                if os.path.exists(cand):
                    actual_path = cand
                    break

        if not os.path.exists(actual_path):
            return None

        # 🔥 快速模式：仅读取最后N行
        if fast:
            if actual_path.lower().endswith(".parquet"):
                # Parquet：使用pyarrow直接读取最后100行（最快）
                try:
                    import pyarrow.parquet as pq

                    # 方法1：读取metadata获取总行数
                    parquet_file = pq.ParquetFile(actual_path)
                    total_rows = parquet_file.metadata.num_rows

                    # 只读最后100行
                    if total_rows > 100:
                        # 使用pyarrow的切片读取（比pandas快）
                        table = parquet_file.read_row_groups([parquet_file.num_row_groups - 1])
                        df = table.to_pandas()
                        df = df.iloc[-100:]
                    else:
                        df = pd.read_parquet(actual_path)
                except Exception:
                    # 回退：直接pandas读取（会读全部，但也够快）
                    df = pd.read_parquet(actual_path)
                    df = df.iloc[-100:] if len(df) > 100 else df
            else:
                # CSV：使用tail命令（Linux）或pandas
                try:
                    # 方法1：系统tail（最快）
                    import subprocess

                    result = subprocess.run(
                        ["tail", "-n", "100", actual_path], capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        from io import StringIO

                        df = pd.read_csv(StringIO(result.stdout))
                    else:
                        raise Exception("tail failed")
                except Exception:
                    # 方法2：pandas读取最后N行
                    df = pd.read_csv(actual_path)
                    df = df.iloc[-100:] if len(df) > 100 else df
        else:
            # 完整模式：读取全文件
            df = read_df_auto(actual_path)

        # 查找时间列
        ts_col = None
        for cand in ["timestamp", "time", "datetime", "ts"]:
            if cand in df.columns:
                ts_col = cand
                break

        if ts_col:
            # 智能检测：整数用毫秒，字符串自动推断
            if pd.api.types.is_integer_dtype(df[ts_col]):
                ts = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
            else:
                ts = pd.to_datetime(df[ts_col], errors="coerce")
            return ts.max()

        # 如果有索引且是时间类型
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.max()

        # 尝试第一列
        if len(df.columns) > 0:
            first_col = df.iloc[:, 0]
            if pd.api.types.is_integer_dtype(first_col):
                ts = pd.to_datetime(first_col, unit="ms", errors="coerce")
            else:
                ts = pd.to_datetime(first_col, errors="coerce")
            if ts.notna().any():
                return ts.max()

        return None
    except Exception:
        return None


def print_latest_timestamp(path: str, label: str = "目前最新日期", fast: bool = True) -> None:
    """
    打印文件的最新时间戳（统一格式）

    Args:
        path: 文件路径
        label: 打印标签
        fast: 是否使用快速模式（仅读最后100行）
    """
    try:
        latest_ts = get_last_timestamp(path, fast=fast)
        if isinstance(latest_ts, pd.Timestamp) and not pd.isna(latest_ts):
            print(f"📅 {label}：{latest_ts.strftime('%Y年%m月%d日 %H:%M')}")
    except Exception:
        pass


def print_latest_timestamp_from_df(df: pd.DataFrame, label: str = "目前最新日期") -> None:
    """
    从DataFrame打印最新时间戳（已加载到内存的情况）

    Args:
        df: DataFrame（应该有timestamp列或DatetimeIndex）
        label: 打印标签
    """
    try:
        latest_ts = None

        # 方法1：从索引获取
        if isinstance(df.index, pd.DatetimeIndex):
            latest_ts = df.index.max()

        # 方法2：从timestamp列获取
        need_fallback = False
        if latest_ts is None:
            need_fallback = True
        elif isinstance(latest_ts, pd.Timestamp) and pd.isna(latest_ts):
            need_fallback = True
        if need_fallback:
            for ts_col in ["timestamp", "time", "datetime", "ts"]:
                if ts_col in df.columns:
                    ts = pd.to_datetime(df[ts_col], errors="coerce")
                    latest_ts = ts.max()
                    break

        if isinstance(latest_ts, pd.Timestamp) and not pd.isna(latest_ts):
            print(f"📅 {label}：{latest_ts.strftime('%Y年%m月%d日 %H:%M')}")
    except Exception:
        pass


# =========================
# 统一 IO 管理器（集中式路径与读写）
# =========================


class IOManager:
    """
    统一的 IO 管理器：基于 main_config.yaml 的 io 配置与模板，提供标准路径与读写接口。
    约束：
    - Step1 强制 CSV
    - Step2-4 默认 Parquet（或遵从 io.output_format: csv|parquet|both）
    - Step5 强制 NPZ（本类不负责写入 NPZ，只负责路径）
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config or {}
        self.io_cfg = self.cfg.get("io", {}) or {}
        self.patterns = self.io_cfg.get("filename_patterns") or {}

        # 目录
        self.base_dir = self._abspath(
            self.io_cfg.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
        )
        self.downloads_dir = self._abspath(
            self.io_cfg.get("downloads_dir") or os.path.join(self.base_dir, "rl_live", "data_downloads")
        )
        self.kline_dir = self._abspath(
            self.io_cfg.get("kline_dir") or os.path.join(self.base_dir, "rl_live", "kline")
        )
        self.indicators_dir = self._abspath(
            self.io_cfg.get("indicators_dir") or os.path.join(self.base_dir, "rl_live", "ind")
        )
        self.merged_dir = self._abspath(
            self.io_cfg.get("merged_dir") or os.path.join(self.base_dir, "rl_live", "merged")
        )
        self.rl_ready_dir = self._abspath(
            self.io_cfg.get("rl_ready_dir") or os.path.join(self.base_dir, "rl_live", "data_ready")
        )

        # 其他配置
        self.output_format = (
            str(self.io_cfg.get("output_format", "parquet")).lower().strip()
        )  # 仅影响 Step2-4
        self.overwrite = bool(self.io_cfg.get("overwrite", False))

        # 渲染上下文（用于模板 {a.b}）
        self.context = self._build_context()

    # ---------- 公共 API ----------
    def path_for(self, kind: str, *, timeframe: Optional[str] = None, roll: bool = False) -> str:
        # 🔥 关键：若未提供或为空字符串，则对需要周期的种类自动回退到 main_config 的 base_download
        if (timeframe is None or (isinstance(timeframe, str) and not timeframe.strip())) and kind in (
            "download",
            "kline",
            "indicator",
        ):
            base_tf = (self.cfg.get("timeframes", {}) or {}).get("base_download") or "1m"
            timeframe = str(base_tf).strip()
        # 确保 timeframe 是字符串且非空
        if timeframe is None:
            timeframe = "1m"
        timeframe = str(timeframe).strip() or "1m"
        dir_path, template = self._dir_and_template(kind, roll=roll)
        name = self._render_template(template, timeframe=timeframe)
        return self._abspath(os.path.join(dir_path, name))

    def resolve_existing(self, path: str, prefer: Optional[List[str]] = None) -> Optional[str]:
        """
        根据偏好顺序（默认 ['parquet','csv']）解析实际存在的文件。
        若 path 存在则直接返回；否则尝试互换扩展名或附加扩展名。
        """
        prefer = prefer or ["parquet", "csv"]
        ap = self._abspath(path)
        if os.path.exists(ap):
            return ap
        root, ext = os.path.splitext(ap)
        cand_exts: List[str] = []
        if ext.lower() in (".csv", ".parquet"):
            other = ".parquet" if ext.lower() == ".csv" else ".csv"
            cand_exts = [other]
        else:
            cand_exts = ["." + e for e in prefer]
        for e in cand_exts:
            cand = root + e
            if os.path.exists(cand):
                return cand
        return None

    def read_table(self, kind: str, *, timeframe: Optional[str] = None, roll: bool = False) -> pd.DataFrame:
        """统一读取表格（自动兜底 .parquet/.csv）。"""
        path = self.path_for(kind, timeframe=timeframe, roll=roll)
        actual = self.resolve_existing(path)
        # 兼容历史路径：indicator/kline 存在子目录 <timeframe>/ 的情况
        if not actual and timeframe:
            name = os.path.basename(path)
            tf_variants = [str(timeframe), str(timeframe).lower(), str(timeframe).upper()]
            if kind == "indicator":
                for tf_dir in tf_variants:
                    alt = self._abspath(os.path.join(self.indicators_dir, tf_dir, name))
                    actual = self.resolve_existing(alt)
                    if actual:
                        break
            elif kind == "kline":
                for tf_dir in tf_variants:
                    alt = self._abspath(os.path.join(self.kline_dir, tf_dir, name))
                    actual = self.resolve_existing(alt)
                    if actual:
                        break
        if not actual:
            raise FileNotFoundError(path)
        return read_df_auto(actual)

    def write_table(
        self, kind: str, df: pd.DataFrame, *, timeframe: Optional[str] = None, roll: bool = False
    ) -> List[str]:
        """
        统一写表：返回实际写入的文件列表。
        - Step1(download): 强制 CSV
        - Step2-4(kline/indicator/merged): 遵从 io.output_format
        - Step5: 本函数不负责
        """
        path = self.path_for(kind, timeframe=timeframe, roll=roll)
        root, ext = os.path.splitext(path)
        written: List[str] = []

        if kind == "download":
            out = root + ".csv"
            self._ensure_dir(out)
            df.reset_index().to_csv(out, index=False)
            written.append(out)
            return written

        if kind in ("kline", "indicator", "merged"):
            fmt = self.output_format
            if fmt == "parquet":
                out = root + ".parquet"
                self._ensure_dir(out)
                df.to_parquet(out, index=True)
                written.append(out)
            elif fmt == "csv":
                out = root + ".csv"
                self._ensure_dir(out)
                df.reset_index().to_csv(out, index=False)
                written.append(out)
            elif fmt == "both":
                out_pq = root + ".parquet"
                out_csv = root + ".csv"
                self._ensure_dir(out_pq)
                self._ensure_dir(out_csv)
                df.to_parquet(out_pq, index=True)
                df.reset_index().to_csv(out_csv, index=False)
                written.extend([out_pq, out_csv])
            else:
                # 默认 parquet
                out = root + ".parquet"
                self._ensure_dir(out)
                df.to_parquet(out, index=True)
                written.append(out)
            return written

        # 其他 kind（如 rl_features/rl_labels）仅提供路径，不负责写入
        return written

    # ---------- 内部工具 ----------
    def _dir_and_template(self, kind: str, *, roll: bool) -> tuple[str, str]:
        # 目录选择
        dir_map = {
            "download": self.downloads_dir,
            "kline": self.kline_dir,
            "indicator": self.indicators_dir,
            "merged": self.merged_dir,
            "rl": self.rl_ready_dir,
            "rl_features": self.rl_ready_dir,
            "rl_labels": self.rl_ready_dir,
        }
        directory = dir_map.get(kind, self.base_dir)

        # 模板选择
        # 优先使用配置中的模板，否则降级为合理默认
        default_patterns = {
            "download": "{symbol.trading_pair_std}_{market_type}_{timeframe}.csv",
            "kline": "{symbol.trading_pair_std}_{timeframe}.parquet",
            "kline_roll": "{symbol.trading_pair_std}_{timeframe}_roll.parquet",
            "indicator": "{symbol.trading_pair_std}_{timeframe}_indicators.parquet",
            "merged": "{symbol.trading_pair_std}_$timeframe$_merged.parquet",
            "rl": "{symbol.trading_pair_std}_rl.npz",
            "rl_features": "{symbol.trading_pair_std}_$timeframe$_rl_features.npz",
            "rl_labels": "{symbol.trading_pair_std}_$timeframe$_rl_labels.npz",
        }

        if kind == "kline" and roll:
            template_key = "kline_roll"
        else:
            template_key = kind if kind in default_patterns else "merged"

        template = self.patterns.get(template_key, default_patterns[template_key])
        return directory, template

    def _render_template(self, template: str, *, timeframe: Optional[str]) -> str:
        """
        渲染模板占位符：
        - {a.b} 单花括号：由 ConfigLoader 在加载配置时替换（静态配置值）
        - $xxx$ 美元符号：由 IOManager 运行时替换（动态运行时参数，如 timeframe）
        """
        # 额外上下文（运行时）
        runtime_ctx = dict(self.context)
        # 🔥 关键：timeframe 必须有值
        if timeframe is None or (isinstance(timeframe, str) and not timeframe.strip()):
            timeframe = runtime_ctx.get("timeframes.base_download", "1m")
        # 确保是字符串且非空
        timeframe = str(timeframe).strip() or "1m"
        runtime_ctx["timeframe"] = timeframe

        def deep_get(d: Dict[str, Any], dotted: str) -> Any:
            cur: Any = d
            for part in dotted.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur

        # 🔥 关键：替换 $xxx$ 运行时参数
        def repl_runtime(m: re.Match) -> str:
            key = m.group(1)
            # 运行时参数优先从 runtime_ctx 获取
            if key in runtime_ctx:
                val = runtime_ctx[key]
            else:
                val = deep_get(self.cfg, key)
            if val is None:
                raise KeyError(f"运行时参数 '${key}$' 未找到，模板：{template}")
            return str(val)

        # 替换 $xxx$ 运行时参数
        result = re.sub(r"\$([\w\.]+)\$", repl_runtime, template)

        # 兜底：替换单花括号（理论上已被 ConfigLoader 替换）
        def repl_single(m: re.Match) -> str:
            key = m.group(1)
            if key in runtime_ctx:
                val = runtime_ctx[key]
            else:
                val = deep_get(self.cfg, key)
            if val is None:
                return ""
            return str(val)

        result = re.sub(r"\{([\w\.]+)\}", repl_single, result)
        return result

    def _build_context(self) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        # 常用字段（避免模板缺失时兜底）
        symbol = self.cfg.get("symbol", {}) or {}
        ctx["symbol.trading_pair_std"] = symbol.get("trading_pair_std", "ETH_USDT")
        ctx["symbol.trading_pair_exchange"] = symbol.get("trading_pair_exchange", "ETH/USDT")
        # ConfigLoader.load_main_config 已将 market_type 提升到顶层并大写
        ctx["market_type"] = self.cfg.get("market_type", symbol.get("market_type", "SWAP")).upper()
        # timeframes.base_download
        tf = self.cfg.get("timeframes", {}) or {}
        ctx["timeframes.base_download"] = tf.get("base_download", "1m")
        return ctx

    def get_min_resample_timeframe(self) -> str:
        """获取 resample_targets 中最小的时间周期（单位按 m/h/d 解析）。"""
        tf_cfg = self.cfg.get("timeframes", {}) or {}
        targets = tf_cfg.get("resample_targets") or []
        base_download = tf_cfg.get("base_download", "1m")
        if not targets:
            return base_download

        def _to_minutes(s: str) -> int:
            try:
                s2 = str(s).strip().lower()
                m = re.match(r"^(\d+)([mhd])$", s2)
                if not m:
                    return 10**9
                val = int(m.group(1))
                unit = m.group(2)
                mult = 1 if unit == "m" else (60 if unit == "h" else 1440)
                return val * mult
            except Exception:
                return 10**9

        try:
            return sorted(targets, key=_to_minutes)[0]
        except Exception:
            return base_download

    @staticmethod
    def _abspath(p: str) -> str:
        return os.path.abspath(p)

    @staticmethod
    def _ensure_dir(path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
