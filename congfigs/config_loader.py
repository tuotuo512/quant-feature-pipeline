"""
配置加载器 - 统一管理所有配置文件的读取和验证
用途: 为数据处理流水线提供标准化的配置接口
"""

import os
import re
import copy
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class ConfigLoader:
    """配置加载器 - 负责读取和验证YAML配置"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器

        参数:
            config_dir: 配置文件目录，默认为当前脚本所在目录
        """
        if config_dir is None:
            # 当前文件位于 congfigs/ 目录，默认使用该目录
            current_file = Path(__file__).resolve()
            config_dir = current_file.parent

        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")

        print(f"配置目录: {self.config_dir}")

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        加载YAML配置文件

        参数:
            filename: 配置文件名

        返回:
            配置字典
        """
        file_path = self.config_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        print(f"[OK] 已加载配置文件: {filename}")
        return config

    # ================= 新增：主配置与占位符插值/级联合并 =================
    def load_main_config(self) -> Dict[str, Any]:
        """加载并解析 main_config.yaml（含占位符内部插值 + 自动推导）。"""
        try:
            cfg = self.load_yaml("main_config.yaml")
        except FileNotFoundError:
            return {}

        # [HOT] 自动推导：根据 rl_build.source_mode 推导 timeframes.include_rolling
        rl_build = cfg.get("rl_build", {})
        source_mode = rl_build.get("source_mode", "fixed")

        if "timeframes" not in cfg:
            cfg["timeframes"] = {}

        # 推导逻辑：sliding → include_rolling=true；fixed → false
        cfg["timeframes"]["include_rolling"] = source_mode == "sliding"

        # [HOT] 提取 market_type 到顶层，便于占位符引用 {market_type}
        symbol_cfg = cfg.get("symbol", {})
        market_type = symbol_cfg.get("market_type", "swap")
        cfg["market_type"] = market_type.upper()  # 转大写用于文件命名（SWAP/SPOT）

        print(
            f"[INFER] 推导: rl_build.source_mode='{source_mode}' -> timeframes.include_rolling={cfg['timeframes']['include_rolling']}"
        )
        print(
            f"[INFER] 推导: symbol.market_type='{market_type}' -> market_type='{cfg['market_type']}' (用于文件命名)"
        )

        # 多轮插值，解决主配置内部相互引用（如 io.downloads_dir 引用 io.base_dir）
        resolved = self._resolve_placeholders_multi_pass(cfg, cfg, max_passes=4)
        return resolved

    # ================= 配置对齐校验 =================
    def validate_live_alignment(self, live_cfg: Dict[str, Any]) -> None:
        """
        校验 live_overrides.yaml 与主配置的一致性。
        若发现关键字段不一致，直接抛出 ValueError 阻断运行。
        """
        if not isinstance(live_cfg, dict):
            raise ValueError("live_overrides 配置无效：应为 dict")

        main_cfg = self.load_main_config()
        issues = []

        # 1. 周期契约：base_period 与 resample_targets
        main_timeframes = main_cfg.get("timeframes", {}) or {}
        main_resample = [str(p) for p in (main_timeframes.get("resample_targets") or [])]
        expected_base = main_resample[0] if main_resample else ""

        fc_cfg = live_cfg.get("features_contract") or {}
        live_base = str(fc_cfg.get("base_period") or "")
        live_periods = [str(p) for p in (fc_cfg.get("periods_in_use") or [])]

        if expected_base and live_base and live_base != expected_base:
            issues.append(
                f"features_contract.base_period 应与 main_config.timeframes.resample_targets[0] 对齐： 当前 live={live_base}, expected={expected_base}"
            )

        if main_resample:
            expected_periods = sorted(set(main_resample))
            current_periods = sorted(set(live_periods))
            if current_periods and current_periods != expected_periods:
                issues.append(
                    "features_contract.periods_in_use 应与 main_config.timeframes.resample_targets 完全一致："
                    f" 当前 live={current_periods}, expected={expected_periods}"
                )

        # 2. 特征源路径：应指向 main_config.io.rl_ready_dir 下的标准命名文件
        # 🔥 支持 {symbol} 占位符（从 live_overrides.yaml 的 exchanges 配置动态获取）
        io_cfg = main_cfg.get("io", {}) or {}
        rl_ready_dir = io_cfg.get("rl_ready_dir")
        
        # 🔥 优先从 live_cfg.exchanges 获取 symbol
        symbol_std = self._resolve_symbol_from_exchanges(live_cfg)
        if not symbol_std:
            symbol_cfg = main_cfg.get("symbol", {}) or {}
            symbol_std = symbol_cfg.get("trading_pair_std")
        
        source_cfg = fc_cfg.get("source") or {}
        source_pattern = source_cfg.get("path_pattern")
        if rl_ready_dir and symbol_std and live_base and source_pattern:
            expected_path = os.path.join(rl_ready_dir, f"{symbol_std}_{live_base}_rl_features.npz")
            try:
                # 🔥 支持 {symbol} 和 {base_period} 两个占位符
                resolved_pattern = source_pattern.format(symbol=symbol_std, base_period=live_base)
            except KeyError:
                resolved_pattern = source_pattern
            if os.path.abspath(resolved_pattern) != os.path.abspath(expected_path):
                issues.append(
                    "features_contract.source.path_pattern 应指向主配置 rl_ready_dir 下的标准文件："
                    f" 当前路径={resolved_pattern}, 期望={expected_path}"
                )

        # 3. microbatch / save_n：预检与流水线应共用同一窗口
        micro_cfg = (main_cfg.get("online") or {}).get("microbatch") or {}
        expected_micro_len = micro_cfg.get("length")
        preheat_cfg = live_cfg.get("preheat") or {}
        live_save_n = preheat_cfg.get("save_n")
        if expected_micro_len and live_save_n and int(expected_micro_len) != int(live_save_n):
            issues.append(
                f"live_overrides.preheat.save_n ({live_save_n}) 应与 main_config.online.microbatch.length ({expected_micro_len}) 一致"
            )

        if issues:
            msg = "live_overrides 与主配置存在不一致:\n- " + "\n- ".join(issues)
            raise ValueError(msg)

    def _resolve_symbol_from_exchanges(self, live_cfg: Dict[str, Any]) -> Optional[str]:
        """
        🔥 从 live_overrides.yaml 的 exchanges 配置读取启用的交易所的 symbol
        
        Returns:
            symbol_std（如 "BTC_USDT"），如果未配置则返回 None
        """
        exchanges_cfg = live_cfg.get("exchanges", {}) or {}
        
        for ex_name in ["okx", "bitget"]:
            ex_cfg = exchanges_cfg.get(ex_name, {}) or {}
            if ex_cfg.get("enabled", False):
                symbol = ex_cfg.get("symbol")
                if symbol:
                    # 解析：例如 "BTC/USDT:USDT" → "BTC_USDT"
                    return symbol.split(":")[0].replace("/", "_")
        
        return None

    def load_yaml_with_main(self, filename: str) -> Dict[str, Any]:
        """加载任意YAML文件，并使用 main_config 进行占位符插值 + 合并主配置。"""
        cfg = self.load_yaml(filename)
        main_cfg = self.load_main_config()

        # 先解析占位符（Step YAML中的 {io.xxx} 等引用）
        resolved_cfg = self._resolve_placeholders_multi_pass(cfg, main_cfg, max_passes=4)

        # 再将主配置的关键字段合并到 Step 配置（确保 Step 可以访问全局配置）
        merged = self._deep_merge_dicts(main_cfg, resolved_cfg)

        return merged

    def load_step1_config(self) -> "Step1DataConfig":
        """加载Step1数据下载配置"""
        config_dict = self.load_yaml_with_main("step1_data download.yaml")
        return Step1DataConfig(config_dict)

    def load_step2_config(self) -> Dict[str, Any]:
        """加载Step2配置（自动套用主配置占位符）。"""
        return self.load_yaml_with_main("step2_resample.yaml")

    def load_step3_config(self) -> Dict[str, Any]:
        """加载Step3配置；若 step3_indicators.yaml 缺失则回退到 base_indicators.yaml。"""
        try:
            return self.load_yaml_with_main("step3_indicators.yaml")
        except FileNotFoundError:
            print("[WARN] step3_indicators.yaml 未找到，回退到 base_indicators.yaml")
            return self.load_yaml_with_main("base_indicators.yaml")

    def load_step4_config(self) -> Dict[str, Any]:
        """加载Step4配置（自动套用主配置占位符）。"""
        return self.load_yaml_with_main("step4_merge.yaml")

    def load_step5_config(self) -> Dict[str, Any]:
        """加载Step5配置（自动套用主配置占位符）。"""
        return self.load_yaml_with_main("step5_mapping.yaml")

    # ================= 辅助：占位符解析 =================
    def _resolve_placeholders_multi_pass(
        self,
        cfg: Dict[str, Any],
        context: Dict[str, Any],
        max_passes: int = 3,
    ) -> Dict[str, Any]:
        """对 cfg 进行多轮占位符解析，支持 {a.b.c} 形式；若整值即占位符，则返回原类型对象。"""
        result = copy.deepcopy(cfg)
        for _ in range(max_passes):
            before = yaml.dump(result, allow_unicode=True)
            result = self._resolve_placeholders_once(result, context)
            after = yaml.dump(result, allow_unicode=True)
            if before == after:
                break
            context = self._deep_merge_dicts(context, result)
        return result

    def _resolve_placeholders_once(self, obj: Any, context: Dict[str, Any]) -> Any:
        if isinstance(obj, dict):
            return {k: self._resolve_placeholders_once(v, context) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_placeholders_once(v, context) for v in obj]
        if isinstance(obj, str):
            return self._interpolate_string(obj, context)
        return obj

    def _interpolate_string(self, s: str, context: Dict[str, Any]) -> Any:
        s = s.strip()
        m = re.fullmatch(r"\{([\w\.]+)\}", s)
        if m:
            key = m.group(1)
            val = self._deep_get(context, key)
            return copy.deepcopy(val) if val is not None else s

        def repl(match: re.Match) -> str:
            key = match.group(1)
            val = self._deep_get(context, key)
            return "" if val is None else str(val)

        return re.sub(r"\{([\w\.]+)\}", repl, s)

    def _deep_get(self, data: Dict[str, Any], dotted_key: str) -> Any:
        cur = data
        for part in dotted_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(base, dict):
            return copy.deepcopy(override)
        result = copy.deepcopy(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = self._deep_merge_dicts(result[k], v)
            else:
                result[k] = copy.deepcopy(v)
        return result


class Step1DataConfig:
    """Step1数据下载配置的结构化对象"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        从配置字典初始化

        参数:
            config_dict: 从YAML加载的配置字典
        """
        self.raw_config = config_dict

        self._parse_symbol_config()
        self._parse_time_range_config()
        self._parse_timeframes_config()
        self._parse_fetch_strategy_config()
        self._parse_network_config()
        self._parse_api_auth_config()
        self._parse_output_config()
        self._parse_logging_config()

    def _parse_symbol_config(self):
        exchange_cfg = self.raw_config.get("exchange", {})
        self.exchange_name = exchange_cfg.get("name", "binance")

        valid_exchanges = ["binance", "okx"]
        if self.exchange_name not in valid_exchanges:
            raise ValueError(f"无效的交易所: {self.exchange_name}，可选: binance | okx")

        symbol_cfg = self.raw_config.get("symbol", {})
        # 🔥 优先使用 trading_pair_exchange，兼容旧配置的 trading_pair
        self.trading_pair = symbol_cfg.get("trading_pair_exchange") or symbol_cfg.get("trading_pair", "ETH/USDT")
        self.trading_pair_std = symbol_cfg.get("trading_pair_std", self.trading_pair.replace("/", "_"))
        self.market_type = symbol_cfg.get("market_type", "swap")

        valid_market_types = ["spot", "swap"]
        if self.market_type not in valid_market_types:
            raise ValueError(f"无效的市场类型: {self.market_type}，可选: spot(现货) | swap(永续合约)")

    def _parse_time_range_config(self):
        time_cfg = self.raw_config.get("time_range", {})
        self.time_mode = time_cfg.get("mode", "incremental")

        if self.time_mode == "incremental":
            incr_cfg = time_cfg.get("incremental", {})
            self.days_if_missing = incr_cfg.get("days_if_missing", 60)
            self.fill_missing = incr_cfg.get("fill_missing", True)
            self.initial_start = incr_cfg.get("initial_start", None)

        elif self.time_mode == "full":
            full_cfg = time_cfg.get("full", {})
            self.years_of_data = full_cfg.get("years_of_data", 2)

        elif self.time_mode == "days":
            days_cfg = time_cfg.get("days", {})
            self.recent_days = days_cfg.get("recent_days", 90)

        elif self.time_mode == "custom":
            custom_cfg = time_cfg.get("custom", {})
            self.start_date = custom_cfg.get("start_date")
            self.end_date = custom_cfg.get("end_date")

            if self.start_date:
                try:
                    datetime.strptime(self.start_date, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"start_date 格式错误: {self.start_date}，应为 YYYY-MM-DD")

            if self.end_date:
                try:
                    datetime.strptime(self.end_date, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"end_date 格式错误: {self.end_date}，应为 YYYY-MM-DD")

    def _parse_timeframes_config(self):
        tf_cfg = self.raw_config.get("timeframes", {})

        if "target" in tf_cfg:
            self.timeframe = tf_cfg["target"]
            self.timeframes = [self.timeframe]
        elif "multi" in tf_cfg:
            self.timeframes = tf_cfg["multi"]
            self.timeframe = self.timeframes[0] if self.timeframes else "5m"
        else:
            self.timeframe = "5m"
            self.timeframes = ["5m"]

    def _parse_fetch_strategy_config(self):
        fetch_cfg = self.raw_config.get("fetch_strategy", {})
        self.retry_count = fetch_cfg.get("retry_count", 5)
        self.batch_size = fetch_cfg.get("batch_size", 500)
        self.delay_ms = fetch_cfg.get("delay_ms", 500)
        self.timeout_ms = fetch_cfg.get("timeout_ms", 6000)

    def _parse_network_config(self):
        net_cfg = self.raw_config.get("network", {})
        self.auto_detect = net_cfg.get("auto_detect", True)
        self.use_proxy = net_cfg.get("use_proxy", True)
        self.proxy_url = net_cfg.get("proxy_url", "http://127.0.0.1:18081")

        conn_cfg = net_cfg.get("connectivity_check", {})
        self.connectivity_enabled = conn_cfg.get("enabled", True)
        self.test_google = conn_cfg.get("test_google", True)
        self.test_binance = conn_cfg.get("test_binance", True)
        self.strict_mode = conn_cfg.get("strict_mode", False)

    def _parse_api_auth_config(self):
        auth_cfg = self.raw_config.get("api_auth", {})
        self.use_env_auth = auth_cfg.get("use_env", True)
        self.require_auth = auth_cfg.get("require_auth", False)

    def _parse_output_config(self):
        out_cfg = self.raw_config.get("output", {})
        _home_default = os.path.join(os.path.expanduser("~"), "FinRL_bn", "data", "data_downloads")
        self.base_dir = out_cfg.get("base_dir", _home_default)
        self.filename_pattern = out_cfg.get("filename_pattern", "{symbol}_{timeframe}.csv")

        qc_cfg = out_cfg.get("quality_check", {})
        self.remove_duplicates = qc_cfg.get("remove_duplicates", True)
        self.fill_missing_values = qc_cfg.get("fill_missing_values", True)
        self.check_completeness = qc_cfg.get("check_completeness", True)
        self.add_time_features = qc_cfg.get("add_time_features", True)

    def _parse_logging_config(self):
        log_cfg = self.raw_config.get("logging", {})
        self.verbose = log_cfg.get("verbose", True)
        self.show_progress = log_cfg.get("show_progress", True)
        self.progress_interval = log_cfg.get("progress_interval", 5)
        self.save_log = log_cfg.get("save_log", False)
        self.log_file = log_cfg.get("log_file", "data_download.log")

    def get_output_filename(self, timeframe: Optional[str] = None) -> str:
        if timeframe is None:
            timeframe = self.timeframe

        pattern = self.filename_pattern
        if pattern.endswith("_.csv"):
            return pattern[:-5] + f"_{timeframe}.csv"

        try:
            return pattern.format(
                symbol=self.trading_pair.replace("/", "_"),
                timeframe=timeframe,
                start_date="",
                end_date="",
            )
        except (KeyError, ValueError):
            return f"{self.trading_pair.replace('/', '_')}_{self.market_type.upper()}_{timeframe}.csv"

    def get_output_path(self, timeframe: Optional[str] = None) -> str:
        filename = self.get_output_filename(timeframe)
        return os.path.join(self.base_dir, filename)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("[CLIPBOARD] 数据下载配置摘要")
        print("=" * 60)
        print(f"交易所: {self.exchange_name}")
        print(f"交易对: {self.trading_pair}")
        print(f"市场类型: {self.market_type}")
        print(f"时间周期: {', '.join(self.timeframes)}")
        print(f"时间模式: {self.time_mode}")

        if self.time_mode == "incremental":
            print(f"  - 本地无文件时抓取天数: {self.days_if_missing}")
            print(f"  - 补齐缺失K线: {self.fill_missing}")
            if self.initial_start:
                print(f"  - 初始起点: {self.initial_start}")
        elif self.time_mode == "full":
            print(f"  - 历史数据年数: {self.years_of_data}")
        elif self.time_mode == "days":
            print(f"  - 最近天数: {self.recent_days}")
        elif self.time_mode == "custom":
            print(f"  - 开始日期: {self.start_date}")
            print(f"  - 结束日期: {self.end_date}")

        print(f"\n网络配置:")
        print(f"  - 自动检测: {self.auto_detect}")
        print(f"  - 使用代理: {self.use_proxy}")
        if self.use_proxy:
            print(f"  - 代理地址: {self.proxy_url}")

        print(f"\n获取策略:")
        print(f"  - 重试次数: {self.retry_count}")
        print(f"  - 批次大小: {self.batch_size}")
        print(f"  - 延迟(ms): {self.delay_ms}")
        print(f"  - 超时(ms): {self.timeout_ms}")

        print(f"\n输出配置:")
        print(f"  - 保存目录: {self.base_dir}")
        print(f"  - 文件名: {self.get_output_filename()}")
        print(f"  - 去重: {self.remove_duplicates}")
        print(f"  - 补缺: {self.fill_missing_values}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    print("测试配置加载器...")

    try:
        loader = ConfigLoader()
        config = loader.load_step1_config()
        config.print_summary()

        print("\n[OK] 配置加载测试成功！")

    except Exception as e:
        print(f"\n[ERROR] 配置加载测试失败: {e}")
        import traceback

        traceback.print_exc()
