#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载脚本 - 支持多交易所、多市场类型
用途: 下载原始K线数据（训练+实盘共用）
配置: congfigs/step1_data download.yaml + main_config.yaml
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
import time
from typing import Optional

try:
    import pandas as pd
    import ccxt
except ImportError as e:
    print(f"❌ 导入依赖失败: {e}")
    print("请运行: pip install pandas ccxt")
    sys.exit(1)

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入统一代理管理器
try:
    from common.http_proxy.proxy_manager import get_proxy_manager
except ImportError:
    # 如果找不到，尝试从项目根目录导入
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from common.http_proxy.proxy_manager import get_proxy_manager


class EnhancedCCXTProcessor:
    """
    增强版的CCXT处理器，独立于FinRL实现
    支持多种时间周期和更好的数据获取策略
    """

    def __init__(
        self,
        exchange_name="binance",
        market_type="spot",
        use_proxy: Optional[bool] = None,
        proxy_url: Optional[str] = None,
        timeout=3000,
    ):
        """
        初始化增强版CCXT处理器

        参数:
            exchange_name: 交易所名称 ('binance', 'okx')
            market_type: 市场类型 ('spot', 'swap')
            use_proxy: 是否强制使用代理（True=强制开启，False=强制关闭，None=自动检测）
            proxy_url: 代理服务器地址（仅在 use_proxy=True 时生效）
            timeout: 请求超时时间(毫秒)
        """
        print(f"初始化CCXT处理器 (交易所: {exchange_name}, 市场类型: {market_type})...")
        self.exchange_name = exchange_name.lower()
        self.market_type = str(market_type).lower()
        self.use_proxy = use_proxy
        self.proxy_url = proxy_url

        # 根据交易所获取API密钥环境变量名
        if self.exchange_name == "okx":
            api_key = os.environ.get("OKX_API_KEY", "")
            api_secret = os.environ.get("OKX_API_SECRET", "")
            api_passphrase = os.environ.get("OKX_API_PASSPHRASE", "")
        else:  # binance
            api_key = os.environ.get("BINANCE_API_KEY", "")
            api_secret = os.environ.get("BINANCE_API_SECRET", "")
            api_passphrase = None

        # 设置交易所客户端配置
        exchange_config = {
            "timeout": timeout,
            "enableRateLimit": True,
            "options": {
                "recvWindow": 60000,  # 增加接收窗口时间
                "adjustForTimeDifference": True,  # 自动调整时间差
                "keepAlive": True,  # 保持连接活跃
            },
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        }

        # 根据市场类型与交易所设置 defaultType（spot 与 swap）
        if market_type == "swap":
            # Binance 永续在 CCXT 使用 'future'; OKX 永续在 CCXT 使用 'swap'
            if self.exchange_name == "binance":
                exchange_config["options"]["defaultType"] = "future"
            elif self.exchange_name == "okx":
                exchange_config["options"]["defaultType"] = "swap"
            else:
                exchange_config["options"]["defaultType"] = "swap"
            print(
                f"已设置市场类型为: swap(永续合约) -> CCXT defaultType: {exchange_config['options']['defaultType']}"
            )
            # 为永续合约设置常见选项
            exchange_config["options"]["defaultMarginMode"] = "cross"
            exchange_config["options"]["createMarketBuyOrderRequiresPrice"] = False
            exchange_config["options"]["fetchTickerQuoteAsset"] = True
            exchange_config["options"]["broker"] = "CCXT"
            print("已为永续合约配置额外参数")
        elif market_type == "spot":
            exchange_config["options"]["defaultType"] = "spot"
            print(
                f"已设置市场类型为: spot(现货) -> CCXT defaultType: {exchange_config['options']['defaultType']}"
            )
        else:
            raise ValueError(f"不支持的市场类型: '{market_type}'，仅支持 spot(现货) 或 swap(永续合约)")

        # 如果API密钥在环境变量中存在
        if api_key and api_secret:
            exchange_config["apiKey"] = api_key
            exchange_config["secret"] = api_secret
            if api_passphrase:
                exchange_config["password"] = api_passphrase
            print(f"✅ 使用API密钥连接{self.exchange_name}")
        else:
            print(f"⚠️ 公开数据模式（无API密钥）")

        # 初始化CCXT交易所
        if self.exchange_name == "okx":
            self.exchange = ccxt.okx(exchange_config)
        else:  # binance
            self.exchange = ccxt.binance(exchange_config)
        print(f"✅ {self.exchange_name}交易所初始化完成")

        # 代理设置：优先使用显式配置，否则使用ProxyManager自动模式
        if self.use_proxy is True and self.proxy_url:
            print(f"设置代理（手动指定）: {self.proxy_url}")
            self.exchange.proxies = {"http": self.proxy_url, "https": self.proxy_url}
        elif self.use_proxy is False:
            print("不使用代理，直连模式")
            try:
                self.exchange.proxies = None
            except Exception:
                self.exchange.proxies = {}
            try:
                self.exchange.session.trust_env = False
            except Exception:
                pass
        else:
            print("使用ProxyManager自动检测代理配置...")
            proxy_manager = get_proxy_manager()
            proxy_config = proxy_manager.get_proxy_config()
            if proxy_config["use_proxy"] and proxy_config["proxies"]:
                self.exchange.proxies = proxy_config["proxies"]
                print(f"已应用代理: {proxy_config['http_proxy']}")
            else:
                print("ProxyManager检测为直连模式")
                try:
                    self.exchange.proxies = None
                except Exception:
                    self.exchange.proxies = {}
                try:
                    self.exchange.session.trust_env = False
                except Exception:
                    pass

        # 设置请求超时参数
        timeout_ms = int(max(3000, min(timeout, 15000)))
        self.exchange.timeout = timeout_ms
        self.exchange.httpOptions = {"timeout": timeout_ms, "keepAlive": True}

        # 支持的时间周期
        self.supported_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]

        # 同步时间
        self._sync_time()

        # 连接测试
        connection_success = self._test_connection()
        if not connection_success:
            print("⚠️ 连接测试失败，数据获取可能会失败")

    def _sync_time(self):
        """同步本地时间与服务器时间"""
        try:
            server_time = self.exchange.fetch_time()
            if server_time is None:
                raise RuntimeError("未获取到服务器时间")

            local_time = int(time.time() * 1000)
            time_diff = int(server_time) - local_time
            self.exchange.options["timeDifference"] = time_diff

            if abs(time_diff) > 1000:
                print(f"⚠️ 时间差: {time_diff}ms (已自动调整)")
            else:
                print(f"✅ 时间同步正常 (差异: {time_diff}ms)")

        except Exception as e:
            print(f"⚠️ 时间同步失败: {e}")

    def format_date(self, year, month, day):
        """
        将年月日转换为所需的日期格式

        参数:
            year: 年份，如2024
            month: 月份，如1
            day: 日期，如1

        返回:
            格式化的日期字符串，如"2024-01-01"
        """
        return f"{year}-{month:02d}-{day:02d}"

    def _test_connection(self):
        """测试与交易所的连接（按交易所/市场类型选择可用符号）"""
        try:
            # 确保加载市场
            try:
                self.exchange.load_markets()
            except Exception:
                pass

            # 优先候选列表
            candidate_symbols = []
            if self.exchange_name == "okx":
                if self.market_type == "swap":
                    candidate_symbols = ["ETH/USDT:USDT", "BTC/USDT:USDT", "ETH/USDT", "BTC/USDT"]
                else:
                    candidate_symbols = ["ETH/USDT", "BTC/USDT"]
            else:  # binance 及其他
                if self.market_type == "swap":
                    candidate_symbols = ["ETH/USDT", "BTC/USDT"]
                else:
                    candidate_symbols = ["ETH/USDT", "BTC/USDT"]

            # 在交易所支持的符号中选一个可用的
            symbols = getattr(self.exchange, "symbols", None) or []
            chosen = None
            for s in candidate_symbols:
                if s in symbols:
                    chosen = s
                    break
            if chosen is None and symbols:
                chosen = symbols[0]

            # 实测连接
            ticker = self.exchange.fetch_ticker(self._normalize_symbol(chosen or candidate_symbols[0]))
            last_price = ticker.get("last") if isinstance(ticker, dict) else None
            print(f"✅ 连接测试成功 (symbol: {chosen}, 当前价格: {last_price})")
            return True
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        """按交易所/市场类型规范化交易对符号，避免因符号不一致导致请求失败"""
        sym = str(symbol).upper().replace(" ", "")
        if self.exchange_name == "okx" and self.market_type == "swap":
            # OKX 永续在 CCXT 统一使用形如 ETH/USDT:USDT
            if ":USDT" not in sym and sym.endswith("/USDT"):
                return sym + ":USDT"
        return sym

    def fetch_data(
        self, symbol, timeframe, start_date, end_date, retry_count=3, batch_size=1000, delay_ms=300
    ):
        """
        获取指定交易对和时间范围的数据并保存为CSV

        参数:
            symbol: 交易对名称，如"ETH/USDT"
            timeframe: 时间周期，如"1d", "4h", "1h", "30m", "5m"
            start_date: 开始日期，格式为"YYYY-MM-DD"
            end_date: 结束日期，格式为"YYYY-MM-DD"
            retry_count: 失败重试次数
            batch_size: 每批获取的K线数量
            delay_ms: 请求间隔延迟(毫秒)

        返回:
            保存好的数据文件路径
        """
        # 检查时间周期是否支持
        if timeframe not in self.supported_timeframes:
            print(f"警告: 时间周期 {timeframe} 不在支持列表中，可能会导致数据不完整")

        # 设置保存路径
        raw_data_path = self._get_data_path(symbol)  # 例如: data/data_downloads/raw/eth

        # 格式化日期
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(
            days=1, seconds=-1
        )  # 结束时间设为当天23:59:59
        # 如果终止日期超出当前时间，自动截断到当前时间
        now_dt = datetime.now()
        if end_datetime > now_dt:
            print("结束日期超出当前时间，已自动截断到当前时间")
            end_datetime = now_dt

        # 获取数据
        print(f"\n=== 开始获取 {symbol} {timeframe} 数据 ===")
        print(f"时间范围: {start_date} 至 {end_date}")

        # === 统一命名规范：直接保存到目标位置，格式 {SYMBOL}_{TIMEFRAME}.csv ===
        os.makedirs(raw_data_path, exist_ok=True)
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        final_file_path = os.path.join(raw_data_path, filename)
        print(f"数据将保存到: {os.path.abspath(final_file_path)}")
        # === 命名规范结束 ===

        # 使用改进的方法获取数据（启用分批增量写入）
        df = self.download_data(
            symbol=symbol,
            start_date=start_datetime,
            end_date=end_datetime,
            time_interval=timeframe,
            retry_count=retry_count,
            batch_size=batch_size,
            delay_ms=delay_ms,
            output_file=final_file_path,
        )

        if df is not None and not df.empty:
            # 添加日期信息字段，方便后续处理
            df["date_str"] = df.index.strftime("%Y-%m-%d")
            df["time_str"] = df.index.strftime("%H:%M:%S")
            df["day_of_week"] = df.index.dayofweek
            # df['hour_of_day'] = df.index.hour
            # df['minute_of_hour'] = df.index.minute

            # 检查数据质量
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                print(f"⚠️ 警告: 数据中包含 {missing_count} 个缺失值")
                # 填充缺失值
                df = df.ffill().bfill()
                print("已填充缺失值")

            # 检查重复索引
            duplicate_count = df.index.duplicated().sum()
            if duplicate_count > 0:
                print(f"⚠️ 警告: 数据中包含 {duplicate_count} 个重复索引")
                # 删除重复行
                df = df[~df.index.duplicated(keep="first")]
                print("已删除重复行")

            # 保存CSV文件（若启用分批写入，download_data 已完成写入；此处覆盖保存用于兜底确保一致性）
            df.to_csv(final_file_path, index=True)
            print(f"✅ 数据已保存至 {final_file_path}")
            print(f"时间范围: {df.index[0]} - {df.index[-1]}")
            print(f"总记录数: {len(df)}")

            # 分析数据完整性
            expected_intervals = self._calculate_expected_intervals(start_datetime, end_datetime, timeframe)
            completeness = min(100.0, (len(df) / expected_intervals) * 100) if expected_intervals > 0 else 0
            print(f"数据完整性: {completeness:.2f}% (预期记录数: {expected_intervals})")

            return final_file_path  # 返回新的文件路径
        else:
            print("❌ 获取数据失败")
            return None

    def _calculate_expected_intervals(self, start_date, end_date, timeframe):
        """计算指定时间范围内应有的K线数量"""
        # 时间周期转换为分钟
        tf_minutes = self._timeframe_to_minutes(timeframe)
        if tf_minutes == 0:  # 对于日线等情况
            return 0

        # 计算总分钟数
        total_minutes = (end_date - start_date).total_seconds() / 60
        # 考虑市场开放时间 (加入这个因素会更准确，但为简化暂时忽略)
        # 币安是24/7交易，所以直接计算
        return total_minutes / tf_minutes

    def _timeframe_to_minutes(self, timeframe):
        """将时间周期转换为分钟数"""
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 60 * 24
        return 0

    def _get_data_path(self, symbol: str):
        """获取数据保存路径，并在raw下为symbol创建子目录"""
        # 提取交易对的基础名称
        symbol_lower = symbol.lower()
        if "/" in symbol_lower:
            base_symbol = symbol_lower.split("/")[0]
        elif symbol_lower.endswith("usdt"):
            base_symbol = symbol_lower[:-4]
        else:
            base_symbol = symbol_lower

        # 确定根数据目录 - 修改这里
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(project_root, "data", "data_downloads", "raw")  # 修改后的路径

        # 检查根数据目录是否存在，如果不存在则创建
        if not os.path.exists(data_root):
            # 如果根目录不存在，尝试上一级（可能脚本在子目录运行）
            parent_root = os.path.dirname(project_root)
            # 尝试新的替代路径
            data_root_alt = os.path.join(parent_root, "data", "data_downloads", "raw")
            # 检查是否存在 data/data_downloads 目录
            if os.path.exists(os.path.join(parent_root, "data", "data_downloads")):
                data_root = data_root_alt
            # 如果还是找不到，则在当前工作目录下创建
            elif not os.path.exists(data_root):
                data_root = os.path.join(os.getcwd(), "data", "data_downloads", "raw")
            os.makedirs(data_root, exist_ok=True)

        # 构建特定交易对的子目录路径
        symbol_specific_path = os.path.join(data_root, base_symbol)

        # 确保子目录存在
        os.makedirs(symbol_specific_path, exist_ok=True)

        print(f"数据将保存到特定子目录: {os.path.abspath(symbol_specific_path)}")
        return symbol_specific_path

    def download_data(
        self,
        symbol,
        start_date,
        end_date,
        time_interval,
        retry_count=5,
        batch_size=500,
        delay_ms=500,
        output_file: str | None = None,
    ):
        """
        改进的数据下载方法，提供更详细的进度信息和更好的异常处理

        参数:
            symbol: 交易对
            start_date: 开始日期 datetime对象
            end_date: 结束日期 datetime对象
            time_interval: 时间周期
            retry_count: 失败重试次数
            batch_size: 每批获取的K线数量（自动优化）
            delay_ms: 请求间隔延迟(毫秒，自动优化)

        返回:
            DataFrame 包含OHLCV数据
        """
        # 🚀 根据交易所自动优化批次大小
        if self.exchange_name == "okx":
            # OKX历史数据限制：最多300根
            batch_size = min(batch_size, 300)
        elif self.exchange_name == "binance":
            # Binance可以支持1000根
            batch_size = min(batch_size, 1000)

        # 🚀 自动优化延迟（确保不触发限流）
        if delay_ms < 100:
            print(f"⚠️ 延迟过低({delay_ms}ms)，自动调整为100ms以避免限流")
            delay_ms = 100
        # 转换为时间戳 (毫秒)
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)

        print(f"\n📥 开始下载: {symbol} {time_interval}")
        print(f"⏰ 时间范围: {start_date} 至 {end_date}")
        print(
            f"🚀 优化参数: {batch_size}根/批, {delay_ms}ms延迟 (速度: ~{int(batch_size * 1000 / delay_ms * 60)}根/分钟)"
        )

        # 返回完整数据的列表
        all_candles = []

        # 当前时间戳
        current_since = since

        # 计数器
        total_fetched = 0
        batch_count = 0
        consecutive_failures = 0

        # 计算预期总批次
        expected_batches = self._estimate_batches(since, until, time_interval, limit=batch_size)
        print(f"预计需要获取约 {expected_batches} 批数据")

        # 增量写入控制
        header_written = False
        if output_file:
            header_written = os.path.exists(output_file)

        # 循环获取所有数据
        while current_since < until:
            retry = 0
            success = False
            no_data_this_round = False

            while retry < retry_count and not success:
                try:
                    if batch_count % 5 == 0 or batch_count == 0:
                        print(
                            f"请求批次 #{batch_count + 1}: 从 {datetime.fromtimestamp(current_since / 1000)}"
                        )
                    else:
                        print(".", end="", flush=True)  # 简化的进度显示

                    # 重新计算每批获取数量，确保最后一批不会超出结束时间
                    remaining_time = until - current_since
                    tf_ms = self._timeframe_to_ms(time_interval)
                    remaining_candles = remaining_time / tf_ms
                    current_limit = min(batch_size, int(remaining_candles) + 10)  # 额外获取几条以确保覆盖

                    # 获取K线数据
                    candles = self.exchange.fetch_ohlcv(
                        symbol=self._normalize_symbol(symbol),
                        timeframe=time_interval,
                        since=current_since,
                        limit=current_limit,
                    )

                    if not candles or len(candles) == 0:
                        print("\n没有获取到数据，可能已到达数据末尾或指定时间内无交易")
                        no_data_this_round = True
                        break

                    # 添加到总列表
                    all_candles.extend(candles)

                    # 更新计数
                    current_batch_size = len(candles)
                    total_fetched += current_batch_size
                    batch_count += 1
                    consecutive_failures = 0  # 重置连续失败计数

                    # 获取最后一个K线的时间作为下一批的开始
                    current_since = candles[-1][0] + 1  # +1毫秒避免重复

                    # 分批增量写入到CSV，减少长任务中的数据丢失风险
                    if output_file:
                        try:
                            batch_df = pd.DataFrame(
                                candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
                            )
                            batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], unit="ms")
                            write_mode = "a" if header_written else "w"
                            batch_df.to_csv(
                                output_file, index=False, mode=write_mode, header=not header_written
                            )
                            header_written = True
                        except Exception as werr:
                            print(f"增量写入失败: {werr}")

                    # 显示当前进度（每5批或累计超过特定数量时）
                    if batch_count % 5 == 0 or batch_count == 1:
                        datetime.fromtimestamp(candles[-1][0] / 1000)
                        progress = min(100, round((current_since - since) / (until - since) * 100))
                        print(
                            f"\n批次 {batch_count}/{expected_batches} ({progress}%): "
                            f"获取了 {current_batch_size} 条K线，累计: {total_fetched} 条"
                        )

                    # 添加延迟避免API限制
                    self.exchange.sleep(delay_ms)

                    # 标记成功
                    success = True

                except ccxt.AuthenticationError as auth_err:
                    retry += 1
                    print(f"\n获取数据时发生认证错误 (尝试 {retry}/{retry_count}): {auth_err}")
                    print("请检查API密钥权限和IP白名单设置。")
                    if retry >= retry_count:
                        print("认证错误达到最大重试次数，放弃数据获取。")
                        return None
                    self.exchange.sleep(2000)  # 等待2秒再试

                except ccxt.DDoSProtection as ddos_err:
                    retry += 1
                    wait_time = retry * 3  # 降低等待时长
                    print(f"\nDDoS保护机制触发 (尝试 {retry}/{retry_count}): {ddos_err}")
                    print(f"等待 {wait_time} 秒后重试...")
                    self.exchange.sleep(wait_time * 1000)

                except ccxt.ExchangeNotAvailable as not_avail_err:
                    retry += 1
                    wait_time = retry * 5  # 降低等待时长
                    print(f"\n交易所不可用 (尝试 {retry}/{retry_count}): {not_avail_err}")
                    print(f"等待 {wait_time} 秒后重试...")
                    self.exchange.sleep(wait_time * 1000)

                except ccxt.RequestTimeout as timeout_err:
                    retry += 1
                    wait_time = retry * 2  # 降低等待时长
                    print(f"\n请求超时 (尝试 {retry}/{retry_count}): {timeout_err}")
                    print(f"等待 {wait_time} 秒后重试...")
                    self.exchange.sleep(wait_time * 1000)

                except Exception as e:
                    retry += 1
                    print(f"\n获取数据出错 (尝试 {retry}/{retry_count}): {e}")
                    print(f"错误类型: {type(e).__name__}")

                    if retry < retry_count:
                        wait_time = retry * 5  # 递增等待时间
                        print(f"等待 {wait_time} 秒后重试...")
                        self.exchange.sleep(wait_time * 1000)
                    else:
                        print("达到最大重试次数")
                        # 如果是第一批次就失败，直接退出
                        if batch_count == 0:
                            print("❌ 第一批次就失败，可能是网络或配置问题，停止数据获取")
                            return None
                        else:
                            print("尝试跳过当前批次")
                            # 尝试向前推进时间戳
                            tf_ms = self._timeframe_to_ms(time_interval)
                            current_since += tf_ms * min(batch_size // 2, 10)  # 跳过一些K线，但不要太多
                            consecutive_failures += 1

            # 近尾部无数据的提前终止：如果接近结束且本轮无数据，直接跳出
            if no_data_this_round:
                tf_ms = self._timeframe_to_ms(time_interval)
                remaining_time = until - current_since
                if remaining_time <= tf_ms * 3:
                    print("接近时间范围末尾且无数据，提前结束循环")
                    break

            # 如果连续多次批次失败，暂停一段时间或减小批量大小
            if not success:
                consecutive_failures += 1

                if consecutive_failures >= 3:
                    print(f"\n警告: 连续 {consecutive_failures} 次批次获取失败")
                    print("可能是遇到了API限制或数据稀疏区域")

                    if consecutive_failures >= 5:
                        print("连续失败次数过多，暂停20秒后继续（保持批量大小不变）...")
                        self.exchange.sleep(20000)
                    else:
                        print("暂停10秒后继续...")
                        self.exchange.sleep(10000)

        if not all_candles:
            print("没有获取到任何数据")
            return None

        print(f"\n数据获取完成: 总共 {len(all_candles)} 条K线")

        # 将数据转换为DataFrame
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # 转换时间戳
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # 删除重复行
        df = df[~df.index.duplicated(keep="first")]

        # 排序
        df.sort_index(inplace=True)

        print(f"数据处理完成: {symbol} {time_interval}")
        print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"总记录数: {len(df)}")

        return df

    def _timeframe_to_ms(self, timeframe):
        """将时间周期转换为毫秒数"""
        minutes = self._timeframe_to_minutes(timeframe)
        return minutes * 60 * 1000

    def _estimate_batches(self, since, until, timeframe, limit):
        """估计需要的批次数量"""
        # 计算总时间范围(毫秒)
        time_range_ms = until - since

        # 计算单个时间周期的毫秒数
        tf_ms = self._timeframe_to_ms(timeframe)

        if tf_ms == 0:
            return 0

        # 估计K线总数
        estimated_candles = time_range_ms / tf_ms

        # 估计批次数
        estimated_batches = estimated_candles / limit

        return int(estimated_batches) + 1

    def fetch_data_for_days(self, symbol, timeframe, days):
        """
        获取最近N天的数据

        参数:
            symbol: 交易对
            timeframe: 时间周期
            days: 天数

        返回:
            保存好的数据文件路径
        """
        today = datetime.now()
        start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        return self.fetch_data(symbol, timeframe, start_date, end_date)

    def fetch_data_by_year(self, symbol, timeframe, year):
        """
        获取指定年份的数据

        参数:
            symbol: 交易对
            timeframe: 时间周期
            year: 年份，如2023

        返回:
            保存好的数据文件路径
        """
        start_date = f"{year}-01-01"

        # 如果是当前年份，则只获取到当前日期
        if year == datetime.now().year:
            end_date = datetime.now().strftime("%Y-%m-%d")
        else:
            end_date = f"{year}-12-31"

        return self.fetch_data(symbol, timeframe, start_date, end_date)

    def fetch_multi_timeframe_data(self, symbol, timeframes=None, start_date=None, end_date=None, days=365):
        """
        获取多个时间周期的数据

        参数:
            symbol: 交易对
            timeframes: 时间周期列表，默认使用所有支持的
            start_date: 开始日期，格式为"YYYY-MM-DD"，默认根据days参数计算
            end_date: 结束日期，格式为"YYYY-MM-DD"，默认为今天
            days: 如未指定开始日期，则获取最近days天的数据

        返回:
            保存好的数据文件路径列表
        """
        # 如果未指定时间周期，使用所有支持的
        if timeframes is None:
            timeframes = self.supported_timeframes

        # 设置结束日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 设置开始日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        print(f"\n=== 开始获取 {symbol} 的多时间周期数据 ===")
        print(f"时间周期: {', '.join(timeframes)}")
        print(f"时间范围: {start_date} 至 {end_date}")

        # 创建合理的数据获取计划 - 不同周期获取不同的历史长度
        fetch_plan = self._create_fetch_plan(timeframes, start_date, end_date)

        # 执行获取计划
        results = []
        for tf, dates in fetch_plan.items():
            tf_start = dates["start"]
            tf_end = dates["end"]
            print(f"\n开始获取 {tf} 周期数据: {tf_start} 至 {tf_end}")
            file_path = self.fetch_data(symbol, tf, tf_start, tf_end)
            if file_path:
                results.append(file_path)

        return results

    def update_base_csv(
        self,
        symbol: str,
        base_tf: str = "5m",
        output_dir: str | None = None,
        days_if_missing: int = 60,
        fill_missing: bool = True,
        initial_start_str: str | None = None,
        symbol_std_override: str | None = None,
    ):
        """
        增量更新任意基础周期数据（如 1m/3m/5m/15m 等），并与本地已有文件合并；
        下载过程中将批次数据直接追加写入目标 CSV（边下边存）。

        参数:
            symbol: 交易对，如 "ETH/USDT"
            base_tf: 基础周期，如 "1m" / "5m"
            output_dir: 输出目录，默认使用项目根目录下 data/data_downloads
            days_if_missing: 如果本地不存在文件，初次抓取的天数
            fill_missing: 是否按基础周期补齐时间网格
            initial_start_str: 初始起点(仅当本地无文件时生效)
            symbol_std_override: 🔥 外部传入的标准化币种名（用于文件命名），优先级最高

        返回:
            最终写入的 CSV 绝对路径
        """
        # 计算输出目录（受 main.io 控制；缺失时基于 base_dir 推导）
        if output_dir is None:
            try:
                from .congfigs.config_loader import ConfigLoader as _CL  # 相对导入
            except ImportError:
                from features_engineering.congfigs.config_loader import ConfigLoader as _CL  # 绝对导入
            _loader = _CL()
            _main_cfg = _loader.load_main_config() or {}
            _io = _main_cfg.get("io", {}) or {}
            _base_dir = _io.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
            output_dir = _io.get("downloads_dir") or f"{_base_dir}/rl_live/data_downloads"
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 🔥 目标文件名优先级：
        # 1. symbol_std_override（外挂传入，最高优先级）
        # 2. 从 symbol 参数推导（ETH/USDT → ETH_USDT）
        # 注意：不再从 main_config 读取，避免多币种下载时文件名混淆
        if symbol_std_override:
            symbol_for_filename = symbol_std_override
        else:
            # 从交易所格式推导：ETH/USDT:USDT → ETH_USDT
            symbol_for_filename = symbol.replace("/", "_").replace(":USDT", "").replace(":USD", "")
        
        market_tag = (self.market_type or "").upper() if isinstance(self.market_type, str) else ""
        filename = f"{symbol_for_filename}_{market_tag}_{base_tf}.csv"
        output_path = os.path.join(output_dir, filename)

        print(f"\n=== 更新 {base_tf} 聚合数据文件 ===")
        print(f"目标文件: {output_path}")

        # 尝试读取已存在的数据
        existing_df = None
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path, parse_dates=[0], index_col=0)
                existing_df.index.name = "timestamp"
                keep_cols = [
                    c for c in ["open", "high", "low", "close", "volume"] if c in existing_df.columns
                ]
                existing_df = existing_df[keep_cols]
                print(
                    f"已读取本地历史数据: {len(existing_df)} 条, 时间范围: {existing_df.index.min()} ~ {existing_df.index.max()}"
                )
            except Exception as e:
                print(f"读取本地文件失败，将重新构建: {e}")
                existing_df = None

        now_dt = datetime.now()

        # 解析初始起点字符串（在本地无历史文件时使用）
        def _parse_initial_start(s: str) -> datetime:
            for fmt in ("%Y-%m-%d-%H-%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    pass
            raise ValueError(f"无法解析 initial_start_str: {s}")

        # 计算开始抓取时间（向前回溯5个基础周期，避免缺口）
        base_minutes = self._timeframe_to_minutes(base_tf) or 1
        backtrack = timedelta(minutes=base_minutes * 5)
        step_delta = timedelta(minutes=base_minutes)

        if existing_df is not None and not existing_df.empty:
            last_ts = existing_df.index.max()
            start_dt = (last_ts + step_delta) - backtrack
            print(f"增量更新起点: {start_dt} (本地最后一条: {last_ts})")
        else:
            if initial_start_str:
                start_dt = _parse_initial_start(initial_start_str)
                print(f"本地无历史文件，按固定起点抓取，起点: {start_dt}")
            else:
                start_dt = now_dt - timedelta(days=days_if_missing)
                print(f"本地无历史文件，首次抓取近 {days_if_missing} 天数据，起点: {start_dt}")

        # 若起止无效则直接保存现有数据（可选补齐）
        if start_dt >= now_dt:
            print("本地数据已是最新，无需增量抓取")
            final_df = existing_df
        else:
            incr_df = self.download_data(
                symbol=symbol,
                start_date=start_dt,
                end_date=now_dt,
                time_interval=base_tf,
                output_file=output_path,
            )

            if incr_df is None or incr_df.empty:
                print("未获取到增量数据，使用本地已有数据")
                final_df = existing_df
            else:
                print(
                    f"获取到增量: {len(incr_df)} 条, 时间范围: {incr_df.index.min()} ~ {incr_df.index.max()}"
                )
                if existing_df is not None and not existing_df.empty:
                    final_df = pd.concat([existing_df, incr_df])
                else:
                    final_df = incr_df

        if final_df is None or final_df.empty:
            print("❌ 无可写入的数据")
            return None

        # 去重、排序
        final_df = final_df[~final_df.index.duplicated(keep="last")].sort_index()

        # 可选: 按基础周期补齐时间网格
        if fill_missing:
            # 将 base_tf 转换为 pandas 频率规则
            unit = base_tf.strip().lower()[-1]
            num = int(base_tf[:-1]) if base_tf[:-1].isdigit() else 1
            if unit == "m":
                freq = f"{num}min"
            elif unit == "h":
                freq = f"{num}h"
            elif unit == "d":
                freq = f"{num}d"
            elif unit == "w":
                freq = f"{num}w"
            else:
                freq = "1min"
            full_index = pd.date_range(start=final_df.index.min(), end=final_df.index.max(), freq=freq)
            final_df = final_df.reindex(full_index)
            final_df.index.name = "timestamp"

            if "close" in final_df.columns:
                final_df["close"] = final_df["close"].ffill()
            for col in ["open", "high", "low"]:
                if col in final_df.columns:
                    if "close" in final_df.columns:
                        final_df[col] = final_df[col].fillna(final_df["close"])
                    else:
                        final_df[col] = final_df[col].ffill()
            if "volume" in final_df.columns:
                final_df["volume"] = final_df["volume"].fillna(0.0)

        # 保存
        final_df.to_csv(output_path, index=True)
        print(f"✅ 已写入: {output_path}")
        print(f"最终时间范围: {final_df.index.min()} ~ {final_df.index.max()}  (共 {len(final_df)} 条)")

        # 使用统一的工具函数打印最新时间
        from tools.io_paths import print_latest_timestamp_from_df

        print_latest_timestamp_from_df(final_df)

        return output_path

    def _create_fetch_plan(self, timeframes, start_date, end_date):
        """
        创建合理的数据获取计划，为不同周期设置不同的历史长度

        参数:
            timeframes: 时间周期列表
            start_date: 开始日期，格式为"YYYY-MM-DD"
            end_date: 结束日期，格式为"YYYY-MM-DD"
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # 创建获取计划字典
        plan = {}

        # 为不同时间周期设置不同的历史长度
        for tf in timeframes:
            tf_start = start_dt

            # 小周期减少历史长度，减轻存储压力
            if tf == "1m":
                # 1分钟数据最多获取30天
                days_to_fetch = min(30, (end_dt - start_dt).days)
                tf_start = end_dt - timedelta(days=days_to_fetch)
            elif tf == "3m":
                # 3分钟数据最多获取60天
                days_to_fetch = min(60, (end_dt - start_dt).days)
                tf_start = end_dt - timedelta(days=days_to_fetch)
            elif tf == "5m":
                # 5分钟数据最多获取90天
                days_to_fetch = min(90, (end_dt - start_dt).days)
                tf_start = end_dt - timedelta(days=days_to_fetch)
            elif tf == "15m" or tf == "30m":
                # 15和30分钟数据最多获取180天
                days_to_fetch = min(180, (end_dt - start_dt).days)
                tf_start = end_dt - timedelta(days=days_to_fetch)

            plan[tf] = {"start": tf_start.strftime("%Y-%m-%d"), "end": end_date}

        return plan

    def fetch_all_timeframes(self, symbol, years=2):
        """
        按照最优实践获取所有时间周期的数据

        参数:
            symbol: 交易对
            years: 获取多少年的数据

        返回:
            保存好的数据文件路径列表
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

        # 为每个时间周期创建专门的获取计划
        plan = {
            "1m": {
                "start": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end": end_date,
            },
            "3m": {
                "start": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
                "end": end_date,
            },
            "5m": {
                "start": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                "end": end_date,
            },
            "15m": {
                "start": (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
                "end": end_date,
            },
            "30m": {
                "start": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "end": end_date,
            },
            "1h": {"start": start_date, "end": end_date},
            "4h": {"start": start_date, "end": end_date},
            "1d": {"start": start_date, "end": end_date},
        }

        results = []
        for tf, dates in plan.items():
            print(f"\n开始获取 {symbol} {tf} 数据: {dates['start']} 至 {dates['end']}")
            file_path = self.fetch_data(symbol, tf, dates["start"], dates["end"])
            if file_path:
                results.append(file_path)

        return results


def run_step1_default(
    days: int = 200,
    exchange_name: str | None = None,
    symbol_exchange: str | None = None,
    market_type: str | None = None,
    base_tf: str | None = None,
    downloads_dir: str | None = None,
) -> None:
    """默认下载最近 days 天到 main.io.downloads_dir，并按 base_download 命名补齐聚合CSV。"""
    from features_engineering.congfigs.config_loader import ConfigLoader

    loader = ConfigLoader()
    main_cfg = loader.load_main_config() or {}

    exchange_name = exchange_name or (main_cfg.get("exchange", {}) or {}).get("name", "binance")
    symbol_exchange = symbol_exchange or (main_cfg.get("symbol", {}) or {}).get("trading_pair_exchange", "ETH/USDT")
    # 🔥 从 main_config 读取 trading_pair_std 用于文件命名
    symbol_std = (main_cfg.get("symbol", {}) or {}).get("trading_pair_std", None)
    market_type = market_type or (main_cfg.get("symbol", {}) or {}).get("market_type", "swap")
    base_tf = base_tf or (main_cfg.get("timeframes", {}) or {}).get("base_download", "1m")
    _io = main_cfg.get("io", {}) or {}
    if downloads_dir:
        downloads_dir = os.path.abspath(downloads_dir)
    else:
        _base_dir = _io.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
        downloads_dir = _io.get("downloads_dir") or f"{_base_dir}/rl_live/data_downloads"

    print(
        f"\n🚀 Step1 默认模式: 下载最近 {days} 天 | {exchange_name} {symbol_exchange} ({market_type}) {base_tf}"
    )
    fetcher = EnhancedCCXTProcessor(exchange_name=exchange_name, market_type=market_type)
    # 🔥 默认模式：从 main_config 读取 trading_pair_std 控制文件名
    fetcher.update_base_csv(
        symbol=symbol_exchange,
        base_tf=base_tf,
        output_dir=downloads_dir,
        days_if_missing=days,
        fill_missing=True,
        initial_start_str=None,
        symbol_std_override=symbol_std,  # 🔥 从 main_config 读取
    )


def run_step1_with_override(
    days: int = 280,
    symbol_std: str = "ETH_USDT",
    symbol_exchange: str = "ETH/USDT",
    exchange_name: str = "okx",
    market_type: str = "swap",
    base_tf: str = "1m",
    downloads_dir: str | None = None,
) -> None:
    """
    带完整参数覆盖的 Step1 下载（供每日调度器调用）
    
    Args:
        days: 下载最近多少天
        symbol_std: 标准化币种名（用于文件命名），如 "ETH_USDT"
        symbol_exchange: 交易所格式币种名，如 "ETH/USDT"
        exchange_name: 交易所名称，如 "okx" / "binance"
        market_type: 市场类型，如 "swap" / "spot"
        base_tf: 基础时间周期，如 "1m"
        downloads_dir: 输出目录（可选，默认从 main_config 读取）
    """
    from features_engineering.congfigs.config_loader import ConfigLoader

    loader = ConfigLoader()
    main_cfg = loader.load_main_config() or {}

    # 确定输出目录
    if downloads_dir:
        downloads_dir = os.path.abspath(downloads_dir)
    else:
        _io = main_cfg.get("io", {}) or {}
        _base_dir = _io.get("base_dir") or os.path.join(os.path.expanduser("~"), "FinRL_bn", "data")
        downloads_dir = _io.get("downloads_dir") or f"{_base_dir}/rl_live/data_downloads"

    print(f"\n🚀 Step1 覆盖模式: {symbol_std} ({symbol_exchange})")
    print(f"   交易所: {exchange_name} | 市场: {market_type} | 周期: {base_tf} | 天数: {days}")
    print(f"   输出目录: {downloads_dir}")

    fetcher = EnhancedCCXTProcessor(exchange_name=exchange_name, market_type=market_type)
    
    # 🔥 关键：使用 symbol_std_override 控制文件名，与 main_config 完全解耦
    fetcher.update_base_csv(
        symbol=symbol_exchange,
        base_tf=base_tf,
        output_dir=downloads_dir,
        days_if_missing=days,
        fill_missing=True,
        initial_start_str=None,
        symbol_std_override=symbol_std,  # 🔥 外挂传入的标准化名称
    )


# 如果直接运行此脚本：默认下载最近200天基础周期并补齐写回
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📥 Step1 默认下载（最近200天）")
    print("=" * 60)
    run_step1_default(days=200)
