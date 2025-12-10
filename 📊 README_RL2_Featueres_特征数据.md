# 📊 RL 特征数据规格说明

---

## 1. 设计目标

- **极简 & 工业化**：保留 Step4 融合后的“原味”列名，通过轻量归一化让特征既稳定又易于追踪。  
- **上下游统一**：同一份特征结构服务于离线训练、预检门禁、实盘策略。  
- **Transformer 友好**：明确特征分组和含义，方便多头注意力捕捉不同维度。  
- **无信息泄漏**：严格使用历史数据，所有时间戳都来自训练 CSV / 统一 pipeline。

---

## 2. 输出概览（完整版：45 维）

### 2.1 数据集规格

| 数据集 | 路径 | 特征数 | 样本数 | 用途 |
|--------|------|--------|--------|------|
| **离线训练集** | `data/rl_live/data_ready/ETH_USDT_3m_rl_features.npz` | **45维** | ~75万 | RL模型训练、策略优化 |
| **在线预检集** | `data_trading/preflight_features/preflight_rl_features.npz` | **29维** | 2000 | 实盘前特征校验 |

> ⚠️ **关键差异**：离线训练使用完整45维特征（包含ATR、RSI等），在线预检使用精简29维（去除部分冗余特征以提高实时性能）

### 2.2 完整特征分组（45维详解）

| 分组 | 代表列 | 数量 | 取值范围 | 说明 |
|------|--------|------|----------|------|
| 🎯 **市场状态** `market_state` | `3m_market_state` | 4 | `{-1, 1}` | SuperTrend方向，离散信号 |
| 📈 **动量** `momentum` | `3m_mom` | 4 | `[-1, 1]` | 真滑窗重算，连续平滑 |
| 📊 **波动带宽** `band_width` | `3m_bb_width` | 4 | `[0.03, 0.97]` | 布林带宽度，log归一化 |
| 📢 **成交量** `volume` | `3m_volume` | 4 | `[0, 1]` | 滚动分位归一化 |
| 💥 **ATR百分比** `atr_pct` | `3m_atr_pct` | 4 | `[0, ∞)` | 平均真实波幅/价格 |
| 🌊 **已实现波动率** `rv` | `3m_rv` | 4 | `[0, 0.1]` | 对数收益率方差 |
| 🎚️ **RSI系列** `rsi_continuous` | `3m_rsi` | 12 | `[-1, 1]` | 连续RSI + 事件标签（超买/超卖） |
| ⏰ **时间编码** `time_encoding` | `time_day_sin` | 4 | `[-1, 1]` | 周期性时间特征（sin/cos） |
| 💰 **价格** `price` | `3m_close` | 4 | 原始价格 | OHLC四价 |
| 📉 **收益率** `return` | `ret_3m_log` | 1 | `[-1, 1]` | 对数收益率 |

**总计：45 维特征 = 4×7 + 12 + 4 + 4 + 1**

### 2.3 周期分布（4个时间尺度）

每个指标覆盖 4 个周期：`3m`, `15m`, `30m`, `2h`

```
3m  (基础周期) ━━━━━━━━━━━━━━━► 高频信号
15m (短期趋势) ━━━━━━━━━━━━━━━► 短线波段
30m (中期趋势) ━━━━━━━━━━━━━━━► 日内趋势
2h  (长期趋势) ━━━━━━━━━━━━━━━► 宏观方向
```

### 2.4 NPZ 文件结构

```python
{
    'version': '1.0',
    'observations': (753640, 45),      # 特征矩阵
    'feature_names': [...],            # 特征名称列表（45个）
    'feature_groups': [...],           # 特征分组标签（45个）
    'timestamps': (753640,),           # Unix时间戳（毫秒）
    'prices': (753640, 4),             # OHLC价格矩阵
    'schema_sha': '...',               # 特征列表哈希校验
    'metadata': {...}                  # 生成时间、周期等元信息
}
```

> ✅ 具体列名与顺序可以在 Step5 summary 文件或预检输出中查看；如需增删列，请同步更新 `step5_mapping.yaml` 与预检门禁。

---

## 3. 特征分组与处理细则

### 3.1 市场状态 `market_state`
- 来源：`{period}_supertrend_direction`、`EMA7/EMA55`。  
- 规则：`{period}_market_state ∈ {-1, 1}`。  
- 用途：给策略快速提供趋势标签（涨/跌），统一训练与实盘判断逻辑。

### 3.2 动量 `momentum`
- 来源：`{period}_mom20`。  
- 真滑窗：每 3 分钟用最新 `3m_close` 重算所有大周期动量，避免阶跃。  
- 归一化：Step5 内部做滚动 Z-score（或根据配置直接保留原值）。

### 3.3 波动 `band_width`
- 处理流程：`log1p` → 全局 2%-98% 分位截断 → 组合快/慢分位 → 平滑 → 缩放至 `[0.03,0.97]`。  
- 意义：去掉过多的 0/1 粘边，呈现更稳定的波动度刻画。

### 3.4 成交量 `volume`
- 处理：`log1p` + 滚动分位数归一化到 `[0,1]`。  
- 语义：低量能≈0，高量能≈1。

### 3.5 RSI `rsi_continuous`（12维）
每个周期（3m/15m/30m）包含4个子特征：
- **`{period}_rsi`**：连续值 `(rsi - 50) / 50` 裁剪到 `[-1,1]`
- **`{period}_rsi_event`**：RSI事件标记（是否触发超买/超卖）
- **`{period}_rsi_overbought`**：超买标志（RSI ≥ 70）
- **`{period}_rsi_oversold`**：超卖标志（RSI ≤ 30）

**用途**：提供动量震荡指标的连续值 + 离散事件标签，方便规则系统直接使用

**注意**：2h周期无RSI特征（周期过长，RSI信号滞后）

### 3.6 ATR百分比 `atr_pct`（4维）
- 处理：`ATR / close_price`，表示波动幅度占价格的百分比
- 取值：通常 `[0, 0.1]`，极端行情可能更高
- 用途：衡量市场波动强度，辅助仓位管理

### 3.7 已实现波动率 `rv`（4维）
- 计算：对数收益率的滚动标准差
- 取值：通常 `[0, 0.1]`
- 用途：评估实际波动水平，与ATR互补

### 3.8 时间编码 `time_encoding`（4维）
- **格式**：`sin/cos` 周期性编码（不是整数）
- **特征**：
  - `time_day_sin`, `time_day_cos`：星期几（7天周期）
  - `time_hour_sin`, `time_hour_cos`：小时（24小时周期）
- **取值范围**：`[-1, 1]`
- **用途**：捕捉交易时间的周期性规律（如周末效应、交易时段）

> ⚠️ 注意：与文档早期版本不同，当前使用 sin/cos 编码（便于RL模型学习），而非整数编码

### 3.9 价格基准 `price`（4维）
- 保留完整OHLC：`3m_open`, `3m_high`, `3m_low`, `3m_close`
- 原始价格（未归一化），用于计算收益、回测等
- 仅基础周期（3m），其他周期通过滑窗计算

### 3.10 收益率 `return`（1维）
- **`ret_3m_log`**：对数收益率 `log(close_t / close_{t-1})`
- 取值范围：理论 `(-∞, +∞)`，实际约 `[-0.05, 0.05]`
- 用途：计算实际收益、构建奖励函数

---

## 4. 真滑窗（Real Sliding Window）详解

> 目的：让大周期动量在基础周期上“连续滚动”，避免旧方案的阶梯式跳变。

旧方案（静态填充）：
```text
12:00 ━━ 15m_mom = 0.14 (新K线)
12:03 ━━ 15m_mom = 0.14 (填充)  ← 数值不变
12:06 ━━ 15m_mom = 0.14 (填充)
12:09 ━━ 15m_mom = 0.14 (填充)
12:12 ━━ 15m_mom = 0.14 (填充)
12:15 ━━ 15m_mom = 0.40 (新K线)  ← 突然跳变
```

新方案（真滑窗）：
```text
12:00 ━━ 15m_mom = 0.14 (用 12:00 的 3m_close 重算)
12:03 ━━ 15m_mom = 0.16 (用 12:03 的 3m_close 重算)
12:06 ━━ 15m_mom = 0.19 (用 12:06 的 3m_close 重算)
12:09 ━━ 15m_mom = 0.22 (用 12:09 的 3m_close 重算)
12:12 ━━ 15m_mom = 0.25 (用 12:12 的 3m_close 重算)
12:15 ━━ 15m_mom = 0.40 (新 15m K 线)
```

- 实现位置：`step4_merge_features.py` 末尾（持久化前）。  
- 配置开关：`main_config.yaml.merge.enable_real_sliding`（默认开启）。  
- 支持指标：当前仅 `mom`；未来可扩展 `bb_width` 等。

关闭方式：
```yaml
merge:
  enable_real_sliding: false
```

---

## 5. 数据质量与验收

- Step5 自动输出 summary：均值/方差/越界比例/极值。  
- 预检阶段门禁：  
  - 特征 MAE < `0.02`  
  - MaxAbs < `0.05`  
  - Corr > `0.98`  
  - 动量特征 MAE 需 < `1e-6`（离线/在线一致性）  
- 监控：  
  - `data_trading/monitoring/probes/feature_snapshot.csv`（上线后）  
  - `data_trading/preflight_features/preflight_rl_features_summary.txt`

---

## 6. 使用示例

### 6.1 离线训练数据加载（45维）

```python
import numpy as np

path = "/root/FinRL_bn/data/rl_live/data_ready/ETH_USDT_3m_rl_features.npz"
with np.load(path, allow_pickle=True) as z:
    features = z["observations"]          # (753640, 45)
    names = z["feature_names"].tolist()   # 45个特征名
    groups = z["feature_groups"]          # 45个分组标签
    timestamps = z["timestamps"]          # Unix时间戳（毫秒）
    prices = z["prices"]                  # (753640, 4) OHLC

print(f"特征数: {len(names)}")            # 45
print(f"样本数: {features.shape[0]:,}")   # 753,640
print(f"时间范围: {timestamps[0]} ~ {timestamps[-1]}")
```

### 6.2 在线预检数据加载（29维）

```python
path = "/root/FinRL_bn/data_trading/preflight_features/preflight_rl_features.npz"
with np.load(path, allow_pickle=True) as z:
    features = z["observations"]          # (2000, 29)
    names = z["feature_names"].tolist()   # 29个特征名
    
print(f"预检特征数: {len(names)}")        # 29
print(f"预检样本数: {features.shape[0]}")  # 2000
```

### 6.3 按分组提取特征

```python
# 提取所有RSI相关特征
rsi_features = []
rsi_indices = []
for i, (name, group) in enumerate(zip(names, groups)):
    if 'rsi' in group.lower():
        rsi_features.append(name)
        rsi_indices.append(i)

print(f"RSI特征: {rsi_features}")
print(f"RSI索引: {rsi_indices}")

# 提取RSI特征矩阵
rsi_data = features[:, rsi_indices]
```

### 6.4 策略优化器使用

```python
from rl_long.strategy_optimizer.core.optimal_strategy_finder import OptimalStrategyFinder

# 策略优化器会自动加载特征
finder = OptimalStrategyFinder(config_path="param_grids/main_strategy.yaml")
strategy = finder.find_optimal_strategy()

print(f"使用特征: {len(finder.feature_names)}个")
print(f"总收益: {strategy.total_return*100:.2f}%")
```

**关键说明：**
- 观测向量即状态向量（`observations == states`），无需二次拼装
- 特征顺序固定，训练和测试必须一致
- 预检阶段会自动校验特征列表哈希（`schema_sha`）

---

## 7. 常见问题

**Q1. 离线45维 vs 在线29维，为什么不一致？**  
- 离线训练：保留所有特征（ATR、RSI全集），用于模型学习完整市场信息
- 在线预检：精简到29维（去除冗余），提高实时计算速度
- **策略优化器**：自动使用离线45维（需要完整特征）
- **实盘交易**：使用在线29维（性能优先）

**Q2. 为什么最近测试 `2h_market_state` 全是 -1.0？**  
**这不是bug！是市场真实走势！** 已验证（2025-11-18）：

**全局数据正常**（75万条，5个月）：
- 正值比例: 49.28%
- 负值比例: 50.72%
- 唯一值: [-1, 1]
- ✅ 数据质量良好

**最近5天异常**：
- 转负时间: 2025-11-11 20:00（持续5.6天全负）
- 价格跌幅: -8.39%（3485 → 3192）
- 其他周期: 3m(53%正), 15m(35%正), 30m(50%正)
- ✅ 符合2小时趋势判断逻辑

**为什么策略优化器测试碰到全负？**
- 默认配置使用"最近5天"数据（`recent_days: 5`）
- 刚好撞上这5.6天持续下跌期
- 触发"趋势反转强制平仓"规则 → 收益0%

**解决方案**：
1. ✅ **已完成**：禁用过度保守的硬规则（`trend_reversal_exit`）
2. ✅ **推荐**：扩大测试周期到30天（`recent_days: 30`）
3. 🔧 **规则优化**：改用"连续N天负值"而非"单次负值"触发平仓

**成交量特征bug**（真实问题）：
- ❌ `15m_volume`, `30m_volume`, `2h_volume`: std=0.000
- 原因：成交量未参与真滑窗重算（仅动量参与）
- 临时方案：优先使用 `3m_volume`

**Q3. 想新增特征/窗口？**  
在 `base_indicators.yaml`、`step5_mapping.yaml` 调整，重新跑 `run2_offline_pipeline.py + 四阶段预检`。

**Q4. 在线/离线动量 MAE ≈ 1e-3？**  
说明仍在使用旧滑窗或缓存脏数据，请确认真滑窗开启、缓存清理、预检重复执行。

**Q5. RSI事件标签如何使用？**  
```python
# 检测超买信号
if features[t, names.index('15m_rsi_overbought')] == 1.0:
    # 触发超买事件
    pass
```

---

## 8. 术语速查

- **真滑窗**：Real Sliding Window，在基础周期上重算大周期指标
- **save_n**：预检/在线模式一次生成的特征条数
- **microbatch_length**：在线追新时每批对齐的窗口长度
- **schema_sha**：特征列名列表的哈希，用于契约验证（确保离线/在线特征一致）
- **observations**：特征矩阵（`shape=(样本数, 特征数)`），即RL的状态向量
- **feature_groups**：特征分组标签，用于分类管理和可视化

---

## 9. 完整特征列表速查

### 离线训练集（45维）

```
# 市场状态 (4)
3m_market_state, 15m_market_state, 30m_market_state, 2h_market_state

# 动量 (4)
3m_mom, 15m_mom, 30m_mom, 2h_mom

# 波动带宽 (4)
3m_bb_width, 15m_bb_width, 30m_bb_width, 2h_bb_width

# 成交量 (4)
3m_volume, 15m_volume, 30m_volume, 2h_volume

# ATR百分比 (4)
3m_atr_pct, 15m_atr_pct, 30m_atr_pct, 2h_atr_pct

# 已实现波动率 (4)
3m_rv, 15m_rv, 30m_rv, 2h_rv

# RSI系列 (12)
3m_rsi, 3m_rsi_event, 3m_rsi_overbought, 3m_rsi_oversold,
15m_rsi, 15m_rsi_event, 15m_rsi_overbought, 15m_rsi_oversold,
30m_rsi, 30m_rsi_event, 30m_rsi_overbought, 30m_rsi_oversold

# 时间编码 (4)
time_day_sin, time_day_cos, time_hour_sin, time_hour_cos

# 价格 (4)
3m_open, 3m_high, 3m_low, 3m_close

# 收益率 (1)
ret_3m_log
```

### 在线预检集（29维）

精简版，去除 ATR、部分RSI标签，保留核心特征

---

> 维护：QBot Team
> 更新时间：2025-11-18  
> 如需了解数据流水线整体架构，请参考《💹 features_engineering 总览》。
