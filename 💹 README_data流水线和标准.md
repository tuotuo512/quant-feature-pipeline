# 💹 features_engineering 总览

> 目标：把“训练 → 预检 → 实盘”三条链路的特征逻辑全部收敛到统一流水线，做到**一个配置源、一个特征工厂、一个预检出口**。

---

## 1. 模块定位 & 目录结构

```text
/root/FinRL_bn/features_engineering/
├── congfigs/                     # ✅ 唯一配置源（主/分步 YAML + 配置加载器）
│   └── config_loader.py          # 配置装载与推导
├── unified_feature_pipeline.py   # 统一特征工厂入口 generate_rl_features(...)
├── run1_step1_data.py            # 下载 1m 基础数据（阶段性验证）
├── run2_offline_pipeline.py      # Step2~5 离线流水线 CLI 包装，会按顺序执行 Step2→Step3→Step4→Step5，跑完一次完整的特征生成
├── run3_featueres_unified.py     # Step5 导出 + 契约校验（离线） 主要验证用的！！
├── step{2,3,4,5}_*.py            # 各 Step 纯逻辑实现
├── tools/                        # 共用工具（IO、补齐、真滑窗等）
├── auto_features_daily/          # 🔌 独立外挂模块，每日自动更新数据
└── docs (本文件 + 特征细节 README)
```

- **核心原则**  
  - **配置唯一**：所有路径、周期、窗口都从 `congfigs/*.yaml` 读取，不再散落在代码里。  
  - **流水线统一**：`unified_feature_pipeline.generate_rl_features()` 是训练/预检/实盘的唯一入口。  
  - **产物标准**：输出 Parquet（中间态）+ NPZ（最终契约），命名遵循 `main_config.yaml.io.filename_patterns`。  
  - **日志透明**：Step5 与预检阶段会自动生成 summary / health report，记录于 `data/` 或 `data_trading/`。

---

## 2. 数据流水线

```text
📥 Step1  下载 1m 基础数据  →  data/rl_live/data_downloads/ETH_USDT_SWAP_1m.csv
📊 Step2  重采样多周期      →  data/rl_live/kline/ETH_USDT_{tf}.parquet
📈 Step3  指标计算          →  data/rl_live/ind/ETH_USDT_{tf}_indicators.parquet
🔗 Step4  融合多周期        →  data/rl_live/merged/ETH_USDT_3m_merged.parquet
🤖 Step5  RL 特征导出       →  data/rl_live/data_ready/ETH_USDT_3m_rl_features.npz
```

- Step2/3/4 默认全量；可通过 CLI 参数或配置开启增量 / 指定时间段。  
- 真滑窗（Real Sliding Window）在 Step4 中完成，用基础周期 `close` 重算大周期动量，避免阶跃（详细说明见《📊 README_RL2_Featueres_特征数据.md》）。  
- Step5 由 `UnifiedFeatureEngine` 输出 29 维极简特征，并自动做健康检查（市场状态、动量、波动、量能等）。

### Step4 原始特征 = 经验池唯一真相源

- 训练端、实盘端、预检体验池全部直接消费 Step4 merged 的 **141 维原始特征矩阵**（未归一化）。  
- Step5 `*_rl_features.npz` 仅供 RL 模型使用；经验仓位、市场模式、人工规则都依赖 Step4 原始尺度。  
- `rl_long.data_processor` 在切分数据时会校验 raw_features ↔ Step5 行数、时间戳严格对齐，缺失即中止。  
- `HierarchicalExperienceRepository` 内置守卫：若接收到的特征向量有效占比 < 50% 或全为 NaN，将抛出 `ExperienceDataError`。  
- 任何时候新增/修改特征，都只需要更新 Step4/Step5 流水线与配置；经验池和实盘只需重新拉取即可。

---

## 3. 配置体系（`congfigs/`）

| 文件 | 作用 | 关键字段 |
|------|------|----------|
| `main_config.yaml` | 全局唯一配置 | `timeframes`, `io`, `merge`, `online/preheat` |
| `step1_data download.yaml` | 下载策略 | API、起止时间、缺失补齐 |
| `step2_resample.yaml` | 重采样参数 | 目标周期、填充策略、输出格式 |
| `base_indicators.yaml` | 指标窗口 | 动量/RSI/BBands 等（被 ConfigLoader 合并） |
| `step4_merge.yaml` | 融合行为 | 对齐方式、缺失填充、真滑窗开关 |
| `step5_mapping.yaml` | Step5 特征映射 | 特征分组、归一化、默认窗口 |

> **注意**：在线特征与离线训练共用同一份配置。若需要临时覆盖，请通过 `live_trading/config/live_overrides.yaml` 的 `preheat` / `features_contract` 段落注入，不要直接改代码。

---

## 📊 真滑窗说明

真滑窗（Real Sliding Window）的原理、配置以及效果对比，已并入《📊 RL 特征数据规格说明》的相关章节。请在主文档中维护和查阅，避免内容重复。

- 配置入口：`main_config.yaml.merge.enable_real_sliding`
- 实现位置：`features_engineering/step4_merge_features.py`、`tools/real_sliding_simple.py`
- 验收门禁：预检阶段的动量一致性（MAE / MaxAbs / Corr）

如需扩展到新的指标或调整行为，请先更新主 README，再同步代码与预检阈值。本文件仅保留索引作用。谢谢 🙏。

---

## 4. 运行入口

```bash
# ① 下载基础数据（可选）
python features_engineering/run1_step1_data.py --exchange okx --symbol ETH/USDT --days 90

# ② 离线流水线（Step2~5 全量）
python features_engineering/run2_offline_pipeline.py --sample_ratio 0.01

# ③ 仅导出 Step5（复用已有 merged）
python features_engineering/run3_featueres_unified.py --use-existing-merged

# ④ 统一入口（供其他模块直接调用）
python - <<'PY'
from features_engineering.unified_feature_pipeline import generate_rl_features
cfg = {
    "mode": "offline",
    "symbol": "ETH_USDT",
    "base_period": "3m",
    "output_dir": "/root/FinRL_bn/data/rl_live/data_ready"
}
generate_rl_features(cfg)
PY
```

运行时，所有路径会通过 `tools/io_paths.py` 自动解析，无需手动拼接。

---

## 5. 与预检 / 实盘的联动

```js
features_engineering.generate_rl_features(mode='online')  ← 预检阶段 1 & 实盘追新调用
│
├── live_trading/preflight/run_preflight_seed.py      # 阶段1：生成 + 对比训练NPZ
├── live_trading/preflight/run_preflight_experience.py # 阶段2：经验池对比
├── live_trading/preflight/run2_preflight_model.py     # 阶段3：模型推理对比
└── live_trading/preflight/run3_preflight_full.py       # 阶段4：完整流水线
```

- 阶段1 会生成 `data_trading/preflight_features/preflight_rl_features.npz`，供阶段2/3/4 共用。  
- 实盘运行 (`runner/living_pipeline.py`) 也直接调用 `generate_rl_features(mode='online')`，保证线上线下一致。  
- 详细预检说明参见《live_trading/preflight/🎨 预检架构.MD》。

---

## 6. 开发 & 升级流程（精简版）

1. **实验**：在单独分支 / 临时工厂验证，修改 `congfigs/*.yaml` 并记录变更。  
2. **离线验证**：运行 `run2_offline_pipeline.py`，确认新的配置/逻辑可产出期望 NPZ。  
3. **预检四阶段**：依次执行 `run_preflight_seed.py → run_preflight_experience.py → run2_preflight_model.py → run3_preflight_full.py`，门禁全部通过（指标见预检文档）。  
4. **上线**：实盘 dry-run 或直接启动，重点关注 `data_trading/monitoring/probes` 下的监控产物。  
5. **记录**：在两个 README 中更新新增特征或流程说明，确保团队一致理解。

---

## 7. 常见问题

### Q1. Step3 提示 “未找到 K 线文件”？

```text
❌ 未找到K线文件: /root/FinRL_bn/data/rl_live/kline/ETH_USDT_3m.parquet
```

排查步骤：

1. `ls -lh data/rl_live/kline/*.parquet` 确认文件存在。  
2. 文件名应为 `ETH_USDT_3m.parquet`（无 `_SWAP` 后缀）。  
3. `main_config.yaml.symbol.trading_pair_std` 必须等于 `ETH_USDT`。

> 2025-11-04 已修复 `read_kline()` 的优先级与异常处理，若仍报错先清理旧缓存再重试。

### Q2. 离线与在线特征存在 0.05 以上差异？

- 检查 `base_indicators.yaml` 与 `step5_mapping.yaml` 的窗口是否与训练同步。  
- 确认 `live_overrides.yaml.preheat` 中 `save_n / microbatch_length / min_lookback` 与 `main_config.yaml` 一致。  
- 排除缓存脏数据后重跑阶段1预检；若仍超阈值，可提高 `save_n` 或放宽预检门限。

### Q3. 想关闭真滑窗？

```yaml
merge:
  enable_real_sliding: false
```

设置后重新生成 Step4/Step5 产物即可（不推荐，除非做回溯对比）。

---

## 8. 资料索引

- 《📊 README_RL2_Featueres_特征数据.md》：详解 29 维特征组成、真滑窗原理、归一化策略。  
- 《live_trading/preflight/🎨 预检架构.MD》：预检四阶段的门禁与流程。  
- `data_trading/monitoring/probes/`：在线监控输出（缓存追新、特征分布、执行阻断等）。
