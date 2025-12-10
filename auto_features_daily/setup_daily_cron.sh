#!/bin/bash
# ========================================
# 🔌 每日数据更新外挂 - Cron 安装脚本
# 用途: 一键配置 cron 定时任务
# ========================================

set -e

# 配置（自动检测路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONDA_ENV="finrl_ml_env"
CONDA_PATH="/root/miniconda3"
LOG_DIR="${PROJECT_ROOT}/logs/auto_features_daily"
PYTHON_BIN="${CONDA_PATH}/envs/${CONDA_ENV}/bin/python"
SCRIPT_PATH="${SCRIPT_DIR}/run_daily_features.py"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "🔌 FinRL 每日数据更新外挂 - Cron 安装"
echo "========================================"
echo "📁 项目根目录: $PROJECT_ROOT"
echo "📄 调度脚本: $SCRIPT_PATH"

# 检查 Python 环境
if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${RED}❌ Python 环境不存在: $PYTHON_BIN${NC}"
    echo "请先创建 conda 环境: conda create -n $CONDA_ENV python=3.10"
    exit 1
fi
echo -e "${GREEN}✅ Python 环境: $PYTHON_BIN${NC}"

# 检查脚本
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}❌ 调度脚本不存在: $SCRIPT_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 调度脚本存在${NC}"

# 创建日志目录
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✅ 日志目录: $LOG_DIR${NC}"

# 生成 cron 条目
# 每日 00:10 UTC (08:10 北京时间)
CRON_DAILY="10 0 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${SCRIPT_PATH} >> ${LOG_DIR}/cron_\$(date +\\%Y\\%m\\%d).log 2>&1"

# 开机自动运行
CRON_BOOT="@reboot sleep 60 && cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${SCRIPT_PATH} --boot >> ${LOG_DIR}/boot_\$(date +\\%Y\\%m\\%d).log 2>&1"

echo ""
echo "========================================"
echo "📋 将添加以下 Cron 任务:"
echo "========================================"
echo ""
echo -e "${YELLOW}1. 每日定时 (00:10 UTC / 08:10 北京):${NC}"
echo "   $CRON_DAILY"
echo ""
echo -e "${YELLOW}2. 开机自动 (@reboot):${NC}"
echo "   $CRON_BOOT"
echo ""

# 询问确认
read -p "是否继续安装? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 获取当前 crontab
CURRENT_CRON=$(crontab -l 2>/dev/null || echo "")

# 检查是否已存在
if echo "$CURRENT_CRON" | grep -q "auto_features_daily"; then
    echo -e "${YELLOW}⚠️ 检测到已有相关 cron 任务${NC}"
    read -p "是否覆盖? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
    # 移除旧条目
    CURRENT_CRON=$(echo "$CURRENT_CRON" | grep -v "auto_features_daily")
fi

# 添加新条目
NEW_CRON="${CURRENT_CRON}
# 🔌 FinRL 每日数据更新外挂 (00:10 UTC)
${CRON_DAILY}
# 🔌 FinRL 开机自动更新外挂
${CRON_BOOT}
"

# 安装 crontab
echo "$NEW_CRON" | crontab -

echo ""
echo -e "${GREEN}========================================"
echo "✅ Cron 任务安装成功!"
echo "========================================${NC}"
echo ""
echo "查看当前 cron: crontab -l"
echo "查看日志: tail -f ${LOG_DIR}/cron_*.log"
echo ""
echo "手动测试运行:"
echo "  cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${SCRIPT_PATH} --force"
echo ""
