#!/bin/bash
# AI Learning Website - 一键部署脚本
# 用法: ./scripts/deploy.sh "提交信息"

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  AI Learning Website 部署脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取提交信息
COMMIT_MSG="${1:-update: 更新内容}"

# 检查是否有改动
if git diff --quiet && git diff --cached --quiet; then
    echo -e "${YELLOW}没有检测到更改，跳过提交${NC}"
else
    echo -e "${GREEN}[1/4] 添加文件到暂存区...${NC}"
    git add .

    echo -e "${GREEN}[2/4] 创建提交...${NC}"
    git commit -m "$COMMIT_MSG

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

    echo -e "${GREEN}[3/4] 推送到远程仓库...${NC}"
    git push origin main

    echo -e "${GREEN}[4/4] 完成!${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "仓库: https://github.com/neko233-com/ai-learning-website"
echo -e "网站: https://neko233-com.github.io/ai-learning-website/"
echo -e "${GREEN}========================================${NC}"
