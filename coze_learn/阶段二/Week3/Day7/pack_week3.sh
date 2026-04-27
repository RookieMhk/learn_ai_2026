#!/bin/bash
# Week 3 打包脚本
# 将 Day1-Day7 所有产出物打包成 Week3.zip

echo "=========================================="
echo "  Week 3 打包脚本"
echo "=========================================="

# 设置目录
WEEK3_DIR="长期计划/个性化学习计划/outputs/阶段二/Week3"
OUTPUT_FILE="Week3.zip"

# 切换到Week3目录
cd "$WEEK3_DIR" || exit 1

# 删除旧zip（如果存在）
if [ -f "$OUTPUT_FILE" ]; then
    echo "删除旧文件: $OUTPUT_FILE"
    rm "$OUTPUT_FILE"
fi

# 创建新的zip（包含所有目录）
echo "开始打包..."
zip -r "$OUTPUT_FILE" Day1 Day2 Day3 Day4 Day5 Day6 Day7 -x "*.DS_Store" "*/__pycache__/*"

# 检查结果
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo ""
    echo "=========================================="
    echo "  ✅ 打包完成！"
    echo "  文件: $OUTPUT_FILE"
    echo "  大小: $SIZE"
    echo "=========================================="
    echo ""
    echo "包含内容:"
    unzip -l "$OUTPUT_FILE" | tail -10
else
    echo "❌ 打包失败！"
    exit 1
fi
