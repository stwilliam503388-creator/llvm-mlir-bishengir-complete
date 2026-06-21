#!/bin/bash
# trace-to-ascendnpu.sh — 在 AscendNPU-IR 源码中定位与给定名称相关的文件
#
# 用法:
#   bash trace-to-ascendnpu.sh <关键词>
#   bash trace-to-ascendnpu.sh linalg-to-hfusion
#   bash trace-to-ascendnpu.sh hivm.vadd
#   bash trace-to-ascendnpu.sh matmul
#
# 前提: ascendnpu-ir 源码在本地，路径为 ~/hermes-workspace/ascendnpu-ir/
#       如果路径不同，设置环境变量 ASCENDNPU_IR_DIR

set -e

ASCEND="${ASCENDNPU_IR_DIR:-$HOME/hermes-workspace/ascendnpu-ir}"

if [ ! -d "$ASCEND" ]; then
    echo "❌ 未找到 AscendNPU-IR 源码: $ASCEND"
    echo "   请设置环境变量: export ASCENDNPU_IR_DIR=/path/to/ascendnpu-ir"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "用法: bash trace-to-ascendnpu.sh <关键词>"
    echo ""
    echo "示例:"
    echo "  bash trace-to-ascendnpu.sh linalg-to-hfusion   # 找转换 Pass"
    echo "  bash trace-to-ascendnpu.sh elemwise_binary      # 找 dialect 定义"
    echo "  bash trace-to-ascendnpu.sh hivm                 # 找 dialect 目录"
    echo "  bash trace-to-ascendnpu.sh matmul               # 找测试用例"
    exit 0
fi

QUERY="$1"

echo "╔══════════════════════════════════════════════╗"
echo "║  AscendNPU-IR 源码追踪                      ║"
echo "║  搜索: $QUERY"
echo "║  目录: $ASCEND"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 在 .cpp 文件中搜索
echo "━━━ 实现文件 (.cpp) ━━━"
find "$ASCEND" -name "*.cpp" -exec grep -l "$QUERY" {} \; 2>/dev/null | head -10

echo ""
echo "━━━ 定义文件 (.td) ━━━"
find "$ASCEND" -name "*.td" -exec grep -l "$QUERY" {} \; 2>/dev/null | head -10

echo ""
echo "━━━ 测试文件 (.mlir) ━━━"
find "$ASCEND" -name "*.mlir" -exec grep -l "$QUERY" {} \; 2>/dev/null | head -10

echo ""
echo "━━━ 头文件 (.h) ━━━"
find "$ASCEND" -name "*.h" -exec grep -l "$QUERY" {} \; 2>/dev/null | head -10
