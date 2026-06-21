#!/bin/bash
# compare.sh — 运行所有优化 variant 并输出对比表
#
# 对应 AscendNPU-IR:
#   本脚本的 V0-V3 对比对应 bishengir 的 3 种降级路径:
#   V0: bishengir-opt --convert-linalg-to-hfusion          (无优化)
#   V2: bishengir-opt --convert-hfusion-to-hivm             (向量化)
#   V3: hfusion.cube_matmul → hivm.mmul                     (硬件映射)
#   对应源码: bishengir/lib/Conversion/ 下三个 Pass 目录
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  matmul 优化方案对比                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 基准行数 (从 bishengir-demo 根目录取)
BASE=$(mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine --convert-scf-to-cf --convert-func-to-llvm \
  "$SCRIPT_DIR/../test-cases/matmul_4x4x4.mlir" 2>&1 | wc -l)

echo "输入文件: matmul_4x4x4.mlir (linalg.matmul, 1 行)"
echo ""

# 运行每个 variant
results=""
for v in 0 1 2 3; do
    echo "━━━ 运行 Variant $v ━━━"
    bash "$SCRIPT_DIR/variant${v}_"*.sh 2>&1 | tail -3
    echo ""
done

# 收集行数
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  对比总表                                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

V0=$(wc -l < "$SCRIPT_DIR/results/variant0_llvm.mlir" 2>/dev/null || echo "N/A")
V1=$(wc -l < "$SCRIPT_DIR/results/variant1_tiling_llvm.mlir" 2>/dev/null || echo "N/A")
V2=$(wc -l < "$SCRIPT_DIR/results/variant2_vector_llvm.mlir" 2>/dev/null || echo "N/A")
V3=$(wc -l < "$SCRIPT_DIR/results/variant3_hw_llvm.mlir" 2>/dev/null || echo "N/A")

echo "  Variant    优化策略                   LLVM行数   对比基准"
echo "  ──────────────────────────────────────────────────────────"
echo "  V0         无优化 (基准)               ${V0}行      -"
echo "  V1         循环分块 (tile=2x2x1)       ${V1}行      +$((V1 - BASE))行"
echo "  V2         向量化 (tile+vectorize)     ${V2}行      +$((V2 - BASE))行"
echo "  V3         硬件映射 (模拟 mmul)        ${V3}行      +$((V3 - BASE))行"
echo ""

echo "  结果保存在: $SCRIPT_DIR/results/"
