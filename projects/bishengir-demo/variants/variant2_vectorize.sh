#!/bin/bash
# variant2_vectorize.sh — 向量化优化
# 原理: 用 SIMD 指令一次处理 2 个元素，减少指令数
# 等价于 bishengir 的 hivm.vadd 向量指令
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATMUL="$SCRIPT_DIR/../test-cases/matmul_4x4x4.mlir"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"

echo "═══ Variant 2: 向量化 (tile+super-vectorize) ═══"
# Stage 1: 先分块再向量化
mlir-opt \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=2,2,1" \
  --affine-super-vectorize="virtual-vector-size=2" \
  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant2_vector_stage1.mlir" | wc -l
echo "行 (vectorize 后)"

# Stage 2: 降级向量操作
mlir-opt \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=2,2,1" \
  --affine-super-vectorize="virtual-vector-size=2" \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant2_vector_llvm.mlir" | wc -l
echo "行 (LLVM)"
