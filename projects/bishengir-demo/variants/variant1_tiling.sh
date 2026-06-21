#!/bin/bash
# variant1_tiling.sh — 循环分块优化 (Tiling)
# 原理: 将 4x4 矩阵切成 2x2 的子块，提高 cache 局部性
# 效果: 代码量增加（多了 tile 循环），但运行性能提升
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATMUL="$SCRIPT_DIR/../test-cases/matmul_4x4x4.mlir"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"

echo "═══ Variant 1: 循环分块 (tile-sizes=2,2,1) ═══"
mlir-opt \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-sizes=2,2,1" \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant1_tiling_llvm.mlir" | wc -l
echo "行"
