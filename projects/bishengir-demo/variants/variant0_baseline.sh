#!/bin/bash
# variant0_baseline.sh — 基准: 无优化的标准降级
#
# 对应 AscendNPU-IR:
#   等价命令: bishengir-opt --convert-linalg-to-hfusion input.mlir
#   区别: bishengir 输出 hfusion.elemwise_binary,
#         本 demo 输出 affine.for.
#   对应源码: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATMUL="$SCRIPT_DIR/../test-cases/matmul_4x4x4.mlir"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"

echo "═══ Variant 0: 基准降级（无优化） ═══"
mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant0_llvm.mlir" | wc -l
echo "行"
