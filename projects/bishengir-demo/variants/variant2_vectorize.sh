1|#!/bin/bash
2|# variant2_vectorize.sh — 向量化优化
3|#
4|# 对应 AscendNPU-IR:
5|#   等价命令: bishengir-opt --convert-linalg-to-hfusion --convert-hfusion-to-hivm input.mlir
6|#   bishengir 的 hivm.vadd 是一条向量指令，
7|#   本 demo 的 --affine-super-vectorize 模拟同等功能.
8|#   对应源码: bishengir/lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp
9|#
10|# 原理: 用 SIMD 指令一次处理 2 个元素，减少指令数
11|set -e
12|SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
13|MATMUL="$SCRIPT_DIR/../test-cases/03_advanced/01_matmul.mlir"
14|RESULT_DIR="$SCRIPT_DIR/results"
15|mkdir -p "$RESULT_DIR"
16|
17|echo "═══ Variant 2: 向量化 (tile+super-vectorize) ═══"
18|# Stage 1: 先分块再向量化
19|mlir-opt \
20|  --convert-linalg-to-affine-loops \
21|  --affine-loop-tile="tile-sizes=2,2,1" \
22|  --affine-super-vectorize="virtual-vector-size=2" \
23|  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant2_vector_stage1.mlir" | wc -l
24|echo "行 (vectorize 后)"
25|
26|# Stage 2: 降级向量操作
27|mlir-opt \
28|  --convert-linalg-to-affine-loops \
29|  --affine-loop-tile="tile-sizes=2,2,1" \
30|  --affine-super-vectorize="virtual-vector-size=2" \
31|  --lower-affine \
32|  --convert-scf-to-cf \
33|  --convert-func-to-llvm \
34|  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant2_vector_llvm.mlir" | wc -l
35|echo "行 (LLVM)"
36|