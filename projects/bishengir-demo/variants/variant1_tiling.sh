1|#!/bin/bash
2|# variant1_tiling.sh — 循环分块优化 (Tiling)
3|#
4|# 对应 AscendNPU-IR:
5|#   等价命令: bishengir-opt --convert-linalg-to-hfusion --affine-loop-tile input.mlir
6|#   bishengir 内部 K 维度隐含分块行为，
7|#   与本 demo 的 tile-sizes=2,2,1 目的相同——提高 cache 局部性.
8|#   对应源码: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
9|#
10|# 原理: 将 4x4 矩阵切成 2x2 的子块，提高 cache 局部性
11|# 效果: 代码量增加（多了 tile 循环），但运行性能提升
12|set -e
13|SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
14|MATMUL="$SCRIPT_DIR/../test-cases/03_advanced/01_matmul.mlir"
15|RESULT_DIR="$SCRIPT_DIR/results"
16|mkdir -p "$RESULT_DIR"
17|
18|echo "═══ Variant 1: 循环分块 (tile-sizes=2,2,1) ═══"
19|mlir-opt \
20|  --convert-linalg-to-affine-loops \
21|  --affine-loop-tile="tile-sizes=2,2,1" \
22|  --lower-affine \
23|  --convert-scf-to-cf \
24|  --convert-func-to-llvm \
25|  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant1_tiling_llvm.mlir" | wc -l
26|echo "行"
27|