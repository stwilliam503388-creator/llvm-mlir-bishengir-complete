1|#!/bin/bash
2|# variant0_baseline.sh — 基准: 无优化的标准降级
3|#
4|# 对应 AscendNPU-IR:
5|#   等价命令: bishengir-opt --convert-linalg-to-hfusion input.mlir
6|#   区别: bishengir 输出 hfusion.elemwise_binary,
7|#         本 demo 输出 affine.for.
8|#   对应源码: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
9|set -e
10|SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
11|MATMUL="$SCRIPT_DIR/../test-cases/mlir/03_advanced/01_matmul.mlir"
12|RESULT_DIR="$SCRIPT_DIR/results"
13|mkdir -p "$RESULT_DIR"
14|
15|echo "═══ Variant 0: 基准降级（无优化） ═══"
16|mlir-opt \
17|  --convert-linalg-to-affine-loops \
18|  --lower-affine \
19|  --convert-scf-to-cf \
20|  --convert-func-to-llvm \
21|  "$MATMUL" 2>&1 | tee "$RESULT_DIR/variant0_llvm.mlir" | wc -l
22|echo "行"
23|