1|#!/bin/bash
2|# compare.sh — 运行所有优化 variant 并输出对比表
3|#
4|# 对应 AscendNPU-IR:
5|#   本脚本的 V0-V3 对比对应 bishengir 的 3 种降级路径:
6|#   V0: bishengir-opt --convert-linalg-to-hfusion          (无优化)
7|#   V2: bishengir-opt --convert-hfusion-to-hivm             (向量化)
8|#   V3: hfusion.cube_matmul → hivm.mmul                     (硬件映射)
9|#   对应源码: bishengir/lib/Conversion/ 下三个 Pass 目录
10|set -e
11|
12|SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
13|export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
14|
15|echo "╔══════════════════════════════════════════════════════════════╗"
16|echo "║  matmul 优化方案对比                                       ║"
17|echo "╚══════════════════════════════════════════════════════════════╝"
18|echo ""
19|
20|# 基准行数 (从 bishengir-demo 根目录取)
21|BASE=$(mlir-opt \
22|  --convert-linalg-to-affine-loops \
23|  --lower-affine --convert-scf-to-cf --convert-func-to-llvm \
24|  "$SCRIPT_DIR/../test-cases/03_advanced/01_matmul.mlir" 2>&1 | wc -l)
25|
26|echo "输入文件: matmul_4x4x4.mlir (linalg.matmul, 1 行)"
27|echo ""
28|
29|# 运行每个 variant
30|results=""
31|for v in 0 1 2 3; do
32|    echo "━━━ 运行 Variant $v ━━━"
33|    bash "$SCRIPT_DIR/variant${v}_"*.sh 2>&1 | tail -3
34|    echo ""
35|done
36|
37|# 收集行数
38|echo "╔══════════════════════════════════════════════════════════════╗"
39|echo "║  对比总表                                                  ║"
40|echo "╚══════════════════════════════════════════════════════════════╝"
41|echo ""
42|
43|V0=$(wc -l < "$SCRIPT_DIR/results/variant0_llvm.mlir" 2>/dev/null || echo "N/A")
44|V1=$(wc -l < "$SCRIPT_DIR/results/variant1_tiling_llvm.mlir" 2>/dev/null || echo "N/A")
45|V2=$(wc -l < "$SCRIPT_DIR/results/variant2_vector_llvm.mlir" 2>/dev/null || echo "N/A")
46|V3=$(wc -l < "$SCRIPT_DIR/results/variant3_hw_llvm.mlir" 2>/dev/null || echo "N/A")
47|
48|echo "  Variant    优化策略                   LLVM行数   对比基准"
49|echo "  ──────────────────────────────────────────────────────────"
50|echo "  V0         无优化 (基准)               ${V0}行      -"
51|echo "  V1         循环分块 (tile=2x2x1)       ${V1}行      +$((V1 - BASE))行"
52|echo "  V2         向量化 (tile+vectorize)     ${V2}行      +$((V2 - BASE))行"
53|echo "  V3         硬件映射 (模拟 mmul)        ${V3}行      +$((V3 - BASE))行"
54|echo ""
55|
56|echo "  结果保存在: $SCRIPT_DIR/results/"
57|