// MatMul — 矩阵乘法
//
// 功能: C = A @ B, 4×4 × 4×4 → 4×4
// AI 角色: 深度学习最核心的算子 — 占 LLM 算力的 60-80%
//   Linear 层: y = x @ W^T + b. Attention Q/K/V 投影. FFN up/down projection.
//   bishengir 保持 1 行 NPU 指令 (hivm.mmul), 不展开为标量.
// 应用场景: Linear / Attention / FFN, 无处不在
// MLIR 模式: linalg.matmul named op, 1行→74行LLVM (74×膨胀)
// 对应 bishengir: hfusion.cube_matmul → hivm.mmul (1行NPU指令)
//
1|// ==- matmul_4x4x4.mlir - 矩阵乘法（bishengir demo）-==//
2|//
3|// 对应 AscendNPU-IR 源码:
4|//   输入格式: bishengir/test/Conversion/LinalgToHFusion/matmul-to-hfusion.mlir
5|//   Pass 实现: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
6|//   等价 bishengir-opt 命令:
7|//     bishengir-opt --convert-linalg-to-hfusion
8|//   预期 bishengir 输出: hfusion.cube_matmul
9|//   （bishengir 保持 1 行语义，本 demo 展开为 74 行）
10|//
11|// bishengir 保留 hfusion.cube_matmul 并直接映射到 Ascend NPU Cube 硬件单元,
12|// 无需展开到标量指令。本文件的标准降级展示了"如果不保留语义"会怎样.
13|//===
14|
15|module {
16|  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
17|    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
18|                 outs(%C : memref<4x4xf32>)
19|    return
20|  }
21|}
22|