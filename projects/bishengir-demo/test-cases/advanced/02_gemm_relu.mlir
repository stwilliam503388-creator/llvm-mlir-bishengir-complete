// GEMM + ReLU — 矩阵乘+激活融合
//
// 功能: y = ReLU(x @ W), 先矩阵乘后激活
// AI 角色: MLP 层的标准融合模式 — 编译器可将 matmul+relu 合并为单个 kernel
//   减少一次中间 buffer 读写, 可提升 1.3-1.5 倍性能.
// 应用场景: MLP 层 / FFN 层 (TVM/XLA/Triton 核心优化)
// MLIR 模式: linalg.matmul + linalg.generic 两阶段 pipeline
// 对应 bishengir: 融合优化可合并为单个 kernel
//
1|// ==- gemm_relu_4x4.mlir - GEMM + ReLU 融合 -==//
2|// y = relu(x @ W) — Linear + Activation 融合模式
3|// 用途: MLP 层的标准模式，展示 two-pass lowering
4|// 对应 AscendNPU-IR: 融合优化可将 matmul + relu 合并为单个 kernel
5|// 降级: 8 行 → 77 行 LLVM
6|
7|module {
8|  func.func @gemm_relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
9|    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%C : memref<4x4xf32>)
10|    %c0 = arith.constant 0.0 : f32
11|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%C : memref<4x4xf32>) outs(%C : memref<4x4xf32>) {
12|    ^bb0(%c: f32, %out: f32):
13|      %gt = arith.cmpf ogt, %c, %c0 : f32
14|      %relu = arith.select %gt, %c, %c0 : f32
15|      linalg.yield %relu : f32
16|    }
17|    return
18|  }
19|}
20|