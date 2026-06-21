// ==- gemm_relu_4x4.mlir - GEMM + ReLU 融合 -==//
// y = relu(x @ W) — Linear + Activation 融合模式
// 用途: MLP 层的标准模式，展示 two-pass lowering
// 对应 AscendNPU-IR: 融合优化可将 matmul + relu 合并为单个 kernel
// 降级: 8 行 → 77 行 LLVM

module {
  func.func @gemm_relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%C : memref<4x4xf32>)
    %c0 = arith.constant 0.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%C : memref<4x4xf32>) outs(%C : memref<4x4xf32>) {
    ^bb0(%c: f32, %out: f32):
      %gt = arith.cmpf ogt, %c, %c0 : f32
      %relu = arith.select %gt, %c, %c0 : f32
      linalg.yield %relu : f32
    }
    return
  }
}
