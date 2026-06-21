// MatMul (原始版) — 矩阵乘法 (对应 triton/03_advanced/20_matmul.py) ⭐⭐⭐
// 公式: C = A @ B, 4x4 x 4x4 -> 4x4
// 一句话: bishengir 原始矩阵乘法测试用例, 与 01_matmul.mlir 功能相同
// 专业角色: linalg.matmul named op, bishengir 保留为 1 行 hivm.mmul NPU 指令
// 用在哪: Linear / Attention / FFN
// 降级: linalg.matmul, 1行→74行LLVM (74×)
// bishengir: hfusion.cube_matmul → hivm.mmul (1行NPU指令)
//
module {
  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
                 outs(%C : memref<4x4xf32>)
    return
  }
}