// MatMul — 矩阵乘法
// 公式: C = A @ B, 4x4 x 4x4 -> 4x4
// 一句话: 深度学习最核心操作, 占LLM算力60-80%
// 专业角色: Linear 层, Attention Q/K/V 投影, FFN up/down
// 用在哪: Linear / Attention / FFN
// 降级: linalg.matmul, 1行->74行LLVM (74x展开)
// bishengir: hfusion.cube_matmul -> hivm.mmul (1行NPU指令)

module {
  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%C : memref<4x4xf32>)
    return
  }
}
