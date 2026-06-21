// ==- softmax_4.mlir - Softmax 指数部分 -==//
//
// y[i] = exp(x[i]) — softmax 的第一步（元素级指数）
// 完整 softmax: exp(x - max(x)) / sum(exp(x - max(x)))
// 本文件只演示 exp 部分，reduce + div 由其他用例覆盖
// 对应 AscendNPU-IR: 逐元素 math 操作映射到 hfusion
// 等价 bishengir-opt: --convert-linalg-to-hfusion

module {
  func.func @softmax_exp(%A: memref<4xf32>, %B: memref<4xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %exp = math.exp %a : f32
      linalg.yield %exp : f32
    }
    return
  }
}
