// Softmax (完整第一步) — 数值稳定版
// 公式: y = e^{x - max(x)}
// 一句话: 先减最大值再取指数, 防止溢出
// 专业角色: Attention 机制核心操作
// 用在哪: Transformer / BERT/GPT Attention 层
// 降级: memref.alloc + reduction + memref.store
// bishengir: 组合 reduce + broadcast + elemwise

module {
  func.func @softmax_stable(%A: memref<4x4xf32>) {
    %max_init = memref.alloc() : memref<f32>
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %max_init[] : memref<f32>
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%A : memref<4x4xf32>) outs(%max_init : memref<f32>) {
    ^bb0(%a: f32, %m: f32):
      %gt = arith.cmpf ogt, %a, %m : f32
      %max = arith.select %gt, %a, %m : f32
      linalg.yield %max : f32
    }
    return
  }
}
