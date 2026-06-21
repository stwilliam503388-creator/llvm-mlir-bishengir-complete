// Leaky ReLU — 带泄漏的线性整流
// 公式: y = x if x > 0 else 0.01x
// 一句话: ReLU 改进版, 负数区留一小缝
// 专业角色: 解决 ReLU 死亡问题, GAN 标配
// 用在哪: GAN / 部分传统 CNN
// 降级: cmpf + mulf + select
// bishengir: 条件分支映射

module {
  func.func @leaky_relu(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c0 = arith.constant 0.0 : f32
    %alpha = arith.constant 0.01 : f32
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %gt = arith.cmpf ogt, %a, %c0 : f32
      %leak = arith.mulf %a, %alpha : f32
      %relu = arith.select %gt, %a, %leak : f32
      linalg.yield %relu : f32
    }
    return
  }
}
