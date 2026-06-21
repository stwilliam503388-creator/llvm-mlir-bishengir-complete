// Reduce Max — 最大值归约
// 公式: max = max(所有元素)
// 一句话: 从一堆数里找出最大的
// 专业角色: Softmax 数值稳定性关键, 先减 max 防 exp 溢出
// 用在哪: Softmax 数值稳定 / Max Pooling
// 降级: reduction + cmpf + select
// bishengir: hfusion.reduce {fun = max}

module {
  func.func @reduce_max(%A: memref<4x4xf32>, %init: memref<f32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%A : memref<4x4xf32>) outs(%init : memref<f32>) {
    ^bb0(%a: f32, %b: f32):
      %gt = arith.cmpf ogt, %a, %b : f32
      %max = arith.select %gt, %a, %b : f32
      linalg.yield %max : f32
    }
    return
  }
}
