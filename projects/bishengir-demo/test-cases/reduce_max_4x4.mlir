// ==- reduce_max_4x4.mlir - 归约求最大值 -==//
//
// max = max(x[i][j]) — 求矩阵最大值（用于 softmax 数值稳定）
// 对应 AscendNPU-IR: hfusion.reduce {fun = max}
// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp

module {
  func.func @reduce_max(%A: memref<4x4xf32>, %init: memref<f32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>],
      iterator_types = ["reduction", "reduction"]
    } ins(%A : memref<4x4xf32>) outs(%init : memref<f32>) {
    ^bb0(%a: f32, %b: f32):
      %gt = arith.cmpf ogt, %a, %b : f32
      %max = arith.select %gt, %a, %b : f32
      linalg.yield %max : f32
    }
    return
  }
}
