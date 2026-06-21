// ==- reduce_sum_4x4.mlir - 归约求和 -==//
//
// sum = Σx[i][j] — 把 4x4 矩阵归约为标量
// 对应 AscendNPU-IR:
//   hfusion.reduce {fun = add, axes = [0, 1]}
// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
// 等价 bishengir-opt: --convert-linalg-to-hfusion
//
// MLIR 关键概念: iterator_types = ["reduction", "reduction"]
//   "parallel" = 输出索引与输入索引一致
//   "reduction" = 输出索引比输入少一维（归约）

module {
  func.func @reduce_sum(%A: memref<4x4xf32>, %init: memref<f32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>],
      iterator_types = ["reduction", "reduction"]
    } ins(%A : memref<4x4xf32>) outs(%init : memref<f32>) {
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }
    return
  }
}
