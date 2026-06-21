// ==- layer_norm_4x4.mlir - Layer Normalization -==//
// LN: y = (x - mean) / sqrt(var + eps) * gamma + beta
// 用途: Transformer 每一层后的归一化
// 对应 AscendNPU-IR: 需 reduction + broadcast + elemwise 组合
// 本文件演示: 归一化计算的平方差部分
// 降级: 7 行 → 44 行 LLVM

module {
  func.func @layer_norm(%A: memref<4x4xf32>, %mean: memref<f32>, %var: memref<f32>, %B: memref<4x4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %diff = arith.subf %a, %b : f32
      %sq = arith.mulf %diff, %diff : f32
      linalg.yield %sq : f32
    }
    return
  }
}
