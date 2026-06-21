// ==- batch_norm_4x4_part1.mlir - Batch Norm (reduce mean) -==//
// BN 第一步: 计算每个通道的均值
// 将 4x4 矩阵按列 (i) 归约为 4 个均值
// 降级: 5 行 → 48 行 LLVM

module {
  func.func @bn_mean(%A: memref<4x4xf32>, %mean: memref<4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (j)>], iterator_types = ["reduction", "parallel"]} ins(%A : memref<4x4xf32>) outs(%mean : memref<4xf32>) {
    ^bb0(%a: f32, %m: f32):
      %sum = arith.addf %m, %a : f32
      linalg.yield %sum : f32
    }
    return
  }
}
