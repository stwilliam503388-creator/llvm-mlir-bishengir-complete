// ==- batch_norm_4x4_part2.mlir - Batch Norm (normalize) -==//
// BN 第二步: (x - mean) / sqrt(var + eps) * gamma + beta
// Broadcasting mean/var/gamma/beta 到每个元素
// 降级: 9 行 → 87 行 LLVM

module {
  func.func @bn_norm(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %mean: memref<4xf32>, %var: memref<4xf32>, %gamma: memref<4xf32>, %beta: memref<4xf32>) {
    %eps = arith.constant 1.0e-5 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A, %mean, %var, %gamma, %beta : memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %m: f32, %v: f32, %g: f32, %bt: f32, %b: f32):
      %norm = arith.subf %a, %m : f32
      %std = arith.addf %v, %eps : f32
      %sqrt = math.sqrt %std : f32
      %div = arith.divf %norm, %sqrt : f32
      %scaled = arith.mulf %div, %g : f32
      %out = arith.addf %scaled, %bt : f32
      linalg.yield %out : f32
    }
    return
  }
}
