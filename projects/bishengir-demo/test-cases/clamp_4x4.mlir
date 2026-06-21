// ==- clamp_4x4.mlir - 数值裁剪 -==//
// clamp(x, min, max) — 将 tensor 值限制在 [min, max] 内
// 用途: 梯度裁剪 / 激活值截断
// 降级: 7 行 → 53 行 LLVM

module {
  func.func @clamp(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    %min = arith.constant -1.0 : f32
    %max = arith.constant 1.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %c1 = arith.cmpf olt, %a, %min : f32
      %clamp_min = arith.select %c1, %min, %a : f32
      %c2 = arith.cmpf ogt, %clamp_min, %max : f32
      %clamped = arith.select %c2, %max, %clamp_min : f32
      linalg.yield %clamped : f32
    }
    return
  }
}
