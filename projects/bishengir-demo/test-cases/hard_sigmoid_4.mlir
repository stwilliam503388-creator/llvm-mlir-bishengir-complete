// ==- hard_sigmoid_4.mlir - Hard Sigmoid 激活 -==//
// hard_sigmoid(x) = clamp(0.2*x + 0.5, 0, 1)
// 用途: MobileNet 等轻量化模型（计算量比 sigmoid 小）
// 对应 AscendNPU-IR: arith.mulf + addf + max/minimumf
// 降级: 7 行 → 39 行 LLVM

module {
  func.func @hard_sigmoid(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %c05 = arith.constant 0.2 : f32
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %mul = arith.mulf %a, %c05 : f32
      %add = arith.addf %mul, %c05 : f32
      %clamped = arith.maximumf %add, %c0 : f32
      %out = arith.minimumf %clamped, %c1 : f32
      linalg.yield %out : f32
    }
    return
  }
}
