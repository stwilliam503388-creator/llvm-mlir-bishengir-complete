// ==- sigmoid_4.mlir - Sigmoid 激活函数 -==//
// σ(x) = 1 / (1 + exp(-x))
// 用途: 二分类输出层 / RNN 门控
// 对应 AscendNPU-IR: 逐元素 math.exp + arith.divf
// 降级: 5 行 → 33 行 LLVM

module {
  func.func @sigmoid(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c1 = arith.constant 1.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %neg = arith.negf %a : f32
      %exp = math.exp %neg : f32
      %one = arith.addf %exp, %c1 : f32
      %sig = arith.divf %c1, %one : f32
      linalg.yield %sig : f32
    }
    return
  }
}
