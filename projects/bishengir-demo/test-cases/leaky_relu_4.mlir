// ==- leaky_relu_4.mlir - Leaky ReLU 激活 -==//
// leaky_relu(x) = x if x > 0 else 0.01*x
// 用途: 解决 ReLU 死亡问题，GAN 等模型常用
// 对应 AscendNPU-IR: arith.cmpf + select 模式
// 降级: 7 行 → 37 行 LLVM

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
