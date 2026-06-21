// ==- prelu_4x4.mlir - PReLU 激活 -==//
// prelu(x) = x if x > 0 else alpha*x (alpha 可学习)
// 用途: 图像分类/超分辨率
// 降级: 7 行 → 52 行 LLVM

module {
  func.func @prelu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    %alpha = arith.constant 0.25 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %neg = arith.mulf %a, %alpha : f32
      %gt = arith.cmpf ogt, %a, %c0 : f32
      %p = arith.select %gt, %a, %neg : f32
      linalg.yield %p : f32
    }
    return
  }
}
