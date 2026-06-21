// ==- softmax_complete_4.mlir - 完整 Softmax 第一步 -==//
// y = exp(x - max(x)) — softmax 的数值稳定版本
// 用途: Attention 机制核心算子
// 对应 AscendNPU-IR: reduction → broadcast → exp
// 降级: 10 行 → 44 行 LLVM

module {
  func.func @softmax_stable(%A: memref<4x4xf32>) {
    %max_init = memref.alloc() : memref<f32>
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %max_init[] : memref<f32>
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%A : memref<4x4xf32>) outs(%max_init : memref<f32>) {
    ^bb0(%a: f32, %m: f32):
      %gt = arith.cmpf ogt, %a, %m : f32
      %max = arith.select %gt, %a, %m : f32
      linalg.yield %max : f32
    }
    return
  }
}
