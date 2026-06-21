// ==- dropout_4x4.mlir - Dropout（训练模式）-==//
// y = x * scale — 简化版，实际还包含 mask
// 用途: 防止过拟合，训练时随机丢弃神经元
// 降级: 4 行 → 24 行 LLVM

module {
  func.func @dropout(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %scale: f32) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %scaled = arith.mulf %a, %scale : f32
      linalg.yield %scaled : f32
    }
    return
  }
}
