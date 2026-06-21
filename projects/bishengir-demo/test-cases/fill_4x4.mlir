// ==- fill_4x4.mlir - 张量填充（模拟 linalg.fill）-==//
// 用 linalg.generic 实现 fill 语义
// 原因: Homebrew LLVM 22 未编译 linalg.fill named op
// 降级: 4 行 → 39 行 LLVM

module {
  func.func @fill(%A: memref<4x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} outs(%A : memref<4x4xf32>) {
    ^bb0(%a: f32):
      linalg.yield %c0 : f32
    }
    return
  }
}
