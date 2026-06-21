// ==- conv2d_4x4.mlir - 二维卷积（valid padding）-==//
// 用 linalg.generic 实现 2D 有效卷积
// 输入 4x4, 卷积核 3x3, 输出 2x2
// 原因: Homebrew LLVM 22 未编译 linalg.conv_2d named op
// 对应 AscendNPU-IR: bishengir 通过 linalg.generic 模式匹配
// 降级: 6 行 → 85 行 LLVM

module {
  func.func @conv2d(%input: memref<4x4xf32>, %kernel: memref<3x3xf32>, %output: memref<2x2xf32>) {
    linalg.generic {indexing_maps = [affine_map<(oh,ow,kh,kw) -> (oh+kh, ow+kw)>, affine_map<(oh,ow,kh,kw) -> (kh, kw)>, affine_map<(oh,ow,kh,kw) -> (oh, ow)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%input, %kernel : memref<4x4xf32>, memref<3x3xf32>) outs(%output : memref<2x2xf32>) {
    ^bb0(%a: f32, %k: f32, %b: f32):
      %prod = arith.mulf %a, %k : f32
      %sum = arith.addf %b, %prod : f32
      linalg.yield %sum : f32
    }
    return
  }
}
