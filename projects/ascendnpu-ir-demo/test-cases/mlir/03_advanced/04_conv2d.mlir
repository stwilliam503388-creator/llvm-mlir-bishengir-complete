// Conv2D — 二维卷积 (valid padding)
// 公式: 输入4x4, kernel 3x3, stride=1 -> 输出2x2
// 一句话: 卷积核在输入上滑动提取特征
// 专业角色: CNN 核心算子, 参数共享+局部连接
// 用在哪: ResNet/VGG/YOLO/CNN 全系
// 降级: linalg.generic + reduction x2, 6行->85行LLVM
// bishengir: 可被 pattern matching 优化
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: arith.addf

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
