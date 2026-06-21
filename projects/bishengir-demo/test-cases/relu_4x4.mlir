// ==- relu_4x4.mlir - ReLU 激活函数 -==//
//
// y = max(x, 0) — 最常用的激活函数，视觉和语言模型都有
// 对应 AscendNPU-IR: bishengir 通过 linalg.generic 做逐元素操作
// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
// 等价 bishengir-opt: --convert-linalg-to-hfusion
// 预期 bishengir 输出: hfusion.elemwise_unary {fun = relu}
//
// 降级效果:  5 行输入 → affine (约 20 行) → LLVM (约 40 行)

module {
  func.func @relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %gt = arith.cmpf ogt, %a, %c0 : f32
      %relu = arith.select %gt, %a, %c0 : f32
      linalg.yield %relu : f32
    }
    return
  }
}
